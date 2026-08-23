//! Stmt — Block and statement compilation. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Compile a Block expression.
    ///
    /// Compiles stmts in order; the trailing expression's NodeId is the Block's result.
    pub(super) fn compile_block(
        &mut self,
        stmts: &[crate::ast::Ast::StmtId],
        trailing: &Option<crate::ast::Ast::ExprId>,
    ) -> NodeId {
        self.enter_scope();
        let prev_effect = self.current_effect;
        // Bug #66: If this is the function body's top-level block, do NOT extract defers —
        // they must stay in defer_table for function-exit execution. Only nested blocks
        // extract block-scoped defers. The flag is set by compile_function_body and reset
        // here so that nested blocks within the function body see `false`.
        let is_function_top_block = self.in_function_top_block;
        self.in_function_top_block = false;
        // Initialize last_effect to prev_effect so the block's first statement depends on prior effects
        // (e.g. the store nodes of global var initialization in the entry function), ensuring that
        // load/call inside the block run only after prior side effects complete.
        let mut last_effect: Option<NodeId> = prev_effect;
        self.current_effect = None;
        // Bug #66: Record the defer_table length at block entry so we can extract
        // block-scoped defers and run them at block exit (LIFO).
        let defer_mark = self
            .current_function_sg
            .map(|sg| self.graph.subgraphs[sg.0 as usize].defer_table.len())
            .unwrap_or(0);
        for &stmt_id in stmts {
            // Set current_effect so subsequent effect nodes (e.g. WriteBack) depend on the prior effect
            self.current_effect = last_effect;
            // Statements are not in tail position (Return internally restores in_tail_position = true for its value)
            let prev_tail = self.in_tail_position;
            self.in_tail_position = false;
            let effect = self.compile_stmt(stmt_id);
            self.in_tail_position = prev_tail;
            if let Some(eff) = effect {
                // Control-flow nodes (CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR) have their
                // prior side-effect dependencies baked into inputs in compile_stmt; no signal relocation
                // is needed. chain_effects is only used for sequential linking of non-control-flow statements.
                let chained = self.chain_effects(last_effect, eff);
                last_effect = Some(chained);
            }
        }
        // The trailing expression inherits the block's effect chain on compilation,
        // ensuring Call nodes in the trailing expression depend on prior effects (consistent with stmts)
        self.current_effect = last_effect;
        let result = match trailing {
            Some(expr_id) => {
                let result_node = self.compile_expr(*expr_id);
                self.chain_effects(last_effect, result_node)
            }
            None => last_effect.unwrap_or_else(|| self.compile_void_const()),
        };
        // Bug #66: Block-scoped defer cleanup — extract defers registered inside this block
        // and generate LIFO cleanup Call nodes after the block result. This ensures defers
        // declared inside `{ ... }` execute when the block exits, not when the function exits.
        // The extracted defers are removed from the function-level defer_table to prevent
        // double execution at function exit.
        // Skip for function body top-level block: those defers must stay in defer_table for
        // function-exit execution (run_defers_sync / process_frame).
        let (result, _defer_effect) = if is_function_top_block {
            (result, None)
        } else {
            self.compile_block_defer_cleanup(defer_mark, result)
        };
        // defer cleanup effects are chained into `result` via CF_SEQ inside
        // compile_block_defer_cleanup, so they flow to consumers through the block's
        // return value. No separate last_effect update is needed (current_effect is
        // restored to prev_effect below).
        self.current_effect = prev_effect;
        self.exit_scope();
        result
    }

    /// Bug #66: Extract block-scoped defers registered after `defer_mark` and generate
    /// LIFO cleanup Call nodes. The defers are removed from the function-level defer_table.
    /// The cleanup nodes are chained after `result` via CF_SEQ, preserving the result value.
    /// Returns (result_node, cleanup_effect) where cleanup_effect is the last defer Call node
    /// (to be used as last_effect for subsequent statements).
    pub(super) fn compile_block_defer_cleanup(
        &mut self,
        defer_mark: usize,
        result: NodeId,
    ) -> (NodeId, Option<NodeId>) {
        let cur_sg = match self.current_function_sg {
            Some(sg) => sg,
            None => return (result, None), // No function subgraph — nothing to do
        };
        let defer_table = &mut self.graph.subgraphs[cur_sg.0 as usize].defer_table;
        if defer_table.len() <= defer_mark {
            return (result, None); // No new defers in this block
        }
        // Extract block-scoped defers (drain entries after defer_mark).
        let block_defers: Vec<crate::ir::Ir::DeferEntry> =
            defer_table.drain(defer_mark..).collect();
        // Generate cleanup by reusing the loop-defer machinery (CF_DEFER_REGISTER + CF_DEFER_RUN).
        // Each block-scoped defer is registered onto the runtime defer_stack via a
        // CF_DEFER_REGISTER node (which snapshots the defer's captured values), then a single
        // CF_DEFER_RUN node drains the stack in LIFO order, executing each defer body as a proper
        // defer frame (with parent_frame_ptr/root_frame_ptr set so the body can read/write outer
        // variables via the frame chain). This mirrors how loops run defer-in-loop bodies and
        // fixes two issues:
        //   - The defer body must run as a defer frame (NOT a regular Call via make_call, which
        //     gives a node_offset=0 frame that cannot reach outer scope via the frame chain).
        //   - The block result value must be preserved: cleanup nodes are chained BEFORE `result`
        //     via CF_SEQ (which returns its LAST input's value), so the final node yields `result`.
        // Generate cleanup by reusing the loop-defer machinery (CF_BLOCK_DEFER_REGISTER +
        // CF_DEFER_RUN). Each block-scoped defer is registered onto the runtime defer_stack via a
        // CF_BLOCK_DEFER_REGISTER node (which snapshots the defer's captured values), then a single
        // CF_DEFER_RUN node drains the stack in LIFO order, executing each defer body as a proper
        // defer frame (with parent_frame_ptr/root_frame_ptr set so the body can read/write outer
        // variables via the frame chain). This mirrors how loops run defer-in-loop bodies.
        //
        // ORDERING (critical): in the dataflow scheduler every node is scheduled independently
        // based on its OWN inputs. A node with zero inputs is enqueued at frame start and would
        // fire before prior effects (e.g. global-var initialization) complete, causing the defer
        // body's reads of outer/global variables to observe stale/null values. To prevent this,
        // each register/run node takes the accumulated effect chain as a DIRECT input:
        //   - CF_BLOCK_DEFER_REGISTER treats input[0] as an effect-ordering dependency and uses
        //     inputs[1..] as the captured NodeIds.
        //   - CF_DEFER_RUN ignores all inputs (it reads defer_stack) but still requires them ready.
        // The block result value is preserved by wrapping the final run node + `result` in a
        // CF_SEQ (which returns its LAST input's value, i.e. `result`).
        let mut last_defer_call: Option<NodeId> = None;
        let mut effect_dep: NodeId = result;
        // Iterate in source (registration) order so the register nodes push onto defer_stack in
        // the same order; CF_DEFER_RUN then drains in LIFO (rev) order, running the
        // last-declared defer first — matching the function-level defer semantics.
        for entry in block_defers.iter() {
            // Build inputs: [effect_dep] ++ captured_inputs.
            let mut reg_inputs: Vec<NodeId> = Vec::with_capacity(entry.captured_inputs.len() + 1);
            reg_inputs.push(effect_dep);
            reg_inputs.extend_from_slice(&entry.captured_inputs);
            let inputs_off = self.graph.inputs_pool.push(&reg_inputs);
            let reg_node = self.graph.add_node(Node {
                kind: NodeKind::Call,
                input_count: reg_inputs.len() as u8,
                inputs_offset: inputs_off,
                compute_fn: CF_BLOCK_DEFER_REGISTER,
            });
            self.graph.set_call_target(reg_node, entry.body_subgraph);
            effect_dep = reg_node;
            last_defer_call = Some(reg_node);
        }
        // CF_DEFER_RUN node: drains defer_stack in LIFO order and runs each defer body as a defer
        // frame. Give it `effect_dep` as a direct input so it cannot fire before the register
        // nodes (and thus before the block's prior effects) complete.
        let run_off = self.graph.inputs_pool.push(&[effect_dep]);
        let run_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: 1,
            inputs_offset: run_off,
            compute_fn: CF_DEFER_RUN,
        });
        if last_defer_call.is_none() {
            last_defer_call = Some(run_node);
        }
        // Wrap run_node + result in CF_SEQ so the block's value is `result` (CF_SEQ returns its
        // last input's value). Both inputs must be ready before the SEQ computes, so the defer
        // cleanup side effects are guaranteed to complete before any consumer reads the value.
        let result_node = self.chain_effects(Some(run_node), result);
        (result_node, last_defer_call)
    }

    /// Compile a statement, returning an effect node (to be sequentially linked into the block result node).
    /// Returns None for pure declarations (variable bindings); their value node is automatically reachable via variable references.
    pub(super) fn compile_stmt(&mut self, stmt_id: crate::ast::Ast::StmtId) -> Option<NodeId> {

        // Skip analyzer-flagged dead statements (unreachable code / dead declarations / dead stores); emit no IR nodes
        if self.is_dead_stmt(stmt_id) {
            return None;
        }
        let spanned = self.current_module().arena.stmt(stmt_id);
        let stmt = &spanned.node;
        match stmt {
            crate::ast::Ast::Stmt::ValDecl { name, value, .. } => {
                let value_node = self.compile_subexpr(*value);
                // Create an independent copy node for the val declaration (CF_SEQ single input = identity),
                // so the val binding owns an independent node ID rather than aliasing the source node.
                // This ensures that closures capturing the val variable capture the snapshot value at
                // declaration time, rather than the current value of the source variable (which may be a var).
                // For example: in a while loop, `val captured = i` followed by `fun() { captured }`;
                // without a copy node, captured aliases i's node and all closures read i's final value
                // after the loop ends. With a copy node, captured owns an independent node (within the
                // loop body subgraph scope); in the main frame that node is not ready, so the
                // same_function path falls back to the closure's Cell upvalue, returning the correct snapshot.
                let copy_off = self.graph.inputs_pool.push(&[value_node]);
                let copy_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 1,
                    inputs_offset: copy_off,
                    compute_fn: CF_SEQ,
                });
                // Cell backing (place model): scalar bindings lower to a Cell
                // at the DECL SITE — order-independent (`&x` before or after
                // any read/write), and `&x` is zero-cost (the cell exists).
                // All-vars functions cell-back every scalar `var` (C1-③④);
                // elsewhere only address-taken bindings (C1-①). Captured
                // bindings stay plain (the capture machinery snapshots the
                // binding node).
                if self.decl_cell_backing_eligible(name, *value, false) {
                    let off = self.graph.inputs_pool.push(&[copy_node]);
                    let cell_node = self.graph.add_node(Node {
                        kind: NodeKind::UnOp,
                        input_count: 1,
                        inputs_offset: off,
                        compute_fn: CF_CELL_ALLOC,
                    });
                    self.bind_cell(name, cell_node);
                    self.track_cell_decl(name, cell_node, copy_node);
                    return Some(cell_node);
                }
                self.declare_var(name, copy_node);
                Some(copy_node)
            }
            crate::ast::Ast::Stmt::VarDecl { name, value, .. } => {
                let value_node = self.compile_subexpr(*value);
                // A `var` initialized from an IDENTIFIER reference must own a FRESH
                // slot. The initializer expression is an alias of the referenced
                // binding's node; aliasing it as this var's home lets later
                // WriteBacks (loop-body home sync) overwrite the SHARED node's
                // slot, corrupting every other reader of that binding — observed
                // as: loop-level `val st`, branch-local `var e = st`, inner-loop
                // `e = e + 1` writebacks clobbering st (st read back as e's value).
                // A one-input CF_SEQ node is a fresh identity copy; non-reference
                // initializers already produce fresh exclusive nodes.
                let is_ref = matches!(
                    &self.current_module().arena.expr(*value).node,
                    crate::ast::Ast::Expr::Ident(_)
                );
                let home = if is_ref {
                    let off = self.graph.inputs_pool.push(&[value_node]);
                    self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 1,
                        inputs_offset: off,
                        compute_fn: CF_SEQ,
                    })
                } else {
                    value_node
                };
                // Cell backing (see ValDecl above); all-vars functions back
                // every scalar `var`.
                if self.decl_cell_backing_eligible(name, *value, true) {
                    let off = self.graph.inputs_pool.push(&[home]);
                    let cell_node = self.graph.add_node(Node {
                        kind: NodeKind::UnOp,
                        input_count: 1,
                        inputs_offset: off,
                        compute_fn: CF_CELL_ALLOC,
                    });
                    self.bind_cell(name, cell_node);
                    self.track_cell_decl(name, cell_node, home);
                    return Some(cell_node);
                }
                self.declare_var(name, home);
                Some(home)
            }
            crate::ast::Ast::Stmt::Expression { expr } => {
                let expr_node = self.compile_subexpr(*expr);
                Some(expr_node)
            }
            crate::ast::Ast::Stmt::Assignment { target, value } => {
                let raw_val = self.compile_subexpr(*value);
                // Link current_effect: ensures the assignment expression executes only after prior effects
                // (e.g. an if-Gate with continue) complete. Prevents statements after continue from running early.
                let val_node = self.chain_effects(self.current_effect, raw_val);
                let target_expr = &self.current_module().arena.expr(*target).node;
                // Array index assignment arr[i] = x: emit a CF_ARRAY_STORE node (three inputs: arr, index, value)
                if let crate::ast::Ast::Expr::Index { recv, index } = target_expr {
                    let arr_node = self.compile_subexpr(*recv);
                    let idx_node = self.compile_subexpr(*index);
                    let off = self.graph.inputs_pool.push(&[arr_node, idx_node, val_node]);
                    let store_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 3,
                        inputs_offset: off,
                        compute_fn: CF_ARRAY_STORE,
                    });
                    return Some(store_node);
                }
                // `*ref = value` → CF_DEREF_WRITE through the shared Cell.
                // MUST return the node: falling through to None silently
                // dropped the store (this branch was missing entirely — the
                // Assign.rs copy serves a different, unreachable path — so
                // `*r = v` never emitted a node and never executed).
                // `current_effect` rides as a direct trailing input (scheduler
                // ordering only): consecutive deref writes must not fire out
                // of order (the value-side chain does not order the write
                // itself).
                if let crate::ast::Ast::Expr::Deref(ref_inner) = target_expr {
                    let ref_node = self.compile_subexpr(*ref_inner);
                    let (input_count, inputs_offset) = match self.current_effect {
                        Some(eff) => (3, self.graph.inputs_pool.push(&[ref_node, val_node, eff])),
                        None => (2, self.graph.inputs_pool.push(&[ref_node, val_node])),
                    };
                    let write_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count,
                        inputs_offset,
                        compute_fn: CF_DEREF_WRITE,
                    });
                    return Some(write_node);
                }
                if let crate::ast::Ast::Expr::Ident(name) = target_expr {
                    // Cell-backed binding (place model): `x = v` writes
                    // through the shared Cell (same node shape as `*r = v`)
                    // and does NOT rebind the name: reads route through the
                    // cell, and a rebind would fork the two stores (plain
                    // node vs cell). `current_effect` is a direct trailing
                    // input (scheduler ordering only): without it,
                    // consecutive cell writes could fire out of order and
                    // the LAST-scheduled write would win.
                    if let Some(cell_node) = self.lookup_cell_binding(name) {
                        let (input_count, inputs_offset) = match self.current_effect {
                            Some(eff) => (
                                3,
                                self.graph.inputs_pool.push(&[cell_node, val_node, eff]),
                            ),
                            None => (2, self.graph.inputs_pool.push(&[cell_node, val_node])),
                        };
                        let write_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count,
                            inputs_offset,
                            compute_fn: CF_DEREF_WRITE,
                        });
                        // Forwarding memory: this store is now the cell's
                        // known current value.
                        self.track_cell_store(cell_node, val_node);
                        return Some(write_node);
                    }
                    // Implicit-this field assignment: `field = value` inside a method body
                    // resolves to `this.field = value`.
                    if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(*target).cloned() {
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        let off = self.graph.inputs_pool.push(&[this_node, val_node]);
                        let set_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: off,
                            compute_fn: CF_RECORD_FIELD_SET,
                        });
                        self.graph.set_field_set_name(set_node, field.to_string());
                        return Some(set_node);
                    }
                    // B/C deletion (2026-08-22): the WriteBack ladder is gone —
                    // every assigned name is cell-backed (all-vars: locals,
                    // assigned params, captured vars via ④'s shared-cell
                    // upvalues; converters via B2/B3 register cells), so the
                    // cell path above already handled the assignable shapes.
                    // What remains: global stores and plain local rebinds.
                    if let Some(slot) = self.lookup_global_var(name) {
                        // Global variable -> global_store, returning an effect node to ensure scheduled execution
                        let store_node = self.compile_global_store(val_node, slot);
                        return Some(store_node);
                    }
                    self.bind_var(name, val_node);
                }
                None
            }
            crate::ast::Ast::Stmt::FieldAssignment { object, field, value } => {
                let obj_node = self.compile_subexpr(*object);
                let val_node = self.compile_subexpr(*value);
                let inputs_offset = self.graph.inputs_pool.push(&[obj_node, val_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                Some(set_node)
            }
            crate::ast::Ast::Stmt::CompoundAssignment { target, op, value } => {
                let target_expr = &self.current_module().arena.expr(*target).node;
                // `*ref op= value` → deref-read, op, deref-write through the
                // shared Cell (statement-level branch; previously missing, the
                // compound store was silently dropped).
                if let crate::ast::Ast::Expr::Deref(ref_inner) = target_expr {
                    let raw_val = self.compile_subexpr(*value);
                    let val_node = self.chain_effects(self.current_effect, raw_val);
                    let ref_node = self.compile_subexpr(*ref_inner);
                    // Read carries the effect as a direct trailing input —
                    // without it the read can fire before prior writes and
                    // compound from a stale cell value.
                    let (r_count, r_off) = match self.current_effect {
                        Some(eff) => (2, self.graph.inputs_pool.push(&[ref_node, eff])),
                        None => (1, self.graph.inputs_pool.push(&[ref_node])),
                    };
                    let read_node = self.graph.add_node(Node {
                        kind: NodeKind::UnOp,
                        input_count: r_count,
                        inputs_offset: r_off,
                        compute_fn: CF_DEREF_READ,
                    });
                    let bin_compute = self.compound_assign_op_to_compute_fn(*op, *target);
                    let bin_off = self.graph.inputs_pool.push(&[read_node, val_node]);
                    let result_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: bin_off,
                        compute_fn: bin_compute,
                    });
                    // Effect rides as a direct trailing input (scheduler
                    // ordering only) — consecutive deref writes must not fire
                    // out of order.
                    let (w_count, w_off) = match self.current_effect {
                        Some(eff) => (3, self.graph.inputs_pool.push(&[ref_node, result_node, eff])),
                        None => (2, self.graph.inputs_pool.push(&[ref_node, result_node])),
                    };
                    let write_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: w_count,
                        inputs_offset: w_off,
                        compute_fn: CF_DEREF_WRITE,
                    });
                    return Some(write_node);
                }
                if let crate::ast::Ast::Expr::Ident(name) = target_expr {
                    let val_node = self.compile_subexpr(*value);
                    let bin_compute = self.compound_assign_op_to_compute_fn(*op, *target);
                    // Cell-backed binding: `x op= v` — read-modify-write
                    // through the shared Cell; no name rebind (reads route
                    // through the cell).
                    if let Some(cell_node) = self.lookup_cell_binding(name) {
                        // Read carries the effect as a direct trailing input
                        // too: without it the read can fire before prior
                        // cell/deref writes and compound from a stale value.
                        let (r_count, r_off) = match self.current_effect {
                            Some(eff) => (2, self.graph.inputs_pool.push(&[cell_node, eff])),
                            None => (1, self.graph.inputs_pool.push(&[cell_node])),
                        };
                        let read_node = self.graph.add_node(Node {
                            kind: NodeKind::UnOp,
                            input_count: r_count,
                            inputs_offset: r_off,
                            compute_fn: CF_DEREF_READ,
                        });
                        let bin_off = self.graph.inputs_pool.push(&[read_node, val_node]);
                        let raw_result = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: bin_off,
                            compute_fn: bin_compute,
                        });
                        let result_node = self.chain_effects(self.current_effect, raw_result);
                        // Write carries the effect as a direct trailing input
                        // (scheduler ordering only) — see the Assignment
                        // cell branch for why the SEQ wrapper is not enough
                        // for the write itself.
                        let (w_count, w_off) = match self.current_effect {
                            Some(eff) => (
                                3,
                                self.graph.inputs_pool.push(&[cell_node, result_node, eff]),
                            ),
                            None => (2, self.graph.inputs_pool.push(&[cell_node, result_node])),
                        };
                        let write_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: w_count,
                            inputs_offset: w_off,
                            compute_fn: CF_DEREF_WRITE,
                        });
                        self.track_cell_store(cell_node, result_node);
                        return Some(write_node);
                    }
                    // Implicit-this field compound assignment: `field op= value` inside a
                    // method body resolves to `this.field op= value`.
                    if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(*target).cloned() {
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        // Read the current field value.
                        let get_off = self.graph.inputs_pool.push(&[this_node]);
                        let get_node = self.graph.add_node(Node {
                            kind: NodeKind::FieldAccess,
                            input_count: 1,
                            inputs_offset: get_off,
                            compute_fn: CF_RECORD_FIELD_GET,
                        });
                        self.graph.set_field_set_name(get_node, field.to_string());
                        // Operation.
                        let bin_off = self.graph.inputs_pool.push(&[get_node, val_node]);
                        let raw_result = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: bin_off,
                            compute_fn: bin_compute,
                        });
                        let result_node = self.chain_effects(self.current_effect, raw_result);
                        // Write back.
                        let set_off = self.graph.inputs_pool.push(&[this_node, result_node]);
                        let set_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: set_off,
                            compute_fn: CF_RECORD_FIELD_SET,
                        });
                        self.graph.set_field_set_name(set_node, field.to_string());
                        return Some(set_node);
                    }
                    // B/C deletion: the WriteBack ladder is gone (all assigned
                    // names are cell-backed; the cell store path higher up
                    // handles them). Remaining: globals store, plain rebind.
                    // Read current value: local var > global var > placeholder
                    let cur_node = if let Some(n) = self.lookup_var(name) {
                        n
                    } else if let Some(slot) = self.lookup_global_var(name) {
                        self.compile_global_load(slot)
                    } else {
                        self.compile_placeholder()
                    };
                    let off = self.graph.inputs_pool.push(&[cur_node, val_node]);
                    let raw_result = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: off,
                        compute_fn: bin_compute,
                    });
                    // Link current_effect: prevents a compound assignment after continue from running early
                    let result_node = self.chain_effects(self.current_effect, raw_result);
                    if self.lookup_global_var(name).is_some() && self.lookup_var(name).is_none() {
                        // Global variable -> global_store. Return the store node so it is chained
                        // into the block's effect chain (last_effect), otherwise the store would be
                        // orphaned and dropped (the global has no local binding to keep it alive).
                        let slot = self.lookup_global_var(name).unwrap();
                        let store_node = self.compile_global_store(result_node, slot);
                        self.current_effect = Some(store_node);
                        Some(store_node)
                    } else {
                        self.bind_var(name, result_node);
                        None
                    }
                } else {
                    // Non-Ident target (FieldAccess/Index/Deref): delegate to
                    // compile_compound_assign which handles read-modify-write for these.
                    let set_node = self.compile_compound_assign(*op, *target, *value);
                    self.current_effect = Some(set_node);
                    Some(set_node)
                }
            }
            crate::ast::Ast::Stmt::Return { value } => {
                let prev_effect = self.current_effect;
                let return_val_node = match value {
                    Some(expr_id) => {
                        let prev_tail = self.in_tail_position;
                        self.in_tail_position = true;
                        let r = self.compile_expr(*expr_id);
                        self.in_tail_position = prev_tail;
                        r
                    }
                    None => self.compile_void_const(),
                };
                // CF_RETURN: inputs[0] = return value, inputs[1] = prior side-effect dependency (optional)
                // The prior side-effect dependency ensures the return signal fires only after prior statements complete
                let (off, count) = match prev_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[return_val_node, eff]), 2),
                    None => (self.graph.inputs_pool.push(&[return_val_node]), 1),
                };
                let return_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_RETURN,
                });
                Some(return_node)
            }
            crate::ast::Ast::Stmt::Throw { expr } => {
                let prev_effect = self.current_effect;
                let expr_node = self.compile_subexpr(*expr);
                // CF_THROW_WRAP_ERR: inputs[0] = thrown value, inputs[1] = prior side-effect dependency (optional)
                // compute_throw_wrap_err directly returns NodeResult::Return(ThrowVal(Err(v)))
                let (off, count) = match prev_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[expr_node, eff]), 2),
                    None => (self.graph.inputs_pool.push(&[expr_node]), 1),
                };
                let wrap_node = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_THROW_WRAP_ERR,
                });
                Some(wrap_node)
            }
            crate::ast::Ast::Stmt::Break => {
                // CF_BREAK: optional inputs[0] = prior side-effect dependency
                let (off, count) = match self.current_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[eff]), 1),
                    None => (self.graph.inputs_pool.push(&[]), 0),
                };
                let n = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_BREAK,
                });
                Some(n)
            }
            crate::ast::Ast::Stmt::Continue => {
                // CF_CONTINUE: optional inputs[0] = prior side-effect dependency
                // The engine-side complete_and_wake_caller detects Continue -> reset_loop_iteration for the next round
                // (Sema guarantees continue is always inside a loop)
                let (off, count) = match self.current_effect {
                    Some(eff) => (self.graph.inputs_pool.push(&[eff]), 1),
                    None => (self.graph.inputs_pool.push(&[]), 0),
                };
                let n = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: count,
                    inputs_offset: off,
                    compute_fn: CF_CONTINUE,
                });
                Some(n)
            }
            crate::ast::Ast::Stmt::While { condition, body } => {
                let while_sg = self.register_while_subgraph(*condition, *body);
                // Launch call with the loop-carried params' initial values
                // (args FIRST — compute_call_launch takes inputs[..param_count]
                // — then the effect dependency).
                let entry_args = std::mem::take(&mut self.while_entry_args);
                let mut call_inputs = entry_args;
                if let Some(eff) = self.current_effect {
                    call_inputs.push(eff);
                }
                let inputs_off = self.graph.inputs_pool.push(&call_inputs);
                let call_node = self.graph.add_node(Node {
                    kind: NodeKind::Call,
                    input_count: call_inputs.len() as u8,
                    inputs_offset: inputs_off,
                    compute_fn: CF_CALL_LAUNCH,
                });
                self.graph.set_call_target(call_node, while_sg);
                Some(call_node)
            }
            crate::ast::Ast::Stmt::Loop { body } => {
                let loop_sg = self.register_loop_subgraph(*body);
                let call_node = self.compile_recursive_call(loop_sg);
                Some(call_node)
            }
            crate::ast::Ast::Stmt::For {
                name,
                iterable,
                body,
            } => {
                // For loop = iterable (already an iterator) -> recursive subgraph (next() + is_null + body)
                let iterable_node = self.compile_subexpr(*iterable);
                // Obtain iterable type info from Sema (type name + whether it is a trait object)
                let (iter_type_name, is_trait_object) = self.lookup_expr_iter_info(*iterable);
                // Register the For loop subgraph (static dispatch: bind next() by type name; trait objects go through vtable)
                let for_sg = self.register_for_subgraph(
                    name,
                    *body,
                    iter_type_name.as_deref(),
                    is_trait_object,
                );
                // Start the loop: Call(for_sg, [iterable_node])
                let call_node = self.make_call(for_sg, &[iterable_node]);
                Some(call_node)
            }
            crate::ast::Ast::Stmt::Defer { expr } => {
                if self.in_loop_body {
                    // Defer-in-loop: compile as CF_DEFER_REGISTER node.
                    // The defer body subgraph + captured values are pushed onto
                    // the loop frame's defer_stack at runtime; CF_DEFER_RUN (in void_sg) drains
                    // it in LIFO order at loop exit.
                    // Defer barrier: the body runs at exit time — its reads
                    // must LOAD (no forwarding of compile-time values) and its
                    // stores must not leak into subsequent forwarding.
                    let defer_barrier = self.cell_barrier_enter();
                    let (body_sg, _captured_inputs) = self.compile_branch_subgraph(*expr);
                    self.cell_barrier_exit_defer(defer_barrier);
                    // Unified capture model: snapshot the loop variable (if any)
                    // and any Snapshot-mode captures, so each defer body reads
                    // per-iteration values rather than final values.
                    // Reference-mode captures (var bindings like an accumulator)
                    // are NOT snapshotted here — they are read live via the
                    // frame chain at defer-run time, so successive loop
                    // iterations' defers accumulate correctly (LIFO over the
                    // shared latest value).
                    let loop_var = self.loop_stack.last().and_then(|lc| lc.loop_var_node);
                    let sema_captures = self.lookup_captures(*expr);
                    let mut inputs: Vec<NodeId> = Vec::new();
                    if let Some(n) = loop_var {
                        inputs.push(n);
                    }
                    for cap in sema_captures {
                        // Only Snapshot-mode captures need per-iteration
                        // snapshotting; Reference-mode captures are read live.
                        if cap.mode != crate::sema::Sema::CaptureMode::Snapshot {
                            continue;
                        }
                        if let Some(node) = self.lookup_var(cap.name.as_ref()) {
                            if !inputs.contains(&node) {
                                inputs.push(node);
                            }
                        }
                    }
                    let inputs_off = self.graph.inputs_pool.push(&inputs);
                    let reg_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: inputs.len() as u8,
                        inputs_offset: inputs_off,
                        compute_fn: CF_DEFER_REGISTER,
                    });
                    self.graph.set_call_target(reg_node, body_sg);
                    Some(reg_node)
                } else {
                    // Function-level defer: execution-gated dynamic registration.
                    //
                    // Historically this pushed a static entry onto the function
                    // subgraph's defer_table, which every frame-exit path drained
                    // unconditionally — a defer whose statement was never reached
                    // (error `?`-exit BEFORE the binding it captures) still ran,
                    // reading an unbound frame slot; with an await inside the
                    // body that fed garbage into the async machinery and crashed
                    // natively (silent exit 127 on the second occurrence). Now
                    // the register node sits in the statement stream, so only
                    // reached defers run; it pushes onto the FUNCTION frame's
                    // runtime defer_stack (see compute_block_defer_register),
                    // which finish_frame / run_defers_sync drain at frame exit.
                    //
                    // The ONLY input is the effect dependency, captured BEFORE
                    // compile_branch_subgraph: that helper compiles the body
                    // expression into the parent's node space first (then carves
                    // the same_function branch range out of it) and leaves
                    // current_effect pointing at a node INSIDE that nested range
                    // — a node the parent frame never executes, which would leave
                    // this register permanently unready. Captured before, it
                    // anchors the register after prior statements (await/call
                    // bindings chain effects, so `val f = ...await()?; defer
                    // f.close()` orders correctly). Captured variables are
                    // deliberately NOT scheduling inputs (same nested-range
                    // hazard); defer bodies read outer variables live via the
                    // frame chain at drain time (Bug #47).
                    let eff_input = self.current_effect;
                    // Defer barrier (same as the loop-defer site): reads load,
                    // stores don't leak into subsequent forwarding.
                    let defer_barrier = self.cell_barrier_enter();
                    let (body_sg, _branch_captures) = self.compile_branch_subgraph(*expr);
                    self.cell_barrier_exit_defer(defer_barrier);
                    // Bug #49 flag: mark the function sg so later local reassignments
                    // emit WriteBacks (defer bodies read the LATEST value).
                    if let Some(fn_sg) = self.current_function_sg {
                        self.function_defer_sgs.insert(fn_sg);
                    }
                    let mut inputs: Vec<NodeId> = Vec::with_capacity(1);
                    if let Some(eff) = eff_input {
                        inputs.push(eff);
                    }
                    let inputs_off = self.graph.inputs_pool.push(&inputs);
                    let reg_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: inputs.len() as u8,
                        inputs_offset: inputs_off,
                        compute_fn: CF_BLOCK_DEFER_REGISTER,
                    });
                    self.graph.set_call_target(reg_node, body_sg);
                    Some(reg_node)
                }
            }
            crate::ast::Ast::Stmt::LocalDecl { decl } => {
                match decl.as_ref() {
                    crate::ast::Ast::Decl::FunDecl {
                        name, params, body, is_async, extern_c_body, ..
                    } => {
                        if extern_c_body.is_some() {
                            return None;
                        }
                        let construct_node =
                            self.compile_lambda(params, *body, *is_async, Some(name), None);
                        self.bind_var(name, construct_node);
                        Some(construct_node)
                    }
                    crate::ast::Ast::Decl::TypeDecl { name, def, .. } => {
                        // Register nested type fields into the current scope (unified with top-level types via type_scope_stack lookup).
                        // Canonical type name (module-qualified for user modules) —
                        // matches registration and the runtime identity.
                        let canonical: String = self.sema.resolve_type_key(name);
                        match def {
                            crate::ast::Ast::TypeDef::Record { fields } => {
                                let field_names: Vec<String> = fields.iter().map(|f| f.name.to_string()).collect();
                                self.bind_type_fields(name, TypeFieldInfo {
                                    field_names,
                                    type_name: canonical.clone(),
                                    kind: RecordLitKind::Record,
                                });
                            }
                            crate::ast::Ast::TypeDef::Adt { constructors } => {
                                // Register the type name + each constructor name (mapped to the type name)
                                self.bind_type_fields(name, TypeFieldInfo {
                                    field_names: Vec::new(),
                                    type_name: canonical.clone(),
                                    kind: RecordLitKind::Adt,
                                });
                                for ctor in constructors {
                                    let field_names: Vec<String> = ctor.fields.iter()
                                        .map(|f| f.name.unwrap_or("_").to_string())
                                        .collect();
                                    self.bind_type_fields(ctor.name, TypeFieldInfo {
                                        field_names,
                                        type_name: canonical.clone(),
                                        kind: RecordLitKind::Adt,
                                    });
                                }
                            }
                            crate::ast::Ast::TypeDef::Newtype { name: nt_name, .. } => {
                                self.bind_type_fields(nt_name, TypeFieldInfo {
                                    field_names: Vec::new(),
                                    type_name: canonical.clone(),
                                    kind: RecordLitKind::Newtype,
                                });
                            }
                            crate::ast::Ast::TypeDef::Alias { .. } => {}
                        }
                        None
                    }
                    // Trait declaration: Sema registers the type; the IR layer generates no code
                    crate::ast::Ast::Decl::TraitDecl { .. } => None,
                    _ => None,
                }
            }
        }
    }

}
