//! Loops — For / while loop subgraph registration. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Look up an expression's type information from sema (used for For-loop dispatch
    /// decisions).
    /// Returns `(type name, is_trait_object)`:
    /// - `(Some("RangeIterator"), false)` -> static dispatch to `"RangeIterator.next"`.
    /// - `(Some("Iterator"), true)` -> vtable dynamic dispatch (an inline_trait value).
    /// - `(None, false)` -> type inference failed; fall back to vtable dispatch.
    pub(super) fn lookup_expr_iter_info(&self, expr: crate::ast::Ast::ExprId) -> (Option<String>, bool) {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            return (info.type_name.as_deref().map(|s| s.to_string()), info.is_trait_object);
        }
        (None, false)
    }

    /// Register the For-loop subgraph (a recursive subgraph; `param_count=1` to receive the
    /// iterator).
    ///
    /// Structure:
    /// - `for_sg` (`param_count=1`): receives the iterator.
    ///   - `param_0` = iterator.
    ///   - `next_call` = `Call("Iterator.next", [param_0])` // returns `T?`.
    ///   - `is_null_node` = `UnOp(is_null, [next_call])`.
    ///   - `body_sg` (`param_count=2`): iterator + current value (bind name, compile body,
    ///     tail-recurses into `for_sg`).
    ///   - `void_sg` (`param_count=0`): exit.
    ///   - `gate` = `Gate(is_null_node)`: `true` -> `void_sg` (exit), `false` -> `body_sg`
    ///     (continue).
    ///
    /// Execution: `next()` returns non-null -> `body_sg` executes and then tail-recurses into
    /// `for_sg`; returns null -> `void_sg` exits.
    /// A Break signal terminates the `body_sg` frame -> the Gate completes -> `for_sg` ends.
    /// Continue is compiled as `Call(for_sg, [iter_param]) + Return` signal -> tail-recurses
    /// into the next iteration.
    pub(super) fn register_for_subgraph(
        &mut self,
        name: &str,
        body: crate::ast::Ast::ExprId,
        iter_type_name: Option<&str>,
        is_trait_object: bool,
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        // Placeholder registration (reserve the id first to allow recursive references).
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 1,
            entry_node: NodeId(node_start),
            return_node: NodeId(node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });

        // param_0 = iterator.
        let iter_off = self.graph.inputs_pool.push(&[]);
        let iter_param = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset: iter_off,
            compute_fn: CF_NOOP,
        });

        // next_call = Call("{iter_type_name}.next", [iter_param]) -> T?
        // Dynamic dispatch: when `is_trait_object=true`, go through the vtable (look up `next` at
        // runtime from the TraitVal).
        // Static dispatch: concrete types are bound by mangled type name (e.g. "ArrayIter.next").
        // Fallback: type inference failed (None) -> vtable dispatch.
        let next_call = if is_trait_object || iter_type_name.is_none() {
            let trait_name = iter_type_name.as_deref().unwrap_or("Iterator");
            self.make_vtable_call(iter_param, trait_name, "next")
        } else {
            let next_method = format!("{}.next", iter_type_name.unwrap());
            self.make_call_by_name(&next_method, &[iter_param])
        };

        // is_null_node = UnOp(is_null, [next_call]).
        let is_null_off = self.graph.inputs_pool.push(&[next_call]);
        let is_null_node = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset: is_null_off,
            compute_fn: CF_IS_NULL, // is_null
        });

        // body_sg (param_count=2: iterator + current value).
        // Reset `current_effect = None` (same as `register_while_subgraph`, to avoid an
        // external effect dependency causing a deadlock after `reset_loop_iteration` of the
        // loop-body frame).
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let body_sg = self.compile_for_body_subgraph(body, sg_id, name);

        // void_sg (exit): includes CF_DEFER_RUN to drain defer-in-loop entries at loop exit.
        let void_sg = self.compile_defer_run_subgraph();
        self.current_effect = prev_effect;

        // gate = Gate(is_null_node): true -> void_sg, false -> body_sg (inputs=[iter_param,
        // next_call]).
        let gate_off = self.graph.inputs_pool.push(&[is_null_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                capture: false,
                condition_input: is_null_node,
                branches: vec![
                    (true, void_sg, vec![]),
                    (false, body_sg, vec![iter_param, next_call]),
                ],
            },
        );

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::For;
        sg.cond_node = Some(is_null_node);
        sg.iter_next_node = Some(next_call);
        sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![next_call],
            reset_to_one: vec![is_null_node],
            reset_condition_tree: vec![],
            fused_carries: Vec::new(),
            condition_tree_plan: Vec::new(),
            carries_value: Vec::new(),
            carries_cell: Vec::new(),
        });
        sg_id
    }

    /// Compile the For-loop body subgraph (`param_count=2`: iterator + current value).
    ///
    /// - `param_0` = iterator (for tail recursion).
    /// - `param_1` = current value (bound to the loop variable `name`).
    /// - Compiles `body`; at the end emits a tail-recursive `Call(for_sg, [param_0])` (depends
    ///   on `body_last` to preserve ordering).
    pub(super) fn compile_for_body_subgraph(
        &mut self,
        body: crate::ast::Ast::ExprId,
        for_sg: SubGraphId,
        name: &str,
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;

        // param_0 = iterator (a node inside body_sg, injected by the Gate branch inputs).
        let iter_off = self.graph.inputs_pool.push(&[]);
        let iter_param = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset: iter_off,
            compute_fn: CF_NOOP,
        });

        // param_1 = current value (bound to the loop variable `name`).
        let val_off = self.graph.inputs_pool.push(&[]);
        let val_param = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset: val_off,
            compute_fn: CF_NOOP,
        });

        self.enter_scope();
        self.bind_var(name, val_param);
        self.loop_stack.push(LoopContext {
            sg: for_sg,
            iter_node: Some(iter_param),
            body_node_start: node_start,
            loop_var_node: Some(val_param),
        });

        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        self.cell_barrier_enter();
        let prev_in_loop = self.in_loop_body;
        self.in_loop_body = true;
        let body_last = self.compile_expr(body);
        self.in_loop_body = prev_in_loop;
        self.current_sg_start = prev_sg_start;
        self.cell_barrier_exit();

        self.loop_stack.pop();
        self.exit_scope();

        // Tail-recursion eliminated: `return_node = body_last`; frame reuse is handled by the
        // Engine's `reset_loop_iteration`.
        let node_end = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_end)),
            param_count: 2,
            entry_node: NodeId(node_start),
            return_node: body_last,
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::LoopBody,
            loop_parent_sg: Some(for_sg),
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        sg_id
    }

    /// Register the While-loop subgraph (a recursive subgraph).
    ///
    /// Structure:
    /// - `cond_node` = `compile_expr(condition)`.
    /// - `gate_node` = `Gate(cond)`: `true` -> `body_sg` (tail recursion), `false` -> `void_sg`
    ///   (exit).
    /// - `body_sg`: compiles `body`; at the end emits a `Call` back to `while_sg` (depends on
    ///   the body's trailing node to preserve ordering).
    ///
    /// Execution: when `cond` is true, `body_sg` executes and then tail-recurses into
    /// `while_sg`; when false, `void_sg` exits.
    /// A Break signal terminates the `body_sg` frame -> `while_sg`'s Gate completes -> the loop
    /// ends.
    /// Continue is compiled as `Call(while_sg) + Return` signal -> tail-recurses into the next
    /// iteration (skipping the rest of the body).
    pub(super) fn register_while_subgraph(
        &mut self,
        condition: crate::ast::Ast::ExprId,
        body: crate::ast::Ast::ExprId,
    ) -> SubGraphId {
        // ── Place-model phi: loop-carried cell params ──
        // Cells assigned in the body become while-sg PARAMS: the entry call
        // passes their current values; each iteration reset copies the body's
        // final values back into the param slots (ResetPlan carries, applied
        // in reset_loop_iteration — which also pokes the condition chain).
        // Condition AND body reads forward to the param nodes (zero-cost
        // edges): the per-iteration loop-carried LOAD is eliminated — the
        // transport is the engine's native slot injection, not a heap read.
        // Entry-arg nodes are emitted in the ENCLOSING frame (before the sg's
        // node range): the forwarded value node when known, else a one-time
        // cell load.
        let mut assigned_names = rustc_hash::FxHashSet::default();
        collect_assigned_names(&self.current_module().arena, body, &mut assigned_names);
        let mut carried_cells: Vec<NodeId> = Vec::new();
        let entry_args: Vec<NodeId> = carried_cells
            .iter()
            .map(|&cell| match self.cell_forwarded_value(cell) {
                Some(v) => v,
                None => {
                    // Outer-loop barrier cleared the memory: one-time load of
                    // the cell in the enclosing frame (effect-ordered like
                    // every cell read).
                    let (count, off) = match self.current_effect {
                        Some(eff) => (2, self.graph.inputs_pool.push(&[cell, eff])),
                        None => (1, self.graph.inputs_pool.push(&[cell])),
                    };
                    self.graph.add_node(Node {
                        kind: NodeKind::UnOp,
                        input_count: count,
                        inputs_offset: off,
                        compute_fn: CF_DEREF_READ,
                    })
                }
            })
            .collect();
        // Expose the entry args for the caller's launch call (Stmt.rs);
        // restored at the end (nested loops clobber the field).
        self.while_entry_args = entry_args.clone();

        let node_start = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        // Placeholder registration (reserve the id first to allow recursive references).
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: carried_cells.len() as u8,
            entry_node: NodeId(node_start),
            return_node: NodeId(node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        // Param placeholder Consts: the FIRST nodes of the sg (positional
        // convention — engine injects call/carry values into these slots).
        let param_nodes: Vec<NodeId> = carried_cells
            .iter()
            .map(|_| {
                let off = self.graph.inputs_pool.push(&[]);
                self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset: off,
                    compute_fn: CF_NOOP,
                })
            })
            .collect();

        // Place-model forwarding barrier (loop-like): the condition and body
        // re-evaluate every iteration — reads must not forward pre-loop
        // values (stale from iteration 2 on). Clears for the rest of the
        // function; the body's own stores forward within one iteration.
        self.cell_barrier_enter();
        // Seed the forwarding memory: reads of carried cells (condition AND
        // body) forward to the param nodes — the injected slot IS the current
        // value, refreshed per iteration by the carries.
        for (i, &cell) in carried_cells.iter().enumerate() {
            self.cell_values.insert(cell, param_nodes[i]);
        }
        // Compile the condition.
        // Reset `current_effect = None` to avoid creating CF_SEQ nodes inside the loop subgraph
        // that depend on the external effect chain.
        // After `reset_loop_iteration`, the loop-body frame's value table is cleared and external
        // effect nodes are not re-copied; CF_SEQ nodes depending on them would stay pending
        // forever, causing a deadlock.
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let mut cond_node = self.compile_subexpr(condition);
        // A bare-variable condition (`while cont`) compiles to the variable's
        // EXISTING binding node, which physically lives in the enclosing
        // function body — OUTSIDE this while subgraph. The per-iteration
        // reset_condition_tree DFS only collects in-sg nodes, so an external
        // cond_node leaves the loop with nothing to re-evaluate: after the
        // first body round the ready queue stays empty and the loop silently
        // exits — unless a WriteBack happens to poke the gate, which is why
        // loops whose condition variable is (re)written every round masked
        // the bug (#104 layer 2). Wrap external conditions in an in-sg
        // identity node (CF_SEQ) so each iteration re-reads the current slot
        // value and re-fires the gate.
        if cond_node.0 < node_start {
            let off = self.graph.inputs_pool.push(&[cond_node]);
            cond_node = self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset: off,
                compute_fn: CF_SEQ,
            });
        }
        // body subgraph (trailing tail-recursive call to while_sg).
        let body_sg = self.compile_loop_body_subgraph(body, sg_id, false);
        // void subgraph (false branch; loop ends): includes CF_DEFER_RUN for defer-in-loop.
        let void_sg = self.compile_defer_run_subgraph();
        self.current_effect = prev_effect;
        // Phi carries: the body's final value per carried cell — the store's
        // value node when statically known (unconditional store: a node that
        // re-fires every iteration), else the CELL itself (conditional store:
        // the engine derefs through it at reset time).
        let mut carries_value: Vec<(NodeId, NodeId)> = Vec::new();
        let mut carries_cell: Vec<(NodeId, NodeId)> = Vec::new();
        for (i, &cell) in carried_cells.iter().enumerate() {
            match self.cell_values.get(&cell) {
                Some(&v) if v != param_nodes[i] => carries_value.push((param_nodes[i], v)),
                _ => carries_cell.push((param_nodes[i], cell)),
            }
        }
        self.cell_barrier_exit();

        // Gate node: cond true -> body_sg, false -> void_sg.
        let gate_off = self.graph.inputs_pool.push(&[cond_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                capture: false,
                condition_input: cond_node,
                branches: vec![
                    (true, body_sg, vec![]),
                    (false, void_sg, vec![]),
                ],
            },
        );

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::While;
        sg.cond_node = Some(cond_node);
        sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![],
            reset_to_one: vec![],
            reset_condition_tree: vec![cond_node],
            fused_carries: Vec::new(),
            condition_tree_plan: Vec::new(),
            carries_value,
            carries_cell,
        });
        // Nested loops compiled inside the body clobbered the field — restore
        // so THIS loop's caller reads the right entry args.
        self.while_entry_args = entry_args;
        sg_id
    }

}

/// True when `body` contains a `continue` belonging to THIS loop (not one
/// inside a nested while/loop/for — those bind their own continues). Such
/// bodies jump via a no-arg Call(while_sg), which cannot carry phi params.
pub(super) fn body_has_direct_continue(
    arena: &crate::ast::Ast::AstArena<'_>,
    expr: crate::ast::Ast::ExprId,
) -> bool {
    expr_has_direct_continue(arena, expr)
}

fn expr_has_direct_continue(arena: &crate::ast::Ast::AstArena<'_>, expr: crate::ast::Ast::ExprId) -> bool {
    use crate::ast::Ast;
    match &arena.expr(expr).node {
        Ast::Expr::Block { stmts, trailing } => {
            stmts.iter().any(|&s| stmt_has_direct_continue(arena, s))
                || trailing.map(|t| expr_has_direct_continue(arena, t)).unwrap_or(false)
        }
        Ast::Expr::If { cond, then_branch, else_branch } => {
            expr_has_direct_continue(arena, *cond)
                || expr_has_direct_continue(arena, *then_branch)
                || else_branch.map(|e| expr_has_direct_continue(arena, e)).unwrap_or(false)
        }
        Ast::Expr::Match { scrutinee, arms } => {
            expr_has_direct_continue(arena, *scrutinee)
                || arms.iter().any(|a| {
                    a.guard.map(|g| expr_has_direct_continue(arena, g)).unwrap_or(false)
                        || expr_has_direct_continue(arena, a.body)
                })
        }
        // Nested loops bind their own continues — do not descend.
        _ => false,
    }
}

fn stmt_has_direct_continue(arena: &crate::ast::Ast::AstArena<'_>, stmt: crate::ast::Ast::StmtId) -> bool {
    use crate::ast::Ast;
    match &arena.stmt(stmt).node {
        Ast::Stmt::Continue => true,
        Ast::Stmt::ValDecl { value, .. }
        | Ast::Stmt::VarDecl { value, .. }
        | Ast::Stmt::Expression { expr: value, .. }
        | Ast::Stmt::Throw { expr: value } => expr_has_direct_continue(arena, *value),
        Ast::Stmt::Assignment { target, value } | Ast::Stmt::CompoundAssignment { target, value, .. } => {
            expr_has_direct_continue(arena, *target) || expr_has_direct_continue(arena, *value)
        }
        Ast::Stmt::FieldAssignment { object, value, .. } => {
            expr_has_direct_continue(arena, *object) || expr_has_direct_continue(arena, *value)
        }
        Ast::Stmt::Return { value } => value.map(|v| expr_has_direct_continue(arena, v)).unwrap_or(false),
        Ast::Stmt::Defer { expr } => expr_has_direct_continue(arena, *expr),
        Ast::Stmt::While { condition, body } => {
            expr_has_direct_continue(arena, *condition) || expr_has_direct_continue(arena, *body)
        }
        Ast::Stmt::For { iterable, body, .. } => {
            expr_has_direct_continue(arena, *iterable) || expr_has_direct_continue(arena, *body)
        }
        Ast::Stmt::Loop { body } => expr_has_direct_continue(arena, *body),
        Ast::Stmt::Break => false,
        Ast::Stmt::LocalDecl { .. } => false,
    }
}
