//! Recursion — Recursion-to-loop transforms (tail & non-tail) + loop-body subgraphs. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Tail-recursion to iteration: consumes `TailRecInfo` to construct the `while_sg` IR.
    ///
    /// Called by `compile_function` when it detects `MemoStrategy::TailRecToLoop`.
    /// The parameter nodes have already been created and `bind_var`'d by `compile_function`;
    /// this method constructs the loop structure.
    ///
    /// Structure:
    /// - `while_sg`: `cond = NOT(base_case condition)`, `Gate(cond)` -> `body_sg` / `exit_sg`.
    /// - `body_sg`: compiles the original function body (`tail_rec_ctx` intercepts tail calls as
    ///   `WriteBack + Call(while_sg)`).
    /// - `exit_sg`: compiles the base-case return value.
    pub(super) fn compile_tail_rec_to_loop(
        &mut self,
        name: &str,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        info: &crate::pass::Analyzer::TailRecInfo,
    ) -> NodeId {
        // L3' slot transport: eligible shapes ride PARAM slots + phi carries
        // instead of parameter-register Cells.
        if self.tail_rec_slot_eligible(params, info) {
            return self.compile_tail_rec_to_loop_slots(name, body_expr, params, info);
        }

        // 1. Collect parameter nodes (already `bind_var`'d by `compile_function`).
        let param_nodes: Vec<NodeId> = params
            .iter()
            .filter_map(|p| self.lookup_var(p.name))
            .collect();

        // 1b. B2 parameter-register cells: each param gets a Cell initialized
        // with the incoming argument. The names re-bind to the cells, so body
        // reads / condition reads route through CF_DEREF_READ (live per
        // iteration) and tail-call stores write CF_DEREF_WRITE — the old
        // WriteBack param registers are gone. The allocs chain into the
        // effect stream ahead of the while Call (same ordering contract as
        // the ③ assigned-param cells).
        let mut param_cells: Vec<NodeId> = Vec::with_capacity(param_nodes.len());
        for (param, &pn) in params.iter().zip(param_nodes.iter()) {
            let off = self.graph.inputs_pool.push(&[pn]);
            let cell_node = self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset: off,
                compute_fn: CF_CELL_ALLOC,
            });
            self.bind_cell(param.name, cell_node);
            self.track_cell_decl(param.name, cell_node, pn);
            self.current_effect = Some(self.chain_effects(self.current_effect, cell_node));
            param_cells.push(cell_node);
        }

        // 2. Placeholder-register while_sg.
        let node_start = self.graph.nodes.len() as u32;
        let while_sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: while_sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 0,
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

        // 3. Build the loop condition `cond_node` (within while_sg's node_range).
        // Forwarding barrier first (loop-like): the condition re-evaluates every
        // iteration — reads of the param cells must be real loads, not forwards
        // of the initial argument nodes (stale from iteration 2 on).
        self.cell_barrier_enter();
        let cond_node = self.build_tail_rec_cond(&info.base_cases, &info.rec_branches);

        // 4. Set `tail_rec_ctx` (`compile_call` intercepts self-calls as
        //    cell stores + Continue barrier).
        self.tail_rec_ctx = Some(TailRecCtx {
            self_name: name.to_string(),
            param_cells,
            slot_params: Vec::new(),
        });

        // 5. Compile body_sg: compiles the original function body (LoopBody; after completion
        //    `reset_loop_iteration` auto-jumps back).
        //    A tail call `self(args)` is intercepted by `compile_call` as a WriteBack (no Call,
        //    no tail_call).
        //    The base-case path is also compiled, but `cond` guarantees it never executes (DCE
        //    can eliminate it).
        //    Force `in_tail_position = true`: a void function's `in_tail_position` defaults to
        //    `false` (line 5208 of `compile_function`: `!is_void_fn`), but in the body_sg of a
        //    tail-recursion transform, self-calls must be intercepted in tail position as a
        //    WriteBack; otherwise a genuine recursive Call node would be generated, causing an
        //    infinite loop (the loop condition is based on the initial parameter values, which
        //    are never updated).
        let prev_effect = self.current_effect;
        let prev_tail = self.in_tail_position;
        self.current_effect = None;
        self.in_tail_position = true;
        let body_sg = self.compile_loop_body_subgraph(body_expr, while_sg_id, true);
        self.in_tail_position = prev_tail;
        self.current_effect = prev_effect;

        // 6. Clear `tail_rec_ctx`.
        self.tail_rec_ctx = None;

        // 7. Compile exit_sg: compiles the base-case return value.
        //    v1 supports a single base_case: directly compile the return-value expression.
        //    For multiple base_cases, take the first one with a condition (`cond` guarantees
        //    only one holds).
        let exit_expr = info.base_cases
            .iter()
            .find(|(c, _)| c.is_some())
            .or_else(|| info.base_cases.first())
            .map(|(_, ret)| *ret)
            .unwrap_or(body_expr);
        let (exit_sg, exit_inputs) = self.compile_branch_subgraph(exit_expr);

        // 8. Gate(cond): true -> body_sg, false -> exit_sg.
        let gate_inputs: Vec<NodeId> = match self.current_effect {
            Some(eff) => vec![cond_node, eff],
            None => vec![cond_node],
        };
        let gate_off = self.graph.inputs_pool.push(&gate_inputs);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: gate_inputs.len() as u8,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                capture: false,
                condition_input: cond_node,
                branches: vec![
                    (true, body_sg, Vec::new()),
                    (false, exit_sg, exit_inputs),
                ],
            },
        );

        // 9. Fill in while_sg metadata.
        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[while_sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::TailRec;
        sg.cond_node = Some(cond_node);

        // 10. Create a Call node to launch while_sg (consistent with
        //     `register_while_subgraph` + `compile_recursive_call`).
        //     while_sg runs as a `same_function` subgraph frame; after body_sg completes,
        //     `reset_loop_iteration` reads while_sg's `loop_kind=While` + `cond_node` to correctly
        //     reset the loop.
        //     If `gate_node` were returned directly, while_sg's nodes would execute in the
        //     function's main subgraph frame, where `reset_loop_iteration` would read
        //     `loop_kind=None` -> loop reset would fail.
        let call_node = self.compile_recursive_call(while_sg_id);
        call_node
    }

    /// L3' eligibility: the shape where params can ride while_sg PARAM slots
    /// + ResetPlan phi carries instead of parameter-register Cells.
    ///
    /// - exactly one base case WITH a synthesizable condition and exactly one
    ///   rec branch (a single tail-call site — the carry source is unambiguous;
    ///   multiple sites would leave the engine no statically known final node);
    /// - the recorded condition, every rec arg, and the base return expr are
    ///   pure and reference nothing but the params: they compile in the while
    ///   frame / exit sg where body-local bindings do not exist, and a
    ///   condition not parameter-derived could diverge from the body's own if;
    /// - no param is cell-backed at entry (assigned params, ③), lambda-captured
    ///   or address-taken — those already ride entry Cells.
    fn tail_rec_slot_eligible(
        &self,
        params: &[crate::ast::Ast::Param<'_>],
        info: &crate::pass::Analyzer::TailRecInfo,
    ) -> bool {
        use crate::ast::Ast;
        if info.base_cases.len() != 1 || info.rec_branches.len() != 1 {
            return false;
        }
        let (base_cond, base_ret) = info.base_cases[0];
        if base_cond.is_none() {
            return false;
        }
        for param in params {
            if self.lookup_var(param.name).is_none() {
                return false;
            }
            if self.lookup_cell_binding(param.name).is_some() {
                return false;
            }
            if self.fn_address_taken.contains(param.name)
                || self.fn_lambda_captured.contains(param.name)
            {
                return false;
            }
        }
        let arena = &self.current_module().arena;
        let mut param_names = rustc_hash::FxHashSet::default();
        for param in params {
            param_names.insert(param.name);
        }
        if !expr_params_only_pure(arena, base_cond.unwrap(), &param_names) {
            return false;
        }
        if !expr_params_only_pure(arena, base_ret, &param_names) {
            return false;
        }
        info.rec_branches[0]
            .1
            .iter()
            .all(|&a| expr_params_only_pure(arena, a, &param_names))
    }

    /// L3' slot transport — the Cell-free tail-rec loop.
    ///
    /// Contrast with the Cell path above:
    /// - params ride while_sg PARAM slots: the entry call injects the incoming
    ///   args; each iteration's `ResetPlan.carries_value` copies the
    ///   (speculatively hoisted) next-iteration args back into the slots.
    ///   Zero heap Cells, zero deref loads/stores per iteration.
    /// - the single tail call lowers to a bare void node: no stores and NO
    ///   Continue barrier. `loop_kind` is While, so the body's normal
    ///   completion IS the continue signal; the loop exits through the gate's
    ///   false branch (exit_sg compiles the base-case value).
    /// - the rec args are hoisted into the while frame and chained into the
    ///   condition root as ordering deps, so the per-iteration condition-tree
    ///   reset re-fires them BEFORE the gate can relaunch the body, and their
    ///   slots (mirrored into the body frame at launch) are the carry sources.
    ///
    /// Eligibility (`tail_rec_slot_eligible`) guarantees cond/args/exit
    /// reference nothing but params, so the speculative hoist can never
    /// observe a different state than the recursive evaluation would.
    fn compile_tail_rec_to_loop_slots(
        &mut self,
        name: &str,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        info: &crate::pass::Analyzer::TailRecInfo,
    ) -> NodeId {
        // 1. Entry args: the function's own param bindings (eligibility proved
        //    them plain — no entry Cells, no captures).
        let entry_args: Vec<NodeId> = params
            .iter()
            .filter_map(|p| self.lookup_var(p.name))
            .collect();

        // 2. Placeholder while_sg; param_count = P (first-P-node convention:
        //    the entry call and the phi carries inject into these slots).
        let node_start = self.graph.nodes.len() as u32;
        let while_sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: while_sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: params.len() as u8,
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

        // 3. Param Const nodes — the FIRST nodes inside while_sg.
        let param_nodes: Vec<NodeId> = params
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

        // 4. Rebind the names to the slots for cond / args / body / exit
        //    compilation (scope-local; the function-level binding keeps
        //    pointing at the entry-arg nodes).
        self.enter_scope();
        for (param, &pn) in params.iter().zip(param_nodes.iter()) {
            self.bind_var(param.name, pn);
        }

        // 5. Condition = NOT(base_cond) — reads the param slots directly
        //    (no loads; the params are the loop's own injected slots).
        let cond_node = self.build_tail_rec_cond(&info.base_cases, &info.rec_branches);

        // 6. Hoist the rec args into the while frame. EVERY arg is chained
        //    into the condition root (arg_0 -> ... -> arg_{P-1} -> cond, via
        //    CF_SEQ passthroughs) — only tree members are reset and re-fired
        //    per iteration, so a non-chained arg would compute once and its
        //    stale value would be carried forever. The chain also orders the
        //    gate: it cannot relaunch the body before fresh values exist.
        //    Each chain link's value is its arg's value (SEQ returns the last
        //    input), so the carries point straight at the chain links.
        let arg_exprs = info.rec_branches[0].1.clone();
        let mut arg_nodes: Vec<NodeId> = Vec::with_capacity(arg_exprs.len());
        let mut chain_prev: Option<NodeId> = None;
        for &a in &arg_exprs {
            let raw = self.compile_subexpr(a);
            let rep = match chain_prev {
                Some(prev) => self.chain_effects(Some(prev), raw),
                None => raw,
            };
            arg_nodes.push(rep);
            chain_prev = Some(rep);
        }
        let cond_root = match chain_prev {
            Some(last) => self.chain_effects(Some(last), cond_node),
            None => cond_node,
        };

        // 7. Slot-mode ctx: compile_call lowers the single self-call to a
        //    bare void node (args already hoisted; no stores, no barrier).
        self.tail_rec_ctx = Some(TailRecCtx {
            self_name: name.to_string(),
            param_cells: Vec::new(),
            slot_params: param_nodes.clone(),
        });

        // 8. body_sg — generic body compile (leading statements, the dead
        //    base arm, nested structures all keep their exact recursive
        //    semantics). The converter stamp is lifted so the body's plain
        //    if-arms qualify for E7 same-frame execution (the rec arm is a
        //    single void Const once the call is intercepted).
        let prev_effect = self.current_effect;
        let prev_tail = self.in_tail_position;
        let prev_conv = self.graph.converter_scope;
        self.graph.converter_scope = false;
        self.current_effect = None;
        self.in_tail_position = true;
        let body_sg = self.compile_loop_body_subgraph(body_expr, while_sg_id, true);
        self.in_tail_position = prev_tail;
        self.current_effect = prev_effect;
        self.graph.converter_scope = prev_conv;

        // 9. exit_sg = the base-case return value.
        let (exit_sg, exit_inputs) = self.compile_slot_exit_sg(info.base_cases[0].1);

        self.exit_scope();

        // 10. Gate(cond_root): true -> body_sg, false -> exit_sg.
        let gate_off = self.graph.inputs_pool.push(&[cond_root]);
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
                condition_input: cond_root,
                branches: vec![
                    (true, body_sg, Vec::new()),
                    (false, exit_sg, exit_inputs),
                ],
            },
        );

        // 11. Metadata: loop_kind = While (the body's None-completion
        //     continues via reset; the TailRec kind's None-means-base-case
        //     exit does not apply — the base case never runs in the body).
        //     The reset plan's condition tree roots at cond_root (the plan
        //     precompute flattens it, hoisted args included) and the carries
        //     are pure value copies arg_i -> param slot i.
        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[while_sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = gate_node;
        sg.loop_kind = LoopKind::While;
        sg.cond_node = Some(cond_root);
        sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![],
            reset_to_one: vec![],
            reset_condition_tree: vec![cond_root],
            fused_carries: Vec::new(),
            condition_tree_plan: Vec::new(),
            carries_value: param_nodes
                .iter()
                .zip(arg_nodes.iter())
                .map(|(&p, &a)| (p, a))
                .collect(),
            carries_cell: vec![],
        });

        // 12. Entry call: args FIRST (compute_call_launch takes the first
        //     param_count inputs as the argument vector), trailing effect
        //     dep for ordering — mirrors compile_recursive_call's contract.
        let mut call_inputs = entry_args;
        if let Some(eff) = self.current_effect {
            call_inputs.push(eff);
        }
        let call_off = self.graph.inputs_pool.push(&call_inputs);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: call_inputs.len() as u8,
            inputs_offset: call_off,
            compute_fn: CF_CALL_LAUNCH,
        });
        self.graph.set_call_target(call_node, while_sg_id);
        call_node
    }

    /// exit_sg for the slot transport: the base-case return value. A bare
    /// param read lowers to the outer slot node itself, which would leave
    /// exit_sg empty with an out-of-range return_node — wrap it in an in-sg
    /// identity node. Richer exprs produce in-sg nodes naturally.
    fn compile_slot_exit_sg(
        &mut self,
        exit_expr: crate::ast::Ast::ExprId,
    ) -> (SubGraphId, Vec<NodeId>) {
        use crate::ast::Ast;
        if let Ast::Expr::Ident(n) = &self.current_module().arena.expr(exit_expr).node {
            if let Some(outer) = self.lookup_var(n) {
                let node_start = self.graph.nodes.len() as u32;
                let off = self.graph.inputs_pool.push(&[outer]);
                let wrap = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset: off,
                    compute_fn: CF_SEQ,
                });
                let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
                self.graph.add_subgraph(SubGraph {
                    converter_generated: false,
                    id: sg_id,
                    node_range: (NodeId(node_start), NodeId(node_start + 1)),
                    param_count: 0,
                    entry_node: wrap,
                    return_node: wrap,
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
                return (sg_id, Vec::new());
            }
        }
        self.compile_branch_subgraph(exit_expr)
    }

    /// Build the loop condition for tail-recursion-to-iteration.
    /// Rules:
    /// - Has a base_case with `Some(cond)`: `cond = AND(NOT(base_cond_i))`.
    /// - No base_case with `Some(cond)`: `cond = OR(rec_cond_i)` (implemented via De Morgan).
    /// - Neither: `cond = Const(true)` (should not happen).
    pub(super) fn build_tail_rec_cond(
        &mut self,
        base_cases: &[(Option<crate::ast::Ast::ExprId>, crate::ast::Ast::ExprId)],
        rec_branches: &[(Option<crate::ast::Ast::ExprId>, Vec<crate::ast::Ast::ExprId>)],
    ) -> NodeId {
        let base_conds: Vec<crate::ast::Ast::ExprId> = base_cases
            .iter()
            .filter_map(|(c, _)| *c)
            .collect();

        if !base_conds.is_empty() {
            // cond = AND(NOT(base_cond_i)).
            let mut negated_nodes: Vec<NodeId> = Vec::new();
            for c in &base_conds {
                let cond_node = self.compile_subexpr(*c);
                let not_off = self.graph.inputs_pool.push(&[cond_node]);
                negated_nodes.push(self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset: not_off,
                    compute_fn: CF_NOT_BOOL,
                }));
            }
            let mut result = negated_nodes[0];
            for n in &negated_nodes[1..] {
                let and_off = self.graph.inputs_pool.push(&[result, *n]);
                result = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: and_off,
                    compute_fn: CF_AND_BOOL,
                });
            }
            result
        } else {
            // No base_case with `Some(cond)`: `cond = OR(rec_cond_i) = NOT(AND(NOT(rec_cond_i)))`.
            // If a rec_branch has `cond=None` (a fallback branch, e.g. the `else` of an if-else),
            // then some rec path always executes, so `cond` should be `Const(true)`, and the
            // rec/base split is dispatched by body_sg's internal Gate + Continue signal.
            let has_none_rec = rec_branches.iter().any(|(c, _)| c.is_none());
            let rec_conds: Vec<crate::ast::Ast::ExprId> = rec_branches
                .iter()
                .filter_map(|(c, _)| *c)
                .collect();
            if rec_conds.is_empty() || has_none_rec {
                // No synthesizable ExprId condition, or a fallback rec branch exists
                // (match/if-else tail recursion).
                // `cond = Const(true)`; body_sg always executes, and the rec/base split is
                // distinguished by the Continue signal:
                //   A rec arm's WriteBack sets Continue -> the loop continues;
                //   a base arm has no WriteBack -> None -> the loop exits (returns body_sg's
                //   return value).
                let off = self.graph.inputs_pool.push(&[]);
                let true_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset: off,
                    compute_fn: CF_NOOP,
                });
                self.graph.const_values[true_node.0 as usize] = Some(ConstValue::Bool(true));
                true_node
            } else {
                let mut negated: Vec<NodeId> = Vec::new();
                for c in &rec_conds {
                    let cn = self.compile_subexpr(*c);
                    let not_off = self.graph.inputs_pool.push(&[cn]);
                    negated.push(self.graph.add_node(Node {
                        kind: NodeKind::UnOp,
                        input_count: 1,
                        inputs_offset: not_off,
                        compute_fn: CF_NOT_BOOL,
                    }));
                }
                let mut and_result = negated[0];
                for n in &negated[1..] {
                    let and_off = self.graph.inputs_pool.push(&[and_result, *n]);
                    and_result = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: and_off,
                        compute_fn: CF_AND_BOOL,
                    });
                }
                let not_off = self.graph.inputs_pool.push(&[and_result]);
                self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset: not_off,
                    compute_fn: CF_NOT_BOOL,
                })
            }
        }
    }

    // ---- Non-tail-recursion-to-iteration helpers ----

    /// Create an i32 constant node.
    pub(super) fn make_i32_const(&mut self, val: i32) -> NodeId {
        let n = self.compile_const();
        self.graph.const_values[n.0 as usize] = Some(ConstValue::I32(val));
        n
    }

    /// Create a binary-operation node.
    pub(super) fn make_binop(&mut self, lhs: NodeId, rhs: NodeId, cf: ComputeFnId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[lhs, rhs]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: cf,
        })
    }

    /// Create an array-store node `arr[idx] = val`.
    pub(super) fn make_array_store(&mut self, arr: NodeId, idx: NodeId, val: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[arr, idx, val]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 3,
            inputs_offset: off,
            compute_fn: CF_ARRAY_STORE,
        })
    }

    /// Create an array-index node `arr[idx]`.
    pub(super) fn make_array_index(&mut self, arr: NodeId, idx: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[arr, idx]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: CF_ARRAY_INDEX,
        })
    }

    /// Create a Continue-signal barrier node (depends on `dep`; triggers the Continue signal).
    pub(super) fn make_continue_barrier(&mut self, dep: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[dep]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 1,
            inputs_offset: off,
            compute_fn: CF_CONTINUE,
        });
        n
    }

    /// Non-tail-recursion to iteration: consumes `NonTailRecInfo` to construct the
    /// "work-stack + while-loop + state-machine" IR.
    ///
    /// Called by `compile_function` when it detects `MemoStrategy::NonTailRecToLoop`.
    /// The parameter nodes have already been created and `bind_var`'d by `compile_function`;
    /// this method constructs the loop structure.
    ///
    /// Structure:
    /// - Function subgraph: param nodes + local variables (stack, sp, result) + initial frame
    ///   push + `Call(while_sg)`.
    /// - `while_sg`: `cond = sp > 0`, `Gate(cond)` -> `body_sg` / `result_sg`.
    /// - `body_sg` (LoopBody): pop a frame -> read `param_cur`/state/saved -> Gate chain
    ///   dispatches by state.
    /// - `state_N_sg`: compile the function body (`non_tail_rec_ctx` intercepts self-calls as
    ///   `push + barrier(Continue)`).
    /// - `result_sg`: returns `result_node`.
    ///
    /// Frame layout (stride = param_count + 1 + max_saved):
    /// `[param_0, ..., param_{P-1}, state, saved_0, ..., saved_{max_saved-1}]`
    pub(super) fn compile_non_tail_rec_to_loop(
        &mut self,
        name: &str,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        info: &crate::pass::Analyzer::NonTailRecInfo,
    ) -> NodeId {
        let param_count = info.param_count;
        let call_sites: Vec<crate::ast::Ast::ExprId> = info.call_sites.clone();
        let num_call_sites = call_sites.len();
        let max_saved = num_call_sites.saturating_sub(1);
        let stride = (param_count + 1 + max_saved) as u32;

        // 1. Collect parameter nodes (compile_function has already bind_var'd them)
        let param_nodes: Vec<NodeId> = params
            .iter()
            .filter_map(|p| self.lookup_var(p.name))
            .collect();

        // 2. Create local variables: stack_node (empty array), sp_cell (Cell=i1),
        //    result_cell (Cell=void). B3: the converter's registers are Cells —
        //    sp/result live across iterations as engine state (the old
        //    WriteBack home-slot registers are gone). The allocs chain into the
        //    effect stream ahead of the while Call (same ordering contract as
        //    the ③ assigned-param cells).
        let stack_off = self.graph.inputs_pool.push(&[]);
        let stack_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 0,
            inputs_offset: stack_off,
            compute_fn: CF_ARRAY_CONSTRUCT,
        });
        let one_init = self.make_i32_const(1);
        let sp_cell_off = self.graph.inputs_pool.push(&[one_init]);
        let sp_cell = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset: sp_cell_off,
            compute_fn: CF_CELL_ALLOC,
        });
        let void_init = self.compile_void_const();
        let result_cell_off = self.graph.inputs_pool.push(&[void_init]);
        let result_cell = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset: result_cell_off,
            compute_fn: CF_CELL_ALLOC,
        });

        // 3. Push the initial frame: stack[0..P] = params, stack[P] = 0 (INIT), stack[P+1..] = 0
        // All array_stores must be chained into the effect chain to ensure Call(while_sg) executes after the stack is filled.
        let zero_init = self.make_i32_const(0);
        let mut init_effect: Option<NodeId> = None;
        init_effect = Some(self.chain_effects(init_effect, sp_cell));
        init_effect = Some(self.chain_effects(init_effect, result_cell));
        for i in 0..param_count {
            let idx = self.make_i32_const(i as i32);
            let store = self.make_array_store(stack_node, idx, param_nodes[i]);
            init_effect = Some(self.chain_effects(init_effect, store));
        }
        let state_zero_idx = self.make_i32_const(param_count as i32);
        let state_zero_store = self.make_array_store(stack_node, state_zero_idx, zero_init);
        init_effect = Some(self.chain_effects(init_effect, state_zero_store));
        for i in 0..max_saved {
            let idx = self.make_i32_const((param_count + 1 + i) as i32);
            let store = self.make_array_store(stack_node, idx, zero_init);
            init_effect = Some(self.chain_effects(init_effect, store));
        }
        self.current_effect = init_effect;

        // 4. Placeholder-register while_sg
        let while_node_start = self.graph.nodes.len() as u32;
        let while_sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: while_sg_id,
            node_range: (NodeId(while_node_start), NodeId(while_node_start)),
            param_count: 0,
            entry_node: NodeId(while_node_start),
            return_node: NodeId(while_node_start),
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

        // 5. cond_node: sp > 0 (within while_sg node_range). sp is read through
        // its Cell: the load is an input of the comparison, so it sits in the
        // condition tree and re-fires every iteration with the current value.
        let zero_cond = self.make_i32_const(0);
        let sp_cond_load_off = self.graph.inputs_pool.push(&[sp_cell]);
        let sp_cond_load = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset: sp_cond_load_off,
            compute_fn: CF_DEREF_READ,
        });
        let cond_node = self.make_binop(sp_cond_load, zero_cond, CF_GT_I32);

        // Save the init effect chain (cell allocs + stack init); body_sg compilation will reset current_effect
        let init_effect_chain = self.current_effect;

        // 6. Compile body_sg (LoopBody: pop + read frame + state dispatch)
        let body_sg = self.compile_non_tail_rec_body_sg(
            body_expr,
            params,
            name,
            &call_sites,
            while_sg_id,
            stack_node,
            sp_cell,
            result_cell,
            param_count,
            max_saved,
            stride,
        );

        // Restore the init effect chain so Call(while_sg) depends on the init code (including sp=1 WriteBack)
        self.current_effect = init_effect_chain;

        // 7. Compile result_sg (false branch): load the final result through the
        // result Cell (the load node lives in result_sg; the cell was last
        // written by the deepest completed state).
        let result_sg = {
            let rs_start = self.graph.nodes.len() as u32;
            let off = self.graph.inputs_pool.push(&[result_cell]);
            let passthrough = self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset: off,
                compute_fn: CF_DEREF_READ,
            });
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(SubGraph {
                converter_generated: false,
                id: sg_id,
                node_range: (NodeId(rs_start), NodeId(rs_start + 1)),
                param_count: 0,
                entry_node: NodeId(rs_start),
                return_node: passthrough,
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
            sg_id
        };

        // 8. Gate(cond): true → body_sg, false → result_sg
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
                    (true, body_sg, Vec::new()),
                    (false, result_sg, Vec::new()),
                ],
            },
        );

        // 9. Populate while_sg metadata
        let while_node_end = self.graph.nodes.len() as u32;
        let while_sg = &mut self.graph.subgraphs[while_sg_id.0 as usize];
        while_sg.node_range = (NodeId(while_node_start), NodeId(while_node_end));
        while_sg.entry_node = NodeId(while_node_start);
        while_sg.return_node = gate_node;
        while_sg.loop_kind = LoopKind::While;
        while_sg.cond_node = Some(cond_node);
        while_sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![],
            reset_to_one: vec![],
            reset_condition_tree: vec![cond_node],
            fused_carries: Vec::new(),
            condition_tree_plan: Vec::new(),
            carries_value: Vec::new(),
            carries_cell: Vec::new(),
        });

        // 10. Create the Call node to launch while_sg
        let call_node = self.compile_recursive_call(while_sg_id);
        call_node
    }

    /// Compile the body_sg for non-tail-recursion-to-iteration (LoopBody subgraph).
    ///
    /// Structure:
    /// 1. Pop: sp = sp - 1 (WriteBack), read the stack frame
    /// 2. Read param_cur[i] = stack[sp * stride + i]
    /// 3. Read state = stack[sp * stride + P]
    /// 4. Read saved[i] = stack[sp * stride + P + 1 + i]
    /// 5. Gate chain dispatches by state to each state_N_sg
    pub(super) fn compile_non_tail_rec_body_sg(
        &mut self,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        self_name: &str,
        call_sites: &[crate::ast::Ast::ExprId],
        while_sg_id: SubGraphId,
        stack_node: NodeId,
        sp_cell: NodeId,
        result_cell: NodeId,
        param_count: usize,
        max_saved: usize,
        stride: u32,
    ) -> SubGraphId {
        let body_node_start = self.graph.nodes.len() as u32;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = body_node_start;

        // Push the loop context (Continue signal jump-back target = while_sg)
        self.loop_stack.push(LoopContext {
            sg: while_sg_id,
            iter_node: None,
            body_node_start,
            loop_var_node: None,
        });

        // W3C: pre-register the outer body subgraph (registered at the end of
        // this function); awaits compiled in the pop/frame-read section below
        // register their EventSourceDecls directly into it.
        let body_sg = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: body_sg,
            node_range: (NodeId(body_node_start), NodeId(body_node_start)),
            param_count: 0,
            entry_node: NodeId(body_node_start),
            return_node: NodeId(body_node_start),
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
        let prev_branch_sg_outer = self.current_branch_sg;
        self.current_branch_sg = Some(body_sg);

        // 1. Pop: sp = sp - 1 through the Cell. The load is ordered after the
        // body's entry effect (trailing dep — compute reads inputs[0]); the
        // store re-binds the Cell so the condition's next re-evaluation and
        // the next iteration's pop read the decremented value.
        let sp_pop_load_off = self.graph.inputs_pool.push(&[sp_cell]);
        let sp_pop_load = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset: sp_pop_load_off,
            compute_fn: CF_DEREF_READ,
        });
        let one_pop = self.make_i32_const(1);
        let sp_minus_1 = self.make_binop(sp_pop_load, one_pop, CF_SUB_I32);
        let pop_store_off = self.graph.inputs_pool.push(&[sp_cell, sp_minus_1]);
        let pop_store = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: pop_store_off,
            compute_fn: CF_DEREF_WRITE,
        });
        self.current_effect = Some(pop_store);

        // 2. Read the stack frame: frame_base = sp_minus_1 * stride
        let stride_node = self.make_i32_const(stride as i32);
        let frame_base = self.make_binop(sp_minus_1, stride_node, CF_MUL_I32);

        // param_cur[i] = stack[frame_base + i]
        let mut param_cur: Vec<NodeId> = Vec::with_capacity(param_count);
        for i in 0..param_count {
            let offset = self.make_i32_const(i as i32);
            let idx = self.make_binop(frame_base, offset, CF_ADD_I32);
            param_cur.push(self.make_array_index(stack_node, idx));
        }

        // state = stack[frame_base + param_count]
        let state_offset = self.make_i32_const(param_count as i32);
        let state_idx = self.make_binop(frame_base, state_offset, CF_ADD_I32);
        let state_node = self.make_array_index(stack_node, state_idx);

        // saved[i] = stack[frame_base + param_count + 1 + i]
        let mut saved_nodes: Vec<NodeId> = Vec::with_capacity(max_saved);
        for i in 0..max_saved {
            let offset = self.make_i32_const((param_count + 1 + i) as i32);
            let idx = self.make_binop(frame_base, offset, CF_ADD_I32);
            saved_nodes.push(self.make_array_index(stack_node, idx));
        }

        // Chain param_cur / state_node / saved_nodes into the effect chain,
        // ensuring they are ready before the dispatch Gate launches state_N_sg.
        // Otherwise the Gate only depends on cmp(state_node==i), which could launch
        // state_N_sg before param_cur is computed, causing the frame copy to receive void parameter values.
        for &pc in &param_cur {
            self.current_effect = Some(self.chain_effects(self.current_effect, pc));
        }
        self.current_effect = Some(self.chain_effects(self.current_effect, state_node));
        for &sn in &saved_nodes {
            self.current_effect = Some(self.chain_effects(self.current_effect, sn));
        }
        let frame_read_effect = self.current_effect;

        // 3. Compile each state_N_sg (each state compiles the function body; non_tail_rec_ctx intercepts self-calls)
        let num_states = call_sites.len() + 1;
        let mut state_sgs: Vec<SubGraphId> = Vec::with_capacity(num_states);

        for state_idx in 0..num_states {
            // Build call_result_map:
            // state 0: empty (all calls are fresh)
            // state N: call_sites[0..N-2] -> saved[0..N-2],
            //          call_sites[N-1] -> RESULT_CELL_MARKER (the most recent
            //          call's result lives in the result Cell; the consumer
            //          synthesizes a CF_DEREF_READ in its own state sg)
            let mut call_result_map: rustc_hash::FxHashMap<crate::ast::Ast::ExprId, NodeId> =
                rustc_hash::FxHashMap::default();
            for i in 0..state_idx {
                if i + 1 < state_idx {
                    call_result_map.insert(call_sites[i], saved_nodes[i]);
                } else {
                    // i == state_idx - 1: the most recently completed call result is in the result Cell
                    call_result_map.insert(call_sites[i], RESULT_CELL_MARKER);
                }
            }

            // Set up non_tail_rec_ctx
            self.non_tail_rec_ctx = Some(NonTailRecCtx {
                self_name: self_name.to_string(),
                param_nodes: param_cur.clone(),
                stack_node,
                sp_cell,
                result_cell,
                call_result_map,
                truncated: false,
                stride,
                param_count,
                max_saved,
                current_state: state_idx as u32,
                saved_nodes: saved_nodes.clone(),
            });

            // Compile state_N_sg
            let sg_node_start = self.graph.nodes.len() as u32;
            let prev_sg_start_inner = self.current_sg_start;
            self.current_sg_start = sg_node_start;

            // W3C: pre-register THIS state's subgraph so awaits in the body
            // register their EventSourceDecls directly into it.
            let state_sg = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(SubGraph {
                converter_generated: false,
                id: state_sg,
                node_range: (NodeId(sg_node_start), NodeId(sg_node_start)),
                param_count: 0,
                entry_node: NodeId(sg_node_start),
                return_node: NodeId(sg_node_start),
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
            let prev_branch_sg = self.current_branch_sg;
            self.current_branch_sg = Some(state_sg);

            self.enter_scope();
            // Bind parameter names to param_cur nodes (instead of the function's param nodes)
            for (i, param) in params.iter().enumerate() {
                if i < param_cur.len() {
                    self.bind_var(param.name, param_cur[i]);
                }
            }

            let prev_effect_inner = self.current_effect;
            let prev_tail = self.in_tail_position;
            self.current_effect = None;
            self.in_tail_position = false;
            let body_node = self.compile_expr(body_expr);
            self.in_tail_position = prev_tail;
            self.current_effect = prev_effect_inner;

            // Clear non_tail_rec_ctx
            self.non_tail_rec_ctx = None;
            self.exit_scope();
            self.current_sg_start = prev_sg_start_inner;

            // Always store the body result through the result Cell.
            // Recursion path: the barrier's Continue signal terminates state_sg before the store executes,
            //   so the store does not run.
            // Base case path: the body completes normally, and the store writes the result to the Cell.
            let return_store_off = self.graph.inputs_pool.push(&[result_cell, body_node]);
            let return_node = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 2,
                inputs_offset: return_store_off,
                compute_fn: CF_DEREF_WRITE,
            });

            let sg_node_end = self.graph.nodes.len() as u32;
            self.current_branch_sg = prev_branch_sg;
            {
                let sgm = &mut self.graph.subgraphs[state_sg.0 as usize];
                sgm.node_range = (NodeId(sg_node_start), NodeId(sg_node_end));
                sgm.return_node = return_node;
            }
            state_sgs.push(state_sg);
        }

        // 4. Build the Gate chain dispatching by state (build the else chain from back to front)
        let void_sg = self.compile_void_subgraph();
        let mut false_sg = void_sg;
        let mut dispatch_gate: NodeId = NodeId(u32::MAX); // sentinel value, always overwritten in the loop

        // Reset current_effect to prevent the Gate chain from depending on residual effects from state compilation
        self.current_effect = None;

        for i in (0..num_states).rev() {
            let wrap_start = self.graph.nodes.len() as u32;

            // cmp = state_node == i
            let state_const = self.make_i32_const(i as i32);
            let cmp = self.make_binop(state_node, state_const, CF_EQ_I32);
            // Make cmp depend on frame_read_effect to ensure param_cur / saved_nodes are ready
            // before launching state_N_sg (otherwise the frame copy receives void parameter values)
            let cmp_eff = self.chain_effects(frame_read_effect, cmp);

            // Gate(cmp_eff): true → state_sgs[i], false → false_sg
            let gate_inputs: Vec<NodeId> = vec![cmp_eff];
            let gate_off = self.graph.inputs_pool.push(&gate_inputs);
            let gate_node = self.graph.add_node(Node {
                kind: NodeKind::Gate,
                input_count: gate_inputs.len() as u8,
                inputs_offset: gate_off,
                compute_fn: CF_GATE_LAUNCH,
            });
            self.graph.set_gate_branches(
                gate_node,
                GateBranches {
                capture: false,
                    condition_input: cmp_eff,
                    branches: vec![
                        (true, state_sgs[i], Vec::new()),
                        (false, false_sg, Vec::new()),
                    ],
                },
            );

            if i == 0 {
                // The first Gate stays in body_sg
                dispatch_gate = gate_node;
            } else {
                // Wrap as a subgraph, serving as the previous Gate's false branch
                let wrap_end = self.graph.nodes.len() as u32;
                let wrap_sg = SubGraphId(self.graph.subgraphs.len() as u32);
                self.graph.add_subgraph(SubGraph {
                    converter_generated: false,
                    id: wrap_sg,
                    node_range: (NodeId(wrap_start), NodeId(wrap_end)),
                    param_count: 0,
                    entry_node: NodeId(wrap_start),
                    return_node: gate_node,
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
                false_sg = wrap_sg;
            }
        }

        // 5. Pop the loop context and register body_sg
        self.loop_stack.pop();
        self.current_sg_start = prev_sg_start;
        self.current_branch_sg = prev_branch_sg_outer;

        let body_node_end = self.graph.nodes.len() as u32;
        {
            let sgm = &mut self.graph.subgraphs[body_sg.0 as usize];
            sgm.node_range = (NodeId(body_node_start), NodeId(body_node_end));
            sgm.return_node = dispatch_gate;
            sgm.loop_kind = LoopKind::LoopBody;
            sgm.loop_parent_sg = Some(while_sg_id);
        }
        body_sg
    }

    /// Register the Loop subgraph (no condition; terminates via Break).
    ///
    /// Structure (unified with While, cond is always true):
    /// - cond_node = Const(true)
    /// - gate_node = Gate(cond): true -> body_sg, false -> void_sg (unreachable)
    /// - body_sg: compiles the body, not tail-recursive (frame reuse on the Engine side)
    ///
    /// Execution: after the body runs, the Engine's reset_loop_iteration resets the Gate to re-execute;
    /// the Break signal terminates body_sg -> Gate completes -> the loop ends.
    pub(super) fn register_loop_subgraph(&mut self, body: crate::ast::Ast::ExprId) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 0,
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

        // cond_node = Const(true) (loop has no condition; always true)
        let cond_node = self.compile_bool_const(true);
        // Reset current_effect = None (same as register_while_subgraph, to avoid deadlock from
        // external effect dependencies after the loop body's reset_loop_iteration)
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // body subgraph (not tail-recursive)
        let body_sg = self.compile_loop_body_subgraph(body, sg_id, true);
        // void subgraph (unreachable branch; used on break exit): includes CF_DEFER_RUN for defer-in-loop.
        let void_sg = self.compile_defer_run_subgraph();
        self.current_effect = prev_effect;

        // Gate node: cond(true) -> body_sg, false -> void_sg (unreachable)
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
        sg.loop_kind = LoopKind::Loop;
        sg.cond_node = Some(cond_node);
        sg.reset_plan = Some(ResetPlan {
            reset_to_zero: vec![],
            reset_to_one: vec![],
            reset_condition_tree: vec![cond_node],
            fused_carries: Vec::new(),
            condition_tree_plan: Vec::new(),
            carries_value: Vec::new(),
            carries_cell: Vec::new(),
        });
        sg_id
    }

    /// Compile the loop body subgraph: compiles the body, not tail-recursive (frame reuse handled by Engine-side reset_loop_iteration).
    ///
    /// `loop_sg` is the while_sg of a While or the loop_sg of a Loop.
    /// return_node = body_last (the body's last node); the Engine detects LoopBody completion and resets the loop.
    pub(super) fn compile_loop_body_subgraph(
        &mut self,
        body: crate::ast::Ast::ExprId,
        loop_sg: SubGraphId,
        clear_cell_values: bool,
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        // Place-model forwarding barrier (loop-like): pre-loop values are
        // stale from iteration 2 on, and post-loop reads must load. `while`
        // bodies pass clear=false — the barrier ran before the CONDITION
        // compile (register_while_subgraph), and the condition's cell LOADS
        // are recorded in the forwarding memory: body reads of the same cells
        // forward to them (same iteration, condition dominates the body) —
        // saving one load per cell per iteration.
        if clear_cell_values {
            self.cell_barrier_enter();
        }
        // Push the loop context (continue jump target; While/Loop have no iterator parameter)
        self.loop_stack.push(LoopContext {
            sg: loop_sg,
            iter_node: None,
            body_node_start: node_start,
            loop_var_node: None,
        });

        // Enter a new scope so that bind_var inside the loop body does not overwrite
        // the function-level binding. This is critical for WriteBack target resolution:
        // when an if/match branch inside the loop body assigns to an outer variable,
        // lookup_root_frame_var must find the function-level declaration (not an
        // intermediate node produced by the loop body's bind_var).
        self.enter_scope();

        // W3C: pre-register the body subgraph (stable id across the body
        // compile) and make it the innermost region so build_await_node
        // registers EventSourceDecls directly here (Bug #24 class, structural
        // fix — replaces the post-hoc drain from the function sg).
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_start)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node: NodeId(node_start),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::LoopBody,
            loop_parent_sg: Some(loop_sg),
            cond_node: None,
            function_id: self.current_function_id,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        let prev_branch_sg = self.current_branch_sg;
        self.current_branch_sg = Some(sg_id);

        let prev_in_loop = self.in_loop_body;
        self.in_loop_body = true;
        let body_last = self.compile_expr(body);
        self.in_loop_body = prev_in_loop;
        self.loop_stack.pop();
        self.exit_scope();
        self.current_sg_start = prev_sg_start;
        self.current_branch_sg = prev_branch_sg;
        // While bodies (clear=false) did not enter a barrier here — the
        // body's final cell_values (the phi carry sources) must SURVIVE for
        // register_while_subgraph to read; its own barrier_exit after the
        // void_sg compile performs the final clear.
        if clear_cell_values {
            self.cell_barrier_exit();
        }
        let node_end = self.graph.nodes.len() as u32;
        {
            let sgm = &mut self.graph.subgraphs[sg_id.0 as usize];
            sgm.node_range = (NodeId(node_start), NodeId(node_end));
            sgm.return_node = body_last;
        }
        sg_id
    }

}

/// L3' eligibility walker: the expr is pure and its free identifiers are all
/// params (literals and param-derived arithmetic/casts only). Calls, records,
/// arrays, field/index reads, references and string interpolations are all
/// rejected — they either have effects/allocations or may reference bindings
/// that do not exist in the while frame.
fn expr_params_only_pure(
    arena: &crate::ast::Ast::AstArena<'_>,
    expr: crate::ast::Ast::ExprId,
    param_names: &rustc_hash::FxHashSet<&str>,
) -> bool {
    use crate::ast::Ast;
    match &arena.expr(expr).node {
        Ast::Expr::IntLit { .. }
        | Ast::Expr::FloatLit { .. }
        | Ast::Expr::BoolLit(_)
        | Ast::Expr::CharLit(_)
        | Ast::Expr::NullLit
        | Ast::Expr::VoidLit => true,
        Ast::Expr::Ident(n) => param_names.contains(n),
        Ast::Expr::Binary { lhs, rhs, .. } => {
            expr_params_only_pure(arena, *lhs, param_names)
                && expr_params_only_pure(arena, *rhs, param_names)
        }
        Ast::Expr::Unary { operand, .. } => expr_params_only_pure(arena, *operand, param_names),
        Ast::Expr::As { expr, .. } => expr_params_only_pure(arena, *expr, param_names),
        _ => false,
    }
}
