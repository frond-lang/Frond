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
        // 1. Collect parameter nodes (already `bind_var`'d by `compile_function`).
        let param_nodes: Vec<NodeId> = params
            .iter()
            .filter_map(|p| self.lookup_var(p.name))
            .collect();

        // 2. Placeholder-register while_sg.
        let node_start = self.graph.nodes.len() as u32;
        let while_sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
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
        let cond_node = self.build_tail_rec_cond(&info.base_cases, &info.rec_branches);

        // 4. Set `tail_rec_ctx` (`compile_call` intercepts self-calls as `WriteBack +
        //    Call(while_sg)`).
        self.tail_rec_ctx = Some(TailRecCtx {
            self_name: name.to_string(),
            param_nodes,
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
        let body_sg = self.compile_loop_body_subgraph(body_expr, while_sg_id);
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

        // 2. Create local variables: stack_node (empty array), sp_node (0), result_node (void)
        let stack_off = self.graph.inputs_pool.push(&[]);
        let stack_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 0,
            inputs_offset: stack_off,
            compute_fn: CF_ARRAY_CONSTRUCT,
        });
        let sp_node = self.make_i32_const(0);
        let result_node = self.compile_void_const();

        // 3. Push the initial frame: stack[0..P] = params, stack[P] = 0 (INIT), stack[P+1..] = 0; sp = 1
        // All array_stores must be chained into the effect chain to ensure Call(while_sg) executes after the stack is filled.
        let zero_init = self.make_i32_const(0);
        let mut init_effect: Option<NodeId> = None;
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
        let one_init = self.make_i32_const(1);
        let sp_init_wb = self.compile_writeback_node(one_init, sp_node);
        self.current_effect = Some(self.chain_effects(init_effect, sp_init_wb));

        // 4. Placeholder-register while_sg
        let while_node_start = self.graph.nodes.len() as u32;
        let while_sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
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

        // 5. cond_node: sp > 0 (within while_sg node_range)
        let zero_cond = self.make_i32_const(0);
        let cond_node = self.make_binop(sp_node, zero_cond, CF_GT_I32);

        // Save the init effect chain (including sp=1 WriteBack); body_sg compilation will reset current_effect
        let init_effect_chain = self.current_effect;

        // 6. Compile body_sg (LoopBody: pop + read frame + state dispatch)
        let body_sg = self.compile_non_tail_rec_body_sg(
            body_expr,
            params,
            name,
            &call_sites,
            while_sg_id,
            stack_node,
            sp_node,
            result_node,
            param_count,
            max_saved,
            stride,
        );

        // Restore the init effect chain so Call(while_sg) depends on the init code (including sp=1 WriteBack)
        self.current_effect = init_effect_chain;

        // 7. Compile result_sg (false branch, returns result_node)
        let result_sg = {
            let rs_start = self.graph.nodes.len() as u32;
            let off = self.graph.inputs_pool.push(&[result_node]);
            let passthrough = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 1,
                inputs_offset: off,
                compute_fn: CF_SEQ,
            });
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(SubGraph {
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
            condition_tree_plan: Vec::new(),
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
        sp_node: NodeId,
        result_node: NodeId,
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

        // 1. Pop: sp = sp - 1 (WriteBack to sp_node)
        let one_pop = self.make_i32_const(1);
        let sp_minus_1 = self.make_binop(sp_node, one_pop, CF_SUB_I32);
        let pop_wb = self.compile_writeback_node(sp_minus_1, sp_node);
        self.current_effect = Some(pop_wb);

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
            // state N: call_sites[0..N-2] -> saved[0..N-2], call_sites[N-1] -> result_node
            let mut call_result_map: rustc_hash::FxHashMap<crate::ast::Ast::ExprId, NodeId> =
                rustc_hash::FxHashMap::default();
            for i in 0..state_idx {
                if i + 1 < state_idx {
                    call_result_map.insert(call_sites[i], saved_nodes[i]);
                } else {
                    // i == state_idx - 1: the most recently completed call result is in result_node
                    call_result_map.insert(call_sites[i], result_node);
                }
            }

            // Set up non_tail_rec_ctx
            self.non_tail_rec_ctx = Some(NonTailRecCtx {
                self_name: self_name.to_string(),
                param_nodes: param_cur.clone(),
                stack_node,
                sp_node,
                result_node,
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

            // Always WriteBack the body result to result_node.
            // Recursion path: the barrier's Continue signal terminates state_sg before the WriteBack executes,
            //   so the WriteBack does not run.
            // Base case path: the body completes normally, and the WriteBack writes the result to result_node.
            let return_node = self.compile_writeback_node(body_node, result_node);

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
        let body_sg = self.compile_loop_body_subgraph(body, sg_id);
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
            condition_tree_plan: Vec::new(),
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
    ) -> SubGraphId {
        let node_start = self.graph.nodes.len() as u32;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
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
        let node_end = self.graph.nodes.len() as u32;
        {
            let sgm = &mut self.graph.subgraphs[sg_id.0 as usize];
            sgm.node_range = (NodeId(node_start), NodeId(node_end));
            sgm.return_node = body_last;
        }
        sg_id
    }

}
