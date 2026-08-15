//! Call — Call machinery: direct / vtable / method calls, intrinsics, extern-C ABI. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Create a sequence node: after `prev_effect` completes, returns `current_node`'s value.
    ///
    /// Used for statement-order chaining: ensures nodes depending on `current_node` execute only
    /// after `prev_effect` completes.
    /// `compute_seq` (idx 48) takes all inputs and returns the value of the last input.
    pub(super) fn chain_effects(&mut self, prev: Option<NodeId>, current: NodeId) -> NodeId {
        match prev {
            Some(prev_node) => {
                let off = self.graph.inputs_pool.push(&[prev_node, current]);
                self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_SEQ, // compute_seq
                })
            }
            None => current,
        }
    }

    /// Create a Call node targeting `target_sg` (no input dependency; immediately ready).
    ///
    /// Used for the initial call of a loop and for `continue` jumps.
    pub(super) fn compile_recursive_call(&mut self, target_sg: SubGraphId) -> NodeId {
        // Append `current_effect` as an implicit dependency (consistent with `compile_call`),
        // ensuring the while/loop recursive Call executes only after prior statements (e.g. an
        // array literal) complete.
        // Otherwise the Call node has no input dependency and could launch the subgraph frame
        // before the `arr` value is ready, preventing the frame from copying `arr`'s Ref value
        // and causing `arr[0]` inside the loop body to return `<non-scalar>`.
        let mut inputs: Vec<NodeId> = Vec::new();
        if let Some(eff) = self.current_effect {
            inputs.push(eff);
        }
        let input_count = inputs.len() as u8;
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        self.graph.set_call_target(call_node, target_sg);
        call_node
    }

    /// Create a Call node targeting `target_sg`, passing the given argument nodes.
    pub(super) fn make_call(&mut self, target_sg: SubGraphId, arg_nodes: &[NodeId]) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(arg_nodes);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: arg_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        self.graph.set_call_target(call_node, target_sg);
        call_node
    }

    /// Create a Call node targeting a function name (looked up via `func_subgraphs`).
    pub(super) fn make_call_by_name(&mut self, name: &str, arg_nodes: &[NodeId]) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(arg_nodes);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: arg_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        match self.resolve_func("make_call_by_name", name, None, None) {
            Some(Ok(target_sg)) => self.graph.set_call_target(call_node, target_sg),
            Some(Err(diag)) => self.errors.push(diag),
            // A target-less Call node evaluates to void at runtime and silently
            // poisons downstream arithmetic — fail the build instead.
            None => self.errors.push(format!(
                "call to unknown function '{}': no target subgraph was bound",
                name
            )),
        }
        call_node
    }

    /// Create a vtable dynamic-dispatch Call node (method call on a trait value).
    ///
    /// Unlike `make_call_by_name`: the target subgraph id is looked up at runtime from the
    /// TraitVal's vtable rather than bound at compile time. Used when a For-loop iterable is a
    /// trait value (`Iterator<T>`).
    pub(super) fn make_vtable_call(&mut self, recv_node: NodeId, trait_name: &str, method_name: &str) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node]);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        let method_idx = self.sema.get_trait_def(trait_name)
            .and_then(|td| td.methods.iter().position(|m| m.name.as_ref() == method_name))
            .map(|i| i as u16);
        match method_idx {
            Some(idx) => self.graph.set_vtable_call(call_node, idx),
            None => self.errors.push(format!(
                "internal: trait method '{}' not found in trait '{}' for vtable dispatch",
                method_name, trait_name)),
        }
        call_node
    }

    /// Compile a function call.
    ///
    /// If the callee is a known function name -> Call node + set_call_target.
    /// If the callee is a type name (e.g. `Iterator(arr, 0)`) -> compile into a record-construction node.
    pub(super) fn compile_call(
        &mut self,
        call_expr_id: crate::ast::Ast::ExprId,
        callee: crate::ast::Ast::ExprId,
        args: &[crate::ast::Ast::ExprId],
    ) -> NodeId {
        // Generic calls prefer the monomorphization-instance path: inline expansion does not handle
        // type-parameter substitution, so inlining a generic function would leave type parameter T
        // unresolved to a concrete type in the body.
        let call_inst_key = crate::sema::Sema::module_expr_key(
            self.expr_key_module(),
            call_expr_id.0 as u64,
        );
        let is_generic_call = self.sema.call_instantiations.contains_key(&call_inst_key);

        // -- Inline expansion: call sites flagged by the analyzer compile the callee body directly instead of launching a subgraph --
        // pure function + small body + non-recursive -> bind actuals to formals, compile body, avoid call overhead
        // generic calls skip inlining (type parameters need replacement via the monomorph instance subgraph)
        if !is_generic_call {
            if let Some(callee_func) = self.inline_target(call_expr_id) {
                if let crate::ast::Ast::Decl::FunDecl { params, body, .. } =
                    &self.current_module().declarations[callee_func.0 as usize].node
                {
                    return self.compile_inline_expansion(*body, params, args);
                }
            }
        }

        let callee_expr = self.current_module().arena.expr(callee);

        // ── reflect top-level function interception (safety net) ──
        // Historically `format(x)` / `type_name(x)` were top-level wrapper functions;
        // they have been removed in favor of direct method calls (`x.format()`).
        // This interception remains as a safety net: if a future top-level reflect
        // wrapper is reintroduced, it lowers directly to CF_REFLECT_* without going
        // through FFI dispatch. Today it is effectively dead code (no such functions
        // are declared, so Sema rejects bare `format(x)` before reaching here).
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if let Some(cf) = reflect_top_level_cf(name) {
                let mut inputs = Vec::with_capacity(args.len());
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                return self.graph.add_node(Node {
                    kind: NodeKind::Call,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: cf,
                });
            }
        }

        // Built-in constructor detection: Ok(val) / Err(record) / channel(capacity)
        // Lowered via the BUILTIN_CTORS registry lookup; missed error types fall through to the record construction path below
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if !self.func_subgraphs.contains_key(*name) {
                if let Some(lower) = BUILTIN_CTORS.iter().find_map(|(n, l)| (*n == *name).then_some(l)) {
                    return match lower {
                        // Ok(val) -> compute_throw_ok (idx 44), input = val
                        BuiltinCtorLower::Ok => {
                            let mut inputs = Vec::with_capacity(args.len());
                            for &arg in args {
                                inputs.push(self.compile_subexpr(arg));
                            }
                            let inputs_offset = self.graph.inputs_pool.push(&inputs);
                            self.graph.add_node(Node {
                                kind: NodeKind::Call,
                                input_count: inputs.len() as u8,
                                inputs_offset,
                                compute_fn: CF_THROW_OK, // throw_ok
                            })
                        }
                        // Err(...) -> first record_construct, then wrap with throw_err (idx 45)
                        BuiltinCtorLower::Err => {
                            let inner = self.compile_record_like(crate::ir::Compute::CTOR_ERR, args);
                            let inputs_offset = self.graph.inputs_pool.push(&[inner]);
                            self.graph.add_node(Node {
                                kind: NodeKind::Call,
                                input_count: 1,
                                inputs_offset,
                                compute_fn: CF_THROW_ERR, // throw_err
                            })
                        }
                        // channel(capacity) -> compute_channel_create (idx 283), input = args
                        BuiltinCtorLower::Channel => {
                            let mut inputs = Vec::with_capacity(args.len());
                            for &arg in args {
                                inputs.push(self.compile_subexpr(arg));
                            }
                            let inputs_offset = self.graph.inputs_pool.push(&inputs);
                            self.graph.add_node(Node {
                                kind: NodeKind::BinOp,
                                input_count: inputs.len() as u8,
                                inputs_offset,
                                compute_fn: CF_CHANNEL_CREATE, // compute_channel_create
                            })
                        }
                    };
                }
            }
        }

        // Type constructor / ADT / Newtype constructor detection: callee is an Ident and not a known function
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if !self.func_subgraphs.contains_key(*name) {
                // First look up the type name (Record or single-constructor ADT), then the constructor name of a multi-constructor ADT
                let tf_info = self.lookup_type_field_names(name)
                    .or_else(|| self.lookup_constructor_field_names(name));
                if let Some(info) = tf_info {
                    // Compile into a construction node (compute_record_construct = 29, dispatches to HeapObj by kind)
                    let mut inputs = Vec::with_capacity(args.len());
                    for &arg in args {
                        inputs.push(self.compile_subexpr(arg));
                    }
                    let inputs_offset = self.graph.inputs_pool.push(&inputs);
                    let node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: inputs.len() as u8,
                        inputs_offset,
                        compute_fn: CF_RECORD_CONSTRUCT, // record_construct
                    });
                    self.graph.set_record_lit_info(node, RecordLitInfo {
                        type_name: info.type_name.clone(),
                        field_names: info.field_names.into_iter().map(Some).collect(),
                        constructor: name.to_string(),
                        kind: info.kind,
                    });
                    return node;
                }
            }
        }

        // Closure call detection: callee is an Ident, not a known function, but bound in scope (variable holds a Closure/Partial)
        // -> use compute_closure_call (idx 41); inputs[0] = callable value node, inputs[1..1+arg_count] = call arguments
        // current_effect appended at the end as an implicit dependency (ensures Call executes only after prior effects complete)
        // arg_count metadata records the actual argument count (excluding closure value and effect), used for chained partial-application detection
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if !self.func_subgraphs.contains_key(*name) {
                if let Some(closure_node) = self.lookup_var(name) {
                    let mut inputs = Vec::with_capacity(args.len() + 2);
                    inputs.push(closure_node);
                    for &arg in args {
                        inputs.push(self.compile_subexpr(arg));
                    }
                    if let Some(eff) = self.current_effect {
                        inputs.push(eff);
                    }
                    let inputs_offset = self.graph.inputs_pool.push(&inputs);
                    let call_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: inputs.len() as u8,
                        inputs_offset,
                        compute_fn: CF_CLOSURE_CALL, // compute_closure_call
                    });
                    self.graph.set_closure_call_arg_count(call_node, args.len() as u8);
                    return call_node;
                }
            }
        }

        // @extern("C") / @builtin call detection: does not launch a sub-frame; calls Ffi::wrapper
        // (extern C) or the Rust #[no_mangle] fn (builtin, e.g. reflect) directly via FFI dispatch.
        // current_effect appended at the end as an implicit dependency (ensures the Call executes
        // only after prior effects complete).
        //
        // @extern("C") call detection: all stdlib `#{ }#` functions go through
        // CF_DYN_FFI_CALL (dlsym self-lookup + Abi::call_dynamic). There is no longer a
        // CF_FFI_CALL / wrapper table. The C symbol name is uniformly `frond_extern_<name>`
        // (generated by build.rs).
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            // @internal externs are denied BEFORE the FFI dispatch: falling
            // through instead reaches the generic binding path, whose
            // resolve_func guard emits the diagnostic exactly once.
            if !self.internal_access_blocked(name) && self.is_extern_c_func(name) {
                let sig = self.build_abi_sig(name);
                let c_symbol = format!("frond_extern_{name}");
                let mut inputs = Vec::with_capacity(args.len() + 1);
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                if let Some(eff) = self.current_effect {
                    inputs.push(eff);
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                let node = self.graph.add_node(Node {
                    kind: NodeKind::Call,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: CF_DYN_FFI_CALL,
                });
                self.graph.set_dyn_ffi_info(
                    node,
                    crate::ir::Ir::DynFfiInfo {
                        symbol: c_symbol,
                        sig,
                        arg_count: args.len() as u8,
                    },
                );
                return node;
            }
        }

        // Partial application detection: callee is a known function name, but actual count < target function formal count
        // -> generate a partial_construct node (idx 286), producing a HeapObj::Partial
        // bound_args = the already-supplied actuals (binding leading parameters in the original function's parameter order)
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            if let Some(Ok(target_sg)) = self.resolve_func("partial_detect", name, None, None) {
                if let Some(sg) = self.graph.subgraphs.get(target_sg.0 as usize) {
                    let param_count = sg.param_count as usize;
                    if args.len() < param_count {
                        let mut bound_inputs = Vec::with_capacity(args.len());
                        for &arg in args {
                            bound_inputs.push(self.compile_subexpr(arg));
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&bound_inputs);
                        let partial_node = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: bound_inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_PARTIAL_CONSTRUCT, // compute_partial_construct
                        });
                        self.graph.set_partial_info(partial_node, PartialInfo {
                            subgraph_id: target_sg,
                            bound_count: args.len() as u8,
                        });
                        return partial_node;
                    }
                }
            }
        }

        // Dynamic closure call: callee is a non-Ident expression (e.g. arr[i], field.access,
        // a closure literal invoked directly fun() {...}(), etc.), evaluated at runtime to a Closure/Partial.
        // Use compute_closure_call (idx 41) for dynamic invocation; inputs[0] = callable value node.
        // current_effect appended at the end as an implicit dependency (consistent with the Ident closure call path).
        if !matches!(&callee_expr.node, crate::ast::Ast::Expr::Ident(_)) {
            let callable_node = self.compile_subexpr(callee);
            let mut inputs = Vec::with_capacity(args.len() + 2);
            inputs.push(callable_node);
            for &arg in args {
                inputs.push(self.compile_subexpr(arg));
            }
            if let Some(eff) = self.current_effect {
                inputs.push(eff);
            }
            let inputs_offset = self.graph.inputs_pool.push(&inputs);
            let call_node = self.graph.add_node(Node {
                kind: NodeKind::Call,
                input_count: inputs.len() as u8,
                inputs_offset,
                compute_fn: CF_CLOSURE_CALL, // compute_closure_call
            });
            self.graph.set_closure_call_arg_count(call_node, args.len() as u8);
            return call_node;
        }

        // Tail-recursion-to-iteration interception: currently compiling in a TailRecToLoop body and callee is self_name,
        // and in tail position (to avoid recursive calls in arguments being mistakenly intercepted),
        // generate WriteBack(param, actual) instead of Call(self).
        // body_sg is a LoopBody; after it completes, reset_loop_iteration automatically jumps back to while_sg to re-evaluate cond.
        if self.in_tail_position && self.tail_rec_ctx.is_some() {
            // Tail-recursion interception: generate WriteBack(param, actual) instead of Call(self)
        }
        if self.in_tail_position {
            if let Some(ctx) = &self.tail_rec_ctx.clone() {
                if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
                    if *name == ctx.self_name {
                    // Compile all actual-argument expressions (evaluate first, then WriteBack, to avoid races between parameters)
                    let arg_nodes: Vec<NodeId> = args
                        .iter()
                        .map(|&a| self.compile_subexpr(a))
                        .collect();
                    // Perform a WriteBack for each parameter (writing back to the function-level param nodes).
                    // Barrier mechanism: the first WriteBack depends on all arg_nodes;
                    // subsequent WriteBacks chain-depend on the previous WriteBack.
                    // This ensures all actual-argument expressions finish evaluating before any WriteBack executes,
                    // preventing a+b from reading the already-WriteBack-updated value of a.
                    //
                    // Only the last WriteBack uses CF_TAILREC_WRITEBACK (sets Continue);
                    // non-last WriteBacks use CF_WRITEBACK (do not set Continue).
                    // Reason: the Continue signal causes the frame to exit immediately and skip notify_downstream;
                    // if every WriteBack set Continue, subsequent chained WriteBacks would never become ready to execute.
                    let wb_count = arg_nodes.len().min(ctx.param_nodes.len());
                    let mut last_wb: Option<NodeId> = None;
                    for (i, &arg_node) in arg_nodes.iter().enumerate() {
                        if i < ctx.param_nodes.len() {
                            let mut wb_inputs = vec![arg_node];
                            if i == 0 {
                                // First WB: a barrier depending on all other arg_nodes
                                for &other in &arg_nodes[1..] {
                                    wb_inputs.push(other);
                                }
                            } else if let Some(prev_wb) = last_wb {
                                // Subsequent WBs: depend on the previous WB (chain ordering)
                                wb_inputs.push(prev_wb);
                            }
                            let is_last = i + 1 == wb_count;
                            let compute_fn = if is_last {
                                CF_TAILREC_WRITEBACK
                            } else {
                                CF_WRITEBACK
                            };
                            let wb_off = self.graph.inputs_pool.push(&wb_inputs);
                            let wb_node = self.graph.add_node(Node {
                                kind: NodeKind::Call,
                                input_count: wb_inputs.len() as u8,
                                inputs_offset: wb_off,
                                compute_fn,
                            });
                            self.graph.set_writeback_target(wb_node, ctx.param_nodes[i]);
                            self.current_effect = Some(wb_node);
                            last_wb = Some(wb_node);
                        }
                    }
                    // Return the last WriteBack node (after body_sg completes, reset_loop_iteration automatically jumps back)
                    return last_wb.unwrap_or_else(|| {
                        let off = self.graph.inputs_pool.push(&[]);
                        self.graph.add_node(Node {
                            kind: NodeKind::Const,
                            input_count: 0,
                            inputs_offset: off,
                            compute_fn: CF_NOOP,
                        })
                    });
                    }
                }
            }
        }

        // Non-tail-recursion-to-iteration interception: a self-call in non-tail position is replaced with push continuation + push sub-task + barrier(Continue)
        // Only intercepted when non_tail_rec_ctx is set (during state_N_sg compilation in compile_non_tail_rec_body_sg)
        if !self.in_tail_position && self.non_tail_rec_ctx.is_some() {
            let ctx_clone = self.non_tail_rec_ctx.clone();
            if let Some(ctx) = &ctx_clone {
                if let crate::ast::Ast::Expr::Ident(callee_name) = &callee_expr.node {
                    if *callee_name == ctx.self_name {
                        // 1. Check call_result_map: if the current call is already in the map, return the mapped node
                        if let Some(&mapped) = ctx.call_result_map.get(&call_expr_id) {
                            return mapped;
                        }
                        // 2. If already truncated, return a void constant (no Call node generated)
                        if ctx.truncated {
                            return self.compile_void_const();
                        }
                        // 3. Intercept: push continuation frame + push sub-task frame + barrier(Continue)

                        // Save current_effect: compile_subexpr may modify it;
                        // restore after compiling the actuals to ensure the store chain starts from the correct effect.
                        let saved_effect = self.current_effect;
                        let arg_nodes: Vec<NodeId> = args
                            .iter()
                            .map(|&a| self.compile_subexpr(a))
                            .collect();
                        self.current_effect = saved_effect;

                        let stride = ctx.stride;
                        let param_count = ctx.param_count;
                        let max_saved = ctx.max_saved;
                        let current_state = ctx.current_state as usize;
                        let stack_node = ctx.stack_node;
                        let sp_node = ctx.sp_node;
                        let result_node = ctx.result_node;

                        // Compute stack indices: base_cont = sp * stride, base_task = (sp + 1) * stride
                        // sp has already been decremented by pop (sp_node = original_sp - 1)
                        // cont writes to the slot freed by pop (overwriting the consumed frame); task writes to the next slot
                        // sp_new = sp + 2; on pop, sp-1 reads task first (LIFO), then cont
                        let one_const = self.make_i32_const(1);
                        let sp_plus_1 = self.make_binop(sp_node, one_const, CF_ADD_I32);
                        let two_const = self.make_i32_const(2);
                        let sp_plus_2 = self.make_binop(sp_node, two_const, CF_ADD_I32);
                        let stride_val = self.make_i32_const(stride as i32);
                        let base_cont = self.make_binop(sp_node, stride_val, CF_MUL_I32);
                        let base_task = self.make_binop(sp_plus_1, stride_val, CF_MUL_I32);

                        // Push continuation frame (write to the slot freed by pop)
                        // stack[base_cont + 0..P] = current parameters (param_cur nodes)
                        // All stores must be chained into the effect chain via chain_effects,
                        // ensuring the barrier triggers Continue only after all stores complete.
                        for i in 0..param_count {
                            let offset = self.make_i32_const(i as i32);
                            let idx = self.make_binop(base_cont, offset, CF_ADD_I32);
                            let store = self.make_array_store(stack_node, idx, ctx.param_nodes[i]);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }
                        // stack[base_cont + P] = state_after (current state + 1)
                        let state_after = self.make_i32_const((current_state + 1) as i32);
                        let state_offset_cont = self.make_i32_const(param_count as i32);
                        let state_idx_cont = self.make_binop(base_cont, state_offset_cont, CF_ADD_I32);
                        let state_store_cont =
                            self.make_array_store(stack_node, state_idx_cont, state_after);
                        self.current_effect = Some(self.chain_effects(self.current_effect, state_store_cont));
                        // stack[base_cont + P + 1..P + 1 + num_saved] = saved values
                        // For state S: slot j = saved_nodes[j] (j < S-1), result_node (j == S-1), 0 (j >= S)
                        let zero_saved = self.make_i32_const(0);
                        for j in 0..max_saved {
                            let offset = self.make_i32_const((param_count + 1 + j) as i32);
                            let idx = self.make_binop(base_cont, offset, CF_ADD_I32);
                            let val = if j < current_state {
                                if j + 1 < current_state {
                                    ctx.saved_nodes[j]
                                } else {
                                    // j == current_state - 1
                                    result_node
                                }
                            } else {
                                zero_saved
                            };
                            let store = self.make_array_store(stack_node, idx, val);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }

                        // Push sub-task frame (top of stack; read first on pop)
                        // stack[base_task + 0..P] = actuals (arg_nodes)
                        for i in 0..param_count {
                            let offset = self.make_i32_const(i as i32);
                            let idx = self.make_binop(base_task, offset, CF_ADD_I32);
                            let store = self.make_array_store(stack_node, idx, arg_nodes[i]);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }
                        // stack[base_task + P] = 0 (INIT state)
                        let state_offset_task = self.make_i32_const(param_count as i32);
                        let state_idx_task = self.make_binop(base_task, state_offset_task, CF_ADD_I32);
                        let state_store_task =
                            self.make_array_store(stack_node, state_idx_task, zero_saved);
                        self.current_effect = Some(self.chain_effects(self.current_effect, state_store_task));
                        // stack[base_task + P + 1..P + 1 + max_saved] = 0
                        for j in 0..max_saved {
                            let offset = self.make_i32_const((param_count + 1 + j) as i32);
                            let idx = self.make_binop(base_task, offset, CF_ADD_I32);
                            let store = self.make_array_store(stack_node, idx, zero_saved);
                            self.current_effect = Some(self.chain_effects(self.current_effect, store));
                        }

                        // WriteBack sp = sp + 2 (chained into effect to ensure it runs after all stores)
                        let sp_new = self.chain_effects(self.current_effect, sp_plus_2);
                        let sp_wb = self.compile_writeback_node(sp_new, sp_node);
                        self.current_effect = Some(sp_wb);

                        // Create the barrier node (Continue signal; blocks subsequent expression execution)
                        let barrier = self.make_continue_barrier(sp_wb);
                        self.current_effect = Some(barrier);

                        // Set the truncated flag
                        if let Some(ctx) = &mut self.non_tail_rec_ctx {
                            ctx.truncated = true;
                        }

                        return barrier;
                    }
                }
            }
        }

        // Regular function call
        // current_effect appended at the end as an implicit dependency (ensures Call executes only after prior effects complete)
        let mut inputs = Vec::with_capacity(args.len() + 1);
        for &arg in args {
            inputs.push(self.compile_subexpr(arg));
        }
        if let Some(eff) = self.current_effect {
            inputs.push(eff);
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        // Default sync call compute_fn (idx 36); async functions use compute_async_call_launch (idx 39)
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH, // compute_call_launch (sync)
        });

        // Bind the target subgraph — single-point resolution (see
        // `resolve_func`): current-module mangled → generic instance →
        // package key → bare (builtin/user only, collision-checked).
        if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
            let inst_id = self.sema.call_instantiations.get(&call_inst_key);
            let mangled = inst_id.map(|&id| format!("{}#{}", name, id));
            match self.resolve_func("compile_call", name, mangled.as_deref(), None) {
                Some(Ok(target_sg)) => {
                    self.graph.set_call_target(call_node, target_sg);
                    // is_async is derived at runtime by compute_call_launch from has_suspend;
                    // here we only check has_suspend to decide whether the call can be marked tail.
                    let is_async = self.graph.subgraphs.get(target_sg.0 as usize)
                        .is_some_and(|sg| sg.has_suspend);
                    // Tail-call marker: tail position + sync function + has call_target -> runtime switch_subgraph frame reuse
                    if self.in_tail_position && !is_async {
                        self.graph.set_tail_call(call_node);
                    }
                }
                Some(Err(diag)) => {
                    self.errors.push(diag);
                }
                // Nothing matched: the callee is not a declared function. A
                // target-less Call node evaluates to void at runtime and
                // silently poisons downstream arithmetic — fail the build,
                // with a qualified-name suggestion when one exists (std
                // functions have no bare slot, so a bare std call lands here).
                None => {
                    let mut quals: Vec<&String> = self.func_subgraphs
                        .keys()
                        .filter(|k| {
                            k.rsplit_once('.').map(|(_, f)| f == *name).unwrap_or(false)
                        })
                        .collect();
                    quals.sort();
                    let hint = if quals.is_empty() {
                        String::new()
                    } else {
                        format!(
                            " (did you mean {}?)",
                            quals.iter().map(|q| q.as_str()).collect::<Vec<_>>().join(" / ")
                        )
                    };
                    self.errors.push(format!(
                        "call to unknown function '{}' in module '{}': no target subgraph was bound{}",
                        name, self.current_module().name, hint
                    ));
                }
            }
        }

        call_node
    }

    /// Inline expansion: compile the callee body with formals bound to actual nodes.
    ///
    /// Enter a new scope -> compile actuals -> bind formal names -> compile body (non-tail position) -> exit scope.
    /// Does not generate a Call node or launch a subgraph; embeds the body's IR directly into the current function.
    pub(super) fn compile_inline_expansion(
        &mut self,
        body: crate::ast::Ast::ExprRef,
        params: &[crate::ast::Ast::Param<'_>],
        args: &[crate::ast::Ast::ExprId],
    ) -> NodeId {
        self.enter_scope();
        // Compile actuals and bind to formal names (actual nodes are compiled in the current scope context)
        for (param, &arg) in params.iter().zip(args.iter()) {
            let arg_node = self.compile_subexpr(arg);
            self.bind_var(param.name, arg_node);
        }
        // W4c: bodies with early `return` / `?` cannot be compiled straight into
        // the caller (the Return signal is function-scoped and would terminate
        // the CALLER — the old has_return/has_propagate inline blacklist).
        // Instead compile the body as a branch subgraph launched by a CAPTURE
        // Gate: the branch's Return signal is captured as the Gate's value
        // (exactly what a non-inlined call does — throw values flow as data,
        // Bug #65 semantics) and the caller keeps executing.
        let early_exit = {
            let arena = &self.current_module().arena;
            crate::pass::Analyzer::has_return(body, arena)
                || crate::pass::Analyzer::has_propagate(body, arena)
        };
        if early_exit {
            let body_node = self.compile_inline_wrap(body);
            self.exit_scope();
            return body_node;
        }
        // Compile the callee body (non-tail position; inline expansion does not preserve tail-call semantics)
        let body_node = self.compile_subexpr(body);
        self.exit_scope();
        body_node
    }

    /// W4c: capture-Gate inline wrap — `Gate(true -> body_sg, false -> void)`
    /// with `capture: true`. The body branch's Return signal becomes the
    /// Gate's value instead of propagating to the caller frame.
    fn compile_inline_wrap(&mut self, body: crate::ast::Ast::ExprRef) -> NodeId {
        // Non-tail: a tail call inside the wrapped body must not get the
        // caller's tail-call treatment (branch frames cannot tail-switch into
        // cross-function subgraphs).
        let prev_tail = self.in_tail_position;
        self.in_tail_position = false;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let (body_sg, body_inputs) = self.compile_branch_subgraph(body);
        let void_sg = self.compile_void_subgraph();
        self.current_effect = prev_effect;
        self.in_tail_position = prev_tail;

        let cond_node = self.compile_bool_const(true);
        let gate_inputs: Vec<NodeId> = match self.current_effect {
            Some(eff) => vec![cond_node, eff],
            None => vec![cond_node],
        };
        let inputs_offset = self.graph.inputs_pool.push(&gate_inputs);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: gate_inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            GateBranches {
                condition_input: cond_node,
                branches: vec![
                    (true, body_sg, body_inputs),
                    (false, void_sg, Vec::new()),
                ],
                capture: true,
            },
        );
        // Order subsequent statements after the inlined body (the call-site
        // analogue of chaining a Call node into the effect chain).
        self.current_effect = Some(self.chain_effects(self.current_effect, gate_node));
        gate_node
    }

    /// Look up a type declaration's field info (by type name).
    ///
    /// Uniformly searches layer by layer through type_scope_stack (top-level + nested types share the same lookup path).
    pub(super) fn lookup_type_field_names(&self, type_name: &str) -> Option<TypeFieldInfo> {
        self.lookup_type_fields(type_name)
    }

    /// Look up the field info of a specified constructor in a multi-constructor ADT.
    ///
    /// Uniformly searches layer by layer through type_scope_stack (top-level + nested types share the same lookup path).
    pub(super) fn lookup_constructor_field_names(&self, constructor_name: &str) -> Option<TypeFieldInfo> {
        self.lookup_type_fields(constructor_name)
    }

    /// Check if `Type.Ctor` is a qualified constructor access (IR-side).
    /// Returns `(type_name, ctor_name, field_names, kind, is_nullary)` for
    /// constructing the IR node.
    pub(super) fn check_qualified_ctor_ir(
        &self,
        type_name: &str,
        ctor_name: &str,
    ) -> Option<(String, String, Vec<Option<String>>, RecordLitKind, bool)> {
        let &type_idx = self.sema.type_def_index.get(type_name)?;
        let type_def = &self.sema.type_defs[&type_idx];
        let ctor = type_def
            .constructors
            .iter()
            .find(|c| c.name.as_ref() == ctor_name)?;
        let field_names: Vec<Option<String>> = ctor
            .field_names
            .iter()
            .map(|n| n.as_deref().map(String::from))
            .collect();
        let is_nullary = ctor.field_type_reprs.is_empty();
        let kind = match type_def.kind {
            crate::sema::Sema::TypeDefKind::Adt => RecordLitKind::Adt,
            crate::sema::Sema::TypeDefKind::Record => RecordLitKind::Record,
            crate::sema::Sema::TypeDefKind::Newtype => RecordLitKind::Newtype,
            crate::sema::Sema::TypeDefKind::Alias => RecordLitKind::Record,
        };
        Some((
            ctor.type_name.to_string(),
            ctor.name.to_string(),
            field_names,
            kind,
            is_nullary,
        ))
    }

    /// Check whether a function name is an @extern("C") function (has extern_c_body).
    pub(super) fn is_extern_c_func(&self, name: &str) -> bool {
        let modules: Vec<&crate::ast::Ast::Module<'_>> =
            std::iter::once(self.module).chain(self.builtin_modules.iter().copied()).collect();
        for m in modules {
            if let Some(d) = m.find_function(name) {
                if let crate::ast::Ast::Decl::FunDecl { extern_c_body, .. } = &d.node {
                    if extern_c_body.is_some() {
                        return true;
                    }
                }
            }
        }
        false
    }

    // (is_builtin_func removed: @builtin attribute eliminated. Reflect primitives
    // now lower as auto-impl trait methods / top-level wrappers to CF_REFLECT_*
    // compute_fns, never reaching the @extern dispatch path.)

    /// Build an AbiSig from a @extern("C") #{ }# function's declared params/return_type.
    /// str params are expanded to (Ptr, Int) two slots. Unknown types fall back to Int{64}.
    pub(super) fn build_abi_sig(&self, name: &str) -> crate::ffi::Abi::AbiSig {
        use crate::ast::Ast::Decl;
        let mut params = Vec::new();
        let mut ret = crate::ffi::Abi::AbiType::Void;
        for m in std::iter::once(self.module).chain(self.builtin_modules.iter().copied()) {
            if let Some(d) = m.find_function(name) {
                if let Decl::FunDecl { params: decl_params, return_type, .. } = &d.node {
                    // Use THIS module's arena (m.arena), not self.module.arena — stdlib functions
                    // have their TypeRefs in the builtin module's arena.
                    let arena = &m.arena;
                    for p in decl_params.iter() {
                        let ty_name = type_name_in_arena(p.type_annotation, arena);
                        self.push_abi_types(&ty_name, &mut params);
                    }
                    if let Some(rt) = return_type {
                        let rt_name = type_name_in_arena(Some(*rt), arena);
                        ret = self.abi_type_of(&rt_name);
                    }
                    break;
                }
            }
        }
        crate::ffi::Abi::AbiSig::new(params, ret)
    }

    /// Map a Frond type name to AbiType. str is handled separately by push_abi_types (two slots).
    pub(super) fn abi_type_of(&self, ty_name: &str) -> crate::ffi::Abi::AbiType {
        crate::ffi::Abi::abi_type_from_name(ty_name)
    }

    /// Reads a `Lib.embed` resource file, resolving `rel` against the entry
    /// module's directory (falling back to the process CWD when the entry has
    /// no source path, e.g. stdin).
    pub(super) fn read_embed_resource(&self, rel: &str) -> Result<Vec<u8>, String> {
        let base = self
            .module
            .source_path
            .and_then(|p| std::path::Path::new(p).parent())
            .map(|d| d.to_path_buf())
            .unwrap_or_default();
        let path = base.join(rel);
        std::fs::read(&path).map_err(|e| format!("cannot read '{}': {}", path.display(), e))
    }

    /// Registers an embedded resource (deduplicated by path name), returning
    /// its index into `graph.resources`.
    pub(super) fn add_embed_resource(&mut self, name: &str, bytes: Vec<u8>) -> u32 {
        if let Some(i) = self.graph.resources.iter().position(|(n, _)| &**n == name) {
            return i as u32;
        }
        self.graph.resources.push((
            std::sync::Arc::from(name),
            std::sync::Arc::from(bytes.into_boxed_slice()),
        ));
        (self.graph.resources.len() - 1) as u32
    }

    /// Extracts the `ForeignFn[R]` return ABI tag from a `lib.lookup(...)`
    /// call's static type (`Throw<ForeignFn<R>, FfiError>`). 0 (void) on any
    /// shape mismatch — Sema guarantees the annotation path, so this is a
    /// defensive fallback.
    fn lib_lookup_ret_tag(&self, call_expr_id: crate::ast::Ast::ExprId) -> u8 {
        if std::env::var("FROND_DEBUG_LIB").is_ok() {
            let dbg = self.expr_type_handle(call_expr_id).map(|h| {
                let r = self.type_arena.resolve(h);
                let inner = match self.type_arena.get(r) {
                    crate::types::Type::Throw(_) => {
                        let (v, e) = self.type_arena.throw_parts(r);
                        let vr = self.type_arena.resolve(v);
                        let ff_inner = match self.type_arena.get(vr) {
                            crate::types::Type::ForeignFn(_) => {
                                let ret = self.type_arena.foreign_fn_ret(vr);
                                let rr = self.type_arena.resolve(ret);
                                format!(" ForeignFn ret={:?} err={:?}", self.type_arena.get(rr), self.type_arena.get(self.type_arena.resolve(e)))
                            }
                            other => format!(" value={:?} err={:?}", other, self.type_arena.get(self.type_arena.resolve(e))),
                        };
                        format!("Throw{{{:?}{}}}", self.type_arena.get(vr), ff_inner)
                    }
                    other => format!("{:?}", other),
                };
                format!("{:?} -> {}", self.type_arena.get(r), inner)
            });
            eprintln!("[LIB-RET-TAG] expr {:?} ty = {:?}", call_expr_id, dbg);
        }
        self.expr_type_handle(call_expr_id)
            .map(|h| {
                let resolved = self.type_arena.resolve(h);
                if let crate::types::Type::Throw(_) = self.type_arena.get(resolved) {
                    let (v, _) = self.type_arena.throw_parts(resolved);
                    let fv = self.type_arena.resolve(v);
                    if let crate::types::Type::ForeignFn(_) = self.type_arena.get(fv) {
                        let r = self.type_arena.foreign_fn_ret(fv);
                        let rt = self.type_arena.get(self.type_arena.resolve(r));
                        return crate::ir::Compute::abi_name_to_lib_ret_kind(rt.name());
                    }
                }
                0
            })
            .unwrap_or(0)
    }

    /// Push AbiType(s) for a Frond type name. str and u8[] expand to (Ptr, Int) two
    /// slots, mirroring the DataLen C-side expansion in ffi/Gen.rs (`{p}_data`/`{p}_len`).
    pub(super) fn push_abi_types(&self, ty_name: &str, out: &mut Vec<crate::ffi::Abi::AbiType>) {
        crate::ffi::Abi::push_abi_types_for_name(ty_name, out)
    }

    /// Compile a method call.
    ///
    /// Method dispatch uniformly goes through the (type_id, method_idx) path:
    /// - intrinsic methods (await/len/send/recv/close/bytes/cancel etc.) are flagged via
    ///   MethodSigInfo.intrinsic and lowered directly to a compute_fn node
    /// - type/trait methods are compiled into Call nodes, looking up method_subgraphs via (type_id, method_idx)
    pub(super) fn compile_method_call(
        &mut self,
        call_expr_id: crate::ast::Ast::ExprId,
        recv: crate::ast::Ast::ExprId,
        method: &str,
        args: &[crate::ast::Ast::ExprId],
        recv_node_override: Option<NodeId>,
    ) -> NodeId {
        // Lib.open(path) / Lib.embed(path): builtin native-library constructors.
        // The receiver `Lib` is a type name (flagged module_func_recv by sema),
        // never compiled as a value node. embed requires a string literal so
        // the file can be captured into the artifact's resources at build time.
        if let crate::ast::Ast::Expr::Ident("Lib") = &self.current_module().arena.expr(recv).node {
            let recv_key = crate::sema::Sema::module_expr_key(
                self.expr_key_module(),
                recv.0 as u64,
            );
            if self.sema.module_func_recv_exprs.contains(&recv_key)
                && (method == "open" || method == "embed")
            {
                let span = self.current_module().arena.expr(call_expr_id).span;
                if args.len() != 1 {
                    self.errors.push(format!(
                        "Lib.{} takes exactly 1 argument (path: str) at line {}",
                        method, span.line
                    ));
                    return self.compile_placeholder();
                }
                match method {
                    "open" => {
                        let mut inputs = Vec::with_capacity(2);
                        inputs.push(self.compile_subexpr(args[0]));
                        if let Some(eff) = self.current_effect {
                            inputs.push(eff);
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&inputs);
                        return self.graph.add_node(Node {
                            kind: NodeKind::Call,
                            input_count: inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_LIB_OPEN,
                        });
                    }
                    "embed" => {
                        // Compile-time capture: the path must be a string literal.
                        let lit = match &self.current_module().arena.expr(args[0]).node {
                            crate::ast::Ast::Expr::StrLit(s) => Some(*s),
                            _ => None,
                        };
                        let rel_path = match lit {
                            Some(p) => p,
                            None => {
                                self.errors.push(format!(
                                    "Lib.embed requires a string-literal path at line {} (the file is captured into the artifact at build time)",
                                    span.line
                                ));
                                return self.compile_placeholder();
                            }
                        };
                        let bytes = match self.read_embed_resource(rel_path) {
                            Ok(b) => b,
                            Err(e) => {
                                self.errors.push(format!(
                                    "Lib.embed('{}') at line {}: {}",
                                    rel_path, span.line, e
                                ));
                                return self.compile_placeholder();
                            }
                        };
                        let res_idx = self.add_embed_resource(rel_path, bytes);
                        let mut inputs = Vec::with_capacity(2);
                        inputs.push(self.compile_subexpr(args[0]));
                        if let Some(eff) = self.current_effect {
                            inputs.push(eff);
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&inputs);
                        let node = self.graph.add_node(Node {
                            kind: NodeKind::Call,
                            input_count: inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_LIB_EMBED,
                        });
                        self.graph.set_embed_info(node, res_idx);
                        return node;
                    }
                    _ => unreachable!(),
                }
            }
        }

        // Qualified-name constructor: Type.Ctor(args) (constructor with parameters)
        if let crate::ast::Ast::Expr::Ident(type_name) = &self.current_module().arena.expr(recv).node {
            if let Some((ctor_type_name, ctor_name, field_names, kind, is_nullary)) =
                self.check_qualified_ctor_ir(type_name, method)
            {
                if !is_nullary {
                    let mut inputs = Vec::with_capacity(args.len());
                    for &arg in args {
                        inputs.push(self.compile_subexpr(arg));
                    }
                    let inputs_offset = self.graph.inputs_pool.push(&inputs);
                    let node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: inputs.len() as u8,
                        inputs_offset,
                        compute_fn: CF_RECORD_CONSTRUCT,
                    });
                    self.graph.set_record_lit_info(node, RecordLitInfo {
                        type_name: ctor_type_name,
                        field_names,
                        constructor: ctor_name,
                        kind,
                    });
                    return node;
                }
            }
        }

        // super.method(args): static dispatch to the bound trait-default
        // subgraph. Bypasses the vtable (Path 1) and the type's own override
        // (Path 2) entirely — sema resolved the target at inference time
        // (super_dispatches) and the monomorph phase guarantees the default
        // subgraph exists (super_targets).
        if let crate::ast::Ast::Expr::Ident("super") = &self.current_module().arena.expr(recv).node {
            return self.compile_super_method_call(call_expr_id, method, args);
        }

        // When the caller supplies a pre-compiled receiver node (e.g. implicit-this method
        // calls where `this` is already bound), skip re-compiling the recv expression.
        let recv_node = match recv_node_override {
            Some(n) => n,
            None => self.compile_subexpr(recv),
        };

        // -- intrinsic lowering --
        // First look up the language-level intrinsic flag in sema method_dispatches (await/recv);
        // on miss, fall back to (type_id, method_idx) lookup of MethodSigInfo.intrinsic (send/close/len etc.).
        // When conditions are not met (e.g. argument count mismatch), fall through to the Call node path.
        let dispatch_intrinsic = {
            let key = crate::sema::Sema::module_expr_key(
                self.expr_key_module(),
                call_expr_id.0 as u64,
            );
            self.sema.method_dispatches.get(&key).and_then(|d| d.intrinsic)
        };
        let intrinsic = dispatch_intrinsic.or_else(|| self.lookup_intrinsic(recv, method));
        if let Some(intrinsic) = intrinsic {
            if let Some(node) = self.try_lower_intrinsic(recv, recv_node, args, intrinsic) {
                return node;
            }
        }

        // ── Lib / ForeignFn builtin methods (structural, reflect-style dispatch) ──
        // lookup/has_symbol/close on `Lib`; call (any arity) on `ForeignFn[R]`.
        // Pairs with the Sema-side lib_method_return_type recognition.
        {
            let recv_ty = self.expr_type_name(recv).unwrap_or("");
            if recv_ty == "Lib" {
                match method {
                    "lookup" if args.len() == 2 => {
                        let ret_tag = self.lib_lookup_ret_tag(call_expr_id);
                        let mut inputs = vec![recv_node];
                        for &arg in args {
                            inputs.push(self.compile_subexpr(arg));
                        }
                        if let Some(eff) = self.current_effect {
                            inputs.push(eff);
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&inputs);
                        let node = self.graph.add_node(Node {
                            kind: NodeKind::Call,
                            input_count: inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_LIB_LOOKUP,
                        });
                        self.graph.set_lib_ret_kind(node, ret_tag);
                        return node;
                    }
                    "has_symbol" if args.len() == 1 => {
                        let mut inputs = vec![recv_node];
                        for &arg in args {
                            inputs.push(self.compile_subexpr(arg));
                        }
                        if let Some(eff) = self.current_effect {
                            inputs.push(eff);
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&inputs);
                        return self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_LIB_HAS_SYMBOL,
                        });
                    }
                    "close" if args.is_empty() => {
                        let mut inputs = vec![recv_node];
                        if let Some(eff) = self.current_effect {
                            inputs.push(eff);
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&inputs);
                        return self.graph.add_node(Node {
                            kind: NodeKind::UnOp,
                            input_count: inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_LIB_CLOSE,
                        });
                    }
                    _ => {}
                }
            } else if recv_ty == "ForeignFn" && method == "call" {
                // Any arity: [ffn, args...] + effect; arg count via the shared
                // closure_call_arg_count metadata slot.
                let mut inputs = Vec::with_capacity(1 + args.len() + 1);
                inputs.push(recv_node);
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                if let Some(eff) = self.current_effect {
                    inputs.push(eff);
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                let node = self.graph.add_node(Node {
                    kind: NodeKind::Call,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: CF_FFN_CALL,
                });
                self.graph.set_closure_call_arg_count(node, args.len() as u8);
                return node;
            }
        }

        // Path 0: module function call (recv is a constructor/module namespace; recv is not passed)
        // recv flagged by sema MethodCall path 0a/0b: ModuleRef.free_func(args) / TypeName.free_func(args)
        // recv is not passed as an argument (free_func is a free function; it does not take recv)
        // Generic calls prefer call_instantiations lookup to bind the specialized subgraph by mangled name;
        // non-generic calls fall back to the bare name (same mangled-lookup logic as compile_call)
        {
            let recv_key = crate::sema::Sema::module_expr_key(
                self.expr_key_module(),
                recv.0 as u64,
            );
            if self.sema.module_func_recv_exprs.contains(&recv_key) {
                let call_inst_key = crate::sema::Sema::module_expr_key(
                    self.expr_key_module(),
                    call_expr_id.0 as u64,
                );
                let inst_id = self.sema.call_instantiations.get(&call_inst_key);
                let mangled = inst_id.map(|&id| format!("{}#{}", method, id));
                // Qualified-first: Sema classified the recv identifier as a
                // module name, so it carries the qualifier a bare lookup
                // would discard. `resolve_func` tries current-module mangled
                // → instance → package key → `Recv.method` (this shape) →
                // bare. The liveness/DCE gaps that forced the earlier revert
                // are closed structurally now (NodeRef door + unified
                // liveness closure in the Optimizer).
                let recv_ident = match &self.current_module().arena.expr(recv).node {
                    crate::ast::Ast::Expr::Ident(n) => Some(*n),
                    _ => None,
                };
                match self.resolve_func("path0_module_recv", method, mangled.as_deref(), recv_ident) {
                    Some(Ok(target_sg)) => {
                        let mut inputs = Vec::with_capacity(args.len() + 1);
                        for &arg in args {
                            inputs.push(self.compile_subexpr(arg));
                        }
                        if let Some(eff) = self.current_effect {
                            inputs.push(eff);
                        }
                        let inputs_offset = self.graph.inputs_pool.push(&inputs);
                        let call_node = self.graph.add_node(Node {
                            kind: NodeKind::Call,
                            input_count: inputs.len() as u8,
                            inputs_offset,
                            compute_fn: CF_CALL_LAUNCH,
                        });
                        self.graph.set_call_target(call_node, target_sg);
                        return call_node;
                    }
                    Some(Err(diag)) => {
                        self.errors.push(diag);
                    }
                    None => {
                        self.errors.push(format!(
                            "module function call '{}.{}' did not resolve to any target subgraph",
                            recv_ident.unwrap_or("<?>"), method
                        ));
                    }
                }
            }
        }

        {
            // Type-driven method dispatch: (type_id, method_idx) structured key lookup into method_subgraphs
            // current_effect appended at the end as an implicit dependency (ensures Call executes only after prior effects complete)
            let mut inputs = Vec::with_capacity(2 + args.len());
            inputs.push(recv_node);
            for &arg in args {
                inputs.push(self.compile_subexpr(arg));
            }
            if let Some(eff) = self.current_effect {
                inputs.push(eff);
            }
            let inputs_offset = self.graph.inputs_pool.push(&inputs);
            let call_node = self.graph.add_node(Node {
                kind: NodeKind::Call,
                input_count: inputs.len() as u8,
                inputs_offset,
                compute_fn: CF_CALL_LAUNCH,
            });

            // Dispatch priority (semantic priority, not fallback):
            //   1. trait object dynamic dispatch (recv type is a trait -> vtable runtime dispatch)
            //   2. type's own method / trait method override: (type_id, method_idx) lookup into method_subgraphs
            //   3. trait default method: (type_id, trait_def_idx, method_idx_in_trait) lookup into trait_default_subgraphs

            // Path 1: trait object dynamic dispatch (vtable)
            if self.is_trait_object_recv(recv) {
                // Look up method_idx from trait_def.methods (consistent with TraitValue.method_values index)
                let trait_name = self.expr_type_name(recv).unwrap_or("").to_string();
                let method_idx = self.sema.get_trait_def(&trait_name)
                    .and_then(|td| td.methods.iter().position(|m| m.name.as_ref() == method))
                    .map(|i| i as u16);
                match method_idx {
                    Some(idx) => {
                        self.graph.set_vtable_call(call_node, idx);
                        // Populate the vtable_fallback_dispatch table so that when a concrete
                        // record (not a TraitVal) is passed as a trait-typed parameter, the
                        // runtime can statically dispatch via (method_idx, type_name) → SubGraphId.
                        // Enumerate all types implementing this trait via the witness table.
                        self.populate_vtable_fallback(&trait_name, idx, method);
                    }
                    None => self.errors.push(format!(
                        "internal: trait method '{}' not found in trait '{}' for vtable dispatch",
                        method, &trait_name)),
                }
                return call_node;
            }

            // Path 2: type's own method / trait method override
            // When recv_node_override is set (implicit-this call), the callee ExprId does not carry
            // the receiver's type info; use current_method_type (set by the enclosing method compile).
            let recv_type: Option<(&str, u16)> = if recv_node_override.is_some() {
                self.current_method_type.as_ref().map(|(n, id)| (n.as_ref(), *id))
            } else {
                self.expr_type_name(recv).zip(self.expr_type_id(recv))
            };
            if let Some((type_name, type_id)) = recv_type {
                if let Some(method_idx) = self.sema.lookup_method_idx(type_name, method) {
                    if let Some(&target_sg) = self.method_subgraphs.get(&(type_id, method_idx)) {
                        self.graph.set_call_target(call_node, target_sg);
                        return call_node;
                    }
                }
            }

            // Path 3: trait default method (type does not override; fall back to the monomorphized specialized version of the trait default impl)
            let path3_type_id = if recv_node_override.is_some() {
                self.current_method_type.as_ref().map(|(_, id)| *id)
            } else {
                self.expr_type_id(recv)
            };
            if let Some(type_id) = path3_type_id {
                for trait_def in self.sema.trait_defs.values() {
                    if !self.type_implements_trait(type_id, &trait_def.name) {
                        continue;
                    }
                    if let Some(method_idx) = trait_def
                        .methods
                        .iter()
                        .position(|m| m.name.as_ref() == method && m.has_body)
                    {
                        if let Some(&trait_idx) = self.sema.trait_def_index.get(trait_def.name.as_ref()) {
                            if let Some(&target_sg) = self.trait_default_subgraphs.get(&(type_id, trait_idx, method_idx as u16)) {
                                self.graph.set_call_target(call_node, target_sg);
                                return call_node;
                            }
                        }
                    }
                }
            }

            // Path 4: free-function method call (recv.method(args) -> method(recv, args))
            // When the method name matches a top-level free function, recv is passed as the first argument.
            // Single-point resolution: builtin/user free functions resolve bare; std free
            // functions (no bare slot) resolve only through the earlier qualified paths.
            if let Some(Ok(target_sg)) = self.resolve_func("path4_free_fn", method, None, None) {
                self.graph.set_call_target(call_node, target_sg);
                return call_node;
            }

            call_node
        }
    }

    /// Compile `super.method(args)` — a static call to the specialized
    /// trait-default subgraph of the enclosing type.
    ///
    /// The receiver is the in-scope `this` binding (super is a layer view of
    /// `this`, not a value); the target comes from sema's `super_dispatches`
    /// keyed by this call expression. On any lookup miss (which sema should
    /// have prevented) an error is recorded and a no-op node is returned rather
    /// than falling through — the normal paths would re-dispatch to the
    /// override and recurse infinitely.
    fn compile_super_method_call(
        &mut self,
        call_expr_id: crate::ast::Ast::ExprId,
        method: &str,
        args: &[crate::ast::Ast::ExprId],
    ) -> NodeId {
        let key = crate::sema::Sema::module_expr_key(
            self.expr_key_module(),
            call_expr_id.0 as u64,
        );
        let dispatch = self.sema.super_dispatches.get(&key).copied();
        let type_id = self.current_method_type.as_ref().map(|(_, id)| *id);
        let target_sg = dispatch
            .zip(type_id)
            .and_then(|((trait_idx, method_idx), tid)| {
                self.trait_default_subgraphs
                    .get(&(tid, trait_idx, method_idx))
                    .copied()
            });

        let noop = |this: &mut Self| -> NodeId {
            let inputs_offset = this.graph.inputs_pool.push(&[]);
            this.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            })
        };

        let target_sg = match target_sg {
            Some(sg) => sg,
            None => {
                let type_name = self
                    .current_method_type
                    .as_ref()
                    .map(|(n, _)| n.to_string())
                    .unwrap_or_default();
                self.errors.push(format!(
                    "super call '{}' could not be resolved to a trait-default subgraph (type '{}')",
                    method, type_name
                ));
                return noop(self);
            }
        };

        // `this` of the enclosing method is the receiver argument.
        let recv_node = match self.lookup_var("this") {
            Some(n) => n,
            None => {
                self.errors.push(
                    "super call: the receiver 'this' is not in scope here (super is only \
                     supported in the direct body of a type method)"
                        .into(),
                );
                return noop(self);
            }
        };

        let mut inputs = Vec::with_capacity(2 + args.len());
        inputs.push(recv_node);
        for &arg in args {
            inputs.push(self.compile_subexpr(arg));
        }
        if let Some(eff) = self.current_effect {
            inputs.push(eff);
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        let call_node = self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_CALL_LAUNCH,
        });
        self.graph.set_call_target(call_node, target_sg);
        call_node
    }

    /// Look up MethodSigInfo.intrinsic via (type_id, method_idx) and return the lowering strategy.
    ///
    /// Intrinsic methods of built-in types (e.g. Async.await, Channel.send, Array.len) have the intrinsic
    /// field annotated when Sema registers the synthetic TypeDefInfo; this lookups uniformly, without special-casing by method name.
    pub(super) fn lookup_intrinsic(
        &self,
        recv: crate::ast::Ast::ExprId,
        method: &str,
    ) -> Option<crate::sema::Sema::IntrinsicKind> {
        // First: reflect trait methods (auto-impl). Recognized structurally by
        // method name, so every type — including builtins and generic type vars —
        // gets reflect methods without needing witness-table registration.
        if let Some((kind, _argc)) = reflect_method_intrinsic(method) {
            // Guard: only lower as reflect intrinsic if the receiver's type does
            // NOT already define a real method of the same name (user override wins).
            let shadows = self.expr_type_name(recv)
                .and_then(|tn| self.sema.lookup_method_idx(tn, method))
                .is_some();
            if !shadows {
                return Some(kind);
            }
        }
        let type_name = self.expr_type_name(recv)?;
        let type_id = self.expr_type_id(recv)?;
        let method_idx = self.sema.lookup_method_idx(type_name, method)?;
        let sig = self.sema.get_method_sig(type_id, method_idx)?;
        sig.intrinsic
    }

    /// Try to lower to a compute_fn node based on IntrinsicKind.
    ///
    /// Returns None when conditions are not met (e.g. argument count mismatch, recv type mismatch);
    /// the caller should fall through to the Call node path.
    pub(super) fn try_lower_intrinsic(
        &mut self,
        recv: crate::ast::Ast::ExprId,
        recv_node: NodeId,
        args: &[crate::ast::Ast::ExprId],
        kind: crate::sema::Sema::IntrinsicKind,
    ) -> Option<NodeId> {
        use crate::sema::Sema::IntrinsicKind;
        match kind {
            // await: unconditionally lower to Await (EventSource + Await dual node)
            IntrinsicKind::Await if args.is_empty() => {
                Some(self.build_await_node(recv, recv_node))
            }
            // recv: only lower to Await when recv's type is Channel/Receiver
            IntrinsicKind::ChannelAwait if args.is_empty() => {
                if self.infer_event_source_kind(recv) == crate::ir::Ir::EventSourceKind::Channel {
                    Some(self.build_await_node(recv, recv_node))
                } else {
                    None
                }
            }
            // cancel/len/close/bytes: single-node unary op (no arguments)
            IntrinsicKind::UnOp(idx) if args.is_empty() => {
                let inputs_offset = self.graph.inputs_pool.push(&[recv_node]);
                Some(self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset,
                    compute_fn: ComputeFnId(idx),
                }))
            }
            // send(value): binary op, inputs = [recv, value]
            IntrinsicKind::BinOp(idx) => {
                let mut inputs = vec![recv_node];
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                Some(self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: ComputeFnId(idx),
                }))
            }
            // compare_exchange(expected, new): ternary op, inputs = [recv, expected, new]
            IntrinsicKind::TriOp(idx) => {
                let mut inputs = vec![recv_node];
                for &arg in args {
                    inputs.push(self.compile_subexpr(arg));
                }
                let inputs_offset = self.graph.inputs_pool.push(&inputs);
                Some(self.graph.add_node(Node {
                    kind: NodeKind::TriOp,
                    input_count: inputs.len() as u8,
                    inputs_offset,
                    compute_fn: ComputeFnId(idx),
                }))
            }
            _ => None, // argument mismatch; fall through to the Call node path
        }
    }

    /// Check whether a type implements a specified trait (queries any method slot via witness_table).
    pub(super) fn type_implements_trait(&self, type_id: u16, trait_name: &str) -> bool {
        for entry in self.sema.witness_table.entries() {
            if entry.trait_name.as_ref() == trait_name && entry.type_id == type_id {
                return true;
            }
        }
        false
    }

    /// Populate the vtable_fallback_dispatch table for a trait method call site.
    ///
    /// For every concrete type that implements `trait_name`, resolve the method's
    /// subgraph via `(type_id, method_idx_in_type_def)` and store it keyed by
    /// `(vtable_method_idx, type_name)`. At runtime, when a vtable call receives a
    /// concrete record (not a TraitVal), `compute_call_launch` looks up the value's
    /// `type_name` here to statically dispatch.
    pub(super) fn populate_vtable_fallback(&mut self, trait_name: &str, vtable_idx: u16, method_name: &str) {
        // Approach: scan all TypeDecls in the user module (top-level + local) for types that
        // have a method matching `method_name`. For each, register the method subgraph keyed by
        // (vtable_idx, type_name). This handles both explicit trait declarations (`: Trait`) and
        // structural trait implementations (methods present without explicit declaration).
        //
        // First try the witness table (explicit declarations); if empty, fall back to scanning
        // all types with a matching method name.
        let mut entries: Vec<(u16, u16)> = Vec::new();
        for entry in self.sema.witness_table.entries() {
            if entry.trait_name.as_ref() != trait_name {
                continue;
            }
            if let Some(type_method_idx) = self.sema.witness_table.resolve_method(trait_name, entry.type_id, method_name) {
                entries.push((entry.type_id, type_method_idx));
            }
        }
        // Structural fallback: if witness table has no entries for this trait, scan all types
        // that have a method with the matching name. This supports `type Dog { fun name(): str }`
        // being passed as `Animal` without an explicit `: Animal` declaration.
        if entries.is_empty() {
            for (&type_idx, type_def) in &self.sema.type_defs {
                for (m_idx, m) in type_def.methods.iter().enumerate() {
                    if m.name.as_ref() == method_name {
                        entries.push((crate::types::dynamic_type_id(type_idx), m_idx as u16));
                        break;
                    }
                }
            }
        }
        for (type_id, type_method_idx) in entries {
            if let Some(&sg) = self.method_subgraphs.get(&(type_id, type_method_idx)) {
                if let Some(name) = self.type_name_from_id(type_id) {
                    self.graph.vtable_fallback_dispatch.insert((vtable_idx, name.into_boxed_str()), sg);
                }
            }
        }
    }

    /// Reverse-lookup a type_name from a dynamic type_id.
    pub(super) fn type_name_from_id(&self, type_id: u16) -> Option<String> {
        for (&type_idx, type_def) in &self.sema.type_defs {
            if crate::types::dynamic_type_id(type_idx) == type_id {
                return Some(type_def.name.as_ref().to_string());
            }
        }
        None
    }

    /// Determine whether recv is a trait object (needs runtime dynamic dispatch).
    ///
    /// Look up recv's type name; if it is a trait name registered in sema.trait_defs, it needs vtable dynamic dispatch.
    pub(super) fn is_trait_object_recv(&self, recv: crate::ast::Ast::ExprId) -> bool {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), recv.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            if let Some(tn) = &info.type_name {
                return self
                    .sema
                    .trait_defs
                    .values()
                    .any(|td| td.name.as_ref() == tn.as_ref());
            }
        }
        false
    }

    /// Get the type_id of an expression (looked up from SemaResult.expr_types).
    ///
    /// type_id computation is consistent with populate_witness_table: type_def_index[name] + FIRST_DYNAMIC_TYPE_ID.
    /// When Sema has no record (self in a specialized trait default method version), look up sema's
    /// TraitDefaultInstance.type_name to get the concrete implementation type name, then query type_def_index.
    pub(super) fn expr_type_id(&self, expr: crate::ast::Ast::ExprId) -> Option<u16> {
        // self in a specialized trait default method version: consume sema's TraitDefaultInstance.type_name
        if let Some(idx) = self.current_trait_default_idx {
            if let crate::ast::Ast::Expr::Ident(name) = &self.module.arena.expr(expr).node {
                if *name == "this" {
                    if let Some(inst) = self.sema.trait_default_instances.get(idx) {
                        return self
                            .sema
                            .type_def_index
                            .get(inst.type_name.as_ref())
                            .map(|&idx| crate::types::dynamic_type_id(idx));
                    }
                }
            }
        }
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr.0 as u64);
        let info = self.sema.expr_types.get(&key)?;
        // Consistent with expr_type_name: prefer type_name, fall back to Type::name()
        // (built-in structural variants like array/nullable/str/Throw return their registered name via Type::name();
        // "unknown" only appears in degenerate paths where Adt/Record arena lookup fails).
        let type_name = info
            .type_name
            .as_deref()
            .unwrap_or_else(|| self.type_arena.get(info.ty).name());
        self.sema
            .type_def_index
            .get(type_name)
            .map(|&idx| crate::types::dynamic_type_id(idx))
    }

}
