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

    /// S2 ctor-resolution consumption (NAME_RESOLUTION_PLAN): build the
    /// construction info from sema's recorded canonical type for this call
    /// expr, instead of re-resolving the bare name through the string tables
    /// (first-wins under cross-module same-name types). Constructor entry by
    /// name, with arity fallback.
    pub(super) fn ctor_tf_info_from_resolution(
        &self,
        call_expr_id: crate::ast::Ast::ExprId,
        ctor_name: &str,
        args_len: usize,
    ) -> Option<TypeFieldInfo> {
        let ckey = crate::sema::Sema::module_expr_key(
            self.expr_key_module(),
            call_expr_id.0 as u64,
        );
        let &sym = self.sema.ctor_resolutions.get(&ckey)?;
        let canonical = self.sema.symbols.resolve(sym);
        let idx = self.sema.type_def_idx(canonical)?;
        let def = self.sema.type_defs.get(&idx)?;
        let ctor = def
            .constructors
            .iter()
            .find(|cc| cc.name.as_ref() == ctor_name)
            .or_else(|| {
                def.constructors
                    .iter()
                    .find(|cc| cc.field_type_reprs.len() == args_len)
            })?;
        let kind = match def.kind {
            crate::sema::Sema::TypeDefKind::Adt => RecordLitKind::Adt,
            crate::sema::Sema::TypeDefKind::Record => RecordLitKind::Record,
            crate::sema::Sema::TypeDefKind::Newtype => RecordLitKind::Newtype,
            crate::sema::Sema::TypeDefKind::Alias => RecordLitKind::Record,
        };
        Some(TypeFieldInfo {
            field_names: ctor
                .field_names
                .iter()
                .map(|n| n.as_deref().unwrap_or("_").to_string())
                .collect(),
            type_name: ctor.type_name.to_string(),
            kind,
        })
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
                // S2 (NAME_RESOLUTION_PLAN): sema already made the ONE resolution
                // decision for this construct (recorded on the call expr as the
                // constructed type's canonical Sym). Consume it first — the
                // string-keyed scope lookups below (bare-name, first-wins,
                // registration-order-sensitive) are a FALLBACK whose hits are
                // measured via FROND_TRACE_S2; the bug classes they caused
                // (empty field tables from variant/type name collisions, std
                // bare-name hijacks) are bypassed entirely on the ID path.
                let mut tf_info: Option<TypeFieldInfo> =
                    self.ctor_tf_info_from_resolution(call_expr_id, name, args.len());
                let s2_hit = tf_info.is_some();
                // String-path fallback (S1-era): bare-name scope layers, then the
                // ctor table with arity correction。测量口径:仅当字符串路径
                // 真正产出构造条目(而非普通函数调用落空)才计回退命中。
                let mut tf_info = match tf_info {
                    Some(info) => Some(info),
                    None => self
                        .lookup_type_field_names(name)
                        .or_else(|| self.lookup_constructor_field_names(name)),
                };
                if !s2_hit && tf_info.is_some() {
                    if std::env::var("FROND_TRACE_S2").is_ok() {
                        eprintln!(
                            "[s2-fallback:construct] name={} mod={} arity={}",
                            *name,
                            self.current_module().name,
                            args.len()
                        );
                    }
                }
                if let Some(info) = tf_info.as_ref() {
                    if info.field_names.len() != args.len() && !args.is_empty() {
                        if let Some(exact) = self.lookup_ctor_info_by_arity(name, args.len()) {
                            tf_info = Some(exact);
                        }
                    }
                }
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
                // Cell-backed binding: the callable routes through the cell —
                // the FORWARDED value node when the Builder knows it (zero
                // cost), else a CF_DEREF_READ load. `lookup_var` alone would
                // hand back the CELL node itself (its value is a Ref(Arc<Cell>),
                // not callable).
                let cell_backed = self.lookup_cell_binding(name).and_then(|c| {
                    self.cell_forwarded_value(c).or_else(|| {
                        let (count, off) = match self.current_effect {
                            Some(eff) => (2, self.graph.inputs_pool.push(&[c, eff])),
                            None => (1, self.graph.inputs_pool.push(&[c])),
                        };
                        let load = self.graph.add_node(Node {
                            kind: NodeKind::UnOp,
                            input_count: count,
                            inputs_offset: off,
                            compute_fn: CF_DEREF_READ,
                        });
                        self.track_cell_store(c, load);
                        Some(load)
                    })
                });
                if let Some(closure_node) = cell_backed.or_else(|| self.lookup_var(name)) {
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
        // store the actuals through the parameter-register Cells instead of Call(self).
        // body_sg is a LoopBody; after it completes, reset_loop_iteration automatically jumps back to while_sg to re-evaluate cond.
        if self.in_tail_position && self.tail_rec_ctx.is_some() {
            // Tail-recursion interception: cell stores instead of Call(self)
        }
        if self.in_tail_position {
            if let Some(ctx) = &self.tail_rec_ctx.clone() {
                if let crate::ast::Ast::Expr::Ident(name) = &callee_expr.node {
                    if *name == ctx.self_name {
                    // Compile all actual-argument expressions (evaluate first, then store, to avoid races between parameters)
                    let arg_nodes: Vec<NodeId> = args
                        .iter()
                        .map(|&a| self.compile_subexpr(a))
                        .collect();
                    // B2: store each actual through the parameter-register Cell
                    // (CF_DEREF_WRITE). Barrier mechanism mirrors the old
                    // WriteBack chain: the first store depends on all arg_nodes
                    // (all actuals finish evaluating before any store executes —
                    // `self(a, a+1)` must not read an already-updated cell);
                    // subsequent stores chain-depend on the previous store.
                    // Inputs past [cell, value] are ordering-only deps (the
                    // compute fn reads inputs[0..2]) — same convention as the
                    // assignment path's trailing effect input.
                    let mut last_store: Option<NodeId> = None;
                    for (i, &arg_node) in arg_nodes.iter().enumerate() {
                        if i < ctx.param_cells.len() {
                            let mut store_inputs = vec![ctx.param_cells[i], arg_node];
                            if i == 0 {
                                // First store: a barrier depending on all other arg_nodes
                                for &other in &arg_nodes[1..] {
                                    store_inputs.push(other);
                                }
                            } else if let Some(prev_store) = last_store {
                                // Subsequent stores: depend on the previous store (chain ordering)
                                store_inputs.push(prev_store);
                            }
                            let store_off = self.graph.inputs_pool.push(&store_inputs);
                            let store_node = self.graph.add_node(Node {
                                kind: NodeKind::BinOp,
                                input_count: store_inputs.len() as u8,
                                inputs_offset: store_off,
                                compute_fn: CF_DEREF_WRITE,
                            });
                            self.track_cell_store(ctx.param_cells[i], arg_node);
                            self.current_effect = Some(store_node);
                            last_store = Some(store_node);
                        }
                    }
                    // Continue barrier (terminates the body-sg frame; the loop
                    // re-evaluates the condition against the updated cells).
                    let barrier_dep = match last_store {
                        Some(s) => s,
                        None => match self.current_effect {
                            Some(eff) => eff,
                            None => {
                                let off = self.graph.inputs_pool.push(&[]);
                                self.graph.add_node(Node {
                                    kind: NodeKind::Const,
                                    input_count: 0,
                                    inputs_offset: off,
                                    compute_fn: CF_NOOP,
                                })
                            }
                        },
                    };
                    let barrier = self.make_continue_barrier(barrier_dep);
                    self.current_effect = Some(barrier);
                    // Return the barrier node (after body_sg completes, reset_loop_iteration automatically jumps back)
                    return barrier;
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
                        // 1. Check call_result_map: if the current call is already in the map,
                        //    return the mapped node. The RESULT_CELL_MARKER case synthesizes a
                        //    CF_DEREF_READ of the result Cell HERE (in the consuming state sg):
                        //    it executes after the producing state's store (states run
                        //    sequentially), reading the current Cell value.
                        if let Some(&mapped) = ctx.call_result_map.get(&call_expr_id) {
                            if mapped == RESULT_CELL_MARKER {
                                let (lc, lo) = match self.current_effect {
                                    Some(eff) => (2, self.graph.inputs_pool.push(&[ctx.result_cell, eff])),
                                    None => (1, self.graph.inputs_pool.push(&[ctx.result_cell])),
                                };
                                let load = self.graph.add_node(Node {
                                    kind: NodeKind::UnOp,
                                    input_count: lc,
                                    inputs_offset: lo,
                                    compute_fn: CF_DEREF_READ,
                                });
                                return load;
                            }
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
                        let sp_cell = ctx.sp_cell;

                        // B3: read the current sp through its Cell. The pop store
                        // (body sg, iteration start) has completed before this
                        // state sg runs, so the Cell holds the post-pop value.
                        let sp_load_off = self.graph.inputs_pool.push(&[sp_cell]);
                        let sp_load = self.graph.add_node(Node {
                            kind: NodeKind::UnOp,
                            input_count: 1,
                            inputs_offset: sp_load_off,
                            compute_fn: CF_DEREF_READ,
                        });

                        // Compute stack indices: base_cont = sp * stride, base_task = (sp + 1) * stride
                        // sp has already been decremented by pop (sp = original_sp - 1)
                        // cont writes to the slot freed by pop (overwriting the consumed frame); task writes to the next slot
                        // sp_new = sp + 2; on pop, sp-1 reads task first (LIFO), then cont
                        let one_const = self.make_i32_const(1);
                        let sp_plus_1 = self.make_binop(sp_load, one_const, CF_ADD_I32);
                        let two_const = self.make_i32_const(2);
                        let sp_plus_2 = self.make_binop(sp_load, two_const, CF_ADD_I32);
                        let stride_val = self.make_i32_const(stride as i32);
                        let base_cont = self.make_binop(sp_load, stride_val, CF_MUL_I32);
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
                        // For state S: slot j = saved_nodes[j] (j < S-1), result-load (j == S-1), 0 (j >= S)
                        // The result-load reads the result Cell HERE (this state sg):
                        // the value was stored by the previous state's completion.
                        let zero_saved = self.make_i32_const(0);
                        let mut result_load: Option<NodeId> = None;
                        for j in 0..max_saved {
                            let offset = self.make_i32_const((param_count + 1 + j) as i32);
                            let idx = self.make_binop(base_cont, offset, CF_ADD_I32);
                            let val = if j < current_state {
                                if j + 1 < current_state {
                                    ctx.saved_nodes[j]
                                } else {
                                    // j == current_state - 1: the most recent call's result
                                    let load = match result_load {
                                        Some(l) => l,
                                        None => {
                                            let (lc, lo) = match self.current_effect {
                                                Some(eff) => (2, self.graph.inputs_pool.push(&[ctx.result_cell, eff])),
                                                None => (1, self.graph.inputs_pool.push(&[ctx.result_cell])),
                                            };
                                            let l = self.graph.add_node(Node {
                                                kind: NodeKind::UnOp,
                                                input_count: lc,
                                                inputs_offset: lo,
                                                compute_fn: CF_DEREF_READ,
                                            });
                                            result_load = Some(l);
                                            l
                                        }
                                    };
                                    load
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

                        // sp = sp + 2 through the Cell (chained into effect to ensure it runs after all stores)
                        let sp_new = self.chain_effects(self.current_effect, sp_plus_2);
                        let sp_store_off = self.graph.inputs_pool.push(&[sp_cell, sp_new]);
                        let sp_store = self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset: sp_store_off,
                            compute_fn: CF_DEREF_WRITE,
                        });
                        self.current_effect = Some(sp_store);

                        // Create the barrier node (Continue signal; blocks subsequent expression execution)
                        let barrier = self.make_continue_barrier(sp_store);
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
        let mut arg_nodes = Vec::with_capacity(params.len());
        for (param, &arg) in params.iter().zip(args.iter()) {
            let arg_node = self.compile_subexpr(arg);
            arg_nodes.push(arg_node);
            self.bind_var(param.name, arg_node);
        }
        // Inline-body param slotting (mirrors ③ in compile_function_body):
        // an ASSIGNED param's cell stores keep the loop/branch machinery out
        // of the caller's inline copy. Without this, an inlined loop assigning
        // the param compiles WriteBacks against the raw arg node.
        // Branch contexts (if arm, &&-RHS — current_branch_sg set) create the
        // cells too, but chain them onto a FRESH in-branch effect root:
        // chaining onto the incoming (parent) effect would wire a
        // branch-internal SEQ to a non-dominating outer node (⑤ V2 stall),
        // while skipping the chain entirely lets an internal gate launch
        // before the alloc executes — its completion can't reach the nested
        // branch frame's pending table and stores are silently dropped (⑥).
        // The reset is sound: parent effects completed before the branch frame
        // launched, so in-branch ordering is the only constraint.
        if self.fn_all_vars_slot {
            let in_branch = self.current_branch_sg.is_some();
            let mut assigned = rustc_hash::FxHashSet::default();
            collect_assigned_names_deep(&self.current_module().arena, body, &mut assigned);
            for param in params {
                if assigned.contains(param.name) {
                    if let Some(arg_node) = self.lookup_var(param.name) {
                        let off = self.graph.inputs_pool.push(&[arg_node]);
                        let cell_node = self.graph.add_node(Node {
                            kind: NodeKind::UnOp,
                            input_count: 1,
                            inputs_offset: off,
                            compute_fn: CF_CELL_ALLOC,
                        });
                        self.bind_cell(param.name, cell_node);
                        self.track_cell_decl(param.name, cell_node, arg_node);
                        if in_branch {
                            // Root the branch-local effect chain at the alloc:
                            // no SEQ onto the incoming parent effect (⑤ V2
                            // stall class), yet every subsequent in-branch
                            // effect — including internal gate effect chains —
                            // depends on the alloc, so nested-sg stores can
                            // never outrun it (⑥ silent-drop class). Parent
                            // effects completed before this branch frame
                            // launched; in-branch ordering is the only
                            // constraint.
                            self.current_effect = Some(cell_node);
                        } else {
                            self.current_effect =
                                Some(self.chain_effects(self.current_effect, cell_node));
                        }
                    }
                }
            }
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
            // The wrap Gate's branch reads outer nodes (the actuals) via the
            // FRAME SNAPSHOT, not through SSA edges — so the actuals must be
            // COMPUTED before the Gate fires. `compile_subexpr` does not feed
            // current_effect, so chain the arg nodes in explicitly; otherwise
            // the Gate can launch the branch while an arg (e.g. a `&x` RefOf)
            // is still uncomputed and the branch body reads null.
            for arg_node in arg_nodes {
                self.current_effect = Some(self.chain_effects(self.current_effect, arg_node));
            }
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

    /// Arity-exact constructor lookup through sema's ctor table
    /// (`ctor_def_index`: bare ctor name → (type_idx << 16 | ctor_idx)).
    /// Returns the field list of the first constructor whose field count
    /// equals `arity`, keyed by canonical type name.
    pub(super) fn lookup_ctor_info_by_arity(&self, name: &str, arity: usize) -> Option<TypeFieldInfo> {
        let packed_list = self.sema.ctor_def_list(name)?;
        for &packed in packed_list.iter() {
            let type_idx = (packed >> 16) as u16;
            let ci = (packed & 0xffff) as usize;
            let def = self.sema.type_defs.get(&type_idx)?;
            let ctor = def.constructors.get(ci)?;
            if ctor.field_type_reprs.len() == arity {
                let field_names: Vec<String> = ctor
                    .field_names
                    .iter()
                    .map(|n| n.as_deref().unwrap_or("_").to_string())
                    .collect();
                let kind = match def.kind {
                    crate::sema::Sema::TypeDefKind::Adt => RecordLitKind::Adt,
                    crate::sema::Sema::TypeDefKind::Record => RecordLitKind::Record,
                    crate::sema::Sema::TypeDefKind::Newtype => RecordLitKind::Newtype,
                    crate::sema::Sema::TypeDefKind::Alias => RecordLitKind::Record,
                };
                return Some(TypeFieldInfo {
                    field_names,
                    type_name: ctor.type_name.to_string(),
                    kind,
                });
            }
        }
        None
    }

    /// Look up a type declaration's field info (by type name).
    ///
    /// Uniformly searches layer by layer through type_scope_stack (top-level + nested types share the same lookup path).
    pub(super) fn alloc_base_dispatch_idx(&mut self) -> u16 {
        let idx = self.next_base_dispatch_idx;
        self.next_base_dispatch_idx = self.next_base_dispatch_idx.wrapping_add(1).max(0x8000);
        idx
    }

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
        // Module-scoped: the AST type segment resolves to its canonical key.
        let canonical = self.sema.resolve_type_key_in(self.current_module().name, type_name);
        let type_idx = self.sema.type_def_idx(&canonical)?;
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
    /// Whether walking `name`'s FIRST-base links (the layout-prefix chain)
/// reaches `ancestor` (exclusive of `name` itself).
fn first_base_chain_reaches(
    sema: &crate::sema::Sema::SemaResult,
    name: &str,
    ancestor: &str,
) -> bool {
    let mut current: Option<Box<str>> = Some(name.into());
    let mut hops = 0usize;
    while let Some(cn) = current {
        if hops > 64 {
            return false;
        }
        let Some(def) = sema.get_type_def(cn.as_ref()) else { return false };
        let Some(first) = def.bases.first() else { return false };
        if first.as_ref() == ancestor {
            return true;
        }
        current = Some(first.clone());
        hops += 1;
    }
    false
}

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
                    "address_of" if args.len() == 1 => {
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
                            compute_fn: CF_LIB_ADDRESS_OF,
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
                // Qualified-first: Sema classified the receiver as a module
                // namespace, so the qualifier it carries must reach
                // resolve_func — including DOTTED receivers. A bare Ident
                // (`Instant.now()`) yields the short key; a dotted chain
                // (`std.time.Instant.now()`) yields the full module path, so
                // step 3 probes the full mangled key instead of discarding the
                // qualifier and falling through to the package/bare families,
                // which trip on cross-module duplicate names (`now` exists in
                // both SystemTime and Instant).
                //
                // Sema's import-resolved module path (module_func_call_targets)
                // is AUTHORITATIVE: bind DIRECTLY by the full mangled key and
                // bypass the string-key families entirely. The short-key
                // tripwire records HISTORICAL collisions (user `Parse.parse`
                // vs `std.json.Parse.parse`) and would veto the call as
                // "ambiguous" even though sema already adjudicated the target
                // through the import binding.
                let sema_target = self
                    .sema
                    .module_func_call_targets
                    .get(&recv_key)
                    .cloned();
                let authoritative = sema_target.as_ref().and_then(|mp| {
                    // Generic calls prefer the sema-chosen monomorphization.
                    if let Some(m) = mangled.as_deref() {
                        if let Some(&sg) = self.func_subgraphs.get(m) {
                            return Some((mp.clone(), m.to_string(), sg));
                        }
                    }
                    let full = format!("{}.{}", mp, method);
                    self.func_subgraphs
                        .get(full.as_str())
                        .map(|&sg| (mp.clone(), full, sg))
                });
                let recv_qualifier = sema_target
                    .map(|p| p.to_string())
                    .or_else(|| self.dotted_qualifier_of(recv));
                // @internal guard must cover the authoritative fast path too —
                // resolve_func's entry check never runs when we bind directly,
                // and the qualified form (Recv.__helper) must stay closed to
                // user code either way (negative: internal_std_helper_qualified).
                if authoritative.is_some() && self.internal_access_blocked(method) {
                    self.errors.push(self.internal_access_diag(method));
                }
                let resolved: Option<Result<SubGraphId, String>> = match authoritative {
                    Some((mp, key, sg)) => {
                        self.log_call_bind("path0_sema_target", method, Some(&mp), &key, sg);
                        Some(Ok(sg))
                    }
                    None => {
                        self.resolve_func("path0_module_recv", method, mangled.as_deref(), recv_qualifier.as_deref())
                    }
                };
                match resolved {
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
                        // Module-qualified type-constructor call
                        // (`std.time.DateTime.DateTime(...)`): the trailing
                        // segment names a type / ADT constructor — lower to
                        // the same record-construct node the bare
                        // `DateTime(...)` / short `DateTime.DateTime(...)`
                        // forms use. Depth-general: only the final segment
                        // matters; the qualifier chain just selects the
                        // module (already validated by sema).
                        // S2 ID path first: sema (MethodCall Path 0a)
                        // adjudicated the qualified ctor — consume the
                        // canonical resolution so the bare-name ctor tables
                        // cannot mis-pick under cross-module same-name types.
                        let tf_info = self
                            .ctor_tf_info_from_resolution(call_expr_id, method, args.len())
                            .or_else(|| self.lookup_type_field_names(method))
                            .or_else(|| self.lookup_constructor_field_names(method));
                        if let Some(info) = tf_info {
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
                                type_name: info.type_name.clone(),
                                field_names: info.field_names.into_iter().map(Some).collect(),
                                constructor: method.to_string(),
                                kind: info.kind,
                            });
                            return node;
                        }
                        self.errors.push(format!(
                            "module function call '{}.{}' did not resolve to any target subgraph",
                            recv_qualifier.as_deref().unwrap_or("<?>"), method
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

            if std::env::var("FROND_TRACE_CTOR").is_ok() {
                let k = crate::sema::Sema::module_expr_key(self.expr_key_module(), recv.0 as u64);
                let tn = self.sema.expr_types.get(&k).map(|i| i.type_name.clone());
                let tid = self.expr_type_id(recv);
                eprintln!("[dispatch-recv] method={} type_name={:?} type_id={:?} recv_expr={} cur_mod={}", method, tn, tid, recv.0, self.current_module().name);
            }
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

            // Path 2: type's own method / trait method override.
            // S4: sema recorded the resolved (type_def_idx, method_idx) at
            // check time — consume it FIRST. The name/type-id lookups below
            // are the measured fallback (FROND_TRACE_S2).
            let dkey = crate::sema::Sema::module_expr_key(
                self.expr_key_module(),
                call_expr_id.0 as u64,
            );
            let s4_target = self.sema.dispatch_targets.get(&dkey).copied();
            let s4_tyname_tid = s4_target.and_then(|(tidx, _)| {
                self.sema
                    .type_defs
                    .get(&tidx)
                    .map(|d| (d.name.to_string(), crate::types::dynamic_type_id(tidx)))
            });
            let recv_type: Option<(String, u16)> = if let Some((tn, tid)) = s4_tyname_tid {
                Some((tn, tid))
            } else if recv_node_override.is_some() {
                if std::env::var("FROND_TRACE_S2").is_ok() {
                    eprintln!(
                        "[s2-fallback:dispatch] method={} mod={}",
                        method,
                        self.current_module().name
                    );
                }
                self.current_method_type.as_ref().map(|(n, id)| (n.to_string(), *id))
            } else {
                if std::env::var("FROND_TRACE_S2").is_ok() {
                    eprintln!(
                        "[s2-fallback:dispatch] method={} mod={}",
                        method,
                        self.current_module().name
                    );
                }
                self.expr_type_name(recv).map(|n| n.to_string()).zip(self.expr_type_id(recv))
            };
            if let Some((type_name, type_id)) = recv_type {
                let method_idx = match s4_target.map(|(_, midx)| midx) {
                    Some(midx) => Some(midx),
                    None => self
                        .sema
                        .lookup_method_idx_by_type_id(type_id, method)
                        .or_else(|| self.sema.lookup_method_idx(type_name.as_str(), method)),
                };
                if std::env::var("FROND_TRACE_GET").is_ok() && method == "get" {
                    let chosen_sg = method_idx
                        .and_then(|idx| self.method_subgraphs.get(&(type_id, idx)).copied());
                    let sg_name = chosen_sg.map(|sg| {
                        let fid = self.graph.subgraphs[sg.0 as usize].function_id;
                        self.graph.sg_names.get(fid as usize).cloned().unwrap_or_default()
                    });
                    eprintln!(
                        "[get-dispatch] recv_ty={} tid={} s4={:?} midx={:?} sg={:?} sg_fn={:?} call_expr={} mod={}",
                        type_name, type_id, s4_target, method_idx, chosen_sg, sg_name, call_expr_id.0, self.current_module().name
                    );
                }
                if let Some(method_idx) = method_idx {
                    if let Some(&target_sg) = self.method_subgraphs.get(&(type_id, method_idx)) {
                        // Inheritance dynamic dispatch: when any type's
                        // FIRST-base chain reaches this receiver type, a child
                        // value may flow in here. Inherited methods are
                        // compiled per child (late binding), so the correct
                        // subgraph depends on the value's ACTUAL type — emit a
                        // vtable-style dispatch keyed by (site_idx, type_name)
                        // covering the base and every chain descendant.
                        let has_descendants = self.sema.type_defs.values().any(|d| {
                            !d.bases.is_empty()
                                && Self::first_base_chain_reaches(self.sema, d.name.as_ref(), type_name.as_str())
                        });
                        if has_descendants {
                            let site_idx = self.alloc_base_dispatch_idx();
                            self.graph.set_vtable_call(call_node, site_idx);
                            self.graph
                                .vtable_fallback_dispatch
                                .insert((site_idx, type_name.clone().into()), target_sg);
                            let descendants: Vec<(Box<str>, u16, u16)> = self
                                .sema
                                .type_defs
                                .iter()
                                .filter(|(_, d)| {
                                    !d.bases.is_empty()
                                        && Self::first_base_chain_reaches(self.sema, d.name.as_ref(), type_name.as_str())
                                })
                                .filter_map(|(&idx, d)| {
                                    let did = crate::types::dynamic_type_id(idx);
                                    self.sema
                                        .lookup_method_idx(d.name.as_ref(), method)
                                        .map(|mi| (d.name.clone(), did, mi))
                                })
                                .collect();
                            for (dname, did, mi) in descendants {
                                if let Some(&dsg) = self.method_subgraphs.get(&(did, mi)) {
                                    self.graph
                                        .vtable_fallback_dispatch
                                        .insert((site_idx, dname.into()), dsg);
                                }
                            }
                            return call_node;
                        }
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
                // Collect every implementing trait that provides a specialized
                // default for this method, then select deterministically: a
                // trait that is a DESCENDANT of all other providers wins
                // (trait inheritance — the child's declaration shadows its
                // parents'); plain multi-trait conflicts never reach here
                // (sema's R3 rejects them at check time).
                let mut candidates: Vec<(Box<str>, SubGraphId)> = Vec::new();
                for trait_def in self.sema.trait_defs.values() {
                    if !self.type_implements_trait(type_id, &trait_def.name) {
                        continue;
                    }
                    if let Some(method_idx) = trait_def
                        .methods
                        .iter()
                        .position(|m| m.name.as_ref() == method && m.has_body)
                    {
                        if let Some(trait_idx) = self.sema.trait_def_idx(trait_def.name.as_ref()) {
                            if let Some(&target_sg) = self.trait_default_subgraphs.get(&(type_id, trait_idx, method_idx as u16)) {
                                candidates.push((trait_def.name.clone(), target_sg));
                            }
                        }
                    }
                }
                let chosen: Option<SubGraphId> = if candidates.len() == 1 {
                    candidates.first().map(|(_, sg)| *sg)
                } else if candidates.len() > 1 {
                    candidates
                        .iter()
                        .find(|(c, _)| {
                            candidates.iter().all(|(o, _)| {
                                o.as_ref() == c.as_ref()
                                    || self
                                        .sema
                                        .trait_parent_closure(c.as_ref())
                                        .iter()
                                        .any(|a| a.as_ref() == o.as_ref())
                            })
                        })
                        .map(|(_, sg)| *sg)
                } else {
                    None
                };
                if let Some(target_sg) = chosen {
                    self.graph.set_call_target(call_node, target_sg);
                    return call_node;
                }
            }

            // Path 4: free-function method call (recv.method(args) -> method(recv, args))
            // When the method name matches a top-level free function, recv is passed as the first argument.
            // Single-point resolution: builtin/user free functions resolve bare; std free
            // functions (no bare slot) resolve only through the earlier qualified paths.
            // Generic callees bind the sema-chosen monomorphization via
            // call_instantiations (same lookup as Path 0): without it the call
            // lands on the UNSPECIALIZED body, whose soft-typed comparisons
            // work for i32 by accident and silently misorder f64/str elements.
            let call_inst_key = crate::sema::Sema::module_expr_key(
                self.expr_key_module(),
                call_expr_id.0 as u64,
            );
            let inst_mangled = self
                .sema
                .call_instantiations
                .get(&call_inst_key)
                .map(|&id| format!("{}#{}", method, id));
            if let Some(Ok(target_sg)) = self.resolve_func("path4_free_fn", method, inst_mangled.as_deref(), None) {
                self.graph.set_call_target(call_node, target_sg);
                return call_node;
            }

            // No dispatch path resolved: report at compile time instead of
            // leaving a target-less Call node that panics the engine at
            // runtime ("no call_target ... broken compiler invariant"). The
            // usual cause is a receiver whose inferred type degraded to
            // Unknown/TypeVar (e.g. a match whose arms failed to join).
            {
                let recv_desc = self
                    .expr_type_name(recv)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown type".to_string());
                let span = self.current_module().arena.expr(call_expr_id).span;
                let recv_node_dbg = format!("{:?}", &self.current_module().arena.expr(recv).node);
                self.errors.push(format!(
                    "cannot dispatch method '{}' on receiver of {} at line {}, column {} [recv_expr={} mod={} key_mod={} inst={:?} recv_node={}]",
                    method, recv_desc, span.line, span.column, recv.0, self.current_module().name,
                    self.expr_key_module(), self.current_instance_id, recv_node_dbg
                ));
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
        // Base-type super (inheritance): a sema-recorded base dispatch wins
        // over the trait-default layer — target is the BASE's own method
        // subgraph, receiver is the enclosing `this` (child values are
        // layout-prefix compatible with the base's field ids).
        if let Some((base_name, base_method)) = self.sema.super_base_dispatches.get(&key) {
            let base_tid = self.sema.type_def_idx(base_name.as_ref())
                .map(|idx| crate::types::dynamic_type_id(idx));
            let base_mi = self.sema.lookup_method_idx_in(
                self.current_module().name,
                base_name.as_ref(),
                base_method.as_ref(),
            );
            if let (Some(tid), Some(mi)) = (base_tid, base_mi) {
                if let Some(&sg) = self.method_subgraphs.get(&(tid, mi)) {
                    let recv_node = match self.lookup_var("this") {
                        Some(n) => n,
                        None => {
                            self.errors.push(
                                "super call: receiver 'this' not in scope (base super is only                                  supported in the direct body of a type method)"
                                    .into(),
                            );
                            let off = self.graph.inputs_pool.push(&[]);
                            return self.graph.add_node(Node {
                                kind: NodeKind::Const,
                                input_count: 0,
                                inputs_offset: off,
                                compute_fn: CF_NOOP,
                            });
                        }
                    };
                    let mut inputs = vec![recv_node];
                    for &a in args {
                        inputs.push(self.compile_subexpr(a));
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
                    self.graph.set_call_target(call_node, sg);
                    return call_node;
                }
            }
            self.errors.push(format!(
                "super call '{}.{}' could not resolve a base method subgraph",
                base_name, base_method
            ));
        }
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
    /// Flatten a pure `a.b.c` identifier/field-access chain into its dotted
    /// text ("a.b.c"); None when the chain root is not an identifier.
    /// Depth-general: the whole receiver chain becomes the qualifier, which
    /// for a module-qualified receiver (`std.time.Instant`) equals the module
    /// logical path — exactly the full mangled key func_subgraphs registers
    /// (`std.time.Instant.now`).
    fn dotted_qualifier_of(&self, mut expr: crate::ast::Ast::ExprId) -> Option<String> {
        let mut parts: Vec<&'a str> = Vec::new();
        loop {
            match &self.current_module().arena.expr(expr).node {
                crate::ast::Ast::Expr::FieldAccess { recv, field, .. } => {
                    parts.push(field);
                    expr = *recv;
                }
                crate::ast::Ast::Expr::Ident(n) => {
                    parts.push(n);
                    parts.reverse();
                    return Some(parts.join("."));
                }
                _ => return None,
            }
        }
    }

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
                .and_then(|tn| self.sema.lookup_method_idx_in(self.current_module().name, tn, method))
                .is_some();
            if !shadows {
                return Some(kind);
            }
        }
        // `T?.is_null()`: nullable is a type constructor with no dispatch table,
        // so the (type_id, method_idx) path below cannot see it — lower directly
        // to CF_IS_NULL (34). Sema's lookup_method_type special-cases the same
        // call for typing; without this the call built a no-target Call node
        // and panicked the engine at runtime.
        if method == "is_null" {
            if let Some(h) = self.expr_type_handle(recv) {
                let r = self.type_arena.resolve(h);
                if matches!(self.type_arena.get(r), crate::types::Type::Nullable(_)) {
                    return Some(crate::sema::Sema::IntrinsicKind::UnOp(34));
                }
            }
        }
        let type_name = self.expr_type_name(recv)?;
        let type_id = self.expr_type_id(recv)?;
        let method_idx = self.sema.lookup_method_idx_in(self.current_module().name, type_name, method)?;
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
        // First try the witness table (explicit declarations); a type whose
        // witness method_slots MISS the name (the method is a TRAIT DEFAULT —
        // slots only carry the type's own methods) resolves to the
        // specialized trait-default subgraph; the structural scan remains
        // the last resort for undeclared implementations.
        let trait_idx_and_mpos: Option<(u16, u16)> = self
            .sema
            .trait_def_idx(trait_name)
            .zip(
                self.sema
                    .get_trait_def(trait_name)
                    .and_then(|td| {
                        td.methods
                            .iter()
                            .position(|m| m.name.as_ref() == method_name)
                            .map(|p| p as u16)
                    }),
            );
        let mut entries: Vec<(u16, u16)> = Vec::new();
        let mut resolved_any = false;
        for entry in self.sema.witness_table.entries() {
            if entry.trait_name.as_ref() != trait_name {
                continue;
            }
            if let Some(type_method_idx) = self.sema.witness_table.resolve_method(trait_name, entry.type_id, method_name) {
                entries.push((entry.type_id, type_method_idx));
                resolved_any = true;
            } else if let Some((trait_idx, trait_mpos)) = trait_idx_and_mpos {
                // Trait-default method: the specialized subgraph compiled for
                // this (type, trait, method) triple.
                if let Some(&sg) = self
                    .trait_default_subgraphs
                    .get(&(entry.type_id, trait_idx, trait_mpos))
                {
                    if let Some(name) = self.type_name_from_id(entry.type_id) {
                        self.graph
                            .vtable_fallback_dispatch
                            .insert((vtable_idx, name.into_boxed_str()), sg);
                        resolved_any = true;
                    }
                }
            }
        }
        if resolved_any {
            for (type_id, type_method_idx) in entries {
                if let Some(&sg) = self.method_subgraphs.get(&(type_id, type_method_idx)) {
                    if let Some(name) = self.type_name_from_id(type_id) {
                        self.graph.vtable_fallback_dispatch.insert((vtable_idx, name.into_boxed_str()), sg);
                    }
                }
            }
            return;
        }
        // Structural fallback: if witness table has no entries for this trait, scan all types
        // that have a method with the matching name. This supports `type Dog { fun name(): str }`
        // being passed as `Animal` without an explicit `: Animal` declaration.
        for (&type_idx, type_def) in &self.sema.type_defs {
            for (m_idx, m) in type_def.methods.iter().enumerate() {
                if m.name.as_ref() == method_name {
                    let tid = crate::types::dynamic_type_id(type_idx);
                    if let Some(&sg) = self.method_subgraphs.get(&(tid, m_idx as u16)) {
                        if let Some(name) = self.type_name_from_id(tid) {
                            self.graph.vtable_fallback_dispatch.insert((vtable_idx, name.into_boxed_str()), sg);
                        }
                    }
                    break;
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
                            .type_def_idx(inst.type_name.as_ref())
                            .map(|idx| crate::types::dynamic_type_id(idx));
                    }
                }
            }
        }
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr.0 as u64);
        let info = self.sema.expr_types.get(&key)?;
        // Consistent with expr_type_name: prefer type_name, fall back to Type::source_name()
        // (built-in structural variants like array/nullable/str/Throw return their registered name via source_name();
        // "unknown" only appears in degenerate paths where Adt/Record arena lookup fails).
        let type_name = info
            .type_name
            .as_deref()
            .unwrap_or_else(|| self.type_arena.get(info.ty).source_name());
        // Concrete array names ("u8[]") address the synthetic builtin "array"
        // type def. Bare legacy names resolve module-scoped.
        let key = crate::sema::Sema::SemaResult::canonical_type_name(type_name);
        let type_idx = match self.sema.type_def_idx(key) {
            Some(idx) => idx,
            None => {
                let resolved = self.sema.resolve_type_key_in(self.current_module().name, key);
                if resolved == key {
                    return None;
                }
                match self.sema.type_def_idx(&resolved) {
                    Some(idx) => idx,
                    None => return None,
                }
            }
        };
        Some(crate::types::dynamic_type_id(type_idx))
    }

}
