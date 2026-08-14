//! Function — Function & method compilation: bodies, memoize, monomorph, methods. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Look up the memo strategy for the current function/method (generic entry point, unified
    /// `FuncId` query).
    ///
    /// Obtains the `FuncId` by looking up the mangled name in `CallGraph.name_to_func`:
    /// - FunDecl / lambda / monomorph instance: `self_type = None`, mangled = name.
    /// - Method: `self_type = Some(type_name)`, mangled = `"{type_name}.{name}"`.
    ///
    /// The mangled-name format matches how methods are registered in `build_call_graph`
    /// (`"Type.method"`).
    /// `memo_pass` already makes the unique decision; a function has at most one strategy.
    pub(super) fn lookup_memo_strategy(
        &self,
        name: &str,
        self_type: Option<&str>,
    ) -> Option<crate::pass::Analyzer::MemoStrategy> {
        let report = self.current_analysis()?;
        let mangled: String = match self_type {
            Some(t) => format!("{}.{}", t, name),
            None => name.to_string(),
        };
        let func_id = *report.call_graph.name_to_func.get(&mangled)?;
        report.memo.candidates.iter()
            .find(|c| c.func == func_id)
            .map(|c| c.strategy.clone())
    }

    /// Unified function-body compilation entry point: queries the memo strategy and dispatches
    /// to the corresponding compilation path.
    ///
    /// All function compilation entry points (`compile_function` /
    /// `compile_monomorph_instance` / `compile_builtin_method` / `compile_user_method` /
    /// `compile_lambda`) call this method, ensuring memoize / tail_rec / non_tail_rec
    /// optimizations apply uniformly to FunDecls, methods, lambdas, and monomorph instances.
    ///
    /// `self_type`: pass `Some(type_name)` for methods, `None` otherwise. Used to construct the
    /// mangled name for the `FuncId` lookup.
    /// Precondition: the caller has set `current_sg_start = node_start` (`compile_memoize`
    /// relies on this value to compute parameter node ids = `current_sg_start + param_index`).
    ///
    /// `is_async`: whether the enclosing function is declared `async`. When true and the body
    /// expression's inferred type is `Async<T>` (Bug #79), an implicit await node is inserted so
    /// the function returns the resolved `T` rather than the raw async handle. This implements
    /// transparent async forwarding — `async fun f(): Async<T> { g() }` where `g(): Async<T>`
    /// automatically awaits `g()` and returns its result.
    pub(super) fn compile_function_body(
        &mut self,
        name: &str,
        self_type: Option<&str>,
        body_expr: crate::ast::Ast::ExprId,
        params: &[crate::ast::Ast::Param<'_>],
        is_void_fn: bool,
        is_async: bool,
    ) -> NodeId {
        let prev_tail = self.in_tail_position;
        self.in_tail_position = !is_void_fn;
        // Bug #66: Mark that the next compile_block call is the function body's top-level block.
        // compile_block reads and resets this flag so that only nested blocks extract
        // block-scoped defers; function-level defers stay in defer_table for function-exit execution.
        let prev_top_block = self.in_function_top_block;
        self.in_function_top_block = true;
        // Unified memo-strategy query (memo_pass already makes the unique decision; mutually
        // exclusive).
        let strategy = self.lookup_memo_strategy(name, self_type);
        let r = match strategy {
            Some(crate::pass::Analyzer::MemoStrategy::TailRecToLoop { info }) => {
                self.compile_tail_rec_to_loop(name, body_expr, params, &info)
            }
            Some(crate::pass::Analyzer::MemoStrategy::NonTailRecToLoop { info }) => {
                self.compile_non_tail_rec_to_loop(name, body_expr, params, &info)
            }
            Some(crate::pass::Analyzer::MemoStrategy::Memoize { cache_key, .. }) => {
                self.compile_memoize(name, body_expr, params, &cache_key)
            }
            _ => {
                let node = self.compile_expr(body_expr);
                // Bug #79: Auto-await forwarding. If this is an async function and the body
                // expression's inferred type is Async<T>, the body produces an async handle (a
                // raw i32 async_id) rather than the resolved value T. Insert an implicit await
                // node to resolve it, so the function returns T (which the runtime wraps back
                // into Async<T> for the caller). Sema already validates type compatibility
                // (unify_return_type handles Async<X> vs Async<Y> by unifying inner types).
                if is_async && self.expr_type_is_async(body_expr) {
                    self.build_await_node(body_expr, node)
                } else {
                    node
                }
            }
        };
        self.in_tail_position = prev_tail;
        self.in_function_top_block = prev_top_block;
        r
    }

    /// Memoization cache: consumes the Memoize strategy and inserts a cache-check Gate at the
    /// function entry and a cache-write after the body.
    ///
    /// Called by `compile_function` when it detects `MemoStrategy::Memoize`.
    /// The parameter nodes have already been created and `bind_var`'d by `compile_function`;
    /// this method constructs the cache structure.
    ///
    /// Structure:
    /// - `memo_check` node: inputs = the parameter nodes; returns `record(hit, value)`.
    /// - `field_get(hit)` -> `Gate(hit)` for branching.
    /// - `hit=true` branch: `field_get(value)` as the return value (passthrough subgraph).
    /// - `hit=false` branch: compile the function body normally + `memo_store(args, body_result)`.
    ///
    /// Recursive calls remain ordinary Calls (on cache hit they return directly without
    /// expanding).
    pub(super) fn compile_memoize(
        &mut self,
        _name: &str,
        body_expr: crate::ast::Ast::ExprId,
        _params: &[crate::ast::Ast::Param<'_>],
        cache_key: &crate::pass::Analyzer::CacheKeySpec,
    ) -> NodeId {
        // Allocate a cache-table index.
        let table_index = self.memo_table_count;
        self.memo_table_count += 1;

        // Collect the parameter nodes that participate in the cache key (by
        // `cache_key.param_indices`).
        // The parameter nodes are the first `param_count` nodes of the subgraph
        // (`current_sg_start` is the start of the function subgraph).
        let param_nodes: Vec<NodeId> = cache_key.param_indices.iter()
            .map(|&idx| {
                let node_id = self.current_sg_start + idx;
                NodeId(node_id)
            })
            .collect();
        let memo_param_count = param_nodes.len() as u8;

        // 1. Create the memo_check node: inputs = the parameter nodes.
        let check_inputs = self.graph.inputs_pool.push(&param_nodes);
        let memo_check_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: param_nodes.len() as u8,
            inputs_offset: check_inputs,
            compute_fn: CF_MEMO_CHECK,
        });
        self.graph.set_memo_info(memo_check_node, crate::ir::Ir::MemoInfo {
            table_index,
            param_count: memo_param_count,
        });

        // 2. Extract the `hit` field from the record returned by `memo_check` (used as the Gate
        //    condition).
        let hit_inputs = self.graph.inputs_pool.push(&[memo_check_node]);
        let hit_node = self.graph.add_node(Node {
            kind: NodeKind::FieldAccess,
            input_count: 1,
            inputs_offset: hit_inputs,
            compute_fn: CF_RECORD_FIELD_GET,
        });
        self.graph.set_field_set_name(hit_node, "hit".to_string());

        // 3. hit=true branch subgraph: extract the `value` field from the record (cache hit:
        //    return the cached value directly).
        //    Uses the `compile_branch_subgraph` pattern: an independent subgraph + frame-chain
        //    passthrough to access `memo_check_node`.
        let hit_sg = {
            let node_start = self.graph.nodes.len() as u32;
            self.enter_scope();
            let prev_sg_start = self.current_sg_start;
            self.current_sg_start = node_start;
            let v_inputs = self.graph.inputs_pool.push(&[memo_check_node]);
            let value_node = self.graph.add_node(Node {
                kind: NodeKind::FieldAccess,
                input_count: 1,
                inputs_offset: v_inputs,
                compute_fn: CF_RECORD_FIELD_GET,
            });
            self.graph.set_field_set_name(value_node, "value".to_string());
            self.current_sg_start = prev_sg_start;
            self.exit_scope();
            let node_end = self.graph.nodes.len() as u32;
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(crate::ir::Ir::SubGraph {
                id: sg_id,
                node_range: (NodeId(node_start), NodeId(node_end)),
                param_count: 0,
                entry_node: NodeId(node_start),
                return_node: value_node,
                has_suspend: false,
                event_source_decls: Vec::new(),
                defer_table: Vec::new(),
                loop_kind: crate::ir::Ir::LoopKind::None,
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

        // 4. hit=false branch subgraph: compile the function body normally + memo_store (cache
        //    miss: compute and write to the cache).
        //    Uses the `compile_branch_subgraph` pattern: an independent subgraph + frame-chain
        //    passthrough to access parameters and recursive calls.
        let miss_sg = {
            let node_start = self.graph.nodes.len() as u32;
            self.enter_scope();
            let prev_sg_start = self.current_sg_start;
            self.current_sg_start = node_start;
            let prev_effect = self.current_effect;
            self.current_effect = None;
            // Recursive Calls inside miss_sg are NOT tagged as tail_calls: `switch_subgraph`
            // frame reuse for a tail_call would skip the Memoize Gate structure, losing the
            // return value (the recursive Call reuses the miss_sg frame to execute the callee
            // subgraph, causing a `value_table` index mismatch -> returns null).
            // Force non-tail position so recursive Calls go through the normal Call path and
            // create a new frame, correctly returning the result.
            let prev_tail = self.in_tail_position;
            self.in_tail_position = false;
            let body_node = self.compile_expr(body_expr);
            self.in_tail_position = prev_tail;
            // memo_store: inputs = the parameter nodes + body_node.
            let mut store_inputs = param_nodes.clone();
            store_inputs.push(body_node);
            let store_off = self.graph.inputs_pool.push(&store_inputs);
            let store_node = self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: store_inputs.len() as u8,
                inputs_offset: store_off,
                compute_fn: CF_MEMO_STORE,
            });
            self.graph.set_memo_info(store_node, crate::ir::Ir::MemoInfo {
                table_index,
                param_count: memo_param_count,
            });
            self.current_effect = prev_effect;
            self.current_sg_start = prev_sg_start;
            self.exit_scope();
            let node_end = self.graph.nodes.len() as u32;
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(crate::ir::Ir::SubGraph {
                id: sg_id,
                node_range: (NodeId(node_start), NodeId(node_end)),
                param_count: 0,
                entry_node: NodeId(node_start),
                return_node: store_node,
                has_suspend: false,
                event_source_decls: Vec::new(),
                defer_table: Vec::new(),
                loop_kind: crate::ir::Ir::LoopKind::None,
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

        // 5. Gate node: hit true -> hit_sg (return the cached value), false -> miss_sg (compute
        //    + write to the cache).
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let gate_off = self.graph.inputs_pool.push(&[hit_node]);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: 1,
            inputs_offset: gate_off,
            compute_fn: CF_GATE_LAUNCH,
        });
        self.graph.set_gate_branches(
            gate_node,
            crate::ir::Ir::GateBranches {
                condition_input: hit_node,
                branches: vec![
                    (true, hit_sg, vec![]),
                    (false, miss_sg, vec![]),
                ],
            },
        );
        self.current_effect = prev_effect;
        gate_node
    }

    /// Look up a function's location across the user module and builtin modules.
    /// Returns None = user module, Some(i) = builtin_modules[i].
    pub(super) fn find_function_location(&self, name: &str) -> Option<Option<usize>> {
        if self.module.find_function(name).is_some() {
            return Some(None);
        }
        for (i, builtin_mod) in self.builtin_modules.iter().enumerate() {
            if builtin_mod.find_function(name).is_some() {
                return Some(Some(i));
            }
        }
        None
    }

    /// Compile a function into a subgraph (supports cross-module: user module + builtin modules).
    ///
    /// If the function does not exist or its declaration type mismatches, records a compile error and returns a placeholder subgraph (error recovery).
    pub fn compile_function(&mut self, name: &str) -> SubGraphId {
        let location = match self.find_function_location(name) {
            Some(loc) => loc,
            None => {
                self.errors.push(format!("function {} not found", name));
                return self.register_subgraph_placeholder(name, 0, false);
            }
        };

        let module = match location {
            None => self.module,
            Some(i) => self.builtin_modules[i],
        };

        // Set the current compiling module (compile_expr accesses the AST arena via current_module())
        let prev_builtin = self.compiling_builtin;
        self.compiling_builtin = match location {
            None => None,
            Some(i) => Some(self.builtin_modules[i]),
        };

        let (body_expr, is_async, params, is_entry, return_type) = match module.find_function(name) {
            Some(d) => match &d.node {
                crate::ast::Ast::Decl::FunDecl {
                    body,
                    is_async,
                    params,
                    is_entry,
                    return_type,
                    ..
                } => (*body, *is_async, params.clone(), *is_entry, *return_type),
                _ => {
                    self.errors.push(format!("{} is not a function", name));
                    self.compiling_builtin = prev_builtin;
                    return self.register_subgraph_placeholder(name, 0, false);
                }
            },
            None => {
                self.errors.push(format!("function {} not found", name));
                self.compiling_builtin = prev_builtin;
                return self.register_subgraph_placeholder(name, 0, false);
            }
        };
        let param_count = params.len();

        // Reuse the pre-registered sg_id (created by the build() pre-registration pass) to avoid duplicate subgraphs
        let sg_id = if let Some(&existing) = self.func_subgraphs.get(name) {
            existing
        } else {
            let new_id = self.register_subgraph_placeholder(name, param_count as u8, is_async);
            self.func_subgraphs.insert(name.to_string(), new_id);
            new_id
        };
        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // Set current_sg_start = node_start so that sub-functions like compile_memoize can correctly
        // reference parameter nodes (param node id = node_start + param_index)
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        self.enter_scope();
        // Record the parameter scope depth so defer body compilation can truncate
        // body-local rebindings and resolve external vars to parameter nodes.
        let prev_param_scope_depth = self.param_scope_depth;
        self.param_scope_depth = self.scope_stack.len();

        // Create parameter nodes (Const placeholders; values are injected at runtime by start_subgraph)
        // These nodes must be the first param_count nodes of the subgraph
        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Entry function: compile all modules' top-level var/val declaration initializations before the function body
        // Global variables are written to the shared storage via global_store; all functions read via global_load
        // Switch compiling_builtin per module so compile_subexpr accesses the correct AST arena
        if is_entry && !self.top_level_var_decls.is_empty() {
            let decls: Vec<(Option<usize>, crate::ast::Ast::StmtId)> = std::mem::take(&mut self.top_level_var_decls);
            for (mod_idx, stmt_id) in &decls {
                let prev_builtin = self.compiling_builtin;
                self.compiling_builtin = match mod_idx {
                    None => None,
                    Some(i) => Some(self.builtin_modules[*i]),
                };
                let module = self.current_module();
                let stmt = &module.arena.stmt(*stmt_id).node;
                let (name, value_expr) = match stmt {
                    crate::ast::Ast::Stmt::VarDecl { name, value, .. } => (*name, *value),
                    crate::ast::Ast::Stmt::ValDecl { name, value, .. } => (*name, *value),
                    _ => { self.compiling_builtin = prev_builtin; continue; }
                };
                let init_node = self.compile_subexpr(value_expr);
                let slot = self.global_var_slots.get(name).copied()
                    .expect("global var slot must exist after collection");
                let store_node = self.compile_global_store(init_node, slot);
                self.current_effect = Some(store_node);
                self.compiling_builtin = prev_builtin;
            }
            self.top_level_var_decls = decls;
        }

        // Tail-call optimization is enabled only for non-void functions: the trailing expression of a
        // void function is a side effect (e.g. println("done")) and should not be tail-called
        // (switch_subgraph would lose the current frame state).
        // Consumes sema's FuncSigInfo.return_type to determine void (builtin modules fall back to AST).
        let is_void_fn = self.sema.get_func_sig(name)
            .map(|sig| matches!(self.type_arena.get(sig.return_type), crate::sema::Sema::Type::Void))
            .unwrap_or_else(|| match return_type {
                None => true,
                Some(tr) => {
                    matches!(module.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
                }
            });
        // Compute is_async before compile_function_body so it can auto-await Async<T> bodies (Bug #79).
        let fn_is_async = self.sema.get_func_sig(name)
            .map(|sig| sig.is_async)
            .unwrap_or(is_async);
        let return_node = self.compile_function_body(name, None, body_expr, &params, is_void_fn, fn_is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.param_scope_depth = prev_param_scope_depth;
        self.current_function_sg = None;
        self.compiling_builtin = prev_builtin;

        let node_end = self.graph.nodes.len() as u32;
        let debug_mod_name = if std::env::var("KUZO_DEBUG_BUILD").is_ok() {
            Some(self.current_module().name.to_string())
        } else {
            None
        };
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        if let Some(ref mod_name) = debug_mod_name {
            eprintln!("[COMPILE-FN] name={:?} sg_id={} nodes=[{},{}) param_count={} mod={:?}",
                name, sg_id.0, node_start, node_end, param_count, mod_name);
        }
        // Consumes sema's FuncSigInfo.is_async (builtin modules fall back to AST is_async)
        sg.has_suspend = fn_is_async;
        sg.function_id = sg_id.0;

        self.func_subgraphs.insert(name.to_string(), sg_id);
        sg_id
    }

    /// Compile a monomorphization instance into a specialized subgraph.
    ///
    /// Differences from `compile_function`:
    /// - Registers under a mangled name (`func_name#instance_id`) in func_subgraphs to avoid clashing with the non-generic version
    /// - Sets `current_type_args` (type param name -> TypeHandle) for cast/expr_type_name queries
    /// - Clears `current_type_args` after compilation
    ///
    /// Called only for generic instances with type_args (non-generic instances are handled by compile_function).
    pub(super) fn compile_monomorph_instance(&mut self, instance: &crate::sema::Sema::MonomorphInstance) {
        let func_name = instance.func_name.as_ref();

        // Look up the function declaration location (user module or builtin module)
        let location = match self.find_function_location(func_name) {
            Some(loc) => loc,
            None => {
                self.errors.push(format!("monomorph instance function {} not found", func_name));
                return;
            }
        };

        let module = match location {
            None => self.module,
            Some(i) => self.builtin_modules[i],
        };

        let prev_builtin = self.compiling_builtin;
        self.compiling_builtin = match location {
            None => None,
            Some(i) => Some(self.builtin_modules[i]),
        };

        let (body_expr, is_async, params, return_type) = match module.find_function(func_name) {
            Some(d) => match &d.node {
                crate::ast::Ast::Decl::FunDecl {
                    body, is_async, params, return_type, ..
                } => (*body, *is_async, params.clone(), *return_type),
                _ => {
                    self.errors.push(format!("{} is not a function", func_name));
                    self.compiling_builtin = prev_builtin;
                    return;
                }
            },
            None => {
                self.errors.push(format!("monomorph instance function {} not found", func_name));
                self.compiling_builtin = prev_builtin;
                return;
            }
        };
        let param_count = params.len();

        // Build the type parameter map: type_params name -> type_args TypeHandle
        // type_params come from FuncSigInfo (in the same order as instance.type_args)
        let type_param_names: Vec<String> = self.sema.get_func_sig(func_name)
            .map(|sig| sig.type_params.iter().map(|n| n.to_string()).collect())
            .unwrap_or_default();
        let prev_type_args = std::mem::take(&mut self.current_type_args);
        self.current_type_args = type_param_names.iter().zip(instance.type_args.iter())
            .map(|(name, &h)| (name.clone(), h))
            .collect();
        let prev_instance_id = self.current_instance_id;
        self.current_instance_id = Some(instance.instance_id);

        // Mangled name: func_name#instance_id (consistent with sema's cache_key format func_name#hash)
        let mangled = format!("{}#{}", func_name, instance.instance_id);

        // Pre-register the subgraph (reusing the placeholder mechanism)
        let sg_id = if let Some(&existing) = self.func_subgraphs.get(mangled.as_str()) {
            existing
        } else {
            let new_id = self.register_subgraph_placeholder(&mangled, param_count as u8, is_async);
            self.func_subgraphs.insert(mangled.clone(), new_id);
            new_id
        };
        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // Set current_sg_start = node_start so compile_memoize in compile_function_body
        // can correctly reference parameter nodes (id = node_start + param_index)
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        self.enter_scope();
        let prev_param_scope_depth = self.param_scope_depth;
        self.param_scope_depth = self.scope_stack.len();

        // Create parameter nodes (Const placeholders; values are injected at runtime by start_subgraph)
        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Compile the function body (unified entry: memoize/tail_rec/non_tail_rec apply to generic instances too)
        let is_void_fn = self.sema.get_func_sig(func_name)
            .map(|sig| matches!(self.type_arena.get(sig.return_type), crate::sema::Sema::Type::Void))
            .unwrap_or_else(|| match return_type {
                None => true,
                Some(tr) => {
                    matches!(module.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
                }
            });
        // Compute is_async before compile_function_body so it can auto-await Async<T> bodies (Bug #79).
        let fn_is_async = self.sema.get_func_sig(func_name)
            .map(|sig| sig.is_async)
            .unwrap_or(is_async);
        let return_node = self.compile_function_body(func_name, None, body_expr, &params, is_void_fn, fn_is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.param_scope_depth = prev_param_scope_depth;
        self.current_function_sg = None;
        self.compiling_builtin = prev_builtin;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        // Consumes sema's FuncSigInfo.is_async (builtin modules fall back to AST is_async)
        sg.has_suspend = fn_is_async;
        sg.function_id = sg_id.0;

        // Restore the outer type_args context
        self.current_type_args = prev_type_args;
        self.current_instance_id = prev_instance_id;
    }

    /// Compile a TypeDecl method in a builtin module (looked up in method_subgraphs via (type_id, method_idx)).
    pub(super) fn compile_builtin_method(&mut self, type_name: &str, method_idx: usize) {
        // Look up the method data in builtin modules (indexed directly by method_idx)
        let found = self.builtin_modules.iter().enumerate().find_map(|(mod_i, m)| {
            for d in &m.declarations {
                if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                    if *name == type_name {
                        if let Some(method) = methods.get(method_idx) {
                            if method.body.is_some() {
                                return Some((
                                    mod_i,
                                    method.name,
                                    method.body.unwrap(),
                                    method.is_async,
                                    method.params.clone(),
                                    method.return_type,
                                ));
                            }
                        }
                    }
                }
            }
            None
        });

        let (mod_i, method_name, body_expr, is_async, params, return_type) = match found {
            Some(x) => x,
            None => return,
        };

        let m = self.builtin_modules[mod_i];

        // Obtain the pre-registered sg_id from method_subgraphs (created in build() step 0a)
        let type_id = match self.sema.type_def_index.get(type_name) {
            Some(&idx) => crate::types::dynamic_type_id(idx),
            None => return,
        };
        let sg_id = match self.method_subgraphs.get(&(type_id, method_idx as u16)) {
            Some(&sg) => sg,
            None => return,
        };

        let prev = self.compiling_builtin;
        self.compiling_builtin = Some(m);
        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let prev_method_type = self.current_method_type.take();
        self.current_method_type = Some((type_name.into(), type_id));
        self.enter_scope();
        let prev_param_scope_depth = self.param_scope_depth;
        self.param_scope_depth = self.scope_stack.len();

        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Unified entry: memoize/tail_rec/non_tail_rec apply to builtin methods too
        // (self_type = Some(type_name) builds the mangled name "Type.method" to look up FuncId)
        let is_void_fn = match return_type {
            None => true,
            Some(tr) => {
                matches!(m.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
            }
        };
        let return_node = self.compile_function_body(method_name, Some(type_name), body_expr, &params, is_void_fn, is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.param_scope_depth = prev_param_scope_depth;
        self.current_function_sg = None;
        self.current_method_type = prev_method_type;
        self.compiling_builtin = prev;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = sg_id.0;
    }

    /// Compile a TypeDecl method in the user module (looked up in method_subgraphs via (type_id, method_idx)).
    pub(super) fn compile_user_method(&mut self, type_name: &str, method_idx: usize) {
        // Look up the method data in the user module (indexed directly by method_idx).
        // Search top-level declarations first, then local types declared inside function bodies.
        let found = self.module.declarations.iter().find_map(|d| {
            if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &d.node {
                if *name == type_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            return Some((
                                method.name,
                                method.body.unwrap(),
                                method.is_async,
                                method.params.clone(),
                                method.return_type,
                            ));
                        }
                    }
                }
            }
            None
        });
        // Fallback: search local types (declared inside function bodies via Stmt::LocalDecl)
        let found = match found {
            Some(x) => Some(x),
            None => self.find_type_method_full(type_name, method_idx),
        };

        let (method_name, body_expr, is_async, params, return_type) = match found {
            Some(x) => x,
            None => return,
        };

        // Obtain the pre-registered sg_id from method_subgraphs (created in build() step 0a)
        let type_id = match self.sema.type_def_index.get(type_name) {
            Some(&idx) => crate::types::dynamic_type_id(idx),
            None => return,
        };
        let sg_id = match self.method_subgraphs.get(&(type_id, method_idx as u16)) {
            Some(&sg) => sg,
            None => return,
        };

        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let prev_method_type = self.current_method_type.take();
        self.current_method_type = Some((type_name.into(), type_id));
        self.enter_scope();
        self.param_scope_depth = self.scope_stack.len();

        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        // Unified entry: memoize/tail_rec/non_tail_rec apply to user methods too
        // (self_type = Some(type_name) builds the mangled name "Type.method" to look up FuncId)
        let is_void_fn = match return_type {
            None => true,
            Some(tr) => {
                matches!(self.module.arena.ty(tr).node, crate::ast::Ast::TypeNode::Named { name } if crate::value::ValueTag::from_name(name).is_some_and(|t| t.family() == crate::types::TypeFamily::Void))
            }
        };
        let return_node = self.compile_function_body(method_name, Some(type_name), body_expr, &params, is_void_fn, is_async);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_sg_start = prev_sg_start;
        self.current_function_sg = None;
        self.current_method_type = prev_method_type;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = sg_id.0;
    }

    /// Compile the monomorphized specialization of a trait default method (generates a dedicated subgraph for a given impl type).
    ///
    /// A trait default method is the dispatch target when a type does not override it. For each type
    /// implementing the trait, a specialized subgraph is generated so that self in the body has
    /// concrete type information, allowing self.method() calls to statically bind to the correct
    /// method subgraph via path 2 (the type's own methods).
    pub(super) fn compile_trait_default_method(&mut self, trait_name: &str, method_idx: usize, impl_type_name: &str, instance_idx: usize) {
        // Look up the TraitDecl method with a body in the user module (indexed directly by method_idx)
        let found = self.module.declarations.iter().find_map(|d| {
            if let crate::ast::Ast::Decl::TraitDecl { name, methods, .. } = &d.node {
                if *name == trait_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            return Some((
                                method.body.unwrap(),
                                method.is_async,
                                method.params.clone(),
                            ));
                        }
                    }
                }
            }
            None
        });

        let (body_expr, is_async, params) = match found {
            Some(x) => x,
            None => return,
        };

        // Obtain the pre-registered specialized subgraph sg_id from trait_default_subgraphs
        let trait_idx = match self.sema.trait_def_index.get(trait_name) {
            Some(&idx) => idx,
            None => return,
        };
        let type_id = match self.sema.type_def_index.get(impl_type_name) {
            Some(&idx) => crate::types::dynamic_type_id(idx),
            None => return,
        };
        let sg_id = match self.trait_default_subgraphs.get(&(type_id, trait_idx, method_idx as u16)) {
            Some(&sg) => sg,
            None => return,
        };

        let node_start = self.graph.nodes.len() as u32;

        self.current_function_sg = Some(sg_id);
        self.current_function_id = sg_id.0;
        // Record the current specialization instance index; expr_type_name/expr_type_id use it to look
        // up sema's TraitDefaultInstance.type_name for self's concrete type (consumes sema output).
        self.current_trait_default_idx = Some(instance_idx);
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let prev_method_type = self.current_method_type.take();
        self.current_method_type = Some((impl_type_name.into(), type_id));
        self.enter_scope();

        for param in &params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }

        let return_node = self.compile_expr(body_expr);
        self.exit_scope();
        self.current_effect = prev_effect;
        self.current_function_sg = None;
        self.current_trait_default_idx = None;
        self.current_method_type = prev_method_type;

        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = sg_id.0;
    }

    /// Find a TypeDecl method by `(type_name, method_idx)`, searching both top-level
    /// declarations AND local types declared inside function bodies. Returns
    /// `(method_name, params_count, is_async)`.
    ///
    /// This is used by the IR build to pre-register and compile local type methods
    /// (step 0a-local / step 2b-local), complementing `compile_user_method` which only
    /// searches `self.module.declarations`.
    pub(super) fn find_type_method(
        &self,
        type_name: &str,
        method_idx: usize,
    ) -> Option<(&'static str, u8, bool)> {
        let module = self.module;
        let arena = &module.arena;
        // 1. Top-level declarations
        for decl in &module.declarations {
            if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &decl.node {
                if *name == type_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            return Some((
                            // SAFETY: method.name is &'a str tied to module lifetime; leak to 'static
                            // (acceptable: the module outlives the entire build).
                            Box::leak(method.name.to_string().into_boxed_str()),
                            method.params.len() as u8,
                            method.is_async,
                            ));
                        }
                    }
                }
            }
        }
        // 2. Recurse into function bodies
        let mut found: Option<(&'static str, u8, bool)> = None;
        for decl in &module.declarations {
            match &decl.node {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    if self.find_type_method_in_expr(*body, arena, type_name, method_idx, &mut found) {
                        return found;
                    }
                }
                crate::ast::Ast::Decl::TypeDecl { methods, .. }
                | crate::ast::Ast::Decl::TraitDecl { methods, .. } => {
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            if self.find_type_method_in_expr(body, arena, type_name, method_idx, &mut found) {
                                return found;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        found
    }

    pub(super) fn find_type_method_in_expr(
        &self,
        expr_id: crate::ast::Ast::ExprId,
        arena: &crate::ast::Ast::AstArena<'_>,
        type_name: &str,
        method_idx: usize,
        found: &mut Option<(&'static str, u8, bool)>,
    ) -> bool {
        let expr = &arena.expr(expr_id).node;
        match expr {
            crate::ast::Ast::Expr::Block { stmts, trailing } => {
                for s in stmts {
                    if self.find_type_method_in_stmt(*s, arena, type_name, method_idx, found) {
                        return true;
                    }
                }
                if let Some(t) = trailing {
                    return self.find_type_method_in_expr(*t, arena, type_name, method_idx, found);
                }
                false
            }
            _ => false,
        }
    }

    pub(super) fn find_type_method_in_stmt(
        &self,
        stmt_id: crate::ast::Ast::StmtId,
        arena: &crate::ast::Ast::AstArena<'_>,
        type_name: &str,
        method_idx: usize,
        found: &mut Option<(&'static str, u8, bool)>,
    ) -> bool {
        let stmt = &arena.stmt(stmt_id).node;
        if let crate::ast::Ast::Stmt::LocalDecl { decl } = stmt {
            if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = decl.as_ref() {
                if *name == type_name {
                    if let Some(method) = methods.get(method_idx) {
                        if method.body.is_some() {
                            *found = Some((
                                Box::leak(method.name.to_string().into_boxed_str()),
                                method.params.len() as u8,
                                method.is_async,
                            ));
                            return true;
                        }
                    }
                }
                // Recurse into the type's method bodies
                for m in methods.iter() {
                    if let Some(body) = m.body {
                        if self.find_type_method_in_expr(body, arena, type_name, method_idx, found) {
                            return true;
                        }
                    }
                }
            }
            if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                return self.find_type_method_in_expr(*body, arena, type_name, method_idx, found);
            }
        }
        false
    }

    /// Full version of find_type_method that returns the complete method data needed by
    /// `compile_user_method`: `(method_name, body_expr, is_async, params, return_type)`.
    /// Searches top-level then local types. Used as the fallback in compile_user_method.
    pub(super) fn find_type_method_full(
        &self,
        type_name: &str,
        method_idx: usize,
    ) -> Option<(&'static str, crate::ast::Ast::ExprId, bool, Vec<crate::ast::Ast::Param<'static>>, Option<crate::ast::Ast::TypeId>)> {
        let module = self.module;
        let arena = &module.arena;
        // Collect from a local helper that returns full method data
        let mut result: Option<(&'static str, crate::ast::Ast::ExprId, bool, Vec<crate::ast::Ast::Param<'static>>, Option<crate::ast::Ast::TypeId>)> = None;
        for decl in &module.declarations {
            match &decl.node {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    self.find_type_method_full_in_expr(*body, arena, type_name, method_idx, &mut result);
                }
                crate::ast::Ast::Decl::TypeDecl { methods, .. }
                | crate::ast::Ast::Decl::TraitDecl { methods, .. } => {
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            self.find_type_method_full_in_expr(body, arena, type_name, method_idx, &mut result);
                        }
                    }
                }
                _ => {}
            }
            if result.is_some() { return result; }
        }
        result
    }

    pub(super) fn find_type_method_full_in_expr(
        &self,
        expr_id: crate::ast::Ast::ExprId,
        arena: &crate::ast::Ast::AstArena<'_>,
        type_name: &str,
        method_idx: usize,
        found: &mut Option<(&'static str, crate::ast::Ast::ExprId, bool, Vec<crate::ast::Ast::Param<'static>>, Option<crate::ast::Ast::TypeId>)>,
    ) {
        let expr = &arena.expr(expr_id).node;
        if let crate::ast::Ast::Expr::Block { stmts, .. } = expr {
            for s in stmts {
                let stmt = &arena.stmt(*s).node;
                if let crate::ast::Ast::Stmt::LocalDecl { decl } = stmt {
                    if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = decl.as_ref() {
                        if *name == type_name {
                            if let Some(method) = methods.get(method_idx) {
                                if method.body.is_some() {
                                    // SAFETY: leak method fields to 'static (module outlives build)
                                    let m_name: &'static str = Box::leak(method.name.to_string().into_boxed_str());
                                    let m_params: Vec<crate::ast::Ast::Param<'static>> = method.params.iter()
                                        .map(|p| crate::ast::Ast::Param {
                                            name: Box::leak(p.name.to_string().into_boxed_str()),
                                            type_annotation: p.type_annotation,
                                        })
                                        .collect();
                                    *found = Some((m_name, method.body.unwrap(), method.is_async, m_params, method.return_type));
                                    return;
                                }
                            }
                        }
                        // Recurse into method bodies
                        for m in methods.iter() {
                            if let Some(body) = m.body {
                                self.find_type_method_full_in_expr(body, arena, type_name, method_idx, found);
                                if found.is_some() { return; }
                            }
                        }
                    }
                    if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                        self.find_type_method_full_in_expr(*body, arena, type_name, method_idx, found);
                        if found.is_some() { return; }
                    }
                }
            }
        }
    }

    /// Recursively collect all local `TypeDecl`s declared inside function bodies
    /// (`Stmt::LocalDecl(TypeDecl)`) across the user module.
    ///
    /// Local types are registered by Sema into `type_def_index` (so `type_id` is available),
    /// and their methods are checked, but the IR build's step 0a / step 2b only scanned
    /// top-level `m.declarations`. This collector walks into Block expressions, match arms,
    /// if branches, lambda bodies, loops, etc. to surface nested type declarations so their
    /// method subgraphs get pre-registered and compiled.
    ///
    /// Returns `Vec<(type_name, method_idx)>` pairs mirroring the step 2b format.
    pub(super) fn collect_local_type_methods(&self) -> Vec<(String, usize)> {
        let mut result = Vec::new();
        let module = self.module;
        let arena = &module.arena;
        // Scan top-level declarations for function/method bodies that may contain local types.
        for decl in &module.declarations {
            match &decl.node {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    self.collect_local_types_from_expr(*body, arena, &mut result);
                }
                crate::ast::Ast::Decl::TypeDecl { methods, .. }
                | crate::ast::Ast::Decl::TraitDecl { methods, .. } => {
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            self.collect_local_types_from_expr(body, arena, &mut result);
                        }
                    }
                }
                crate::ast::Ast::Decl::ExprDecl { expr, stmt, .. } => {
                    self.collect_local_types_from_expr(*expr, arena, &mut result);
                    if let Some(s) = stmt {
                        self.collect_local_types_from_stmt(*s, arena, &mut result);
                    }
                }
                _ => {}
            }
        }
        result
    }

    pub(super) fn collect_local_types_from_stmt(
        &self,
        stmt_id: crate::ast::Ast::StmtId,
        arena: &crate::ast::Ast::AstArena<'_>,
        out: &mut Vec<(String, usize)>,
    ) {
        let stmt = &arena.stmt(stmt_id).node;
        match stmt {
            crate::ast::Ast::Stmt::LocalDecl { decl } => {
                if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = decl.as_ref() {
                    for (idx, m) in methods.iter().enumerate() {
                        if m.body.is_some() {
                            out.push((name.to_string(), idx));
                        }
                    }
                    // Recurse into the type's own method bodies (nested local types)
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            self.collect_local_types_from_expr(body, arena, out);
                        }
                    }
                }
                // Also recurse into local FunDecl bodies (functions can nest types)
                if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                    self.collect_local_types_from_expr(*body, arena, out);
                }
            }
            crate::ast::Ast::Stmt::ValDecl { value, .. }
            | crate::ast::Ast::Stmt::VarDecl { value, .. }
            | crate::ast::Ast::Stmt::Expression { expr: value, .. }
            | crate::ast::Ast::Stmt::Return { value: Some(value), .. }
            | crate::ast::Ast::Stmt::Throw { expr: value, .. }
            | crate::ast::Ast::Stmt::Defer { expr: value, .. } => {
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Stmt::Assignment { target, value, .. }
            | crate::ast::Ast::Stmt::CompoundAssignment { target, value, .. } => {
                self.collect_local_types_from_expr(*target, arena, out);
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Stmt::FieldAssignment { object, value, .. } => {
                self.collect_local_types_from_expr(*object, arena, out);
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Stmt::For { body, .. }
            | crate::ast::Ast::Stmt::While { body, .. }
            | crate::ast::Ast::Stmt::Loop { body } => {
                self.collect_local_types_from_expr(*body, arena, out);
            }
            _ => {}
        }
    }

    pub(super) fn collect_local_types_from_expr(
        &self,
        expr_id: crate::ast::Ast::ExprId,
        arena: &crate::ast::Ast::AstArena<'_>,
        out: &mut Vec<(String, usize)>,
    ) {
        let expr = &arena.expr(expr_id).node;
        match expr {
            crate::ast::Ast::Expr::Block { stmts, trailing } => {
                for s in stmts {
                    self.collect_local_types_from_stmt(*s, arena, out);
                }
                if let Some(t) = trailing {
                    self.collect_local_types_from_expr(*t, arena, out);
                }
            }
            crate::ast::Ast::Expr::If { cond, then_branch, else_branch } => {
                self.collect_local_types_from_expr(*cond, arena, out);
                self.collect_local_types_from_expr(*then_branch, arena, out);
                if let Some(e) = else_branch {
                    self.collect_local_types_from_expr(*e, arena, out);
                }
            }
            crate::ast::Ast::Expr::Match { scrutinee, arms } => {
                self.collect_local_types_from_expr(*scrutinee, arena, out);
                for arm in arms {
                    if let Some(g) = arm.guard {
                        self.collect_local_types_from_expr(g, arena, out);
                    }
                    self.collect_local_types_from_expr(arm.body, arena, out);
                }
            }
            crate::ast::Ast::Expr::Lambda { body, .. } => match body {
                crate::ast::Ast::LambdaBody::Block(e) | crate::ast::Ast::LambdaBody::Expression(e) => {
                    self.collect_local_types_from_expr(*e, arena, out);
                }
            },
            // Expressions that contain sub-expressions
            crate::ast::Ast::Expr::Call { callee, args, .. } => {
                self.collect_local_types_from_expr(*callee, arena, out);
                for a in args {
                    self.collect_local_types_from_expr(*a, arena, out);
                }
            }
            crate::ast::Ast::Expr::MethodCall { recv, args, .. }
            | crate::ast::Ast::Expr::SafeMethodCall { recv, args, .. } => {
                self.collect_local_types_from_expr(*recv, arena, out);
                for a in args {
                    self.collect_local_types_from_expr(*a, arena, out);
                }
            }
            crate::ast::Ast::Expr::Binary { lhs, rhs, .. }
            | crate::ast::Ast::Expr::Assign { target: lhs, value: rhs }
            | crate::ast::Ast::Expr::Elvis { lhs, rhs } => {
                self.collect_local_types_from_expr(*lhs, arena, out);
                self.collect_local_types_from_expr(*rhs, arena, out);
            }
            crate::ast::Ast::Expr::CompoundAssign { target, value, .. } => {
                self.collect_local_types_from_expr(*target, arena, out);
                self.collect_local_types_from_expr(*value, arena, out);
            }
            crate::ast::Ast::Expr::As { expr, .. } => {
                self.collect_local_types_from_expr(*expr, arena, out);
            }
            _ => {}
        }
    }

}
