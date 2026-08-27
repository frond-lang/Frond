//! Lambda — Lazy-evaluation lowering: select / lambda / inline trait / lazy. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Compile a `select` expression.
    ///
    /// Each `SelectArm` is compiled into an independent subgraph (event-source check + body).
    /// The Gate node (`compute_select_gate`) selects the first ready branch: if a ready branch
    /// exists, it launches that branch's subgraph; if none is ready, the frame suspends,
    /// registers all event sources to wait, and wakes up to re-check when any event arrives.
    pub(super) fn compile_select(&mut self, arms: &[crate::ast::Ast::SelectArm<'_>]) -> NodeId {
        let mut branches = Vec::with_capacity(arms.len());

        for arm in arms {
            let (event_kind, event_source_node, body_expr, binding) = match arm {
                crate::ast::Ast::SelectArm::Receive { channel_expr, body, binding } => {
                    // `channel_expr` is of the form `ch.recv()`: at compile time we must take the
                    // receiver of `recv` (the channel value), not the entire method call
                    // (`recv()` returns the received value, not the channel itself).
                    // Determined via the intrinsic flag in sema's `method_dispatches` (eliminates
                    // string guards).
                    let ch_node = match &self.current_module().arena.expr(*channel_expr).node {
                        crate::ast::Ast::Expr::MethodCall { recv, .. } => {
                            let key = crate::sema::Sema::module_expr_key(
                                self.expr_key_module(),
                                channel_expr.0 as u64,
                            );
                            let is_recv = self.sema.method_dispatches.get(&key)
                                .and_then(|d| d.intrinsic)
                                .is_some_and(|i| i == crate::sema::Sema::IntrinsicKind::ChannelAwait);
                            if is_recv {
                                self.compile_subexpr(*recv)
                            } else {
                                self.compile_subexpr(*channel_expr)
                            }
                        }
                        _ => self.compile_subexpr(*channel_expr),
                    };
                    (EventSourceKind::Channel, ch_node, *body, *binding)
                }
                crate::ast::Ast::SelectArm::Timeout { duration, body } => {
                    let dur_node = self.compile_subexpr(*duration);
                    (EventSourceKind::Timer, dur_node, *body, None)
                }
            };

            // Create a subgraph for each branch: register a placeholder first (node_range is
            // back-filled after compilation).
            let node_start = self.graph.nodes.len() as u32;
            let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
            self.graph.add_subgraph(SubGraph {
                converter_generated: false,
                id: sg_id,
                node_range: (NodeId(node_start), NodeId(node_start)),
                param_count: 0,
                entry_node: NodeId(node_start),
                return_node: NodeId(node_start),
                has_suspend: true,
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

            let prev_sg = self.current_function_sg;
            self.current_function_sg = Some(sg_id);
            self.enter_scope();

            // When the arm has a binding (`ch.recv() => v => body`), the received value is bound
            // to `binding` inside the body. Emit a 0-input parameter node as the FIRST node of the
            // branch subgraph; the runtime injects the recv'd value into it (param_count=1) when the
            // branch is selected. The body then references `binding` via this node.
            let mut branch_param_count: u8 = 0;
            if let Some(name) = binding {
                let off = self.graph.inputs_pool.push(&[]);
                let param_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset: off,
                    compute_fn: CF_NOOP,
                });
                self.bind_var(name, param_node);
                branch_param_count = 1;
            }

            // Compile the body (variable bindings inside the body live in the subgraph scope).
            let result_node = self.compile_expr(body_expr);

            self.exit_scope();
            self.current_function_sg = prev_sg;

            let node_end = self.graph.nodes.len() as u32;
            let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
            sg.node_range = (NodeId(node_start), NodeId(node_end));
            sg.return_node = result_node;
            sg.param_count = branch_param_count;

            branches.push(SelectBranch {
                subgraph_id: sg_id,
                event_kind,
                event_source_node,
            });
        }

        // Create the Gate node (the core of `select`: choose the first ready branch).
        // The Gate depends on every branch's `event_source_node` + `current_effect`, ensuring
        // readiness is checked only after all event sources (channel/timer) are evaluated.
        let mut gate_inputs: Vec<NodeId> = branches.iter().map(|b| b.event_source_node).collect();
        if let Some(eff) = self.current_effect {
            gate_inputs.push(eff);
        }
        let gate_off = self.graph.inputs_pool.push(&gate_inputs);
        let gate_node = self.graph.add_node(Node {
            kind: NodeKind::Gate,
            input_count: gate_inputs.len() as u8,
            inputs_offset: gate_off,
            compute_fn: CF_SELECT_GATE, // compute_select_gate
        });
        self.graph.set_select_info(gate_node, SelectInfo { branches });

        gate_node
    }

    /// Compile a `Lambda` expression into a closure subgraph + closure construct node.
    ///
    /// Appended-parameter model: captured variables are appended to the end of the subgraph's
    /// parameter list.
    /// - Subgraph `param_count` = lambda parameter count + captured-variable count.
    /// - The first N nodes of the subgraph are lambda parameter nodes; the rest are captured
    ///   upvalue parameter nodes.
    /// - A closure construct node (`compute_fn` 40) is created in the current scope; its inputs
    ///   are the captured value nodes.
    /// - The value of the Lambda expression is the closure construct node (at runtime it produces
    ///   a Closure heap object).
    pub(super) fn compile_lambda(
        &mut self,
        params: &[crate::ast::Ast::Param<'_>],
        body_expr: crate::ast::Ast::ExprId,
        is_async: bool,
        fn_name: Option<&str>,
        lambda_expr_id: Option<crate::ast::Ast::ExprId>,
    ) -> NodeId {
        // 1. Capture analysis (unified): consume Sema's authoritative capture
        //    table for this nested scope. This replaces the builder's own
        //    `collect_free_idents_expr` re-scan. Each CaptureInfo carries the
        //    variable name and a by-val (Snapshot) / by-ref (Reference) mode.
        //
        //    Node resolution still uses `lookup_var` + the `captured_scopes`
        //    chain (unchanged): the Sema table is the single source of truth for
        //    *which* variables are captured and *how*, but the NodeId is an IR
        //    concept that only the builder can resolve.
        let param_names: rustc_hash::FxHashSet<&str> =
            params.iter().map(|p| p.name).collect();
        // Lookup Sema's capture table. For Lambda expressions the key is the
        // Lambda's own ExprId; for nested function declarations (Stmt::LocalDecl)
        // there is no Lambda expr, so Sema records captures under the *body*
        // expression's ExprId — fall back to that.
        let sema_captures: Vec<crate::sema::Sema::CaptureInfo> = {
            let key_id = lambda_expr_id.unwrap_or(body_expr);
            self.lookup_captures(key_id).to_vec()
        };
        let mut captured: Vec<(String, NodeId)> = Vec::new();
        let mut pending_cell_captures: Vec<(String, NodeId)> = Vec::new();

        // Self-reference detection: a named function that references itself in its body becomes an
        // upvalue placeholder.
        // At runtime `compute_closure_call` injects the closure value into that slot, enabling
        // recursive calls.
        let self_upvalue_idx = if let Some(fname) = fn_name {
            if !param_names.contains(fname)
                && sema_captures.iter().any(|c| c.name.as_ref() == fname)
            {
                let void_node = self.compile_void_const();
                let idx = captured.len();
                captured.push((fname.to_string(), void_node));
                idx as i32
            } else {
                -1
            }
        } else {
            -1
        };

        for cap in &sema_captures {
            let ident = cap.name.as_ref();
            if param_names.contains(ident) {
                continue;
            }
            // Skip the self-reference name (already added as a placeholder upvalue).
            if Some(ident) == fn_name && self_upvalue_idx >= 0 {
                continue;
            }
            if let Some(node) = self.lookup_var(ident) {
                if !captured.iter().any(|(n, _)| n.as_str() == ident) {
                    // If the variable has already been captured by an outer lambda, use the outer
                    // original node.
                    // This ensures the WriteBack target points to the outermost defining node (root
                    // frame), not an intermediate lambda's upvalue parameter node (an
                    // intermediate-frame copy).
                    let outer_node = self.captured_scopes.iter().rev()
                        .find_map(|scope| scope.iter()
                            .find(|(n, _)| n.as_str() == ident)
                            .map(|(_, node)| *node))
                            .unwrap_or(node);
                    captured.push((ident.to_string(), outer_node));
                }
            }
        }

        // No Step 7: the declaring frame keeps its original variable bindings.
        // Cross-function upvalue visibility is handled at runtime: compute_closure_construct
        // wraps upvalues in Cells, compute_closure_call preserves Cell references, and
        // compute_writeback's Path 3 writes to Cell upvalues for escaped closures.
        // For defer bodies (same-function branch frames), WriteBack's frame-chain paths
        // (Path 0/1) propagate upvalue mutations within the function's frame chain.

        let param_count = (params.len() + captured.len()) as u8;

        // 2. Register a placeholder subgraph (node range is filled in after compilation).
        let sg_id = self.register_subgraph_placeholder("", param_count, is_async);
        let node_start = self.graph.nodes.len() as u32;

        // 3. Enter the lambda scope: create lambda parameter nodes first, then captured upvalue
        //    parameter nodes, and `bind_var` all of them (capture nodes shadow outer bindings of
        //    the same name within the lambda scope).
        self.enter_scope();
        for param in params {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let param_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            self.bind_var(param.name, param_node);
        }
        for (name, _outer_node) in &captured {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let upvalue_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            // Cell-capture registration (place model 4): when the outer
            // binding is cell-backed, the upvalue carries the SHARED cell Arc
            // (construct wraps it once, call unwraps that layer). Routing the
            // body's reads/writes through deref ops on it gives by-ref
            // capture semantics via the shared cell — no WriteBack machinery.
            // Without this, reads would leak the Arc as a plain value.
            // NOTE: the eligibility check MUST run before bind_var — once the
            // name rebinds to the upvalue, the binding-identity guard
            // correctly reports it no longer points at the outer cell.
            let is_cell_capture = self.lookup_cell_binding(name).is_some();
            self.bind_var(name, upvalue_node);
            if is_cell_capture {
                // Registration is DEFERRED to just before the body compile:
                // the owner tag must match the id the BODY compiles under
                // (escaping lambdas switch function_id below), or the body's
                // lookups filter their own registration out and fall back to
                // the WriteBack path.
                pending_cell_captures.push((name.clone(), upvalue_node));
            }
        }

        // 4. Compile the body to obtain the return node.
        //    Set `current_sg_start = node_start` so `is_in_current_subgraph` correctly
        //    identifies nodes inside the lambda (including upvalue placeholder nodes), preventing
        //    captured-variable assignments from taking the local path by mistake.
        //    Reset `current_effect = None` to isolate the lambda body from the outer effect chain,
        //    ensuring that a block with no trailing expression returns the `void_const` created
        //    inside the lambda rather than an outer effect node.
        //    Push onto `captured_scopes` so that `Assignment` can recognize captured variables
        //    and create WriteBacks.

        // Escape analysis (Bug #41 + Bug #40 loop capture):
        // Consume the analyzer's unified escape table; the IR performs no parallel escape
        // analysis.
        // 1. Tail-position escape (Bug #41): a lambda at the tail position of its enclosing
        //    function -> the defining frame is destroyed after the function returns.
        // 2. Loop-body capture escape (Bug #40): a lambda captures a loop-body local variable
        //    -> after the loop-body frame is destroyed, the access reads null.
        // Both cases require allocating an independent `function_id` and taking the cross-function
        // Cell path to persist the upvalue.
        let escapes = lambda_expr_id.is_some_and(|id| {
            self.current_analysis()
                .map_or(false, |r| {
                    r.escape.lookup(id).is_some_and(|info| {
                        matches!(
                            info,
                            crate::pass::Analyzer::EscapeInfo::Escapes(
                                crate::pass::Analyzer::EscapeKind::Lambda { .. }
                            )
                        )
                    })
                })
        });

        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let prev_effect = self.current_effect;
        self.current_effect = None;
        // Set `current_function_sg` so defer statements can register into the lambda subgraph's
        // `defer_table`.
        // If left unset, defer bodies would be lost (`current_function_sg` would be `None` or point
        // to the outer function).
        let prev_func_sg = self.current_function_sg;
        self.current_function_sg = Some(sg_id);
        // Escaping lambdas use an independent `function_id` so that subgraphs inside the body
        // (if-else/match branches, etc.) inherit this id and are distinguished from the enclosing
        // function -> cross-function Cell path.
        let prev_func_id = self.current_function_id;
        if escapes {
            self.current_function_id = sg_id.0;
        }
        self.captured_scopes.push(captured.clone());
        // Deferred cell-capture registration: AFTER the escape function_id
        // switch, so the owner tag matches the body's compile-time id.
        if !pending_cell_captures.is_empty() {
            let owner = self.current_function_id;
            for (name, upvalue_node) in &pending_cell_captures {
                if let Some(scope) = self.cell_bound.last_mut() {
                    scope.insert(name.clone(), (*upvalue_node, owner));
                }
            }
            pending_cell_captures.clear();
        }

        // Unified entry: memoize/tail_rec/non_tail_rec apply equally to closures
        // (the lambda is not in the call_graph, so `lookup_memo_strategy` returns None -> the
        // default `compile_expr` path is taken).
        let lambda_name = fn_name.unwrap_or("");
        // Bug #97: tail `expr?` must re-wrap into Ok when the lambda returns Throw.
        // The lambda's declared `: T` lives in its ExprInfo (a Fn type handle).
        let fn_returns_throw = lambda_expr_id
            .and_then(|lid| {
                let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), lid.0 as u64);
                self.sema.expr_types.get(&key).map(|info| info.ty)
            })
            .map(|t| self.handle_returns_throw(t))
            .unwrap_or(false);
        let return_node = self.compile_function_body(lambda_name, None, body_expr, params, false, is_async, fn_returns_throw);

        self.current_sg_start = prev_sg_start;
        self.current_effect = prev_effect;
        self.current_function_sg = prev_func_sg;
        self.current_function_id = prev_func_id;
        self.captured_scopes.pop();
        self.exit_scope();

        // 5. Update the subgraph's node_range + return_node + function_id + upvalue metadata.
        // `function_id`: escaping lambdas use an independent id (`sg_id.0`); non-escaping lambdas
        // inherit the outer id.
        // - Escaping: `same_function=false` -> cross-function Cell path (defining frame destroyed;
        //   Cell persists the upvalue).
        // - Non-escaping: `same_function=true` -> frame-chain path (defining frame alive; shared
        //   state).
        // `upvalue_count` + `upvalue_outer_nodes` are used by `start_subgraph` to inject the
        // current parent-frame values on `same_function` calls (capture-by-reference semantics).
        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = is_async;
        sg.function_id = if escapes { sg_id.0 } else { prev_func_id };
        sg.upvalue_count = captured.len() as u8;
        sg.upvalue_outer_nodes = captured.iter().map(|(_, n)| *n).collect();

        // Register captured variables into `captured_vars`: when an outer `Assignment` assigns to
        // one of these variables, it must emit a WriteBack to the original node, so a
        // `same_function` closure call reads the latest value.
        for (name, node) in &captured {
            self.captured_vars.entry(name.clone()).or_insert(*node);
        }

        // 6. Create the closure construct node in the current scope (inputs = captured outer
        //    nodes, `compute_fn` 40).
        let upvalue_nodes: Vec<NodeId> = captured.iter().map(|(_, n)| *n).collect();
        let inputs_offset = self.graph.inputs_pool.push(&upvalue_nodes);
        let construct_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: upvalue_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_CLOSURE_CONSTRUCT, // compute_closure_construct
        });
        self.graph.set_closure_info(
            construct_node,
            ClosureInfo {
                subgraph_id: sg_id,
                arity: params.len() as u8,
                self_upvalue_idx,
            },
        );
        construct_node
    }

    /// Compile an `inline_trait` expression: each method is compiled into a subgraph (with upvalue
    /// captures), and a TraitValue construct node (`compute_fn=266`) is created that at runtime
    /// packs the closures together.
    ///
    /// Method-subgraph compilation mirrors `compile_lambda`: free-variable analysis ->
    /// placeholder subgraph -> enter scope (params + upvalues) -> compile body -> fill
    /// `node_range`.
    /// The upvalues of all methods are concatenated in order as the construct node's inputs.
    pub(super) fn compile_inline_trait(&mut self, expr_id: crate::ast::Ast::ExprId, methods: &[crate::ast::Ast::MethodDecl<'_>]) -> NodeId {
        // Infer the trait name (from `sema.expr_types` as `Type::TraitObject`).
        let trait_name = self.expr_type_name(expr_id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        // Reorder the `inline_trait` methods by the declaration order in `trait_def.methods`,
        // ensuring `method_values[i]` corresponds to `trait_def.methods[i]`, so the vtable can
        // dispatch by `method_idx` positional index (Task 10).
        let method_order: Vec<usize> = if let Some(trait_def) = self.sema.get_trait_def(&trait_name) {
            let mut order: Vec<usize> = (0..methods.len()).collect();
            order.sort_by_key(|&i| {
                trait_def.methods.iter().position(|tm| tm.name.as_ref() == methods[i].name)
                    .unwrap_or(usize::MAX)
            });
            order
        } else {
            (0..methods.len()).collect()
        };

        let mut method_names: Vec<String> = Vec::with_capacity(methods.len());
        let mut method_entries: Vec<TraitMethodEntry> = Vec::with_capacity(methods.len());
        let mut all_upvalue_nodes: Vec<NodeId> = Vec::new();

        for &idx in &method_order {
            let m = &methods[idx];
            let body_expr = match m.body {
                Some(b) => b,
                None => {
                    self.errors.push(format!(
                        "compile_inline_trait: method {} has no body (inline_trait requires all methods to have bodies)",
                        m.name
                    ));
                    continue;
                }
            };

            // 1. Free-variable analysis: collect outer variables referenced in the body (excluding
            //    the method's own parameters).
            let param_names: rustc_hash::FxHashSet<&str> =
                m.params.iter().map(|p| p.name).collect();
            let mut ident_names: Vec<String> = Vec::new();
            self.collect_free_idents_expr(body_expr, &mut ident_names);
            let mut captured: Vec<(String, NodeId)> = Vec::new();
            for name in &ident_names {
                if param_names.contains(name.as_str()) {
                    continue;
                }
                if let Some(node) = self.lookup_var(name) {
                    if !captured.iter().any(|(n, _)| n == name) {
                        captured.push((name.clone(), node));
                    }
                }
            }

            let param_count = (m.params.len() + captured.len()) as u8;

            // 2. Register a placeholder subgraph.
            let sg_id = self.register_subgraph_placeholder("", param_count, m.is_async);
            let node_start = self.graph.nodes.len() as u32;

            // 3. Enter the method scope: parameter nodes + upvalue nodes.
            self.enter_scope();
            for param in &m.params {
                let inputs_offset = self.graph.inputs_pool.push(&[]);
                let param_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset,
                    compute_fn: CF_NOOP,
                });
                self.bind_var(param.name, param_node);
            }
            for (name, _outer_node) in &captured {
                let inputs_offset = self.graph.inputs_pool.push(&[]);
                let upvalue_node = self.graph.add_node(Node {
                    kind: NodeKind::Const,
                    input_count: 0,
                    inputs_offset,
                    compute_fn: CF_NOOP,
                });
                // Cell-capture registration (see compile_lambda): the check
                // MUST precede bind_var (binding-identity guard).
                let is_cell_capture = self.lookup_cell_binding(name).is_some();
                self.bind_var(name, upvalue_node);
                if is_cell_capture {
                    if let Some(scope) = self.cell_bound.last_mut() {
                        scope.insert(
                            name.clone(),
                            (upvalue_node, self.current_function_id),
                        );
                    }
                }
            }

            // 4. Compile the method body.
            //    Set `current_sg_start` + reset `current_effect` + push `captured_scopes`
            //    (consistent with `compile_lambda`).
            let prev_sg_start = self.current_sg_start;
            self.current_sg_start = node_start;
            let prev_effect = self.current_effect;
            self.current_effect = None;
            self.captured_scopes.push(captured.clone());

            let return_node = self.compile_expr(body_expr);

            self.current_sg_start = prev_sg_start;
            self.current_effect = prev_effect;
            self.captured_scopes.pop();
            self.exit_scope();

            // 5. Fill in the subgraph's `node_range` + upvalue metadata.
            let node_end = self.graph.nodes.len() as u32;
            let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
            sg.node_range = (NodeId(node_start), NodeId(node_end));
            sg.entry_node = NodeId(node_start);
            sg.return_node = return_node;
            sg.has_suspend = m.is_async;
            sg.upvalue_count = captured.len() as u8;
            sg.upvalue_outer_nodes = captured.iter().map(|(_, n)| *n).collect();

            // 6. Collect upvalue nodes + record method info.
            let upvalue_count = captured.len() as u8;
            for (_, n) in &captured {
                all_upvalue_nodes.push(*n);
            }
            method_names.push(m.name.to_string());
            method_entries.push(TraitMethodEntry {
                subgraph_id: sg_id,
                arity: m.params.len() as u8,
                upvalue_count,
            });
        }

        // Construct the TraitValue construct node (inputs = the concatenated upvalues of all
        // methods).
        let inputs_offset = self.graph.inputs_pool.push(&all_upvalue_nodes);
        let construct_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: all_upvalue_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_TRAIT_CONSTRUCT, // compute_trait_construct
        });
        self.graph.set_trait_construct_info(
            construct_node,
            TraitConstructInfo {
                trait_name,
                method_names,
                methods: method_entries,
            },
        );
        construct_node
    }

    /// Compile a `lazy` expression: the operand is compiled into a zero-parameter thunk
    /// subgraph, and a LazyValue construct node (`compute_fn=267`) is created that at runtime
    /// produces an unevaluated LazyValue.
    ///
    /// The thunk subgraph captures outer free variables (the same capture mechanism as for
    /// lambdas). On the first force it launches the subgraph to compute the value; the result is
    /// cached for reuse by subsequent forces.
    pub(super) fn compile_lazy(&mut self, expr_id: crate::ast::Ast::ExprId, operand: crate::ast::Ast::ExprId) -> NodeId {
        let _ = expr_id; // trait_name inference is not yet needed; the parameter is retained for
                         // future force semantics.
        // 1. Free-variable analysis.
        let mut ident_names: Vec<String> = Vec::new();
        self.collect_free_idents_expr(operand, &mut ident_names);
        let mut captured: Vec<(String, NodeId)> = Vec::new();
        let mut pending_cell_captures: Vec<(String, NodeId)> = Vec::new();
        for name in &ident_names {
            if let Some(node) = self.lookup_var(name) {
                if !captured.iter().any(|(n, _)| n == name) {
                    captured.push((name.clone(), node));
                }
            }
        }

        let param_count = captured.len() as u8;

        // 2. Register a placeholder subgraph (thunk: no explicit parameters, only upvalues).
        let sg_id = self.register_subgraph_placeholder("", param_count, false);
        let node_start = self.graph.nodes.len() as u32;

        // 3. Enter the thunk scope: upvalue nodes.
        self.enter_scope();
        for (name, _outer_node) in &captured {
            let inputs_offset = self.graph.inputs_pool.push(&[]);
            let upvalue_node = self.graph.add_node(Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset,
                compute_fn: CF_NOOP,
            });
            // Cell-capture registration (place model 4): when the outer
            // binding is cell-backed, the upvalue carries the SHARED cell Arc
            // (construct wraps it once, call unwraps that layer). Routing the
            // body's reads/writes through deref ops on it gives by-ref
            // capture semantics via the shared cell — no WriteBack machinery.
            // Without this, reads would leak the Arc as a plain value.
            // NOTE: the eligibility check MUST run before bind_var — once the
            // name rebinds to the upvalue, the binding-identity guard
            // correctly reports it no longer points at the outer cell.
            let is_cell_capture = self.lookup_cell_binding(name).is_some();
            self.bind_var(name, upvalue_node);
            if is_cell_capture {
                // Registration is DEFERRED to just before the body compile:
                // the owner tag must match the id the BODY compiles under
                // (escaping lambdas switch function_id below), or the body's
                // lookups filter their own registration out and fall back to
                // the WriteBack path.
                pending_cell_captures.push((name.clone(), upvalue_node));
            }
        }

        // 4. Register deferred cell captures (no escape id switch in lazy
        //    thunks — registering here keeps the pattern uniform), then
        //    compile the operand.
        if !pending_cell_captures.is_empty() {
            let owner = self.current_function_id;
            for (name, upvalue_node) in &pending_cell_captures {
                if let Some(scope) = self.cell_bound.last_mut() {
                    scope.insert(name.clone(), (*upvalue_node, owner));
                }
            }
            pending_cell_captures.clear();
        }
        let return_node = self.compile_expr(operand);
        self.exit_scope();

        // 5. Fill in the subgraph's `node_range`.
        let node_end = self.graph.nodes.len() as u32;
        let sg = &mut self.graph.subgraphs[sg_id.0 as usize];
        sg.node_range = (NodeId(node_start), NodeId(node_end));
        sg.entry_node = NodeId(node_start);
        sg.return_node = return_node;
        sg.has_suspend = false;

        // 6. Construct the LazyValue construct node (inputs = upvalues).
        let upvalue_nodes: Vec<NodeId> = captured.iter().map(|(_, n)| *n).collect();
        let inputs_offset = self.graph.inputs_pool.push(&upvalue_nodes);
        let construct_node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: upvalue_nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_LAZY_CONSTRUCT, // compute_lazy_construct
        });
        self.graph.set_lazy_construct_info(
            construct_node,
            LazyConstructInfo { thunk_sg: sg_id },
        );
        construct_node
    }

}
