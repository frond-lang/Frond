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
        let prev_in_loop = self.in_loop_body;
        self.in_loop_body = true;
        let body_last = self.compile_expr(body);
        self.in_loop_body = prev_in_loop;
        self.current_sg_start = prev_sg_start;

        self.loop_stack.pop();
        self.exit_scope();

        // Tail-recursion eliminated: `return_node = body_last`; frame reuse is handled by the
        // Engine's `reset_loop_iteration`.
        let node_end = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
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
        let node_start = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        // Placeholder registration (reserve the id first to allow recursive references).
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

        // Compile the condition.
        // Reset `current_effect = None` to avoid creating CF_SEQ nodes inside the loop subgraph
        // that depend on the external effect chain.
        // After `reset_loop_iteration`, the loop-body frame's value table is cleared and external
        // effect nodes are not re-copied; CF_SEQ nodes depending on them would stay pending
        // forever, causing a deadlock.
        let prev_effect = self.current_effect;
        self.current_effect = None;
        let cond_node = self.compile_subexpr(condition);
        // body subgraph (trailing tail-recursive call to while_sg).
        let body_sg = self.compile_loop_body_subgraph(body, sg_id);
        // void subgraph (false branch; loop ends): includes CF_DEFER_RUN for defer-in-loop.
        let void_sg = self.compile_defer_run_subgraph();
        self.current_effect = prev_effect;

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
        });
        sg_id
    }

}
