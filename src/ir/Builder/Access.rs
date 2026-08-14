//! Access — Field / index / slice / record / array / atomic / await lowering. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Build an Await node: EventSource declaration + Await node (spec 4.5; not ready -> frame suspends).
    ///
    /// Shared by await/recv: infer event-source kind -> register EventSourceDecl -> generate the Await node.
    pub(super) fn build_await_node(
        &mut self,
        recv: crate::ast::Ast::ExprId,
        recv_node: NodeId,
    ) -> NodeId {
        let es_inputs_offset = self.graph.inputs_pool.push(&[]);
        let es_node = self.graph.add_node(Node {
            kind: NodeKind::EventSource,
            input_count: 0,
            inputs_offset: es_inputs_offset,
            compute_fn: CF_NOOP, // noop
        });
        let event_kind = self.infer_event_source_kind(recv);
        // W3C: register into the INNERMOST subgraph being compiled (branch /
        // loop body) so `compute_await`'s `frame.subgraph_id` lookup finds the
        // declaration (Bug #24 class, fixed structurally instead of by
        // post-hoc migration). The branch context only applies when it belongs
        // to the function being compiled — a lambda compiled inside a branch
        // arm switches current_function_id, so awaits in its body register to
        // the lambda's own subgraph instead.
        let current_sg = self
            .current_branch_sg
            .filter(|b| {
                self.graph
                    .subgraphs
                    .get(b.0 as usize)
                    .map(|sg| sg.function_id == self.current_function_id)
                    .unwrap_or(false)
            })
            .or(self.current_function_sg);
        if let Some(sg_id) = current_sg {
            if let Some(sg) = self.graph.subgraphs.get_mut(sg_id.0 as usize) {
                sg.event_source_decls.push(EventSourceDecl {
                    node: es_node,
                    kind: event_kind,
                });
            }
        }
        // current_effect appended at the end as an implicit dependency (consistent with compile_call):
        // ensures await executes only after prior effects (e.g. producer.await()) complete,
        // otherwise result_ch.recv() would become ready before producer.await() and suspend on an empty channel, causing deadlock.
        let mut await_inputs = vec![recv_node];
        if let Some(eff) = self.current_effect {
            await_inputs.push(eff);
        }
        let await_inputs_offset = self.graph.inputs_pool.push(&await_inputs);
        let await_node = self.graph.add_node(Node {
            kind: NodeKind::Await,
            input_count: await_inputs.len() as u8,
            inputs_offset: await_inputs_offset,
            compute_fn: CF_AWAIT, // compute_await
        });
        self.graph.set_await_event_source(await_node, es_node);
        await_node
    }

    /// Infer the event-source kind from the recv expression.
    ///
    /// Async<T> -> AsyncJoin, Channel<T>/Receiver<T> -> Channel, Timer -> Timer
    /// default -> AsyncJoin (5a-2 primarily supports awaiting async handles)
    pub(super) fn infer_event_source_kind(&self, recv: crate::ast::Ast::ExprId) -> EventSourceKind {
        // Look up the recv's type name in Sema expr_types
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), recv.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            if let Some(ref tn) = info.type_name {
                let tn = tn.as_ref();
                // Built-in generics + Timer: derived from Type::from_type_name + family() (eliminates string matching)
                if let Some(ty) = crate::types::Type::from_type_name(tn) {
                    use crate::types::TypeFamily;
                    match ty.family() {
                        TypeFamily::Async => return EventSourceKind::AsyncJoin,
                        TypeFamily::Channel | TypeFamily::Receiver => return EventSourceKind::Channel,
                        TypeFamily::Timer => return EventSourceKind::Timer,
                        _ => {}
                    }
                }
            }
        }
        EventSourceKind::AsyncJoin
    }

    /// Check if an expression's inferred type is `Async<T>` (Bug #79: auto-await forwarding).
    /// Returns false when the type is unknown or not Async, unlike `infer_event_source_kind`
    /// which defaults to AsyncJoin.
    pub(super) fn expr_type_is_async(&self, expr_id: crate::ast::Ast::ExprId) -> bool {
        let key = crate::sema::Sema::module_expr_key(self.expr_key_module(), expr_id.0 as u64);
        if let Some(info) = self.sema.expr_types.get(&key) {
            if let Some(ref tn) = info.type_name {
                if let Some(ty) = crate::types::Type::from_type_name(tn.as_ref()) {
                    return ty.family() == crate::types::TypeFamily::Async;
                }
            }
        }
        false
    }

    /// Compile a field access.
    ///
    /// Binds compute_record_field_get, storing only the field name as the runtime by-name lookup key.
    pub(super) fn compile_field_access(
        &mut self,
        _expr_id: crate::ast::Ast::ExprId,
        recv: crate::ast::Ast::ExprId,
        field: &str,
    ) -> NodeId {
        // Qualified-name constructor: Type.Ctor (zero-parameter constructor)
        if let crate::ast::Ast::Expr::Ident(type_name) = &self.current_module().arena.expr(recv).node {
            if let Some((ctor_type_name, ctor_name, field_names, kind, is_nullary)) =
                self.check_qualified_ctor_ir(type_name, field)
            {
                if is_nullary {
                    let inputs_offset = self.graph.inputs_pool.push(&[]);
                    let node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 0,
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

        // Cross-module constant access (Math.PI): sema has recorded the recv's expr key → mangled
        // name in module_const_recv_exprs. On a hit, skip recv compilation and look up the mangled
        // name in global_var_slots to emit compile_global_load, sharing the local global var path.
        let recv_key = crate::sema::Sema::module_expr_key(
            self.expr_key_module(),
            recv.0 as u64,
        );
        if let Some(mangled) = self.sema.module_const_recv_exprs.get(&recv_key) {
            if let Some(&slot) = self.global_var_slots.get(mangled.as_str()) {
                return self.compile_global_load(slot);
            }
        }
        let recv_node = self.compile_subexpr(recv);
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::FieldAccess,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_RECORD_FIELD_GET, // record_field_get
        });
        // Uniformly store the field name as the runtime lookup key:
        // Record/Adt both resolve via find_field(name) by name, no compile-time field_idx needed
        self.graph.set_field_set_name(node, field.to_string());
        node
    }

    /// Build a FieldAccess node for an implicit-this field read.
    ///
    /// When a bare identifier inside a method body resolves to an instance field (recorded by
    /// sema on `ExprInfo.implicit_this`), the IR synthesizes `this.<field>`. The receiver node is
    /// the `this` binding already compiled for the method body; this helper mirrors
    /// `compile_field_access` but skips recv re-compilation and qualified-ctor/global-const
    /// detection (neither applies to an implicit `this` receiver).
    pub(super) fn build_implicit_field_access(&mut self, this_node: NodeId, field: &str) -> NodeId {
        // Chain with `current_effect` to mirror the explicit `this.<field>` path:
        // `compile_expr` for `Ident("this")` chains the bound `this` node through
        // `current_effect` (line 588), ensuring the field read executes only after
        // prior side effects (e.g. a prior `pos = pos + 1` WriteBack) complete.
        // Without this chain, an implicit field read could observe a stale value
        // before an in-flight assignment updates the instance field, breaking
        // iterator-style mutation (e.g. `next()` reading `pos` after `pos = pos + 1`).
        let this_node = self.chain_effects(self.current_effect, this_node);
        let inputs_offset = self.graph.inputs_pool.push(&[this_node]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::FieldAccess,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_RECORD_FIELD_GET, // record_field_get
        });
        self.graph.set_field_set_name(node, field.to_string());
        node
    }

    /// Compile an index access.
    pub(super) fn compile_index(&mut self, recv: crate::ast::Ast::ExprId, index: crate::ast::Ast::ExprId) -> NodeId {
        let recv_node = self.compile_subexpr(recv);
        let index_node = self.compile_subexpr(index);
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node, index_node]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset,
            compute_fn: CF_ARRAY_INDEX, // array_index
        })
    }

    /// Compile a slice `recv[start..end]` (inclusive=false) or `recv[start..=end]` (inclusive=true).
    ///
    /// Three-input node (recv, start, end); the inclusive flag is stored in graph.slice_inclusive.
    /// At runtime, str is sliced by code point and array by element.
    pub(super) fn compile_slice(
        &mut self,
        recv: crate::ast::Ast::ExprId,
        start: crate::ast::Ast::ExprId,
        end: crate::ast::Ast::ExprId,
        inclusive: bool,
    ) -> NodeId {
        let recv_node = self.compile_subexpr(recv);
        let start_node = self.compile_subexpr(start);
        let end_node = self.compile_subexpr(end);
        let inputs_offset = self.graph.inputs_pool.push(&[recv_node, start_node, end_node]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 3,
            inputs_offset,
            compute_fn: CF_SLICE, // compute_slice
        });
        self.graph.set_slice_inclusive(node, inclusive);
        node
    }

    /// Compile a record construction (by positional args + type name).
    ///
    /// Used for `Err(args)` / `IOError(args)` and similar constructor calls; field names are auto-generated as `_0`, `_1`, ...
    pub(super) fn compile_record_like(&mut self, type_name: &str, args: &[crate::ast::Ast::ExprId]) -> NodeId {
        let mut inputs = Vec::with_capacity(args.len());
        for &arg in args {
            inputs.push(self.compile_subexpr(arg));
        }
        let field_names: Vec<Option<String>> = (0..args.len())
            .map(|i| Some(format!("_{}", i)))
            .collect();
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_RECORD_CONSTRUCT, // record_construct
        });
        self.graph.set_record_lit_info(
            node,
            RecordLitInfo {
                type_name: type_name.to_string(),
                field_names,
                constructor: type_name.to_string(),
                kind: RecordLitKind::Record,
            },
        );
        node
    }

    /// Compile a record construction expression.
    /// Allocation sites marked non-escaping by the analyzer use the stack-alloc compute_fn (288).
    pub(super) fn compile_record_lit(&mut self, expr_id: crate::ast::Ast::ExprId, fields: &[crate::ast::Ast::RecordFieldExpr<'_>]) -> NodeId {
        let mut inputs = Vec::with_capacity(fields.len());
        let mut field_names = Vec::with_capacity(fields.len());
        for field in fields {
            inputs.push(self.compile_subexpr(field.value));
            field_names.push(Some(field.name.to_string()));
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        // Stack-alloc marker: non-escaping allocations use compute_record_construct_stack (288)
        let compute_fn = if self.should_stack_alloc(expr_id) {
            CF_RECORD_CONSTRUCT_STACK // record_construct_stack
        } else {
            CF_RECORD_CONSTRUCT // record_construct
        };
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn,
        });
        self.graph.set_record_lit_info(
            node,
            RecordLitInfo {
                type_name: "Record".to_string(),
                field_names,
                constructor: "Record".to_string(),
                kind: RecordLitKind::Record,
            },
        );
        node
    }

    /// Compile a record extension expression `(...base, field: value, ...)`.
    ///
    /// inputs[0] = base record; inputs[1..] = update field values.
    /// RecordExtendInfo stores the update field name list (in order, corresponding to inputs[1..]).
    /// At runtime, clones fields from base, replaces/appends by update field names, and builds a new RecordValue.
    pub(super) fn compile_record_extend(
        &mut self,
        base: crate::ast::Ast::ExprId,
        updates: &[crate::ast::Ast::RecordFieldExpr<'_>],
    ) -> NodeId {
        let mut inputs = Vec::with_capacity(1 + updates.len());
        let mut update_names = Vec::with_capacity(updates.len());
        inputs.push(self.compile_subexpr(base));
        for field in updates {
            inputs.push(self.compile_subexpr(field.value));
            update_names.push(field.name.to_string());
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        let node = self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn: CF_RECORD_EXTEND, // record_extend
        });
        self.graph.set_record_extend_info(node, RecordExtendInfo { update_names });
        node
    }

    /// Compile an atomic construction expression `atomic expr`.
    ///
    /// Single-input node; at runtime wraps the value as an AtomicValue (an atomic container sharing the underlying memory).
    pub(super) fn compile_atomic(&mut self, operand: crate::ast::Ast::ExprId) -> NodeId {
        let operand_node = self.compile_subexpr(operand);
        let inputs_offset = self.graph.inputs_pool.push(&[operand_node]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_ATOMIC_CONSTRUCT, // atomic_construct
        })
    }

    /// Compile an array construction expression.
    /// Allocation sites marked non-escaping by the analyzer use the stack-alloc compute_fn (289).
    /// When `fill` is present (`[value, ..count]`), uses compute_array_fill (321).
    pub(super) fn compile_array_lit(&mut self, expr_id: crate::ast::Ast::ExprId, elements: &[crate::ast::Ast::ExprRef], fill: Option<(crate::ast::Ast::ExprRef, crate::ast::Ast::ExprRef)>) -> NodeId {
        if let Some((value, count)) = fill {
            let val_node = self.compile_subexpr(value);
            let count_node = self.compile_subexpr(count);
            let inputs = [val_node, count_node];
            let inputs_offset = self.graph.inputs_pool.push(&inputs);
            return self.graph.add_node(Node {
                kind: NodeKind::BinOp,
                input_count: 2,
                inputs_offset,
                compute_fn: CF_ARRAY_FILL,
            });
        }
        let mut inputs = Vec::with_capacity(elements.len());
        for &elem in elements {
            inputs.push(self.compile_subexpr(elem));
        }
        let inputs_offset = self.graph.inputs_pool.push(&inputs);
        // Stack-alloc marker: non-escaping allocations use compute_array_construct_stack (289)
        let compute_fn = if self.should_stack_alloc(expr_id) {
            CF_ARRAY_CONSTRUCT_STACK // array_construct_stack
        } else {
            CF_ARRAY_CONSTRUCT // array_construct
        };
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn,
        })
    }

}
