//! Assign — Assignment lowering: ident / assign / compound assign. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Compile an `Expr::Ident(name)` reference: local var, captured/outer var, implicit-this
    /// field access, global load, or nullary ADT/type constructor.
    pub(super) fn compile_ident(
        &mut self,
        expr_id: crate::ast::Ast::ExprId,
        name: &str,
    ) -> NodeId {
        // Static reference rebinding: after `&x` registered this binding, name
        // reads route through the shared Cell (CF_DEREF_READ on the RefOf
        // node), so `*r = v` writes are observed in source order.
        // `current_effect` is appended as a DIRECT input (scheduler-ordering
        // only — the compute fn reads inputs[0]): a CF_SEQ wrapper would not
        // stop the read from firing before a prior cell/deref write (the SEQ
        // merely waits for both, it does not order input computation). Same
        // pattern as `compile_global_load`.
        if let Some(entry) = self.ref_rebind_active(name) {
            let (input_count, inputs_offset) = match self.current_effect {
                Some(eff) => (2, self.graph.inputs_pool.push(&[entry.ref_node, eff])),
                None => (1, self.graph.inputs_pool.push(&[entry.ref_node])),
            };
            return self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count,
                inputs_offset,
                compute_fn: CF_DEREF_READ,
            });
        }
        match self.lookup_var(name) {
            Some(node_id) => {
                // When `current_effect` exists, create a CF_SEQ dependency node to ensure the
                // variable read executes only after prior side effects complete.
                // This prevents an expression from reading a stale value before a WriteBack in
                // a while/loop subgraph updates it.
                // Consistent with the `current_effect` dependency mechanism in
                // `compile_global_load`.
                match self.current_effect {
                    Some(eff) => self.chain_effects(Some(eff), node_id),
                    None => node_id,
                }
            }
            None => {
                // Implicit this: bare identifier resolved to an instance field by sema.
                // Sema marks such accesses on `ExprInfo.implicit_this`; synthesize an explicit
                // `this.<field>` FieldAccess node. The method variant is handled in the Call
                // branch (it needs the argument list).
                if let Some(access) = self.expr_implicit_this(expr_id).cloned() {
                    if let crate::sema::Sema::ImplicitThisAccess::Field(field) = access {
                        let this_node = self
                            .lookup_var("this")
                            .expect("this binding must exist in method body");
                        return self.build_implicit_field_access(this_node, &field);
                    }
                    // Method variant handled in Call branch.
                }
                match self.lookup_global_var(name) {
                    Some(slot) => self.compile_global_load(slot),
                    None => {
                        // Nullary ADT / type constructor detection: when an Ident is neither a
                        // local nor a global variable, check whether it is a nullary constructor
                        // (e.g. `Lt`/`Leaf`/`Red`) and compile it as a nullary construct node.
                        // Parameterized constructors (non-empty `field_names`) are not handled
                        // here (they go through the `Call` path with arguments).
                        // A newtype always has an inner value, so it can never be nullary.
                        let tf_info = self.lookup_constructor_field_names(name)
                            .or_else(|| self.lookup_type_field_names(name));
                        match tf_info {
                            Some(info) if info.field_names.is_empty() && info.kind != RecordLitKind::Newtype => {
                                let inputs_offset = self.graph.inputs_pool.push(&[]);
                                let node = self.graph.add_node(Node {
                                    kind: NodeKind::BinOp,
                                    input_count: 0,
                                    inputs_offset,
                                    compute_fn: CF_RECORD_CONSTRUCT, // record_construct
                                });
                                self.graph.set_record_lit_info(node, RecordLitInfo {
                                    type_name: info.type_name.clone(),
                                    field_names: Vec::new(),
                                    constructor: name.to_string(),
                                    kind: info.kind,
                                });
                                node
                            }
                            _ => self.compile_const(),
                        }
                    }
                }
            }
        }
    }

    /// Compile an `Expr::Assign { target, value }` expression.
    ///
    /// Used for assignments in expression contexts such as defer bodies.
    /// Consistent with the `Ident` logic of `Stmt::Assignment`:
    ///   captured variable -> WriteBack; outer variable -> WriteBack;
    ///   global variable -> global_store; local -> bind_var.
    pub(super) fn compile_assign(
        &mut self,
        target: crate::ast::Ast::ExprId,
        value: crate::ast::Ast::ExprId,
    ) -> NodeId {
        let raw_val = self.compile_subexpr(value);
        let val_node = self.chain_effects(self.current_effect, raw_val);
        let target_expr = &self.current_module().arena.expr(target).node;
        match target_expr {
            crate::ast::Ast::Expr::Ident(name) => {
                // Static reference rebinding (mirror of Stmt::Assignment): `x = v`
                // where `&x` registered this binding writes through the shared
                // Cell; no rebind (reads route through the cell). Effect rides
                // as a direct trailing input (scheduler ordering only).
                if let Some(entry) = self.ref_rebind_active(name) {
                    let (input_count, inputs_offset) = match self.current_effect {
                        Some(eff) => (3, self.graph.inputs_pool.push(&[entry.ref_node, val_node, eff])),
                        None => (2, self.graph.inputs_pool.push(&[entry.ref_node, val_node])),
                    };
                    let write_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count,
                        inputs_offset,
                        compute_fn: CF_DEREF_WRITE,
                    });
                    self.current_effect = Some(write_node);
                    return self.compile_void_const();
                }
                // Implicit-this field assignment: `field = value` inside a method body
                // resolves to `this.field = value`. Without this, the bare name would
                // create a local binding instead of mutating the instance field.
                if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(target).cloned() {
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
                    // Bug #99: chain the field set into the effect order so later reads
                    // of the field in the same method execute after the write (e.g. the
                    // exponent `if` reading `pos` after the fraction loop advanced it).
                    self.current_effect = Some(self.chain_effects(self.current_effect, set_node));
                    return self.compile_void_const();
                }
                let captured_source = self.captured_scopes.iter().rev()
                    .find_map(|scope| scope.iter()
                        .find(|(n, _)| n.as_str() == *name)
                        .map(|(_, node)| *node));
                if let Some(source) = captured_source {
                    let wb_node = self.compile_writeback_node(val_node, source);
                    self.bind_var(name, val_node);
                    self.current_effect = Some(wb_node);
                } else if let Some(outer_node) = self.lookup_var(name) {
                    if !self.is_in_current_subgraph(outer_node) {
                        let wb_node = self.compile_writeback_node(val_node, outer_node);
                        self.bind_var(name, val_node);
                        self.current_effect = Some(wb_node);
                    } else if let Some(&captured_node) = self.captured_vars.get(*name) {
                        let wb_node = self.compile_writeback_node(val_node, captured_node);
                        self.bind_var(name, val_node);
                        self.current_effect = Some(wb_node);
                    } else {
                        self.bind_var(name, val_node);
                    }
                } else if let Some(slot) = self.lookup_global_var(name) {
                    let store_node = self.compile_global_store(val_node, slot);
                    self.current_effect = Some(store_node);
                } else {
                    self.bind_var(name, val_node);
                }
            }
            crate::ast::Ast::Expr::FieldAccess { recv: obj, field } => {
                let obj_node = self.compile_subexpr(*obj);
                let off = self.graph.inputs_pool.push(&[obj_node, val_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                self.current_effect = Some(self.chain_effects(self.current_effect, set_node));
            }
            // `recv?.field = value`: skip the assignment when `obj` is null.
            crate::ast::Ast::Expr::SafeAccess { recv: obj, field } => {
                let obj_node = self.compile_subexpr(*obj);
                let off = self.graph.inputs_pool.push(&[obj_node, val_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                self.graph.set_safe_op(set_node);
                self.current_effect = Some(self.chain_effects(self.current_effect, set_node));
            }
            // `*ref = value` → compute_deref_write(282)
            crate::ast::Ast::Expr::Deref(ref_inner) => {
                let ref_node = self.compile_subexpr(*ref_inner);
                let off = self.graph.inputs_pool.push(&[ref_node, val_node]);
                let write_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: CF_DEREF_WRITE, // compute_deref_write
                });
                self.current_effect = Some(self.chain_effects(self.current_effect, write_node));
            }
            _ => {}
        }
        self.compile_void_const()
    }

    /// Compile an `Expr::CompoundAssign { op, target, value }` expression: `target op= value`.
    pub(super) fn compile_compound_assign(
        &mut self,
        op: crate::ast::Ast::CompoundAssignOp,
        target: crate::ast::Ast::ExprId,
        value: crate::ast::Ast::ExprId,
    ) -> NodeId {
        let val_node = self.compile_subexpr(value);
        let target_expr = &self.current_module().arena.expr(target).node;
        let bin_compute = self.compound_assign_op_to_compute_fn(op, target);
        match target_expr {
            crate::ast::Ast::Expr::Ident(name) => {
                // Implicit-this field compound assignment: `field op= value` inside a
                // method body resolves to `this.field op= value`.
                if let Some(crate::sema::Sema::ImplicitThisAccess::Field(field)) = self.expr_implicit_this(target).cloned() {
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
                    // Operation.
                    let bin_off = self.graph.inputs_pool.push(&[get_node, val_node]);
                    let result_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: bin_off,
                        compute_fn: bin_compute,
                    });
                    // Write back.
                    let set_off = self.graph.inputs_pool.push(&[this_node, result_node]);
                    let set_node = self.graph.add_node(Node {
                        kind: NodeKind::BinOp,
                        input_count: 2,
                        inputs_offset: set_off,
                        compute_fn: CF_RECORD_FIELD_SET,
                    });
                    self.graph.set_field_set_name(set_node, field.to_string());
                    return result_node;
                }
                let cur_node = self
                    .lookup_var(name)
                    .unwrap_or_else(|| self.compile_placeholder());
                let off = self.graph.inputs_pool.push(&[cur_node, val_node]);
                let result_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: off,
                    compute_fn: bin_compute,
                });
                self.bind_var(name, result_node);
                result_node
            }
            crate::ast::Ast::Expr::FieldAccess { recv: obj, field }
            | crate::ast::Ast::Expr::SafeAccess { recv: obj, field } => {
                let obj_node = self.compile_subexpr(*obj);
                // Read the current field value.
                let get_off = self.graph.inputs_pool.push(&[obj_node]);
                let get_node = self.graph.add_node(Node {
                    kind: NodeKind::FieldAccess,
                    input_count: 1,
                    inputs_offset: get_off,
                    compute_fn: CF_RECORD_FIELD_GET, // record_field_get
                });
                // The field_get node needs the field name metadata to know which field to extract.
                // compute_record_field_get reads the name via field_set_name (same metadata as field_set).
                self.graph.set_field_set_name(get_node, field.to_string());
                // Operation.
                let bin_off = self.graph.inputs_pool.push(&[get_node, val_node]);
                let result_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: bin_off,
                    compute_fn: bin_compute,
                });
                // Write back. The set_node MUST be chained into the effect graph,
                // otherwise DCE drops it and the field mutation never executes.
                let set_off = self.graph.inputs_pool.push(&[obj_node, result_node]);
                let set_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: set_off,
                    compute_fn: CF_RECORD_FIELD_SET, // record_field_set
                });
                self.graph.set_field_set_name(set_node, field.to_string());
                self.chain_effects(self.current_effect, set_node)
            }
            // `*ref op= value` -> read Cell + operation + write back to Cell.
            crate::ast::Ast::Expr::Deref(ref_inner) => {
                let ref_node = self.compile_subexpr(*ref_inner);
                // Read the current value: compute_deref_read (281).
                let read_off = self.graph.inputs_pool.push(&[ref_node]);
                let read_node = self.graph.add_node(Node {
                    kind: NodeKind::UnOp,
                    input_count: 1,
                    inputs_offset: read_off,
                    compute_fn: CF_DEREF_READ,
                });
                // Operation.
                let bin_off = self.graph.inputs_pool.push(&[read_node, val_node]);
                let result_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: bin_off,
                    compute_fn: bin_compute,
                });
                // Write back to Cell: compute_deref_write (282).
                let write_off = self.graph.inputs_pool.push(&[ref_node, result_node]);
                let _write_node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset: write_off,
                    compute_fn: CF_DEREF_WRITE,
                });
                result_node
            }
            _ => self.compile_void_const(),
        }
    }

}
