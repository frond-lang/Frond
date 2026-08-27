//! Ops — Operator selection and binary / unary / short-circuit / cast lowering. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Compile a bool AND node (used for guard conditions pattern && guard).
    pub(super) fn compile_bool_and(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[lhs, rhs]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: CF_AND_BOOL, // and_bool
        })
    }

    /// Compile a bool OR node (used for or-patterns p1 | p2).
    pub(super) fn compile_bool_or(&mut self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let off = self.graph.inputs_pool.push(&[lhs, rhs]);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: 2,
            inputs_offset: off,
            compute_fn: CF_OR_BOOL, // or_bool
        })
    }

    /// Type family: returns the `TypeFamily` (caller merges integer variants with `|` to dispatch by bit-width).
    /// i8/i16/u8/u16/u32/char -> SignedInt32/UnsignedInt32/Char; i64/u64/isize/usize -> SignedInt64/UnsignedInt64;
    /// i128/u128 -> SignedInt128/UnsignedInt128; bool -> Bool; floats -> Float.
    pub(super) fn int_family(ty_name: &str) -> crate::types::TypeFamily {
        match crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta) {
            Some(m) => m.family,
            None => crate::types::TypeFamily::SignedInt32, // unknown integer type falls back to the Int32 path
        }
    }

    /// Get TypeFamily from a type name (including non-scalar built-in types like Str).
    /// Unlike int_family, this method uses ValueTag::from_name + family() directly,
    /// bypassing scalar_meta, so for str it returns TypeFamily::Str instead of falling back to SignedInt32.
    pub(super) fn type_family(ty_name: &str) -> crate::types::TypeFamily {
        match crate::value::ValueTag::from_name(ty_name) {
            Some(tag) => tag.family(),
            None => crate::types::TypeFamily::SignedInt32,
        }
    }

    /// Arithmetic/bitwise compute_fn table lookup: returns the arithmetic base by type name.
    /// Integer types: 12 consecutive indices each (add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot);
    /// float types: 6 consecutive indices each (add/sub/mul/div/mod/neg, no bitwise ops).
    /// Returns None when the type does not support arithmetic.
    /// The base comes from `scalar_meta`, kept in single-point sync with the compute_fn_table! indices.
    pub(super) fn arith_base(ty_name: &str) -> Option<u32> {
        crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta).map(|m| m.arith_base)
    }

    /// Select a compute_fn id by op + expression type.
    pub(super) fn select_binary_compute_fn(
        &mut self,
        op: crate::ast::Ast::BinaryOp,
        binary_expr_id: crate::ast::Ast::ExprId,
        lhs_expr: crate::ast::Ast::ExprId,
        _rhs_expr: crate::ast::Ast::ExprId,
    ) -> ComputeFnId {
        // Consume the sema-promoted type: binary_expr_id's ExprInfo.type_name is the binary
        // operation's result type inferred by sema. For arithmetic, the result type is the promoted
        // operand type (i32+f64 -> f64); for comparisons, the result type is bool, so the operand type
        // must be used to select the compute_fn.
        // Check in two steps to avoid borrow conflicts: first check whether the lhs type exists and report errors, then get the type reference
        let has_lhs_ty = self.expr_type_name(lhs_expr).is_some();
        if !has_lhs_ty {
            self.errors.push(format!(
                "internal: missing ExprInfo for expr {:?} in binary_op", lhs_expr));
        }
        let lhs_ty = self.expr_type_name(lhs_expr).unwrap_or("i32");
        let ty_name = match self.expr_type_name(binary_expr_id) {
            Some(t) if Self::type_family(t) == crate::types::TypeFamily::Bool => lhs_ty,  // comparison: use operand type
            Some(t) => t,             // arithmetic: use promoted type
            None => lhs_ty,           // no sema record: fall back to lhs type
        };
        let ty_meta = crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta);
        let is_float = ty_meta.as_ref().map(|m| m.is_float).unwrap_or(false);
        // f128 needs a dedicated comparison path: going through to_f64 drops 60 bits of precision, causing distinct f128 values to be misjudged as equal
        let is_f128 = crate::value::ValueTag::from_name(ty_name) == Some(crate::value::ValueTag::F128);
        // is_int: non-float and non-bool (reuses the TypeFamily enum, eliminating string comparison)
        let is_int = !is_float && Self::int_family(ty_name) != crate::types::TypeFamily::Bool;
        let base = Self::arith_base(ty_name);

        // Elvis (??) operator: returns rhs when lhs is null, otherwise lhs.
        // Does not depend on operand types; must be handled before the str/composite-type branches.
        if matches!(op, crate::ast::Ast::BinaryOp::Elvis) {
            return CF_ELVIS;
        }

        // ==/!= for nullable types: ?. short-circuit or null literals produce Value::Null,
        // and dedicated comparison functions for str/i32 etc. do not handle Null (heap_obj() returns None, always false).
        // Dispatch to CF_EQ_OBJ/CF_NE_OBJ (value_equals_with_arena correctly handles Null discrimination).
        if matches!(op, crate::ast::Ast::BinaryOp::Eq | crate::ast::Ast::BinaryOp::NotEq)
            && self.expr_is_nullable(lhs_expr)
        {
            return match op {
                crate::ast::Ast::BinaryOp::Eq => CF_EQ_OBJ,
                crate::ast::Ast::BinaryOp::NotEq => CF_NE_OBJ,
                _ => unreachable!(),
            };
        }

        // str + str -> string concatenation (compute_str_concat, 269)
        if Self::type_family(ty_name) == crate::types::TypeFamily::Str
            && matches!(op, crate::ast::Ast::BinaryOp::Add)
        {
            return CF_STR_CONCAT;
        }

        // str comparison -> dedicated str comparison compute_fn (292-297)
        // Does not go through the i32 path: str has no as_i32 semantics; the i32 path would always be 0, yielding wrong results
        if Self::type_family(ty_name) == crate::types::TypeFamily::Str {
            return match op {
                crate::ast::Ast::BinaryOp::Eq => CF_EQ_STR,
                crate::ast::Ast::BinaryOp::NotEq => CF_NE_STR,
                crate::ast::Ast::BinaryOp::Lt => CF_LT_STR,
                crate::ast::Ast::BinaryOp::Gt => CF_GT_STR,
                crate::ast::Ast::BinaryOp::LtEq => CF_LE_STR,
                crate::ast::Ast::BinaryOp::GtEq => CF_GE_STR,
                _ => CF_EQ_STR, // arithmetic etc. already handled above; unreachable here
            };
        }

        // ==/!= for composite types (record/adt/newtype/array/closure/throw etc.) ->
        // generic semantic comparison compute_fn (298-299). Going through the i32 path would make
        // as_i32() always 0, judging all composite types as equal.
        // Rationale: scalar_meta being None means a non-scalar type. At this point Str and Nullable
        // are already handled above; the remaining None cases are all composite types (Array/Ref/Fn/Adt/Record/...).
        // scalar_meta is the single source of truth for scalar types, so is_none() is the necessary and sufficient condition for composite types.
        if matches!(op, crate::ast::Ast::BinaryOp::Eq | crate::ast::Ast::BinaryOp::NotEq)
            && ty_meta.is_none()
        {
            return match op {
                crate::ast::Ast::BinaryOp::Eq => CF_EQ_OBJ,
                crate::ast::Ast::BinaryOp::NotEq => CF_NE_OBJ,
                _ => unreachable!(),
            };
        }

        // Arithmetic (add/sub/mul/div/mod): supported by both integers and floats; look up by concrete type
        // Integer index order: add(0) sub(1) mul(2) div(3) mod(4) bitand(5) bitor(6) bitxor(7) shl(8) shr(9) neg(10) bitnot(11)
        // Float index order: add(0) sub(1) mul(2) div(3) mod(4) neg(5)
        let arith_offset = |op: &crate::ast::Ast::BinaryOp| -> Option<u32> {
            match op {
                crate::ast::Ast::BinaryOp::Add => Some(0),
                crate::ast::Ast::BinaryOp::Sub => Some(1),
                crate::ast::Ast::BinaryOp::Mul => Some(2),
                crate::ast::Ast::BinaryOp::Div => Some(3),
                crate::ast::Ast::BinaryOp::Mod => Some(4),
                _ => None,
            }
        };
        if let Some(off) = arith_offset(&op) {
            if let Some(b) = base {
                return ComputeFnId(b + off);
            }
            // Unknown type falls back to the i32 path. CAUTION: this is
            // load-bearing — monomorphized generic arithmetic routinely lands
            // here with junk type names ('offset', '_', soft typevars), and
            // as_i32/as_int reads the runtime int values correctly for the
            // 32-bit range. Erroring here breaks the stdlib build; the real
            // fix is hardening soft types so concrete arithmetic never
            // reaches this fallback (floats/large i64s silently truncate).
            return ComputeFnId(114 + off); // i32 full-family base (renumbered 2026-08-22)
        }

        // Bitwise (bitand/bitor/bitxor/shl/shr): only supported by integers
        if is_int {
            let bit_offset = match op {
                crate::ast::Ast::BinaryOp::BitAnd => Some(5),
                crate::ast::Ast::BinaryOp::BitOr => Some(6),
                crate::ast::Ast::BinaryOp::BitXor => Some(7),
                crate::ast::Ast::BinaryOp::Shl => Some(8),
                crate::ast::Ast::BinaryOp::Shr => Some(9),
                _ => None,
            };
            if let Some(off) = bit_offset {
                if let Some(b) = base {
                    return ComputeFnId(b + off);
                }
                return ComputeFnId(CF_ADD_I32_FULL.0 + off); // fall back to i32 (see arithmetic note above)
            }
        }

        // Comparison: result is bool; input read by type family
        // fam is the TypeFamily enum; use | to merge signed/unsigned integer variants to dispatch by bit-width (compiler exhaustive check)
        let fam = Self::int_family(ty_name);
        use crate::types::TypeFamily;
        // u128 comparisons get dedicated computes: the i128 domain cannot hold
        // the upper half of u128 (bit-reinterpretation inverts the ordering
        // above 2^127). Routed before the cmp_arm cascade.
        if matches!(fam, TypeFamily::UnsignedInt128) {
            match op {
                crate::ast::Ast::BinaryOp::Eq => return CF_EQ_U128,
                crate::ast::Ast::BinaryOp::NotEq => return CF_NE_U128,
                crate::ast::Ast::BinaryOp::Lt => return CF_LT_U128,
                crate::ast::Ast::BinaryOp::Gt => return CF_GT_U128,
                crate::ast::Ast::BinaryOp::LtEq => return CF_LE_U128,
                crate::ast::Ast::BinaryOp::GtEq => return CF_GE_U128,
                _ => {}
            }
        }
        // The 6 comparison ops share an f128->float->(bool)->i128->i64->i32 cascade; a macro removes the repetition.
        // Eq/NotEq have a Bool branch; Lt/Gt/LtEq/GtEq have no Bool branch (bool cannot be ordered).
        // The macro only expands the cascade block (=> right side); match patterns stay explicit to preserve the compiler's exhaustive check.
        macro_rules! cmp_arm {
            ($f128:ident, $f64:ident, $bool:ident, $i128:ident, $i64:ident, $i32:ident) => {
                if is_f128 { $f128 }
                else if is_float { $f64 }
                else if fam == TypeFamily::Bool { $bool }
                // Unsigned 64-bit (u64/usize): compare in the I128 domain —
                // zero-extension preserves values, while the signed I64
                // compares misread the sign bit (values above i63::MAX came
                // out negative, e.g. `u >= 0` was false for a large usize).
                // Eq/NotEq are bit-pattern exact in I128 too.
                else if matches!(fam, TypeFamily::UnsignedInt64) { $i128 }
                else if matches!(fam, TypeFamily::SignedInt128) { $i128 }
                else if matches!(fam, TypeFamily::SignedInt64) { $i64 }
                else { $i32 }
            };
            ($f128:ident, $f64:ident, $i128:ident, $i64:ident, $i32:ident) => {
                if is_f128 { $f128 }
                else if is_float { $f64 }
                else if matches!(fam, TypeFamily::UnsignedInt64) { $i128 }
                else if matches!(fam, TypeFamily::SignedInt128) { $i128 }
                else if matches!(fam, TypeFamily::SignedInt64) { $i64 }
                else { $i32 }
            };
        }
        match op {
            crate::ast::Ast::BinaryOp::Eq => cmp_arm!(CF_EQ_F128, CF_EQ_F64, CF_EQ_BOOL, CF_EQ_I128, CF_EQ_I64, CF_EQ_I32),
            crate::ast::Ast::BinaryOp::NotEq => cmp_arm!(CF_NE_F128, CF_NE_F64, CF_NE_BOOL, CF_NE_I128, CF_NE_I64, CF_NE_I32),
            crate::ast::Ast::BinaryOp::Lt => cmp_arm!(CF_LT_F128, CF_LT_F64, CF_LT_I128, CF_LT_I64, CF_LT_I32),
            crate::ast::Ast::BinaryOp::Gt => cmp_arm!(CF_GT_F128, CF_GT_F64, CF_GT_I128, CF_GT_I64, CF_GT_I32),
            crate::ast::Ast::BinaryOp::LtEq => cmp_arm!(CF_LE_F128, CF_LE_F64, CF_LE_I128, CF_LE_I64, CF_LE_I32),
            crate::ast::Ast::BinaryOp::GtEq => cmp_arm!(CF_GE_F128, CF_GE_F64, CF_GE_I128, CF_GE_I64, CF_GE_I32),
            crate::ast::Ast::BinaryOp::And => CF_AND_BOOL, // and_bool
            crate::ast::Ast::BinaryOp::Or => CF_OR_BOOL,  // or_bool
            crate::ast::Ast::BinaryOp::RefEq => CF_REF_EQ,          // ref_eq
            crate::ast::Ast::BinaryOp::RefNeq => CF_REF_NEQ,         // ref_neq
            crate::ast::Ast::BinaryOp::ConcatList => CF_CONCAT_LIST,     // concat_list
            crate::ast::Ast::BinaryOp::Range => CF_RANGE,          // range
            crate::ast::Ast::BinaryOp::RangeInclusive => CF_RANGE_INCLUSIVE, // range_inclusive
            crate::ast::Ast::BinaryOp::Elvis => CF_ELVIS,          // elvis
            _ => CF_NOOP,
        }
    }

    /// Select a unary operation compute_fn id by op + operand expression type.
    pub(super) fn select_unary_compute_fn(
        &mut self,
        op: crate::ast::Ast::UnaryOp,
        operand_expr: crate::ast::Ast::ExprId,
    ) -> ComputeFnId {
        let ty_name = self.expr_type_name_checked(operand_expr, "unary_op");
        let is_float = crate::value::ValueTag::from_name(ty_name).and_then(scalar_meta).map(|m| m.is_float).unwrap_or(false);
        let base = Self::arith_base(ty_name);
        match op {
            crate::ast::Ast::UnaryOp::Not => CF_NOT_BOOL, // not_bool
            crate::ast::Ast::UnaryOp::Neg => {
                // integer neg is at base+10; float neg is at base+5
                if let Some(b) = base {
                    let off = if is_float { 5 } else { 10 };
                    return ComputeFnId(b + off);
                }
                CF_NEG_I32_FULL // fall back to neg_i32
            }
            crate::ast::Ast::UnaryOp::BitNot => {
                // integers only; bitnot is at base+11
                if let Some(b) = base {
                    return ComputeFnId(b + 11);
                }
                CF_BITNOT_I32_FULL // fall back to bitnot_i32
            }
        }
    }

    /// Compile a binary operation.
    pub(super) fn compile_binary(
        &mut self,
        op: crate::ast::Ast::BinaryOp,
        binary_expr_id: crate::ast::Ast::ExprId,
        lhs: crate::ast::Ast::ExprId,
        rhs: crate::ast::Ast::ExprId,
    ) -> NodeId {
        // Range/RangeInclusive compiled as a range_iter(start, end, inclusive) function call
        // (Range itself is an iterator; the For loop statically dispatches via RangeIterator.next)
        match op {
            crate::ast::Ast::BinaryOp::Range | crate::ast::Ast::BinaryOp::RangeInclusive => {
                let lhs_node = self.compile_subexpr(lhs);
                let rhs_node = self.compile_subexpr(rhs);
                let inclusive = matches!(op, crate::ast::Ast::BinaryOp::RangeInclusive);
                let bool_node = self.compile_bool_const(inclusive);
                // RangeIterator is i64-based; operands of any integer width are
                // widened explicitly here (implicit promotion was removed
                // language-wide, Bug #60).
                let lhs_node = self.make_i64_cast_node(lhs_node);
                let rhs_node = self.make_i64_cast_node(rhs_node);
                self.make_call_by_name("range_iter", &[lhs_node, rhs_node, bool_node])
            }
            // Bug #38: &&/|| short-circuit evaluation -- lowered to a Gate conditional branch, ensuring RHS is
            // evaluated only when LHS does not satisfy the short-circuit condition (same conditional dataflow as the if expression).
            //   lhs && rhs  =>  if lhs { rhs } else { false }
            //   lhs || rhs  =>  if lhs { true } else { rhs }
            crate::ast::Ast::BinaryOp::And | crate::ast::Ast::BinaryOp::Or => {
                self.compile_short_circuit(op, lhs, rhs)
            }
            _ => {
                // str + non-str / non-str + str -> convert the non-string operand to a string via
                // compute_reflect_format, then concat with str_concat
                // (same lowering path as string interpolation "{expr}")
                if matches!(op, crate::ast::Ast::BinaryOp::Add) {
                    let lhs_ty = self.expr_type_name(lhs).unwrap_or("");
                    let rhs_ty = self.expr_type_name(rhs).unwrap_or("");
                    let lhs_is_str = Self::type_family(lhs_ty) == crate::types::TypeFamily::Str;
                    let rhs_is_str = Self::type_family(rhs_ty) == crate::types::TypeFamily::Str;
                    if lhs_is_str || rhs_is_str {
                        let lhs_node = self.compile_subexpr(lhs);
                        let rhs_node = self.compile_subexpr(rhs);
                        let lhs_final = if lhs_is_str {
                            lhs_node
                        } else {
                            self.make_reflect_format_node(lhs_node)
                        };
                        let rhs_final = if rhs_is_str {
                            rhs_node
                        } else {
                            self.make_reflect_format_node(rhs_node)
                        };
                        let inputs_offset =
                            self.graph.inputs_pool.push(&[lhs_final, rhs_final]);
                        return self.graph.add_node(Node {
                            kind: NodeKind::BinOp,
                            input_count: 2,
                            inputs_offset,
                            compute_fn: CF_STR_CONCAT,
                        });
                    }
                }
                // Operands are not in tail position: their values are consumed by the operation node, not returned directly.
                let lhs_node = self.compile_subexpr(lhs);
                let rhs_node = self.compile_subexpr(rhs);
                let inputs_offset = self.graph.inputs_pool.push(&[lhs_node, rhs_node]);
                let compute_fn = self.select_binary_compute_fn(op, binary_expr_id, lhs, rhs);
                let node = self.graph.add_node(Node {
                    kind: NodeKind::BinOp,
                    input_count: 2,
                    inputs_offset,
                    compute_fn,
                });
                // Compile-time SIMD batching marker: scalar type + op -> runtime batches by (tag, op) group
                if let Some(info) = self.binary_batch_info(op, lhs) {
                    self.graph.set_batch_info(node, info);
                }
                node
            }
        }
    }

    /// Bug #38: compile &&/|| short-circuit evaluation.
    ///
    /// Uses a Gate conditional branch to ensure RHS is evaluated only when LHS does not satisfy the short-circuit condition:
    ///   lhs && rhs  =>  if lhs { rhs } else { false }
    ///   lhs || rhs  =>  if lhs { true } else { rhs }
    ///
    /// Consistent with compile_if's Gate pattern: cond_node + then_sg + else_sg.
    /// The then/else branch bodies are Const nodes (short-circuit value) or the RHS expression (the branch needing evaluation).
    pub(super) fn compile_short_circuit(
        &mut self,
        op: crate::ast::Ast::BinaryOp,
        lhs: crate::ast::Ast::ExprId,
        rhs: crate::ast::Ast::ExprId,
    ) -> NodeId {
        let cond_node = self.compile_subexpr(lhs);
        let is_and = matches!(op, crate::ast::Ast::BinaryOp::And);
        // `compile_branch_subgraph` does not isolate `current_effect` (same
        // contract as `compile_if`): an effectful RHS body leaves it pointing
        // at branch-internal nodes, and the gate below must never reference
        // those — its pending lives in the PARENT frame, where a child-sg
        // node's completion can never be observed (silent stall). Save/restore
        // around both branch compiles.
        let prev_effect = self.current_effect;
        // && : lhs=true -> evaluate rhs ; lhs=false -> false (short-circuit)
        // || : lhs=true -> true (short-circuit)   ; lhs=false -> evaluate rhs
        let (then_sg, then_inputs) = if is_and {
            let r = self.compile_branch_subgraph(rhs);
            self.current_effect = prev_effect;
            r
        } else {
            self.compile_bool_branch(true)
        };
        let (else_sg, else_inputs) = if is_and {
            self.compile_bool_branch(false)
        } else {
            let r = self.compile_branch_subgraph(rhs);
            self.current_effect = prev_effect;
            r
        };
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
                capture: false,
                condition_input: cond_node,
                branches: vec![
                    (true, then_sg, then_inputs),
                    (false, else_sg, else_inputs),
                ],
            },
        );
        gate_node
    }

    /// Compile a constant bool branch (short-circuit value), used for &&'s false branch and ||'s true branch.
    pub(super) fn compile_bool_branch(&mut self, value: bool) -> (SubGraphId, Vec<NodeId>) {
        let node_start = self.graph.nodes.len() as u32;
        self.enter_scope();
        let prev_sg_start = self.current_sg_start;
        self.current_sg_start = node_start;
        let return_node = self.compile_bool_const(value);
        self.current_sg_start = prev_sg_start;
        self.exit_scope();
        let node_end = self.graph.nodes.len() as u32;
        let sg_id = SubGraphId(self.graph.subgraphs.len() as u32);
        self.graph.add_subgraph(SubGraph {
            converter_generated: false,
            id: sg_id,
            node_range: (NodeId(node_start), NodeId(node_end)),
            param_count: 0,
            entry_node: NodeId(node_start),
            return_node,
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: crate::ir::Ir::LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: self.current_function_sg
                .map(|sg| sg.0)
                .unwrap_or(0),
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        (sg_id, Vec::new())
    }

    /// Map a Frond BinaryOp + type name to a BatchInfo (batchable op + scalar type combination).
    ///
    /// Returns None when the op is not SIMD-batchable (e.g. And/Or/RefEq/ConcatList/Range
    /// and other non-scalar arithmetic ops, or non-scalar types).
    pub(super) fn binary_batch_info(
        &self,
        op: crate::ast::Ast::BinaryOp,
        lhs_expr: crate::ast::Ast::ExprId,
    ) -> Option<BatchInfo> {
        use crate::ast::Ast::BinaryOp;
        use crate::value::{BinOp as VBinOp, CmpOp as VCmpOp};

        let ty = self.expr_type_name(lhs_expr)?;
        let tag = Self::ty_name_to_scalar_tag(ty)?;
        let is_float = scalar_meta(tag).map(|m| m.is_float).unwrap_or(false);

        let batch_op = match op {
            BinaryOp::Add => BatchOp::Bin(VBinOp::Add),
            BinaryOp::Sub => BatchOp::Bin(VBinOp::Sub),
            BinaryOp::Mul => BatchOp::Bin(VBinOp::Mul),
            BinaryOp::Div => BatchOp::Bin(VBinOp::Div),
            BinaryOp::Mod => BatchOp::Bin(VBinOp::Mod),
            BinaryOp::BitAnd if !is_float => BatchOp::Bin(VBinOp::Band),
            BinaryOp::BitOr if !is_float => BatchOp::Bin(VBinOp::Bor),
            BinaryOp::BitXor if !is_float => BatchOp::Bin(VBinOp::Bxor),
            BinaryOp::Shl if !is_float => BatchOp::Bin(VBinOp::Shl),
            BinaryOp::Shr if !is_float => BatchOp::Bin(VBinOp::Shr),
            BinaryOp::Eq => BatchOp::Cmp(VCmpOp::Eq),
            BinaryOp::NotEq => BatchOp::Cmp(VCmpOp::Ne),
            BinaryOp::Lt => BatchOp::Cmp(VCmpOp::Lt),
            BinaryOp::Gt => BatchOp::Cmp(VCmpOp::Gt),
            BinaryOp::LtEq => BatchOp::Cmp(VCmpOp::Le),
            BinaryOp::GtEq => BatchOp::Cmp(VCmpOp::Ge),
            // And/Or/RefEq/RefNeq/ConcatList/Range/RangeInclusive/Elvis -> not batchable
            _ => return None,
        };
        Some(BatchInfo { tag, op: batch_op })
    }

    /// Map a Frond UnaryOp + type name to a BatchInfo.
    ///
    /// Neg (integer/float negation) and BitNot (integer bitwise not) are batchable;
    /// Not (bool logical not) does not go through SIMD batching.
    pub(super) fn unary_batch_info(
        &self,
        op: crate::ast::Ast::UnaryOp,
        operand_expr: crate::ast::Ast::ExprId,
    ) -> Option<BatchInfo> {
        use crate::ast::Ast::UnaryOp;
        use crate::value::UnaryOp as VUnaryOp;

        let ty = self.expr_type_name(operand_expr)?;
        let tag = Self::ty_name_to_scalar_tag(ty)?;
        let is_float = scalar_meta(tag).map(|m| m.is_float).unwrap_or(false);

        let batch_op = match op {
            UnaryOp::Neg => BatchOp::Unary(VUnaryOp::Neg),
            UnaryOp::BitNot if !is_float => BatchOp::Unary(VUnaryOp::Bnot),
            // Not (bool logical not) -> not batchable
            _ => return None,
        };
        Some(BatchInfo { tag, op: batch_op })
    }

    /// Map a type name to a ValueTag (delegates to `ValueTag::from_name`, single-point sync with Value).
    pub(super) fn ty_name_to_scalar_tag(ty: &str) -> Option<crate::value::ValueTag> {
        crate::value::ValueTag::from_name(ty)
    }

    /// Wraps a node in an explicit scalar cast node targeting i64
    /// (used by range lowering: RangeIterator is i64-based).
    fn make_i64_cast_node(&mut self, input: NodeId) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[input]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_CAST_SCALAR, // compute_cast_scalar
        });
        self.graph.set_cast_target_type(node, "i64".to_string());
        node
    }

    /// Compile a type cast `expr as T`.
    ///
    /// Two codegen paths, both single-node:
    ///   - target is str: `compute_cast_to_str` (idx 277) — covers scalar/char/bool/array→str
    ///   - scalar→scalar: `compute_cast_scalar` (idx 278) — covers all int↔int/int↔float/char↔int
    /// Render an `as`-target TypeRef to a type-name string. Honors
    /// type-parameter substitution (`current_type_args`) and array suffixes
    /// (`u8[]`, nested `u8[][]`); other nodes fall back to "i64" (legacy).
    /// Nullable is NOT peeled here — only compile_as_cast peels it, and only
    /// at the top level: `i32?[]` (array of nullables) must stay distinct
    /// from `(i32[])?`, and its rendered element name ("i32?") fails
    /// ValueTag::from_name, giving the permissive reference-cast passthrough.
    fn render_cast_target_name(&self, ty: crate::ast::Ast::TypeRef) -> String {
        let s = &self.current_module().arena.types[ty.0 as usize];
        match &s.node {
            crate::ast::Ast::TypeNode::Nullable { inner } => {
                format!("{}?", self.render_cast_target_name(*inner))
            }
            crate::ast::Ast::TypeNode::Named { name } => {
                let name = *name;
                // Type-parameter replacement (monomorphization instance context)
                if let Some((_, h)) = self.current_type_args.iter().find(|(n, _)| n == name) {
                    if let Some(resolved) = self.type_arena.type_name_concrete(*h) {
                        resolved
                    } else {
                        name.to_string()
                    }
                } else {
                    name.to_string()
                }
            }
            crate::ast::Ast::TypeNode::Array { element_type, .. } => {
                format!("{}[]", self.render_cast_target_name(*element_type))
            }
            _ => "i64".to_string(),
        }
    }

    pub(super) fn compile_as_cast(
        &mut self,
        expr: crate::ast::Ast::ExprId,
        target: crate::ast::Ast::TypeRef,
    ) -> NodeId {
        // Get the target type name (array suffixes included: "u8[]", ...).
        // In a generic context, target may be a type-parameter name (e.g. "T"); look up
        // current_type_args to replace it with the concrete type name.
        // Nullable wrappers (`f32?`) peel to the inner scalar: the runtime Value
        // of a nullable scalar IS the scalar (null is the Null sentinel), so the
        // cast targets the base type and null passes through at runtime.
        // (Top-level peel only — see render_cast_target_name.)
        let target_ty = {
            let mut ty = target;
            loop {
                let s = &self.current_module().arena.types[ty.0 as usize];
                match &s.node {
                    crate::ast::Ast::TypeNode::Nullable { inner } => ty = *inner,
                    _ => break self.render_cast_target_name(ty),
                }
            }
        };

        // Get the source type name (from Sema expr_types; peel a trailing '?'
        // the same way so scalar-tag matching sees the base scalar).
        let source_ty = self
            .expr_type_name(expr)
            .unwrap_or("i64")
            .trim_end_matches('?')
            .to_string();

        let input = self.compile_subexpr(expr);

        // Path 1: any type -> str
        if Self::type_family(&target_ty) == crate::types::TypeFamily::Str {
            let inputs_offset = self.graph.inputs_pool.push(&[input]);
            return self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset,
                compute_fn: CF_CAST_TO_STR, // compute_cast_to_str
            });
        }

        // Path 2: scalar -> scalar (int<->int, int<->float, float<->float, bool<->int, char<->int)
        if Self::ty_name_to_scalar_tag(&source_ty).is_some()
            && Self::ty_name_to_scalar_tag(&target_ty).is_some()
        {
            let inputs_offset = self.graph.inputs_pool.push(&[input]);
            let node = self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset,
                compute_fn: CF_CAST_SCALAR, // compute_cast_scalar
            });
            self.graph.set_cast_target_type(node, target_ty.clone());
            return node;
        }

        // Path 3: array target (`x as u8[]`). Array targets used to fall into
        // the optimistic scalar-cast fallback below — non-Named targets were
        // misread as "i64", destroying the value (the cast result behaved as
        // an empty array). The array cast carries the value through and
        // converts scalar elements when the tags differ.
        if target_ty.ends_with("[]") {
            let inputs_offset = self.graph.inputs_pool.push(&[input]);
            let node = self.graph.add_node(Node {
                kind: NodeKind::UnOp,
                input_count: 1,
                inputs_offset,
                compute_fn: CF_CAST_ARRAY, // compute_cast_array
            });
            self.graph.set_cast_target_type(node, target_ty.clone());
            return node;
        }

        // Fallback: source or target is a generic type parameter whose concrete type is not yet
        // known (resolved later by monomorphization). Emit a scalar cast node optimistically;
        // `compute_cast_scalar` reads the concrete target at runtime via `cast_target_type`.
        let inputs_offset = self.graph.inputs_pool.push(&[input]);
        let node = self.graph.add_node(Node {
            kind: NodeKind::UnOp,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_CAST_SCALAR, // compute_cast_scalar
        });
        self.graph.set_cast_target_type(node, target_ty.clone());
        node
    }

}
