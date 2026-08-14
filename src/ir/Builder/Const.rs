//! Const — Constant and string-literal lowering (incl. reflect format node). Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Compile a sub-expression (not in tail position).
    ///
    /// The value of a sub-expression (operand, function argument, `if` condition, field-access
    /// base, etc.) is consumed by its parent expression rather than returned directly as the
    /// function result, so it is never in tail position: `in_tail_position` is turned off before
    /// compilation and restored afterwards. This prevents a `Call` inside a sub-expression from
    /// being mis-tagged as a tail call (otherwise `switch_subgraph` frame reuse would swap away
    /// the current frame, breaking the parent expression's execution of the remaining
    /// sub-expressions / operation nodes; e.g. in `fib(n-1)+fib(n-2)`, mis-tagging `fib(n-1)` as
    /// a tail call would cause `fib(n-2)` and the addition node to never execute).
    pub(super) fn compile_subexpr(&mut self, expr_id: crate::ast::Ast::ExprId) -> NodeId {
        let prev_tail = self.in_tail_position;
        self.in_tail_position = false;
        let node = self.compile_expr(expr_id);
        self.in_tail_position = prev_tail;
        node
    }

    /// Compile a constant expression (no inputs).
    pub(super) fn compile_const(&mut self) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        })
    }

    /// Compile a void constant node (used when `return`/`break`/`continue` has no value).
    pub(super) fn compile_void_const(&mut self) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[n.0 as usize] = Some(ConstValue::Void);
        n
    }

    /// Compile a constant expression carrying a raw value, populating `const_values`.
    pub(super) fn compile_const_with_value(&mut self, expr_id: crate::ast::Ast::ExprId) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let node_id = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        match self.parse_const_value(expr_id) {
            Ok(cv) => {
                self.graph.const_values[node_id.0 as usize] = cv;
            }
            Err(msg) => {
                self.graph.const_values[node_id.0 as usize] = None;
                self.errors.push(msg);
            }
        }
        node_id
    }

    /// Parse a constant value from an AST expression.
    ///
    /// Return value semantics:
    /// - `Ok(Some(cv))`: a valid constant literal that passed type-range checks.
    /// - `Ok(None)`: a non-constant expression (e.g. a variable reference) that cannot be folded
    ///   into a constant.
    /// - `Err(msg)`: constant-literal parsing failed (syntax error or value out of target-type
    ///   range).
    pub(super) fn parse_const_value(&mut self, expr_id: crate::ast::Ast::ExprId) -> Result<Option<ConstValue>, String> {
        let spanned = self.current_module().arena.expr(expr_id);
        let span = spanned.span;
        match &spanned.node {
            crate::ast::Ast::Expr::IntLit { raw, suffix } => {
                // Suffix takes priority; when absent, consult the type inferred by sema to pick
                // the corresponding integer ConstValue, ensuring the literal's runtime tag matches
                // the contextual type.
                let ty = suffix
                    .map(|s| s.to_string())
                    .or_else(|| self.expr_type_name(expr_id).map(|s| s.to_string()));
                let ty_name = match ty.as_deref() {
                    Some(t) => t,
                    None => return Err(format!(
                        "internal: missing ExprInfo for int literal expr {:?}", expr_id)),
                };

                // The u128 range (0..=2^128-1) exceeds i128, so parse directly with
                // `u128::from_str_radix`.
                // As with float-suffix dispatch, u128 is the only integer type whose range
                // exceeds i128; the dedicated parse path is mathematically necessary, not a
                // special-case judgement.
                if crate::value::ValueTag::from_name(ty_name) == Some(crate::value::ValueTag::U128) {
                    let v = parse_int_to_u128(raw, span)?;
                    return Ok(Some(ConstValue::U128(v)));
                }

                // Parse the integer: supports 0x/0o/0b prefixes and underscore separators.
                let v = parse_int_to_i128(raw, span)?;

                // Range check + type conversion (generic approach, unified for all integer types
                // via a macro).
                Ok(Some(check_int_range(v, ty_name, raw, span)?))
            }
            crate::ast::Ast::Expr::FloatLit { raw, suffix } => {
                // Strip underscore separators (Rust's `parse` does not accept underscores).
                let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
                let is_hex = cleaned.starts_with("0x") || cleaned.starts_with("0X");
                let cv = match suffix {
                    None | Some("f64") => {
                        if is_hex { parse_hex_float_f64(&cleaned).map(ConstValue::F64) }
                        else { cleaned.parse::<f64>().ok().map(ConstValue::F64) }
                    }
                    Some("f32") => {
                        if is_hex { parse_hex_float_f32(&cleaned).map(ConstValue::F32) }
                        else { cleaned.parse::<f32>().ok().map(ConstValue::F32) }
                    }
                    Some("f16") => {
                        if is_hex { parse_hex_float_f16(&cleaned).map(ConstValue::F16) }
                        else {
                            cleaned.parse::<f64>()
                                .ok()
                                .map(|f| ConstValue::F16(crate::value::F16::from_f64(f).to_bits()))
                        }
                    }
                    Some("f128") => {
                        if is_hex { parse_hex_float_f128(&cleaned).map(ConstValue::F128) }
                        else { parse_decimal_f128(&cleaned).map(ConstValue::F128) }
                    }
                    _ => {
                        if is_hex { parse_hex_float_f64(&cleaned).map(ConstValue::F64) }
                        else { cleaned.parse::<f64>().ok().map(ConstValue::F64) }
                    }
                };
                Ok(cv)
            }
            crate::ast::Ast::Expr::BoolLit(b) => Ok(Some(ConstValue::Bool(*b))),
            crate::ast::Ast::Expr::CharLit(c) => Ok(Some(ConstValue::Char(*c))),
            crate::ast::Ast::Expr::StrLit(s) => {
                let (offset, len) = self.intern_str(s);
                Ok(Some(ConstValue::Str { offset, len }))
            }
            crate::ast::Ast::Expr::NullLit => Ok(Some(ConstValue::Null)),
            crate::ast::Ast::Expr::VoidLit => Ok(Some(ConstValue::Void)),
            _ => Ok(None),
        }
    }

    /// Compile a placeholder node (for Expr variants not yet implemented at this stage).
    pub(super) fn compile_placeholder(&mut self) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        })
    }

    /// Compile a bool constant node.
    pub(super) fn compile_bool_const(&mut self, b: bool) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[n.0 as usize] = Some(ConstValue::Bool(b));
        n
    }

    /// Compile a string constant node (used for string literals in pattern matching).
    pub(super) fn compile_str_const(&mut self, s: &str) -> NodeId {
        let (offset, len) = self.intern_str(s);
        let inputs_offset = self.graph.inputs_pool.push(&[]);
        let n = self.graph.add_node(Node {
            kind: NodeKind::Const,
            input_count: 0,
            inputs_offset,
            compute_fn: CF_NOOP,
        });
        self.graph.const_values[n.0 as usize] = Some(ConstValue::Str { offset, len });
        n
    }

    /// Compile string interpolation: lower `"text {expr} more {expr}"` into a chained str_concat.
    ///
    /// Each Literal part is compiled into a string constant node;
    /// each Expression part is converted to a string via `compute_reflect_format` (idx 290);
    /// all parts are chained together via `compute_str_concat` (idx 269).
    pub(super) fn compile_str_interp(
        &mut self,
        parts: &[crate::ast::Ast::InterpolationPart<'_>],
    ) -> NodeId {
        if parts.is_empty() {
            return self.compile_str_const("");
        }

        // Collect all part nodes
        let mut nodes: Vec<NodeId> = Vec::with_capacity(parts.len());
        for part in parts {
            match part {
                crate::ast::Ast::InterpolationPart::Literal(text) => {
                    if !text.is_empty() {
                        nodes.push(self.compile_str_const(text));
                    }
                }
                crate::ast::Ast::InterpolationPart::Expression(expr_id) => {
                    let expr_node = self.compile_subexpr(*expr_id);
                    // Convert any value to a string via compute_reflect_format
                    // (a standalone compute_fn, not going through FFI dispatch, with built-in lazy force)
                    let inputs_offset = self.graph.inputs_pool.push(&[expr_node]);
                    let reflect_node = self.graph.add_node(Node {
                        kind: NodeKind::Call,
                        input_count: 1,
                        inputs_offset,
                        compute_fn: CF_REFLECT_FORMAT, // compute_reflect_format
                    });
                    nodes.push(reflect_node);
                }
            }
        }

        // Single part: return directly
        if nodes.len() == 1 {
            return nodes[0];
        }

        // Multi-input one-shot concat: O(n) one-shot concatenation, replacing chained O(n²) concat
        let inputs_offset = self.graph.inputs_pool.push(&nodes);
        self.graph.add_node(Node {
            kind: NodeKind::BinOp,
            input_count: nodes.len() as u8,
            inputs_offset,
            compute_fn: CF_STR_MULTI_CONCAT,
        })
    }

    /// Convert any value node to a string node via compute_reflect_format (idx 290).
    /// Used to convert non-string operands to strings for `str + non-str` concatenation (same as string interpolation).
    pub(super) fn make_reflect_format_node(&mut self, value_node: NodeId) -> NodeId {
        let inputs_offset = self.graph.inputs_pool.push(&[value_node]);
        self.graph.add_node(Node {
            kind: NodeKind::Call,
            input_count: 1,
            inputs_offset,
            compute_fn: CF_REFLECT_FORMAT,
        })
    }

}
