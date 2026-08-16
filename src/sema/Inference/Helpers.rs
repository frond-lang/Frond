//! Helpers — Small shared inference helpers and literal-name utilities. Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Returns the TypeHandle for a builtin scalar type (helper).
    pub(super) fn make_builtin(&mut self, ty: Type) -> TypeHandle {
        self.arena.make(ty)
    }

    /// Constructs the dedicated Type variant for a builtin generic type (Throw/Channel/Async/Lazy/Atomic/Sender/Receiver).
    /// Falls back to Type::Generic on arity mismatch (fault-tolerant; sema already constrains builtin generic arity).
    pub(super) fn make_builtin_generic(&mut self, name: Box<str>, args: Box<[TypeHandle]>) -> TypeHandle {
        match name.as_ref() {
            "Throw" if args.len() == 2 => self.arena.make_throw(args[0], args[1]),
            "Channel" if args.len() == 1 => self.arena.make_channel(args[0]),
            "Async" if args.len() == 1 => self.arena.make_async(args[0]),
            "Lazy" if args.len() == 1 => self.arena.make_lazy(args[0]),
            "Atomic" if args.len() == 1 => self.arena.make_atomic(args[0]),
            "Sender" if args.len() == 1 => self.arena.make_sender(args[0]),
            "Receiver" if args.len() == 1 => self.arena.make_receiver(args[0]),
            "ForeignFn" if args.len() == 1 => self.arena.make_foreign_fn(args[0]),
            _ => self.arena.make_generic(name, args),
        }
    }

    /// Determines whether an expression is a literal (used by peer_type_binary callers).
    pub(super) fn expr_is_literal(ast: &AstArena<'_>, expr: ExprId) -> bool {
        matches!(
            ast.expr(expr).node,
            Expr::IntLit { .. }
                | Expr::FloatLit { .. }
                | Expr::BoolLit(_)
                | Expr::CharLit(_)
                | Expr::StrLit(_)
                | Expr::NullLit
                | Expr::VoidLit
        )
    }

    /// Returns true if the expression has an explicitly declared numeric type
    /// that cannot be silently promoted. This includes:
    /// - Suffixed numeric literals (e.g. `1i32`, `2.0f64`)
    /// - Identifier references (variables with declared types)
    /// Computed expressions (binary ops, calls, etc.) are NOT "explicitly typed"
    /// because their type may result from bare-literal promotion internally.
    pub(super) fn expr_is_explicitly_typed_numeric(ast: &AstArena<'_>, expr: ExprId) -> bool {
        match ast.expr(expr).node {
            Expr::IntLit { suffix: Some(_), .. }
            | Expr::FloatLit { suffix: Some(_), .. } => true,
            Expr::Ident(_) => true,
            _ => false,
        }
    }

    /// Check numeric binary operation type compatibility (Bug #73, #74).
    ///
    /// Rules (consistent with user preference: Rust-style strict typing,
    /// bare literals promotable, explicitly typed operands require cast):
    /// 1. Types already equal → OK
    /// 2. Cross-category (int vs float): always error (no implicit int↔float conversion)
    /// 3. Same category different bit width: error only if both sides are
    ///    explicitly typed (suffixed literal or variable identifier)
    pub(super) fn check_numeric_binop_compat(
        &mut self,
        ast: &AstArena<'_>,
        lhs: ExprId,
        rhs: ExprId,
        left_ty: TypeHandle,
        right_ty: TypeHandle,
        span: crate::ast::Ast::Span,
    ) {
        // If types are already equal, no issue.
        if types_equal(self.arena, left_ty, right_ty) {
            return;
        }

        let lc = self.arena.get(left_ty);
        let rc = self.arena.get(right_ty);

        let left_str = format!("{}", self.arena.display(left_ty));
        let right_str = format!("{}", self.arena.display(right_ty));

        // Cross-category (int vs float): always error — no implicit int↔float conversion.
        if (lc.is_int() && rc.is_float()) || (lc.is_float() && rc.is_int()) {
            self.add_error_at(
                &format!(
                    "type mismatch: cannot operate on '{}' and '{}' without explicit cast (int/float category mismatch)",
                    left_str, right_str
                ),
                span.line,
                span.column,
            );
            return;
        }

        // Same category, different bit widths: error only if both sides are
        // explicitly typed (suffixed literal or variable identifier).
        // Computed expressions (e.g. `1.0 / 0.0`) may derive their type from
        // bare-literal promotion, so they are not treated as "explicitly typed".
        let left_explicit = Self::expr_is_explicitly_typed_numeric(ast, lhs);
        let right_explicit = Self::expr_is_explicitly_typed_numeric(ast, rhs);
        if left_explicit && right_explicit {
            self.add_error_at(
                &format!(
                    "type mismatch: cannot operate on '{}' and '{}' without explicit cast (different bit widths)",
                    left_str, right_str
                ),
                span.line,
                span.column,
            );
        }
    }

    /// Dereferences a ref/nullable type, returning the inner type; for non-ref/nullable types returns the original type.
    /// SafeAccess `?.` on a Nullable needs to unwrap the inner type to look up fields, matching how method calls unwrap Nullable.
    pub(super) fn unwrap_ref(&self, ty: TypeHandle) -> TypeHandle {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Type::Ref(_) => self.arena.ref_parts(resolved).0,
            Type::Nullable(_) => self.arena.nullable_inner(resolved),
            _ => resolved,
        }
    }

    /// Structurally extracts the element type from an iterator type.
    /// Covers all standard iterator shapes:
    /// - Array<T> → T (arrays are not iterators, but the element type is extracted for constraints).
    /// - ArrayIter<T> / Iter<T> / RangeIterator → T.
    /// - Iterator of Map<K,V> → Entry<K,V>.
    /// - Str → char.
    /// - Throw<T,E> → T (destructured directly so that iterating over a Throw yields value-typed elements).
    /// Returns None on failure (the caller falls back to fresh_type_var + a constraint).
    pub(super) fn extract_iterator_element(&mut self, h: TypeHandle) -> Option<TypeHandle> {
        let ty = self.arena.get(h);
        match ty {
            Type::Array(_) => Some(self.arena.array_parts(h).0),
            Type::Str => Some(self.make_builtin(Type::Char)),
            Type::Generic(_) => {
                let (name, args) = self.arena.generic_parts(h);
                match name {
                    // Standard iterators: ArrayIter<T>, Iter<T>, RangeIterator (no args; element is i64).
                    "ArrayIter" | "Iter" if args.len() == 1 => Some(args[0]),
                    "RangeIterator" => Some(self.make_builtin(Type::I64)),
                    // Map iterators return Entry<K,V>.
                    "MapIter" | "MapKeys" | "MapValues" if args.len() == 1 => Some(args[0]),
                    "Map" if args.len() == 2 => {
                        let entry_ty = self.arena.make_generic(
                            "Entry".into(),
                            args.to_vec().into_boxed_slice(),
                        );
                        Some(entry_ty)
                    }
                    _ => None,
                }
            }
            Type::Throw(_) => Some(self.arena.throw_parts(h).0),
            _ => None,
        }
    }

    /// Return type for an auto-impl reflect trait method, or `None` if `method`
    /// is not a reflect method. This pairs with `Builder::reflect_method_intrinsic`
    /// to give every type access to reflect methods without explicit trait impl.
    ///
    /// Must stay in sync with:
    /// - `ir/Builder.rs::reflect_method_intrinsic` (method-name → IntrinsicKind)
    /// - `ir/Compute.rs` compute_reflect_* (the runtime implementation)
    pub(super) fn reflect_method_return_type(&mut self, method: &str, arg_count: usize) -> Option<TypeHandle> {
        // Nullary reflect methods (receiver only, arg_count == 0).
        let nullary: Option<TypeHandle> = match method {
            "format" | "type_name" | "kind" | "constructor" => Some(self.make_builtin(Type::Str)),
            "size" | "alignment" => Some(self.make_builtin(Type::U32)),
            "field_count" => Some(self.make_builtin(Type::U16)),
            _ => None,
        };
        if let Some(ty) = nullary {
            return (arg_count == 0).then_some(ty);
        }
        // Unary-index reflect methods (receiver + index, arg_count == 1).
        let unary: Option<TypeHandle> = match method {
            "field_name" => Some(self.make_builtin(Type::Str)),
            _ => None,
        };
        if let Some(ty) = unary {
            return (arg_count == 1).then_some(ty);
        }
        None
    }

    /// Return type for a `Lib` / `ForeignFn` builtin method, or `None` if
    /// (recv_ty, method) is not a Lib-family method call. This pairs with
    /// `ir/Builder` structural lowering and `ir/Compute.rs` compute_lib_* /
    /// compute_ffn_call (the runtime).
    ///
    /// `lookup`'s `R` is a FRESH TYPE VAR: the method-call path does not
    /// propagate expected types, so R is solved either by the caller unifying
    /// `expected` (when present) or downstream — the `Ok(f)` pattern binding
    /// plus a `val f: ForeignFn<u64>` annotation, or a use of `f.call`'s
    /// result. The IR layer reads the solved R at build time
    /// (lib_lookup_ret_tag).
    pub(super) fn lib_method_return_type(
        &mut self,
        recv_ty: Type,
        recv_handle: TypeHandle,
        method: &str,
        arg_count: usize,
        expected: Option<TypeHandle>,
    ) -> Option<TypeHandle> {
        match (recv_ty, method) {
            (Type::Lib, "lookup") if arg_count == 2 => {
                let r = self.arena.fresh_type_var();
                let ff = self.arena.make_foreign_fn(r);
                let err = self.ffi_error_ty();
                let ret = self.arena.make_throw(ff, err);
                if let Some(exp) = expected {
                    self.unify_or_constrain(ret, exp);
                }
                Some(ret)
            }
            (Type::Lib, "has_symbol") if arg_count == 1 => Some(self.make_builtin(Type::Bool)),
            (Type::Lib, "close") if arg_count == 0 => Some(self.make_builtin(Type::Void)),
            (Type::ForeignFn(_), "call") => {
                // Any arity ≥ 0; the return type is the receiver's R.
                let ret = self.arena.foreign_fn_ret(recv_handle);
                let err = self.ffi_error_ty();
                Some(self.arena.make_throw(ret, err))
            }
            _ => None,
        }
    }

    /// The `FfiError` type as seen from the Lib builtin methods. Declared in
    /// `builtin/error/FfiError.frond`; resolved by name here (Adt unify is
    /// name-based, so this handle interops with the declared one).
    pub(super) fn ffi_error_ty(&mut self) -> TypeHandle {
        self.arena.make_adt("FfiError".into(), Box::new([]))
    }

    /// Integer suffix → corresponding integer TypeHandle (derived from `BUILTIN_TABLE`; returns `None` on miss).
    pub(super) fn int_suffix_to_type(&mut self, suffix: &str) -> Option<TypeHandle> {
        let tag = crate::types::ValueTag::from_name(suffix)?;
        if tag.is_int() {
            Some(self.arena.from_scalar_name(suffix))
        } else {
            None
        }
    }

    /// Float suffix → corresponding float TypeHandle (derived from `BUILTIN_TABLE`; returns `None` on miss).
    pub(super) fn float_suffix_to_type(&mut self, suffix: &str) -> Option<TypeHandle> {
        let tag = crate::types::ValueTag::from_name(suffix)?;
        if tag.is_float() {
            Some(self.arena.from_scalar_name(suffix))
        } else {
            None
        }
    }

    /// Bug #61: render the type annotation string — if the annotation is a named type and is an alias, preserve the alias name.
    pub(super) fn display_type_annotation(
        &self,
        ta: AstTypeRef,
        ast: &AstArena<'_>,
        annot_ty: TypeHandle,
    ) -> String {
        let type_node = &ast.ty(ta).node;
        if let crate::ast::Ast::TypeNode::Named { name } = type_node {
            if let Some(td) = self.sema_result.get_type_def(name) {
                if td.kind == TypeDefKind::Alias {
                    return (*name).to_string();
                }
            }
        }
        format!("{}", self.arena.display(annot_ty))
    }

}

/// Range-check an integer literal's raw text against the target scalar type's range.
/// Returns `Some(error message)` when out of range or unparseable; `None` when in range.
/// Mirrors `ir::Builder::check_int_range` so sema and IR report consistently (Bug #72: stage consistency).
pub(super) fn check_int_literal_range(raw: &str, tag: crate::types::ValueTag) -> Option<String> {
    let cleaned: String = raw.chars().filter(|c| *c != '_').collect();
    let (digits, radix) = cleaned
        .strip_prefix("0x").map(|s| (s, 16u32))
        .or_else(|| cleaned.strip_prefix("0o").map(|s| (s, 8)))
        .or_else(|| cleaned.strip_prefix("0b").map(|s| (s, 2)))
        .unwrap_or((cleaned.as_str(), 10));
    // i128/u128 literals cannot overflow their own parse; only syntax errors are possible.
    match tag {
        crate::types::ValueTag::I128 => {
            return match i128::from_str_radix(digits, radix) {
                Ok(_) => None,
                Err(_) => Some(format!("invalid integer literal '{}'", raw)),
            };
        }
        crate::types::ValueTag::U128 => {
            return match u128::from_str_radix(digits, radix) {
                Ok(_) => None,
                Err(_) => Some(format!("invalid integer literal '{}'", raw)),
            };
        }
        _ => {}
    }
    let (min, max, name): (i128, i128, &str) = match tag {
        crate::types::ValueTag::I8    => (i8::MIN    as i128, i8::MAX    as i128, "i8"),
        crate::types::ValueTag::I16   => (i16::MIN   as i128, i16::MAX   as i128, "i16"),
        crate::types::ValueTag::I32   => (i32::MIN   as i128, i32::MAX   as i128, "i32"),
        crate::types::ValueTag::I64   => (i64::MIN   as i128, i64::MAX   as i128, "i64"),
        crate::types::ValueTag::U8    => (0,                   u8::MAX    as i128, "u8"),
        crate::types::ValueTag::U16   => (0,                   u16::MAX   as i128, "u16"),
        crate::types::ValueTag::U32   => (0,                   u32::MAX   as i128, "u32"),
        crate::types::ValueTag::U64   => (0,                   u64::MAX   as i128, "u64"),
        crate::types::ValueTag::Isize => (isize::MIN as i128, isize::MAX as i128, "isize"),
        crate::types::ValueTag::Usize => (0,                   usize::MAX as i128, "usize"),
        _ => return None,
    };
    match i128::from_str_radix(digits, radix) {
        Ok(v) if v < min || v > max => Some(format!(
            "integer literal '{}' is out of range for {} (valid range: {}..={})",
            raw, name, min, max
        )),
        Ok(_) => None,
        Err(_) => Some(format!("invalid integer literal '{}'", raw)),
    }
}

/// Inference context: encapsulates all state needed for type inference.
///
/// Lifetime: a single TypeArena is shared across the whole module's sema stage; InferContext holds a `&mut` reference to it.
/// Instantiation-mode context: used when resolving types in a monomorphized function body.
///
/// Design: two-phase writes to avoid aliasing conflicts.
/// - Runtime type results are staged in local_expr_types.
/// - After the run completes, take_local_expr_types() transfers them into MonomorphInstance.expr_types.
///
/// Does not hold func_decls (lifetime-bearing references); monomorphization triggers in the Call branch are orchestrated externally.

/// Builtin scalar name → Type (single derivation point, replacing the three previously duplicated
/// match sites).
///
/// Derived from `Type::BUILTIN_TABLE`: look up ValueTag by name, then dispatch to Type by ValueTag.
/// The name → ValueTag mapping comes from a single source of truth.
///
/// Type names are uniformly lower-case (consistent with .frond source syntax): null/void/bool/char/str
/// and the numeric types.
pub(super) fn name_to_concrete(name: &str) -> Option<Type> {
    use crate::types::{builtin_info_by_name, ValueTag};
    let info = builtin_info_by_name(name)?;
    let ct = match info.value_tag {
        ValueTag::I8 => Type::I8,
        ValueTag::I16 => Type::I16,
        ValueTag::I32 => Type::I32,
        ValueTag::I64 => Type::I64,
        ValueTag::I128 => Type::I128,
        ValueTag::U8 => Type::U8,
        ValueTag::U16 => Type::U16,
        ValueTag::U32 => Type::U32,
        ValueTag::U64 => Type::U64,
        ValueTag::U128 => Type::U128,
        ValueTag::Isize => Type::Isize,
        ValueTag::Usize => Type::Usize,
        ValueTag::F16 => Type::F16,
        ValueTag::F32 => Type::F32,
        ValueTag::F64 => Type::F64,
        ValueTag::F128 => Type::F128,
        ValueTag::Bool => Type::Bool,
        ValueTag::Char => Type::Char,
        ValueTag::Ref => Type::Str,   // str's value_tag is Ref.
        ValueTag::Null => Type::Null,
        ValueTag::Void => Type::Void,
    };
    Some(ct)
}

/// Returns all numeric builtin type names + Type (derived from BUILTIN_TABLE).
///
/// Replaces the original static `NUMERIC_BUILTIN_NAMES` table; automatically syncs with
/// BUILTIN_TABLE changes.
/// Includes all scalars (including bool/char, consistent with the original table); excludes
/// str/null/void.
pub(super) fn numeric_builtin_names() -> Vec<(&'static str, Type)> {
    use crate::types::{BUILTIN_TABLE, ValueTag};
    BUILTIN_TABLE.iter()
        .filter(|s| !matches!(s.value_tag, ValueTag::Ref | ValueTag::Null | ValueTag::Void))
        .filter_map(|s| {
            let ct = name_to_concrete(s.name)?;
            Some((s.name, ct))
        })
        .collect()
}


// ── Missing return value check (Bug: non-void function with no tail expression) ──

/// A `return value` or `throw` anywhere in the function body (outside nested
/// lambdas/defer bodies) means the body may exit with a value. An unconditional
/// `loop { ... }` that contains no `break` never exits normally (diverging),
/// which also satisfies "the function cannot fall off the end".
fn body_stmt_has_exit(ast: &AstArena<'_>, stmt: StmtId) -> bool {
    match &ast.stmt(stmt).node {
        Stmt::Return { value } => value.is_some(),
        Stmt::Throw { .. } => true,
        // A nested function's returns belong to it, not the enclosing function.
        Stmt::LocalDecl { .. } => false,
        Stmt::Defer { .. } => false,
        Stmt::Loop { body } => {
            body_expr_has_exit(ast, *body) || !stmt_tree_has_break(ast, *body)
        }
        Stmt::For { body, .. } | Stmt::While { body, .. } => body_expr_has_exit(ast, *body),
        Stmt::Expression { expr } => body_expr_has_exit(ast, *expr),
        _ => false,
    }
}

fn body_expr_has_exit(ast: &AstArena<'_>, expr: ExprId) -> bool {
    match &ast.expr(expr).node {
        // Stop at lambda boundaries: their returns/throws are their own.
        Expr::Lambda { .. } => false,
        Expr::Block { stmts, trailing } => {
            stmts.iter().any(|s| body_stmt_has_exit(ast, *s))
                || trailing.map(|t| body_expr_has_exit(ast, t)).unwrap_or(false)
        }
        Expr::If { then_branch, else_branch, .. } => {
            body_expr_has_exit(ast, *then_branch)
                || else_branch.map(|e| body_expr_has_exit(ast, e)).unwrap_or(false)
        }
        Expr::Match { arms, .. } => arms.iter()
            .any(|arm| body_expr_has_exit(ast, arm.body)),
        _ => false,
    }
}

fn stmt_tree_has_break(ast: &AstArena<'_>, expr: ExprId) -> bool {
    match &ast.expr(expr).node {
        Expr::Lambda { .. } => false,
        Expr::Block { stmts, trailing } => {
            stmts.iter().any(|s| match &ast.stmt(*s).node {
                Stmt::Break => true,
                Stmt::Expression { expr } => stmt_tree_has_break(ast, *expr),
                Stmt::For { body, .. } | Stmt::While { body, .. } | Stmt::Loop { body } =>
                    stmt_tree_has_break(ast, *body),
                _ => false,
            }) || trailing.map(|t| stmt_tree_has_break(ast, t)).unwrap_or(false)
        }
        Expr::If { then_branch, else_branch, .. } => {
            stmt_tree_has_break(ast, *then_branch)
                || else_branch.map(|e| stmt_tree_has_break(ast, e)).unwrap_or(false)
        }
        Expr::Match { arms, .. } => arms.iter()
            .any(|arm| stmt_tree_has_break(ast, arm.body)),
        _ => false,
    }
}

impl<'a> InferContext<'a> {
    /// Rejects a function/lambda whose declared return type is non-void but whose
    /// body is a block with no trailing expression and no `return value`/`throw`
    /// statement. Such a body previously compiled silently and returned garbage
    /// at runtime (e.g. an i32 function returning `2.71875f16`, or a str function
    /// leaking `Ok(void)`).
    pub(super) fn check_missing_return_value(
        &mut self,
        what: &str,
        ret_ty: TypeHandle,
        body: ExprId,
        ast: &AstArena<'_>,
        line: u32,
        column: u32,
    ) {
        // Async<X> carries X as its produced value: Async<void> needs no value.
        let mut ret_ty = ret_ty;
        loop {
            let resolved = self.arena.resolve(ret_ty);
            match self.arena.get(resolved) {
                Type::Void => return,
                Type::Async(_) => ret_ty = self.arena.async_value(resolved),
                _ => break,
            }
        }
        let (stmts, trailing) = match &ast.expr(body).node {
            Expr::Block { stmts, trailing } => (stmts, trailing),
            // A non-block body expression is itself the return value.
            _ => return,
        };
        if trailing.is_some() {
            return;
        }
        if stmts.iter().any(|s| body_stmt_has_exit(ast, *s)) {
            return;
        }
        let ret_str = format!("{}", self.arena.display(ret_ty));
        self.add_error_at(
            &format!(
                "missing return value: {what} declares return type '{ret_str}' but its body has no trailing expression and no 'return'/'throw' statement"
            ),
            line,
            column,
        );
    }
}
