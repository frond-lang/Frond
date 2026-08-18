//! StmtInfer — Statement and local-decl inference. Mechanically split from Inference.rs (no logic changes).

use super::*;

impl<'a> InferContext<'a> {
    /// Shared logic for ValDecl and VarDecl: type-check the annotation and value,
    /// detect mutability-changing shadowing (Bug #76), and define the binding.
    pub(super) fn check_local_decl(
        &mut self,
        name: &str,
        type_annotation: Option<crate::ast::Ast::TypeRef>,
        value: crate::ast::Ast::ExprRef,
        is_mutable: bool,
        ast: &AstArena<'_>,
        env: EnvId,
        stmt: StmtId,
    ) -> TypeHandle {
        // kind_check the type annotation.
        if let Some(ta) = type_annotation {
            let mut errors = Vec::new();
            check_type_node(self.sema_result, ast, ta, &[], &mut errors);
            for e in errors {
                self.sema_result.add_error(e);
            }
        }
        let expected_ty = type_annotation.map(|ta| self.type_from_ast(ta, ast));
        let val_ty = self.infer_expr(value, ast, env, expected_ty);
        let bind_ty = if let Some(ta) = type_annotation {
            let annot_ty = self.type_from_ast(ta, ast);
            if self.try_widen_unify(annot_ty, val_ty).is_err() {
                // Literal → nullable annotation: `val a: i64? = 42`. Bare
                // literals never promote toward a Nullable expected type, so
                // the literal stays i32 while the annotation is i64? —
                // re-infer the literal against the INNER scalar (running the
                // normal literal promotion) and accept when that unifies.
                let mut unified = false;
                if Self::expr_is_literal(ast, value) {
                    let annot_resolved = self.arena.resolve(annot_ty);
                    if let Type::Nullable(_) = self.arena.get(annot_resolved) {
                        let inner = self.arena.nullable_inner(annot_resolved);
                        let re_ty = self.infer_expr(value, ast, env, Some(inner));
                        if self.try_widen_unify(inner, re_ty).is_ok() {
                            unified = true;
                        }
                    }
                }
                if !unified {
                    // Bug #61: if the type annotation is a named type and is an alias, preserve the alias name rather than unfolding the underlying type.
                    let annot_str = self.display_type_annotation(ta, ast, annot_ty);
                    let val_str = format!("{}", self.arena.display(val_ty));
                    let span = ast.ty(ta).span;
                    self.add_error_at(
                        &format!(
                            "type annotation mismatch: expected '{}', found '{}'",
                            annot_str, val_str
                        ),
                        span.line,
                        span.column,
                    );
                }
            }
            annot_ty
        } else {
            val_ty
        };

        // Bug #76: detect mutability-changing shadowing (val→var or var→val).
        // Same-mutability shadowing (val→val, var→var) is allowed.
        let key = (env.0, name.to_string());
        if let Some(&prev_mutable) = self.local_mutability.get(&key) {
            if prev_mutable != is_mutable {
                let prev_kw = if prev_mutable { "var" } else { "val" };
                let new_kw = if is_mutable { "var" } else { "val" };
                let span = ast.stmt(stmt).span;
                self.add_error_at(
                    &format!(
                        "cannot shadow {} '{}' with {} {}: mutability mismatch",
                        prev_kw, name, new_kw, name
                    ),
                    span.line,
                    span.column,
                );
            }
        }
        // Record mutability for this scope.
        self.local_mutability.insert(key, is_mutable);

        // Define the binding. Use redefine to allow same-mutability shadowing
        // (define returns false without updating when the name already exists).
        if self.sema_result.env.define(env, name, bind_ty) {
            // New binding — already inserted.
        } else {
            // Name already exists — shadowing. Use redefine to update the binding.
            self.sema_result.env.redefine(env, name, bind_ty);
        }
        // Return the value's inferred type so callers (e.g. Block divergence
        // analysis) can detect `val/var x = <never>` as a diverging statement.
        val_ty
    }

    /// Infers a statement's type. Returns `Some(ty)` when the statement produces a value (expression statements).
    pub fn infer_stmt(
        &mut self,
        stmt: StmtId,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> Option<TypeHandle> {
        let node = &ast.stmt(stmt).node;
        match node {
            Stmt::ValDecl { name, type_annotation, value, .. } => {
                let val_ty = self.check_local_decl(name, *type_annotation, *value, false, ast, env, stmt);
                // Propagate Never so Block-level divergence analysis detects
                // `val x = <never>` (e.g. a val bound to an if/match/block that
                // always diverges) as a diverging statement. Bug #84 generality.
                if matches!(self.arena.get(self.arena.resolve(val_ty)), Type::Never) {
                    Some(val_ty)
                } else {
                    None
                }
            }
            Stmt::VarDecl { name, type_annotation, value, .. } => {
                let val_ty = self.check_local_decl(name, *type_annotation, *value, true, ast, env, stmt);
                if matches!(self.arena.get(self.arena.resolve(val_ty)), Type::Never) {
                    Some(val_ty)
                } else {
                    None
                }
            }
            Stmt::Assignment { target, value } => {
                // `*ref = value` compiles but the store is silently lost at runtime
                // (compute_deref_write boxes scalars into an orphan Cell; heap refs
                // are a no-op), so reject it until true write-through lands. The
                // in-place paths that DO work keep compiling: `(*ref).field = v`
                // (record_field_set) and `(*ref)[i] = v` (array element store).
                if matches!(ast.expr(*target).node, crate::ast::Ast::Expr::Deref(_)) {
                    let span = ast.stmt(stmt).span;
                    self.add_error_at(
                        "cannot assign through `*ref`: dereference writes are not implemented; \
                         mutate in place via `(*ref).field = ...` or `(*ref)[index] = ...` instead",
                        span.line,
                        span.column,
                    );
                }
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                if self.arena.unify(target_ty, val_ty).is_err() {
                    let target_str = format!("{}", self.arena.display(target_ty));
                    let val_str = format!("{}", self.arena.display(val_ty));
                    let span = ast.stmt(stmt).span;
                    self.add_error_at(
                        &format!(
                            "assignment type mismatch: cannot assign '{}' to '{}'",
                            val_str, target_str
                        ),
                        span.line,
                        span.column,
                    );
                }
                None
            }
            Stmt::FieldAssignment { object, value, .. } => {
                let _ = self.infer_expr(*object, ast, env, None);
                let _ = self.infer_expr(*value, ast, env, None);
                None
            }
            Stmt::CompoundAssignment { target, value, .. } => {
                // Same silent-store hole as plain assignment: `*ref += value`
                // lowers to a deref write that never reaches the binding.
                if matches!(ast.expr(*target).node, crate::ast::Ast::Expr::Deref(_)) {
                    let span = ast.stmt(stmt).span;
                    self.add_error_at(
                        "cannot compound-assign through `*ref`: dereference writes are not implemented; \
                         mutate in place via `(*ref).field = ...` or `(*ref)[index] = ...` instead",
                        span.line,
                        span.column,
                    );
                }
                let target_ty = self.infer_expr(*target, ast, env, None);
                let value_ty = self.infer_expr(*value, ast, env, None);
                // Bug #95: compound assignment must enforce the same strict numeric
                // rules as the equivalent binary expression — `x += 2.0` (i32 += f64)
                // used to compile AND silently drop the store; `x += 2i64` silently
                // cross-width converted. Literal values still adapt via
                // peer_type_binary (so `u8_var += 2` keeps working).
                let rt = self.arena.resolve(target_ty);
                let rv = self.arena.resolve(value_ty);
                if self.arena.get(rt).is_numeric() && self.arena.get(rv).is_numeric() {
                    let span = ast.stmt(stmt).span;
                    self.check_numeric_binop_compat(ast, *target, *value, rt, rv, span);
                    let _ = peer_type_binary(
                        self.arena,
                        rt,
                        rv,
                        Self::expr_is_literal(ast, *target),
                        Self::expr_is_literal(ast, *value),
                    );
                } else if self.arena.unify(rt, rv).is_err() {
                    let span = ast.stmt(stmt).span;
                    let t_str = format!("{}", self.arena.display(rt));
                    let v_str = format!("{}", self.arena.display(rv));
                    self.add_error_at(
                        &format!(
                            "compound assignment type mismatch: target '{}' is not compatible with value '{}'",
                            t_str, v_str
                        ),
                        span.line,
                        span.column,
                    );
                }
                None
            }
            Stmt::Expression { expr } => {
                let ty = self.infer_expr(*expr, ast, env, None);
                Some(ty)
            }
            Stmt::Return { value } => {
                if let Some(v) = value {
                    // Propagate expected_return to infer_expr so expressions that depend on
                    // the expected constraint (NullLit, match arm, etc.) infer correctly and
                    // avoid creating orphan TypeVars.
                    let expected = self.expected_return;
                    let val_ty = self.infer_expr(*v, ast, env, expected);
                    if let Some(fn_ret) = self.expected_return {
                        if self.unify_return_type(fn_ret, val_ty).is_err() {
                            let ret_str = format!("{}", self.arena.display(fn_ret));
                            let val_str = format!("{}", self.arena.display(val_ty));
                            let span = ast.stmt(stmt).span;
                            self.add_error_at(
                                &format!(
                                    "return type mismatch: expected '{}', found '{}'",
                                    ret_str, val_str
                                ),
                                span.line,
                                span.column,
                            );
                        }
                        // Sync Throw-returning functions must return a Throw VALUE.
                        // A bare payload (often via `return expr?`, which unwraps)
                        // leaks non-Throw where Throw was declared — the
                        // from_datetime_utc/scanln bug class. Async funs are
                        // exempt: their expected_return is Async-wrapped, so the
                        // first check below filters them. TypeVar/Unknown values
                        // may still solve to Throw via the fixpoint — skip them.
                        if matches!(self.arena.get(self.arena.resolve(fn_ret)), Type::Throw(_)) {
                            match self.arena.get(self.arena.resolve(val_ty)) {
                                Type::Throw(_)
                                | Type::TypeVar(_)
                                | Type::Unknown
                                | Type::Never => {}
                                _ => {
                                    let ret_str = format!("{}", self.arena.display(fn_ret));
                                    let val_str = format!("{}", self.arena.display(val_ty));
                                    let span = ast.stmt(stmt).span;
                                    self.add_error_at(
                                        &format!(
                                            "'return' value must be Throw-wrapped: expected '{}', found '{}' (wrap in Ok(..)/Err(..) or return the Throw value directly; 'expr?' unwraps)",
                                            ret_str, val_str
                                        ),
                                        span.line,
                                        span.column,
                                    );
                                }
                            }
                        }
                    }
                    Some(val_ty)
                } else {
                    Some(self.make_builtin(Type::Void))
                }
            }
            Stmt::Defer { expr } => {
                let _ = self.infer_expr(*expr, ast, env, None);
                // Record the defer's capture list. Defer semantics require
                // observing values at function/block exit, so ALL captures are
                // forced to Reference mode (this directly resolves Bug #49:
                // the `val`-snapshot vs defer-latest tension is eliminated
                // because defer never snapshots).
                if self.instantiation_ctx.is_none() {
                    let captures = self.compute_captures(ast, *expr, &[], true);
                    let key = module_expr_key(&self.current_module_name, expr.0 as u64);
                    self.sema_result.put_captures(key, &self.current_module_name, captures);
                }
                None
            }
            Stmt::Throw { expr } => {
                let thrown_ty = self.infer_expr(*expr, ast, env, None);
                let span = ast.stmt(stmt).span;
                self.check_throw_stmt(thrown_ty, span.line, span.column);
                None
            }
            Stmt::Break | Stmt::Continue => None,
            Stmt::For { name, iterable, body } => {
                let span = ast.stmt(stmt).span;
                let iterable_ty = self.infer_expr(*iterable, ast, env, None);
                let child_env = self.sema_result.env.child(env);
                // Structurally extract the element type from the iterator type and build a
                // constraint rather than using an isolated fresh_type_var.
                // Covers: Array<T>, ArrayIter<T>, RangeIterator, Str→char, Map<K,V>→Entry<K,V>, etc.
                // On extraction failure, falls back to fresh_type_var and lets the fixpoint solver
                // resolve via constraint.
                let item_ty = {
                    let resolved = self.arena.resolve(iterable_ty);
                    let ct = self.arena.get(resolved);
                    // Check whether iterable is a non-iterator type (Array/Str/primitive)
                    let is_non_iterator = match ct {
                        Type::Array(_) => true,
                        // for-in over a bare str must be rejected: the IR passes the
                        // iterable to `.next` verbatim and `str.next` does not exist,
                        // which hangs the engine (Bug #93). Use str_iter(s).
                        Type::Str => true,
                        ct if ct.is_scalar() => true,
                        _ => false,
                    };
                    if is_non_iterator {
                        let type_name = match ct {
                            // Element-concrete name ("i32[]") reads better in
                            // the diagnostic than the bare "array".
                            Type::Array(_) => self
                                .arena
                                .type_name_concrete(resolved)
                                .unwrap_or_else(|| "array".to_string()),
                            _ => ct.name().to_string(),
                        };
                        self.add_error_at(
                            &format!(
                                "type '{}' does not implement Iterator; For loops require an iterator type. Use arr.iter() for arrays, str_iter(s) for strings",
                                type_name
                            ),
                            span.line,
                            span.column,
                        );
                    }
                    // Iterator protocol is null-terminated (`Iterator<T>.next() -> T?`):
                    // when the ELEMENT type is itself nullable, a null element is
                    // indistinguishable from end-of-iteration and the loop would stop
                    // early. Reject with guidance to iterate by index instead.
                    if let Some(elem) = self.extract_iterator_element(resolved) {
                        let elem_r = self.arena.resolve(elem);
                        if matches!(self.arena.get(elem_r), Type::Nullable(_)) {
                            self.add_error_at(
                                "cannot use for-in over an iterator whose element type is nullable: the null-terminated iterator protocol treats a null element as end-of-iteration. Iterate by index instead: `var i: usize = 0; while i < arr.len() { val e = arr[i] ... }`",
                                span.line,
                                span.column,
                            );
                        }
                    }
                    // Structured element type extraction.
                    self.extract_iterator_element(resolved).unwrap_or_else(|| {
                        // Extraction failed: use fresh_type_var and register a constraint for
                        // the fixpoint solver.
                        let fv = self.arena.fresh_type_var();
                        // Build ArrayIter<fv> as the expected iterator type and unify/constrain
                        // it with the actual iterable_ty.
                        let expected_iter = self.arena.make_generic(
                            "ArrayIter".into(),
                            vec![fv].into_boxed_slice(),
                        );
                        self.unify_or_constrain(iterable_ty, expected_iter);
                        fv
                    })
                };
                self.sema_result.env.define(child_env, name, item_ty);
                let _ = self.infer_expr(*body, ast, child_env, None);
                None
            }
            Stmt::While { condition, body } => {
                let cond_ty = self.infer_expr(*condition, ast, env, None);
                let bool_ty = self.make_builtin(Type::Bool);
                if self.arena.unify(cond_ty, bool_ty).is_err() {
                    let cond_str = format!("{}", self.arena.display(cond_ty));
                    let span = ast.stmt(stmt).span;
                    self.add_error_at(
                        &format!(
                            "while condition must be bool, found '{}'",
                            cond_str
                        ),
                        span.line,
                        span.column,
                    );
                }
                let _ = self.infer_expr(*body, ast, env, None);
                None
            }
            Stmt::Loop { body } => {
                let _ = self.infer_expr(*body, ast, env, None);
                None
            }
            Stmt::LocalDecl { decl } => {
                // Route through check_decl uniformly: nested function/type/trait declarations
                // share the same processing path.
                // LocalDecl's Box<Decl> has no span; the enclosing Stmt provides it.
                // For nested function declarations, record the capture list (the
                // nested function captures outer-scope variables; self-reference
                // to its own name is excluded).
                if let crate::ast::Ast::Decl::FunDecl { name, params, body, .. } = decl.as_ref() {
                    if self.instantiation_ctx.is_none() {
                        let mut param_names: Vec<&str> = params.iter().map(|p| p.name).collect();
                        param_names.push(name); // self-reference is not a capture
                        let captures = self.compute_captures(ast, *body, &param_names, false);
                        let key = module_expr_key(&self.current_module_name, body.0 as u64);
                        self.sema_result.put_captures(key, &self.current_module_name, captures);
                    }
                }
                self.check_decl(decl.as_ref(), ast.stmt(stmt).span, ast, env);
                None
            }

        }
    }

    // ── infer_pattern ──

}
