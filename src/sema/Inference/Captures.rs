//! Captures — Capture analysis: scoped free-ident collection and capture modes. Mechanically split from Inference.rs (no logic changes).

// =========================================================================
// phase5: InferContext extensions — expression/statement/pattern inference + module check entry
//
// Ported from `src/sema/type_check.zig`: inferExpr / inferStmt / inferPattern /
// registerBuiltins / checkModuleWithName.
// =========================================================================


use super::*;

impl<'a> InferContext<'a> {
    /// Stores the inferred type as ExprInfo into sema_result.expr_types.
    pub(super) fn store_expr_info(&mut self, expr: ExprId, ty: TypeHandle) {
        let resolved = self.arena.resolve(ty);
        let ct = self.arena.get(resolved);
        // Concrete name: arrays render element-concretely ("u8[]") instead of
        // the bare "array" (ExprInfo.type_name feeds the IR builder's
        // expr_type_name).
        let type_name: Option<String> = self.arena.type_name_concrete(resolved);
        let is_ref = matches!(ct, Type::Ref(_));
        let is_raw_ref = matches!(ct, Type::Ref(_)) && self.arena.ref_parts(resolved).1;

        let is_trait_object = matches!(ct, Type::TraitObject(_));

        let key = if let Some(ref ictx) = self.instantiation_ctx {
            // Instantiation mode: compute the key with the instance's module name; write to the instance-local staging table + global resolved_types.
            module_expr_key(&ictx.module_name, expr.0 as u64)
        } else {
            // HM mode: compute the key with the current module name.
            module_expr_key(&self.current_module_name, expr.0 as u64)
        };

        // In instantiation mode, the original HM pass may have set `implicit_this`
        // on the ExprInfo (marking bare identifiers/calls that resolve to implicit
        // `this` field/method access). The instantiation pass re-infers types with
        // concrete type_args but does NOT set up `this_binding_stack`, so
        // `pending_implicit_this` is never set. Preserve the original marker by
        // copying it from the pre-existing ExprInfo.
        let implicit_this = if self.instantiation_ctx.is_some() {
            self.sema_result
                .expr_types
                .get(&key)
                .and_then(|info| info.implicit_this.clone())
        } else {
            None
        };

        let info = ExprInfo {
            ty: resolved,
            const_val: None,
            expr_id: expr.0 as u64,
            type_name: type_name.map(|s| s.into_boxed_str()),
            is_trait_object,
            is_ref_type: is_ref,
            is_raw_ref,
            implicit_this,
        };

        if let Some(ref mut ictx) = self.instantiation_ctx {
            ictx.local_expr_types.insert(key, info);
            self.sema_result.resolved_types.insert(key, resolved);
        } else {
            self.sema_result.put_expr(key, info);
            // Record module ownership for incremental purge (expr_types key).
            let mod_name = self.current_module_name.clone();
            self.sema_result.module_ownership.expr_type_keys
                .entry(mod_name)
                .or_default()
                .insert(key);
        }
    }

    // ── Unified capture analysis ──
    //
    // Computes the capture list (outer variables referenced by a nested scope:
    // lambda / defer / nested function) with per-capture mutability. This is the
    // single source of truth consumed by the IR builder (replacing its own
    // `collect_free_idents_expr` re-scan) and by the assignment-decision-tree
    // simplification.
    //
    // Capture mode rules:
    //   - `var` binding            → Reference (reads/writes reflect latest value)
    //   - `val` binding            → Snapshot  (captures declaration-time value)
    //   - parameter                → Snapshot  (params are immutable; no `var param`)
    //   - `this`                   → Reference (receiver is shared; mutations via it)
    //   - defer bodies             → ALL captures forced to Reference (defer semantics
    //                                require reading the value at exit, not at defer-site;
    //                                this directly resolves the Bug #49 tension where a
    //                                `val` snapshot and defer-latest conflicted on the
    //                                same node).
    //
    // The walk is scope-aware: bindings introduced *inside* the nested scope
    // (block ValDecl/VarDecl, for-loop binders, nested lambda params, pattern
    // binders) are excluded — only names resolved from the enclosing scope count
    // as captures. This correctly handles nested lambdas/defers whose own
    // captures are computed independently.

    /// Collect free identifier names referenced by a nested-scope body, excluding
    /// names bound *inside* the body. `bound` is seeded with the nested scope's
    /// own parameter names (plus `self`-reference for named functions); the walk
    /// adds inner bindings as it descends.
    pub(super) fn collect_free_idents_scoped(
        &self,
        ast: &AstArena<'_>,
        expr: ExprId,
        bound: &mut rustc_hash::FxHashSet<String>,
        out: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        let node = &ast.expr(expr).node;
        match node {
            Ast::Expr::Ident(name) => {
                if !bound.contains(*name) {
                    out.insert(name.to_string());
                }
            }
            Ast::Expr::Assign { target, value } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Expr::Binary { lhs, rhs, .. } => {
                self.collect_free_idents_scoped(ast, *lhs, bound, out);
                self.collect_free_idents_scoped(ast, *rhs, bound, out);
            }
            Ast::Expr::CompoundAssign { target, value, .. } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Expr::Unary { operand, .. } => {
                self.collect_free_idents_scoped(ast, *operand, bound, out);
            }
            Ast::Expr::Call { callee, args, .. } => {
                self.collect_free_idents_scoped(ast, *callee, bound, out);
                for &a in args {
                    self.collect_free_idents_scoped(ast, a, bound, out);
                }
            }
            Ast::Expr::MethodCall { recv, args, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
                for &a in args {
                    self.collect_free_idents_scoped(ast, a, bound, out);
                }
            }
            Ast::Expr::FieldAccess { recv, .. }
            | Ast::Expr::RefOf(recv)
            | Ast::Expr::Deref(recv)
            | Ast::Expr::NonNullAssert(recv)
            | Ast::Expr::Propagate(recv)
            | Ast::Expr::As { expr: recv, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
            }
            Ast::Expr::Index { recv, index } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
                self.collect_free_idents_scoped(ast, *index, bound, out);
            }
            Ast::Expr::SafeAccess { recv, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
            }
            Ast::Expr::SafeMethodCall { recv, args, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
                for &a in args {
                    self.collect_free_idents_scoped(ast, a, bound, out);
                }
            }
            Ast::Expr::Slice { recv, start, end, .. } => {
                self.collect_free_idents_scoped(ast, *recv, bound, out);
                self.collect_free_idents_scoped(ast, *start, bound, out);
                self.collect_free_idents_scoped(ast, *end, bound, out);
            }
            Ast::Expr::Elvis { lhs, rhs } => {
                self.collect_free_idents_scoped(ast, *lhs, bound, out);
                self.collect_free_idents_scoped(ast, *rhs, bound, out);
            }
            Ast::Expr::If { cond, then_branch, else_branch } => {
                self.collect_free_idents_scoped(ast, *cond, bound, out);
                self.collect_free_idents_scoped(ast, *then_branch, bound, out);
                if let Some(eb) = else_branch {
                    self.collect_free_idents_scoped(ast, *eb, bound, out);
                }
            }
            Ast::Expr::Block { stmts, trailing } => {
                // Block introduces a new lexical scope: snapshot `bound`, recurse
                // (inner ValDecl/VarDecl names are added to `bound` by the stmt
                // walker), then restore.
                let snapshot = bound.clone();
                for &s in stmts {
                    self.collect_free_idents_stmt_scoped(ast, s, bound, out);
                }
                if let Some(t) = trailing {
                    self.collect_free_idents_scoped(ast, *t, bound, out);
                }
                *bound = snapshot;
            }
            Ast::Expr::Match { scrutinee, arms } => {
                self.collect_free_idents_scoped(ast, *scrutinee, bound, out);
                for arm in arms {
                    let snapshot = bound.clone();
                    self.collect_pattern_binders(ast, arm.pattern, bound);
                    if let Some(g) = arm.guard {
                        self.collect_free_idents_scoped(ast, g, bound, out);
                    }
                    self.collect_free_idents_scoped(ast, arm.body, bound, out);
                    *bound = snapshot;
                }
            }
            Ast::Expr::Lambda { params, body, .. } => {
                // Nested lambda: its params belong to it, not to the outer scope.
                let snapshot = bound.clone();
                for p in params {
                    bound.insert(p.name.to_string());
                }
                match body {
                    Ast::LambdaBody::Block(b) => self.collect_free_idents_scoped(ast, *b, bound, out),
                    Ast::LambdaBody::Expression(e) => self.collect_free_idents_scoped(ast, *e, bound, out),
                }
                *bound = snapshot;
            }
            Ast::Expr::ArrayLit { elements, fill } => {
                for &e in elements {
                    self.collect_free_idents_scoped(ast, e, bound, out);
                }
                if let Some((v, c)) = fill {
                    self.collect_free_idents_scoped(ast, *v, bound, out);
                    self.collect_free_idents_scoped(ast, *c, bound, out);
                }
            }
            Ast::Expr::RecordLit(fields) => {
                for f in fields {
                    self.collect_free_idents_scoped(ast, f.value, bound, out);
                }
            }
            Ast::Expr::RecordExtend { base, updates } => {
                self.collect_free_idents_scoped(ast, *base, bound, out);
                for f in updates {
                    self.collect_free_idents_scoped(ast, f.value, bound, out);
                }
            }
            Ast::Expr::StrInterp(parts) => {
                for part in parts {
                    if let Ast::InterpolationPart::Expression(e) = part {
                        self.collect_free_idents_scoped(ast, *e, bound, out);
                    }
                }
            }
            Ast::Expr::Select(arms) => {
                for arm in arms {
                    match arm {
                        Ast::SelectArm::Receive { channel_expr, binding, body } => {
                            self.collect_free_idents_scoped(ast, *channel_expr, bound, out);
                            // The `binding` name is local to the arm body.
                            let snapshot = bound.clone();
                            if let Some(b) = binding {
                                bound.insert(b.to_string());
                            }
                            self.collect_free_idents_scoped(ast, *body, bound, out);
                            *bound = snapshot;
                        }
                        Ast::SelectArm::Timeout { duration, body } => {
                            self.collect_free_idents_scoped(ast, *duration, bound, out);
                            self.collect_free_idents_scoped(ast, *body, bound, out);
                        }
                    }
                }
            }
            Ast::Expr::Atomic(inner) | Ast::Expr::Lazy(inner) => {
                self.collect_free_idents_scoped(ast, *inner, bound, out);
            }
            Ast::Expr::InlineTrait(methods) => {
                for m in methods {
                    if let Some(body_expr) = m.body {
                        self.collect_free_idents_scoped(ast, body_expr, bound, out);
                    }
                }
            }
            // Literals and other leaf variants: no sub-expressions.
            _ => {}
        }
    }

    /// Statement-level scoped free-identifier collection (companion to
    /// `collect_free_idents_scoped`).
    pub(super) fn collect_free_idents_stmt_scoped(
        &self,
        ast: &AstArena<'_>,
        stmt: StmtId,
        bound: &mut rustc_hash::FxHashSet<String>,
        out: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        let node = &ast.stmt(stmt).node;
        match node {
            Ast::Stmt::ValDecl { name, value, .. }
            | Ast::Stmt::VarDecl { name, value, .. } => {
                self.collect_free_idents_scoped(ast, *value, bound, out);
                bound.insert(name.to_string());
            }
            Ast::Stmt::Expression { expr } => {
                self.collect_free_idents_scoped(ast, *expr, bound, out);
            }
            Ast::Stmt::Assignment { target, value } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Stmt::FieldAssignment { object, value, .. } => {
                self.collect_free_idents_scoped(ast, *object, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Stmt::CompoundAssignment { target, value, .. } => {
                self.collect_free_idents_scoped(ast, *target, bound, out);
                self.collect_free_idents_scoped(ast, *value, bound, out);
            }
            Ast::Stmt::Return { value } => {
                if let Some(v) = value {
                    self.collect_free_idents_scoped(ast, *v, bound, out);
                }
            }
            Ast::Stmt::Throw { expr } => {
                self.collect_free_idents_scoped(ast, *expr, bound, out);
            }
            Ast::Stmt::For { name, iterable, body } => {
                self.collect_free_idents_scoped(ast, *iterable, bound, out);
                let snapshot = bound.clone();
                bound.insert(name.to_string());
                self.collect_free_idents_scoped(ast, *body, bound, out);
                *bound = snapshot;
            }
            Ast::Stmt::While { condition, body } => {
                self.collect_free_idents_scoped(ast, *condition, bound, out);
                self.collect_free_idents_scoped(ast, *body, bound, out);
            }
            Ast::Stmt::Loop { body } => {
                self.collect_free_idents_scoped(ast, *body, bound, out);
            }
            Ast::Stmt::Defer { expr } => {
                self.collect_free_idents_scoped(ast, *expr, bound, out);
            }
            Ast::Stmt::Break | Ast::Stmt::Continue => {}
            Ast::Stmt::LocalDecl { decl } => match decl.as_ref() {
                Ast::Decl::FunDecl { name, params, body, .. } => {
                    // A nested function declaration binds its own name in the
                    // current scope. Its params belong to it. References inside
                    // its body that resolve to the *current* scope are captures
                    // of the current scope — record them, treating the nested
                    // function's name + params as bound.
                    bound.insert(name.to_string());
                    let snapshot = bound.clone();
                    for p in params {
                        bound.insert(p.name.to_string());
                    }
                    self.collect_free_idents_scoped(ast, *body, bound, out);
                    *bound = snapshot;
                }
                _ => {}
            },
        }
    }

    /// Add pattern-bound variable names to `bound`.
    pub(super) fn collect_pattern_binders(
        &self,
        ast: &AstArena<'_>,
        pat: PatternId,
        bound: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        let node = &ast.pattern(pat).node;
        match node {
            Ast::Pattern::Variable { name } => {
                bound.insert(name.to_string());
            }
            Ast::Pattern::Wildcard | Ast::Pattern::Literal(_) => {}
            Ast::Pattern::Constructor { patterns, .. } => {
                for &p in patterns {
                    self.collect_pattern_binders(ast, p, bound);
                }
            }
            Ast::Pattern::Record { fields } => {
                for f in fields {
                    self.collect_pattern_binders(ast, f.pattern, bound);
                }
            }
            Ast::Pattern::OrPattern { left, right } => {
                self.collect_pattern_binders(ast, *left, bound);
                self.collect_pattern_binders(ast, *right, bound);
            }
            Ast::Pattern::Guard { pattern, .. } => {
                self.collect_pattern_binders(ast, *pattern, bound);
            }
        }
    }

    /// Compute the capture list for a nested scope.
    ///
    /// `body_expr` is the scope's body expression; `param_names` are the scope's
    /// own parameter names (excluded from captures); `force_reference` overrides
    /// all captures to `Reference` mode (used for defer).
    pub(super) fn compute_captures(
        &self,
        ast: &AstArena<'_>,
        body_expr: ExprId,
        param_names: &[&str],
        force_reference: bool,
    ) -> Vec<crate::sema::Sema::CaptureInfo> {
        use crate::sema::Sema::{CaptureInfo, CaptureMode};

        let mut bound: rustc_hash::FxHashSet<String> = param_names.iter().map(|s| s.to_string()).collect();
        let mut free: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
        self.collect_free_idents_scoped(ast, body_expr, &mut bound, &mut free);

        // Implicit-this: if we're inside a method body and any free identifier
        // is not a known local/var (i.e. it resolves to an implicit `this.field`),
        // the nested scope needs to capture `this` so escaped closures can reach
        // the receiver. Add `this` to the capture set.
        if self.current_this_type().is_some() && !free.contains("this") {
            let any_implicit = free.iter().any(|name| {
                name.as_str() != "this"
                    && !self.local_mutability.iter().any(|((_, n), _)| n.as_str() == name.as_str())
            });
            if any_implicit {
                free.insert("this".to_string());
            }
        }

        // Deterministic order for stable output.
        let mut names: Vec<String> = free.into_iter().collect();
        names.sort();

        names
            .into_iter()
            .map(|name| {
                let mode = if force_reference {
                    CaptureMode::Reference
                } else {
                    self.capture_mode_for(&name)
                };
                CaptureInfo {
                    name: name.into_boxed_str(),
                    decl_key: 0, // resolved by name on the IR side (lookup_var)
                    mode,
                }
            })
            .collect()
    }

    /// Decide the capture mode for a referenced variable based on its declared
    /// mutability. `this` is always `Reference` (shared receiver). `var`
    /// bindings are `Reference`; `val`/params/unresolved default to `Snapshot`.
    pub(super) fn capture_mode_for(&self, name: &str) -> crate::sema::Sema::CaptureMode {
        use crate::sema::Sema::CaptureMode;
        if name == "this" {
            return CaptureMode::Reference;
        }
        // Search local_mutability for any entry with this name (the env_id
        // component is not known here; a `var` binding will have an entry with
        // `is_mutable == true` somewhere in the table).
        let is_var = self
            .local_mutability
            .iter()
            .any(|((_, n), mutable)| n.as_str() == name && *mutable);
        if is_var {
            CaptureMode::Reference
        } else {
            CaptureMode::Snapshot
        }
    }

    // ── infer_expr ──

}
