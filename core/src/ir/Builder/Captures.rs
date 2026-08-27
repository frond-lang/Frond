//! Captures — Free-identifier collection for capture analysis. Mechanically split from Builder.rs (no logic changes).

use super::*;

impl<'a> IrBuilder<'a> {
    /// Recursively collect all Ident names in an expression (deduplicated, preserving
    /// first-occurrence order).
    ///
    /// A simplified free-variable analysis: traverse common Expr variants collecting identifier
    /// references; the caller excludes lambda parameters and checks outer-scope bindings.
    pub(super) fn collect_free_idents_expr(&self, expr_id: crate::ast::Ast::ExprId, names: &mut Vec<String>) {
        use crate::ast::Ast::LambdaBody;
        let spanned = self.current_module().arena.expr(expr_id);
        match &spanned.node {
            crate::ast::Ast::Expr::Ident(name) => {
                if !names.iter().any(|n| n == name) {
                    names.push((*name).to_string());
                }
            }
            crate::ast::Ast::Expr::Binary { lhs, rhs, .. } => {
                self.collect_free_idents_expr(*lhs, names);
                self.collect_free_idents_expr(*rhs, names);
            }
            crate::ast::Ast::Expr::Unary { operand, .. } => {
                self.collect_free_idents_expr(*operand, names);
            }
            crate::ast::Ast::Expr::As { expr, .. } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Expr::Call { callee, args, .. } => {
                self.collect_free_idents_expr(*callee, names);
                for &a in args {
                    self.collect_free_idents_expr(a, names);
                }
            }
            crate::ast::Ast::Expr::MethodCall { recv, args, .. } => {
                self.collect_free_idents_expr(*recv, names);
                for &a in args {
                    self.collect_free_idents_expr(a, names);
                }
            }
            crate::ast::Ast::Expr::FieldAccess { recv, .. }
            | crate::ast::Ast::Expr::SafeAccess { recv, .. } => {
                self.collect_free_idents_expr(*recv, names);
            }
            crate::ast::Ast::Expr::Index { recv, index } => {
                self.collect_free_idents_expr(*recv, names);
                self.collect_free_idents_expr(*index, names);
            }
            crate::ast::Ast::Expr::Assign { target, value } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Expr::CompoundAssign { target, value, .. } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Expr::RecordLit(fields) => {
                for f in fields {
                    self.collect_free_idents_expr(f.value, names);
                }
            }
            crate::ast::Ast::Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.collect_free_idents_expr(*cond, names);
                self.collect_free_idents_expr(*then_branch, names);
                if let Some(e) = else_branch {
                    self.collect_free_idents_expr(*e, names);
                }
            }
            crate::ast::Ast::Expr::Block { stmts, trailing } => {
                for &s in stmts {
                    self.collect_free_idents_stmt(s, names);
                }
                if let Some(t) = trailing {
                    self.collect_free_idents_expr(*t, names);
                }
            }
            crate::ast::Ast::Expr::Lambda { body, .. } => {
                let inner = match body {
                    LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
                };
                self.collect_free_idents_expr(inner, names);
            }
            crate::ast::Ast::Expr::Match { scrutinee, arms } => {
                self.collect_free_idents_expr(*scrutinee, names);
                for arm in arms {
                    if let Some(g) = arm.guard {
                        self.collect_free_idents_expr(g, names);
                    }
                    self.collect_free_idents_expr(arm.body, names);
                }
            }
            // Single-operand expressions: RefOf/Deref/Propagate/NonNullAssert/Atomic/Lazy.
            crate::ast::Ast::Expr::RefOf(inner)
            | crate::ast::Ast::Expr::Deref(inner)
            | crate::ast::Ast::Expr::Propagate(inner)
            | crate::ast::Ast::Expr::NonNullAssert(inner)
            | crate::ast::Ast::Expr::Atomic(inner)
            | crate::ast::Ast::Expr::Lazy(inner) => {
                self.collect_free_idents_expr(*inner, names);
            }
            // Elvis: `lhs ?: rhs`.
            crate::ast::Ast::Expr::Elvis { lhs, rhs } => {
                self.collect_free_idents_expr(*lhs, names);
                self.collect_free_idents_expr(*rhs, names);
            }
            // Slice: `recv[start..end]` (`inclusive` does not affect ident collection).
            crate::ast::Ast::Expr::Slice { recv, start, end, .. } => {
                self.collect_free_idents_expr(*recv, names);
                self.collect_free_idents_expr(*start, names);
                self.collect_free_idents_expr(*end, names);
            }
            // Safe method call: `recv?.method(args)`.
            crate::ast::Ast::Expr::SafeMethodCall { recv, args, .. } => {
                self.collect_free_idents_expr(*recv, names);
                for &a in args {
                    self.collect_free_idents_expr(a, names);
                }
            }
            // Record extension: `{ base with x: 1, ... }`.
            crate::ast::Ast::Expr::RecordExtend { base, updates } => {
                self.collect_free_idents_expr(*base, names);
                for f in updates {
                    self.collect_free_idents_expr(f.value, names);
                }
            }
            // Array literal `fill` clause: `[value, ..count]`.
            crate::ast::Ast::Expr::ArrayLit { elements, fill } => {
                for &e in elements {
                    self.collect_free_idents_expr(e, names);
                }
                if let Some((v, c)) = fill {
                    self.collect_free_idents_expr(*v, names);
                    self.collect_free_idents_expr(*c, names);
                }
            }
            // String interpolation: may contain `{expr}`.
            crate::ast::Ast::Expr::StrInterp(parts) => {
                for part in parts {
                    if let crate::ast::Ast::InterpolationPart::Expression(e) = part {
                        self.collect_free_idents_expr(*e, names);
                    }
                }
            }
            // select expression: each branch contains channel_expr/duration + body.
            crate::ast::Ast::Expr::Select(arms) => {
                for arm in arms {
                    match arm {
                        crate::ast::Ast::SelectArm::Receive { channel_expr, body, .. } => {
                            self.collect_free_idents_expr(*channel_expr, names);
                            self.collect_free_idents_expr(*body, names);
                        }
                        crate::ast::Ast::SelectArm::Timeout { duration, body } => {
                            self.collect_free_idents_expr(*duration, names);
                            self.collect_free_idents_expr(*body, names);
                        }
                    }
                }
            }
            // inline_trait: method bodies may reference outer variables.
            crate::ast::Ast::Expr::InlineTrait(methods) => {
                for m in methods {
                    if let Some(body_expr) = m.body {
                        self.collect_free_idents_expr(body_expr, names);
                    }
                }
            }
            // Constant / no-subexpression variants: IntLit/FloatLit/BoolLit/CharLit/StrLit/NullLit/VoidLit.
            _ => {}
        }
    }

    /// Recursively collect Ident names in a statement (statement version of
    /// `collect_free_idents_expr`).
    pub(super) fn collect_free_idents_stmt(&self, stmt_id: crate::ast::Ast::StmtId, names: &mut Vec<String>) {
        let spanned = self.current_module().arena.stmt(stmt_id);
        match &spanned.node {
            crate::ast::Ast::Stmt::ValDecl { value, .. }
            | crate::ast::Ast::Stmt::VarDecl { value, .. } => {
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::Expression { expr } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Stmt::Assignment { target, value } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::FieldAssignment { object, value, .. } => {
                self.collect_free_idents_expr(*object, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::CompoundAssignment { target, value, .. } => {
                self.collect_free_idents_expr(*target, names);
                self.collect_free_idents_expr(*value, names);
            }
            crate::ast::Ast::Stmt::Return { value } => {
                if let Some(v) = value {
                    self.collect_free_idents_expr(*v, names);
                }
            }
            crate::ast::Ast::Stmt::Throw { expr } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Stmt::For { iterable, body, .. } => {
                self.collect_free_idents_expr(*iterable, names);
                self.collect_free_idents_expr(*body, names);
            }
            crate::ast::Ast::Stmt::While { condition, body } => {
                self.collect_free_idents_expr(*condition, names);
                self.collect_free_idents_expr(*body, names);
            }
            crate::ast::Ast::Stmt::Loop { body } => {
                self.collect_free_idents_expr(*body, names);
            }
            crate::ast::Ast::Stmt::Defer { expr } => {
                self.collect_free_idents_expr(*expr, names);
            }
            crate::ast::Ast::Stmt::Break | crate::ast::Ast::Stmt::Continue => {}
            crate::ast::Ast::Stmt::LocalDecl { decl } => match decl.as_ref() {
                crate::ast::Ast::Decl::FunDecl { body, .. } => {
                    self.collect_free_idents_expr(*body, names);
                }
                _ => {}
            },
        }
    }

}

// =========================================================================
// Place-model C1-① pre-pass: address-taken / lambda-captured name collection
// =========================================================================
//
// Collects, for the function being compiled:
//   address_taken   — names under `&name` OUTSIDE lambda / nested-function
//                     bodies (those decls get Cell-backed at the DECL SITE)
//   lambda_captured — names READ inside lambda / nested-function bodies (a
//                     binding both address-taken and captured stays plain:
//                     the capture machinery snapshots the binding node, which
//                     must remain the raw value, not a Cell Arc)
//
// Lambda-parameter names are excluded from lambda_captured (they are the
// lambda's OWN bindings). The walk is exhaustive with NO wildcard arm: a new
// Expr/Stmt variant breaks this build, forcing the walker to be extended.

impl<'a> IrBuilder<'a> {
    pub(super) fn collect_place_names(
        &self,
        body: crate::ast::Ast::ExprId,
        address_taken: &mut rustc_hash::FxHashSet<String>,
        lambda_captured: &mut rustc_hash::FxHashSet<String>,
    ) {
        let arena = &self.current_module().arena;
        let mut lambda_params: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
        Self::place_walk_expr(arena, body, false, address_taken, lambda_captured, &mut lambda_params);
        for p in lambda_params {
            lambda_captured.remove(&p);
        }
    }

    fn place_walk_stmt(
        arena: &crate::ast::Ast::AstArena<'_>,
        stmt: crate::ast::Ast::StmtId,
        in_lambda: bool,
        at: &mut rustc_hash::FxHashSet<String>,
        lc: &mut rustc_hash::FxHashSet<String>,
        lp: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        match &arena.stmt(stmt).node {
            Ast::Stmt::ValDecl { value, .. } | Ast::Stmt::VarDecl { value, .. } => {
                Self::place_walk_expr(arena, *value, in_lambda, at, lc, lp);
            }
            Ast::Stmt::Expression { expr } => {
                Self::place_walk_expr(arena, *expr, in_lambda, at, lc, lp);
            }
            Ast::Stmt::Assignment { target, value } => {
                Self::place_walk_expr(arena, *target, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *value, in_lambda, at, lc, lp);
            }
            Ast::Stmt::FieldAssignment { object, value, .. } => {
                Self::place_walk_expr(arena, *object, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *value, in_lambda, at, lc, lp);
            }
            Ast::Stmt::CompoundAssignment { target, value, .. } => {
                Self::place_walk_expr(arena, *target, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *value, in_lambda, at, lc, lp);
            }
            Ast::Stmt::Return { value } => {
                if let Some(v) = value {
                    Self::place_walk_expr(arena, *v, in_lambda, at, lc, lp);
                }
            }
            Ast::Stmt::Defer { expr } | Ast::Stmt::Throw { expr } => {
                Self::place_walk_expr(arena, *expr, in_lambda, at, lc, lp);
            }
            Ast::Stmt::Break | Ast::Stmt::Continue => {}
            Ast::Stmt::For { iterable, body, .. } => {
                Self::place_walk_expr(arena, *iterable, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *body, in_lambda, at, lc, lp);
            }
            Ast::Stmt::While { condition, body } => {
                Self::place_walk_expr(arena, *condition, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *body, in_lambda, at, lc, lp);
            }
            Ast::Stmt::Loop { body } => {
                Self::place_walk_expr(arena, *body, in_lambda, at, lc, lp);
            }
            Ast::Stmt::LocalDecl { decl } => {
                // Nested function: its body is a separate function scope —
                // refs inside target ITS locals; free idents are captures.
                if let Ast::Decl::FunDecl { params, body, .. } = decl.as_ref() {
                    for p in params {
                        lp.insert(p.name.to_string());
                    }
                    Self::place_walk_expr(arena, *body, true, at, lc, lp);
                }
            }
        }
    }

    fn place_walk_expr(
        arena: &crate::ast::Ast::AstArena<'_>,
        expr: crate::ast::Ast::ExprId,
        in_lambda: bool,
        at: &mut rustc_hash::FxHashSet<String>,
        lc: &mut rustc_hash::FxHashSet<String>,
        lp: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        let node = &arena.expr(expr).node;
        match node {
            Ast::Expr::IntLit { .. }
            | Ast::Expr::FloatLit { .. }
            | Ast::Expr::BoolLit(_)
            | Ast::Expr::CharLit(_)
            | Ast::Expr::StrLit(_)
            | Ast::Expr::NullLit
            | Ast::Expr::VoidLit => {}
            Ast::Expr::Ident(name) => {
                if in_lambda {
                    lc.insert((*name).to_string());
                }
            }
            Ast::Expr::RefOf(inner) => {
                if !in_lambda {
                    if let Ast::Expr::Ident(name) = &arena.expr(*inner).node {
                        at.insert((*name).to_string());
                    }
                }
                Self::place_walk_expr(arena, *inner, in_lambda, at, lc, lp);
            }
            Ast::Expr::Deref(inner) => {
                Self::place_walk_expr(arena, *inner, in_lambda, at, lc, lp);
            }
            Ast::Expr::Unary { operand, .. } => {
                Self::place_walk_expr(arena, *operand, in_lambda, at, lc, lp);
            }
            Ast::Expr::As { expr, .. } => {
                Self::place_walk_expr(arena, *expr, in_lambda, at, lc, lp);
            }
            Ast::Expr::Propagate(inner) | Ast::Expr::NonNullAssert(inner) => {
                Self::place_walk_expr(arena, *inner, in_lambda, at, lc, lp);
            }
            Ast::Expr::Atomic(inner) | Ast::Expr::Lazy(inner) => {
                Self::place_walk_expr(arena, *inner, in_lambda, at, lc, lp);
            }
            Ast::Expr::StrInterp(parts) => {
                for part in parts {
                    if let Ast::InterpolationPart::Expression(e) = part {
                        Self::place_walk_expr(arena, *e, in_lambda, at, lc, lp);
                    }
                }
            }
            Ast::Expr::Assign { target, value } => {
                Self::place_walk_expr(arena, *target, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *value, in_lambda, at, lc, lp);
            }
            Ast::Expr::CompoundAssign { target, value, .. } => {
                Self::place_walk_expr(arena, *target, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *value, in_lambda, at, lc, lp);
            }
            Ast::Expr::Binary { lhs, rhs, .. } => {
                Self::place_walk_expr(arena, *lhs, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *rhs, in_lambda, at, lc, lp);
            }
            Ast::Expr::Elvis { lhs, rhs } => {
                Self::place_walk_expr(arena, *lhs, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *rhs, in_lambda, at, lc, lp);
            }
            Ast::Expr::Call { callee, args, .. } => {
                Self::place_walk_expr(arena, *callee, in_lambda, at, lc, lp);
                for a in args {
                    Self::place_walk_expr(arena, *a, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::MethodCall { recv, args, .. } => {
                Self::place_walk_expr(arena, *recv, in_lambda, at, lc, lp);
                for a in args {
                    Self::place_walk_expr(arena, *a, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::SafeMethodCall { recv, args, .. } => {
                Self::place_walk_expr(arena, *recv, in_lambda, at, lc, lp);
                for a in args {
                    Self::place_walk_expr(arena, *a, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::FieldAccess { recv, .. } | Ast::Expr::SafeAccess { recv, .. } => {
                Self::place_walk_expr(arena, *recv, in_lambda, at, lc, lp);
            }
            Ast::Expr::Index { recv, index } => {
                Self::place_walk_expr(arena, *recv, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *index, in_lambda, at, lc, lp);
            }
            Ast::Expr::Slice { recv, start, end, .. } => {
                Self::place_walk_expr(arena, *recv, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *start, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *end, in_lambda, at, lc, lp);
            }
            Ast::Expr::ArrayLit { elements, fill } => {
                for e in elements {
                    Self::place_walk_expr(arena, *e, in_lambda, at, lc, lp);
                }
                if let Some((v, n)) = fill {
                    Self::place_walk_expr(arena, *v, in_lambda, at, lc, lp);
                    Self::place_walk_expr(arena, *n, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::RecordLit(fields) => {
                for f in fields {
                    Self::place_walk_expr(arena, f.value, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::RecordExtend { base, updates } => {
                Self::place_walk_expr(arena, *base, in_lambda, at, lc, lp);
                for f in updates {
                    Self::place_walk_expr(arena, f.value, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::Lambda { params, body, .. } => {
                for p in params {
                    lp.insert(p.name.to_string());
                }
                let body_expr = match body {
                    Ast::LambdaBody::Block(e) | Ast::LambdaBody::Expression(e) => *e,
                };
                Self::place_walk_expr(arena, body_expr, true, at, lc, lp);
            }
            Ast::Expr::If { cond, then_branch, else_branch } => {
                Self::place_walk_expr(arena, *cond, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *then_branch, in_lambda, at, lc, lp);
                if let Some(e) = else_branch {
                    Self::place_walk_expr(arena, *e, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::Block { stmts, trailing } => {
                for s in stmts {
                    Self::place_walk_stmt(arena, *s, in_lambda, at, lc, lp);
                }
                if let Some(t) = trailing {
                    Self::place_walk_expr(arena, *t, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::Match { scrutinee, arms } => {
                Self::place_walk_expr(arena, *scrutinee, in_lambda, at, lc, lp);
                for arm in arms {
                    if let Some(g) = arm.guard {
                        Self::place_walk_expr(arena, g, in_lambda, at, lc, lp);
                    }
                    Self::place_walk_expr(arena, arm.body, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::Select(arms) => {
                for arm in arms {
                    Self::place_walk_select_arm(arena, arm, in_lambda, at, lc, lp);
                }
            }
            Ast::Expr::InlineTrait(methods) => {
                for m in methods {
                    if let Some(b) = m.body {
                        for p in &m.params {
                            lp.insert(p.name.to_string());
                        }
                        Self::place_walk_expr(arena, b, true, at, lc, lp);
                    }
                }
            }
        }
    }

    fn place_walk_select_arm(
        arena: &crate::ast::Ast::AstArena<'_>,
        arm: &crate::ast::Ast::SelectArm<'_>,
        in_lambda: bool,
        at: &mut rustc_hash::FxHashSet<String>,
        lc: &mut rustc_hash::FxHashSet<String>,
        lp: &mut rustc_hash::FxHashSet<String>,
    ) {
        use crate::ast::Ast;
        match arm {
            Ast::SelectArm::Receive { channel_expr, body, .. } => {
                Self::place_walk_expr(arena, *channel_expr, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *body, in_lambda, at, lc, lp);
            }
            Ast::SelectArm::Timeout { duration, body } => {
                Self::place_walk_expr(arena, *duration, in_lambda, at, lc, lp);
                Self::place_walk_expr(arena, *body, in_lambda, at, lc, lp);
            }
        }
    }
}
