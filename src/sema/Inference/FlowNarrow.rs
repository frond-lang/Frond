//! FlowNarrow — Flow facts and null-check narrowing. Mechanically split from Inference.rs (no logic changes).

use super::*;

// =========================================================================
// sema v2: Flow-Sensitive Narrowing — general flow fact system.
//
// Design philosophy (original, not borrowed from Kotlin/TS):
// - Generalize the Zig version's nullable narrowing into a general flow fact system.
// - Supports three narrowing kinds: NonNull / IsCheck / ConstructorMatch.
// - DOD: flow facts use arena indices; scopes are stack-managed.
// - Query: lookup_narrowed(path) -> Option<TypeHandle>.
//
// Relationship with the constraint solver:
// Narrowing constraints are managed by FlowContext and do not enter the solver queue
// directly.
// FlowContext is path-sensitive; the solver is path-insensitive.
// =========================================================================

/// Narrowing kinds: covers all flow-sensitive type refinement scenarios in Kuzo.
#[derive(Debug, Clone)]
pub enum NarrowKind {
    /// Non-null narrowing: `if x != null` → x narrows from `Nullable<T>` to `T`.
    NonNull,
    /// Type-test narrowing: `if x is Type` → x narrows to Type.
    /// (Kuzo's `is` expression, similar to Kotlin's smart cast.)
    IsCheck(TypeHandle),
    /// ADT constructor-match narrowing: `match x { Some(v) => ... }` → x narrows to
    /// `Some<T>`.
    /// (GADT type refinement: after constructor matching, type variables gain concrete
    /// information.)
    ConstructorMatch {
        /// Constructor name (e.g. "Some", "None", "Ok", "Err").
        ctor_name: Box<str>,
        /// Bound sub-pattern variable names (used for sub-pattern type refinement).
        bound_vars: Box<[Box<str>]>,
    },
}

/// Flow fact: a type narrowing assertion about a path at some program point.
///
/// `path` is the expression's canonical path (e.g. "x", "obj.field", "a.b.c"), used to
/// reference the same expression from different positions.
#[derive(Debug, Clone)]
pub struct FlowFact {
    /// Expression path (canonicalized as a string).
    pub path: Box<str>,
    /// Narrowed type.
    pub narrowed_ty: TypeHandle,
    /// Narrowing condition.
    pub kind: NarrowKind,
}

/// Flow fact table: stores all flow facts within the current scope.
///
/// DOD: facts in Vec, by_path indexed by FxHashMap.
#[derive(Default)]
pub struct FlowFactTable {
    facts: Vec<FlowFact>,
    /// Indexed by path: path → fact indices.
    by_path: FxHashMap<Box<str>, Vec<u32>>,
}

impl FlowFactTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a flow fact.
    pub fn add(&mut self, fact: FlowFact) {
        let idx = self.facts.len() as u32;
        self.by_path
            .entry(fact.path.clone())
            .or_default()
            .push(idx);
        self.facts.push(fact);
    }

    /// Looks up the latest flow fact for a path (including kind).
    pub fn lookup_fact(&self, path: &str) -> Option<&FlowFact> {
        self.by_path
            .get(path)
            .and_then(|indices| indices.last())
            .and_then(|&idx| self.facts.get(idx as usize))
    }
}

/// Flow context: stack-managed flow fact scopes.
///
/// Push a new scope when entering an if/match branch; pop when leaving.
/// Queries search from the top of the stack downward (inner scopes shadow outer ones).
pub struct FlowContext {
    scopes: Vec<FlowFactTable>,
}

impl Default for FlowContext {
    fn default() -> Self {
        Self::new()
    }
}

impl FlowContext {
    pub fn new() -> Self {
        FlowContext {
            scopes: vec![FlowFactTable::new()], // Root scope.
        }
    }

    /// Enters a new scope (if/match branch).
    pub fn push_scope(&mut self) {
        self.scopes.push(FlowFactTable::new());
    }

    /// Leaves the current scope.
    ///
    /// Does not pop the root scope (keeps at least one layer).
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Adds a flow fact to the current (top) scope.
    pub fn add_fact(&mut self, fact: FlowFact) {
        if let Some(top) = self.scopes.last_mut() {
            top.add(fact);
        }
    }

    /// Looks up the narrowed type for a path: searches from the top of the stack downward.
    ///
    /// Inner-scope narrowings shadow outer ones (path-sensitive).
    ///
    /// Bug #35: A ConstructorMatch fact indicates the scrutinee variable was narrowed to an
    /// ADT type. If the fact's bound_vars contains `path`, then `path` has been shadowed by
    /// a pattern variable (e.g. in `match w { W3(w) => w * w }` the pattern variable `w`
    /// shadows the parameter `w`). In that case the scrutinee's narrowed type does not apply
    /// to the pattern variable; this fact should be skipped so infer_expr falls back to env
    /// lookup to get the pattern variable's correct field type.
    pub fn lookup_narrowed(&self, path: &str) -> Option<TypeHandle> {
        for scope in self.scopes.iter().rev() {
            if let Some(fact) = scope.lookup_fact(path) {
                if let NarrowKind::ConstructorMatch { bound_vars, .. } = &fact.kind {
                    if bound_vars.iter().any(|v| v.as_ref() == path) {
                        continue;
                    }
                }
                return Some(fact.narrowed_ty);
            }
        }
        None
    }

    /// Looks up the flow fact for a path (including kind): searches from the top of the
    /// stack downward.
    pub fn lookup_fact(&self, path: &str) -> Option<&FlowFact> {
        for scope in self.scopes.iter().rev() {
            if let Some(fact) = scope.lookup_fact(path) {
                return Some(fact);
            }
        }
        None
    }

    /// Current scope depth.
    #[inline]
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// Resets to the root scope (called on function switch).
    pub fn reset(&mut self) {
        self.scopes.truncate(1);
        self.scopes[0] = FlowFactTable::new();
    }
}

/// Extracts flow facts from the condition expression of `if cond { ... }`.
///
/// Returns (then_facts, else_facts):
/// - then_facts: narrowings that hold in the then branch.
/// - else_facts: narrowings that hold in the else branch (condition negated).
///
/// Supports:
/// - `x != null` → then: NonNull(x), else: none.
/// - `x == null` → then: none, else: NonNull(x).
/// - `x is Type` → then: IsCheck(x, Type), else: none.
///
/// (ConstructorMatch is handled by the match expression, not in this function.)
pub fn analyze_null_check_facts(
    arena: &TypeArena,
    ast: &AstArena<'_>,
    cond: ExprId,
    env: EnvId,
    env_arena: &EnvArena,
) -> (Vec<FlowFact>, Vec<FlowFact>) {
    let mut then_facts = Vec::new();
    let mut else_facts = Vec::new();

    // If `path_expr` is a nullable variable path and `null_expr` is a null literal,
    // push a NonNull narrowing fact into `facts`.
    let push_nonnull = |path_expr: ExprId, null_expr: ExprId, facts: &mut Vec<FlowFact>| {
        if let Some(path) = expr_path(ast, path_expr) {
            if matches!(ast.expr(null_expr).node, Expr::NullLit) {
                if let Some(ty) = env_arena.lookup(env, &path) {
                    let resolved = arena.resolve(ty);
                    if let Type::Nullable(_) = arena.get(resolved) {
                        facts.push(FlowFact {
                            path: path.into(),
                            narrowed_ty: arena.nullable_inner(resolved),
                            kind: NarrowKind::NonNull,
                        });
                    }
                }
            }
        }
    };

    let cond_node = &ast.expr(cond).node;
    if let Expr::Binary { op, lhs, rhs } = cond_node {
        match op {
            crate::ast::Ast::BinaryOp::NotEq => {
                // `x != null` / `null != x` → then: NonNull(x).
                push_nonnull(*lhs, *rhs, &mut then_facts);
                push_nonnull(*rhs, *lhs, &mut then_facts);
            }
            crate::ast::Ast::BinaryOp::Eq => {
                // `x == null` / `null == x` → else: NonNull(x).
                push_nonnull(*lhs, *rhs, &mut else_facts);
                push_nonnull(*rhs, *lhs, &mut else_facts);
            }
            _ => {}
        }
    }

    (then_facts, else_facts)
}

/// Extracts the canonical path of an expression (used as the flow narrowing identifier).
///
/// Supports:
/// - `Ident(name)` → `name`.
/// - `FieldAccess(recv, field)` → `{recv_path}.{field}`.
/// - Others → None (not narrowable).
pub(super) fn expr_path(ast: &AstArena<'_>, expr: ExprId) -> Option<String> {
    match &ast.expr(expr).node {
        Expr::Ident(name) => Some((*name).to_string()),
        Expr::FieldAccess { recv, field } => {
            let recv_path = expr_path(ast, *recv)?;
            Some(format!("{}.{}", recv_path, field))
        }
        _ => None,
    }
}

/// Extracts the constructor name and bound variable names from a pattern node (for
/// ConstructorMatch narrowing).
///
/// Only handles `Constructor { name, patterns }` patterns: extracts the constructor name and
/// the names of all `Variable` bindings in sub-patterns.
///
/// Other patterns (Wildcard/Literal/Variable/Record/Or/Guard) return None.
pub(super) fn extract_constructor_pattern<'a>(
    pattern: &crate::ast::Ast::Pattern<'a>,
    ast: &'a crate::ast::Ast::AstArena<'a>,
) -> Option<(&'a str, Vec<Box<str>>)> {
    match pattern {
        crate::ast::Ast::Pattern::Constructor { name, patterns } => {
            let mut bound_vars = Vec::new();
            for &pref in patterns {
                collect_pattern_binds(ast, pref, &mut bound_vars);
            }
            Some((*name, bound_vars))
        }
        _ => None,
    }
}

/// Recursively collects the bound variable names in a pattern (the `name` of Variable patterns).
pub(super) fn collect_pattern_binds<'a>(
    ast: &'a crate::ast::Ast::AstArena<'a>,
    pref: crate::ast::Ast::PatternRef,
    out: &mut Vec<Box<str>>,
) {
    match &ast.pattern(pref).node {
        crate::ast::Ast::Pattern::Variable { name } => {
            out.push((*name).into());
        }
        crate::ast::Ast::Pattern::Constructor { patterns, .. } => {
            for &p in patterns {
                collect_pattern_binds(ast, p, out);
            }
        }
        crate::ast::Ast::Pattern::Record { fields } => {
            for f in fields {
                collect_pattern_binds(ast, f.pattern, out);
            }
        }
        crate::ast::Ast::Pattern::OrPattern { left, right } => {
            collect_pattern_binds(ast, *left, out);
            collect_pattern_binds(ast, *right, out);
        }
        crate::ast::Ast::Pattern::Guard { pattern, .. } => {
            collect_pattern_binds(ast, *pattern, out);
        }
        _ => {}
    }
}

