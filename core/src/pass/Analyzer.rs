//! Analyzer.rs — post-Sema static analyzer
//!
//! Produces AnalysisReport: dead code/dead variables/dead functions + memoization strategy.
//! Three-layer multi-pass pipeline with rayon parallelism. Type-driven side-effect classification.
//! See docs/superpowers/specs/2026-08-03-analyzer-design.md

use crate::ast::Ast::{
    AstArena, Decl, Expr, ExprId, InterpolationPart, LambdaBody, Module, Pattern, PatternId,
    SelectArm, Stmt, StmtId, Visibility,
};
use crate::sema::Sema::{module_expr_key, ConstVal, SemaResult};
use crate::types::dynamic_type_id;
use rustc_hash::{FxHashMap, FxHashSet};

// =========================================================================
// Handle types (DOD style, consistent with ExprId/StmtId in Ast.rs)
// =========================================================================

/// Function index (index into Module.declarations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

/// Variable definition site index (index into DefUseGraph.defs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

/// Use site index (index into DefUseGraph.uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UseId(pub u32);

// =========================================================================
// DefUseGraph — def-use chains + live variables
// =========================================================================

/// Variable definition site.
#[derive(Debug, Clone)]
pub struct DefNode {
    pub name: String,
    /// The StmtId of the defining statement (ValDecl/VarDecl) or assignment.
    /// Parameters have no corresponding statement; StmtId(0) is used as a placeholder.
    pub stmt: StmtId,
    /// Containing function.
    pub func: FuncId,
    /// Whether this is a mutable binding (var / assignment target).
    pub is_mutable: bool,
}

/// Variable use site.
#[derive(Debug, Clone)]
pub struct UseNode {
    pub var: VarId,
    /// Expression containing the read.
    pub expr: ExprId,
    /// Containing function.
    pub func: FuncId,
}

/// Per-function def-use chains and live variable sets.
#[derive(Debug, Default)]
pub struct DefUseGraph {
    pub defs: Vec<DefNode>,
    pub uses: Vec<UseNode>,
    /// VarId -> list of use sites.
    pub def_to_uses: Vec<Vec<UseId>>,
    /// Variable name -> definition site (most recent definition within the same function). key = (func, name).
    pub name_to_def: FxHashMap<(FuncId, String), VarId>,
    /// Live-in set at function entry (parameters).
    pub live_in: FxHashMap<FuncId, FxHashSet<VarId>>,
    /// Live-out set at function exit (always empty in this analysis, reserved).
    pub live_out: FxHashMap<FuncId, FxHashSet<VarId>>,
    /// Set of global variable names (top-level VarDecl/ValDecl). Assignments to global
    /// variables inside functions do not register local definition sites, avoiding false dead-variable detection.
    pub global_vars: FxHashSet<String>,
}

impl DefUseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a variable definition and returns its VarId.
    pub fn add_def(&mut self, name: &str, stmt: StmtId, func: FuncId, is_mutable: bool) -> VarId {
        let id = VarId(self.defs.len() as u32);
        self.defs.push(DefNode {
            name: name.to_string(),
            stmt,
            func,
            is_mutable,
        });
        self.def_to_uses.push(Vec::new());
        self.name_to_def.insert((func, name.to_string()), id);
        id
    }

    /// Registers a variable use site.
    pub fn add_use(&mut self, var: VarId, expr: ExprId, func: FuncId) -> UseId {
        let id = UseId(self.uses.len() as u32);
        self.uses.push(UseNode { var, expr, func });
        self.def_to_uses[var.0 as usize].push(id);
        id
    }

    /// Looks up the current definition site of a variable name within a function.
    pub fn lookup(&self, func: FuncId, name: &str) -> Option<VarId> {
        self.name_to_def.get(&(func, name.to_string())).copied()
    }

    /// Whether this variable is never read.
    pub fn is_never_read(&self, var: VarId) -> bool {
        self.def_to_uses[var.0 as usize].is_empty()
    }

    /// Checks whether a variable name has any use site within a function (across all definition sites).
    /// Used for closure-captured mutable variables: a new definition site created by assignment may
    /// have no use site, but the same-named variable has use sites at the old definition site (closure
    /// read), so it should not be classified as dead.
    pub fn is_name_ever_read(&self, func: FuncId, name: &str) -> bool {
        for (i, def) in self.defs.iter().enumerate() {
            if def.func == func && def.name == name {
                if !self.def_to_uses[i].is_empty() {
                    return true;
                }
            }
        }
        false
    }
}

// =========================================================================
// CallGraph — call graph + recursion detection
// =========================================================================

/// Reason a function is retained as reachable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReachableReason {
    // -- Definite entry points (always reachable, never eliminated) --
    /// is_entry=true
    Entry,
    /// extern_c_body is not None
    ExternC,
    /// @extern attribute
    ExternAttr,
    // -- Conservative entry points (possibly reachable, retained but reason tagged) --
    /// trait method
    TraitMethod,
    /// type method (method inside an impl block)
    TypeMethod,
    /// public visibility
    Public,
    // -- Reachability propagation results --
    /// Called by a reachable function
    CalledBy(FuncId),
    /// Required by a memoization candidate dependency
    MemoDependency,
}

impl ReachableReason {
    /// Definite entry: never eliminated by reachability analysis.
    pub fn is_definite(&self) -> bool {
        matches!(self, Self::Entry | Self::ExternC | Self::ExternAttr)
    }
    /// Conservative entry: not eliminated during single-module analysis; may be downgraded by cross-module analysis.
    pub fn is_conservative(&self) -> bool {
        matches!(self, Self::TraitMethod | Self::TypeMethod | Self::Public)
    }
}

/// Call graph.
#[derive(Debug, Default)]
pub struct CallGraph {
    pub nodes: Vec<FuncId>,
    /// caller -> [callee]
    pub edges: FxHashMap<FuncId, Vec<FuncId>>,
    /// callee -> [callers] (reverse graph)
    pub reverse: FxHashMap<FuncId, Vec<FuncId>>,
    /// Directly recursive functions
    pub recursive: FxHashSet<FuncId>,
    /// Mutually recursive SCCs (strongly connected components)
    pub mutually_recursive: Vec<FxHashSet<FuncId>>,
    /// Entry/retention reasons
    pub entry_reasons: FxHashMap<FuncId, ReachableReason>,
    /// Function name -> FuncId (FunDecl name + mangled method name "Type.method")
    pub name_to_func: FxHashMap<String, FuncId>,
    /// Call site ExprId -> callee FuncId.
    /// Only records call sites whose callee is a known function in this module (external functions are not recorded).
    pub call_sites: FxHashMap<ExprId, FuncId>,
    /// Method FuncId -> (decl_idx, method_idx), used to locate method bodies from module.declarations.
    pub func_to_method_loc: FxHashMap<FuncId, (usize, usize)>,
    /// Set of method FuncIds (fast check for whether a FuncId is a method).
    pub method_func_ids: FxHashSet<FuncId>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a call edge caller -> callee (deduplicated).
    pub fn add_edge(&mut self, caller: FuncId, callee: FuncId) {
        let v = self.edges.entry(caller).or_default();
        if !v.contains(&callee) {
            v.push(callee);
        }
        let r = self.reverse.entry(callee).or_default();
        if !r.contains(&caller) {
            r.push(caller);
        }
    }

    /// Whether the given FuncId is a method (rather than a FunDecl).
    #[inline]
    pub fn is_method(&self, func: FuncId) -> bool {
        self.method_func_ids.contains(&func)
    }

    /// Retrieves function/method metadata by FuncId (unified entry point, eliminating scattered FunDecl/Method traversal).
    /// FunDecl -> FuncId = decl_idx; Method -> located via func_to_method_loc.
    pub fn get_func_meta<'a>(&self, func: FuncId, module: &'a Module) -> Option<FuncMetaRef<'a>> {
        if let Some(&(decl_idx, method_idx)) = self.func_to_method_loc.get(&func) {
            // Method
            let decl = module.declarations.get(decl_idx)?;
            if let crate::ast::Ast::Decl::TypeDecl { name, methods, .. } = &decl.node {
                let method = methods.get(method_idx)?;
                return Some(FuncMetaRef {
                    name: method.name,
                    params: &method.params,
                    body: method.body?,
                    is_async: method.is_async,
                    visibility: method.visibility,
                    is_entry: false,
                    self_type: Some(name),
                    func_kind: FuncKind::Method(*name, method_idx),
                });
            }
            None
        } else {
            // FunDecl
            let decl = module.declarations.get(func.0 as usize)?;
            if let crate::ast::Ast::Decl::FunDecl {
                name, params, body, is_async, visibility, is_entry, ..
            } = &decl.node
            {
                Some(FuncMetaRef {
                    name,
                    params,
                    body: *body,
                    is_async: *is_async,
                    visibility: *visibility,
                    is_entry: *is_entry,
                    self_type: None,
                    func_kind: FuncKind::Fun(func.0 as usize),
                })
            } else {
                None
            }
        }
    }

    /// Iterates over all functions (FunDecl + Method), returning (FuncId, FuncMetaRef).
    /// All passes use this method for unified traversal without separately handling FunDecl and TypeDecl.methods.
    pub fn iter_funcs<'a>(&'a self, module: &'a Module) -> impl Iterator<Item = (FuncId, FuncMetaRef<'a>)> + 'a {
        self.nodes.iter().filter_map(move |&fid| {
            self.get_func_meta(fid, module).map(|meta| (fid, meta))
        })
    }
}

/// Function kind: FunDecl or Method.
#[derive(Debug, Clone, Copy)]
pub enum FuncKind<'a> {
    /// FunDecl; value is the declarations index.
    Fun(usize),
    /// Method; value is (type_name, method_idx).
    Method(&'a str, usize),
}

/// Function metadata reference (unified access for FunDecl and Method).
#[derive(Debug, Clone, Copy)]
pub struct FuncMetaRef<'a> {
    pub name: &'a str,
    pub params: &'a [crate::ast::Ast::Param<'a>],
    pub body: crate::ast::Ast::ExprId,
    pub is_async: bool,
    pub visibility: crate::ast::Ast::Visibility,
    pub is_entry: bool,
    /// Self type name for methods (None for FunDecl).
    pub self_type: Option<&'a str>,
    pub func_kind: FuncKind<'a>,
}

// =========================================================================
// PurityTable / EscapeTable — Layer 2 outputs
// =========================================================================

/// Function purity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purity {
    /// Pure function: no side effects, result depends only on arguments, can be memoized.
    Pure,
    /// Impure: has I/O/concurrency/communication side effects.
    Impure,
}

/// Purity table: FuncId -> Purity.
#[derive(Debug, Default)]
pub struct PurityTable {
    pub map: FxHashMap<FuncId, Purity>,
}

impl PurityTable {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn put(&mut self, f: FuncId, p: Purity) {
        self.map.insert(f, p);
    }
    pub fn lookup(&self, f: FuncId) -> Option<Purity> {
        self.map.get(&f).copied()
    }
    pub fn is_pure(&self, f: FuncId) -> bool {
        self.lookup(f) == Some(Purity::Pure)
    }
}

/// Variable/allocation escape information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeInfo {
    /// Does not escape: only used within the function; allocation can be eliminated (if unused).
    NoEscape,
    /// Escapes: tagged with escape kind.
    Escapes(EscapeKind),
}

/// Escape kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeKind {
    /// Heap allocation escape (ArrayLit/RecordLit/RecordExtend) -> stack_alloc optimization.
    Alloc,
    /// Lambda escape (tail position return / loop body capture) -> independent function_id via Cell path.
    Lambda { loop_body_capture: bool },
}

/// Escape table: ExprId (allocation site) -> EscapeInfo.
#[derive(Debug, Default)]
pub struct EscapeTable {
    pub map: FxHashMap<ExprId, EscapeInfo>,
}

impl EscapeTable {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn put(&mut self, e: ExprId, info: EscapeInfo) {
        self.map.insert(e, info);
    }
    pub fn lookup(&self, e: ExprId) -> Option<EscapeInfo> {
        self.map.get(&e).copied()
    }
    pub fn is_no_escape(&self, e: ExprId) -> bool {
        self.lookup(e) == Some(EscapeInfo::NoEscape)
    }
}

// =========================================================================
// SideEffect — type-driven side-effect classification
// =========================================================================

/// Expression side-effect classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideEffect {
    /// No side effects; can be eliminated.
    Pure,
    /// Has side effects; cannot be eliminated.
    Impure,
    /// Allocates but does not escape; no externally observable side effects.
    AllocNoEscape,
}

/// Resolves the receiver type of a method call via sema.expr_types and constructs the mangled name `TypeName.method`.
///
/// Used for call graph edge construction and side-effect classification. Returns None if the receiver type cannot be resolved.
fn resolve_method_mangled(recv: ExprId, method: &str, module_name: &str, sema: &SemaResult) -> Option<String> {
    let key = module_expr_key(module_name, recv.0 as u64);
    let info = sema.expr_types.get(&key)?;
    let type_name = info.type_name.as_deref()?;
    Some(format!("{}.{}", type_name, method))
}

/// Classifies the side effect of a single expression.
///
/// Recursively classifies sub-expressions: only when all sub-expressions are Pure/AllocNoEscape can the result be Pure.
/// Function calls consult the PurityTable; allocations consult the EscapeTable; field accesses consult Type mutability.
/// Method calls resolve the implementation function via sema and check its purity.
pub fn classify_side_effect(
    expr_id: ExprId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
) -> SideEffect {
    let expr = &arena.expr(expr_id).node;
    match expr {
        // -- Pure leaves --
        Expr::IntLit { .. }
        | Expr::FloatLit { .. }
        | Expr::BoolLit(_)
        | Expr::CharLit(_)
        | Expr::StrLit(_)
        | Expr::NullLit
        | Expr::VoidLit
        | Expr::Ident(_) => SideEffect::Pure,

        // -- Pure unary operations (recursively classify operand) --
        Expr::Unary { operand, .. } => classify_side_effect(
            *operand, arena, module_name, sema, purity, escape, func_name_to_id,
        ),
        // -- Type cast `expr as T`: pure (recursively classify operand) --
        Expr::As { expr, .. } => classify_side_effect(
            *expr, arena, module_name, sema, purity, escape, func_name_to_id,
        ),
        Expr::RefOf(inner) | Expr::Deref(inner) | Expr::NonNullAssert(inner) => classify_side_effect(
            *inner, arena, module_name, sema, purity, escape, func_name_to_id,
        ),

        // -- Binary operations (recursively classify both sides) --
        Expr::Binary { lhs, rhs, .. } => {
            let l = classify_side_effect(*lhs, arena, module_name, sema, purity, escape, func_name_to_id);
            let r = classify_side_effect(*rhs, arena, module_name, sema, purity, escape, func_name_to_id);
            combine(l, r)
        }

        // -- if expression: pure only if condition + branches are all pure --
        Expr::If { cond, then_branch, else_branch } => {
            let c = classify_side_effect(*cond, arena, module_name, sema, purity, escape, func_name_to_id);
            let t = classify_side_effect(*then_branch, arena, module_name, sema, purity, escape, func_name_to_id);
            let mut acc = combine(c, t);
            if let Some(e) = else_branch {
                acc = combine(acc, classify_side_effect(*e, arena, module_name, sema, purity, escape, func_name_to_id));
            }
            acc
        }

        // -- Block: pure only if all statements + trailing are pure --
        Expr::Block { stmts, trailing } => {
            let mut acc = SideEffect::Pure;
            for s in stmts {
                acc = combine(acc, classify_stmt_side_effect(*s, arena, module_name, sema, purity, escape, func_name_to_id));
                if acc == SideEffect::Impure {
                    return SideEffect::Impure;
                }
            }
            if let Some(t) = trailing {
                acc = combine(acc, classify_side_effect(*t, arena, module_name, sema, purity, escape, func_name_to_id));
            }
            acc
        }

        // -- Field access: pure if receiver is pure --
        Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => classify_side_effect(
            *recv, arena, module_name, sema, purity, escape, func_name_to_id,
        ),

        // -- Function call: consult PurityTable (includes sema is_async/is_throwing checks) --
        Expr::Call { callee, args, .. } => {
            let callee_purity = if let Expr::Ident(name) = &arena.expr(*callee).node {
                // sema FuncSigInfo lookup: async/throwing functions are always impure.
                // All-owners check: same-named functions across modules each count.
                if sema.func_sigs_named(*name).iter().any(|sig| sig.is_async || sig.is_throwing) {
                    return SideEffect::Impure;
                }
                func_name_to_id.get(*name).and_then(|fid| purity.lookup(*fid))
            } else {
                None
            };
            match callee_purity {
                Some(Purity::Pure) => {
                    let mut acc = SideEffect::Pure;
                    for a in args {
                        acc = combine(acc, classify_side_effect(*a, arena, module_name, sema, purity, escape, func_name_to_id));
                    }
                    acc
                }
                _ => SideEffect::Impure,
            }
        }

        // -- Method call: resolve implementation function via sema and check its purity --
        Expr::MethodCall { recv, method, args, .. } | Expr::SafeMethodCall { recv, method, args, .. } => {
            // Resolve receiver type via sema -> mangled name TypeName.method -> check purity
            let mangled = resolve_method_mangled(*recv, *method, module_name, sema);
            let callee_purity = mangled.as_deref().and_then(|name| {
                // sema MethodSigInfo lookup: async/throwing methods are always impure
                if let Some(dot) = name.rfind('.') {
                    let type_name = &name[..dot];
                    if let Some(method_idx) = sema.lookup_method_idx(type_name, method) {
                        let type_idx = sema.type_def_idx(type_name)?;
                        let type_def = &sema.type_defs[&type_idx];
                        let method_sig = type_def.methods.get(method_idx as usize)?;
                        if method_sig.is_async || method_sig.is_throwing {
                            return Some(Purity::Impure);
                        }
                    }
                }
                func_name_to_id.get(name).and_then(|fid| purity.lookup(*fid))
            });
            match callee_purity {
                Some(Purity::Pure) => {
                    let mut acc = classify_side_effect(*recv, arena, module_name, sema, purity, escape, func_name_to_id);
                    for a in args {
                        acc = combine(acc, classify_side_effect(*a, arena, module_name, sema, purity, escape, func_name_to_id));
                    }
                    acc
                }
                _ => SideEffect::Impure,
            }
        }

        // -- Allocations (array/record literals): consult EscapeTable --
        Expr::ArrayLit { elements, fill } => {
            let mut acc = if escape.is_no_escape(expr_id) {
                SideEffect::AllocNoEscape
            } else {
                return SideEffect::Impure;
            };
            for e in elements {
                acc = combine(acc, classify_side_effect(*e, arena, module_name, sema, purity, escape, func_name_to_id));
                if acc == SideEffect::Impure {
                    return SideEffect::Impure;
                }
            }
            if let Some((val, count)) = fill {
                acc = combine(acc, classify_side_effect(*val, arena, module_name, sema, purity, escape, func_name_to_id));
                acc = combine(acc, classify_side_effect(*count, arena, module_name, sema, purity, escape, func_name_to_id));
            }
            acc
        }
        Expr::RecordLit(fields) => {
            let mut acc = if escape.is_no_escape(expr_id) {
                SideEffect::AllocNoEscape
            } else {
                return SideEffect::Impure;
            };
            for f in fields {
                acc = combine(acc, classify_side_effect(f.value, arena, module_name, sema, purity, escape, func_name_to_id));
                if acc == SideEffect::Impure {
                    return SideEffect::Impure;
                }
            }
            acc
        }
        Expr::RecordExtend { base, updates } => {
            let mut acc = if escape.is_no_escape(expr_id) {
                SideEffect::AllocNoEscape
            } else {
                return SideEffect::Impure;
            };
            acc = combine(acc, classify_side_effect(*base, arena, module_name, sema, purity, escape, func_name_to_id));
            for u in updates {
                acc = combine(acc, classify_side_effect(u.value, arena, module_name, sema, purity, escape, func_name_to_id));
                if acc == SideEffect::Impure {
                    return SideEffect::Impure;
                }
            }
            acc
        }

        // -- Elvis: pure only if both sides are pure --
        Expr::Elvis { lhs, rhs } => {
            let l = classify_side_effect(*lhs, arena, module_name, sema, purity, escape, func_name_to_id);
            let r = classify_side_effect(*rhs, arena, module_name, sema, purity, escape, func_name_to_id);
            combine(l, r)
        }

        // -- Always treated as having side effects --
        Expr::StrInterp(_)
        | Expr::Assign { .. }
        | Expr::CompoundAssign { .. }
        | Expr::Index { .. }
        | Expr::Slice { .. }
        | Expr::Propagate(_)
        | Expr::Lambda { .. }
        | Expr::Match { .. }
        | Expr::Atomic(_)
        | Expr::Lazy(_)
        | Expr::Select(_)
        | Expr::InlineTrait(_) => SideEffect::Impure,
    }
}

/// Classifies the side effect of a statement.
fn classify_stmt_side_effect(
    stmt_id: StmtId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
) -> SideEffect {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::Expression { expr } => classify_side_effect(*expr, arena, module_name, sema, purity, escape, func_name_to_id),
        Stmt::ValDecl { value, .. } | Stmt::VarDecl { value, .. } => {
            classify_side_effect(*value, arena, module_name, sema, purity, escape, func_name_to_id)
        }
        _ => SideEffect::Impure,
    }
}

/// Combines two side-effect classifications: if either is Impure, the result is Impure; otherwise AllocNoEscape takes precedence over Pure.
fn combine(a: SideEffect, b: SideEffect) -> SideEffect {
    match (a, b) {
        (SideEffect::Impure, _) | (_, SideEffect::Impure) => SideEffect::Impure,
        (SideEffect::AllocNoEscape, _) | (_, SideEffect::AllocNoEscape) => SideEffect::AllocNoEscape,
        (SideEffect::Pure, SideEffect::Pure) => SideEffect::Pure,
    }
}

/// Whether the expression has no side effects (Pure or AllocNoEscape are both considered eliminable).
pub fn is_side_effect_free(s: SideEffect) -> bool {
    s != SideEffect::Impure
}

// =========================================================================
// DefUseBuilder — Layer 1: build def-use graph
// =========================================================================

/// Builds the def-use graph. Traverses each function body, collecting ValDecl/VarDecl/Assignment definition
/// sites and Ident use sites. Assignments to global variables (top-level VarDecl/ValDecl) inside functions
/// do not register local definition sites.
pub fn build_def_use(module: &Module, arena: &AstArena) -> DefUseGraph {
    let mut graph = DefUseGraph::new();
    // Collect global variable names (VarDecl/ValDecl nested in top-level ExprDecl)
    for decl in &module.declarations {
        if let Decl::ExprDecl { stmt: Some(stmt_id), .. } = &decl.node {
            let stmt = &arena.stmt(*stmt_id).node;
            if let Stmt::VarDecl { name, .. } | Stmt::ValDecl { name, .. } = stmt {
                graph.global_vars.insert(name.to_string());
            }
        }
    }
    for (idx, decl) in module.declarations.iter().enumerate() {
        if let Decl::FunDecl { params, body, .. } = &decl.node {
            let func = FuncId(idx as u32);
            // Parameters as entry live variables and definition sites (parameters are immutable by default)
            let mut live = FxHashSet::default();
            for p in params {
                // Parameters have no corresponding statement; use StmtId(u32::MAX) as placeholder; DeadVarPass skips based on this
                let v = graph.add_def(p.name, StmtId(u32::MAX), func, false);
                live.insert(v);
            }
            graph.live_in.insert(func, live);
            collect_def_use_expr(*body, arena, func, &mut graph);
        }
    }
    graph
}

/// Recursively collects def-use information in an expression.
fn collect_def_use_expr(expr_id: ExprId, arena: &AstArena, func: FuncId, graph: &mut DefUseGraph) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Ident(name) => {
            if let Some(v) = graph.lookup(func, name) {
                graph.add_use(v, expr_id, func);
            }
        }
        Expr::Block { stmts, trailing } => {
            for s in stmts {
                collect_def_use_stmt(*s, arena, func, graph);
            }
            if let Some(t) = trailing {
                collect_def_use_expr(*t, arena, func, graph);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            collect_def_use_expr(*cond, arena, func, graph);
            collect_def_use_expr(*then_branch, arena, func, graph);
            if let Some(e) = else_branch {
                collect_def_use_expr(*e, arena, func, graph);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_def_use_expr(*lhs, arena, func, graph);
            collect_def_use_expr(*rhs, arena, func, graph);
        }
        Expr::Unary { operand, .. }
        | Expr::As { expr: operand, .. }
        | Expr::RefOf(operand)
        | Expr::Deref(operand)
        | Expr::NonNullAssert(operand)
        | Expr::Propagate(operand)
        | Expr::Atomic(operand)
        | Expr::Lazy(operand) => {
            collect_def_use_expr(*operand, arena, func, graph);
        }
        Expr::Assign { target, value } => {
            if !matches!(&arena.expr(*target).node, Expr::Ident(_)) {
                collect_def_use_expr(*target, arena, func, graph);
            }
            collect_def_use_expr(*value, arena, func, graph);
        }
        Expr::CompoundAssign { target, value, .. } => {
            if let Expr::Ident(name) = &arena.expr(*target).node {
                if let Some(v) = graph.lookup(func, name) {
                    graph.add_use(v, expr_id, func);
                }
            } else {
                collect_def_use_expr(*target, arena, func, graph);
            }
            collect_def_use_expr(*value, arena, func, graph);
        }
        Expr::Call { callee, args, .. } => {
            collect_def_use_expr(*callee, arena, func, graph);
            for a in args {
                collect_def_use_expr(*a, arena, func, graph);
            }
        }
        Expr::MethodCall { recv, args, .. } | Expr::SafeMethodCall { recv, args, .. } => {
            collect_def_use_expr(*recv, arena, func, graph);
            for a in args {
                collect_def_use_expr(*a, arena, func, graph);
            }
        }
        Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => {
            collect_def_use_expr(*recv, arena, func, graph);
        }
        Expr::Index { recv, index } => {
            collect_def_use_expr(*recv, arena, func, graph);
            collect_def_use_expr(*index, arena, func, graph);
        }
        Expr::Slice { recv, start, end, .. } => {
            collect_def_use_expr(*recv, arena, func, graph);
            collect_def_use_expr(*start, arena, func, graph);
            collect_def_use_expr(*end, arena, func, graph);
        }
        Expr::Elvis { lhs, rhs } => {
            collect_def_use_expr(*lhs, arena, func, graph);
            collect_def_use_expr(*rhs, arena, func, graph);
        }
        Expr::ArrayLit { elements, fill } => {
            for e in elements {
                collect_def_use_expr(*e, arena, func, graph);
            }
            if let Some((val, count)) = fill {
                collect_def_use_expr(*val, arena, func, graph);
                collect_def_use_expr(*count, arena, func, graph);
            }
        }
        Expr::RecordLit(fields) => {
            for f in fields {
                collect_def_use_expr(f.value, arena, func, graph);
            }
        }
        Expr::RecordExtend { base, updates } => {
            collect_def_use_expr(*base, arena, func, graph);
            for u in updates {
                collect_def_use_expr(u.value, arena, func, graph);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_def_use_expr(*scrutinee, arena, func, graph);
            for arm in arms {
                // Register variable bindings in patterns (e.g., x in Some(x) => x)
                collect_pattern_binds(arm.pattern, arena, func, graph);
                if let Some(g) = arm.guard {
                    collect_def_use_expr(g, arena, func, graph);
                }
                collect_def_use_expr(arm.body, arena, func, graph);
            }
        }
        Expr::StrInterp(parts) => {
            for p in parts {
                if let InterpolationPart::Expression(e) = p {
                    collect_def_use_expr(*e, arena, func, graph);
                }
            }
        }
        Expr::Lambda { body, .. } => {
            // Lambda parameters are in a separate scope and not registered in the current function's def-use graph;
            // but references to outer variables in the lambda body need to be recorded as use sites (captures).
            let body_expr = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            collect_def_use_expr(body_expr, arena, func, graph);
        }
        Expr::Select(arms) => {
            for arm in arms {
                match arm {
                    SelectArm::Receive { channel_expr, body, .. } => {
                        collect_def_use_expr(*channel_expr, arena, func, graph);
                        collect_def_use_expr(*body, arena, func, graph);
                    }
                    SelectArm::Timeout { duration, body } => {
                        collect_def_use_expr(*duration, arena, func, graph);
                        collect_def_use_expr(*body, arena, func, graph);
                    }
                }
            }
        }
        Expr::InlineTrait(_) => {}
        Expr::IntLit { .. }
        | Expr::FloatLit { .. }
        | Expr::BoolLit(_)
        | Expr::CharLit(_)
        | Expr::StrLit(_)
        | Expr::NullLit
        | Expr::VoidLit => {}
    }
}

/// Recursively collects def-use information in a statement.
fn collect_def_use_stmt(stmt_id: StmtId, arena: &AstArena, func: FuncId, graph: &mut DefUseGraph) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::ValDecl { name, value, .. } => {
            collect_def_use_expr(*value, arena, func, graph);
            graph.add_def(name, stmt_id, func, false);
        }
        Stmt::VarDecl { name, value, .. } => {
            collect_def_use_expr(*value, arena, func, graph);
            graph.add_def(name, stmt_id, func, true);
        }
        Stmt::Assignment { target, value } => {
            // First collect uses in value (may read the old value of target, e.g., x = x + 1)
            collect_def_use_expr(*value, arena, func, graph);
            // Then register the new definition site (overwriting name_to_def so subsequent uses map to the new definition)
            // Assignments to global variables do not register local definition sites (globals are not in function def-use scope)
            if let Expr::Ident(name) = &arena.expr(*target).node {
                if !graph.global_vars.contains(*name) {
                    graph.add_def(name, stmt_id, func, true);
                }
            } else {
                collect_def_use_expr(*target, arena, func, graph);
            }
        }
        Stmt::FieldAssignment { object, value, .. } => {
            collect_def_use_expr(*object, arena, func, graph);
            collect_def_use_expr(*value, arena, func, graph);
        }
        Stmt::CompoundAssignment { target, value, .. } => {
            // Compound assignment x += v: first read old value (use), then collect v, then register new definition (def)
            // Compound assignments to global variables do not register local definition sites
            if let Expr::Ident(name) = &arena.expr(*target).node {
                if let Some(v) = graph.lookup(func, name) {
                    graph.add_use(v, *target, func);
                }
                collect_def_use_expr(*value, arena, func, graph);
                if !graph.global_vars.contains(*name) {
                    graph.add_def(name, stmt_id, func, true);
                }
            } else {
                collect_def_use_expr(*target, arena, func, graph);
                collect_def_use_expr(*value, arena, func, graph);
            }
        }
        Stmt::Expression { expr } => {
            collect_def_use_expr(*expr, arena, func, graph);
        }
        Stmt::Return { value } => {
            if let Some(v) = value {
                collect_def_use_expr(*v, arena, func, graph);
            }
        }
        Stmt::Defer { expr } | Stmt::Throw { expr } => {
            collect_def_use_expr(*expr, arena, func, graph);
        }
        Stmt::For { name, iterable, body } => {
            collect_def_use_expr(*iterable, arena, func, graph);
            graph.add_def(name, stmt_id, func, true);
            collect_def_use_expr(*body, arena, func, graph);
        }
        Stmt::While { condition, body } => {
            collect_def_use_expr(*condition, arena, func, graph);
            collect_def_use_expr(*body, arena, func, graph);
        }
        Stmt::Loop { body } => {
            collect_def_use_expr(*body, arena, func, graph);
        }
        Stmt::LocalDecl { decl } => match decl.as_ref() {
            // Nested function declaration: scan the body for references to outer
            // variables (same as Lambda). Without this, a var captured only by a
            // nested function would be misidentified as dead and skipped by the
            // IR builder, causing the nested function to read `void`.
            crate::ast::Ast::Decl::FunDecl { body, .. } => {
                collect_def_use_expr(*body, arena, func, graph);
            }
            _ => {}
        },
        Stmt::Break | Stmt::Continue => {}
    }
}

/// Recursively collects variable bindings in a pattern and registers them as definition sites.
/// Uses StmtId(u32::MAX) as placeholder (same as parameters); DeadVarPass skips these based on that.
fn collect_pattern_binds(pattern_id: PatternId, arena: &AstArena, func: FuncId, graph: &mut DefUseGraph) {
    let pat = &arena.pattern(pattern_id).node;
    match pat {
        Pattern::Variable { name } => {
            graph.add_def(name, StmtId(u32::MAX), func, false);
        }
        Pattern::Constructor { patterns, .. } => {
            for p in patterns {
                collect_pattern_binds(*p, arena, func, graph);
            }
        }
        Pattern::Record { fields } => {
            for f in fields {
                collect_pattern_binds(f.pattern, arena, func, graph);
            }
        }
        Pattern::OrPattern { left, right } => {
            collect_pattern_binds(*left, arena, func, graph);
            collect_pattern_binds(*right, arena, func, graph);
        }
        Pattern::Guard { pattern, condition } => {
            collect_pattern_binds(*pattern, arena, func, graph);
            collect_def_use_expr(*condition, arena, func, graph);
        }
        Pattern::Wildcard | Pattern::Literal(_) => {}
    }
}

// =========================================================================
// CallGraphBuilder — Layer 1: build call graph + recursion detection + entry marking
// =========================================================================

/// Enum of impure built-in functions: those with I/O/concurrency/communication side effects.
/// Replaces the string slice IMPURE_BUILTINS; this enum is the single source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImpureBuiltinFn {
    Async,
    Lazy,
    Select,
    Send,
    Recv,
}

impl ImpureBuiltinFn {
    /// Looks up the enum by function name (eliminates string slice contains checks).
    #[inline]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "async" => Some(Self::Async),
            "lazy" => Some(Self::Lazy),
            "select" => Some(Self::Select),
            "send" => Some(Self::Send),
            "recv" => Some(Self::Recv),
            _ => None,
        }
    }
}

/// Builds the call graph. Traverses all functions (FunDecl + TypeDecl.methods), collects Call/MethodCall edges,
/// marks entry reasons, and detects recursion. Methods are registered in name_to_func via mangled name "Type.method".
pub fn build_call_graph(module: &Module, arena: &AstArena, sema: &SemaResult) -> CallGraph {
    let module_name = module.name;
    let mut cg = CallGraph::new();
    // First pass: collect all function names -> FuncId
    // FunDecl: FuncId = declarations index
    for (idx, decl) in module.declarations.iter().enumerate() {
        if let Decl::FunDecl { name, .. } = &decl.node {
            let fid = FuncId(idx as u32);
            cg.nodes.push(fid);
            cg.name_to_func.insert(name.to_string(), fid);
        }
    }
    // Method: FuncId = declarations.len() + method_global_idx
    let fun_decl_count = module.declarations.len();
    let mut method_global_idx = 0usize;
    for (decl_idx, decl) in module.declarations.iter().enumerate() {
        if let crate::ast::Ast::Decl::TypeDecl { name: type_name, methods, .. } = &decl.node {
            for (method_idx, method) in methods.iter().enumerate() {
                if method.body.is_some() {
                    let fid = FuncId((fun_decl_count + method_global_idx) as u32);
                    cg.nodes.push(fid);
                    cg.method_func_ids.insert(fid);
                    cg.func_to_method_loc.insert(fid, (decl_idx, method_idx));
                    // Register mangled name "Type.method" (consistent with resolve_method_mangled)
                    let mangled = format!("{}.{}", type_name, method.name);
                    cg.name_to_func.insert(mangled, fid);
                    method_global_idx += 1;
                }
            }
        }
    }
    // Second pass: collect call edges + mark entries (clone nodes to avoid borrow conflicts)
    let method_locs: Vec<(FuncId, usize, usize)> = cg.func_to_method_loc.iter()
        .map(|(&fid, &(d, m))| (fid, d, m))
        .collect();
    let nodes = cg.nodes.clone();
    for &fid in &nodes {
        // Determine whether it is a method and extract metadata: unified return (&str, &[Param], Option<ExprRef>, bool, Visibility, bool, &[Attribute], bool)
        let meta_opt: Option<(&str, &[crate::ast::Ast::Param], Option<crate::ast::Ast::ExprRef>, bool, crate::ast::Ast::Visibility, bool, &[crate::ast::Ast::Attribute], bool)> =
            method_locs.iter().find(|(f, _, _)| *f == fid).and_then(|&(_, decl_idx, method_idx)| {
                let decl = &module.declarations[decl_idx];
                if let crate::ast::Ast::Decl::TypeDecl { methods, .. } = &decl.node {
                    methods.get(method_idx).map(|m| (m.name, &m.params[..], m.body, m.is_async, m.visibility, false, &[][..], false))
                } else {
                    None
                }
            }).or_else(|| {
                module.declarations.get(fid.0 as usize).and_then(|d| {
                    if let Decl::FunDecl { name, params, body, is_async, visibility, is_entry, attributes, extern_c_body, .. } = &d.node {
                        Some((*name, &params[..], Some(*body), *is_async, *visibility, *is_entry, attributes.as_slice(), extern_c_body.is_some()))
                    } else {
                        None
                    }
                })
            });
        if let Some((name, _params, body_opt, _is_async, visibility, is_entry, attrs, ext_c)) = meta_opt {
            mark_entry_reason(&mut cg, fid, is_entry, visibility, ext_c, attrs, name, sema);
            if cg.is_method(fid) {
                cg.entry_reasons.entry(fid).or_insert(ReachableReason::TypeMethod);
            }
            if let Some(body) = body_opt {
                collect_call_edges(body, arena, fid, name, module_name, sema, &mut cg);
            }
        }
    }
    detect_recursion(&mut cg);
    cg
}

/// Marks the reachability entry reason for a function.
fn mark_entry_reason(
    cg: &mut CallGraph,
    func: FuncId,
    is_entry: bool,
    visibility: Visibility,
    has_extern_c: bool,
    attributes: &[crate::ast::Ast::Attribute],
    name: &str,
    sema: &SemaResult,
) {
    if is_entry {
        cg.entry_reasons.insert(func, ReachableReason::Entry);
        return;
    }
    if has_extern_c {
        cg.entry_reasons.insert(func, ReachableReason::ExternC);
        return;
    }
    if attributes.iter().any(|a| a.name == crate::ffi::ATTR_EXTERN) {
        cg.entry_reasons.insert(func, ReachableReason::ExternAttr);
        return;
    }
    // Type method / trait method: name contains '.' (mangled name TypeName.method)
    if let Some(dot) = name.rfind('.') {
        let type_name = &name[..dot];
        let method_name = &name[dot + 1..];
        // Use witness_table to determine if this is a trait method implementation:
        // if the type implements a trait and the method is in the witness_table's method_slots, it is a TraitMethod
        if let Some(type_idx) = sema.type_def_idx(type_name) {
            let type_id = dynamic_type_id(type_idx);
            for entry in sema.witness_table.entries() {
                if entry.type_id == type_id && entry.method_slots.contains_key(method_name) {
                    cg.entry_reasons.insert(func, ReachableReason::TraitMethod);
                    return;
                }
            }
        }
        // Otherwise it is a regular type method
        cg.entry_reasons.insert(func, ReachableReason::TypeMethod);
        return;
    }
    if visibility == Visibility::Public {
        cg.entry_reasons.insert(func, ReachableReason::Public);
        return;
    }
}

/// Recursively collects call edges in a function body.
fn collect_call_edges(
    expr_id: ExprId,
    arena: &AstArena,
    caller: FuncId,
    caller_name: &str,
    module_name: &str,
    sema: &SemaResult,
    cg: &mut CallGraph,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Call { callee, args, .. } => {
            if let Expr::Ident(name) = &arena.expr(*callee).node {
                if *name == caller_name {
                    cg.recursive.insert(caller);
                } else if let Some(&callee_id) = cg.name_to_func.get(*name) {
                    cg.add_edge(caller, callee_id);
                    // Record call site ExprId -> callee function, for use by inline expansion
                    cg.call_sites.insert(expr_id, callee_id);
                }
            }
            collect_call_edges(*callee, arena, caller, caller_name, module_name, sema, cg);
            for a in args {
                collect_call_edges(*a, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::Block { stmts, trailing } => {
            for s in stmts {
                collect_call_edges_stmt(*s, arena, caller, caller_name, module_name, sema, cg);
            }
            if let Some(t) = trailing {
                collect_call_edges(*t, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            collect_call_edges(*cond, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*then_branch, arena, caller, caller_name, module_name, sema, cg);
            if let Some(e) = else_branch {
                collect_call_edges(*e, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_call_edges(*lhs, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*rhs, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::Unary { operand, .. }
        | Expr::As { expr: operand, .. }
        | Expr::RefOf(operand)
        | Expr::Deref(operand)
        | Expr::NonNullAssert(operand)
        | Expr::Propagate(operand)
        | Expr::Atomic(operand)
        | Expr::Lazy(operand) => {
            collect_call_edges(*operand, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::Assign { target, value } | Expr::CompoundAssign { target, value, .. } => {
            collect_call_edges(*target, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*value, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::MethodCall { recv, method, args, .. } | Expr::SafeMethodCall { recv, method, args, .. } => {
            // Resolve method mangled name via sema -> look up name_to_func -> add call edge
            if let Some(mangled) = resolve_method_mangled(*recv, method, module_name, sema) {
                if mangled == caller_name {
                    cg.recursive.insert(caller);
                } else if let Some(&callee_id) = cg.name_to_func.get(&mangled) {
                    cg.add_edge(caller, callee_id);
                }
            }
            collect_call_edges(*recv, arena, caller, caller_name, module_name, sema, cg);
            for a in args {
                collect_call_edges(*a, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => {
            collect_call_edges(*recv, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::Index { recv, index } => {
            collect_call_edges(*recv, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*index, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::Slice { recv, start, end, .. } => {
            collect_call_edges(*recv, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*start, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*end, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::Elvis { lhs, rhs } => {
            collect_call_edges(*lhs, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*rhs, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::ArrayLit { elements, fill } => {
            for e in elements {
                collect_call_edges(*e, arena, caller, caller_name, module_name, sema, cg);
            }
            if let Some((v, c)) = fill {
                collect_call_edges(*v, arena, caller, caller_name, module_name, sema, cg);
                collect_call_edges(*c, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::RecordLit(fields) => {
            for f in fields {
                collect_call_edges(f.value, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::RecordExtend { base, updates } => {
            collect_call_edges(*base, arena, caller, caller_name, module_name, sema, cg);
            for u in updates {
                collect_call_edges(u.value, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_call_edges(*scrutinee, arena, caller, caller_name, module_name, sema, cg);
            for arm in arms {
                if let Some(g) = arm.guard {
                    collect_call_edges(g, arena, caller, caller_name, module_name, sema, cg);
                }
                collect_call_edges(arm.body, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Expr::StrInterp(parts) => {
            for p in parts {
                if let InterpolationPart::Expression(e) = p {
                    collect_call_edges(*e, arena, caller, caller_name, module_name, sema, cg);
                }
            }
        }
        Expr::Select(arms) => {
            for arm in arms {
                match arm {
                    SelectArm::Receive { channel_expr, body, .. } => {
                        collect_call_edges(*channel_expr, arena, caller, caller_name, module_name, sema, cg);
                        collect_call_edges(*body, arena, caller, caller_name, module_name, sema, cg);
                    }
                    SelectArm::Timeout { duration, body } => {
                        collect_call_edges(*duration, arena, caller, caller_name, module_name, sema, cg);
                        collect_call_edges(*body, arena, caller, caller_name, module_name, sema, cg);
                    }
                }
            }
        }
        Expr::Lambda { body, .. } => {
            // Recurse into lambda body: calls in nested lambdas are attributed to the outer caller
            let inner = match body {
                crate::ast::Ast::LambdaBody::Block(e) => *e,
                crate::ast::Ast::LambdaBody::Expression(e) => *e,
            };
            collect_call_edges(inner, arena, caller, caller_name, module_name, sema, cg);
        }
        Expr::InlineTrait(_) => {}
        Expr::IntLit { .. }
        | Expr::FloatLit { .. }
        | Expr::BoolLit(_)
        | Expr::CharLit(_)
        | Expr::StrLit(_)
        | Expr::NullLit
        | Expr::VoidLit
        | Expr::Ident(_) => {}
    }
}

/// Recursively collects call edges in a statement.
fn collect_call_edges_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    caller: FuncId,
    caller_name: &str,
    module_name: &str,
    sema: &SemaResult,
    cg: &mut CallGraph,
) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::ValDecl { value, .. } | Stmt::VarDecl { value, .. } => {
            collect_call_edges(*value, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::Assignment { target, value } | Stmt::CompoundAssignment { target, value, .. } => {
            collect_call_edges(*target, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*value, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::FieldAssignment { object, value, .. } => {
            collect_call_edges(*object, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*value, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::Expression { expr } => {
            collect_call_edges(*expr, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::Return { value } => {
            if let Some(v) = value {
                collect_call_edges(*v, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Stmt::Defer { expr } | Stmt::Throw { expr } => {
            collect_call_edges(*expr, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::For { iterable, body, .. } => {
            collect_call_edges(*iterable, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*body, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::While { condition, body } => {
            collect_call_edges(*condition, arena, caller, caller_name, module_name, sema, cg);
            collect_call_edges(*body, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::Loop { body } => {
            collect_call_edges(*body, arena, caller, caller_name, module_name, sema, cg);
        }
        Stmt::LocalDecl { decl } => {
            // Recurse into nested function body: calls in nested functions are attributed to the outer caller
            if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                collect_call_edges(*body, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Stmt::Break | Stmt::Continue => {}
    }
}

/// Detects direct and mutual recursion (Tarjan SCC).
fn detect_recursion(cg: &mut CallGraph) {
    let mut sccs = tarjan_scc(cg);
    sccs.retain(|s| s.len() > 1);
    // All functions in mutual recursion SCCs are also recursive functions; add them to the recursive set uniformly.
    // This makes cg.recursive the authoritative source for "all recursive functions"; consumers like inline_pass
    // only need to check recursive without separately checking mutually_recursive.
    for scc in &sccs {
        for &func in scc {
            cg.recursive.insert(func);
        }
    }
    cg.mutually_recursive = sccs;
}

/// Tarjan's strongly connected components algorithm.
fn tarjan_scc(cg: &CallGraph) -> Vec<FxHashSet<FuncId>> {
    let mut index_counter: u32 = 0;
    let mut stack: Vec<FuncId> = Vec::new();
    let mut on_stack: FxHashSet<FuncId> = FxHashSet::default();
    let mut indices: FxHashMap<FuncId, u32> = FxHashMap::default();
    let mut lowlinks: FxHashMap<FuncId, u32> = FxHashMap::default();
    let mut sccs: Vec<FxHashSet<FuncId>> = Vec::new();

    fn strongconnect(
        v: FuncId,
        cg: &CallGraph,
        index_counter: &mut u32,
        stack: &mut Vec<FuncId>,
        on_stack: &mut FxHashSet<FuncId>,
        indices: &mut FxHashMap<FuncId, u32>,
        lowlinks: &mut FxHashMap<FuncId, u32>,
        sccs: &mut Vec<FxHashSet<FuncId>>,
    ) {
        indices.insert(v, *index_counter);
        lowlinks.insert(v, *index_counter);
        *index_counter += 1;
        stack.push(v);
        on_stack.insert(v);

        if let Some(callees) = cg.edges.get(&v) {
            for &w in callees {
                if !indices.contains_key(&w) {
                    strongconnect(w, cg, index_counter, stack, on_stack, indices, lowlinks, sccs);
                    let lw = *lowlinks.get(&w).unwrap();
                    let lv = *lowlinks.get(&v).unwrap();
                    lowlinks.insert(v, lv.min(lw));
                } else if on_stack.contains(&w) {
                    let iw = *indices.get(&w).unwrap();
                    let lv = *lowlinks.get(&v).unwrap();
                    lowlinks.insert(v, lv.min(iw));
                }
            }
        }

        if lowlinks.get(&v) == indices.get(&v) {
            let mut scc = FxHashSet::default();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                scc.insert(w);
                if w == v {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    for &v in &cg.nodes {
        if !indices.contains_key(&v) {
            strongconnect(v, cg, &mut index_counter, &mut stack, &mut on_stack, &mut indices, &mut lowlinks, &mut sccs);
        }
    }
    sccs
}

/// Whether a function name is an impure built-in function.
pub fn is_impure_builtin(name: &str) -> bool {
    ImpureBuiltinFn::from_name(name).is_some()
}

// =========================================================================
// PurityAnalyzer — Layer 2: purity fixpoint propagation
// =========================================================================

/// Purity analysis. Initially assumes all functions are pure, traverses function bodies (unified FunDecl + Method)
/// to find directly impure functions (calls to impure built-ins, method calls, select, async/throwing, etc.),
/// then propagates Impure along the reverse call graph.
pub fn analyze_purity(module: &Module, arena: &AstArena, cg: &CallGraph, sema: &SemaResult) -> PurityTable {
    let mut table = PurityTable::new();
    for &fid in &cg.nodes {
        table.put(fid, Purity::Pure);
    }
    let mut direct_impure: FxHashSet<FuncId> = FxHashSet::default();
    // Module-level var/val names: writing any of them makes a function
    // stateful (impure for inlining/memoization purposes).
    let top_level_vars: FxHashSet<&str> = module
        .declarations
        .iter()
        .filter_map(|d| match &d.node {
            crate::ast::Ast::Decl::ExprDecl { stmt: Some(s), .. } => {
                match &arena.stmt(*s).node {
                    Stmt::VarDecl { name, .. } | Stmt::ValDecl { name, .. } => Some(*name),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    // Unified traversal of FunDecl + Method (via cg.iter_funcs)
    let func_metas: Vec<(FuncId, &str, crate::ast::Ast::ExprId, bool)> = cg.iter_funcs(module)
        .map(|(fid, meta)| (fid, meta.name, meta.body, meta.is_async))
        .collect();
    for (caller, name, body, is_async) in func_metas {
        // sema FuncSigInfo lookup: async/throwing functions are always impure
        // (module-qualified: `name` is this module's own function).
        if is_async {
            direct_impure.insert(caller);
            continue;
        }
        if let Some(sig) = sema.get_func_sig_in(module.name, name) {
            if sig.is_async || sig.is_throwing {
                direct_impure.insert(caller);
                continue;
            }
        }
        if is_direct_impure(body, arena, name, module.name, sema, &top_level_vars) {
            direct_impure.insert(caller);
        }
    }
    let mut worklist: Vec<FuncId> = direct_impure.iter().copied().collect();
    for &f in &worklist {
        table.put(f, Purity::Impure);
    }
    while let Some(impure_fn) = worklist.pop() {
        if let Some(callers) = cg.reverse.get(&impure_fn) {
            for &caller in callers {
                if table.lookup(caller) == Some(Purity::Pure) {
                    table.put(caller, Purity::Impure);
                    worklist.push(caller);
                }
            }
        }
    }
    table
}

/// Determines whether a function body is directly impure (contains impure built-in calls, method calls, select, spawn, etc.).
/// Uses sema FuncSigInfo: async/throwing external functions (e.g., println) are also classified as impure.
fn is_direct_impure(body: ExprId, arena: &AstArena, self_name: &str, module_name: &str, sema: &SemaResult, top_level_vars: &FxHashSet<&str>) -> bool {
    // Stateful functions are impure: a body that WRITES a module-level
    // variable (directly or through an element/field of one) must never be
    // inlined or memoized — the expanded body's stores bypass the global
    // slot, freezing the state (rand's inlined next_u64 kept returning the
    // first value forever).
    if writes_top_level_var(body, arena, top_level_vars) {
        return true;
    }
    // Reference writes are equally observable: records and arrays are shared
    // by reference, so `this.field = v` (implicit-this sugar inside methods),
    // `obj.field = v`, `arr[i] = v` and `*r = v` mutate state every alias can
    // see — callers, sibling loop iterations, other methods. A mutating
    // method (`&fn` like an iterator's `&next`) classified pure led to its
    // call being eliminated as a dead `val` declaration (#104) and to unsafe
    // inline/memo plans of the same shape as the rand freeze above.
    if writes_through_reference(body, arena, module_name, sema) {
        return true;
    }
    fn check(expr_id: ExprId, arena: &AstArena, self_name: &str, sema: &SemaResult) -> bool {
        let expr = &arena.expr(expr_id).node;
        match expr {
            Expr::Call { callee, args, .. } => {
                if let Expr::Ident(name) = &arena.expr(*callee).node {
                    if is_impure_builtin(name) {
                        return true;
                    }
                    // sema FuncSigInfo lookup: async/throwing functions (including stdlib external functions) are always impure.
                    // All-owners check: same-named functions across modules each count.
                    if sema.func_sigs_named(name).iter().any(|sig| sig.is_async || sig.is_throwing) {
                        return true;
                    }
                } else {
                    return true;
                }
                if check(*callee, arena, self_name, sema) {
                    return true;
                }
                for a in args {
                    if check(*a, arena, self_name, sema) {
                        return true;
                    }
                }
                false
            }
            Expr::MethodCall { .. } | Expr::SafeMethodCall { .. } | Expr::Select(_) | Expr::InlineTrait(_) => true,
            Expr::Block { stmts, trailing } => {
                for s in stmts {
                    if check_stmt(*s, arena, self_name, sema) {
                        return true;
                    }
                }
                if let Some(t) = trailing {
                    return check(*t, arena, self_name, sema);
                }
                false
            }
            Expr::If { cond, then_branch, else_branch } => {
                check(*cond, arena, self_name, sema)
                    || check(*then_branch, arena, self_name, sema)
                    || else_branch.map_or(false, |e| check(e, arena, self_name, sema))
            }
            Expr::Binary { lhs, rhs, .. } => check(*lhs, arena, self_name, sema) || check(*rhs, arena, self_name, sema),
            Expr::Unary { operand, .. }
            | Expr::RefOf(operand)
            | Expr::Deref(operand)
            | Expr::NonNullAssert(operand)
            | Expr::Propagate(operand)
            | Expr::Atomic(operand)
            | Expr::Lazy(operand) => check(*operand, arena, self_name, sema),
            Expr::Assign { target, value } | Expr::CompoundAssign { target, value, .. } => {
                check(*target, arena, self_name, sema) || check(*value, arena, self_name, sema)
            }
            Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => check(*recv, arena, self_name, sema),
            Expr::Index { recv, index } => check(*recv, arena, self_name, sema) || check(*index, arena, self_name, sema),
            Expr::Slice { recv, start, end, .. } => {
                check(*recv, arena, self_name, sema) || check(*start, arena, self_name, sema) || check(*end, arena, self_name, sema)
            }
            Expr::Elvis { lhs, rhs } => check(*lhs, arena, self_name, sema) || check(*rhs, arena, self_name, sema),
            Expr::ArrayLit { elements, fill } => {
                elements.iter().any(|e| check(*e, arena, self_name, sema))
                    || fill.map_or(false, |(v, c)| check(v, arena, self_name, sema) || check(c, arena, self_name, sema))
            }
            Expr::RecordLit(fields) => fields.iter().any(|f| check(f.value, arena, self_name, sema)),
            Expr::RecordExtend { base, updates } => {
                check(*base, arena, self_name, sema) || updates.iter().any(|u| check(u.value, arena, self_name, sema))
            }
            Expr::Match { scrutinee, arms } => {
                check(*scrutinee, arena, self_name, sema)
                    || arms.iter().any(|a| {
                        a.guard.map_or(false, |g| check(g, arena, self_name, sema)) || check(a.body, arena, self_name, sema)
                    })
            }
            Expr::StrInterp(parts) => parts.iter().any(|p| {
                if let InterpolationPart::Expression(e) = p {
                    check(*e, arena, self_name, sema)
                } else {
                    false
                }
            }),
            Expr::Lambda { .. } => false,
            _ => false,
        }
    }
    fn check_stmt(stmt_id: StmtId, arena: &AstArena, self_name: &str, sema: &SemaResult) -> bool {
        let stmt = &arena.stmt(stmt_id).node;
        match stmt {
            Stmt::ValDecl { value, .. } | Stmt::VarDecl { value, .. } => check(*value, arena, self_name, sema),
            Stmt::Assignment { target, value } | Stmt::CompoundAssignment { target, value, .. } => {
                check(*target, arena, self_name, sema) || check(*value, arena, self_name, sema)
            }
            Stmt::FieldAssignment { object, value, .. } => {
                check(*object, arena, self_name, sema) || check(*value, arena, self_name, sema)
            }
            Stmt::Expression { expr } => check(*expr, arena, self_name, sema),
            Stmt::Return { value } => value.map_or(false, |v| check(v, arena, self_name, sema)),
            Stmt::Defer { expr } | Stmt::Throw { expr } => check(*expr, arena, self_name, sema),
            Stmt::For { iterable, body, .. } => check(*iterable, arena, self_name, sema) || check(*body, arena, self_name, sema),
            Stmt::While { condition, body } => check(*condition, arena, self_name, sema) || check(*body, arena, self_name, sema),
            Stmt::Loop { body } => check(*body, arena, self_name, sema),
            _ => false,
        }
    }
    check(body, arena, self_name, sema)
}

/// Root identifier of an assignment target: `x` → `x`; `arr[i].f` → `arr`.
fn assign_target_root_ident<'a>(target: crate::ast::Ast::ExprId, arena: &AstArena<'a>) -> Option<&'a str> {
    let mut root = target;
    loop {
        match &arena.expr(root).node {
            Expr::Ident(n) => return Some(n),
            Expr::Index { recv, .. }
            | Expr::FieldAccess { recv, .. }
            | Expr::SafeAccess { recv, .. } => root = *recv,
            _ => return None,
        }
    }
}

/// Whether any assignment inside the function body targets a module-level
/// variable (or an element/field of one). Shadowing (a parameter/local named
/// like a global) may over-approximate — that only forgoes an optimization.
/// Whether an assignment target writes THROUGH a reference (observable
/// mutation): field writes (`obj.f = v`, including implicit-this `f = v`
/// inside methods), array element writes (`arr[i] = v`) and deref writes
/// (`*r = v`). Plain local-variable assignments are NOT reference writes —
/// they are function-internal and invisible to callers.
fn writes_through_reference(expr_id: ExprId, arena: &AstArena, module_name: &str, sema: &SemaResult) -> bool {
    match &arena.expr(expr_id).node {
        Expr::Assign { target, .. } | Expr::CompoundAssign { target, .. } => {
            if assign_target_writes_reference(*target, arena, module_name, sema) {
                return true;
            }
        }
        _ => {}
    }
    let mut found = false;
    walk_children_expr(expr_id, arena, |c| {
        if !found {
            found = writes_through_reference(c, arena, module_name, sema);
        }
    });
    if !found {
        walk_children_stmts_of_expr(expr_id, arena, |s| {
            if !found {
                found = writes_through_reference_stmt(s, arena, module_name, sema);
            }
        });
    }
    found
}

fn writes_through_reference_stmt(stmt_id: StmtId, arena: &AstArena, module_name: &str, sema: &SemaResult) -> bool {
    match &arena.stmt(stmt_id).node {
        Stmt::Assignment { target, .. } | Stmt::CompoundAssignment { target, .. } => {
            assign_target_writes_reference(*target, arena, module_name, sema)
        }
        // Explicit field-assignment statements (`obj.f = v` shape): always a
        // reference write regardless of the receiver's root.
        Stmt::FieldAssignment { .. } => true,
        _ => {
            let mut found = false;
            walk_children_stmt(stmt_id, arena, |e| {
                if !found {
                    found = writes_through_reference(e, arena, module_name, sema);
                }
            });
            found
        }
    }
}

/// One assignment target: is it a write through a reference?
fn assign_target_writes_reference(target: ExprId, arena: &AstArena, module_name: &str, sema: &SemaResult) -> bool {
    match &arena.expr(target).node {
        // obj.f = v / arr[i] = v / *r = v
        Expr::FieldAccess { .. } | Expr::SafeAccess { .. } | Expr::Index { .. } | Expr::Deref(_) => true,
        // `f = v` inside a method body resolving to `this.f = v`
        Expr::Ident(_) => {
            let key = module_expr_key(module_name, target.0 as u64);
            sema.expr_types.get(&key).and_then(|info| info.implicit_this.as_ref()).is_some()
        }
        _ => false,
    }
}

fn writes_top_level_var(expr_id: ExprId, arena: &AstArena, top: &FxHashSet<&str>) -> bool {
    match &arena.expr(expr_id).node {
        Expr::Assign { target, .. } | Expr::CompoundAssign { target, .. } => {
            if assign_target_root_ident(*target, arena)
                .map_or(false, |n| top.contains(n))
            {
                return true;
            }
        }
        _ => {}
    }
    let mut found = false;
    walk_children_expr(expr_id, arena, |c| {
        if !found {
            found = writes_top_level_var(c, arena, top);
        }
    });
    if !found {
        walk_children_stmts_of_expr(expr_id, arena, |s| {
            if !found {
                found = writes_top_level_var_stmt(s, arena, top);
            }
        });
    }
    found
}

fn writes_top_level_var_stmt(stmt_id: StmtId, arena: &AstArena, top: &FxHashSet<&str>) -> bool {
    match &arena.stmt(stmt_id).node {
        Stmt::Assignment { target, value, .. } | Stmt::CompoundAssignment { target, value, .. } => {
            assign_target_root_ident(*target, arena).map_or(false, |n| top.contains(n))
                || writes_top_level_var(*value, arena, top)
        }
        Stmt::FieldAssignment { object, value, .. } => {
            assign_target_root_ident(*object, arena).map_or(false, |n| top.contains(n))
                || writes_top_level_var(*value, arena, top)
        }
        _ => {
            let mut found = false;
            walk_children_stmt(stmt_id, arena, |e| {
                if !found {
                    found = writes_top_level_var(e, arena, top);
                }
            });
            found
        }
    }
}

// =========================================================================
// EscapeAnalyzer — Layer 2: escape analysis
// =========================================================================

/// Escape analysis. Traverses each function body and determines whether
/// ArrayLit/RecordLit/RecordExtend allocation sites escape.
pub fn analyze_escape(
    module: &Module,
    arena: &AstArena,
    cg: &CallGraph,
    purity: &PurityTable,
) -> EscapeTable {
    let mut table = EscapeTable::new();
    // Unified traversal of FunDecl + Method
    let func_metas: Vec<(FuncId, &str, crate::ast::Ast::ExprId)> = cg.iter_funcs(module)
        .map(|(fid, meta)| (fid, meta.name, meta.body))
        .collect();
    for (func, name, body) in func_metas {
        mark_allocations(body, arena, &mut table);
        let mut escaping: FxHashSet<ExprId> = FxHashSet::default();
        scan_escapes(body, arena, func, name, cg, purity, &mut escaping);
        for e in escaping {
            if table.lookup(e).is_some() {
                table.put(e, EscapeInfo::Escapes(EscapeKind::Alloc));
            }
        }
    }
    // Lambda escape analysis (Bug #41 tail position escape + Bug #40 loop body capture)
    analyze_lambda_escape(module, arena, &mut table);
    table
}

/// Marks all allocation sites as NoEscape (initial value).
fn mark_allocations(expr_id: ExprId, arena: &AstArena, table: &mut EscapeTable) {
    let expr = &arena.expr(expr_id).node;
    if matches!(expr, Expr::ArrayLit { .. } | Expr::RecordLit(_) | Expr::RecordExtend { .. }) {
        table.put(expr_id, EscapeInfo::NoEscape);
    }
    walk_children_expr(expr_id, arena, |c| mark_allocations(c, arena, table));
    walk_children_stmts_of_expr(expr_id, arena, |s| mark_allocations_stmt(s, arena, table));
}

/// Traverses allocation sites in a statement.
fn mark_allocations_stmt(stmt_id: StmtId, arena: &AstArena, table: &mut EscapeTable) {
    walk_children_stmt(stmt_id, arena, |e| mark_allocations(e, arena, table));
}

/// Scans for escape sites.
fn scan_escapes(
    expr_id: ExprId,
    arena: &AstArena,
    func: FuncId,
    func_name: &str,
    cg: &CallGraph,
    purity: &PurityTable,
    escaping: &mut FxHashSet<ExprId>,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Call { callee, args, .. } => {
            let callee_impure = if let Expr::Ident(name) = &arena.expr(*callee).node {
                cg.name_to_func.get(*name).and_then(|fid| purity.lookup(*fid)) == Some(Purity::Impure)
            } else {
                true
            };
            if callee_impure {
                for a in args {
                    collect_all_allocs(*a, arena, escaping);
                }
            }
        }
        Expr::RecordExtend { base, .. } => {
            collect_all_allocs(*base, arena, escaping);
        }
        _ => {}
    }
    walk_children_expr(expr_id, arena, |c| scan_escapes(c, arena, func, func_name, cg, purity, escaping));
    walk_children_stmts_of_expr(expr_id, arena, |s| scan_escapes_stmt(s, arena, func, func_name, cg, purity, escaping));
}

/// Scans for escape sites in a statement.
fn scan_escapes_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    func: FuncId,
    func_name: &str,
    cg: &CallGraph,
    purity: &PurityTable,
    escaping: &mut FxHashSet<ExprId>,
) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::Return { value } => {
            if let Some(v) = value {
                collect_all_allocs(*v, arena, escaping);
            }
        }
        Stmt::FieldAssignment { value, .. } => {
            collect_all_allocs(*value, arena, escaping);
        }
        Stmt::Throw { expr } => {
            collect_all_allocs(*expr, arena, escaping);
        }
        _ => {}
    }
    walk_children_stmt(stmt_id, arena, |e| scan_escapes(e, arena, func, func_name, cg, purity, escaping));
}

/// Collects all allocation sites in an expression and its sub-expressions.
fn collect_all_allocs(expr_id: ExprId, arena: &AstArena, escaping: &mut FxHashSet<ExprId>) {
    let expr = &arena.expr(expr_id).node;
    if matches!(expr, Expr::ArrayLit { .. } | Expr::RecordLit(_) | Expr::RecordExtend { .. }) {
        escaping.insert(expr_id);
    }
    walk_children_expr(expr_id, arena, |c| collect_all_allocs(c, arena, escaping));
    walk_children_stmts_of_expr(expr_id, arena, |s| collect_all_allocs_stmt(s, arena, escaping));
}

/// Traverses allocation sites in a statement.
fn collect_all_allocs_stmt(stmt_id: StmtId, arena: &AstArena, escaping: &mut FxHashSet<ExprId>) {
    walk_children_stmt(stmt_id, arena, |e| collect_all_allocs(e, arena, escaping));
}

/// Walks the direct child expressions of an expression.
fn walk_children_expr<F: FnMut(ExprId)>(expr_id: ExprId, arena: &AstArena, mut f: F) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit(_)
        | Expr::CharLit(_) | Expr::StrLit(_) | Expr::NullLit | Expr::VoidLit
        | Expr::Ident(_) => {}
        Expr::Unary { operand, .. } | Expr::As { expr: operand, .. } | Expr::RefOf(operand) | Expr::Deref(operand)
        | Expr::NonNullAssert(operand) | Expr::Propagate(operand)
        | Expr::Atomic(operand) | Expr::Lazy(operand) => f(*operand),
        Expr::Binary { lhs, rhs, .. } => { f(*lhs); f(*rhs); }
        Expr::Assign { target, value } | Expr::CompoundAssign { target, value, .. } => { f(*target); f(*value); }
        Expr::Call { callee, args, .. } => { f(*callee); for a in args { f(*a); } }
        Expr::MethodCall { recv, args, .. } | Expr::SafeMethodCall { recv, args, .. } => { f(*recv); for a in args { f(*a); } }
        Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => f(*recv),
        Expr::Index { recv, index } => { f(*recv); f(*index); }
        Expr::Slice { recv, start, end, .. } => { f(*recv); f(*start); f(*end); }
        Expr::Elvis { lhs, rhs } => { f(*lhs); f(*rhs); }
        Expr::ArrayLit { elements, fill } => {
            for e in elements { f(*e); }
            if let Some((v, c)) = fill { f(*v); f(*c); }
        }
        Expr::RecordLit(fields) => { for fd in fields { f(fd.value); } }
        Expr::RecordExtend { base, updates } => { f(*base); for u in updates { f(u.value); } }
        Expr::If { cond, then_branch, else_branch } => {
            f(*cond); f(*then_branch); if let Some(e) = else_branch { f(*e); }
        }
        Expr::Block { trailing, .. } => {
            if let Some(t) = trailing { f(*t); }
        }
        Expr::Match { scrutinee, arms } => {
            f(*scrutinee);
            for arm in arms {
                if let Some(g) = arm.guard { f(g); }
                f(arm.body);
            }
        }
        Expr::StrInterp(parts) => {
            for p in parts {
                if let InterpolationPart::Expression(e) = p { f(*e); }
            }
        }
        Expr::Select(arms) => {
            for arm in arms {
                match arm {
                    SelectArm::Receive { channel_expr, body, .. } => { f(*channel_expr); f(*body); }
                    SelectArm::Timeout { duration, body } => { f(*duration); f(*body); }
                }
            }
        }
        Expr::Lambda { .. } | Expr::InlineTrait(_) => {}
    }
}

/// Walks statements embedded in an expression (only Block).
fn walk_children_stmts_of_expr<F: FnMut(StmtId)>(expr_id: ExprId, arena: &AstArena, mut f: F) {
    if let Expr::Block { stmts, .. } = &arena.expr(expr_id).node {
        for s in stmts { f(*s); }
    }
}

/// Walks the child expressions of a statement (recursing into expressions).
fn walk_children_stmt<F: FnMut(ExprId)>(stmt_id: StmtId, arena: &AstArena, mut f: F) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::ValDecl { value, .. } | Stmt::VarDecl { value, .. } => f(*value),
        Stmt::Assignment { target, value } | Stmt::CompoundAssignment { target, value, .. } => { f(*target); f(*value); }
        Stmt::FieldAssignment { object, value, .. } => { f(*object); f(*value); }
        Stmt::Expression { expr } => f(*expr),
        Stmt::Return { value } => { if let Some(v) = value { f(*v); } }
        Stmt::Defer { expr } | Stmt::Throw { expr } => f(*expr),
        Stmt::For { iterable, body, .. } => { f(*iterable); f(*body); }
        Stmt::While { condition, body } => { f(*condition); f(*body); }
        Stmt::Loop { body } => f(*body),
        Stmt::LocalDecl { .. } | Stmt::Break | Stmt::Continue => {}
    }
}

// =========================================================================
// LambdaEscape — Lambda escape analysis (Bug #41 tail position escape + Bug #40 loop body capture)
// =========================================================================

/// Unified entry point for lambda escape analysis.
///
/// Performs two passes on each FunDecl's body:
/// 1. Tail position escape: calls find_escaping_lambdas, marks as
///    `EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: false })`
/// 2. Loop body capture escape: scans for Lambdas in the body, checks whether they capture
///    loop body local variables, marks as `EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: true })`
fn analyze_lambda_escape(
    module: &Module,
    arena: &AstArena,
    table: &mut EscapeTable,
) {
    for decl in &module.declarations {
        if let Decl::FunDecl { body, .. } = &decl.node {
            // Recursively analyze function body and all nested lambda bodies for escape
            analyze_lambda_escape_recursive(*body, arena, table);
        }
    }
}

/// Performs tail position escape analysis on the current body, then recurses into all nested Lambda bodies.
///
/// The IR's escape_context_stack is stack-based: when compiling each lambda, it scans the body
/// to find escaping nested lambdas. The analyzer must recursively perform the same analysis for each lambda body.
fn analyze_lambda_escape_recursive(
    expr_id: ExprId,
    arena: &AstArena,
    table: &mut EscapeTable,
) {
    // Pass 1: tail position escape (tail-position lambdas in the current body)
    let tail_escaping = find_escaping_lambdas(expr_id, arena);
    for lambda_id in tail_escaping {
        table.put(
            lambda_id,
            EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: false }),
        );
    }
    // Recurse into all nested Lambda bodies to perform the same tail position escape analysis
    walk_lambdas_in_expr(expr_id, arena, &mut |lambda_body| {
        analyze_lambda_escape_recursive(lambda_body, arena, table);
    });
    // Pass 2: loop body capture escape
    let mut loop_body_vars_stack: Vec<FxHashSet<String>> = Vec::new();
    scan_lambda_escapes_in_expr(expr_id, arena, &mut loop_body_vars_stack, table);
}

/// Walks all Lambdas in an expression, calling the callback on each Lambda's body.
fn walk_lambdas_in_expr(
    expr_id: ExprId,
    arena: &AstArena,
    f: &mut impl FnMut(ExprId),
) {
    use crate::ast::Ast::LambdaBody;
    let node = &arena.expr(expr_id).node;
    match node {
        Expr::Lambda { body, .. } => {
            let body_expr = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            f(body_expr);
            // Continue recursing into the lambda body (may have deeper nesting)
            walk_lambdas_in_expr(body_expr, arena, f);
        }
        Expr::Block { stmts, trailing } => {
            for &s in stmts {
                walk_lambdas_in_stmt(s, arena, f);
            }
            if let Some(t) = trailing {
                walk_lambdas_in_expr(*t, arena, f);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            walk_lambdas_in_expr(*cond, arena, f);
            walk_lambdas_in_expr(*then_branch, arena, f);
            if let Some(e) = else_branch {
                walk_lambdas_in_expr(*e, arena, f);
            }
        }
        Expr::Match { scrutinee, arms } => {
            walk_lambdas_in_expr(*scrutinee, arena, f);
            for arm in arms {
                if let Some(g) = arm.guard {
                    walk_lambdas_in_expr(g, arena, f);
                }
                walk_lambdas_in_expr(arm.body, arena, f);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            walk_lambdas_in_expr(*lhs, arena, f);
            walk_lambdas_in_expr(*rhs, arena, f);
        }
        Expr::Assign { target, value } => {
            walk_lambdas_in_expr(*target, arena, f);
            walk_lambdas_in_expr(*value, arena, f);
        }
        Expr::Call { callee, args, .. } => {
            walk_lambdas_in_expr(*callee, arena, f);
            for &a in args {
                walk_lambdas_in_expr(a, arena, f);
            }
        }
        Expr::MethodCall { recv, args, .. } => {
            walk_lambdas_in_expr(*recv, arena, f);
            for &a in args {
                walk_lambdas_in_expr(a, arena, f);
            }
        }
        Expr::ArrayLit { elements, fill } => {
            for &e in elements {
                walk_lambdas_in_expr(e, arena, f);
            }
            if let Some((v, c)) = fill {
                walk_lambdas_in_expr(*v, arena, f);
                walk_lambdas_in_expr(*c, arena, f);
            }
        }
        Expr::RecordLit(fields) => {
            for field in fields {
                walk_lambdas_in_expr(field.value, arena, f);
            }
        }
        Expr::Elvis { lhs, rhs } => {
            walk_lambdas_in_expr(*lhs, arena, f);
            walk_lambdas_in_expr(*rhs, arena, f);
        }
        _ => {}
    }
}

/// Statement version of walk_lambdas_in_expr.
fn walk_lambdas_in_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    f: &mut impl FnMut(ExprId),
) {
    match &arena.stmt(stmt_id).node {
        Stmt::ValDecl { value, .. } | Stmt::VarDecl { value, .. } => {
            walk_lambdas_in_expr(*value, arena, f);
        }
        Stmt::Assignment { value, .. } => {
            walk_lambdas_in_expr(*value, arena, f);
        }
        Stmt::Expression { expr } => {
            walk_lambdas_in_expr(*expr, arena, f);
        }
        Stmt::Return { value } => {
            if let Some(v) = value {
                walk_lambdas_in_expr(*v, arena, f);
            }
        }
        Stmt::For { iterable, body, .. } => {
            walk_lambdas_in_expr(*iterable, arena, f);
            walk_lambdas_in_expr(*body, arena, f);
        }
        Stmt::Defer { expr } => {
            walk_lambdas_in_expr(*expr, arena, f);
        }
        Stmt::Throw { expr } => {
            walk_lambdas_in_expr(*expr, arena, f);
        }
        Stmt::CompoundAssignment { value, .. } => {
            walk_lambdas_in_expr(*value, arena, f);
        }
        Stmt::FieldAssignment { value, .. } => {
            walk_lambdas_in_expr(*value, arena, f);
        }
        Stmt::While { condition, body } => {
            walk_lambdas_in_expr(*condition, arena, f);
            walk_lambdas_in_expr(*body, arena, f);
        }
        Stmt::Loop { body } => {
            walk_lambdas_in_expr(*body, arena, f);
        }
        Stmt::LocalDecl { decl } => {
            // Local function declaration: recurse into function body for escape analysis
            if let Decl::FunDecl { body, .. } = &**decl {
                f(*body);
                walk_lambdas_in_expr(*body, arena, f);
            }
        }
        _ => {}
    }
}

/// Collects variable names holding Lambdas in ValDecl/VarDecl -> ExprId.
fn collect_lambda_vars(
    expr_id: ExprId,
    arena: &AstArena,
    out: &mut FxHashMap<String, ExprId>,
) {
    let node = &arena.expr(expr_id).node;
    match node {
        Expr::Lambda { body, .. } => {
            let body_expr = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            collect_lambda_vars(body_expr, arena, out);
        }
        Expr::Block { stmts, trailing } => {
            for &stmt_id in stmts {
                collect_lambda_vars_stmt(stmt_id, arena, out);
            }
            if let Some(t) = trailing {
                collect_lambda_vars(*t, arena, out);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            collect_lambda_vars(*cond, arena, out);
            collect_lambda_vars(*then_branch, arena, out);
            if let Some(e) = else_branch {
                collect_lambda_vars(*e, arena, out);
            }
        }
        Expr::Match { scrutinee, arms } => {
            collect_lambda_vars(*scrutinee, arena, out);
            for arm in arms {
                if let Some(g) = arm.guard {
                    collect_lambda_vars(g, arena, out);
                }
                collect_lambda_vars(arm.body, arena, out);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_lambda_vars(*lhs, arena, out);
            collect_lambda_vars(*rhs, arena, out);
        }
        Expr::Assign { target, value } => {
            collect_lambda_vars(*target, arena, out);
            collect_lambda_vars(*value, arena, out);
        }
        Expr::Call { callee, args, .. } => {
            collect_lambda_vars(*callee, arena, out);
            for a in args {
                collect_lambda_vars(*a, arena, out);
            }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_lambda_vars(*recv, arena, out);
            for a in args {
                collect_lambda_vars(*a, arena, out);
            }
        }
        Expr::ArrayLit { elements, fill } => {
            for e in elements {
                collect_lambda_vars(*e, arena, out);
            }
            if let Some((v, c)) = fill {
                collect_lambda_vars(*v, arena, out);
                collect_lambda_vars(*c, arena, out);
            }
        }
        Expr::RecordLit(fields) => {
            for f in fields {
                collect_lambda_vars(f.value, arena, out);
            }
        }
        Expr::Elvis { lhs, rhs } => {
            collect_lambda_vars(*lhs, arena, out);
            collect_lambda_vars(*rhs, arena, out);
        }
        _ => {}
    }
}

/// Helper: collects lambda variables from a Stmt.
fn collect_lambda_vars_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    out: &mut FxHashMap<String, ExprId>,
) {
    match &arena.stmt(stmt_id).node {
        Stmt::ValDecl { name, value, .. } | Stmt::VarDecl { name, value, .. } => {
            if let Expr::Lambda { .. } = &arena.expr(*value).node {
                out.insert(name.to_string(), *value);
            }
            // Recursively scan value (lambda body may also contain lambda variables)
            collect_lambda_vars(*value, arena, out);
        }
        Stmt::Assignment { value, .. } => {
            collect_lambda_vars(*value, arena, out);
        }
        Stmt::Expression { expr } => {
            collect_lambda_vars(*expr, arena, out);
        }
        Stmt::Return { value } => {
            if let Some(v) = value {
                collect_lambda_vars(*v, arena, out);
            }
        }
        Stmt::For { iterable, body, .. } => {
            collect_lambda_vars(*iterable, arena, out);
            collect_lambda_vars(*body, arena, out);
        }
        Stmt::Defer { expr } => {
            collect_lambda_vars(*expr, arena, out);
        }
        Stmt::Throw { expr } => {
            collect_lambda_vars(*expr, arena, out);
        }
        Stmt::CompoundAssignment { value, .. } => {
            collect_lambda_vars(*value, arena, out);
        }
        Stmt::FieldAssignment { value, .. } => {
            collect_lambda_vars(*value, arena, out);
        }
        Stmt::While { condition, body } => {
            collect_lambda_vars(*condition, arena, out);
            collect_lambda_vars(*body, arena, out);
        }
        Stmt::Loop { body } => {
            collect_lambda_vars(*body, arena, out);
        }
        // Nested function body may bind lambda variables.
        Stmt::LocalDecl { decl } => match decl.as_ref() {
            crate::ast::Ast::Decl::FunDecl { body, .. } => {
                collect_lambda_vars(*body, arena, out);
            }
            _ => {}
        },
        // Break/Continue do not contain lambda variables
        _ => {}
    }
}

/// Recursively collects tail-position Lambda ExprIds (including Idents holding Lambdas).
///
/// Tail position = the expression's value will be used as the return value of the enclosing lambda.
/// - The body itself is in tail position
/// - Block trailing is in tail position
/// - Return statement value is in tail position
/// - If branches are in tail position (when the If itself is in tail position)
/// - Match arm body is in tail position (when the Match itself is in tail position)
/// - Elvis rhs is in tail position (when the Elvis itself is in tail position)
fn collect_tail_lambdas(
    expr_id: ExprId,
    arena: &AstArena,
    lambda_vars: &FxHashMap<String, ExprId>,
    out: &mut FxHashSet<ExprId>,
) {
    let node = &arena.expr(expr_id).node;
    match node {
        Expr::Lambda { .. } => {
            // Lambda in tail position -> escapes
            out.insert(expr_id);
        }
        Expr::Ident(name) => {
            // Ident in tail position, if it holds a Lambda -> that Lambda escapes
            if let Some(&lambda_id) = lambda_vars.get(*name) {
                out.insert(lambda_id);
            }
        }
        Expr::Block { stmts, trailing } => {
            // Return statement value is in tail position
            for &stmt_id in stmts {
                if let Stmt::Return { value: Some(ret_expr) } = &arena.stmt(stmt_id).node {
                    collect_tail_lambdas(*ret_expr, arena, lambda_vars, out);
                }
            }
            // trailing is in tail position
            if let Some(t) = trailing {
                collect_tail_lambdas(*t, arena, lambda_vars, out);
            }
        }
        Expr::If { then_branch, else_branch, .. } => {
            collect_tail_lambdas(*then_branch, arena, lambda_vars, out);
            if let Some(e) = else_branch {
                collect_tail_lambdas(*e, arena, lambda_vars, out);
            }
        }
        Expr::Match { arms, .. } => {
            for arm in arms {
                collect_tail_lambdas(arm.body, arena, lambda_vars, out);
            }
        }
        Expr::Elvis { rhs, .. } => {
            // Elvis rhs is in tail position (when lhs is null, rhs is the return value)
            collect_tail_lambdas(*rhs, arena, lambda_vars, out);
        }
        _ => {
            // Other expressions are not in tail position; their sub-expressions are not either
        }
    }
}

/// Two-pass scan entry point: collects tail-position escaping Lambdas.
///
/// Pass 1: Collects all ValDecl/VarDecl variables holding Lambdas (name -> lambda ExprId)
/// Pass 2: Recursively collects tail-position Lambdas (including Idents holding Lambdas)
fn find_escaping_lambdas(body: ExprId, arena: &AstArena) -> FxHashSet<ExprId> {
    let mut escaping: FxHashSet<ExprId> = FxHashSet::default();
    let mut lambda_vars: FxHashMap<String, ExprId> = FxHashMap::default();
    collect_lambda_vars(body, arena, &mut lambda_vars);
    collect_tail_lambdas(body, arena, &lambda_vars, &mut escaping);
    escaping
}

/// Recursively collects all Ident names in an expression (deduplicated, preserving first-occurrence order).
///
/// Simplified free-variable analysis: traverses common Expr variants to collect identifier references;
/// the caller excludes lambda parameters and checks outer-scope bindings.
fn collect_free_idents_expr(expr_id: ExprId, arena: &AstArena, names: &mut Vec<String>) {
    let spanned = arena.expr(expr_id);
    match &spanned.node {
        Expr::Ident(name) => {
            if !names.iter().any(|n| n == name) {
                names.push((*name).to_string());
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_free_idents_expr(*lhs, arena, names);
            collect_free_idents_expr(*rhs, arena, names);
        }
        Expr::Unary { operand, .. } => {
            collect_free_idents_expr(*operand, arena, names);
        }
        Expr::Call { callee, args, .. } => {
            collect_free_idents_expr(*callee, arena, names);
            for &a in args {
                collect_free_idents_expr(a, arena, names);
            }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_free_idents_expr(*recv, arena, names);
            for &a in args {
                collect_free_idents_expr(a, arena, names);
            }
        }
        Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => {
            collect_free_idents_expr(*recv, arena, names);
        }
        Expr::Index { recv, index } => {
            collect_free_idents_expr(*recv, arena, names);
            collect_free_idents_expr(*index, arena, names);
        }
        Expr::Assign { target, value } => {
            collect_free_idents_expr(*target, arena, names);
            collect_free_idents_expr(*value, arena, names);
        }
        Expr::CompoundAssign { target, value, .. } => {
            collect_free_idents_expr(*target, arena, names);
            collect_free_idents_expr(*value, arena, names);
        }
        Expr::RecordLit(fields) => {
            for f in fields {
                collect_free_idents_expr(f.value, arena, names);
            }
        }
        Expr::If { cond, then_branch, else_branch } => {
            collect_free_idents_expr(*cond, arena, names);
            collect_free_idents_expr(*then_branch, arena, names);
            if let Some(e) = else_branch {
                collect_free_idents_expr(*e, arena, names);
            }
        }
        Expr::Block { stmts, trailing } => {
            for &s in stmts {
                collect_free_idents_stmt(s, arena, names);
            }
            if let Some(t) = trailing {
                collect_free_idents_expr(*t, arena, names);
            }
        }
        Expr::Lambda { body, .. } => {
            let inner = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            collect_free_idents_expr(inner, arena, names);
        }
        Expr::Match { scrutinee, arms } => {
            collect_free_idents_expr(*scrutinee, arena, names);
            for arm in arms {
                if let Some(g) = arm.guard {
                    collect_free_idents_expr(g, arena, names);
                }
                collect_free_idents_expr(arm.body, arena, names);
            }
        }
        // Single-operand expressions: RefOf/Deref/Propagate/NonNullAssert/Atomic/Lazy
        Expr::RefOf(inner)
        | Expr::Deref(inner)
        | Expr::Propagate(inner)
        | Expr::NonNullAssert(inner)
        | Expr::Atomic(inner)
        | Expr::Lazy(inner) => {
            collect_free_idents_expr(*inner, arena, names);
        }
        Expr::Elvis { lhs, rhs } => {
            collect_free_idents_expr(*lhs, arena, names);
            collect_free_idents_expr(*rhs, arena, names);
        }
        Expr::Slice { recv, start, end, .. } => {
            collect_free_idents_expr(*recv, arena, names);
            collect_free_idents_expr(*start, arena, names);
            collect_free_idents_expr(*end, arena, names);
        }
        Expr::SafeMethodCall { recv, args, .. } => {
            collect_free_idents_expr(*recv, arena, names);
            for &a in args {
                collect_free_idents_expr(a, arena, names);
            }
        }
        Expr::RecordExtend { base, updates } => {
            collect_free_idents_expr(*base, arena, names);
            for f in updates {
                collect_free_idents_expr(f.value, arena, names);
            }
        }
        Expr::ArrayLit { elements, fill } => {
            for &e in elements {
                collect_free_idents_expr(e, arena, names);
            }
            if let Some((v, c)) = fill {
                collect_free_idents_expr(*v, arena, names);
                collect_free_idents_expr(*c, arena, names);
            }
        }
        Expr::StrInterp(parts) => {
            for part in parts {
                if let InterpolationPart::Expression(e) = part {
                    collect_free_idents_expr(*e, arena, names);
                }
            }
        }
        Expr::Select(arms) => {
            for arm in arms {
                match arm {
                    SelectArm::Receive { channel_expr, body, .. } => {
                        collect_free_idents_expr(*channel_expr, arena, names);
                        collect_free_idents_expr(*body, arena, names);
                    }
                    SelectArm::Timeout { duration, body } => {
                        collect_free_idents_expr(*duration, arena, names);
                        collect_free_idents_expr(*body, arena, names);
                    }
                }
            }
        }
        Expr::InlineTrait(methods) => {
            for m in methods {
                if let Some(body_expr) = m.body {
                    collect_free_idents_expr(body_expr, arena, names);
                }
            }
        }
        // Constant/no-subexpression variants: IntLit/FloatLit/BoolLit/CharLit/StrLit/NullLit/VoidLit
        _ => {}
    }
}

/// Recursively collects Ident names in a statement (statement version of collect_free_idents_expr).
fn collect_free_idents_stmt(stmt_id: StmtId, arena: &AstArena, names: &mut Vec<String>) {
    match &arena.stmt(stmt_id).node {
        Stmt::ValDecl { value, .. } | Stmt::VarDecl { value, .. } => {
            collect_free_idents_expr(*value, arena, names);
        }
        Stmt::Expression { expr } => {
            collect_free_idents_expr(*expr, arena, names);
        }
        Stmt::Assignment { target, value } => {
            collect_free_idents_expr(*target, arena, names);
            collect_free_idents_expr(*value, arena, names);
        }
        Stmt::FieldAssignment { object, value, .. } => {
            collect_free_idents_expr(*object, arena, names);
            collect_free_idents_expr(*value, arena, names);
        }
        Stmt::CompoundAssignment { target, value, .. } => {
            collect_free_idents_expr(*target, arena, names);
            collect_free_idents_expr(*value, arena, names);
        }
        Stmt::Return { value } => {
            if let Some(v) = value {
                collect_free_idents_expr(*v, arena, names);
            }
        }
        Stmt::Throw { expr } => {
            collect_free_idents_expr(*expr, arena, names);
        }
        Stmt::For { iterable, body, .. } => {
            collect_free_idents_expr(*iterable, arena, names);
            collect_free_idents_expr(*body, arena, names);
        }
        Stmt::While { condition, body } => {
            collect_free_idents_expr(*condition, arena, names);
            collect_free_idents_expr(*body, arena, names);
        }
        Stmt::Loop { body } => {
            collect_free_idents_expr(*body, arena, names);
        }
        Stmt::Defer { expr } => {
            collect_free_idents_expr(*expr, arena, names);
        }
        Stmt::Break | Stmt::Continue => {}
        Stmt::LocalDecl { decl } => match decl.as_ref() {
            Decl::FunDecl { body, .. } => {
                collect_free_idents_expr(*body, arena, names);
            }
            _ => {}
        },
    }
}

/// Scans for Lambdas in an expression, detecting loop body capture escape.
///
/// Maintains `loop_body_vars_stack` (a stack of loop body local variable name sets). When encountering a Lambda:
/// 1. Collect lambda parameter names (excluding its own parameters)
/// 2. Use collect_free_idents_expr to collect all identifiers in the lambda body
/// 3. Exclude the lambda's own parameter names; the remaining are free variables
/// 4. Check whether any free variable is in any layer of loop_body_vars_stack -> loop body capture escape
///
/// NOTE: This scan is intentionally kept separate from Sema's unified capture
/// table (`SemaResult.captures`). The two serve different purposes:
/// - Sema captures: per-capture mode (Snapshot/Reference) for IR codegen.
/// - Analyzer escape: per-lambda escape classification (loop_body_capture) for
///   function_id allocation.
/// A future cleanup could unify them (the Analyzer could read
/// `SemaResult.captures` instead of re-scanning), but the current duplication
/// is harmless (both produce consistent results) and avoids coupling the
/// Analyzer's AST-only pass to Sema's tables.
fn scan_lambda_escapes_in_expr(
    expr_id: ExprId,
    arena: &AstArena,
    loop_body_vars_stack: &mut Vec<FxHashSet<String>>,
    table: &mut EscapeTable,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Lambda { params, body, .. } => {
            // a. Collect lambda parameter names (excluding its own parameters)
            let param_names: FxHashSet<String> = params.iter().map(|p| p.name.to_string()).collect();
            // b. Collect all identifiers in the lambda body
            let body_expr = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            let mut idents = Vec::new();
            collect_free_idents_expr(body_expr, arena, &mut idents);
            // c. Exclude lambda's own parameter names -> free variables
            // d. Check whether free variables are in the loop body local variable stack
            let captures_loop_var = idents.iter().any(|n| {
                !param_names.contains(n) && loop_body_vars_stack.iter().any(|layer| layer.contains(n))
            });
            if captures_loop_var {
                table.put(
                    expr_id,
                    EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: true }),
                );
            }
            // Continue recursively scanning the lambda body for nested lambdas / loops
            scan_lambda_escapes_in_expr(body_expr, arena, loop_body_vars_stack, table);
        }
        Expr::Block { stmts, trailing } => {
            for &s in stmts {
                scan_lambda_escapes_in_stmt(s, arena, loop_body_vars_stack, table);
            }
            if let Some(t) = trailing {
                scan_lambda_escapes_in_expr(*t, arena, loop_body_vars_stack, table);
            }
        }
        _ => {
            walk_children_expr(expr_id, arena, |c| {
                scan_lambda_escapes_in_expr(c, arena, loop_body_vars_stack, table);
            });
            walk_children_stmts_of_expr(expr_id, arena, |s| {
                scan_lambda_escapes_in_stmt(s, arena, loop_body_vars_stack, table);
            });
        }
    }
}

/// Scans for Lambdas in a statement, detecting loop body capture escape.
///
/// When entering For/While/Loop body, collects all ValDecl/VarDecl variable names defined in the body,
/// pushes them onto the loop body local variable stack; pops on exit.
fn scan_lambda_escapes_in_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    loop_body_vars_stack: &mut Vec<FxHashSet<String>>,
    table: &mut EscapeTable,
) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::For { iterable, body, .. } => {
            // First scan iterable (not inside the loop body)
            scan_lambda_escapes_in_expr(*iterable, arena, loop_body_vars_stack, table);
            // Collect loop body local variables, push onto stack
            let mut body_vars = FxHashSet::default();
            collect_loop_body_vars_expr(*body, arena, &mut body_vars);
            loop_body_vars_stack.push(body_vars);
            scan_lambda_escapes_in_expr(*body, arena, loop_body_vars_stack, table);
            loop_body_vars_stack.pop();
        }
        Stmt::While { condition, body } => {
            scan_lambda_escapes_in_expr(*condition, arena, loop_body_vars_stack, table);
            let mut body_vars = FxHashSet::default();
            collect_loop_body_vars_expr(*body, arena, &mut body_vars);
            loop_body_vars_stack.push(body_vars);
            scan_lambda_escapes_in_expr(*body, arena, loop_body_vars_stack, table);
            loop_body_vars_stack.pop();
        }
        Stmt::Loop { body } => {
            let mut body_vars = FxHashSet::default();
            collect_loop_body_vars_expr(*body, arena, &mut body_vars);
            loop_body_vars_stack.push(body_vars);
            scan_lambda_escapes_in_expr(*body, arena, loop_body_vars_stack, table);
            loop_body_vars_stack.pop();
        }
        Stmt::LocalDecl { decl } => {
            // Local function declaration: separate scope, scan with a fresh loop_body_vars_stack
            if let Decl::FunDecl { body, .. } = &**decl {
                let mut fresh_stack: Vec<FxHashSet<String>> = Vec::new();
                scan_lambda_escapes_in_expr(*body, arena, &mut fresh_stack, table);
            }
        }
        _ => {
            walk_children_stmt(stmt_id, arena, |e| {
                scan_lambda_escapes_in_expr(e, arena, loop_body_vars_stack, table);
            });
        }
    }
}

/// Collects all ValDecl/VarDecl variable names defined in a loop body (does not enter nested lambda/function scopes).
fn collect_loop_body_vars_expr(expr_id: ExprId, arena: &AstArena, vars: &mut FxHashSet<String>) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        // Do not enter nested lambda internal scopes (lambdas have their own parameters and local variables)
        Expr::Lambda { .. } => {}
        _ => {
            walk_children_expr(expr_id, arena, |c| collect_loop_body_vars_expr(c, arena, vars));
            walk_children_stmts_of_expr(expr_id, arena, |s| collect_loop_body_vars_stmt(s, arena, vars));
        }
    }
}

/// Statement version of collect_loop_body_vars_expr.
fn collect_loop_body_vars_stmt(stmt_id: StmtId, arena: &AstArena, vars: &mut FxHashSet<String>) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::ValDecl { name, value, .. } | Stmt::VarDecl { name, value, .. } => {
            vars.insert(name.to_string());
            collect_loop_body_vars_expr(*value, arena, vars);
        }
        // Do not enter nested function internal scopes
        Stmt::LocalDecl { .. } => {}
        _ => {
            walk_children_stmt(stmt_id, arena, |e| collect_loop_body_vars_expr(e, arena, vars));
        }
    }
}

// =========================================================================
// MemoPlan — memoization strategy (Layer 3 shared structure)
// =========================================================================

/// Memoization candidate.
#[derive(Debug, Clone)]
pub struct MemoCandidate {
    pub func: FuncId,
    pub strategy: MemoStrategy,
}

/// Tail recursion parameter transformation info: base case + recursive branch extracted from the function body AST.
/// The Builder layer consumes this info to construct while_sg IR.
#[derive(Debug, Clone, Default)]
pub struct TailRecInfo {
    /// Non-recursive termination branch: (condition expression, return value expression).
    /// Condition is None for the else fallback branch (unconditional termination).
    pub base_cases: Vec<(Option<ExprId>, ExprId)>,
    /// Recursive branch: (condition expression, argument list).
    /// Condition is None for the else fallback branch (unconditional recursion).
    pub rec_branches: Vec<(Option<ExprId>, Vec<ExprId>)>,
}

impl TailRecInfo {
    /// Whether valid: at least one base case and one rec branch.
    pub fn is_valid(&self) -> bool {
        !self.base_cases.is_empty() && !self.rec_branches.is_empty()
    }
}

/// Non-tail recursion to iteration info: transforms non-tail-recursive functions into "work stack + while loop + state machine" IR.
///
/// Core idea: each non-tail self-call in the function body is split into "push continuation + push subtask";
/// after the call returns, dispatch to the corresponding continuation via state number, replacing the call
/// result with a result variable.
///
/// For example, fib(n) = if n < 2 { n } else { fib(n-1) + fib(n-2) } is transformed to:
/// - state 0 (INIT): if n < 2 { result = n } else { push cont(1); push task(n-1); continue }
/// - state 1 (AFTER fib(n-1)): left = result; push cont(2, left); push task(n-2); continue
/// - state 2 (AFTER fib(n-2)): result = saved + result
#[derive(Debug, Clone)]
pub struct NonTailRecInfo {
    /// ExprIds of all non-tail self-calls (in AST traversal order).
    /// The Builder uses this list to assign state numbers: state 0 = INIT, state N = after the Nth call returns.
    pub call_sites: Vec<ExprId>,
    /// The continuation expression ExprId containing all call_sites.
    /// The Builder recompiles this expression for each state, replacing completed calls via call_result_map.
    pub continuation_expr: ExprId,
    /// base case: (condition, return value). Condition is None for else fallback.
    pub base_cases: Vec<(Option<ExprId>, ExprId)>,
    /// Number of function parameters (used to construct stack frames).
    pub param_count: usize,
}

impl NonTailRecInfo {
    pub fn is_valid(&self) -> bool {
        !self.call_sites.is_empty() && !self.base_cases.is_empty()
    }
}

/// Memoization strategy.
#[derive(Debug, Clone)]
pub enum MemoStrategy {
    /// Tail recursion to loop, no caching.
    TailRecToLoop { info: TailRecInfo },
    /// Non-tail recursion to iteration (work stack simulation), no caching.
    NonTailRecToLoop { info: NonTailRecInfo },
    /// Memoization cache.
    Memoize { cache_key: CacheKeySpec, capacity: MemoCapacity },
    /// Loop invariant hoisting.
    LoopInvariantHoist { invariants: Vec<ExprId> },
}

/// Cache key specification: parameter indices participating in the cache key.
#[derive(Debug, Clone)]
pub struct CacheKeySpec {
    pub param_indices: Vec<u32>,
}

/// Cache capacity strategy.
#[derive(Debug, Clone)]
pub enum MemoCapacity {
    Unlimited,
    LRU(usize),
}

/// Memoization plan.
#[derive(Debug, Default)]
pub struct MemoPlan {
    pub candidates: Vec<MemoCandidate>,
}

// =========================================================================
// Unreachable code detection + constant-condition dead branch elimination
// =========================================================================

/// Whether a statement is a control flow terminator (statements after it are unreachable).
fn is_terminator_stmt(stmt_id: StmtId, arena: &AstArena) -> bool {
    matches!(
        &arena.stmt(stmt_id).node,
        Stmt::Return { .. } | Stmt::Break | Stmt::Continue | Stmt::Throw { .. }
    )
}

/// Recursively marks all statements after a terminator in a block as dead (unreachable code).
fn mark_unreachable(expr_id: ExprId, arena: &AstArena, report: &mut DeadCodeReport) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Block { stmts, trailing } => {
            let mut terminated = false;
            for s in stmts {
                if terminated {
                    report.dead_stmts.insert(*s);
                } else if is_terminator_stmt(*s, arena) {
                    terminated = true;
                } else {
                    mark_unreachable_stmt(*s, arena, report);
                }
            }
            if !terminated {
                if let Some(t) = trailing {
                    mark_unreachable(*t, arena, report);
                }
            }
        }
        _ => walk_children_expr(expr_id, arena, |c| mark_unreachable(c, arena, report)),
    }
}

fn mark_unreachable_stmt(stmt_id: StmtId, arena: &AstArena, report: &mut DeadCodeReport) {
    walk_children_stmt(stmt_id, arena, |e| mark_unreachable(e, arena, report));
}

/// Recursively marks all statements in an expression as dead (used for whole dead-branch elimination).
fn mark_all_dead(expr_id: ExprId, arena: &AstArena, report: &mut DeadCodeReport) {
    let expr = &arena.expr(expr_id).node;
    if let Expr::Block { stmts, trailing } = expr {
        for s in stmts {
            report.dead_stmts.insert(*s);
            mark_all_dead_stmt(*s, arena, report);
        }
        if let Some(t) = trailing {
            mark_all_dead(*t, arena, report);
        }
    } else {
        walk_children_expr(expr_id, arena, |c| mark_all_dead(c, arena, report));
    }
}

fn mark_all_dead_stmt(stmt_id: StmtId, arena: &AstArena, report: &mut DeadCodeReport) {
    walk_children_stmt(stmt_id, arena, |e| mark_all_dead(e, arena, report));
}

/// Evaluates a compile-time constant boolean condition. Prefers sema ExprInfo.const_val, falls back to BoolLit.
fn eval_const_bool(expr_id: ExprId, arena: &AstArena, module_name: &str, sema: &SemaResult) -> Option<bool> {
    if let Expr::BoolLit(b) = &arena.expr(expr_id).node {
        return Some(*b);
    }
    let key = module_expr_key(module_name, expr_id.0 as u64);
    let info = sema.expr_types.get(&key)?;
    match &info.const_val {
        Some(ConstVal::Bool(b)) => Some(*b),
        _ => None,
    }
}

// =========================================================================
// DeadCodePass — Layer 3: dead code elimination
// =========================================================================

/// Dead code report.
#[derive(Debug, Default)]
pub struct DeadCodeReport {
    /// Statements that can be safely eliminated.
    pub dead_stmts: FxHashSet<StmtId>,
}

impl DeadCodeReport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_dead(&self, s: StmtId) -> bool {
        self.dead_stmts.contains(&s)
    }
}

/// Dead code analysis pass. Per-function fixpoint iteration: collects valid reads, marks unread and side-effect-free declarations.
/// The preprocessing phase marks unreachable code and constant-condition dead branches.
pub fn dead_code_pass(
    module: &Module,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    cg: &CallGraph,
    def_use: &DefUseGraph,
) -> DeadCodeReport {
    let mut report = DeadCodeReport::new();
    let func_name_to_id = &cg.name_to_func;
    // Unified traversal of FunDecl + Method
    let func_metas: Vec<(FuncId, crate::ast::Ast::ExprId)> = cg.iter_funcs(module)
        .map(|(fid, meta)| (fid, meta.body))
        .collect();
    for (func, body) in func_metas {
        // Preprocessing: unreachable code (after return/break/continue/throw)
        mark_unreachable(body, arena, &mut report);
        // Fixpoint iteration: dead declarations + constant-condition dead branches + dead stores
        analyze_function_dce(body, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, &mut report);
    }
    report
}

/// Performs fixpoint iteration on a function body.
fn analyze_function_dce(
    body: ExprId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    func: FuncId,
    def_use: &DefUseGraph,
    report: &mut DeadCodeReport,
) {
    let mut changed = true;
    while changed {
        changed = false;
        let mut reads: FxHashSet<String> = FxHashSet::default();
        collect_reads_expr(body, arena, report, &mut reads);
        let before = report.dead_stmts.len();
        mark_dead_decls_expr(body, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, &reads, report);
        if report.dead_stmts.len() > before {
            changed = true;
        }
    }
}

/// Recursively collects valid reads in an expression (skipping inits of already-dead declarations).
fn collect_reads_expr(expr_id: ExprId, arena: &AstArena, report: &DeadCodeReport, reads: &mut FxHashSet<String>) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Ident(name) => {
            reads.insert(name.to_string());
        }
        Expr::Block { stmts, trailing } => {
            for s in stmts {
                collect_reads_stmt(*s, arena, report, reads);
            }
            if let Some(t) = trailing {
                collect_reads_expr(*t, arena, report, reads);
            }
        }
        Expr::Assign { target, value } => {
            if !matches!(&arena.expr(*target).node, Expr::Ident(_)) {
                collect_reads_expr(*target, arena, report, reads);
            }
            collect_reads_expr(*value, arena, report, reads);
        }
        // -- Closures: traverse body to collect captured outer variable reads --
        // walk_children_expr does not traverse Lambda; this needs special handling, otherwise
        // closure-captured variables would not be marked as "read", causing their declarations
        // to be falsely classified as dead code.
        Expr::Lambda { body, .. } => {
            let body_expr = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            collect_reads_expr(body_expr, arena, report, reads);
        }
        _ => walk_children_expr(expr_id, arena, |c| collect_reads_expr(c, arena, report, reads)),
    }
}

/// Recursively collects valid reads in a statement.
fn collect_reads_stmt(stmt_id: StmtId, arena: &AstArena, report: &DeadCodeReport, reads: &mut FxHashSet<String>) {
    if report.is_dead(stmt_id) {
        return;
    }
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::ValDecl { value, .. } | Stmt::VarDecl { value, .. } => {
            collect_reads_expr(*value, arena, report, reads);
        }
        Stmt::Assignment { target, value } => {
            if !matches!(&arena.expr(*target).node, Expr::Ident(_)) {
                collect_reads_expr(*target, arena, report, reads);
            }
            collect_reads_expr(*value, arena, report, reads);
        }
        // Nested function declaration: traverse the body to collect outer-variable
        // reads (same as Lambda). Without this, a variable captured only by a
        // nested function would be misidentified as dead.
        Stmt::LocalDecl { decl } => match decl.as_ref() {
            crate::ast::Ast::Decl::FunDecl { body, .. } => {
                collect_reads_expr(*body, arena, report, reads);
            }
            _ => {}
        },
        _ => walk_children_stmt(stmt_id, arena, |e| collect_reads_expr(e, arena, report, reads)),
    }
}

/// Traverses an expression, marking unread and side-effect-free declarations as dead. Also handles constant-condition dead branches.
fn mark_dead_decls_expr(
    expr_id: ExprId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    func: FuncId,
    def_use: &DefUseGraph,
    reads: &FxHashSet<String>,
    report: &mut DeadCodeReport,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Block { stmts, trailing } => {
            for s in stmts {
                mark_dead_decls_stmt(*s, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
            }
            if let Some(t) = trailing {
                mark_dead_decls_expr(*t, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
            }
        }
        // -- Constant-condition dead branches: if true -> else is dead, if false -> then is dead --
        Expr::If { cond, then_branch, else_branch } => {
            if let Some(val) = eval_const_bool(*cond, arena, module_name, sema) {
                if !val {
                    mark_all_dead(*then_branch, arena, report);
                } else if let Some(e) = else_branch {
                    mark_all_dead(*e, arena, report);
                }
            }
            mark_dead_decls_expr(*cond, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
            mark_dead_decls_expr(*then_branch, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
            if let Some(e) = else_branch {
                mark_dead_decls_expr(*e, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
            }
        }
        _ => walk_children_expr(expr_id, arena, |c| {
            mark_dead_decls_expr(c, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report)
        }),
    }
}

/// Traverses a statement, marking unread and side-effect-free declarations as dead. Also handles dead stores (assignments overwritten before being read).
fn mark_dead_decls_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    func: FuncId,
    def_use: &DefUseGraph,
    reads: &FxHashSet<String>,
    report: &mut DeadCodeReport,
) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::ValDecl { name, value, .. } | Stmt::VarDecl { name, value, .. } => {
            if !reads.contains(*name) {
                let se = classify_side_effect(*value, arena, module_name, sema, purity, escape, func_name_to_id);
                if is_side_effect_free(se) {
                    report.dead_stmts.insert(stmt_id);
                }
            }
            mark_dead_decls_expr(*value, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
        }
        // -- Dead store: assignment target is never read within the function and the assignment expression has no side effects --
        // Note: uses reads (not never_read) for determination, because mutable variables may be indirectly read by closures;
        // the new definition site created by assignment has no use site in the def-use graph, but closures read the latest value when called.
        // Implicit-this field assignments (`field = value` resolving to `this.field = value`) are
        // NEVER dead: they mutate instance state visible to other methods and callers.
        // Global-variable assignments are NEVER dead: they write to process-wide shared storage
        // (`global_var_storage[slot]`) visible to all functions/callers, so the store is an observable
        // side effect even when the variable is not read again within this function.
        Stmt::Assignment { target, value } => {
            if let Expr::Ident(name) = &arena.expr(*target).node {
                let key = module_expr_key(module_name, target.0 as u64);
                let is_implicit_this = sema.expr_types.get(&key)
                    .and_then(|info| info.implicit_this.as_ref())
                    .is_some();
                let is_global = def_use.global_vars.contains(*name);
                if !is_implicit_this && !is_global && !reads.contains(*name) {
                    let se = classify_side_effect(*value, arena, module_name, sema, purity, escape, func_name_to_id);
                    if is_side_effect_free(se) {
                        report.dead_stmts.insert(stmt_id);
                    }
                }
            }
            mark_dead_decls_expr(*value, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
        }
        Stmt::CompoundAssignment { target, value, .. } => {
            // Compound assignment x += v: if x is never read within the function and v has no side effects, the whole statement is a dead store
            // Implicit-this field compound assignments are NEVER dead (same rationale as Assignment).
            // Global-variable compound assignments are NEVER dead (the store writes shared storage
            // observable by other functions/callers; see `Stmt::Assignment` above).
            if let Expr::Ident(name) = &arena.expr(*target).node {
                let key = module_expr_key(module_name, target.0 as u64);
                let is_implicit_this = sema.expr_types.get(&key)
                    .and_then(|info| info.implicit_this.as_ref())
                    .is_some();
                let is_global = def_use.global_vars.contains(*name);
                if !is_implicit_this && !is_global && !reads.contains(*name) {
                    let se = classify_side_effect(*value, arena, module_name, sema, purity, escape, func_name_to_id);
                    if is_side_effect_free(se) {
                        report.dead_stmts.insert(stmt_id);
                    }
                }
            }
            mark_dead_decls_expr(*value, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
        }
        // Recurse into nested function bodies so dead code inside them is also detected.
        Stmt::LocalDecl { decl } => match decl.as_ref() {
            crate::ast::Ast::Decl::FunDecl { body, .. } => {
                mark_dead_decls_expr(*body, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
            }
            _ => {}
        },
        _ => walk_children_stmt(stmt_id, arena, |e| {
            mark_dead_decls_expr(e, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report)
        }),
    }
}

// =========================================================================
// DeadVarPass — Layer 3: dead variable elimination
// =========================================================================

/// Dead variable report.
#[derive(Debug, Default)]
pub struct DeadVarReport {
    /// Eliminable variable definition sites.
    pub dead_vars: FxHashSet<VarId>,
}

impl DeadVarReport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_dead(&self, v: VarId) -> bool {
        self.dead_vars.contains(&v)
    }
}

/// Dead variable analysis. Based on DefUseGraph:
/// - Variable definition sites that are never read are dead
/// - Variables corresponding to dead code declaration statements are also marked as dead
pub fn dead_var_pass(
    _module: &Module,
    _arena: &AstArena,
    def_use: &DefUseGraph,
    dead_code: &DeadCodeReport,
) -> DeadVarReport {
    let mut report = DeadVarReport::new();
    for (i, def) in def_use.defs.iter().enumerate() {
        let vid = VarId(i as u32);
        // Skip parameter definition sites (StmtId(u32::MAX))
        if def.stmt.0 == u32::MAX {
            continue;
        }
        // Variables corresponding to dead code declaration statements
        if dead_code.is_dead(def.stmt) {
            report.dead_vars.insert(vid);
            continue;
        }
        // Never read
        // Note: for closure-captured mutable variables, the new definition site created by assignment has no use site,
        // but the same-named variable has use sites at the old definition site (closure read), so it should not be classified as dead.
        if def_use.is_never_read(vid) && !def_use.is_name_ever_read(def.func, &def.name) {
            report.dead_vars.insert(vid);
        }
    }
    report
}

// =========================================================================
// DeadFuncPass — Layer 3: dead function elimination
// =========================================================================

/// Dead function report.
#[derive(Debug, Default)]
pub struct DeadFuncReport {
    /// Eliminable functions.
    pub dead: FxHashSet<FuncId>,
    /// Retention reasons.
    pub reachable_reasons: FxHashMap<FuncId, ReachableReason>,
}

impl DeadFuncReport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_dead(&self, f: FuncId) -> bool {
        self.dead.contains(&f)
    }
}

/// Dead function analysis. Propagates reachability from all entry points along the call graph.
pub fn dead_func_pass(cg: &CallGraph, memo: &MemoPlan) -> DeadFuncReport {
    let mut report = DeadFuncReport::new();
    let mut reachable: FxHashSet<FuncId> = FxHashSet::default();
    for (&fid, reason) in &cg.entry_reasons {
        report.reachable_reasons.insert(fid, reason.clone());
        reachable.insert(fid);
    }
    // Memoization candidates and the pure functions they call are also retained
    for cand in &memo.candidates {
        report.reachable_reasons.insert(cand.func, ReachableReason::MemoDependency);
        reachable.insert(cand.func);
        if let Some(callees) = cg.edges.get(&cand.func) {
            for &callee in callees {
                reachable.insert(callee);
                report.reachable_reasons.insert(callee, ReachableReason::MemoDependency);
            }
        }
    }
    // Worklist: propagate reachability from entries along the call graph
    let mut worklist: Vec<FuncId> = reachable.iter().copied().collect();
    while let Some(f) = worklist.pop() {
        if let Some(callees) = cg.edges.get(&f) {
            for &callee in callees {
                if !reachable.contains(&callee) {
                    reachable.insert(callee);
                    report.reachable_reasons.insert(callee, ReachableReason::CalledBy(f));
                    worklist.push(callee);
                }
            }
        }
    }
    // Unreachable functions are dead
    for &fid in &cg.nodes {
        if !reachable.contains(&fid) {
            report.dead.insert(fid);
        }
    }
    report
}

// =========================================================================
// MemoPass — Layer 3: memoization strategy decision
// =========================================================================

/// Memoization analysis. Strategy decisions (general classification, no special-case branches):
/// - Pure function + recursive (self/mutual):
///   - Tail recursive and info valid -> TailRecToLoop
///   - Non-tail recursive with single call site, no defer, and info valid -> NonTailRecToLoop
///   - Other recursive cases -> Memoize (cache all parameters)
/// - Pure function + contains loops -> LoopInvariantHoist
pub fn memo_pass(
    module: &Module,
    arena: &AstArena,
    cg: &CallGraph,
    purity: &PurityTable,
    escape: &EscapeTable,
    sema: &SemaResult,
) -> MemoPlan {
    let module_name = module.name;
    let func_name_to_id = &cg.name_to_func;
    let mut plan = MemoPlan::default();
    // Unified traversal of FunDecl + Method (via cg.iter_funcs)
    let func_metas: Vec<(FuncId, &str, &[crate::ast::Ast::Param], crate::ast::Ast::ExprId)> =
        cg.iter_funcs(module)
            .map(|(fid, meta)| (fid, meta.name, meta.params, meta.body))
            .collect();
    for (func, name, params, body_expr) in func_metas {
        if !purity.is_pure(func) {
            continue;
        }
        // Recursive functions (self-recursion + mutual recursion handled uniformly)
        if cg.recursive.contains(&func) {
            if is_tail_recursive(body_expr, arena, name) {
                // TailRecToLoop uniformly handles if-else and match tail recursion:
                // - if-else: cond = NOT(base_case_cond), Gate dispatches base/rec
                // - match: cond = Const(true), body_sg internally match Gate dispatches,
                //   rec arm's WriteBack sets Continue -> loop continues,
                //   base arm has no signal -> loop exits (returns body_sg return value)
                let info = extract_tail_rec_info(body_expr, arena, name);
                if info.is_valid() {
                    plan.candidates.push(MemoCandidate {
                        func,
                        strategy: MemoStrategy::TailRecToLoop { info },
                    });
                } else {
                    plan.candidates.push(memoize_all_params(func, params));
                }
            } else if has_non_tail_self_call(body_expr, arena, name) {
                // NonTailRecToLoop only applies when: no defer + info valid + single call site (no overlapping subproblems).
                // All other cases (defer / info invalid / 2+ call sites with overlapping subproblems) go to Memoize.
                let info = extract_non_tail_rec_info(body_expr, arena, name, params.len());
                let can_non_tail_loop = !has_defer(body_expr, arena)
                    && info.is_valid()
                    && info.call_sites.len() < 2;
                if can_non_tail_loop {
                    plan.candidates.push(MemoCandidate {
                        func,
                        strategy: MemoStrategy::NonTailRecToLoop { info },
                    });
                } else {
                    plan.candidates.push(memoize_all_params(func, params));
                }
            } else {
                // Mutual recursion (no self-call) -> Memoize
                plan.candidates.push(memoize_all_params(func, params));
            }
            continue;
        }
        // Mutually recursive pure function SCC: memoize
        if cg.mutually_recursive.iter().any(|scc| scc.contains(&func)) {
            plan.candidates.push(memoize_all_params(func, params));
            continue;
        }
        // Contains loops: collect invariants
        let invariants = collect_loop_invariants(body_expr, arena, module_name, sema, purity, escape, func_name_to_id);
        if !invariants.is_empty() {
            plan.candidates.push(MemoCandidate {
                func,
                strategy: MemoStrategy::LoopInvariantHoist { invariants },
            });
        }
    }
    plan
}

/// Constructs a Memoize candidate: caches all parameters (generic helper, eliminates duplicate construction).
fn memoize_all_params(func: FuncId, params: &[crate::ast::Ast::Param]) -> MemoCandidate {
    let param_indices: Vec<u32> = (0..params.len() as u32).collect();
    MemoCandidate {
        func,
        strategy: MemoStrategy::Memoize {
            cache_key: CacheKeySpec { param_indices },
            capacity: MemoCapacity::Unlimited,
        },
    }
}

/// Determines whether a function body is tail-recursive: at least one path's tail position is a self-call.
/// Supports if-else and Match tail recursion, and all self-calls must be in tail position.
/// The inner ack in ack(m-1, ack(m, n-1)) is a non-tail-position self-call -> rejected.
fn is_tail_recursive(body: ExprId, arena: &AstArena, self_name: &str) -> bool {
    has_tail_call(body, arena, self_name) && !has_non_tail_self_call(body, arena, self_name)
}

/// Checks whether a call to self_name exists in the tail position of an expression.
/// Recurses into if-else, Match arm body, and block trailing.
fn has_tail_call(expr_id: ExprId, arena: &AstArena, self_name: &str) -> bool {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Call { callee, .. } => {
            if let Expr::Ident(name) = &arena.expr(*callee).node {
                *name == self_name
            } else {
                false
            }
        }
        Expr::Block { trailing: Some(t), .. } => has_tail_call(*t, arena, self_name),
        Expr::If { then_branch, else_branch, .. } => {
            has_tail_call(*then_branch, arena, self_name)
                || else_branch.map_or(false, |e| has_tail_call(e, arena, self_name))
        }
        Expr::Match { arms, .. } => {
            // Each arm's body is in tail position
            arms.iter().any(|arm| has_tail_call(arm.body, arena, self_name))
        }
        _ => false,
    }
}

/// Checks whether the function body contains non-tail-position self-calls (e.g., the inner ack in ack(m-1, ack(m, n-1))).
/// Non-tail position = as an argument, operand, field value, etc.
/// If such calls exist, the function is not purely tail-recursive and cannot be safely converted to iteration.
fn has_non_tail_self_call(body: ExprId, arena: &AstArena, self_name: &str) -> bool {
    fn is_self_call(expr_id: ExprId, arena: &AstArena, self_name: &str) -> bool {
        if let Expr::Call { callee, .. } = &arena.expr(expr_id).node {
            if let Expr::Ident(name) = &arena.expr(*callee).node {
                return *name == self_name;
            }
        }
        false
    }
    /// Recursively checks whether sub-expressions contain non-tail-position self-calls.
    /// `in_tail` indicates whether the current expression is in tail position.
    fn check(expr_id: ExprId, arena: &AstArena, self_name: &str, in_tail: bool) -> bool {
        let expr = &arena.expr(expr_id).node;
        match expr {
            Expr::Call { callee, args, .. } => {
                let is_self = is_self_call(expr_id, arena, self_name);
                if is_self && !in_tail {
                    // Non-tail-position self-call -> reject
                    return true;
                }
                if is_self && in_tail {
                    // Tail-position self-call: check whether arguments contain non-tail self-calls
                    return args.iter().any(|&a| check(a, arena, self_name, false));
                }
                // Non-self-call: callee and args are all non-tail position
                check(*callee, arena, self_name, false)
                    || args.iter().any(|&a| check(a, arena, self_name, false))
            }
            Expr::Block { stmts, trailing } => {
                // Expressions in stmts are not in tail position
                for s in stmts {
                    if let Some(e) = stmt_tail_expr(*s, arena) {
                        if check(e, arena, self_name, false) {
                            return true;
                        }
                    }
                }
                trailing.map_or(false, |t| check(t, arena, self_name, in_tail))
            }
            Expr::If { cond, then_branch, else_branch, .. } => {
                check(*cond, arena, self_name, false)
                    || check(*then_branch, arena, self_name, in_tail)
                    || else_branch.map_or(false, |e| check(e, arena, self_name, in_tail))
            }
            Expr::Match { scrutinee, arms, .. } => {
                // Match itself does not block, but non-tail self-calls in arm bodies are detected
                check(*scrutinee, arena, self_name, false)
                    || arms.iter().any(|arm| check(arm.body, arena, self_name, in_tail))
            }
            Expr::Binary { lhs, rhs, .. } => {
                check(*lhs, arena, self_name, false)
                    || check(*rhs, arena, self_name, false)
            }
            Expr::Unary { operand, .. } => check(*operand, arena, self_name, false),
            Expr::ArrayLit { elements, fill } => {
                elements.iter().any(|&e| check(e, arena, self_name, false))
                    || fill.map_or(false, |(v, c)| {
                        check(v, arena, self_name, false)
                            || check(c, arena, self_name, false)
                    })
            }
            Expr::RecordLit(fields) => {
                fields.iter().any(|f| check(f.value, arena, self_name, false))
            }
            Expr::RecordExtend { base, updates } => {
                check(*base, arena, self_name, false)
                    || updates.iter().any(|f| check(f.value, arena, self_name, false))
            }
            Expr::MethodCall { recv, args, .. } => {
                check(*recv, arena, self_name, false)
                    || args.iter().any(|&a| check(a, arena, self_name, false))
            }
            Expr::FieldAccess { recv, .. } => check(*recv, arena, self_name, false),
            Expr::Index { recv, index } => {
                check(*recv, arena, self_name, false)
                    || check(*index, arena, self_name, false)
            }
            Expr::Assign { target, value } => {
                check(*target, arena, self_name, false)
                    || check(*value, arena, self_name, false)
            }
            Expr::CompoundAssign { target, value, .. } => {
                check(*target, arena, self_name, false)
                    || check(*value, arena, self_name, false)
            }
            Expr::Elvis { lhs, rhs } => {
                check(*lhs, arena, self_name, false)
                    || check(*rhs, arena, self_name, false)
            }
            Expr::RefOf(e) | Expr::Deref(e) | Expr::Propagate(e) | Expr::NonNullAssert(e)
            | Expr::Atomic(e) | Expr::Lazy(e) => check(*e, arena, self_name, false),
            _ => false,
        }
    }
    check(body, arena, self_name, true)
}

/// Extracts the expression from a statement (for non-tail-position self-call checks).
fn stmt_tail_expr(stmt_id: crate::ast::Ast::StmtId, arena: &AstArena) -> Option<ExprId> {
    match &arena.stmt(stmt_id).node {
        crate::ast::Ast::Stmt::Expression { expr } => Some(*expr),
        crate::ast::Ast::Stmt::Return { value } => *value,
        crate::ast::Ast::Stmt::ValDecl { value, .. } => Some(*value),
        crate::ast::Ast::Stmt::VarDecl { value, .. } => Some(*value),
        crate::ast::Ast::Stmt::Assignment { value, .. } => Some(*value),
        crate::ast::Ast::Stmt::FieldAssignment { value, .. } => Some(*value),
        crate::ast::Ast::Stmt::CompoundAssignment { value, .. } => Some(*value),
        _ => None,
    }
}

/// Extracts parameter transformation info from a tail-recursive function body.
///
/// Traverses the control-flow branches of the function body, classifying them into
/// base cases (non-recursive terminations) and rec branches (recursive calls).
/// Supported AST shapes:
/// - if cond { return base } else { return self(args) }
/// - if cond1 { ... } else if cond2 { return self(args2) } else { return base }
/// - match scrut { arm1 => return base, arm2 => return self(args) }
/// - block { stmts; trailing_if_or_match }
///
/// Each base case records (condition, return value); each rec branch records (condition, argument list).
/// A `None` condition denotes an else/match-wildcard fallback branch.
fn extract_tail_rec_info(
    body: ExprId,
    arena: &AstArena,
    self_name: &str,
) -> TailRecInfo {
    let mut info = TailRecInfo::default();
    collect_tail_branches(body, arena, self_name, None, &mut info);
    info
}

/// Recursively collects base cases and rec branches from control-flow branches.
/// `cond` is the inherited condition of the current branch (None = fallback/unconditional).
fn collect_tail_branches(
    expr_id: ExprId,
    arena: &AstArena,
    self_name: &str,
    cond: Option<ExprId>,
    info: &mut TailRecInfo,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        // block: recurse into trailing first; when trailing is None, check the final Return
        Expr::Block { stmts, trailing } => {
            if let Some(t) = trailing {
                collect_tail_branches(*t, arena, self_name, cond, info);
            } else if let Some(last) = stmts.last() {
                if let Stmt::Return { value } = &arena.stmt(*last).node {
                    if let Some(v) = value {
                        collect_tail_branches(*v, arena, self_name, cond, info);
                    } else {
                        info.base_cases.push((cond, expr_id));
                    }
                }
            }
        }
        // if: then branch uses Some(cond), else branch uses None (fallback)
        Expr::If { cond: if_cond, then_branch, else_branch, .. } => {
            collect_tail_branches(*then_branch, arena, self_name, Some(*if_cond), info);
            if let Some(eb) = else_branch {
                collect_tail_branches(*eb, arena, self_name, None, info);
            }
        }
        // match: each arm is dispatched separately (pattern conditions cannot be expressed as ExprId, use None)
        Expr::Match { arms, .. } => {
            for arm in arms {
                collect_tail_branches(arm.body, arena, self_name, None, info);
            }
        }
        // tail call: rec branch
        Expr::Call { callee, args, .. } => {
            if let Expr::Ident(name) = &arena.expr(*callee).node {
                if *name == self_name {
                    info.rec_branches.push((cond, args.clone()));
                    return;
                }
            }
            info.base_cases.push((cond, expr_id));
        }
        // non-tail-call expression: base case
        _ => {
            info.base_cases.push((cond, expr_id));
        }
    }
}

// =========================================================================
// Non-tail recursion to iteration: call-site extraction + continuation analysis
// =========================================================================

/// Extracts call sites and continuation info from a non-tail-recursive function body.
///
/// Traverses the function body AST, collecting all non-tail-position self-call ExprIds.
/// continuation_expr = body (the function body itself), because re-evaluating conditions of a
/// pure function yields the same result. The Builder recompiles the body for each state,
/// replacing completed calls via call_result_map.
fn extract_non_tail_rec_info(
    body: ExprId,
    arena: &AstArena,
    self_name: &str,
    param_count: usize,
) -> NonTailRecInfo {
    let mut call_sites = Vec::new();
    let mut base_cases = Vec::new();

    collect_non_tail_calls(body, arena, self_name, true, &mut call_sites, &mut base_cases);

    NonTailRecInfo {
        call_sites,
        continuation_expr: body,
        base_cases,
        param_count,
    }
}

/// Recursively collects non-tail-position self-call ExprIds and base cases.
///
/// `in_tail` indicates whether the current expression is in tail position.
/// - Self-calls in tail position are tail-recursive (handled by Tier A), not collected
/// - Self-calls in non-tail position are collected into call_sites
/// - Non-self-call expressions in tail position are collected as base cases
fn collect_non_tail_calls(
    expr_id: ExprId,
    arena: &AstArena,
    self_name: &str,
    in_tail: bool,
    call_sites: &mut Vec<ExprId>,
    base_cases: &mut Vec<(Option<ExprId>, ExprId)>,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Call { callee, args, .. } => {
            let is_self = if let Expr::Ident(name) = &arena.expr(*callee).node {
                *name == self_name
            } else {
                false
            };
            if is_self {
                if in_tail {
                    // tail-position self-call: tail recursion, does not trigger Tier B
                    base_cases.push((None, expr_id));
                } else {
                    // non-tail-position self-call: collect as call_site
                    call_sites.push(expr_id);
                    // Check arguments for additional self-calls (e.g., the inner call in ack(m-1, ack(m, n-1)))
                    for &a in args {
                        collect_non_tail_calls(a, arena, self_name, false, call_sites, base_cases);
                    }
                }
            } else {
                collect_non_tail_calls(*callee, arena, self_name, false, call_sites, base_cases);
                for &a in args {
                    collect_non_tail_calls(a, arena, self_name, false, call_sites, base_cases);
                }
                if in_tail {
                    base_cases.push((None, expr_id));
                }
            }
        }
        Expr::If { cond, then_branch, else_branch, .. } => {
            collect_non_tail_calls(*cond, arena, self_name, false, call_sites, base_cases);
            collect_non_tail_calls(*then_branch, arena, self_name, in_tail, call_sites, base_cases);
            if let Some(eb) = else_branch {
                collect_non_tail_calls(*eb, arena, self_name, in_tail, call_sites, base_cases);
            }
        }
        Expr::Block { stmts, trailing } => {
            for s in stmts {
                if let Some(e) = stmt_tail_expr(*s, arena) {
                    collect_non_tail_calls(e, arena, self_name, false, call_sites, base_cases);
                }
            }
            if let Some(t) = trailing {
                collect_non_tail_calls(*t, arena, self_name, in_tail, call_sites, base_cases);
            }
        }
        Expr::Binary { lhs, rhs, .. } => {
            collect_non_tail_calls(*lhs, arena, self_name, false, call_sites, base_cases);
            collect_non_tail_calls(*rhs, arena, self_name, false, call_sites, base_cases);
        }
        Expr::Unary { operand, .. } => {
            collect_non_tail_calls(*operand, arena, self_name, false, call_sites, base_cases);
        }
        Expr::Match { scrutinee, arms, .. } => {
            collect_non_tail_calls(*scrutinee, arena, self_name, false, call_sites, base_cases);
            for arm in arms {
                collect_non_tail_calls(arm.body, arena, self_name, in_tail, call_sites, base_cases);
            }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_non_tail_calls(*recv, arena, self_name, false, call_sites, base_cases);
            for &a in args {
                collect_non_tail_calls(a, arena, self_name, false, call_sites, base_cases);
            }
        }
        Expr::FieldAccess { recv, .. } => {
            collect_non_tail_calls(*recv, arena, self_name, false, call_sites, base_cases);
        }
        Expr::Index { recv, index } => {
            collect_non_tail_calls(*recv, arena, self_name, false, call_sites, base_cases);
            collect_non_tail_calls(*index, arena, self_name, false, call_sites, base_cases);
        }
        Expr::Elvis { lhs, rhs } => {
            collect_non_tail_calls(*lhs, arena, self_name, false, call_sites, base_cases);
            collect_non_tail_calls(*rhs, arena, self_name, false, call_sites, base_cases);
        }
        Expr::ArrayLit { elements, fill } => {
            for &e in elements {
                collect_non_tail_calls(e, arena, self_name, false, call_sites, base_cases);
            }
            if let Some((v, c)) = fill {
                collect_non_tail_calls(*v, arena, self_name, false, call_sites, base_cases);
                collect_non_tail_calls(*c, arena, self_name, false, call_sites, base_cases);
            }
        }
        Expr::RecordLit(fields) => {
            for f in fields {
                collect_non_tail_calls(f.value, arena, self_name, false, call_sites, base_cases);
            }
        }
        Expr::RecordExtend { base, updates } => {
            collect_non_tail_calls(*base, arena, self_name, false, call_sites, base_cases);
            for f in updates {
                collect_non_tail_calls(f.value, arena, self_name, false, call_sites, base_cases);
            }
        }
        Expr::Assign { target, value } => {
            collect_non_tail_calls(*target, arena, self_name, false, call_sites, base_cases);
            collect_non_tail_calls(*value, arena, self_name, false, call_sites, base_cases);
        }
        Expr::CompoundAssign { target, value, .. } => {
            collect_non_tail_calls(*target, arena, self_name, false, call_sites, base_cases);
            collect_non_tail_calls(*value, arena, self_name, false, call_sites, base_cases);
        }
        Expr::RefOf(e) | Expr::Deref(e) | Expr::Propagate(e) | Expr::NonNullAssert(e)
        | Expr::Atomic(e) | Expr::Lazy(e) => {
            collect_non_tail_calls(*e, arena, self_name, false, call_sites, base_cases);
        }
        _ => {
            if in_tail {
                base_cases.push((None, expr_id));
            }
        }
    }
}

/// Determines whether the expression is a leaf (has no sub-expressions).
fn is_leaf(expr_id: ExprId, arena: &AstArena) -> bool {
    matches!(
        &arena.expr(expr_id).node,
        Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit(_)
        | Expr::CharLit(_) | Expr::StrLit(_) | Expr::NullLit | Expr::VoidLit
        | Expr::Ident(_)
    )
}

/// Collects loop invariants: pure expressions referenced inside the loop that depend only on
/// variables defined outside the loop and not modified within it.
/// Candidates include loop conditions (While), iteration sources (For), and pure expressions
/// inside the loop body.
fn collect_loop_invariants(
    body: ExprId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
) -> Vec<ExprId> {
    let mut invariants = Vec::new();
    collect_loop_invariants_expr(body, arena, module_name, sema, purity, escape, func_name_to_id, &mut invariants);
    invariants
}

/// Recursively searches for loops (While/Loop/For statements), collecting invariant candidates.
fn collect_loop_invariants_expr(
    expr_id: ExprId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    invariants: &mut Vec<ExprId>,
) {
    // Traverse statements in Block to find loops
    walk_children_stmts_of_expr(expr_id, arena, |s| {
        collect_loop_invariants_stmt(s, arena, module_name, sema, purity, escape, func_name_to_id, invariants);
    });
    // Traverse sub-expressions to find nested Blocks
    walk_children_expr(expr_id, arena, |c| {
        collect_loop_invariants_expr(c, arena, module_name, sema, purity, escape, func_name_to_id, invariants)
    });
}

/// Traverses statements to find loops.
fn collect_loop_invariants_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    invariants: &mut Vec<ExprId>,
) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::While { condition, body } => {
            let modified = collect_modified_vars(*body, arena);
            try_add_invariant(*condition, arena, module_name, sema, purity, escape, func_name_to_id, &modified, invariants);
            collect_invariant_candidates(*body, arena, module_name, sema, purity, escape, func_name_to_id, &modified, invariants);
            collect_loop_invariants_expr(*body, arena, module_name, sema, purity, escape, func_name_to_id, invariants);
        }
        Stmt::Loop { body } => {
            let modified = collect_modified_vars(*body, arena);
            collect_invariant_candidates(*body, arena, module_name, sema, purity, escape, func_name_to_id, &modified, invariants);
            collect_loop_invariants_expr(*body, arena, module_name, sema, purity, escape, func_name_to_id, invariants);
        }
        Stmt::For { name, iterable, body } => {
            let mut modified = collect_modified_vars(*body, arena);
            modified.insert(name.to_string());
            try_add_invariant(*iterable, arena, module_name, sema, purity, escape, func_name_to_id, &modified, invariants);
            collect_invariant_candidates(*body, arena, module_name, sema, purity, escape, func_name_to_id, &modified, invariants);
            collect_loop_invariants_expr(*body, arena, module_name, sema, purity, escape, func_name_to_id, invariants);
        }
        _ => walk_children_stmt(stmt_id, arena, |e| {
            collect_loop_invariants_expr(e, arena, module_name, sema, purity, escape, func_name_to_id, invariants)
        }),
    }
}

/// Collects the set of variable names modified within the loop body.
fn collect_modified_vars(expr_id: ExprId, arena: &AstArena) -> FxHashSet<String> {
    let mut modified = FxHashSet::default();
    collect_modified_vars_inner(expr_id, arena, &mut modified);
    modified
}

fn collect_modified_vars_inner(expr_id: ExprId, arena: &AstArena, modified: &mut FxHashSet<String>) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Assign { target, value } => {
            if let Expr::Ident(name) = &arena.expr(*target).node {
                modified.insert(name.to_string());
            }
            collect_modified_vars_inner(*value, arena, modified);
        }
        Expr::CompoundAssign { target, value, .. } => {
            if let Expr::Ident(name) = &arena.expr(*target).node {
                modified.insert(name.to_string());
            }
            collect_modified_vars_inner(*value, arena, modified);
        }
        _ => walk_children_expr(expr_id, arena, |c| collect_modified_vars_inner(c, arena, modified)),
    }
    walk_children_stmts_of_expr(expr_id, arena, |s| collect_modified_vars_stmt(s, arena, modified));
}

fn collect_modified_vars_stmt(stmt_id: StmtId, arena: &AstArena, modified: &mut FxHashSet<String>) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::Assignment { target, value } => {
            if let Expr::Ident(name) = &arena.expr(*target).node {
                modified.insert(name.to_string());
            }
            collect_modified_vars_inner(*value, arena, modified);
        }
        Stmt::CompoundAssignment { target, value, .. } => {
            if let Expr::Ident(name) = &arena.expr(*target).node {
                modified.insert(name.to_string());
            }
            collect_modified_vars_inner(*value, arena, modified);
        }
        Stmt::FieldAssignment { object, value, .. } => {
            if let Expr::Ident(name) = &arena.expr(*object).node {
                modified.insert(name.to_string());
            }
            collect_modified_vars_inner(*value, arena, modified);
        }
        Stmt::VarDecl { name, value, .. } => {
            modified.insert(name.to_string());
            collect_modified_vars_inner(*value, arena, modified);
        }
        _ => walk_children_stmt(stmt_id, arena, |e| collect_modified_vars_inner(e, arena, modified)),
    }
}

/// Traverses the loop body, collecting all pure expressions that satisfy the invariant condition.
fn collect_invariant_candidates(
    expr_id: ExprId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    modified: &FxHashSet<String>,
    invariants: &mut Vec<ExprId>,
) {
    try_add_invariant(expr_id, arena, module_name, sema, purity, escape, func_name_to_id, modified, invariants);
    walk_children_expr(expr_id, arena, |c| {
        collect_invariant_candidates(c, arena, module_name, sema, purity, escape, func_name_to_id, modified, invariants)
    });
    walk_children_stmts_of_expr(expr_id, arena, |s| {
        collect_invariant_candidates_stmt(s, arena, module_name, sema, purity, escape, func_name_to_id, modified, invariants)
    });
}

fn collect_invariant_candidates_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    modified: &FxHashSet<String>,
    invariants: &mut Vec<ExprId>,
) {
    walk_children_stmt(stmt_id, arena, |e| {
        collect_invariant_candidates(e, arena, module_name, sema, purity, escape, func_name_to_id, modified, invariants)
    });
}

/// Determines whether an expression is a loop-invariant candidate: non-leaf, pure, and
/// all referenced variables are unmodified by the loop.
fn try_add_invariant(
    expr_id: ExprId,
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    purity: &PurityTable,
    escape: &EscapeTable,
    func_name_to_id: &FxHashMap<String, FuncId>,
    modified: &FxHashSet<String>,
    invariants: &mut Vec<ExprId>,
) {
    if is_leaf(expr_id, arena) {
        return;
    }
    let se = classify_side_effect(expr_id, arena, module_name, sema, purity, escape, func_name_to_id);
    if !is_side_effect_free(se) {
        return;
    }
    let mut refs = FxHashSet::default();
    collect_idents(expr_id, arena, &mut refs);
    if refs.iter().any(|name| modified.contains(name)) {
        return;
    }
    invariants.push(expr_id);
}

/// Collects all identifier names referenced within an expression.
fn collect_idents(expr_id: ExprId, arena: &AstArena, refs: &mut FxHashSet<String>) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Ident(name) => {
            refs.insert(name.to_string());
        }
        _ => walk_children_expr(expr_id, arena, |c| collect_idents(c, arena, refs)),
    }
}

// =========================================================================
// DeadParamPass — unused parameter detection
// =========================================================================

/// Unused parameter report.
#[derive(Debug, Default)]
pub struct DeadParamReport {
    /// List of (FuncId, parameter name): parameters never read by the function body.
    pub dead_params: Vec<(FuncId, String)>,
}

/// Detects parameters that are never read by the function body.
/// Parameters are registered in DefUseGraph with StmtId(u32::MAX); is_never_read is used for the check.
pub fn dead_param_pass(module: &Module, def_use: &DefUseGraph, cg: &CallGraph) -> DeadParamReport {
    let mut report = DeadParamReport::default();
    // Unified traversal of FunDecl + Method
    for (func, meta) in cg.iter_funcs(module) {
        for param in meta.params {
            if let Some(vid) = def_use.lookup(func, param.name) {
                if def_use.defs[vid.0 as usize].stmt == StmtId(u32::MAX)
                    && def_use.is_never_read(vid)
                {
                    report.dead_params.push((func, param.name.to_string()));
                }
            }
        }
    }
    report
}

// =========================================================================
// InlinePass — inline candidate analysis
// =========================================================================

/// Inline threshold: functions with AST node count <= this value and pure are recommended for inlining.
const INLINE_SIZE_THRESHOLD: usize = 15;

/// Inline candidate report.
#[derive(Debug, Default)]
pub struct InlineReport {
    /// List of functions recommended for inlining: (FuncId, function body size)
    pub candidates: Vec<(FuncId, usize)>,
    /// Call site ExprId → callee FuncId.
    /// IrBuilder looks up this table when compiling Call; on hit, it inlines the callee body
    /// instead of launching a sub-graph.
    pub expansions: FxHashMap<ExprId, FuncId>,
}

/// Inline candidate analysis: small pure functions + non-recursive + non-async/throwing.
/// Produces inline recommendations for the IR layer to act on.
pub fn inline_pass(
    module: &Module,
    arena: &AstArena,
    cg: &CallGraph,
    purity: &PurityTable,
    sema: &SemaResult,
) -> InlineReport {
    let mut report = InlineReport::default();
    // Allow disabling AST inlining for debugging (FROND_NO_AST_INLINE=1).
    // AST inlining is independent of OptLevel — it happens at compile time,
    // not in the optimizer pipeline.
    if std::env::var("FROND_NO_AST_INLINE").is_ok() {
        return report;
    }
    // Pass 1: collect the set of inlineable functions (unified traversal of FunDecl + Method)
    let mut inlineable: FxHashSet<FuncId> = FxHashSet::default();
    let func_metas: Vec<(FuncId, &str, crate::ast::Ast::ExprId, bool)> = cg.iter_funcs(module)
        .map(|(fid, meta)| (fid, meta.name, meta.body, meta.is_async))
        .collect();
    for (func, name, body, is_async) in func_metas {
        // Non-pure functions are not inlined (may have side-effect dependencies)
        if !purity.is_pure(func) {
            continue;
        }
        // Recursive functions are not inlined (would expand infinitely)
        if cg.recursive.contains(&func) {
            continue;
        }
        // async/throwing functions are not inlined.
        if is_async {
            continue;
        }
        // All-owners check: same-named functions across modules each count.
        if sema.func_sigs_named(name).iter().any(|sig| sig.is_async || sig.is_throwing) {
            continue;
        }
        // Entry functions (Entry/ExternC/ExternAttr) are not inlined
        if let Some(reason) = cg.entry_reasons.get(&func) {
            if reason.is_definite() {
                continue;
            }
        }
        // Functions containing nested functions (Lambda/LocalDecl) are not inlined
        if has_nested_function(body, arena) {
            continue;
        }
        // W4c: bodies with early `return` / `?` ARE now inlinable — the Builder
        // wraps them in a capture Gate (branch subgraph whose Return becomes
        // the call-site value instead of leaking into the caller frame). The
        // Builder re-detects them with the same helpers to pick the wrap path.
        // Functions containing defer statements are not inlined
        if has_defer(body, arena) {
            continue;
        }
        let size = count_expr_nodes(body, arena);
        if size <= INLINE_SIZE_THRESHOLD {
            report.candidates.push((func, size));
            inlineable.insert(func);
        }
    }
    // Pass 2: filter call sites that call inlineable functions, producing expansions
    for (&expr_id, &callee) in &cg.call_sites {
        if inlineable.contains(&callee) {
            report.expansions.insert(expr_id, callee);
        }
    }
    report
}

/// Detects whether an expression contains nested functions (Lambda or FunDecl within LocalDecl).
/// Functions with nested functions should not be inlined: inlining would introduce new sub-graphs
/// whose node ranges conflict with the outer sub-graph's node_range, causing prepare_frame to
/// mistakenly mark nested nodes as never-ready.
fn has_nested_function(expr_id: ExprId, arena: &AstArena) -> bool {
    if matches!(arena.expr(expr_id).node, Expr::Lambda { .. }) {
        return true;
    }
    let mut found = false;
    walk_children_expr(expr_id, arena, |c| {
        if !found {
            found = has_nested_function(c, arena);
        }
    });
    if !found {
        walk_children_stmts_of_expr(expr_id, arena, |s| {
            if !found {
                found = has_nested_function_stmt(s, arena);
            }
        });
    }
    found
}

fn has_nested_function_stmt(stmt_id: StmtId, arena: &AstArena) -> bool {
    let stmt = &arena.stmt(stmt_id).node;
    if let Stmt::LocalDecl { decl } = stmt {
        if matches!(decl.as_ref(), crate::ast::Ast::Decl::FunDecl { .. }) {
            return true;
        }
    }
    let mut found = false;
    walk_children_stmt(stmt_id, arena, |e| {
        if !found {
            found = has_nested_function(e, arena);
        }
    });
    found
}

/// Detects whether an expression contains the `?` propagation operator (Expr::Propagate).
/// Functions with `?` should not be inlined: compute_propagate implements early return via
/// ControlSignal::Return, which is function-scoped; inlining would incorrectly terminate the
/// caller function.
pub(crate) fn has_propagate(expr_id: ExprId, arena: &AstArena) -> bool {
    if matches!(arena.expr(expr_id).node, Expr::Propagate(_)) {
        return true;
    }
    let mut found = false;
    walk_children_expr(expr_id, arena, |c| {
        if !found {
            found = has_propagate(c, arena);
        }
    });
    if !found {
        walk_children_stmts_of_expr(expr_id, arena, |s| {
            if !found {
                found = has_propagate_stmt(s, arena);
            }
        });
    }
    found
}

fn has_propagate_stmt(stmt_id: StmtId, arena: &AstArena) -> bool {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        // defer body is compiled into a separate sub-graph; ? propagation does not affect the outer function
        Stmt::Defer { .. } => false,
        _ => {
            let mut found = false;
            walk_children_stmt(stmt_id, arena, |e| {
                if !found {
                    found = has_propagate(e, arena);
                }
            });
            found
        }
    }
}

/// Detects whether an expression contains defer statements.
/// Functions with defer cannot be inlined: defer registers into the function sub-graph's
/// defer_table; after inlining the function frame is not created, so defer_table is never
/// checked (Bug #47).
/// Defer within lambda bodies is not counted (has_nested_function already excludes functions
/// containing lambdas).
fn has_defer(expr_id: ExprId, arena: &AstArena) -> bool {
    let mut found = false;
    walk_children_stmts_of_expr(expr_id, arena, |s| {
        if !found {
            found = has_defer_stmt(s, arena);
        }
    });
    if !found {
        walk_children_expr(expr_id, arena, |c| {
            if !found {
                found = has_defer(c, arena);
            }
        });
    }
    found
}

fn has_defer_stmt(stmt_id: StmtId, arena: &AstArena) -> bool {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::Defer { .. } => true,
        // Defer within lambda bodies is not counted (lambdas have independent frames)
        // has_nested_function already excludes functions containing lambdas
        _ => {
            let mut found = false;
            walk_children_stmt(stmt_id, arena, |e| {
                if !found {
                    found = has_defer(e, arena);
                }
            });
            found
        }
    }
}

/// Detects whether an expression contains return statements (function-scoped).
/// Return within lambda bodies is not counted (scoped to the lambda).
/// Return within defer bodies is not counted (defer body is compiled into a separate sub-graph).
pub(crate) fn has_return(expr_id: ExprId, arena: &AstArena) -> bool {
    let mut found = false;
    walk_children_stmts_of_expr(expr_id, arena, |s| {
        if !found {
            found = has_return_stmt(s, arena);
        }
    });
    if !found {
        walk_children_expr(expr_id, arena, |c| {
            if !found {
                found = has_return(c, arena);
            }
        });
    }
    found
}

fn has_return_stmt(stmt_id: StmtId, arena: &AstArena) -> bool {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::Return { .. } => true,
        // defer body is compiled into a separate sub-graph; return does not affect the outer function
        Stmt::Defer { .. } => false,
        _ => {
            let mut found = false;
            walk_children_stmt(stmt_id, arena, |e| {
                if !found {
                    found = has_return(e, arena);
                }
            });
            found
        }
    }
}

/// Recursively counts the number of AST nodes in an expression subtree.
fn count_expr_nodes(expr_id: ExprId, arena: &AstArena) -> usize {
    let mut count = 1;
    walk_children_expr(expr_id, arena, |c| {
        count += count_expr_nodes(c, arena);
    });
    walk_children_stmts_of_expr(expr_id, arena, |s| {
        count += count_stmt_nodes(s, arena);
    });
    count
}

fn count_stmt_nodes(stmt_id: StmtId, arena: &AstArena) -> usize {
    let mut count = 1;
    walk_children_stmt(stmt_id, arena, |e| {
        count += count_expr_nodes(e, arena);
    });
    count
}

// =========================================================================
// StackAllocPass — stack allocation recommendations
// =========================================================================

/// Stack allocation recommendation report.
#[derive(Debug, Default)]
pub struct StackAllocReport {
    /// List of allocation site ExprIds eligible for stack allocation (marked NoEscape by EscapeTable)
    pub candidates: Vec<ExprId>,
}

/// Stack allocation recommendations: allocations marked NoEscape by EscapeTable can be
/// converted to stack allocations.
/// Data is produced by EscapeAnalyzer; this pass aggregates and reports it for the IR layer.
pub fn stack_alloc_pass(escape: &EscapeTable) -> StackAllocReport {
    let mut report = StackAllocReport::default();
    for (&expr_id, &info) in escape.map.iter() {
        if info == EscapeInfo::NoEscape {
            report.candidates.push(expr_id);
        }
    }
    report
}

// =========================================================================
// MatchExhaustivenessPass — pattern match exhaustiveness + unreachable arm detection
// =========================================================================

/// Pattern match analysis report.
#[derive(Debug, Default)]
pub struct MatchReport {
    /// Non-exhaustive matches: (match expression ExprId, scrutinee type name, missing constructor names)
    pub non_exhaustive: Vec<(ExprId, String, Vec<String>)>,
    /// Unreachable match arms: (match expression ExprId, arm index)
    pub unreachable_arms: Vec<(ExprId, usize)>,
}

/// Pattern match analysis:
/// - Non-exhaustive detection: if a match on an ADT type does not cover all constructors and
///   has no Wildcard, report the missing constructors
/// - Unreachable arm detection: arms after a Wildcard are unreachable
pub fn match_pass(module: &Module, arena: &AstArena, sema: &SemaResult, cg: &CallGraph) -> MatchReport {
    let mut report = MatchReport::default();
    // Unified traversal of FunDecl + Method
    for (_fid, meta) in cg.iter_funcs(module) {
        analyze_match_expr(meta.body, arena, module.name, sema, &mut report);
    }
    report
}

fn analyze_match_expr(expr_id: ExprId, arena: &AstArena, module_name: &str, sema: &SemaResult, report: &mut MatchReport) {
    let expr = &arena.expr(expr_id).node;
    if let Expr::Match { scrutinee, arms } = expr {
        analyze_single_match(expr_id, *scrutinee, arms, arena, module_name, sema, report);
    }
    // Recurse into sub-expressions
    walk_children_expr(expr_id, arena, |c| {
        analyze_match_expr(c, arena, module_name, sema, report);
    });
    // Recurse into statements within Blocks
    walk_children_stmts_of_expr(expr_id, arena, |s| {
        analyze_match_stmt(s, arena, module_name, sema, report);
    });
}

fn analyze_match_stmt(stmt_id: StmtId, arena: &AstArena, module_name: &str, sema: &SemaResult, report: &mut MatchReport) {
    walk_children_stmt(stmt_id, arena, |e| {
        analyze_match_expr(e, arena, module_name, sema, report);
    });
}

fn analyze_single_match(
    match_expr: ExprId,
    scrutinee: ExprId,
    arms: &[crate::ast::Ast::MatchArm],
    arena: &AstArena,
    module_name: &str,
    sema: &SemaResult,
    report: &mut MatchReport,
) {
    // 1. Unreachable arm detection: arms after a Wildcard are unreachable (a guardless Wildcard terminates matching)
    let mut wildcard_seen = false;
    for (i, arm) in arms.iter().enumerate() {
        let pat = &arena.pattern(arm.pattern).node;
        let is_wildcard = matches!(pat, Pattern::Wildcard);
        let has_guard = arm.guard.is_some() || matches!(pat, Pattern::Guard { .. });
        if wildcard_seen && !has_guard {
            report.unreachable_arms.push((match_expr, i));
        }
        if is_wildcard && !has_guard {
            wildcard_seen = true;
        }
    }

    // 2. Non-exhaustive detection: when the scrutinee type is an ADT, collect all constructor names covered by arms
    let key = module_expr_key(module_name, scrutinee.0 as u64);
    let Some(info) = sema.expr_types.get(&key) else { return };
    let Some(type_name) = info.type_name.as_deref() else { return };
    let Some(type_idx) = sema.type_def_idx(type_name) else { return };
    let type_def = &sema.type_defs[&type_idx];
    // Only ADT types with multiple constructors require exhaustiveness checking
    if type_def.kind != crate::sema::Sema::TypeDefKind::Adt {
        return;
    }
    let all_ctors: Vec<&str> = type_def.constructors.iter().map(|c| c.name.as_ref()).collect();
    if all_ctors.is_empty() {
        return;
    }

    // Collect constructor names covered by arms
    let mut covered: FxHashSet<String> = FxHashSet::default();
    let mut has_wildcard = false;
    for arm in arms {
        let pat = &arena.pattern(arm.pattern).node;
        collect_pattern_ctors(pat, arena, &mut covered, &mut has_wildcard);
    }

    // With a Wildcard, conservatively treat as exhaustive
    if has_wildcard {
        return;
    }

    // Find missing constructors
    let missing: Vec<String> = all_ctors
        .iter()
        .filter(|c| !covered.contains(**c))
        .map(|c| c.to_string())
        .collect();
    if !missing.is_empty() {
        report.non_exhaustive.push((match_expr, type_name.to_string(), missing));
    }
}

/// Recursively collects constructor names covered by a pattern.
fn collect_pattern_ctors(
    pattern: &Pattern,
    arena: &AstArena,
    covered: &mut FxHashSet<String>,
    has_wildcard: &mut bool,
) {
    match pattern {
        Pattern::Wildcard | Pattern::Variable { .. } => {
            *has_wildcard = true;
        }
        Pattern::Literal(_) => {
            // Literal patterns do not cover constructors; conservatively treat as wildcard (may cover partial values)
            *has_wildcard = true;
        }
        Pattern::Constructor { name, patterns } => {
            covered.insert(name.to_string());
            for p in patterns {
                collect_pattern_ctors(&arena.pattern(*p).node, arena, covered, has_wildcard);
            }
        }
        Pattern::Record { .. } => {
            *has_wildcard = true;
        }
        Pattern::OrPattern { left, right } => {
            collect_pattern_ctors(&arena.pattern(*left).node, arena, covered, has_wildcard);
            collect_pattern_ctors(&arena.pattern(*right).node, arena, covered, has_wildcard);
        }
        Pattern::Guard { pattern, .. } => {
            // Patterns with guards do not guarantee coverage; conservatively treat as potentially non-covering
            collect_pattern_ctors(&arena.pattern(*pattern).node, arena, covered, has_wildcard);
        }
    }
}

// =========================================================================
// AnalysisReport — aggregated report + rayon three-layer parallel entry point
// =========================================================================

/// Loop analysis report (populated by LoopAnalysis.rs after IR construction).
#[derive(Debug, Default)]
pub struct LoopAnalysisReport {
    /// List of invariant nodes for each loop body_sg.
    /// key = SubGraphId of body_sg, value = list of invariant NodeIds within body_sg.
    pub invariants: FxHashMap<crate::ir::Ir::SubGraphId, Vec<crate::ir::Ir::NodeId>>,
    /// Unrollable loops.
    /// key = SubGraphId of the loop sub-graph, value = unroll information.
    pub unrollable: FxHashMap<crate::ir::Ir::SubGraphId, UnrollInfo>,
}

/// Loop unrolling information.
#[derive(Debug, Clone)]
pub struct UnrollInfo {
    /// Compile-time known trip count
    pub trip_count: u32,
    /// Binding node of the loop variable within body_sg
    pub loop_var_node: crate::ir::Ir::NodeId,
    /// Loop start value
    pub start_value: i128,
    /// Loop step
    pub step: i128,
    /// SubGraphId of body_sg
    pub body_sg: crate::ir::Ir::SubGraphId,
    /// Original ConstValue of the Range start (used to preserve type consistency)
    pub start_const: crate::ir::Ir::ConstValue,
}

/// Aggregated static analysis report.
#[derive(Debug)]
pub struct AnalysisReport {
    pub def_use: DefUseGraph,
    pub call_graph: CallGraph,
    pub purity: PurityTable,
    pub escape: EscapeTable,
    pub dead_code: DeadCodeReport,
    pub dead_var: DeadVarReport,
    pub dead_func: DeadFuncReport,
    pub memo: MemoPlan,
    pub dead_param: DeadParamReport,
    pub inline: InlineReport,
    pub stack_alloc: StackAllocReport,
    pub match_report: MatchReport,
    pub loop_analysis: LoopAnalysisReport,
}

/// Runs the full three-layer analysis pipeline.
///
/// Layer 1: DefUseBuilder + CallGraphBuilder (rayon::join parallel, no data dependency)
/// Layer 2: PurityAnalyzer → EscapeAnalyzer (sequential, Escape depends on Purity)
/// Layer 3: DeadCodePass + MemoPass can run in parallel; DeadVarPass depends on DeadCodeReport;
///          DeadFuncPass depends on MemoPlan.
/// Layer 4: DeadParam + Inline + StackAlloc + Match (can run in parallel, depend on Layer 1-2 output)
pub fn analyze(module: &Module, arena: &AstArena, sema: &SemaResult) -> AnalysisReport {
    let module_name = module.name;

    // Layer 1: build def-use graph and call graph in parallel
    let (def_use, call_graph) = rayon::join(
        || build_def_use(module, arena),
        || build_call_graph(module, arena, sema),
    );

    // Layer 2: sequential — EscapeAnalyzer depends on PurityTable
    let purity = analyze_purity(module, arena, &call_graph, sema);
    let escape = analyze_escape(module, arena, &call_graph, &purity);

    // Layer 3: DeadCodePass and MemoPass have no mutual dependency, can run in parallel
    let (dead_code, memo) = rayon::join(
        || dead_code_pass(module, arena, module_name, sema, &purity, &escape, &call_graph, &def_use),
        || memo_pass(module, arena, &call_graph, &purity, &escape, sema),
    );
    let dead_var = dead_var_pass(module, arena, &def_use, &dead_code);
    let dead_func = dead_func_pass(&call_graph, &memo);

    // Layer 4: four new passes run in parallel (depend on Layer 1-2 output)
    let ((dead_param, inline), (stack_alloc, match_report)) = rayon::join(
        || rayon::join(
            || dead_param_pass(module, &def_use, &call_graph),
            || inline_pass(module, arena, &call_graph, &purity, sema),
        ),
        || rayon::join(
            || stack_alloc_pass(&escape),
            || match_pass(module, arena, sema, &call_graph),
        ),
    );

    AnalysisReport {
        def_use,
        call_graph,
        purity,
        escape,
        dead_code,
        dead_var,
        dead_func,
        memo,
        dead_param,
        inline,
        stack_alloc,
        match_report,
        loop_analysis: LoopAnalysisReport::default(), // populated by analyze_loops in this file after IR construction
    }
}

// =========================================================================
// Loop analysis pass (merged from LoopAnalysis.rs)
//
// Produces LoopAnalysisReport:
// - Invariant identification: nodes in body_sg that are pure computations with inputs from outside the loop
// - Trip count estimation: compile-time trip count when a For loop iterator is a constant Range
// =========================================================================

use crate::ir::Ir::{
    ComputeFnId, ConstValue, DataFlowGraph, LoopKind, NodeId, SubGraphId,
    CF_CALL_LAUNCH, CF_RANGE, CF_RANGE_INCLUSIVE,
};

/// Maximum number of body nodes to unroll
const MAX_UNROLL_BODY_NODES: usize = 32;
/// Maximum trip count to unroll
const MAX_UNROLL: u32 = 8;

/// Runs loop analysis, populating LoopAnalysisReport.
///
/// This function runs after IR construction and directly analyzes the DataFlowGraph.
/// Analyzer.rs's analyze() runs before IR construction (consuming AST + SemaResult),
/// so loop_analysis must be populated by main.rs calling this function after IR construction.
pub fn analyze_loops(graph: &DataFlowGraph) -> LoopAnalysisReport {
    let mut report = LoopAnalysisReport::default();
    // W1: single derivation point (Bug #99 aliasing-read subtraction included).
    let pure_set = crate::ir::Ir::graph_pure_set(graph);

    // Collect all loop sub-graphs (loop_kind != None and != LoopBody)
    let loop_sgs: Vec<SubGraphId> = graph
        .subgraphs
        .iter()
        .enumerate()
        .filter(|(_, sg)| sg.loop_kind != LoopKind::None && sg.loop_kind != LoopKind::LoopBody)
        .map(|(i, _)| SubGraphId(i as u32))
        .collect();

    for loop_sg_id in &loop_sgs {
        let loop_sg = &graph.subgraphs[loop_sg_id.0 as usize];

        // Find the corresponding body_sg (loop_kind == LoopBody and loop_parent_sg == loop_sg_id)
        let body_sg_id = graph
            .subgraphs
            .iter()
            .enumerate()
            .find(|(_, sg)| {
                sg.loop_kind == LoopKind::LoopBody && sg.loop_parent_sg == Some(*loop_sg_id)
            })
            .map(|(i, _)| SubGraphId(i as u32));

        let Some(body_sg_id) = body_sg_id else { continue };

        // ── Invariant identification ──
        let invariants = find_invariants(graph, *loop_sg_id, body_sg_id, &pure_set);
        if !invariants.is_empty() {
            report.invariants.insert(body_sg_id, invariants);
        }

        // ── Loop unrolling analysis (For loops only) ──
        if loop_sg.loop_kind == LoopKind::For {
            if let Some(unroll_info) = analyze_unroll(graph, *loop_sg_id, body_sg_id) {
                report.unrollable.insert(*loop_sg_id, unroll_info);
            }
        }
    }

    report
}

/// Identifies loop-invariant nodes within body_sg.
fn find_invariants(
    graph: &DataFlowGraph,
    loop_sg_id: SubGraphId,
    body_sg_id: SubGraphId,
    pure_set: &FxHashSet<ComputeFnId>,
) -> Vec<NodeId> {
    let loop_sg = &graph.subgraphs[loop_sg_id.0 as usize];
    let body_sg = &graph.subgraphs[body_sg_id.0 as usize];
    let (body_start, body_end) = body_sg.node_range;

    // Loop-variable-dependent nodes (cond_node, iter_next_node)
    let mut loop_deps: FxHashSet<NodeId> = FxHashSet::default();
    if let Some(c) = loop_sg.cond_node {
        loop_deps.insert(c);
    }
    if let Some(n) = loop_sg.iter_next_node {
        loop_deps.insert(n);
    }

    // Modified set within body_sg (WriteBack machinery deleted B/C 2026-08-22 —
    // cell stores go through the engine-level Cell, not node metadata).
    let modified: FxHashSet<NodeId> = FxHashSet::default();

    // Pre-compute all loop sub-graph ranges (loop_kind != None) to determine whether a node is at function level.
    // The hoist target is the function-level sub-graph; only nodes whose inputs are all at function level
    // (not inside any loop sub-graph) or already identified as invariants can be safely hoisted — nodes
    // depending on loop variables will not be hoisted, because after hoisting to function level their
    // values would no longer change with the loop.
    let loop_ranges: Vec<(u32, u32)> = graph
        .subgraphs
        .iter()
        .filter(|sg| sg.loop_kind != LoopKind::None)
        .map(|sg| (sg.node_range.0 .0, sg.node_range.1 .0))
        .collect();
    let is_func_level = |nid: NodeId| -> bool {
        !loop_ranges.iter().any(|&(s, e)| nid.0 >= s && nid.0 < e)
    };

    // Iteratively identify invariants (multiple scan rounds until fixpoint)
    let mut invariants: Vec<NodeId> = Vec::new();
    let mut invariant_set: FxHashSet<NodeId> = FxHashSet::default();

    let mut changed = true;
    while changed {
        changed = false;
        for idx in (body_start.0 as usize)..(body_end.0 as usize) {
            let nid = NodeId(idx as u32);
            if invariant_set.contains(&nid) {
                continue;
            }

            let node = graph.nodes[idx];

            // Must not be a launch kind (Gate/Call/Await/EventSource — W1 shared predicate)
            if crate::ir::Ir::is_launch_kind(node.kind) {
                continue;
            }

            // Must be a pure computation
            if !pure_set.contains(&node.compute_fn) {
                continue;
            }

            // All inputs must be at function level or already identified as invariants
            let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
            let mut all_invariant = true;
            for &input in inputs {
                if loop_deps.contains(&input) || modified.contains(&input) {
                    all_invariant = false;
                    break;
                }
                if !is_func_level(input) && !invariant_set.contains(&input) {
                    all_invariant = false;
                    break;
                }
            }

            if all_invariant {
                invariant_set.insert(nid);
                invariants.push(nid);
                changed = true;
            }
        }
    }

    invariants
}

/// Analyzes whether a For loop can be unrolled.
fn analyze_unroll(
    graph: &DataFlowGraph,
    loop_sg_id: SubGraphId,
    body_sg_id: SubGraphId,
) -> Option<UnrollInfo> {
    let loop_sg = &graph.subgraphs[loop_sg_id.0 as usize];
    let body_sg = &graph.subgraphs[body_sg_id.0 as usize];

    // Body node count limit
    let body_size = (body_sg.node_range.1.0 - body_sg.node_range.0.0) as usize;
    if body_size > MAX_UNROLL_BODY_NODES {
        return None;
    }

    // Body must not contain break/continue/return/throw
    for idx in (body_sg.node_range.0.0 as usize)..(body_sg.node_range.1.0 as usize) {
        if crate::ir::Ir::is_control_flow_compute_fn(graph.nodes[idx].compute_fn) {
            return None;
        }
    }

    // Search for CF_RANGE / CF_RANGE_INCLUSIVE construction nodes within loop_sg
    let (loop_start, loop_end) = loop_sg.node_range;
    let mut range_node: Option<NodeId> = None;
    let mut range_inclusive = false;
    for idx in (loop_start.0 as usize)..(loop_end.0 as usize) {
        let node = &graph.nodes[idx];
        if node.compute_fn == CF_RANGE {
            range_node = Some(NodeId(idx as u32));
            range_inclusive = false;
            break;
        }
        if node.compute_fn == CF_RANGE_INCLUSIVE {
            range_node = Some(NodeId(idx as u32));
            range_inclusive = true;
            break;
        }
    }
    let range_node = range_node?;

    // Range inputs = [start, end]
    let range_node_struct = graph.nodes[range_node.0 as usize];
    let range_inputs = graph.inputs_pool.get(
        range_node_struct.inputs_offset,
        range_node_struct.input_count,
    );
    if range_inputs.len() < 2 {
        return None;
    }

    let start_cv = graph
        .const_values
        .get(range_inputs[0].0 as usize)
        .and_then(|o| o.as_ref())?;
    let end_cv = graph
        .const_values
        .get(range_inputs[1].0 as usize)
        .and_then(|o| o.as_ref())?;

    let start_val = const_to_i128(start_cv)?;
    let end_val = const_to_i128(end_cv)?;

    let step: i128 = 1;
    let trip_count = if range_inclusive {
        if end_val < start_val {
            return None;
        }
        ((end_val - start_val) / step + 1) as u32
    } else {
        if end_val <= start_val {
            return None;
        }
        ((end_val - start_val) / step) as u32
    };

    if trip_count == 0 || trip_count > MAX_UNROLL {
        return None;
    }

    // iter_next_node must exist and be a Call node (Range next call)
    let iter_next = loop_sg.iter_next_node?;
    let iter_node = &graph.nodes[iter_next.0 as usize];
    if iter_node.compute_fn != CF_CALL_LAUNCH {
        return None;
    }

    // body_sg structure: param_0 = iterator, param_1 = current value (loop variable)
    // loop_var_node = the second parameter node of body_sg (param_1 = current value)
    let loop_var_node = NodeId(body_sg.node_range.0.0 + 1);

    Some(UnrollInfo {
        trip_count,
        loop_var_node,
        start_value: start_val,
        step,
        body_sg: body_sg_id,
        start_const: *start_cv,
    })
}

/// Extracts an i128 from a ConstValue.
fn const_to_i128(cv: &ConstValue) -> Option<i128> {
    use crate::ir::Ir::ConstValue::*;
    match cv {
        I8(v) => Some(*v as i128),
        I16(v) => Some(*v as i128),
        I32(v) => Some(*v as i128),
        I64(v) => Some(*v as i128),
        I128(v) => Some(*v),
        U8(v) => Some(*v as i128),
        U16(v) => Some(*v as i128),
        U32(v) => Some(*v as i128),
        U64(v) => Some(*v as i128),
        U128(v) => Some(*v as i128),
        Isize(v) => Some(*v as i128),
        Usize(v) => Some(*v as i128),
        _ => None,
    }
}
