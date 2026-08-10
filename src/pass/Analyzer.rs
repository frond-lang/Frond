//! Analyzer.rs — Sema 后静态分析器
//!
//! 产出 AnalysisReport：死代码/死变量/死函数 + 记忆化策略。
//! 三层多 pass 管线，rayon 并行。类型驱动副作用判定。
//! 详见 docs/superpowers/specs/2026-08-03-analyzer-design.md

use crate::ast::Ast::{
    AstArena, Decl, Expr, ExprId, InterpolationPart, LambdaBody, Module, Pattern, PatternId,
    SelectArm, Stmt, StmtId, Visibility,
};
use crate::sema::Sema::{module_expr_key, ConstVal, SemaResult};
use crate::types::dynamic_type_id;
use rustc_hash::{FxHashMap, FxHashSet};

// =========================================================================
// 句柄类型（DOD 风格，与 Ast.rs 的 ExprId/StmtId 一致）
// =========================================================================

/// 函数索引（指向 Module.declarations 的下标）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

/// 变量定义点索引（指向 DefUseGraph.defs）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VarId(pub u32);

/// 使用点索引（指向 DefUseGraph.uses）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UseId(pub u32);

// =========================================================================
// DefUseGraph — def-use 链 + 活跃变量
// =========================================================================

/// 变量定义点。
#[derive(Debug, Clone)]
pub struct DefNode {
    pub name: String,
    /// 定义语句（ValDecl/VarDecl）或赋值（Assignment）所在 StmtId。
    /// 参数无对应语句，用 StmtId(0) 占位。
    pub stmt: StmtId,
    /// 所在函数
    pub func: FuncId,
    /// 是否可变绑定（var / 赋值目标）
    pub is_mutable: bool,
}

/// 变量使用点。
#[derive(Debug, Clone)]
pub struct UseNode {
    pub var: VarId,
    /// 读取所在表达式
    pub expr: ExprId,
    /// 所在函数
    pub func: FuncId,
}

/// 每个函数内的 def-use 链 + 活跃变量集。
#[derive(Debug, Default)]
pub struct DefUseGraph {
    pub defs: Vec<DefNode>,
    pub uses: Vec<UseNode>,
    /// VarId -> 使用点列表
    pub def_to_uses: Vec<Vec<UseId>>,
    /// 变量名 -> 定义点（同一函数内最近定义）。key = (func, name)
    pub name_to_def: FxHashMap<(FuncId, String), VarId>,
    /// 函数入口活跃集（参数）
    pub live_in: FxHashMap<FuncId, FxHashSet<VarId>>,
    /// 函数出口活跃集（本分析中恒为空，预留）
    pub live_out: FxHashMap<FuncId, FxHashSet<VarId>>,
    /// 全局变量名集合（顶层 VarDecl/ValDecl）。函数内对全局变量的赋值
    /// 不注册局部定义点，避免被误判为死变量。
    pub global_vars: FxHashSet<String>,
}

impl DefUseGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个变量定义，返回其 VarId。
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

    /// 注册一个变量使用点。
    pub fn add_use(&mut self, var: VarId, expr: ExprId, func: FuncId) -> UseId {
        let id = UseId(self.uses.len() as u32);
        self.uses.push(UseNode { var, expr, func });
        self.def_to_uses[var.0 as usize].push(id);
        id
    }

    /// 查询函数内某变量名的当前定义点。
    pub fn lookup(&self, func: FuncId, name: &str) -> Option<VarId> {
        self.name_to_def.get(&(func, name.to_string())).copied()
    }

    /// 该变量是否从未被读取。
    pub fn is_never_read(&self, var: VarId) -> bool {
        self.def_to_uses[var.0 as usize].is_empty()
    }

    /// 检查函数内某变量名是否有任何使用点（跨所有定义点）。
    /// 用于闭包捕获的可变变量：赋值创建的新定义点可能无使用点，
    /// 但同名变量在旧定义点有使用点（闭包读取），不应判为死变量。
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
// CallGraph — 调用图 + 递归检测
// =========================================================================

/// 函数可达性保留原因。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReachableReason {
    // ── 确定入口（一定可达，永不消除）──
    /// is_entry=true
    Entry,
    /// extern_c_body 非 None
    ExternC,
    /// @extern 属性
    ExternAttr,
    // ── 保守入口（可能可达，不消除但标记原因）──
    /// trait 方法
    TraitMethod,
    /// 类型方法（impl 块内方法）
    TypeMethod,
    /// public 可见性
    Public,
    // ── 可达性传播结果 ──
    /// 被某可达函数调用
    CalledBy(FuncId),
    /// 被记忆化候选依赖
    MemoDependency,
}

impl ReachableReason {
    /// 确定入口：永不可达性分析消除。
    pub fn is_definite(&self) -> bool {
        matches!(self, Self::Entry | Self::ExternC | Self::ExternAttr)
    }
    /// 保守入口：单模块分析时不消除，未来跨模块分析可降级。
    pub fn is_conservative(&self) -> bool {
        matches!(self, Self::TraitMethod | Self::TypeMethod | Self::Public)
    }
}

/// 调用图。
#[derive(Debug, Default)]
pub struct CallGraph {
    pub nodes: Vec<FuncId>,
    /// caller -> [callee]
    pub edges: FxHashMap<FuncId, Vec<FuncId>>,
    /// callee -> [callers]（逆向图）
    pub reverse: FxHashMap<FuncId, Vec<FuncId>>,
    /// 直接递归函数
    pub recursive: FxHashSet<FuncId>,
    /// 相互递归的 SCC（强连通分量）
    pub mutually_recursive: Vec<FxHashSet<FuncId>>,
    /// 入口/保留原因
    pub entry_reasons: FxHashMap<FuncId, ReachableReason>,
    /// 函数名 -> FuncId（FunDecl 名 + 方法 mangled 名 "Type.method"）
    pub name_to_func: FxHashMap<String, FuncId>,
    /// 调用点 ExprId -> 被调函数 FuncId
    /// 仅记录 callee 为本模块已知函数的调用点（外部函数不记录）
    pub call_sites: FxHashMap<ExprId, FuncId>,
    /// 方法 FuncId -> (decl_idx, method_idx)，用于从 module.declarations 定位方法体
    pub func_to_method_loc: FxHashMap<FuncId, (usize, usize)>,
    /// 方法 FuncId 集合（快速判定 FuncId 是否为方法）
    pub method_func_ids: FxHashSet<FuncId>,
}

impl CallGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加调用边 caller -> callee（去重）。
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

    /// 判定 FuncId 是否为方法（而非 FunDecl）。
    #[inline]
    pub fn is_method(&self, func: FuncId) -> bool {
        self.method_func_ids.contains(&func)
    }

    /// 通过 FuncId 获取函数/方法的元数据（统一入口，消除 FunDecl/Method 分散遍历）。
    /// FunDecl → FuncId = decl_idx；Method → 通过 func_to_method_loc 定位。
    pub fn get_func_meta<'a>(&self, func: FuncId, module: &'a Module) -> Option<FuncMetaRef<'a>> {
        if let Some(&(decl_idx, method_idx)) = self.func_to_method_loc.get(&func) {
            // 方法
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

    /// 遍历所有函数（FunDecl + Method），返回 (FuncId, FuncMetaRef)。
    /// 所有 pass 用此方法统一遍历，无需分别处理 FunDecl 和 TypeDecl.methods。
    pub fn iter_funcs<'a>(&'a self, module: &'a Module) -> impl Iterator<Item = (FuncId, FuncMetaRef<'a>)> + 'a {
        self.nodes.iter().filter_map(move |&fid| {
            self.get_func_meta(fid, module).map(|meta| (fid, meta))
        })
    }
}

/// 函数种类：FunDecl 或 Method。
#[derive(Debug, Clone, Copy)]
pub enum FuncKind<'a> {
    /// FunDecl，值为 declarations 索引
    Fun(usize),
    /// Method，值为 (type_name, method_idx)
    Method(&'a str, usize),
}

/// 函数元数据引用（统一 FunDecl 和 Method 的访问）。
#[derive(Debug, Clone, Copy)]
pub struct FuncMetaRef<'a> {
    pub name: &'a str,
    pub params: &'a [crate::ast::Ast::Param<'a>],
    pub body: crate::ast::Ast::ExprId,
    pub is_async: bool,
    pub visibility: crate::ast::Ast::Visibility,
    pub is_entry: bool,
    /// 方法的 self 类型名（FunDecl 为 None）
    pub self_type: Option<&'a str>,
    pub func_kind: FuncKind<'a>,
}

// =========================================================================
// PurityTable / EscapeTable — Layer 2 产出
// =========================================================================

/// 函数纯度。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purity {
    /// 纯函数：无副作用，结果仅依赖参数，可记忆化
    Pure,
    /// 非纯：有 I/O/并发/通信副作用
    Impure,
}

/// 纯度表：FuncId -> Purity。
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

/// 变量/分配逃逸信息。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeInfo {
    /// 不逃逸：仅在函数内使用，分配可消除（若未使用）
    NoEscape,
    /// 逃逸：带种类标记
    Escapes(EscapeKind),
}

/// 逃逸种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeKind {
    /// 堆分配逃逸（ArrayLit/RecordLit/RecordExtend）→ stack_alloc 优化
    Alloc,
    /// Lambda 逃逸（尾位置返回 / 循环体捕获）→ 独立 function_id 走 Cell 路径
    Lambda { loop_body_capture: bool },
}

/// 逃逸表：ExprId(分配点) -> EscapeInfo。
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
// SideEffect — 类型驱动的副作用判定
// =========================================================================

/// 表达式副作用分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SideEffect {
    /// 无副作用，可消除
    Pure,
    /// 有副作用，不可消除
    Impure,
    /// 分配但不逃逸，无外部可观察副作用
    AllocNoEscape,
}

/// 通过 sema.expr_types 解析方法调用的接收者类型，构造 mangled 名 `TypeName.method`。
///
/// 用于调用图边构建与副作用判定。返回 None 表示无法解析接收者类型。
fn resolve_method_mangled(recv: ExprId, method: &str, module_name: &str, sema: &SemaResult) -> Option<String> {
    let key = module_expr_key(module_name, recv.0 as u64);
    let info = sema.expr_types.get(&key)?;
    let type_name = info.type_name.as_deref()?;
    Some(format!("{}.{}", type_name, method))
}

/// 判定单个表达式的副作用。
///
/// 递归判定子表达式：仅当所有子表达式均为 Pure/AllocNoEscape 时才可能为 Pure。
/// 函数调用查 PurityTable；分配查 EscapeTable；字段访问查 Ty 可变性。
/// 方法调用通过 sema 解析实现函数，查其纯度。
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
        // ── 纯叶子 ──
        Expr::IntLit { .. }
        | Expr::FloatLit { .. }
        | Expr::BoolLit(_)
        | Expr::CharLit(_)
        | Expr::StrLit(_)
        | Expr::NullLit
        | Expr::VoidLit
        | Expr::Ident(_) => SideEffect::Pure,

        // ── 纯一元运算（递归判定操作数）──
        Expr::Unary { operand, .. } => classify_side_effect(
            *operand, arena, module_name, sema, purity, escape, func_name_to_id,
        ),
        Expr::RefOf(inner) | Expr::Deref(inner) | Expr::NonNullAssert(inner) => classify_side_effect(
            *inner, arena, module_name, sema, purity, escape, func_name_to_id,
        ),

        // ── 二元运算（递归判定两侧）──
        Expr::Binary { lhs, rhs, .. } => {
            let l = classify_side_effect(*lhs, arena, module_name, sema, purity, escape, func_name_to_id);
            let r = classify_side_effect(*rhs, arena, module_name, sema, purity, escape, func_name_to_id);
            combine(l, r)
        }

        // ── if 表达式：条件 + 分支均纯才纯 ──
        Expr::If { cond, then_branch, else_branch } => {
            let c = classify_side_effect(*cond, arena, module_name, sema, purity, escape, func_name_to_id);
            let t = classify_side_effect(*then_branch, arena, module_name, sema, purity, escape, func_name_to_id);
            let mut acc = combine(c, t);
            if let Some(e) = else_branch {
                acc = combine(acc, classify_side_effect(*e, arena, module_name, sema, purity, escape, func_name_to_id));
            }
            acc
        }

        // ── 块：所有语句 + trailing 均纯才纯 ──
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

        // ── 字段访问：接收者纯即视为纯 ──
        Expr::FieldAccess { recv, .. } | Expr::SafeAccess { recv, .. } => classify_side_effect(
            *recv, arena, module_name, sema, purity, escape, func_name_to_id,
        ),

        // ── 函数调用：查 PurityTable（含 sema 的 is_async/is_throwing 检查）──
        Expr::Call { callee, args, .. } => {
            let callee_purity = if let Expr::Ident(name) = &arena.expr(*callee).node {
                // sema 查 FuncSigInfo：async/throwing 函数一律视为非纯
                if let Some(sig) = sema.get_func_sig(*name) {
                    if sig.is_async || sig.is_throwing {
                        return SideEffect::Impure;
                    }
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

        // ── 方法调用：通过 sema 解析实现函数，查其纯度 ──
        Expr::MethodCall { recv, method, args, .. } | Expr::SafeMethodCall { recv, method, args, .. } => {
            // 通过 sema 解析接收者类型 → mangled 名 TypeName.method → 查纯度
            let mangled = resolve_method_mangled(*recv, *method, module_name, sema);
            let callee_purity = mangled.as_deref().and_then(|name| {
                // sema 查 MethodSigInfo：async/throwing 方法一律非纯
                if let Some(dot) = name.rfind('.') {
                    let type_name = &name[..dot];
                    if let Some(method_idx) = sema.lookup_method_idx(type_name, method) {
                        let &type_idx = sema.type_def_index.get(type_name)?;
                        let type_def = &sema.type_defs[type_idx as usize];
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

        // ── 分配（数组/记录字面量）：查 EscapeTable ──
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

        // ── Elvis：两侧均纯才纯 ──
        Expr::Elvis { lhs, rhs } => {
            let l = classify_side_effect(*lhs, arena, module_name, sema, purity, escape, func_name_to_id);
            let r = classify_side_effect(*rhs, arena, module_name, sema, purity, escape, func_name_to_id);
            combine(l, r)
        }

        // ── 一律视为有副作用 ──
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

/// 判定语句的副作用。
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

/// 合并两个副作用分类：任一 Impure 则 Impure；否则 AllocNoEscape 优先于 Pure。
fn combine(a: SideEffect, b: SideEffect) -> SideEffect {
    match (a, b) {
        (SideEffect::Impure, _) | (_, SideEffect::Impure) => SideEffect::Impure,
        (SideEffect::AllocNoEscape, _) | (_, SideEffect::AllocNoEscape) => SideEffect::AllocNoEscape,
        (SideEffect::Pure, SideEffect::Pure) => SideEffect::Pure,
    }
}

/// 判定表达式是否无副作用（Pure 或 AllocNoEscape 均视为可消除）。
pub fn is_side_effect_free(s: SideEffect) -> bool {
    s != SideEffect::Impure
}

// =========================================================================
// DefUseBuilder — Layer 1：构建 def-use 图
// =========================================================================

/// 构建 def-use 图。遍历每个函数体，收集 ValDecl/VarDecl/Assignment 定义点
/// 与 Ident 使用点。全局变量（顶层 VarDecl/ValDecl）的函数内赋值不注册局部定义点。
pub fn build_def_use(module: &Module, arena: &AstArena) -> DefUseGraph {
    let mut graph = DefUseGraph::new();
    // 收集全局变量名（顶层 ExprDecl 中嵌套的 VarDecl/ValDecl）
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
            // 参数作为入口活跃变量与定义点（参数默认不可变）
            let mut live = FxHashSet::default();
            for p in params {
                // 参数无对应语句，用 StmtId(u32::MAX) 占位，DeadVarPass 据此跳过
                let v = graph.add_def(p.name, StmtId(u32::MAX), func, false);
                live.insert(v);
            }
            graph.live_in.insert(func, live);
            collect_def_use_expr(*body, arena, func, &mut graph);
        }
    }
    graph
}

/// 递归收集表达式中的 def-use。
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
                // 注册 pattern 中的变量绑定（如 Some(x) => x 中的 x）
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
            // lambda 参数是独立作用域，不注册到当前函数的 def-use 图；
            // 但 lambda body 中引用外层变量需要记录为使用点（捕获）。
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

/// 递归收集语句中的 def-use。
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
            // 先收集 value 中的使用（可能读取 target 变量的旧值，如 x = x + 1）
            collect_def_use_expr(*value, arena, func, graph);
            // 再注册新定义点（覆盖 name_to_def，使后续使用映射到新定义）
            // 全局变量的赋值不注册局部定义点（全局变量不在函数 def-use 作用域内）
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
            // 复合赋值 x += v：先读取旧值（use），再收集 v，再注册新定义（def）
            // 全局变量的复合赋值不注册局部定义点
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
        Stmt::LocalDecl { .. } => {}
        Stmt::Break | Stmt::Continue => {}
    }
}

/// 递归收集 pattern 中的变量绑定，注册为定义点。
/// stmt 用 StmtId(u32::MAX) 占位（与参数相同），DeadVarPass 据此跳过这些。
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
// CallGraphBuilder — Layer 1：构建调用图 + 递归检测 + 入口标记
// =========================================================================

/// 内建非纯函数枚举：有 I/O/并发/通信副作用。
/// 替代字符串切片 IMPURE_BUILTINS，单一真相源为此枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImpureBuiltinFn {
    Async,
    Lazy,
    Select,
    Send,
    Recv,
}

impl ImpureBuiltinFn {
    /// 按函数名查枚举（消除字符串切片 contains 判定）。
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

/// 构建调用图。遍历所有函数（FunDecl + TypeDecl.methods），收集 Call/MethodCall 边，
/// 标记入口原因，检测递归。方法通过 mangled 名 "Type.method" 注册到 name_to_func。
pub fn build_call_graph(module: &Module, arena: &AstArena, sema: &SemaResult) -> CallGraph {
    let module_name = module.name;
    let mut cg = CallGraph::new();
    // 第一遍：收集所有函数名 -> FuncId
    // FunDecl: FuncId = declarations 索引
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
                    // 注册 mangled 名 "Type.method"（与 resolve_method_mangled 一致）
                    let mangled = format!("{}.{}", type_name, method.name);
                    cg.name_to_func.insert(mangled, fid);
                    method_global_idx += 1;
                }
            }
        }
    }
    // 第二遍：收集调用边 + 标记入口（clone nodes 避免借用冲突）
    let method_locs: Vec<(FuncId, usize, usize)> = cg.func_to_method_loc.iter()
        .map(|(&fid, &(d, m))| (fid, d, m))
        .collect();
    let nodes = cg.nodes.clone();
    for &fid in &nodes {
        // 判断是否方法并提取元数据：统一返回 (&str, &[Param], Option<ExprRef>, bool, Visibility, bool, &[Attribute], bool)
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

/// 标记函数的可达性入口原因。
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
    // 类型方法 / trait 方法：名称含 '.'（mangled 名 TypeName.method）
    if let Some(dot) = name.rfind('.') {
        let type_name = &name[..dot];
        let method_name = &name[dot + 1..];
        // 利用 witness_table 判定是否为 trait 方法实现：
        // 若该类型实现了某 trait 且该方法在 witness_table 的 method_slots 中，则为 TraitMethod
        if let Some(&type_idx) = sema.type_def_index.get(type_name) {
            let type_id = dynamic_type_id(type_idx);
            for entry in sema.witness_table.entries().iter() {
                if entry.type_id == type_id && entry.method_slots.contains_key(method_name) {
                    cg.entry_reasons.insert(func, ReachableReason::TraitMethod);
                    return;
                }
            }
        }
        // 否则为普通类型方法
        cg.entry_reasons.insert(func, ReachableReason::TypeMethod);
        return;
    }
    if visibility == Visibility::Public {
        cg.entry_reasons.insert(func, ReachableReason::Public);
        return;
    }
}

/// 递归收集函数体中的调用边。
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
                    // 记录调用点 ExprId → 被调函数，供内联展开使用
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
            // 通过 sema 解析方法 mangled 名 → 查 name_to_func → 添加调用边
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
            // 递归进入 lambda body：嵌套 lambda 中的调用归并到外层 caller
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

/// 递归收集语句中的调用边。
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
            // 递归进入嵌套函数 body：嵌套函数中的调用归并到外层 caller
            if let crate::ast::Ast::Decl::FunDecl { body, .. } = decl.as_ref() {
                collect_call_edges(*body, arena, caller, caller_name, module_name, sema, cg);
            }
        }
        Stmt::Break | Stmt::Continue => {}
    }
}

/// 检测直接递归与相互递归（Tarjan SCC）。
fn detect_recursion(cg: &mut CallGraph) {
    let mut sccs = tarjan_scc(cg);
    sccs.retain(|s| s.len() > 1);
    // 相互递归 SCC 中的所有函数也是递归函数，统一加入 recursive 集合。
    // 使 cg.recursive 成为"所有递归函数"的权威来源，inline_pass 等消费者
    // 只需检查 recursive 即可，无需分别检查 mutually_recursive。
    for scc in &sccs {
        for &func in scc {
            cg.recursive.insert(func);
        }
    }
    cg.mutually_recursive = sccs;
}

/// Tarjan 强连通分量算法。
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

/// 判断函数名是否为内建非纯函数。
pub fn is_impure_builtin(name: &str) -> bool {
    ImpureBuiltinFn::from_name(name).is_some()
}

// =========================================================================
// PurityAnalyzer — Layer 2：纯度不动点传播
// =========================================================================

/// 纯度分析。初始假定所有函数为纯，遍历函数体（FunDecl + Method 统一）找出直接非纯的函数
/// （调用内建非纯函数、方法调用、select、async/throwing 等），再沿逆向调用图传播 Impure。
pub fn analyze_purity(module: &Module, arena: &AstArena, cg: &CallGraph, sema: &SemaResult) -> PurityTable {
    let mut table = PurityTable::new();
    for &fid in &cg.nodes {
        table.put(fid, Purity::Pure);
    }
    let mut direct_impure: FxHashSet<FuncId> = FxHashSet::default();
    // 统一遍历 FunDecl + Method（通过 cg.iter_funcs）
    let func_metas: Vec<(FuncId, &str, crate::ast::Ast::ExprId, bool)> = cg.iter_funcs(module)
        .map(|(fid, meta)| (fid, meta.name, meta.body, meta.is_async))
        .collect();
    for (caller, name, body, is_async) in func_metas {
        // sema 查 FuncSigInfo：async/throwing 函数一律视为非纯
        if is_async {
            direct_impure.insert(caller);
            continue;
        }
        if let Some(sig) = sema.get_func_sig(name) {
            if sig.is_async || sig.is_throwing {
                direct_impure.insert(caller);
                continue;
            }
        }
        if is_direct_impure(body, arena, name, sema) {
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

/// 判定函数体是否直接非纯（包含内建非纯调用、方法调用、select、spawn 等）。
/// 通过 sema 查 FuncSigInfo：async/throwing 的外部函数（如 println）也判定为非纯。
fn is_direct_impure(body: ExprId, arena: &AstArena, self_name: &str, sema: &SemaResult) -> bool {
    fn check(expr_id: ExprId, arena: &AstArena, self_name: &str, sema: &SemaResult) -> bool {
        let expr = &arena.expr(expr_id).node;
        match expr {
            Expr::Call { callee, args, .. } => {
                if let Expr::Ident(name) = &arena.expr(*callee).node {
                    if is_impure_builtin(name) {
                        return true;
                    }
                    // sema 查 FuncSigInfo：async/throwing 函数（含 stdlib 外部函数）一律非纯
                    if let Some(sig) = sema.get_func_sig(name) {
                        if sig.is_async || sig.is_throwing {
                            return true;
                        }
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

// =========================================================================
// EscapeAnalyzer — Layer 2：逃逸分析
// =========================================================================

/// 逃逸分析。遍历每个函数体，对 ArrayLit/RecordLit/RecordExtend 分配点
/// 判定是否逃逸。
pub fn analyze_escape(
    module: &Module,
    arena: &AstArena,
    cg: &CallGraph,
    purity: &PurityTable,
) -> EscapeTable {
    let mut table = EscapeTable::new();
    // 统一遍历 FunDecl + Method
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
    // Lambda 逃逸分析（Bug #41 尾位置逃逸 + Bug #40 循环体捕获）
    analyze_lambda_escape(module, arena, &mut table);
    table
}

/// 标记所有分配点为 NoEscape（初始值）。
fn mark_allocations(expr_id: ExprId, arena: &AstArena, table: &mut EscapeTable) {
    let expr = &arena.expr(expr_id).node;
    if matches!(expr, Expr::ArrayLit { .. } | Expr::RecordLit(_) | Expr::RecordExtend { .. }) {
        table.put(expr_id, EscapeInfo::NoEscape);
    }
    walk_children_expr(expr_id, arena, |c| mark_allocations(c, arena, table));
    walk_children_stmts_of_expr(expr_id, arena, |s| mark_allocations_stmt(s, arena, table));
}

/// 遍历语句中的分配点。
fn mark_allocations_stmt(stmt_id: StmtId, arena: &AstArena, table: &mut EscapeTable) {
    walk_children_stmt(stmt_id, arena, |e| mark_allocations(e, arena, table));
}

/// 扫描逃逸点。
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

/// 扫描语句中的逃逸点。
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

/// 收集表达式及其子表达式中所有分配点。
fn collect_all_allocs(expr_id: ExprId, arena: &AstArena, escaping: &mut FxHashSet<ExprId>) {
    let expr = &arena.expr(expr_id).node;
    if matches!(expr, Expr::ArrayLit { .. } | Expr::RecordLit(_) | Expr::RecordExtend { .. }) {
        escaping.insert(expr_id);
    }
    walk_children_expr(expr_id, arena, |c| collect_all_allocs(c, arena, escaping));
    walk_children_stmts_of_expr(expr_id, arena, |s| collect_all_allocs_stmt(s, arena, escaping));
}

/// 遍历语句中的分配点。
fn collect_all_allocs_stmt(stmt_id: StmtId, arena: &AstArena, escaping: &mut FxHashSet<ExprId>) {
    walk_children_stmt(stmt_id, arena, |e| collect_all_allocs(e, arena, escaping));
}

/// 遍历表达式的直接子表达式。
fn walk_children_expr<F: FnMut(ExprId)>(expr_id: ExprId, arena: &AstArena, mut f: F) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit(_)
        | Expr::CharLit(_) | Expr::StrLit(_) | Expr::NullLit | Expr::VoidLit
        | Expr::Ident(_) => {}
        Expr::Unary { operand, .. } | Expr::RefOf(operand) | Expr::Deref(operand)
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

/// 遍历表达式中嵌入的语句（仅 Block）。
fn walk_children_stmts_of_expr<F: FnMut(StmtId)>(expr_id: ExprId, arena: &AstArena, mut f: F) {
    if let Expr::Block { stmts, .. } = &arena.expr(expr_id).node {
        for s in stmts { f(*s); }
    }
}

/// 遍历语句的子表达式（递归到表达式）。
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
// LambdaEscape — Lambda 逃逸分析（Bug #41 尾位置逃逸 + Bug #40 循环体捕获）
// =========================================================================

/// Lambda 逃逸分析统一入口。
///
/// 对每个 FunDecl 的 body 做两遍分析：
/// 1. 尾位置逃逸：调用 find_escaping_lambdas，标记为
///    `EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: false })`
/// 2. 循环体捕获逃逸：扫描 body 中的 Lambda，检查是否捕获了循环体局部变量，
///    标记为 `EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: true })`
fn analyze_lambda_escape(
    module: &Module,
    arena: &AstArena,
    table: &mut EscapeTable,
) {
    for decl in &module.declarations {
        if let Decl::FunDecl { body, .. } = &decl.node {
            // 递归分析函数 body 和所有嵌套 lambda body 的逃逸
            analyze_lambda_escape_recursive(*body, arena, table);
        }
    }
}

/// 对当前 body 做尾位置逃逸分析，然后递归进入所有嵌套 Lambda body。
///
/// IR 的 escape_context_stack 是栈式的：编译每个 lambda 时扫描其 body
/// 找出逃逸的嵌套 lambda。analyzer 需要对每个 lambda body 递归做同样分析。
fn analyze_lambda_escape_recursive(
    expr_id: ExprId,
    arena: &AstArena,
    table: &mut EscapeTable,
) {
    // Pass 1: 尾位置逃逸（当前 body 的尾位置 lambda）
    let tail_escaping = find_escaping_lambdas(expr_id, arena);
    for lambda_id in tail_escaping {
        table.put(
            lambda_id,
            EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: false }),
        );
    }
    // 递归进入所有嵌套 Lambda body，对其做同样的尾位置逃逸分析
    walk_lambdas_in_expr(expr_id, arena, &mut |lambda_body| {
        analyze_lambda_escape_recursive(lambda_body, arena, table);
    });
    // Pass 2: 循环体捕获逃逸
    let mut loop_body_vars_stack: Vec<FxHashSet<String>> = Vec::new();
    scan_lambda_escapes_in_expr(expr_id, arena, &mut loop_body_vars_stack, table);
}

/// 遍历表达式中的所有 Lambda，对每个 Lambda 的 body 调用回调。
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
            // 继续递归进入 lambda body 内部（可能有更深嵌套）
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

/// walk_lambdas_in_expr 的 stmt 版本。
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
            // 局部函数声明：递归进入函数 body 做逃逸分析
            if let Decl::FunDecl { body, .. } = &**decl {
                f(*body);
                walk_lambdas_in_expr(*body, arena, f);
            }
        }
        _ => {}
    }
}

/// 收集 ValDecl/VarDecl 中持有 Lambda 的变量名 → ExprId。
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

/// 辅助：从 Stmt 中收集 lambda 变量。
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
            // 递归扫描 value（lambda body 内可能也有 lambda 变量）
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
        // Break/Continue/LocalDecl 不含 lambda 变量
        _ => {}
    }
}

/// 递归收集尾位置的 Lambda ExprId（包括持有 Lambda 的 Ident）。
///
/// 尾位置 = 表达式的值会被作为 enclosing lambda 的返回值。
/// - body 本身在尾位置
/// - Block trailing 在尾位置
/// - Return 语句值在尾位置
/// - If 分支在尾位置（当 If 本身在尾位置时）
/// - Match arm body 在尾位置（当 Match 本身在尾位置时）
/// - Elvis rhs 在尾位置（当 Elvis 本身在尾位置时）
fn collect_tail_lambdas(
    expr_id: ExprId,
    arena: &AstArena,
    lambda_vars: &FxHashMap<String, ExprId>,
    out: &mut FxHashSet<ExprId>,
) {
    let node = &arena.expr(expr_id).node;
    match node {
        Expr::Lambda { .. } => {
            // Lambda 在尾位置 → 逃逸
            out.insert(expr_id);
        }
        Expr::Ident(name) => {
            // Ident 在尾位置，若持有 Lambda → 该 Lambda 逃逸
            if let Some(&lambda_id) = lambda_vars.get(*name) {
                out.insert(lambda_id);
            }
        }
        Expr::Block { stmts, trailing } => {
            // Return 语句值在尾位置
            for &stmt_id in stmts {
                if let Stmt::Return { value: Some(ret_expr) } = &arena.stmt(stmt_id).node {
                    collect_tail_lambdas(*ret_expr, arena, lambda_vars, out);
                }
            }
            // trailing 在尾位置
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
            // Elvis rhs 在尾位置（当 lhs 为 null 时 rhs 是返回值）
            collect_tail_lambdas(*rhs, arena, lambda_vars, out);
        }
        _ => {
            // 其他表达式不在尾位置，其子表达式也不在尾位置
        }
    }
}

/// 两遍扫描入口：收集尾位置逃逸的 Lambda。
///
/// Pass 1: 收集所有 ValDecl/VarDecl 中持有 Lambda 的变量 (name → lambda ExprId)
/// Pass 2: 递归收集尾位置的 Lambda（包括持有 Lambda 的 Ident）
fn find_escaping_lambdas(body: ExprId, arena: &AstArena) -> FxHashSet<ExprId> {
    let mut escaping: FxHashSet<ExprId> = FxHashSet::default();
    let mut lambda_vars: FxHashMap<String, ExprId> = FxHashMap::default();
    collect_lambda_vars(body, arena, &mut lambda_vars);
    collect_tail_lambdas(body, arena, &lambda_vars, &mut escaping);
    escaping
}

/// 递归收集表达式中的所有 Ident 名称（去重，保留首次出现顺序）。
///
/// 简化版自由变量分析：遍历常见 Expr 变体收集标识符引用，
/// 由调用方排除 lambda 参数并检查外层作用域绑定。
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
        // 单 operand 表达式：RefOf/Deref/Propagate/NonNullAssert/Atomic/Lazy
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
        // 常量/无子表达式变体：IntLit/FloatLit/BoolLit/CharLit/StrLit/NullLit/VoidLit
        _ => {}
    }
}

/// 递归收集语句中的 Ident 名称（collect_free_idents_expr 的语句版本）。
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

/// 扫描表达式中的 Lambda，检测循环体捕获逃逸。
///
/// 维护 `loop_body_vars_stack`（循环体局部变量名栈），遇到 Lambda 时：
/// 1. 收集 lambda 参数名（排除自身参数）
/// 2. 用 collect_free_idents_expr 收集 lambda body 中的所有标识符
/// 3. 排除 lambda 自身参数名后，剩下的就是自由变量
/// 4. 检查自由变量是否有在 loop_body_vars_stack 的任一层中 → 循环体捕获逃逸
fn scan_lambda_escapes_in_expr(
    expr_id: ExprId,
    arena: &AstArena,
    loop_body_vars_stack: &mut Vec<FxHashSet<String>>,
    table: &mut EscapeTable,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        Expr::Lambda { params, body, .. } => {
            // a. 收集 lambda 参数名（排除自身参数）
            let param_names: FxHashSet<String> = params.iter().map(|p| p.name.to_string()).collect();
            // b. 收集 lambda body 中的所有标识符
            let body_expr = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            let mut idents = Vec::new();
            collect_free_idents_expr(body_expr, arena, &mut idents);
            // c. 排除 lambda 自身参数名 → 自由变量
            // d. 检查自由变量是否在循环体局部变量栈中
            let captures_loop_var = idents.iter().any(|n| {
                !param_names.contains(n) && loop_body_vars_stack.iter().any(|layer| layer.contains(n))
            });
            if captures_loop_var {
                table.put(
                    expr_id,
                    EscapeInfo::Escapes(EscapeKind::Lambda { loop_body_capture: true }),
                );
            }
            // 继续递归扫描 lambda body 内部的嵌套 lambda / 循环
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

/// 扫描语句中的 Lambda，检测循环体捕获逃逸。
///
/// 进入 For/While/Loop body 时，收集 body 内所有 ValDecl/VarDecl 定义的变量名，
/// push 到循环体局部变量栈；退出时 pop。
fn scan_lambda_escapes_in_stmt(
    stmt_id: StmtId,
    arena: &AstArena,
    loop_body_vars_stack: &mut Vec<FxHashSet<String>>,
    table: &mut EscapeTable,
) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::For { iterable, body, .. } => {
            // 先扫描 iterable（不在循环体内）
            scan_lambda_escapes_in_expr(*iterable, arena, loop_body_vars_stack, table);
            // 收集循环体局部变量，push 到栈
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
            // 局部函数声明：独立作用域，用全新的 loop_body_vars_stack 扫描
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

/// 收集循环体内所有 ValDecl/VarDecl 定义的变量名（不进入嵌套 lambda/函数作用域）。
fn collect_loop_body_vars_expr(expr_id: ExprId, arena: &AstArena, vars: &mut FxHashSet<String>) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        // 不进入嵌套 lambda 的内部作用域（lambda 有自己的参数和局部变量）
        Expr::Lambda { .. } => {}
        _ => {
            walk_children_expr(expr_id, arena, |c| collect_loop_body_vars_expr(c, arena, vars));
            walk_children_stmts_of_expr(expr_id, arena, |s| collect_loop_body_vars_stmt(s, arena, vars));
        }
    }
}

/// collect_loop_body_vars_expr 的语句版本。
fn collect_loop_body_vars_stmt(stmt_id: StmtId, arena: &AstArena, vars: &mut FxHashSet<String>) {
    let stmt = &arena.stmt(stmt_id).node;
    match stmt {
        Stmt::ValDecl { name, value, .. } | Stmt::VarDecl { name, value, .. } => {
            vars.insert(name.to_string());
            collect_loop_body_vars_expr(*value, arena, vars);
        }
        // 不进入嵌套函数的内部作用域
        Stmt::LocalDecl { .. } => {}
        _ => {
            walk_children_stmt(stmt_id, arena, |e| collect_loop_body_vars_expr(e, arena, vars));
        }
    }
}

// =========================================================================
// MemoPlan — 记忆化策略（Layer 3 共享结构）
// =========================================================================

/// 记忆化候选。
#[derive(Debug, Clone)]
pub struct MemoCandidate {
    pub func: FuncId,
    pub strategy: MemoStrategy,
}

/// 尾递归参数变换信息：从函数体 AST 提取的 base case + 递归分支。
/// Builder 层消费此信息构造 while_sg IR。
#[derive(Debug, Clone, Default)]
pub struct TailRecInfo {
    /// 非递归终止分支：(条件表达式, 返回值表达式)
    /// 条件为 None 表示 else 兜底分支（无条件终止）。
    pub base_cases: Vec<(Option<ExprId>, ExprId)>,
    /// 递归分支：(条件表达式, 实参列表)
    /// 条件为 None 表示 else 兜底分支（无条件递归）。
    pub rec_branches: Vec<(Option<ExprId>, Vec<ExprId>)>,
}

impl TailRecInfo {
    /// 是否有效：至少一个 base case 和一个 rec branch
    pub fn is_valid(&self) -> bool {
        !self.base_cases.is_empty() && !self.rec_branches.is_empty()
    }
}

/// 非尾递归转迭代信息：将非尾递归函数变换为"工作栈 + while 循环 + 状态机"IR。
///
/// 核心思路：函数体中的每个非尾自调用拆分为"push 续延 + push 子任务"，
/// 调用返回后通过 state 号分派到对应续延，用 result 变量替换调用结果。
///
/// 例如 fib(n) = if n < 2 { n } else { fib(n-1) + fib(n-2) } 变换为：
/// - state 0 (INIT): if n < 2 { result = n } else { push cont(1); push task(n-1); continue }
/// - state 1 (AFTER fib(n-1)): left = result; push cont(2, left); push task(n-2); continue
/// - state 2 (AFTER fib(n-2)): result = saved + result
#[derive(Debug, Clone)]
pub struct NonTailRecInfo {
    /// 所有非尾自调用的 ExprId（按 AST 遍历顺序）。
    /// Builder 用此列表分配 state 号：state 0 = INIT，state N = 第 N 个调用返回后。
    pub call_sites: Vec<ExprId>,
    /// 包含所有 call_sites 的续延表达式 ExprId。
    /// Builder 对每个 state 重新编译此表达式，用 call_result_map 替换已完成的调用。
    pub continuation_expr: ExprId,
    /// base case：(条件, 返回值)。条件为 None 表示 else 兜底。
    pub base_cases: Vec<(Option<ExprId>, ExprId)>,
    /// 函数参数数量（用于构造栈帧）。
    pub param_count: usize,
}

impl NonTailRecInfo {
    pub fn is_valid(&self) -> bool {
        !self.call_sites.is_empty() && !self.base_cases.is_empty()
    }
}

/// 记忆化策略。
#[derive(Debug, Clone)]
pub enum MemoStrategy {
    /// 尾递归转循环，不缓存
    TailRecToLoop { info: TailRecInfo },
    /// 非尾递归转迭代（工作栈模拟），不缓存
    NonTailRecToLoop { info: NonTailRecInfo },
    /// 记忆化缓存
    Memoize { cache_key: CacheKeySpec, capacity: MemoCapacity },
    /// 循环不变量外提
    LoopInvariantHoist { invariants: Vec<ExprId> },
}

/// 缓存键规格：参与缓存键的参数下标。
#[derive(Debug, Clone)]
pub struct CacheKeySpec {
    pub param_indices: Vec<u32>,
}

/// 缓存容量策略。
#[derive(Debug, Clone)]
pub enum MemoCapacity {
    Unlimited,
    LRU(usize),
}

/// 记忆化计划。
#[derive(Debug, Default)]
pub struct MemoPlan {
    pub candidates: Vec<MemoCandidate>,
}

// =========================================================================
// 不可达代码检测 + 常量条件死分支消除
// =========================================================================

/// 语句是否为控制流终结符（之后的语句不可达）。
fn is_terminator_stmt(stmt_id: StmtId, arena: &AstArena) -> bool {
    matches!(
        &arena.stmt(stmt_id).node,
        Stmt::Return { .. } | Stmt::Break | Stmt::Continue | Stmt::Throw { .. }
    )
}

/// 递归标记块中终结符之后的所有语句为死（不可达代码）。
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

/// 递归标记表达式中的所有语句为死（用于死分支整体消除）。
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

/// 求值编译时常量布尔条件。优先查 sema ExprInfo.const_val，回退到 BoolLit。
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
// DeadCodePass — Layer 3：死代码消除
// =========================================================================

/// 死代码报告。
#[derive(Debug, Default)]
pub struct DeadCodeReport {
    /// 可安全消除的语句
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

/// 死代码分析遍。逐函数不动点迭代：收集有效读取，标记未读且无副作用的声明。
/// 预处理阶段标记不可达代码和常量条件死分支。
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
    // 统一遍历 FunDecl + Method
    let func_metas: Vec<(FuncId, crate::ast::Ast::ExprId)> = cg.iter_funcs(module)
        .map(|(fid, meta)| (fid, meta.body))
        .collect();
    for (func, body) in func_metas {
        // 预处理：不可达代码（return/break/continue/throw 之后）
        mark_unreachable(body, arena, &mut report);
        // 不动点迭代：死声明 + 常量条件死分支 + 死存储
        analyze_function_dce(body, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, &mut report);
    }
    report
}

/// 对函数体执行不动点迭代。
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

/// 递归收集表达式中的有效读取（跳过已标记死声明的 init）。
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
        // ── 闭包：遍历 body 收集捕获的外层变量读取 ──
        // walk_children_expr 不遍历 Lambda，需专门处理，否则闭包捕获的变量
        // 不被标记为"已读取"，导致其声明被误判为死代码。
        Expr::Lambda { body, .. } => {
            let body_expr = match body {
                LambdaBody::Block(e) | LambdaBody::Expression(e) => *e,
            };
            collect_reads_expr(body_expr, arena, report, reads);
        }
        _ => walk_children_expr(expr_id, arena, |c| collect_reads_expr(c, arena, report, reads)),
    }
}

/// 递归收集语句中的有效读取。
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
        _ => walk_children_stmt(stmt_id, arena, |e| collect_reads_expr(e, arena, report, reads)),
    }
}

/// 遍历表达式，标记未读且无副作用的声明为死。同时处理常量条件死分支。
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
        // ── 常量条件死分支：if true → else 死，if false → then 死 ──
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

/// 遍历语句，标记未读且无副作用的声明为死。同时处理死存储（赋值后未读即被覆盖）。
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
        // ── 死存储：赋值目标在函数内完全不被读取且赋值表达式无副作用 ──
        // 注意：只用 reads 判定（而非 never_read），因为可变变量可能被闭包间接读取，
        // 赋值创建的新定义点在 def-use 图中无使用点，但闭包调用时会读取最新值。
        Stmt::Assignment { target, value } => {
            if let Expr::Ident(name) = &arena.expr(*target).node {
                if !reads.contains(*name) {
                    let se = classify_side_effect(*value, arena, module_name, sema, purity, escape, func_name_to_id);
                    if is_side_effect_free(se) {
                        report.dead_stmts.insert(stmt_id);
                    }
                }
            }
            mark_dead_decls_expr(*value, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
        }
        Stmt::CompoundAssignment { target, value, .. } => {
            // 复合赋值 x += v：若 x 在函数内完全不被读取且 v 无副作用，则整体为死存储
            if let Expr::Ident(name) = &arena.expr(*target).node {
                if !reads.contains(*name) {
                    let se = classify_side_effect(*value, arena, module_name, sema, purity, escape, func_name_to_id);
                    if is_side_effect_free(se) {
                        report.dead_stmts.insert(stmt_id);
                    }
                }
            }
            mark_dead_decls_expr(*value, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report);
        }
        _ => walk_children_stmt(stmt_id, arena, |e| {
            mark_dead_decls_expr(e, arena, module_name, sema, purity, escape, func_name_to_id, func, def_use, reads, report)
        }),
    }
}

// =========================================================================
// DeadVarPass — Layer 3：死变量消除
// =========================================================================

/// 死变量报告。
#[derive(Debug, Default)]
pub struct DeadVarReport {
    /// 可消除的变量定义点
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

/// 死变量分析。基于 DefUseGraph：
/// - 从未被读取的变量定义点为死
/// - 死代码的声明语句对应的变量也标记为死
pub fn dead_var_pass(
    _module: &Module,
    _arena: &AstArena,
    def_use: &DefUseGraph,
    dead_code: &DeadCodeReport,
) -> DeadVarReport {
    let mut report = DeadVarReport::new();
    for (i, def) in def_use.defs.iter().enumerate() {
        let vid = VarId(i as u32);
        // 参数定义点（StmtId(u32::MAX)）跳过
        if def.stmt.0 == u32::MAX {
            continue;
        }
        // 死代码的声明语句对应的变量
        if dead_code.is_dead(def.stmt) {
            report.dead_vars.insert(vid);
            continue;
        }
        // 从未被读取
        // 注意：闭包捕获的可变变量，赋值创建的新定义点无使用点，
        // 但同名变量在旧定义点有使用点（闭包读取），不应判为死。
        if def_use.is_never_read(vid) && !def_use.is_name_ever_read(def.func, &def.name) {
            report.dead_vars.insert(vid);
        }
    }
    report
}

// =========================================================================
// DeadFuncPass — Layer 3：死函数消除
// =========================================================================

/// 死函数报告。
#[derive(Debug, Default)]
pub struct DeadFuncReport {
    /// 可消除函数
    pub dead: FxHashSet<FuncId>,
    /// 保留原因
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

/// 死函数分析。从所有入口出发沿调用图做可达性传播。
pub fn dead_func_pass(cg: &CallGraph, memo: &MemoPlan) -> DeadFuncReport {
    let mut report = DeadFuncReport::new();
    let mut reachable: FxHashSet<FuncId> = FxHashSet::default();
    for (&fid, reason) in &cg.entry_reasons {
        report.reachable_reasons.insert(fid, reason.clone());
        reachable.insert(fid);
    }
    // 记忆化候选及其调用的纯函数也保留
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
    // 工作列表：从入口沿调用图传播可达
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
    // 不可达的函数为死
    for &fid in &cg.nodes {
        if !reachable.contains(&fid) {
            report.dead.insert(fid);
        }
    }
    report
}

// =========================================================================
// MemoPass — Layer 3：记忆化策略决策
// =========================================================================

/// 记忆化分析。决策策略（通用判定，无特例分支）：
/// - 纯函数 + 递归（自/相互）：
///   - 尾递归且 info 有效 → TailRecToLoop
///   - 非尾递归且单调用点且无 defer 且 info 有效 → NonTailRecToLoop
///   - 其他递归情况 → Memoize（缓存全部参数）
/// - 纯函数 + 含循环 → LoopInvariantHoist
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
    // 统一遍历 FunDecl + Method（通过 cg.iter_funcs）
    let func_metas: Vec<(FuncId, &str, &[crate::ast::Ast::Param], crate::ast::Ast::ExprId)> =
        cg.iter_funcs(module)
            .map(|(fid, meta)| (fid, meta.name, meta.params, meta.body))
            .collect();
    for (func, name, params, body_expr) in func_metas {
        if !purity.is_pure(func) {
            continue;
        }
        // 递归函数（自递归 + 相互递归统一处理）
        if cg.recursive.contains(&func) {
            if is_tail_recursive(body_expr, arena, name) {
                // TailRecToLoop 统一处理 if-else 和 match 尾递归：
                // - if-else: cond = NOT(base_case_cond)，Gate 分派 base/rec
                // - match: cond = Const(true)，body_sg 内部 match Gate 分派，
                //   rec arm 的 WriteBack 设置 Continue → 循环继续，
                //   base arm 无信号 → 循环退出（返回 body_sg 返回值）
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
                // NonTailRecToLoop 仅在：无 defer + info 有效 + 单调用点（无重复子问题）时适用。
                // 其余情况（defer / info 无效 / 2+ 调用点有重复子问题）一律走 Memoize。
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
                // 相互递归（无自调用）→ Memoize
                plan.candidates.push(memoize_all_params(func, params));
            }
            continue;
        }
        // 相互递归纯函数 SCC：记忆化
        if cg.mutually_recursive.iter().any(|scc| scc.contains(&func)) {
            plan.candidates.push(memoize_all_params(func, params));
            continue;
        }
        // 含循环：收集不变量
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

/// 构造 Memoize 候选：缓存全部参数（通用 helper，消除重复构造）。
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

/// 判定函数体是否为尾递归：函数体中至少有一条路径的尾位置是对自身的调用。
/// 支持 if-else 和 Match 尾递归，且所有自调用必须在尾位置。
/// ack(m-1, ack(m, n-1)) 的内层 ack 是非尾位置自调用 → 拒绝。
fn is_tail_recursive(body: ExprId, arena: &AstArena, self_name: &str) -> bool {
    has_tail_call(body, arena, self_name) && !has_non_tail_self_call(body, arena, self_name)
}

/// 检查表达式的尾位置是否存在对 self_name 的调用。
/// 递归 if-else、Match arm body 和 block trailing。
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
            // 每个 arm 的 body 都是尾位置
            arms.iter().any(|arm| has_tail_call(arm.body, arena, self_name))
        }
        _ => false,
    }
}

/// 检查函数体是否存在非尾位置的自调用（如 ack(m-1, ack(m, n-1)) 的内层 ack）。
/// 非尾位置 = 作为参数、操作数、字段值等。
/// 若存在此类调用，函数不是纯尾递归，不能安全转迭代。
fn has_non_tail_self_call(body: ExprId, arena: &AstArena, self_name: &str) -> bool {
    fn is_self_call(expr_id: ExprId, arena: &AstArena, self_name: &str) -> bool {
        if let Expr::Call { callee, .. } = &arena.expr(expr_id).node {
            if let Expr::Ident(name) = &arena.expr(*callee).node {
                return *name == self_name;
            }
        }
        false
    }
    /// 递归检查子表达式中是否存在非尾位置自调用。
    /// `in_tail` 表示当前表达式是否在尾位置。
    fn check(expr_id: ExprId, arena: &AstArena, self_name: &str, in_tail: bool) -> bool {
        let expr = &arena.expr(expr_id).node;
        match expr {
            Expr::Call { callee, args, .. } => {
                let is_self = is_self_call(expr_id, arena, self_name);
                if is_self && !in_tail {
                    // 非尾位置的自调用 → 拒绝
                    return true;
                }
                if is_self && in_tail {
                    // 尾位置的自调用：检查参数中是否有非尾自调用
                    return args.iter().any(|&a| check(a, arena, self_name, false));
                }
                // 非自调用：callee 和 args 都是非尾位置
                check(*callee, arena, self_name, false)
                    || args.iter().any(|&a| check(a, arena, self_name, false))
            }
            Expr::Block { stmts, trailing } => {
                // stmts 中的表达式都不是尾位置
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
                // Match 本身不阻止，但 arm body 中的非尾自调用会被检测
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

/// 获取语句中的表达式（用于非尾位置自调用检查）。
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

/// 从尾递归函数体提取参数变换信息。
///
/// 遍历函数体的控制流分支，分类为 base case（非递归终止）和 rec branch（递归调用）。
/// 支持的 AST 结构：
/// - if cond { return base } else { return self(args) }
/// - if cond1 { ... } else if cond2 { return self(args2) } else { return base }
/// - match scrut { arm1 => return base, arm2 => return self(args) }
/// - block { stmts; trailing_if_or_match }
///
/// 每个 base case 记录 (条件, 返回值)；每个 rec branch 记录 (条件, 实参列表)。
/// 条件为 None 表示 else/match wildcard 兜底分支。
fn extract_tail_rec_info(
    body: ExprId,
    arena: &AstArena,
    self_name: &str,
) -> TailRecInfo {
    let mut info = TailRecInfo::default();
    collect_tail_branches(body, arena, self_name, None, &mut info);
    info
}

/// 递归收集控制流分支的 base case 和 rec branch。
/// `cond` 是当前分支的继承条件（None 表示兜底/无条件）。
fn collect_tail_branches(
    expr_id: ExprId,
    arena: &AstArena,
    self_name: &str,
    cond: Option<ExprId>,
    info: &mut TailRecInfo,
) {
    let expr = &arena.expr(expr_id).node;
    match expr {
        // block：优先递归 trailing；trailing 为 None 时检查末尾 Return
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
        // if：then 分支用 Some(cond)，else 分支用 None（兜底）
        Expr::If { cond: if_cond, then_branch, else_branch, .. } => {
            collect_tail_branches(*then_branch, arena, self_name, Some(*if_cond), info);
            if let Some(eb) = else_branch {
                collect_tail_branches(*eb, arena, self_name, None, info);
            }
        }
        // match：每个 arm 单独分派（模式条件无法用 ExprId 表达，用 None）
        Expr::Match { arms, .. } => {
            for arm in arms {
                collect_tail_branches(arm.body, arena, self_name, None, info);
            }
        }
        // 尾调用：rec branch
        Expr::Call { callee, args, .. } => {
            if let Expr::Ident(name) = &arena.expr(*callee).node {
                if *name == self_name {
                    info.rec_branches.push((cond, args.clone()));
                    return;
                }
            }
            info.base_cases.push((cond, expr_id));
        }
        // 非尾调用表达式：base case
        _ => {
            info.base_cases.push((cond, expr_id));
        }
    }
}

// =========================================================================
// 非尾递归转迭代：调用点提取 + 续延分析
// =========================================================================

/// 从非尾递归函数体提取调用点 + 续延信息。
///
/// 遍历函数体 AST，收集所有非尾位置的自调用 ExprId。
/// continuation_expr = body（函数体本身），因为纯函数的条件重评估结果不变，
/// Builder 对每个 state 重新编译 body，用 call_result_map 替换已完成的调用。
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

/// 递归收集非尾位置的自调用 ExprId + base case。
///
/// `in_tail` 表示当前表达式是否在尾位置。
/// - 尾位置的自调用是尾递归（Tier A 处理），不收集
/// - 非尾位置的自调用收集到 call_sites
/// - 非自调用的尾位置表达式收集为 base case
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
                    // 尾位置自调用：尾递归，不触发 Tier B
                    base_cases.push((None, expr_id));
                } else {
                    // 非尾位置自调用：收集为 call_site
                    call_sites.push(expr_id);
                    // 检查参数中是否有更多自调用（如 ack(m-1, ack(m, n-1)) 的内层）
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

/// 判定是否为叶子表达式（无子表达式）。
fn is_leaf(expr_id: ExprId, arena: &AstArena) -> bool {
    matches!(
        &arena.expr(expr_id).node,
        Expr::IntLit { .. } | Expr::FloatLit { .. } | Expr::BoolLit(_)
        | Expr::CharLit(_) | Expr::StrLit(_) | Expr::NullLit | Expr::VoidLit
        | Expr::Ident(_)
    )
}

/// 收集循环不变量：循环内引用、仅依赖循环外定义且循环内未修改变量的纯表达式。
/// 候选包括循环条件（While）、迭代源（For）及循环体内的纯表达式。
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

/// 递归查找循环（While/Loop/For 语句），收集不变量候选。
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
    // 遍历 Block 中的语句查找循环
    walk_children_stmts_of_expr(expr_id, arena, |s| {
        collect_loop_invariants_stmt(s, arena, module_name, sema, purity, escape, func_name_to_id, invariants);
    });
    // 遍历子表达式查找嵌套 Block
    walk_children_expr(expr_id, arena, |c| {
        collect_loop_invariants_expr(c, arena, module_name, sema, purity, escape, func_name_to_id, invariants)
    });
}

/// 遍历语句查找循环。
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

/// 收集循环体内被修改的变量名集合。
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

/// 遍历循环体，收集所有满足不变量条件的纯表达式。
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

/// 判定表达式是否为循环不变量候选：非叶子、纯、引用的变量均未被循环修改。
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

/// 收集表达式中引用的所有标识符名。
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
// DeadParamPass — 未使用参数检测
// =========================================================================

/// 未使用参数报告。
#[derive(Debug, Default)]
pub struct DeadParamReport {
    /// (FuncId, 参数名) 列表：从未被函数体读取的参数
    pub dead_params: Vec<(FuncId, String)>,
}

/// 检测从未被函数体读取的参数。
/// 参数在 DefUseGraph 中以 StmtId(u32::MAX) 注册，is_never_read 判定。
pub fn dead_param_pass(module: &Module, def_use: &DefUseGraph, cg: &CallGraph) -> DeadParamReport {
    let mut report = DeadParamReport::default();
    // 统一遍历 FunDecl + Method
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
// InlinePass — 内联候选分析
// =========================================================================

/// 内联阈值：函数体 AST 节点数 <= 此值且为纯函数时建议内联。
const INLINE_SIZE_THRESHOLD: usize = 15;

/// 内联候选报告。
#[derive(Debug, Default)]
pub struct InlineReport {
    /// 建议内联的函数列表：(FuncId, 函数体大小)
    pub candidates: Vec<(FuncId, usize)>,
    /// 调用点 ExprId → 被调函数 FuncId
    /// IrBuilder 编译 Call 时查此表，命中则内联展开 callee body 而非 launch 子图
    pub expansions: FxHashMap<ExprId, FuncId>,
}

/// 内联候选分析：小纯函数 + 非递归 + 非 async/throwing。
/// 产出内联建议供 IR 层决策。
pub fn inline_pass(
    module: &Module,
    arena: &AstArena,
    cg: &CallGraph,
    purity: &PurityTable,
    sema: &SemaResult,
) -> InlineReport {
    let mut report = InlineReport::default();
    // 第一遍：收集可内联函数集合（统一遍历 FunDecl + Method）
    let mut inlineable: FxHashSet<FuncId> = FxHashSet::default();
    let func_metas: Vec<(FuncId, &str, crate::ast::Ast::ExprId, bool)> = cg.iter_funcs(module)
        .map(|(fid, meta)| (fid, meta.name, meta.body, meta.is_async))
        .collect();
    for (func, name, body, is_async) in func_metas {
        // 非纯函数不内联（可能有副作用依赖）
        if !purity.is_pure(func) {
            continue;
        }
        // 递归函数不内联（会无限展开）
        if cg.recursive.contains(&func) {
            continue;
        }
        // async/throwing 函数不内联
        if is_async {
            continue;
        }
        if let Some(sig) = sema.get_func_sig(name) {
            if sig.is_async || sig.is_throwing {
                continue;
            }
        }
        // 入口函数（Entry/ExternC/ExternAttr）不内联
        if let Some(reason) = cg.entry_reasons.get(&func) {
            if reason.is_definite() {
                continue;
            }
        }
        // 包含嵌套函数（Lambda/LocalDecl）的函数不内联
        if has_nested_function(body, arena) {
            continue;
        }
        // 包含 ? 传播运算符的函数不内联
        if has_propagate(body, arena) {
            continue;
        }
        // 包含 return 语句的函数不内联
        if has_return(body, arena) {
            continue;
        }
        // 包含 defer 语句的函数不内联
        if has_defer(body, arena) {
            continue;
        }
        let size = count_expr_nodes(body, arena);
        if size <= INLINE_SIZE_THRESHOLD {
            report.candidates.push((func, size));
            inlineable.insert(func);
        }
    }
    // 第二遍：从 call_sites 中筛选调用可内联函数的调用点，产出 expansions
    for (&expr_id, &callee) in &cg.call_sites {
        if inlineable.contains(&callee) {
            report.expansions.insert(expr_id, callee);
        }
    }
    report
}

/// 检测表达式中是否包含嵌套函数（Lambda 或 LocalDecl 中的 FunDecl）。
/// 包含嵌套函数的函数不应内联：内联展开会引入新子图，
/// 其节点范围与外层子图 node_range 冲突，导致 prepare_frame 误标为嵌套节点永不就绪。
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

/// 检测表达式中是否包含 `?` 传播运算符（Expr::Propagate）。
/// 包含 ? 的函数不应内联：compute_propagate 通过 ControlSignal::Return 实现提前返回，
/// 该信号是函数级作用域，内联后会错误地终止调用方函数。
fn has_propagate(expr_id: ExprId, arena: &AstArena) -> bool {
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
        // defer body 编译为独立子图，? 传播不影响外层函数
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

/// 检测表达式中是否包含 defer 语句。
/// 含 defer 的函数不可内联：defer 注册到函数子图的 defer_table，
/// 内联后函数帧不创建，defer_table 永远不被检查（Bug #47）。
/// Lambda body 中的 defer 不计入（has_nested_function 已排除含 lambda 的函数）。
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
        // Lambda body 中的 defer 不计入（lambda 有独立帧）
        // has_nested_function 已排除含 lambda 的函数
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

/// 检测表达式中是否包含 return 语句（函数级作用域）。
/// Lambda body 中的 return 不计入（scoped to lambda）。
/// Defer body 中的 return 不计入（defer body 编译为独立子图）。
fn has_return(expr_id: ExprId, arena: &AstArena) -> bool {
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
        // defer body 编译为独立子图，return 不影响外层函数
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

/// 递归统计表达式子树的 AST 节点数。
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
// StackAllocPass — 栈分配建议
// =========================================================================

/// 栈分配建议报告。
#[derive(Debug, Default)]
pub struct StackAllocReport {
    /// 可栈分配的分配点 ExprId 列表（EscapeTable 标记为 NoEscape）
    pub candidates: Vec<ExprId>,
}

/// 栈分配建议：EscapeTable 标记为 NoEscape 的分配可改为栈分配。
/// 数据由 EscapeAnalyzer 产出，此处汇总上报供 IR 层使用。
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
// MatchExhaustivenessPass — 模式匹配完备性 + 不可达 arm 检测
// =========================================================================

/// 模式匹配分析报告。
#[derive(Debug, Default)]
pub struct MatchReport {
    /// 非完备 match：(match 表达式 ExprId, scrutinee 类型名, 缺失的构造器名)
    pub non_exhaustive: Vec<(ExprId, String, Vec<String>)>,
    /// 不可达 match arm：(match 表达式 ExprId, arm 索引)
    pub unreachable_arms: Vec<(ExprId, usize)>,
}

/// 模式匹配分析：
/// - 非完备检测：ADT 类型的 match 若未覆盖所有构造器且无 Wildcard → 报告缺失构造器
/// - 不可达 arm 检测：Wildcard 之后的 arm 不可达
pub fn match_pass(module: &Module, arena: &AstArena, sema: &SemaResult, cg: &CallGraph) -> MatchReport {
    let mut report = MatchReport::default();
    // 统一遍历 FunDecl + Method
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
    // 递归子表达式
    walk_children_expr(expr_id, arena, |c| {
        analyze_match_expr(c, arena, module_name, sema, report);
    });
    // 递归 Block 中的语句
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
    // 1. 不可达 arm 检测：Wildcard 之后的 arm 不可达（无 guard 的 Wildcard 终结匹配）
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

    // 2. 非完备检测：scrutinee 类型为 ADT，收集所有 arm 覆盖的构造器名
    let key = module_expr_key(module_name, scrutinee.0 as u64);
    let Some(info) = sema.expr_types.get(&key) else { return };
    let Some(type_name) = info.type_name.as_deref() else { return };
    let Some(&type_idx) = sema.type_def_index.get(type_name) else { return };
    let type_def = &sema.type_defs[type_idx as usize];
    // 仅 ADT 类型有多个构造器需要检查完备性
    if type_def.kind != crate::sema::Sema::TypeDefKind::Adt {
        return;
    }
    let all_ctors: Vec<&str> = type_def.constructors.iter().map(|c| c.name.as_ref()).collect();
    if all_ctors.is_empty() {
        return;
    }

    // 收集 arm 中覆盖的构造器名
    let mut covered: FxHashSet<String> = FxHashSet::default();
    let mut has_wildcard = false;
    for arm in arms {
        let pat = &arena.pattern(arm.pattern).node;
        collect_pattern_ctors(pat, arena, &mut covered, &mut has_wildcard);
    }

    // 有 Wildcard 则视为完备（保守）
    if has_wildcard {
        return;
    }

    // 找出缺失的构造器
    let missing: Vec<String> = all_ctors
        .iter()
        .filter(|c| !covered.contains(**c))
        .map(|c| c.to_string())
        .collect();
    if !missing.is_empty() {
        report.non_exhaustive.push((match_expr, type_name.to_string(), missing));
    }
}

/// 递归收集 pattern 中覆盖的构造器名。
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
            // 字面量模式不覆盖构造器，保守视为 wildcard（可能覆盖部分值）
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
            // 带 guard 的 pattern 不保证覆盖，保守视为可能未覆盖
            collect_pattern_ctors(&arena.pattern(*pattern).node, arena, covered, has_wildcard);
        }
    }
}

// =========================================================================
// AnalysisReport — 汇总报告 + rayon 三层并行入口
// =========================================================================

/// 循环分析报告（IR 构建后由 LoopAnalysis.rs 填充）。
#[derive(Debug, Default)]
pub struct LoopAnalysisReport {
    /// 每个循环 body_sg 的不变量节点列表。
    /// key = body_sg 的 SubGraphId, value = body_sg 内不变量节点的 NodeId 列表。
    pub invariants: FxHashMap<crate::ir::Ir::SubGraphId, Vec<crate::ir::Ir::NodeId>>,
    /// 可展开的循环。
    /// key = 循环 sg 的 SubGraphId, value = 展开信息。
    pub unrollable: FxHashMap<crate::ir::Ir::SubGraphId, UnrollInfo>,
}

/// 循环展开信息。
#[derive(Debug, Clone)]
pub struct UnrollInfo {
    /// 编译期已知的 trip count
    pub trip_count: u32,
    /// 循环变量在 body_sg 中的绑定节点
    pub loop_var_node: crate::ir::Ir::NodeId,
    /// 循环起始值
    pub start_value: i128,
    /// 循环步进
    pub step: i128,
    /// body_sg 的 SubGraphId
    pub body_sg: crate::ir::Ir::SubGraphId,
    /// Range start 的原始 ConstValue（用于保持类型一致）
    pub start_const: crate::ir::Ir::ConstValue,
}

/// 静态分析汇总报告。
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

/// 运行完整三层管线分析。
///
/// Layer 1：DefUseBuilder + CallGraphBuilder（rayon::join 并行，无数据依赖）
/// Layer 2：PurityAnalyzer → EscapeAnalyzer（顺序执行，Escape 依赖 Purity）
/// Layer 3：DeadCodePass + MemoPass 可并行；DeadVarPass 依赖 DeadCodeReport；
///          DeadFuncPass 依赖 MemoPlan。
/// Layer 4：DeadParam + Inline + StackAlloc + Match（可并行，依赖 Layer 1-2 产出）
pub fn analyze(module: &Module, arena: &AstArena, sema: &SemaResult) -> AnalysisReport {
    let module_name = module.name;

    // Layer 1：并行构建 def-use 图与调用图
    let (def_use, call_graph) = rayon::join(
        || build_def_use(module, arena),
        || build_call_graph(module, arena, sema),
    );

    // Layer 2：顺序执行——EscapeAnalyzer 依赖 PurityTable
    let purity = analyze_purity(module, arena, &call_graph, sema);
    let escape = analyze_escape(module, arena, &call_graph, &purity);

    // Layer 3：DeadCodePass 与 MemoPass 无互相依赖，可并行
    let (dead_code, memo) = rayon::join(
        || dead_code_pass(module, arena, module_name, sema, &purity, &escape, &call_graph, &def_use),
        || memo_pass(module, arena, &call_graph, &purity, &escape, sema),
    );
    let dead_var = dead_var_pass(module, arena, &def_use, &dead_code);
    let dead_func = dead_func_pass(&call_graph, &memo);

    // Layer 4：四个新增 pass 并行（依赖 Layer 1-2 产出）
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
        loop_analysis: LoopAnalysisReport::default(), // 由本文件 analyze_loops 在 IR 构建后填充
    }
}

// =========================================================================
// 循环分析 pass（从 LoopAnalysis.rs 合并）
//
// 产出 LoopAnalysisReport：
// - 不变量识别：body_sg 中纯计算且输入来自循环外的节点
// - trip count 估计：For 循环迭代器为常量 Range 时的编译期 trip count
// =========================================================================

use crate::ir::Ir::{
    ComputeFnId, ConstValue, DataFlowGraph, LoopKind, NodeId, NodeKind, SubGraphId,
    CF_CALL_LAUNCH, CF_RANGE, CF_RANGE_INCLUSIVE, pure_compute_fn_set,
};

/// 最大展开 body 节点数
const MAX_UNROLL_BODY_NODES: usize = 32;
/// 最大展开 trip count
const MAX_UNROLL: u32 = 8;

/// 运行循环分析，填充 LoopAnalysisReport。
///
/// 此函数在 IR 构建后运行，直接分析 DataFlowGraph。
/// Analyzer.rs 的 analyze() 在 IR 构建前运行（消费 AST + SemaResult），
/// 因此 loop_analysis 需要在 IR 构建后由 main.rs 调用此函数填充。
pub fn analyze_loops(graph: &DataFlowGraph) -> LoopAnalysisReport {
    let mut report = LoopAnalysisReport::default();
    let pure_set = pure_compute_fn_set();

    // 收集所有循环子图（loop_kind != None 且 != LoopBody）
    let loop_sgs: Vec<SubGraphId> = graph
        .subgraphs
        .iter()
        .enumerate()
        .filter(|(_, sg)| sg.loop_kind != LoopKind::None && sg.loop_kind != LoopKind::LoopBody)
        .map(|(i, _)| SubGraphId(i as u32))
        .collect();

    for loop_sg_id in &loop_sgs {
        let loop_sg = &graph.subgraphs[loop_sg_id.0 as usize];

        // 找到对应的 body_sg（loop_kind == LoopBody 且 loop_parent_sg == loop_sg_id）
        let body_sg_id = graph
            .subgraphs
            .iter()
            .enumerate()
            .find(|(_, sg)| {
                sg.loop_kind == LoopKind::LoopBody && sg.loop_parent_sg == Some(*loop_sg_id)
            })
            .map(|(i, _)| SubGraphId(i as u32));

        let Some(body_sg_id) = body_sg_id else { continue };

        // ── 不变量识别 ──
        let invariants = find_invariants(graph, *loop_sg_id, body_sg_id, &pure_set);
        if !invariants.is_empty() {
            report.invariants.insert(body_sg_id, invariants);
        }

        // ── 循环展开分析（仅 For 循环）──
        if loop_sg.loop_kind == LoopKind::For {
            if let Some(unroll_info) = analyze_unroll(graph, *loop_sg_id, body_sg_id) {
                report.unrollable.insert(*loop_sg_id, unroll_info);
            }
        }
    }

    report
}

/// 识别 body_sg 中的循环不变量节点。
fn find_invariants(
    graph: &DataFlowGraph,
    loop_sg_id: SubGraphId,
    body_sg_id: SubGraphId,
    pure_set: &FxHashSet<ComputeFnId>,
) -> Vec<NodeId> {
    let loop_sg = &graph.subgraphs[loop_sg_id.0 as usize];
    let body_sg = &graph.subgraphs[body_sg_id.0 as usize];
    let (body_start, body_end) = body_sg.node_range;

    // 循环变量依赖节点（cond_node, iter_next_node）
    let mut loop_deps: FxHashSet<NodeId> = FxHashSet::default();
    if let Some(c) = loop_sg.cond_node {
        loop_deps.insert(c);
    }
    if let Some(n) = loop_sg.iter_next_node {
        loop_deps.insert(n);
    }

    // 修改集：body_sg 内有副作用节点写回的目标
    let mut modified: FxHashSet<NodeId> = FxHashSet::default();
    for idx in (body_start.0 as usize)..(body_end.0 as usize) {
        if let Some(Some(wt)) = graph.writeback_targets.get(idx) {
            modified.insert(*wt);
        }
    }

    // 预计算所有循环子图范围（loop_kind != None），用于判断节点是否在函数级。
    // 外提目标为函数级子图，只有所有 inputs 都在函数级（不在任何循环子图内）
    // 或已判定为不变量的节点才能安全外提——依赖循环变量的节点不会被外提，
    // 因为外提到函数级后值不会随循环变化。
    let loop_ranges: Vec<(u32, u32)> = graph
        .subgraphs
        .iter()
        .filter(|sg| sg.loop_kind != LoopKind::None)
        .map(|sg| (sg.node_range.0 .0, sg.node_range.1 .0))
        .collect();
    let is_func_level = |nid: NodeId| -> bool {
        !loop_ranges.iter().any(|&(s, e)| nid.0 >= s && nid.0 < e)
    };

    // 迭代判定不变量（多轮扫描直到收敛）
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

            // 不能是控制流/调用/事件节点
            if matches!(
                node.kind,
                NodeKind::Gate | NodeKind::Call | NodeKind::Await | NodeKind::EventSource
            ) {
                continue;
            }

            // 必须是纯计算
            if !pure_set.contains(&node.compute_fn) {
                continue;
            }

            // 所有 inputs 必须在函数级或已判定的不变量
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

/// 分析 For 循环是否可展开。
fn analyze_unroll(
    graph: &DataFlowGraph,
    loop_sg_id: SubGraphId,
    body_sg_id: SubGraphId,
) -> Option<UnrollInfo> {
    let loop_sg = &graph.subgraphs[loop_sg_id.0 as usize];
    let body_sg = &graph.subgraphs[body_sg_id.0 as usize];

    // body 节点数限制
    let body_size = (body_sg.node_range.1.0 - body_sg.node_range.0.0) as usize;
    if body_size > MAX_UNROLL_BODY_NODES {
        return None;
    }

    // body 内不能有 break/continue/return/throw
    for idx in (body_sg.node_range.0.0 as usize)..(body_sg.node_range.1.0 as usize) {
        if crate::ir::Ir::is_control_flow_compute_fn(graph.nodes[idx].compute_fn) {
            return None;
        }
    }

    // 在 loop_sg 内查找 CF_RANGE / CF_RANGE_INCLUSIVE 构造节点
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

    // Range 的 inputs = [start, end]
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

    // iter_next_node 必须存在且是 Call 节点（Range next 调用）
    let iter_next = loop_sg.iter_next_node?;
    let iter_node = &graph.nodes[iter_next.0 as usize];
    if iter_node.compute_fn != CF_CALL_LAUNCH {
        return None;
    }

    // body_sg 结构：param_0 = 迭代器, param_1 = 当前值（循环变量）
    // loop_var_node = body_sg 的第二个参数节点（param_1 = 当前值）
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

/// 从 ConstValue 提取 i128。
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
