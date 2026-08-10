//! Inference.rs — 类型推断算法层
//!
//! 从 Sema.rs 拆分。依赖 crate::Sema（类型系统基础）+ crate::Relations（类型关系判定）。
//! 职责：类型推断、约束求解、flow-sensitive narrowing。
//! 单态化实例收集已迁移至 crate::Monomorph（Sema 后阶段统一入口）。

use crate::sema::Sema::*;
use crate::sema::Relations::*;
use crate::ast::Ast::{
    AstArena, BinaryOp, Decl, Expr, ExprId, InterpolationPart, LambdaBody, Module,
    Pattern, PatternId, PatternLiteral, PatternRef, Stmt, StmtId,
    TypeNode, TypeRef as AstTypeRef,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// 生成数值字面量推断的 match 臂（IntLit/FloatLit 共用结构）。
/// `$suffix_fn` 为后缀→TypeHandle 方法，`$predicate` 为 expected 类型谓词，`$fallback` 为默认 Ty 变体。
macro_rules! numeric_lit {
    ($self:expr, $suffix:expr, $expected:expr, $suffix_fn:ident, $predicate:ident, $fallback:ident) => {{
        if let Some(suf) = $suffix {
            if let Some(ty) = $self.$suffix_fn(suf) {
                return ty;
            }
        }
        if let Some(exp) = $expected {
            let resolved = $self.arena.resolve(exp);
            if $self.arena.get(resolved).$predicate() {
                return exp;
            }
        }
        $self.make_builtin(Ty::$fallback)
    }};
}

/// 推导上下文：封装类型推导所需的所有状态。
///
/// 生命周期：整个模块的 sema 阶段共享一个 TypeArena，InferContext 持有 &mut 引用。
/// 实例化模式上下文：单态化函数体类型解析时使用。
///
/// 设计：双阶段写入避免别名冲突
/// - 运行时类型结果暂存到 local_expr_types
/// - 运行结束后 take_local_expr_types() 转移到 MonomorphInstance.expr_types
///
/// 不持有 func_decls（带生命周期引用），Call 分支的单态化触发由外部编排。
pub struct InstantiationCtx {
    /// 当前实例的函数名（用于泛型递归短路：func_name == cur_func_name）
    pub func_name: Box<str>,
    /// 当前实例的 type_args（与 type_params 等长，按位置对应）
    pub type_args: Box<[TypeHandle]>,
    /// 类型参数名 → type_args 索引（快速查找）
    pub type_param_map: FxHashMap<String, u16>,
    /// 被调函数所在模块名（跨模块时 expr_types key 必须与 IR Builder 查找一致）
    pub module_name: String,
    /// 暂存的表达式类型表（key = module_expr_key(module_name, expr_id)）
    pub local_expr_types: FxHashMap<u64, ExprInfo>,
    /// 暂存的 field_accesses 元信息（key = module_expr_key）
    pub local_field_accesses: FxHashMap<u64, FieldAccessInfo>,
    /// 循环检测：正在实例化的 cache_key → instance_id（前向引用支持）
    pub in_progress: FxHashMap<u64, u32>,
}

/// type_binding_stack 和 self_binding_stack 随 impl/trait/fn 块进出而 push/pop。
/// env 为局部变量环境（EnvArena），expected_return 用于反向推导 return 类型。
pub struct InferContext<'a> {
    pub arena: &'a mut TypeArena,
    pub sema_result: &'a mut SemaResult,
    pub type_binding_stack: TypeBindingStack,
    pub self_binding_stack: SelfBindingStack,
    pub env: EnvArena,
    /// 当前函数的期望返回类型（用于反向推导 throw 表达式等）
    pub expected_return: Option<TypeHandle>,
    /// sema v2: 约束求解器（延迟求解 + snapshot/rollback）
    pub solver: ConstraintSolver,
    /// sema v2: flow-sensitive narrowing 上下文（path-sensitive 类型精化）
    pub flow_ctx: FlowContext,
    /// sema v2: witness table（trait 实现的静态分派表）
    pub witness_table: WitnessTable,
    /// 模块路径 → 模块专属 EnvId 的映射。
    ///
    /// 每个模块（含路径前缀）在注册时创建一个专属 env（parent 指向 root_env 或父路径 env），
    /// 模块的函数/类型注册于此 env。ModuleRef 查找时直接在对应 env 中按裸名查找，无需 mangled name。
    ///
    /// 层级结构示例：
    ///   "std"            → env_std (parent=root_env)，绑定 "io"→ModuleRef("std.io", env_std_io)
    ///   "std.io"         → env_std_io (parent=env_std)，绑定 "File"→ModuleRef("std.io.File", env_std_io_file)
    ///   "std.io.File"    → env_std_io_file (parent=env_std_io)，绑定 "open"→Fn(...)
    ///
    /// 这使得 `std.io.File.open(...)` 的查找完全通过 env 链结构化进行：
    ///   std → env_std.lookup("io") → ModuleRef("std.io", env_std_io)
    ///       → env_std_io.lookup("File") → ModuleRef("std.io.File", env_std_io_file)
    ///       → Call: env_std_io_file.lookup("open") → Fn(...)
    pub module_envs: FxHashMap<String, EnvId>,
    /// 当前正在检查的模块的逻辑路径（如 "Math.Geometry"），用于注册 mangled 名
    /// 在 check_module_with_env 开始时设置，供 infer_stmt 等不接收 module 参数的方法使用
    pub current_module_logical_path: Option<String>,
    /// 当前正在检查的模块的专属 EnvId。
    /// 在 check_module_with_env 开始时从 module_envs 中查找，predeclare_declarations 时用于注册符号。
    pub current_module_env: Option<EnvId>,
    /// 当前正在检查的模块的文件名（如 "Math/Geometry.kz"），用于 expr_types 复合 key
    /// 避免不同模块的 ExprId 在全局 expr_types 中冲突
    pub current_module_name: String,
    /// 诊断追踪表：记录每个表达式推断结果的 (TypeHandle, Span)，用于反向定位未解析 TypeVar 的代码位置。
    /// 仅在 KUZO_SEMA_TRACE 启用时填充，避免正常编译的内存开销。
    pub type_trace: Vec<(TypeHandle, crate::ast::Ast::Span)>,
    /// 构造器短名 → 定义该类型的模块 EnvId（Zig @This 语义）。
    ///
    /// 当 `import std.time.Duration` 且模块内定义 `pub type Duration` 时，
    /// predefine 用 redefine 将 ModuleRef 覆盖为构造器 Fn。此映射保留
    /// "类型名 → 源模块 env"，使 MethodCall 路径 0b 能回退查找模块内自由函数
    /// （类型名 == 文件名时，类型视作模块命名空间）。
    pub ctor_module_envs: FxHashMap<String, EnvId>,
    /// 实例化模式上下文：None = HM 模式，Some = 实例化模式
    /// 单态化函数体类型解析时设为 Some，HM 类型检查时为 None
    pub instantiation_ctx: Option<InstantiationCtx>,
}

/// 检查类型是否引用了任何未解析的 TypeVar（在 unresolved_set 中）。
/// 用于诊断阶段反向定位未解析 TypeVar 的表达式位置。
fn type_contains_any_unresolved(
    ty: TypeHandle,
    arena: &TypeArena,
    unresolved_set: &FxHashSet<u32>,
) -> bool {
    let resolved = arena.resolve(ty);
    match arena.get(resolved) {
        Ty::TypeVar(idx) => unresolved_set.contains(&idx),
        Ty::Fn(_) => {
            let (params, return_type) = arena.fn_parts(resolved);
            params.iter().any(|&p| type_contains_any_unresolved(p, arena, unresolved_set))
                || type_contains_any_unresolved(return_type, arena, unresolved_set)
        }
        Ty::Record(_) => arena.record_fields(resolved)
            .iter()
            .any(|f| type_contains_any_unresolved(f.ty, arena, unresolved_set)),
        Ty::Adt(_) => {
            let (_, type_args) = arena.adt_parts(resolved);
            type_args
                .iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        Ty::Nullable(_) => {
            let inner = arena.nullable_inner(resolved);
            type_contains_any_unresolved(inner, arena, unresolved_set)
        }
        Ty::Ref(_) => {
            let (inner, _) = arena.ref_parts(resolved);
            type_contains_any_unresolved(inner, arena, unresolved_set)
        }
        Ty::Generic(_) => {
            let (_, args) = arena.generic_parts(resolved);
            args.iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        Ty::Array(_) => {
            let (element_type, _) = arena.array_parts(resolved);
            type_contains_any_unresolved(element_type, arena, unresolved_set)
        }
        Ty::Throw(_) => {
            let (value_type, error_type) = arena.throw_parts(resolved);
            type_contains_any_unresolved(value_type, arena, unresolved_set)
                || type_contains_any_unresolved(error_type, arena, unresolved_set)
        }
        Ty::Trait(_) => {
            let (_, type_args) = arena.trait_parts(resolved);
            type_args
                .iter()
                .any(|&a| type_contains_any_unresolved(a, arena, unresolved_set))
        }
        _ => false,
    }
}

impl<'a> InferContext<'a> {
    pub fn new(arena: &'a mut TypeArena, sema_result: &'a mut SemaResult) -> Self {
        InferContext {
            arena,
            sema_result,
            type_binding_stack: TypeBindingStack::new(),
            self_binding_stack: SelfBindingStack::new(),
            env: EnvArena::new(),
            expected_return: None,
            solver: ConstraintSolver::new(),
            flow_ctx: FlowContext::new(),
            witness_table: WitnessTable::new(),
            module_envs: FxHashMap::default(),
            current_module_logical_path: None,
            current_module_env: None,
            current_module_name: String::new(),
            type_trace: Vec::new(),
            ctor_module_envs: FxHashMap::default(),
            instantiation_ctx: None,
        }
    }

    // ── 类型绑定栈操作 ──

    /// 进入泛型作用域：为每个类型参数分配 rigid var 并压栈。
    /// 未声明 kind 的参数默认 Star，声明的 kind 用于 HKT 检查。
    pub fn push_type_bindings(&mut self, type_params: &[(&str, Option<SemKind>)]) {
        self.type_binding_stack.push();
        for &(name, ref kind_opt) in type_params {
            let var = match kind_opt {
                Some(kind) => self.arena.fresh_rigid_var_with_kind(kind.clone()),
                None => self.arena.fresh_rigid_var(),
            };
            self.type_binding_stack.insert_top(name, var);
        }
    }

    /// 离开泛型作用域：弹出栈顶帧。
    pub fn pop_type_bindings(&mut self) {
        self.type_binding_stack.pop();
    }

    /// 查询类型参数绑定。
    pub fn lookup_type_binding(&self, name: &str) -> Option<TypeHandle> {
        self.type_binding_stack.lookup(name)
    }

    // ── 实例化模式（单态化函数体类型解析）──

    /// 进入实例化模式：用具体 type_args 替换类型参数绑定。
    ///
    /// 调用前应已 push_type_bindings（rigid var），此方法将栈顶帧的 rigid var
    /// 替换为 type_args 中的具体 TypeHandle（insert_top 内部 HashMap::insert 覆盖同名 key）。
    pub fn enter_instantiation_mode(
        &mut self,
        func_name: Box<str>,
        type_args: Box<[TypeHandle]>,
        type_param_names: &[&str],
        module_name: String,
        in_progress: FxHashMap<u64, u32>,
    ) {
        // 将 type_binding_stack 栈顶的 rigid var 替换为具体 type_args
        for (i, &name) in type_param_names.iter().enumerate() {
            if i < type_args.len() {
                self.type_binding_stack.insert_top(name, type_args[i]);
            }
        }

        // 构建 type_param_map
        let mut type_param_map: FxHashMap<String, u16> = FxHashMap::default();
        for (i, &name) in type_param_names.iter().enumerate() {
            type_param_map.insert(name.to_string(), i as u16);
        }

        self.instantiation_ctx = Some(InstantiationCtx {
            func_name,
            type_args,
            type_param_map,
            module_name,
            local_expr_types: FxHashMap::default(),
            local_field_accesses: FxHashMap::default(),
            in_progress,
        });
    }

    /// 离开实例化模式：取出暂存的 local_expr_types 和 local_field_accesses。
    ///
    /// 调用方负责将返回值转移到 MonomorphInstance。
    pub fn leave_instantiation_mode(
        &mut self,
    ) -> Option<(
        FxHashMap<u64, ExprInfo>,
        FxHashMap<u64, FieldAccessInfo>,
        FxHashMap<u64, u32>,
    )> {
        self.instantiation_ctx.take().map(|ctx| {
            (
                ctx.local_expr_types,
                ctx.local_field_accesses,
                ctx.in_progress,
            )
        })
    }

    // ── Self 绑定栈操作 ──

    /// 进入 type 块：Self 绑定到具体类型。
    /// `self_ty` 应为 `Adt { name, type_args }` 形式，type_args 引用 TypeBindingStack 中的 var。
    pub fn push_self_type(&mut self, self_ty: TypeHandle) {
        self.self_binding_stack.push(self_ty);
    }

    /// 进入 trait 默认方法：Self 绑定到 fresh_rigid_var（待 impl 时 unify 求解）。
    /// 用 rigid var 表示 Self 是模板参数，诊断时自动排除（非 rigid 的未绑定 TypeVar 才报错）。
    pub fn push_self_type_var(&mut self) -> TypeHandle {
        let var = self.arena.fresh_rigid_var();
        self.self_binding_stack.push(var);
        var
    }

    /// 离开 type/trait 块：弹出 Self 绑定。
    pub fn pop_self_type(&mut self) {
        self.self_binding_stack.pop();
    }

    /// 当前 Self 类型（栈顶）。
    pub fn current_self_type(&self) -> Option<TypeHandle> {
        self.self_binding_stack.current()
    }

    // ── 错误记录 ──

    pub fn add_error(&mut self, message: &str) {
        // line=0/column=0 表示无位置信息（sema 推导阶段尚未关联 AST 位置）
        self.sema_result.add_error(SemaError::new(message, 0, 0));
    }

    /// 带位置信息的错误添加（用于有 AST span 上下文的调用点）。
    pub fn add_error_at(&mut self, message: &str, line: u32, column: u32) {
        self.sema_result.add_error(SemaError::new(message, line, column));
    }

    // ── self 参数解析（phase3b）──

    /// 判断参数的 type_annotation 是否为 SelfType（或 RefType<SelfType>）。
    ///
    /// 解析器对 type/trait 块内方法的 `self`/`&self` 自动填充 SelfType 注解，
    /// Sema 通过此类型节点判断是否为 self 参数，而非依赖参数名。
    fn is_self_param(&self, type_annotation: Option<AstTypeRef>, ast: &AstArena<'_>) -> bool {
        match type_annotation {
            Some(ta) => match &ast.ty(ta).node {
                crate::ast::Ast::TypeNode::SelfType => true,
                crate::ast::Ast::TypeNode::RefType { inner } => {
                    matches!(ast.ty(*inner).node, crate::ast::Ast::TypeNode::SelfType)
                }
                _ => false,
            },
            None => false,
        }
    }

    /// 解析 self 参数的类型。
    ///
    /// **语义规则（Rust 移植版，有意改进）**：
    /// - `self` 只能在 type/trait 块内的方法中使用（SelfBindingStack 非空）
    /// - `self` 参数不允许类型注解（解析器自动填 SelfType 或 RefType<SelfType>）
    /// - 顶层 fun 写 self 参数 → 报错
    /// - self 参数有显式 `: Type` 注解 → 报错
    ///
    /// **返回值**：
    /// - `self`（无注解，type 块内）→ scope 类型
    /// - `&self`（无注解，type 块内）→ `Ref<scope类型>`
    /// - 非法用法 → 报错并返回 fresh_type_var（错误恢复）
    pub fn infer_self_param(
        &mut self,
        type_annotation: Option<AstTypeRef>,
        ast: &AstArena<'_>,
    ) -> TypeHandle {
        let self_ty = match self.current_self_type() {
            Some(ty) => ty,
            None => {
                // 从 type_annotation 获取 span（若有），否则无位置信息
                let (line, column) = type_annotation
                    .map(|ta| {
                        let s = ast.ty(ta).span;
                        (s.line, s.column)
                    })
                    .unwrap_or((0, 0));
                self.add_error_at(
                    "self parameter requires enclosing type or trait block",
                    line,
                    column,
                );
                return self.arena.fresh_type_var();
            }
        };

        // 检查类型注解：self 参数不允许显式注解
        // 解析器对 `self`（无 `:`）自动填 SelfType，对 `&self` 填 RefType<SelfType>
        // 用户写 `self: Foo` 走 parse_param 的 `:` 分支，type_annotation 为用户类型
        match type_annotation {
            None => {
                // 无注解（理论上不会出现，解析器总为 self 填充）
                self_ty
            }
            Some(ta) => {
                let tn = &ast.ty(ta).node;
                let span = ast.ty(ta).span;
                match tn {
                    // `self`（解析器自动填 SelfType）→ 返回 scope 类型
                    TypeNode::SelfType => self_ty,
                    // `&self`（解析器自动填 RefType<SelfType>）→ 返回 Ref<scope类型>
                    TypeNode::RefType { inner } => {
                        if matches!(ast.ty(*inner).node, TypeNode::SelfType) {

                            self.arena.make_ref(self_ty, false)
                        } else {
                            // `&self: &Foo` 用户显式写引用注解 → 报错
                            self.add_error_at(
                                "self parameter does not allow explicit type annotation",
                                span.line,
                                span.column,
                            );
                            self.arena.fresh_type_var()
                        }
                    }
                    // `self: Foo` 用户显式写注解 → 报错
                    _ => {
                        self.add_error_at(
                            "self parameter does not allow explicit type annotation",
                            span.line,
                            span.column,
                        );
                        self.arena.fresh_type_var()
                    }
                }
            }
        }
    }

    /// 递归收集类型中的所有 TypeVar idx，填入 subst（值为占位 TypeHandle(0)，仅用 key）。
    fn collect_type_vars(&self, ty: TypeHandle, subst: &mut FxHashMap<u32, TypeHandle>) {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Ty::TypeVar(idx) => {
                subst.entry(idx).or_insert(TypeHandle(0));
            }
            Ty::Fn(_) => {
                let (params, return_type) = self.arena.fn_parts(resolved);
                for &p in params.iter() {
                    self.collect_type_vars(p, subst);
                }
                self.collect_type_vars(return_type, subst);
            }
            Ty::Record(_) => {
                let fields = self.arena.record_fields(resolved);
                for f in fields.iter() {
                    self.collect_type_vars(f.ty, subst);
                }
            }
            Ty::Adt(_) => {
                let (_, type_args) = self.arena.adt_parts(resolved);
                for &a in type_args.iter() {
                    self.collect_type_vars(a, subst);
                }
            }
            Ty::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                self.collect_type_vars(inner, subst)
            }
            Ty::Ref(_) => {
                let (inner, _) = self.arena.ref_parts(resolved);
                self.collect_type_vars(inner, subst)
            }
            Ty::Generic(_) => {
                let (_, args) = self.arena.generic_parts(resolved);
                for &a in args.iter() {
                    self.collect_type_vars(a, subst);
                }
            }
            Ty::Array(_) => {
                let (element_type, _) = self.arena.array_parts(resolved);
                self.collect_type_vars(element_type, subst)
            }
            Ty::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                self.collect_type_vars(value_type, subst);
                self.collect_type_vars(error_type, subst);
            }
            Ty::Trait(_) => {
                let (_, type_args) = self.arena.trait_parts(resolved);
                for &a in type_args.iter() {
                    self.collect_type_vars(a, subst);
                }
            }
            Ty::TraitObject(_) => {}
            _ => {}
        }
    }

    /// 实例化函数类型：为签名中所有未绑定 TypeVar 创建 fresh 非刚性副本。
    ///
    /// 多态内置函数（Ok/i8 等用 rigid var 注册的泛型函数）每次调用时必须实例化，
    /// 否则不同调用的类型约束会相互冲突（第一次调用永久绑定后，后续调用无法 unify）。
    /// 非多态函数（签名无 TypeVar）原样返回。
    fn instantiate_fn_type(&mut self, fn_ty: TypeHandle) -> TypeHandle {
    let resolved = self.arena.resolve(fn_ty);
    // 收集函数签名中所有未绑定 TypeVar idx（collect_type_vars 跟随 resolve，
    // 已绑定的 TypeVar 不会被收集）
    let mut subst: FxHashMap<u32, TypeHandle> = FxHashMap::default();
    if !matches!(self.arena.get(resolved), Ty::Fn(_)) {
        return resolved;
    }
    {
        let (params, return_type) = self.arena.fn_parts(resolved);
        for &p in params.iter() {
            self.collect_type_vars(p, &mut subst);
        }
        self.collect_type_vars(return_type, &mut subst);
    }
    if subst.is_empty() {
        return resolved;
    }
    // 为每个 idx 创建 fresh non-rigid var（collect_type_vars 借用已释放，可安全可变借用）
    let indices: Vec<u32> = subst.keys().copied().collect();
    for idx in indices {
        let fresh = self.arena.fresh_type_var();
        subst.insert(idx, fresh);
    }
    self.substitute_type(resolved, &subst)
}

    /// 类型替换：将类型中的指定 TypeVar（按 idx）替换为绑定表中的类型。
    ///
    /// 递归遍历复合类型，替换匹配的 TypeVar。用于将形参的 rigid var 替换为
    /// 调用点的 fresh 非刚性 var，使其可被 unify 绑定。
    fn substitute_type(&mut self, ty: TypeHandle, subst: &FxHashMap<u32, TypeHandle>) -> TypeHandle {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Ty::TypeVar(idx) => {
                // 命中替换表 → 返回替换类型；否则保持原样
                subst.get(&idx).copied().unwrap_or(resolved)
            }
            Ty::Fn(_) => {
                let (params, return_type) = self.arena.fn_parts(resolved);
                let params: Vec<TypeHandle> = params.to_vec();
                let new_params: Vec<TypeHandle> = params
                    .iter()
                    .map(|&p| self.substitute_type(p, subst))
                    .collect();
                let new_ret = self.substitute_type(return_type, subst);
                self.arena.make_fn(new_params.into_boxed_slice(), new_ret)
            }
            Ty::Record(_) => {
                let fields = self.arena.record_fields(resolved).to_vec();
                let name = self.arena.record_name(resolved).map(|s| s.into());
                let new_fields: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: f.name.clone(),
                        ty: self.substitute_type(f.ty, subst),
                    })
                    .collect();
                self.arena.make_record(new_fields.into_boxed_slice(), name)
            }
            Ty::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                let name: Box<str> = name.into();
                let type_args: Vec<TypeHandle> = type_args.to_vec();
                let new_args: Vec<TypeHandle> = type_args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_adt(name, new_args.into_boxed_slice())
            }
            Ty::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                let new_inner = self.substitute_type(inner, subst);
                self.arena.make_nullable(new_inner)
            }
            Ty::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                let name: Box<str> = name.into();
                let args: Vec<TypeHandle> = args.to_vec();
                let new_args: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_generic(name, new_args.into_boxed_slice())
            }
            Ty::Array(_) => {
                let (element_type, size) = self.arena.array_parts(resolved);
                let new_elem = self.substitute_type(element_type, subst);
                self.arena.make_array(new_elem, size)
            }
            Ty::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                let new_v = self.substitute_type(value_type, subst);
                let new_e = self.substitute_type(error_type, subst);
                self.arena.make_throw(new_v, new_e)
            }
            Ty::Trait(_) => {
                let (name, type_args) = self.arena.trait_parts(resolved);
                let name: Box<str> = name.into();
                let type_args: Vec<TypeHandle> = type_args.to_vec();
                let new_args: Vec<TypeHandle> = type_args
                    .iter()
                    .map(|&a| self.substitute_type(a, subst))
                    .collect();
                self.arena.make_trait(name, new_args.into_boxed_slice())
            }
            Ty::TraitObject(_) => {
                let (trait_name, method_sigs) = self.arena.trait_object_parts(resolved);
                self.arena.make_trait_object(trait_name.into(), method_sigs.to_vec().into_boxed_slice())
            }
            Ty::Ref(_) => {
                let (inner, is_raw) = self.arena.ref_parts(resolved);
                let new_inner = self.substitute_type(inner, subst);
                self.arena.make_ref(new_inner, is_raw)
            }
            // 标量、Never、Unknown、Void、Null 等无子节点 → 原样返回
            _ => resolved,
        }
    }

    // ── 字面量提升 ──
    // v2 收敛：literal_promotion 已由 peer_type_binary 替代，
    // 字面量提升规则内化到 peer_type_binary 中，消除双轨制。

    // ── GADT 推断（phase3e）──

    /// 对构造器模式进行 GADT 类型精化。
    ///
    /// **语义**（移植自 `src/sema/gadt_check.zig` refineConstructorPattern）：
    /// 1. 从 sema_result 查找构造器定义（CtorDefInfo）
    /// 2. 将构造器返回类型与 expected_ty unify，实现类型变量精化
    /// 3. 对子模式按构造器字段类型递归推断
    ///
    /// **返回值**：`true` 表示已由本函数处理（构造器已注册）；
    /// `false` 表示构造器未注册，交由常规模式推断处理。
    ///
    /// **Throw 错误分支**：当 expected_ty 是 Throw 类型且构造器是 error_newtype
    /// ADT 构造器时，`is_throw_error_branch` 标志为真，构造器返回类型与子模式
    /// 统一绑定到 error_type。该标志贯穿返回类型解析与子模式绑定两个步骤，
    /// 无独立早退分支，与常规 GADT 路径走同一控制流。
    pub fn refine_constructor_pattern(
        &mut self,
        ctor_name: &str,
        sub_patterns: &[PatternRef],
        expected_ty: TypeHandle,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> bool {
        // 使用 field_type_reprs（自包含 TypeRepr）替代 field_type_nodes（AST 引用），
        // 避免跨模块使用时 AST arena 不匹配导致 TypeRef 索引指向错误类型节点。
        // return_type_node 仍用 AstTypeRef（GADT 场景少且通常同模块）。
        type CtorInfoSnapshot = (Box<str>, bool, Option<AstTypeRef>, Box<[TypeRepr]>);
        let resolved_expected = self.arena.resolve(expected_ty);

        // 先克隆构造器信息，避免 &CtorDefInfo 借用阻塞后续 &mut self 调用
        let ctor_info: Option<CtorInfoSnapshot> =
            self.find_ctor_def(ctor_name).map(|c| {
                (
                    c.type_name.clone(),
                    c.is_newtype,
                    c.return_type_node,
                    c.field_type_reprs.clone(),
                )
            });

        // Throw<T, E> 内置类型变体匹配：
        // Throw 是内置 sum 类型，变体 Ok(T) / Error(E) 未注册为 CtorDefInfo。
        // - 构造器未注册（如 Ok）→ 值变体 → 子模式绑定到 value_type
        // - 构造器已注册（如 Error ADT、newtype 错误构造器）→ 错误变体 → 子模式绑定到 error_type
        //   （构造器名与 Throw 错误变体名 "Error" 碰撞时，无论 error_type 是何类型，
        //    模式均匹配 Throw 错误变体，子模式绑定到 error_type 而非构造器字段类型）
        if let Ty::Throw(_) = self.arena.get(resolved_expected) {
            let (value_type, error_type) = self.arena.throw_parts(resolved_expected);
            let branch_ty = if ctor_info.is_some() { error_type } else { value_type };
            for &sub_pat in sub_patterns.iter() {
                self.infer_pattern(sub_pat, ast, branch_ty, env);
            }
            return true;
        }

        let (type_name, is_newtype, return_type_node, field_type_reprs) = match ctor_info {
            Some(info) => info,
            None => return false,
        };
        let _ = is_newtype;

        // 解析构造器返回类型（GADT → return_type_node，普通 ADT → type_name 对应的 Adt）
        // 走 InferContext 完整类型解析（type_from_ast），统一所有 TypeNode 变体处理，
        // 消除简化版 resolve_type_node_to_handle 对复杂类型 fresh_type_var 兜底导致的类型丢失。
        let ctor_return_ty = if let Some(rtn) = return_type_node {
            self.type_from_ast(rtn, ast)
        } else {
            self.arena.make_adt(type_name, Box::new([]))
        };

        // unify 构造器返回类型与期望类型，实现 GADT 类型精化
        // 失败时注册约束供不动点迭代重试
        let ctor_compatible = self.arena.unify(ctor_return_ty, expected_ty).is_ok();
        if !ctor_compatible {
            self.unify_or_constrain(ctor_return_ty, expected_ty);
        }

        // 对子模式按构造器字段类型递归推断并绑定变量
        // 当构造器返回类型与期望类型不兼容时（如 Error ADT 用于解包 Throw 的 error_type），
        // 子模式绑定到 expected_ty 而非构造器字段类型，确保模式变量获得正确的运行时类型
        for (i, &sub_pat) in sub_patterns.iter().enumerate() {
            let sub_ty = if !ctor_compatible {
                expected_ty
            } else if i < field_type_reprs.len() {
                self.type_repr_to_handle(&field_type_reprs[i])
            } else {
                self.arena.fresh_type_var()
            };
            self.infer_pattern(sub_pat, ast, sub_ty, env);
        }

        true
    }

    /// 从 sema_result 查找构造器定义（按名称）。
    fn find_ctor_def(&self, ctor_name: &str) -> Option<&CtorDefInfo> {
        self.sema_result.get_ctor_def(ctor_name)
    }
}


// =========================================================================
// phase5: InferContext 扩展 — 类型解析、freshen、结构相等、throw 检查
//
// 新增 InferContext 方法，移植自 `src/sema/type_check.zig` 与 `throw_check.zig`。
// =========================================================================

/// 内置 cast 函数注册表：(函数名, 是否 try 变体)。
/// 新增 cast 变体只需追加一行，无需新增函数名特判分支。
const CAST_BUILTINS: &[(&str, bool)] = &[
    ("__cast_to", false),
    ("__cast_try_to", true),
];

impl<'a> InferContext<'a> {
    // ── 类型解析（typeFromAst）──

    /// 将 AST TypeNode 解析为 TypeHandle（便捷版，无类型参数映射）。
    pub fn type_from_ast(&mut self, type_ref: AstTypeRef, ast: &AstArena<'_>) -> TypeHandle {
        let empty = FxHashMap::default();
        self.type_from_ast_with_params(type_ref, ast, &empty)
    }

    /// 按名称解析为 TypeHandle（别名穿透 + 循环检测）。
    ///
    /// 这是 Named 类型解析的核心：type_param_map → type_binding → 内置标量 →
    /// trait → type_defs 中的 Alias 递归展开 → 用户自定义 Adt。
    /// `visiting` 用于 alias 循环检测（A→B→A），出现循环时返回 Adt(name) 终止。
    fn resolve_name_to_type(
        &mut self,
        name: &str,
        type_param_map: &FxHashMap<String, TypeHandle>,
        visiting: &mut FxHashSet<String>,
    ) -> TypeHandle {
        // 1. 类型参数映射
        if let Some(ty) = type_param_map.get(name) {
            return *ty;
        }
        // 2. 类型绑定栈（泛型作用域）
        if let Some(ty) = self.lookup_type_binding(name) {
            return ty;
        }
        // 3. 内置标量 + str/null/void：派生自 BUILTIN_TABLE
        if let Some(ct) = name_to_concrete(name) {
            return self.arena.make(ct);
        }
        // 4. trait 定义 → Trait 类型
        if self.sema_result.get_trait_def(name).is_some() {
            return self.arena.make_trait(name.into(), Box::new([]));
        }
        // 循环 alias 检测
        if visiting.contains(name) {
            return self.arena.make_adt(name.into(), Box::new([]));
        }
        visiting.insert(name.to_string());
        // 5. 别名穿透：type Name = T → 解析 T
        // 优先使用已解析的 target_type（TypeHandle），覆盖函数/Record/Array 等非命名目标；
        // 退而使用 target_type_name（命名目标，如 type A = B）。
        let (alias_target_ty, alias_target_name): (Option<TypeHandle>, Option<String>) = self
            .sema_result
            .get_type_def(name)
            .filter(|td| td.kind == TypeDefKind::Alias)
            .map(|td| (td.target_type, td.target_type_name.as_deref().map(String::from)))
            .unwrap_or((None, None));
        if let Some(inner_ty) = alias_target_ty {
            visiting.remove(name);
            return inner_ty;
        }
        if let Some(target_name) = alias_target_name {
            let result = self.resolve_name_to_type(&target_name, type_param_map, visiting);
            visiting.remove(name);
            return result;
        }
        visiting.remove(name);
        // 6. 用户自定义类型 → Adt
        self.arena.make_adt(name.into(), Box::new([]))
    }

    /// 将 AST TypeNode 解析为 TypeHandle（完整版，带类型参数映射）。
    ///
    /// 处理所有 TypeNode 变体：Named、SelfType、Generic、Nullable、RefType、RawPtr、
    /// Function、Record、Array、KindAnnotated。内置标量走 from_scalar_name；
    /// 泛型 Throw 特殊处理为 Throw 类型；其余内置泛型构造为 Generic；
    /// 自定义 ADT 构造为 Adt；trait 构造为 Trait。
    pub fn type_from_ast_with_params(
        &mut self,
        type_ref: AstTypeRef,
        ast: &AstArena<'_>,
        type_param_map: &FxHashMap<String, TypeHandle>,
    ) -> TypeHandle {
        let tn = &ast.ty(type_ref).node;
        match tn {
            TypeNode::Named { name } => {
                // 委托 resolve_name_to_type：内置标量 → trait → 别名穿透 → Adt
                let mut visiting = FxHashSet::default();
                self.resolve_name_to_type(name, type_param_map, &mut visiting)
            }
            TypeNode::SelfType => match self.current_self_type() {
                Some(ty) => ty,
                None => {
                    let span = ast.ty(type_ref).span;
                    self.add_error_at("Self type can only be used within type or trait methods", span.line, span.column);
                    self.arena.make(Ty::Void)
                }
            },
            TypeNode::Generic { name, args } => {
                // 递归解析类型实参
                let new_args: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.type_from_ast_with_params(a, ast, type_param_map))
                    .collect();
                let args_box: Box<[TypeHandle]> = new_args.into_boxed_slice();

                // 类型参数映射中的高阶类型（HKT）：F<T> 其中 F 是类型参数
                if let Some(&param_handle) = type_param_map.get(*name) {
                    // kind 检查：验证 F 的 kind 与参数数量和 kind 一致
                    let constructor_kind = self.arena.kind_of(param_handle);
                    // 如果 constructor_kind 不是 Star（即 F 是类型构造器），
                    // 或 args 非空（即 F<T> 应用），执行 kind 检查
                    if !matches!(constructor_kind, SemKind::Star) || !args_box.is_empty() {
                        let arg_kinds: Vec<SemKind> = args_box
                            .iter()
                            .map(|&a| self.arena.kind_of(a))
                            .collect();
                        if let Err(kind_err) = self.arena.check_kind_application(&constructor_kind, &arg_kinds) {
                            // 错误恢复：记录错误但继续构造类型
                            let span = ast.ty(type_ref).span;
                            self.add_error_at(&kind_err, span.line, span.column);
                        }
                    }
                    return self.arena.make_generic((*name).into(), args_box);
                }
                // 内置泛型类型（Throw/Atomic/Async/Channel 等）构造专用 Ty 变体，
                // 不再走 Ty::Generic 路径——避免后续用字符串名匹配识别内置泛型。
                if is_builtin_generic_type(name) {
                    return self.make_builtin_generic((*name).into(), args_box);
                }
                // trait 定义 → Trait 类型
                if self.sema_result.get_trait_def(name).is_some() {
                    return self.arena.make_trait((*name).into(), args_box);
                }
                // 用户自定义泛型 ADT
                let has_type_params = self
                    .sema_result
                    .get_type_def(name)
                    .map(|d| !d.type_params.is_empty())
                    .unwrap_or(false);
                if has_type_params {
                    return self.arena.make_adt((*name).into(), args_box);
                }
                // 兜底：构造 Generic（可能未定义或前向引用，后续使用时报错）
                self.arena.make_generic((*name).into(), args_box)
            }
            TypeNode::Nullable { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_nullable(inner_ty)
            }
            TypeNode::RefType { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_ref(inner_ty, false)
            }
            TypeNode::RawPtr { inner } => {
                let inner_ty = self.type_from_ast_with_params(*inner, ast, type_param_map);
                self.arena.make_ref(inner_ty, true)
            }
            TypeNode::Function { params, return_type } => {
                let new_params: Vec<TypeHandle> = params
                    .iter()
                    .map(|&p| self.type_from_ast_with_params(p, ast, type_param_map))
                    .collect();
                let new_ret = self.type_from_ast_with_params(*return_type, ast, type_param_map);
                self.arena.make_fn(new_params.into_boxed_slice(), new_ret)
            }
            TypeNode::Record { fields } => {
                if fields.is_empty() {
                    return self.arena.make(Ty::Void);
                }
                let new_fields: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: Some(f.name.into()),
                        ty: self.type_from_ast_with_params(f.ty, ast, type_param_map),
                    })
                    .collect();
                self.arena.make_record(new_fields.into_boxed_slice(), None)
            }
            TypeNode::Array { element_type, size } => {
                let elem_ty = self.type_from_ast_with_params(*element_type, ast, type_param_map);
                self.arena.make_array(elem_ty, *size)
            }
            TypeNode::KindAnnotated { inner, .. } => {
                self.type_from_ast_with_params(*inner, ast, type_param_map)
            }
        }
    }

    // ── freshen_type / apply_type_subst ──

    /// 刷新类型：将类型中的未绑定 TypeVar 替换为新的 TypeVar。
    /// 用于从环境查找泛型函数类型时保持各次调用的独立性（替代旧 HM instantiate）。
    pub fn freshen_type(&mut self, ty: TypeHandle) -> TypeHandle {
        // 1. 收集所有未绑定的 TypeVar idx
        let mut free_vars: Vec<u32> = Vec::new();
        self.collect_free_vars(ty, &mut free_vars);
        if free_vars.is_empty() {
            return ty;
        }
        // 2. 为每个 free var 分配 fresh var，构建替换表
        let mut subst: FxHashMap<u32, TypeHandle> = FxHashMap::default();
        for idx in free_vars.iter() {
            let fresh = self.arena.fresh_type_var();
            subst.insert(*idx, fresh);
        }
        // 3. 应用替换
        self.apply_type_subst(ty, &subst)
    }

    /// 递归收集类型中的未绑定 TypeVar idx（去重）。
    ///
    /// 注意：Fn 类型不收集内部 TypeVar。函数类型是"类型方案"（type scheme），
    /// 其自由变量的实例化由调用点的 `instantiate_fn_type` 统一处理。
    /// 若 freshen_type 也实例化 Fn 内部变量，会与 instantiate_fn_type 产生重复实例化，
    /// 导致第一组 fresh 副本成为孤儿（未被任何 unify 引用），最终被报告为未解析 TypeVar。
    fn collect_free_vars(&self, ty: TypeHandle, free_vars: &mut Vec<u32>) {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Ty::TypeVar(idx) => {
                // rigid var 代表泛型参数声明（如 type ArrayIter<T> 中的 T），
                // 在当前作用域是固定的，不应被 freshen 实例化。
                // 仅收集非 rigid 的未绑定 TypeVar（局部推断变量）。
                if !self.arena.type_var(idx).is_rigid && !free_vars.contains(&idx) {
                    free_vars.push(idx);
                }
            }
            // Fn 类型跳过：实例化由 instantiate_fn_type 在调用点处理
            Ty::Fn(_) => {}
            Ty::Record(_) => {
                let fields = self.arena.record_fields(resolved);
                for f in fields.iter() {
                    self.collect_free_vars(f.ty, free_vars);
                }
            }
            Ty::Adt(_) => {
                let (_, type_args) = self.arena.adt_parts(resolved);
                for &a in type_args.iter() {
                    self.collect_free_vars(a, free_vars);
                }
            }
            Ty::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                self.collect_free_vars(inner, free_vars)
            }
            Ty::Ref(_) => {
                let (inner, _) = self.arena.ref_parts(resolved);
                self.collect_free_vars(inner, free_vars)
            }
            Ty::Generic(_) => {
                let (_, args) = self.arena.generic_parts(resolved);
                for &a in args.iter() {
                    self.collect_free_vars(a, free_vars);
                }
            }
            Ty::Array(_) => {
                let (element_type, _) = self.arena.array_parts(resolved);
                self.collect_free_vars(element_type, free_vars)
            }
            Ty::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                self.collect_free_vars(value_type, free_vars);
                self.collect_free_vars(error_type, free_vars);
            }
            Ty::Trait(_) => {
                let (_, type_args) = self.arena.trait_parts(resolved);
                for &a in type_args.iter() {
                    self.collect_free_vars(a, free_vars);
                }
            }
            Ty::TraitObject(_) => {}
            _ => {}
        }
    }

    /// 用替换表替换类型中的 TypeVar（按 idx）。无副作用，返回新类型。
    /// 委托给已有的 `substitute_type` 实现。
    pub fn apply_type_subst(
        &mut self,
        ty: TypeHandle,
        subst: &FxHashMap<u32, TypeHandle>,
    ) -> TypeHandle {
        self.substitute_type(ty, subst)
    }

    // ── types_structurally_equal ──

    /// 无副作用的类型结构相等检查：不修改任何 TypeVar，不触发 unify 副作用。
    /// 用于 trait 方法签名匹配时比较参数类型和返回类型。
    ///
    /// 委托给自由函数 `types_equal`，避免重复维护两套结构相等逻辑。
    pub fn types_structurally_equal(&self, a: TypeHandle, b: TypeHandle) -> bool {
        types_equal(self.arena, a, b)
    }

    // ── throw_check 方法 ──

    /// 统一函数声明的返回类型与函数体推断出的类型。
    /// 针对 nullable/throw 返回类型有特殊放宽：函数体返回 void（早退/抛出）
    /// 时不视为不匹配；否则尝试宽化统一，失败再回退到严格 unify。
    pub fn unify_return_type(
        &mut self,
        declared: TypeHandle,
        inferred: TypeHandle,
    ) -> Result<(), UnifyError> {
        let r_declared = self.arena.resolve(declared);
        let r_inferred = self.arena.resolve(inferred);

        let declared_ty = self.arena.get(r_declared);
        let inferred_ty = self.arena.get(r_inferred);

        // async 函数：声明返回类型应为 Async<X>，body 推断为 Async<Y>
        // 递归统一内部类型 X 与 Y
        if let (Ty::Async(_), Ty::Async(_)) = (declared_ty, inferred_ty) {
            let da = self.arena.async_value(r_declared);
            let ia = self.arena.async_value(r_inferred);
            return self.unify_return_type(da, ia);
        }
        // async 函数 body 直接返回内层值（非 Async 包装）：
        // 声明 Async<X>，body 推断为 Y → 递归统一 X 与 Y
        if let Ty::Async(_) = declared_ty {
            let da = self.arena.async_value(r_declared);
            return self.unify_return_type(da, r_inferred);
        }

        match declared_ty {
            Ty::Nullable(_) => match inferred_ty {
                Ty::Nullable(_) => self.arena.unify(declared, inferred),
                Ty::Void => Ok(()), // 函数体未产生值，与 nullable 兼容
                _ => {
                    let inner_ty = self.arena.nullable_inner(r_declared);
                    match self.try_widen_unify(inner_ty, r_inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(inner_ty, r_inferred),
                    }
                }
            },
            Ty::Throw(_) => match inferred_ty {
                Ty::Throw(_) => {
                    match self.try_widen_unify(declared, inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(declared, inferred),
                    }
                }
                Ty::Void => Ok(()), // 函数体未产生值，与 throw 兼容
                _ => {
                    let (vt, _) = self.arena.throw_parts(r_declared);
                    match self.try_widen_unify(vt, r_inferred) {
                        Ok(_) => Ok(()),
                        Err(_) => self.arena.unify(vt, r_inferred),
                    }
                }
            },
            _ => {
                match self.try_widen_unify(r_declared, r_inferred) {
                    Ok(_) => Ok(()),
                    Err(_) => self.arena.unify(declared, inferred),
                }
            }
        }
    }

    /// 立即 unify 两个类型，失败时注册为 Equality 约束供不动点迭代重试。
    ///
    /// 替代 `let _ = self.arena.unify(t1, t2)` 模式：
    /// - unify 成功 → 立即绑定（保持推断时序优势）
    /// - unify 失败 → 注册 Equality 约束到 solver，由不动点迭代重试
    ///   （其他约束可能先绑定相关 TypeVar，使后续 unify 成功）
    #[inline]
    pub fn unify_or_constrain(&mut self, t1: TypeHandle, t2: TypeHandle) {
        // 实例化模式：跳过 HM 约束求解（类型已在 sema HM 阶段检查）
        if self.instantiation_ctx.is_some() {
            return;
        }
        if self.arena.unify(t1, t2).is_err() {
            self.solver.add_equality(t1, t2);
        }
    }

    /// 尝试对两个类型进行宽化统一，返回统一后的类型。
    /// 先尝试严格 unify；失败时若二者均为数值则按宽化规则择一返回；
    /// 否则针对 nullable/throw 与普通类型、void 等组合做结构性兼容处理。
    pub fn try_widen_unify(
        &mut self,
        t1: TypeHandle,
        t2: TypeHandle,
    ) -> Result<TypeHandle, UnifyError> {
        let r1 = self.arena.resolve(t1);
        let r2 = self.arena.resolve(t2);

        // never 与任何类型统一为对方
        if matches!(self.arena.get(r1), Ty::Never) {
            return Ok(r2);
        }
        if matches!(self.arena.get(r2), Ty::Never) {
            return Ok(r1);
        }

        // 先尝试严格 unify
        if self.arena.unify(r1, r2).is_ok() { return Ok(r1) }

        let c1 = self.arena.get(r1);
        let c2 = self.arena.get(r2);

        // async 穿透：Async<X> 与 Y（非 Async）→ 递归统一 X 与 Y
        // 场景：async 函数体中 Ok(void) 返回 Throw<void, '_E>，
        // expected 为 Async<Throw<void, IOError>>，需穿透 Async 层求解 '_E
        if let Ty::Async(_) = c1 {
            let inner = self.arena.async_value(r1);
            return self.try_widen_unify(inner, r2);
        }
        if let Ty::Async(_) = c2 {
            let inner = self.arena.async_value(r2);
            return self.try_widen_unify(r1, inner);
        }

        // 数值类型之间尝试宽化
        if c1.is_numeric() && c2.is_numeric() {
            if can_coerce_numeric(self.arena, r1, r2) {
                return Ok(r1);
            }
            if can_coerce_numeric(self.arena, r2, r1) {
                return Ok(r2);
            }
            return Err(UnifyError::TypeMismatch);
        }

        match (c1, c2) {
            (Ty::Nullable(_), _) => match c2 {
                Ty::Nullable(_) => {
                    let i1 = self.arena.resolve(self.arena.nullable_inner(r1));
                    let i2 = self.arena.resolve(self.arena.nullable_inner(r2));
                    match self.arena.unify(i1, i2) {
                        Ok(_) => Ok(r1),
                        Err(_) => {
                            if self.arena.get(i1).is_numeric()
                                && self.arena.get(i2).is_numeric()
                                && can_coerce_numeric(self.arena, i1, i2)
                            {
                                Ok(r1)
                            } else {
                                Err(UnifyError::TypeMismatch)
                            }
                        }
                    }
                }
                Ty::Void => Ok(r1), // void 可视为 nullable 的"空值"
                _ => {
                    // nullable<T> 与 T 兼容
                    let inner1_ty = self.arena.nullable_inner(r1);
                    match self.arena.unify(inner1_ty, r2) {
                        Ok(_) => Ok(r1),
                        Err(_) => {
                            let i1 = self.arena.resolve(inner1_ty);
                            let r2r = self.arena.resolve(r2);
                            if self.arena.get(i1).is_numeric()
                                && self.arena.get(r2r).is_numeric()
                                && can_coerce_numeric(self.arena, i1, r2r)
                            {
                                Ok(r1)
                            } else {
                                Err(UnifyError::TypeMismatch)
                            }
                        }
                    }
                }
            },
            (Ty::Throw(_), _) => match c2 {
                Ty::Throw(_) => {
                    let (vt1, et1) = self.arena.throw_parts(r1);
                    let (vt2, et2) = self.arena.throw_parts(r2);
                    let v1 = self.arena.resolve(vt1);
                    let v2 = self.arena.resolve(vt2);
                    let e1 = self.arena.resolve(et1);
                    let e2 = self.arena.resolve(et2);
                    self.arena.unify(e1, e2)?;
                    match self.arena.unify(v1, v2) {
                        Ok(_) => Ok(r1),
                        Err(_) => {
                            match self.try_widen_unify(v1, v2) {
                                Ok(_) => Ok(r1),
                                Err(_) => {
                                    if self.arena.get(v1).is_numeric()
                                        && self.arena.get(v2).is_numeric()
                                        && can_coerce_numeric(self.arena, v1, v2)
                                    {
                                        Ok(r1)
                                    } else {
                                        Err(UnifyError::TypeMismatch)
                                    }
                                }
                            }
                        }
                    }
                }
                Ty::Void => Ok(r1), // void 可视为 throw 的"未取值"
                _ => {
                    // Throw<T, E> 与 T 兼容（仅取值维度）
                    let (vt1_ty, _) = self.arena.throw_parts(r1);
                    match self.arena.unify(vt1_ty, r2) {
                        Ok(_) => Ok(r1),
                        Err(_) => {
                            let v1 = self.arena.resolve(vt1_ty);
                            let r2r = self.arena.resolve(r2);
                            if self.arena.get(v1).is_numeric()
                                && self.arena.get(r2r).is_numeric()
                                && can_coerce_numeric(self.arena, v1, r2r)
                            {
                                Ok(r1)
                            } else {
                                Err(UnifyError::TypeMismatch)
                            }
                        }
                    }
                }
            },
            (Ty::Void, _) => match c2 {
                Ty::Nullable(_) | Ty::Throw(_) => Ok(r2),
                _ => Err(UnifyError::TypeMismatch),
            },
            (_, Ty::Nullable(_)) => {
                // T 与 nullable<T> 兼容，统一为 nullable
                let inner2_ty = self.arena.nullable_inner(r2);
                self.arena.unify(r1, inner2_ty)?;
                Ok(r2)
            }
            (_, Ty::Throw(_)) => {
                // T 与 Throw<T, E> 兼容，统一为 throw
                let (vt2_ty, _) = self.arena.throw_parts(r2);
                self.arena.unify(r1, vt2_ty)?;
                Ok(r2)
            }
            _ => Err(UnifyError::TypeMismatch),
        }
    }

    /// 检查传播操作符 `?` 在表达式上的合法性，并返回展开后的类型。
    ///
    /// `expected_return` 为外层函数的返回类型（可能是 `Async<Throw<V, E>>` 或 `Throw<V, E>`），
    /// 用于统一 error_type，使 throw 传播类型正确。
    ///
    /// - nullable：展开为内层类型
    /// - throw：展开为值类型，并将 error_type 与外层函数的 error_type 统一
    /// - TypeVar：延迟到 solver 求解，返回 fresh_type_var 避免级联误报
    /// - 其它类型：报错并返回原类型
    pub fn check_propagate(
        &mut self,
        resolved_inner: TypeHandle,
        inner_ty: TypeHandle,
        expected_return: Option<TypeHandle>,
        line: u32,
        column: u32,
    ) -> TypeHandle {
        let ct = self.arena.get(resolved_inner);
        match ct {
            Ty::Nullable(_) => self.arena.nullable_inner(resolved_inner),
            Ty::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved_inner);
                // 将 error_type 与外层函数的 error_type 统一（当外层是 throwing 函数时）
                // Kuzo 允许在非 throwing 函数中使用 `?`（失败时 panic/退出），此时不传播 error_type
                if let Some(er) = expected_return {
                    let er_resolved = self.arena.resolve(er);
                    let er_ty = self.arena.get(er_resolved);
                    // async 函数：expected_return 可能是 Async<Throw<V', E'>>
                    let outer_throw_handle = if let Ty::Async(_) = er_ty {
                        Some(self.arena.resolve(self.arena.async_value(er_resolved)))
                    } else {
                        None
                    };
                    let outer_resolved = outer_throw_handle.unwrap_or(er_resolved);
                    if let Ty::Throw(_) = self.arena.get(outer_resolved) {
                        let (_, outer_err) = self.arena.throw_parts(outer_resolved);
                        self.unify_or_constrain(error_type, outer_err);
                    }
                    // 非 Throw 外层（如 void）或 TypeVar：静默跳过，不报错
                }
                value_type
            }
            Ty::TypeVar(_) => {
                // operand 类型尚未确定，延迟到 solver 求解后再判定
                // 返回 fresh_type_var 避免下游方法查找级联误报
                self.arena.fresh_type_var()
            }
            _ => {
                self.add_error_at(
                    "propagation operator '?' cannot be used on a non-nullable, non-throw expression",
                    line,
                    column,
                );
                inner_ty
            }
        }
    }

    /// 检查 throw 语句的表达式类型。
    /// Kuzo 无 try-catch，throw 是通用抛出机制，接受任意 ADT/Record/Throw/TypeVar。
    pub fn check_throw_stmt(&mut self, thrown_ty: TypeHandle, _line: u32, _column: u32) {
        let resolved = self.arena.resolve(thrown_ty);
        let ct = self.arena.get(resolved);
        match ct {
            Ty::TypeVar(_) => return,   // 延迟到统一阶段
            Ty::Throw(_) => return, // throw Error("...") 返回 Throw，合法
            Ty::Adt(_) | Ty::Generic(_) => return, // 错误类型（普通 ADT）
            _ => return, // 保守放行，throw 是通用机制
        }
    }

    // ── infer_expr / infer_stmt / infer_pattern 占位（下方实现）──

    /// 获取内置标量类型的 TypeHandle（辅助）。
    fn make_builtin(&mut self, ty: Ty) -> TypeHandle {
        self.arena.make(ty)
    }

    /// 构造内置泛型类型的专用 Ty 变体（Throw/Channel/Async/Lazy/Atomic/Sender/Receiver）。
    /// arity 不匹配时回退到 Ty::Generic（容错，sema 已对内置泛型 arity 有约束）。
    fn make_builtin_generic(&mut self, name: Box<str>, args: Box<[TypeHandle]>) -> TypeHandle {
        match name.as_ref() {
            "Throw" if args.len() == 2 => self.arena.make_throw(args[0], args[1]),
            "Channel" if args.len() == 1 => self.arena.make_channel(args[0]),
            "Async" if args.len() == 1 => self.arena.make_async(args[0]),
            "Lazy" if args.len() == 1 => self.arena.make_lazy(args[0]),
            "Atomic" if args.len() == 1 => self.arena.make_atomic(args[0]),
            "Sender" if args.len() == 1 => self.arena.make_sender(args[0]),
            "Receiver" if args.len() == 1 => self.arena.make_receiver(args[0]),
            _ => self.arena.make_generic(name, args),
        }
    }

    /// 判断表达式是否为字面量（用于 peer_type_binary 调用方判断）。
    fn expr_is_literal(ast: &AstArena<'_>, expr: ExprId) -> bool {
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

    /// 解引用 ref/nullable 类型，返回 inner；非 ref/nullable 返回原类型。
    /// SafeAccess `?.` 对 Nullable 需要解包内层类型查找字段，与方法调用对 Nullable 的解包保持一致。
    fn unwrap_ref(&self, ty: TypeHandle) -> TypeHandle {
        let resolved = self.arena.resolve(ty);
        match self.arena.get(resolved) {
            Ty::Ref(_) => self.arena.ref_parts(resolved).0,
            Ty::Nullable(_) => self.arena.nullable_inner(resolved),
            _ => resolved,
        }
    }

    /// 从迭代器类型结构化提取元素类型。
    /// 覆盖所有标准迭代器形态：
    /// - Array<T> → T（虽然数组不是迭代器，但提取元素类型用于约束）
    /// - ArrayIter<T> / Iter<T> / RangeIterator → T
    /// - Map<K,V> 的迭代器 → Entry<K,V>
    /// - Str → char
    /// - Throw<T,E> → T（直接解构，便于 For over Throw 时元素为值类型）
    /// 提取失败返回 None（调用方回退到 fresh_type_var + 约束）。
    fn extract_iterator_element(&mut self, h: TypeHandle) -> Option<TypeHandle> {
        let ty = self.arena.get(h);
        match ty {
            Ty::Array(_) => Some(self.arena.array_parts(h).0),
            Ty::Str => Some(self.make_builtin(Ty::Char)),
            Ty::Generic(_) => {
                let (name, args) = self.arena.generic_parts(h);
                match name {
                    // 标准迭代器：ArrayIter<T>、Iter<T>、RangeIterator（无 args，元素为 i64）
                    "ArrayIter" | "Iter" if args.len() == 1 => Some(args[0]),
                    "RangeIterator" => Some(self.make_builtin(Ty::I64)),
                    // Map 迭代器返回 Entry<K,V>
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
            Ty::Throw(_) => Some(self.arena.throw_parts(h).0),
            _ => None,
        }
    }
}

// =========================================================================
// phase5: InferContext 扩展 — 表达式/语句/模式推断 + 模块检查入口
//
// 移植自 `src/sema/type_check.zig` 的 inferExpr / inferStmt / inferPattern /
// registerBuiltins / checkModuleWithName。
// =========================================================================

impl<'a> InferContext<'a> {
    /// 将推断出的类型存储为 ExprInfo 到 sema_result.expr_types。
    fn store_expr_info(&mut self, expr: ExprId, ty: TypeHandle) {
        let resolved = self.arena.resolve(ty);
        let ct = self.arena.get(resolved);
        let type_name: Option<String> = self.arena.type_name(resolved).map(|s| s.to_string());
        let is_ref = matches!(ct, Ty::Ref(_));
        let is_raw_ref = matches!(ct, Ty::Ref(_)) && self.arena.ref_parts(resolved).1;

        let is_trait_object = matches!(ct, Ty::TraitObject(_));
        let info = ExprInfo {
            ty: resolved,
            const_val: None,
            expr_id: expr.0 as u64,
            type_name: type_name.map(|s| s.into_boxed_str()),
            is_trait_object,
            is_ref_type: is_ref,
            is_raw_ref,
        };
        let key = if let Some(ref ictx) = self.instantiation_ctx {
            // 实例化模式：用实例模块名计算 key，写入实例本地暂存表 + 全局 resolved_types
            module_expr_key(&ictx.module_name, expr.0 as u64)
        } else {
            // HM 模式：用当前模块名计算 key
            module_expr_key(&self.current_module_name, expr.0 as u64)
        };

        if let Some(ref mut ictx) = self.instantiation_ctx {
            ictx.local_expr_types.insert(key, info);
            self.sema_result.resolved_types.insert(key, resolved);
        } else {
            self.sema_result.put_expr(key, info);
        }
    }

    // ── infer_expr ──

    /// 推断表达式的类型。这是类型检查的核心入口，递归处理所有表达式变体。
    /// 推断完成后将 ExprInfo 存储到 sema_result。
    pub fn infer_expr(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        let ty = self.infer_expr_inner(expr, ast, env, expected);
        self.store_expr_info(expr, ty);
        // 诊断追踪：仅在 KUZO_SEMA_TRACE 启用时记录 (TypeHandle, Span)
        if std::env::var("KUZO_SEMA_TRACE").is_ok() {
            let span = ast.expr(expr).span;
            self.type_trace.push((ty, span));
        }
        ty
    }

    /// 表达式类型推断内部实现（不存储 ExprInfo）。
    fn infer_expr_inner(
        &mut self,
        expr: ExprId,
        ast: &AstArena<'_>,
        env: EnvId,
        expected: Option<TypeHandle>,
    ) -> TypeHandle {
        let node = &ast.expr(expr).node;
        match node {
            // ── 字面量 ──
            Expr::IntLit { suffix, .. } => numeric_lit!(self, suffix, expected, int_suffix_to_type, is_int, I32),
            Expr::FloatLit { suffix, .. } => numeric_lit!(self, suffix, expected, float_suffix_to_type, is_float, F64),
            Expr::BoolLit(_) => self.make_builtin(Ty::Bool),
            Expr::CharLit(_) => self.make_builtin(Ty::Char),
            Expr::StrLit(_) => self.make_builtin(Ty::Str),
            Expr::StrInterp(parts) => {
                // 递归推断插值内部的子表达式，确保其 ExprInfo 注册到 expr_types。
                // 否则 IR 编译 `select_binary_compute_fn` 因查不到类型回退到 "i32"，
                // 对 bool/str 等非整数类型误分派到 CF_EQ_I32（as_i32 对 bool 恒为 0）。
                for p in parts {
                    if let InterpolationPart::Expression(e) = p {
                        let _ = self.infer_expr(*e, ast, env, None);
                    }
                }
                self.make_builtin(Ty::Str)
            }
            Expr::NullLit => {
                // null 字面量类型为 Nullable<T>，T 通过 expected 约束求解。
                // try_widen_unify 处理所有 expected 类型（Nullable<T> 统一 inner，
                // 其他类型尝试 widen 或报错），无需对 expected 做类型特判。
                let tv = self.arena.fresh_type_var();
                let ty = self.arena.make_nullable(tv);
                if let Some(exp) = expected {
                    if let Err(e) = self.try_widen_unify(exp, ty) {
                        self.add_error(&format!("null literal incompatible with expected type: {}", e));
                    }
                }
                ty
            }
            Expr::VoidLit => self.make_builtin(Ty::Void),

            // ── 标识符 ──
            Expr::Ident(name) => {
                // sema v2: 优先查询 flow narrowing 结果（path-sensitive 类型精化）
                if let Some(narrowed_ty) = self.flow_ctx.lookup_narrowed(name) {
                    return narrowed_ty;
                }
                if let Some(scheme) = self.env.lookup(env, name) {
                    return self.freshen_type(scheme);
                }
                // 实例化模式：临时 InferContext 的 env 不含模块级声明，
                // 从 sema_result 查询（HM 阶段已解析）
                if self.instantiation_ctx.is_some() {
                    // 从 expr_types 查询（HM 阶段已解析该表达式的类型）
                    let key = module_expr_key(&self.current_module_name, expr.0 as u64);
                    if let Some(info) = self.sema_result.get_expr(key) {
                        return info.ty;
                    }
                    // 实例化模式下不报错，返回 fresh_type_var
                    return self.arena.fresh_type_var();
                }
                let span = ast.expr(expr).span;
                self.add_error_at(&format!("undefined variable '{}'", name), span.line, span.column);
                self.arena.fresh_type_var()
            }

            // ── 赋值 ──
            Expr::Assign { target, value } => {
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                self.unify_or_constrain(target_ty, val_ty);
                self.make_builtin(Ty::Void)
            }
            Expr::CompoundAssign { target, value, .. } => {
                let target_ty = self.infer_expr(*target, ast, env, None);
                let val_ty = self.infer_expr(*value, ast, env, Some(target_ty));
                self.unify_or_constrain(target_ty, val_ty);
                target_ty
            }

            // ── 二元运算 ──
            Expr::Binary { op, lhs, rhs } => {
                let left_ty = self.infer_expr(*lhs, ast, env, None);
                let right_ty = self.infer_expr(*rhs, ast, env, None);
                let left_is_lit = Self::expr_is_literal(ast, *lhs);
                let right_is_lit = Self::expr_is_literal(ast, *rhs);
                match op {
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                        let rl = self.arena.resolve(left_ty);
                        let rr = self.arena.resolve(right_ty);
                        if self.arena.get(rl).is_numeric() && self.arena.get(rr).is_numeric() {
                            // v2 收敛：用 peer_type_binary 替代 literal_promotion
                            // 字面量提升规则内化到 peer_type_binary 中
                            return peer_type_binary(
                                self.arena,
                                left_ty,
                                right_ty,
                                left_is_lit,
                                right_is_lit,
                            );
                        }
                        self.unify_or_constrain(left_ty, right_ty);
                        left_ty
                    }
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::RefEq | BinaryOp::RefNeq
                    | BinaryOp::Lt | BinaryOp::Gt | BinaryOp::LtEq | BinaryOp::GtEq => {
                        let rl = self.arena.resolve(left_ty);
                        let rr = self.arena.resolve(right_ty);
                        if self.arena.get(rl).is_numeric() && self.arena.get(rr).is_numeric() {
                            // v2 收敛：比较运算用 peer_type_binary 统一操作数类型
                            let _ = peer_type_binary(
                                self.arena,
                                left_ty,
                                right_ty,
                                left_is_lit,
                                right_is_lit,
                            );
                        } else {
                            self.unify_or_constrain(left_ty, right_ty);
                        }
                        self.make_builtin(Ty::Bool)
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        let bool_ty = self.make_builtin(Ty::Bool);
                        self.unify_or_constrain(left_ty, bool_ty);
                        self.unify_or_constrain(right_ty, bool_ty);
                        bool_ty
                    }
                    BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor
                    | BinaryOp::Shl | BinaryOp::Shr => {
                        self.unify_or_constrain(left_ty, right_ty);
                        left_ty
                    }
                    BinaryOp::ConcatList => {
                        // 数组拼接 a ++ b：左右元素类型必须一致，结果复用左操作数元素类型
                        // 避免创建孤立 fresh_type_var（res_elem 与输入无约束）
                        let left_elem = self.arena.fresh_type_var();
                        let left_arr = self.arena.make_array(left_elem, None);
                        self.unify_or_constrain(left_ty, left_arr);
                        let right_arr = self.arena.make_array(left_elem, None);
                        self.unify_or_constrain(right_ty, right_arr);
                        self.arena.make_array(left_elem, None)
                    }
                    BinaryOp::Range | BinaryOp::RangeInclusive => {
                        // Range 表达式 a..b / a..=b 返回 RangeIterator 类型
                        // （Range 本身是迭代器，For 循环通过 RangeIterator.next 静态分派）
                        let i64_ty = self.make_builtin(Ty::I64);
                        if let Err(e) = self.try_widen_unify(i64_ty, left_ty) {
                            self.add_error(&format!("range operand must be integer: {}", e));
                        }
                        let i64_ty = self.make_builtin(Ty::I64);
                        if let Err(e) = self.try_widen_unify(i64_ty, right_ty) {
                            self.add_error(&format!("range operand must be integer: {}", e));
                        }
                        self.arena.make_generic(
                            "RangeIterator".into(),
                            Box::new([]),
                        )
                    }
                    BinaryOp::Elvis => {
                        let rl = self.arena.resolve(left_ty);
                        if let Ty::Nullable(_) = self.arena.get(rl) {
                            return self.arena.nullable_inner(rl);
                        }
                        // Throw<T,E> ?? rhs → 返回 T（Ok 值类型），与 Nullable 对称（Bug #28）
                        if let Ty::Throw(_) = self.arena.get(rl) {
                            let value_ty = self.arena.throw_parts(rl).0;
                            // unify rhs 到 value_ty，确保默认值类型兼容
                            if let Err(e) = self.try_widen_unify(value_ty, right_ty) {
                                self.add_error(&format!("?? default value incompatible with Throw value type: {}", e));
                            }
                            return value_ty;
                        }
                        left_ty
                    }
                }
            }

            // ── 一元运算 ──
            Expr::Unary { operand, .. } => {
                let _ = self.infer_expr(*operand, ast, env, None);
                // ! / ~ / - 均返回操作数类型
                self.infer_expr(*operand, ast, env, None)
            }

            // ── 引用/解引用 ──
            Expr::RefOf(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_ref(inner_ty, false)
            }
            Expr::Deref(operand) => {
                let operand_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(operand_ty);
                match self.arena.get(resolved) {
                    Ty::Ref(_) => self.arena.ref_parts(resolved).0,
                    _ => operand_ty, // 非引用解引用：返回原类型
                }
            }

            // ── 函数调用 ──
            Expr::Call { callee, args, type_args } => {
                // cast 调用解析：__cast_to<T>(x) / __cast_try_to<T>(x)
                // parser 将 cast(x).to(T) 降级为 __cast_to<T>(x) 普通 Call，
                // sema 推断源类型 S，返回 T（或 Throw<T, CastError> for try_to）
                // 通过 CAST_BUILTINS 注册表查表，避免函数名特判分支
                if let Expr::Ident(name) = &ast.expr(*callee).node {
                    if let Some(is_try) = CAST_BUILTINS
                        .iter()
                        .find_map(|(n, t)| (*n == *name).then_some(*t))
                    {
                        // 推断源表达式类型
                        let _ = self.infer_expr(args[0], ast, env, None);
                        // 从 type_args 取目标类型 T
                        let target_ty = match type_args {
                            Some(ta) if !ta.is_empty() => self.type_from_ast(ta[0], ast),
                            _ => self.arena.fresh_type_var(),
                        };
                        if is_try {
                            let err_ty = self.arena.make_adt(
                                "CastError".into(),
                                Box::new([]),
                            );
                            return self.arena.make_throw(target_ty, err_ty);
                        }
                        return target_ty;
                    }
                }

                let callee_ty = self.infer_expr(*callee, ast, env, None);
                let resolved_callee = self.arena.resolve(callee_ty);

                // 实例化模式：跳过 HM unify（类型已在 sema HM 阶段检查），
                // 仅推断参数类型并返回返回类型。单态化触发由外部编排。
                if self.instantiation_ctx.is_some() {
                    // ModuleRef 调用：从模块 env 查找函数签名
                    if let Ty::ModuleRef(_) = self.arena.get(resolved_callee) {
                        let (path, module_env) = self.arena.module_ref_parts(resolved_callee);
                        if let Some(func_name) = path.rsplit('.').next() {
                            if let Some(fn_ty) = self.env.lookup_local(module_env, func_name) {
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Ty::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                        let _ = self.infer_expr(arg, ast, env, Some(param_ty));
                                    }
                                    return return_type;
                                }
                            }
                        }
                    }
                    // 普通函数调用：推断参数类型，返回返回类型
                    let inst_callee = self.instantiate_fn_type(resolved_callee);
                    if let Ty::Fn(_) = self.arena.get(inst_callee) {
                        let (params, return_type) = self.arena.fn_parts(inst_callee);
                        let params: Vec<TypeHandle> = params.to_vec();
                        for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                            let _ = self.infer_expr(arg, ast, env, Some(param_ty));
                        }
                        return return_type;
                    }
                    // 非 Fn 类型的 callee：报错并返回 Unknown
                    let span = ast.expr(expr).span;
                    let callee_name = self
                        .arena
                        .type_name(resolved_callee)
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("{:?}", self.arena.get(resolved_callee)));
                    self.add_error_at(
                        &format!("cannot call non-function value of type '{}'", callee_name),
                        span.line,
                        span.column,
                    );
                    for &a in args.iter() {
                        let _ = self.infer_expr(a, ast, env, None);
                    }
                    return self.arena.make(Ty::Unknown);
                }

                // ModuleRef 调用：callee 是模块路径引用（如 "std.reflect.Reflect.format"），
                // 直接在 ModuleRef 携带的模块 env 中按末段裸名查找函数签名（不穿透父 env）
                if let Ty::ModuleRef(_) = self.arena.get(resolved_callee) {
                    let (path, module_env) = self.arena.module_ref_parts(resolved_callee);
                    // 末段即函数名（如 "std.reflect.Reflect.format" → "format"）
                    if let Some(func_name) = path.rsplit('.').next() {
                        if let Some(fn_ty) = self.env.lookup_local(module_env, func_name) {
                            // 实例化多态函数类型，避免不同调用的类型约束冲突
                            let inst_fn = self.instantiate_fn_type(fn_ty);
                            if let Ty::Fn(_) = self.arena.get(inst_fn) {
                                let (params, return_type) = self.arena.fn_parts(inst_fn);
                                let params: Vec<TypeHandle> = params.to_vec();
                                if params.len() == args.len() {
                                    for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                                        let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                                        if let Err(e) = self.try_widen_unify(param_ty, arg_ty) {
                                            self.add_error(&format!("argument type incompatible with parameter type: {}", e));
                                        }
                                    }
                                    return return_type;
                                }
                            }
                        }
                    }
                }

                // 实例化多态函数类型（将 rigid var / 未绑定 TypeVar 替换为 fresh non-rigid var），
                // 使每次调用拥有独立的类型变量，避免不同调用的类型约束相互冲突
                let inst_callee = self.instantiate_fn_type(resolved_callee);
                if let Ty::Fn(_) = self.arena.get(inst_callee) {
                    let (params, return_type) = self.arena.fn_parts(inst_callee);
                    let params: Vec<TypeHandle> = params.to_vec();
                    if params.len() == args.len() {
                        for (&param_ty, &arg) in params.iter().zip(args.iter()) {
                            let arg_ty = self.infer_expr(arg, ast, env, Some(param_ty));
                            // unify 失败时注册约束（而非丢弃），让不动点迭代求解参数类型
                            self.unify_or_constrain(param_ty, arg_ty);
                        }
                    }
                    // 始终返回声明的返回类型，避免参数不匹配导致级联类型丢失
                    // 若有 expected 类型，unify 返回类型与 expected，求解返回类型中的未决 TypeVar
                    // （如 Ok(void) 返回 Throw<void, '_E>，expected=Throw<void, IOError> 可求解 E=IOError）
                    if let Some(exp) = expected {
                        self.unify_or_constrain(return_type, exp);
                    }
                    return return_type;
                }
                // 兜底：推断所有参数，unify callee 与 (args -> ret)
                let ret_ty = self.arena.fresh_type_var();
                let arg_types: Vec<TypeHandle> = args
                    .iter()
                    .map(|&a| self.infer_expr(a, ast, env, None))
                    .collect();
                let expected_fn = self.arena.make_fn(
                    arg_types.into_boxed_slice(),
                    ret_ty,
                );
                self.unify_or_constrain(callee_ty, expected_fn);
                ret_ty
            }

            // ── 方法调用 ──
            Expr::MethodCall { recv, method, args, .. }
            | Expr::SafeMethodCall { recv, method, args, .. } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);

                // 路径 0a：ModuleRef recv → 模块路径函数调用
                // 当 recv 是 ModuleRef（如 std.net.UdpSocket）时，method 是模块中的顶层函数，
                // 直接在 ModuleRef 携带的模块 env 中按 method 裸名查找（不穿透父 env）。
                let recv_resolved_0a = self.arena.resolve(recv_ty);
                if let Ty::ModuleRef(_) = self.arena.get(recv_resolved_0a) {
                    let (mod_path, module_env) = self.arena.module_ref_parts(recv_resolved_0a);
                    let found = self.env.lookup_local(module_env, method);
                    // 目录模块语义：当 lookup_local 在当前模块 env 未命中时，
                    // 搜索同目录兄弟模块的 env（如 Math.sqrt 中 sqrt 在 Power.kz，
                    // Math 与 Power 同属 std.math 目录）。
                    let found = found.or_else(|| {
                        self.lookup_sibling_module_fn(mod_path, module_env, method)
                    });
                    if let Some(fn_ty) = found {
                        let inst_fn = self.instantiate_fn_type(fn_ty);
                        if let Ty::Fn(_) = self.arena.get(inst_fn) {
                            let (params, return_type) = self.arena.fn_parts(inst_fn);
                            let params: Vec<TypeHandle> = params.to_vec();
                            let n = params.len().min(args.len());
                            for i in 0..n {
                                let arg_ty = self.infer_expr(args[i], ast, env, Some(params[i]));
                                self.unify_or_constrain(params[i], arg_ty);
                            }
                            // 标记 recv 为模块函数调用接收者，IR 编译时不传 recv
                            //（与路径 0b 一致：ModuleRef recv 的 Module.fun(args) 语义）
                            let recv_key = module_expr_key(
                                &self.current_module_name,
                                recv.0 as u64,
                            );
                            self.sema_result.module_func_recv_exprs.insert(recv_key);
                            return return_type;
                        }
                    }
                }

                // 路径 0b：构造器 recv（类型名 == 模块名）→ 模块函数调用（Zig @This 语义）
                // 当 recv 是类型构造器（Fn，return_type 为 Adt）且类型名与某模块同名时，
                // 在该模块 env 中按 method 裸名查找自由函数。
                // 典型场景：import std.time.Duration 后 Duration.from_millis(100)，
                // 其中 Duration 既是类型又是模块（文件同名，predefine redefine 覆盖了 ModuleRef）。
                if let Ty::Fn(_) = self.arena.get(recv_resolved_0a) {
                    let (_, ret_ty) = self.arena.fn_parts(recv_resolved_0a);
                    let ret_resolved = self.arena.resolve(ret_ty);
                    if let Ty::Adt(_) = self.arena.get(ret_resolved) {
                        let (type_name, _) = self.arena.adt_parts(ret_resolved);
                        if let Some(&mod_env) = self.ctor_module_envs.get(type_name) {
                            if let Some(fn_ty) = self.env.lookup_local(mod_env, method) {
                                let inst_fn = self.instantiate_fn_type(fn_ty);
                                if let Ty::Fn(_) = self.arena.get(inst_fn) {
                                    let (params, return_type) = self.arena.fn_parts(inst_fn);
                                    let params: Vec<TypeHandle> = params.to_vec();
                                    let n = params.len().min(args.len());
                                    for i in 0..n {
                                        let arg_ty = self.infer_expr(args[i], ast, env, Some(params[i]));
                                        self.unify_or_constrain(params[i], arg_ty);
                                    }
                                    // 标记 recv 为模块函数调用接收者，IR 编译时不传 recv
                                    let recv_key = module_expr_key(
                                        &self.current_module_name,
                                        recv.0 as u64,
                                    );
                                    self.sema_result.module_func_recv_exprs.insert(recv_key);
                                    return return_type;
                                }
                            }
                        }
                    }
                }

                // 语言级 intrinsic 标记：await/recv 由 sema 统一识别，
                // 注册到 method_dispatches 供 IR 消费（消除 IR 侧字符串守卫）。
                // await 是通用挂起语义（对所有类型）；recv 仅对 Channel/Receiver 类型标记。
                {
                    let intrinsic = if *method == "await" && args.is_empty() {
                        Some(crate::sema::Sema::IntrinsicKind::Await)
                    } else if *method == "recv" && args.is_empty() {
                        match self.arena.get(recv_resolved_0a) {
                            Ty::Channel(_) | Ty::Receiver(_) => {
                                Some(crate::sema::Sema::IntrinsicKind::ChannelAwait)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    if intrinsic.is_some() {
                        let key = crate::sema::Sema::module_expr_key(
                            &self.current_module_name,
                            expr.0 as u64,
                        );
                        self.sema_result.method_dispatches.insert(
                            key,
                            crate::sema::Sema::DispatchInfo {
                                trait_id: 0,
                                method_idx: 0,
                                impl_fn_idx: 0,
                                instance_id: 0,
                                intrinsic,
                            },
                        );
                    }
                }

                // 路径 1（优先）：类型感知的方法查找
                // 通过 lookup_method_type 按接收者类型查 witness_table / func_sigs / 内置方法，
                // 确保同名方法（如 Instant.add_duration 与 DateTime.add_duration）分派到正确签名。
                let method_fn_ty = self.lookup_method_type(recv_ty, method);
                if let Some(fn_ty) = method_fn_ty {
                    let inst_fn = self.instantiate_fn_type(fn_ty);
                    if let Ty::Fn(_) = self.arena.get(inst_fn) {
                        let (params, return_type) = self.arena.fn_parts(inst_fn);
                        let params: Vec<TypeHandle> = params.to_vec();
                        // 第一个参数是 self，跳过
                        let n = params.len().min(args.len() + 1);
                        for i in 1..n {
                            let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                            self.unify_or_constrain(params[i], arg_ty);
                        }
                        return return_type;
                    }
                }

                // 路径 0（回退）：env 中查找方法名为 Fn 类型的绑定（free function with self 参数）
                // 使用 lookup_with_pred 跳过同名的非函数绑定（如局部变量遮蔽自由函数）。
                // Kuzo 中 `recv.method(args)` 是 `method(recv, args)` 的语法糖
                if let Some(fn_ty) = self.env.lookup_with_pred(env, method, |ty| {
                    let r = self.arena.resolve(ty);
                    matches!(self.arena.get(r), Ty::Fn(_))
                }) {
                    let inst_fn = self.instantiate_fn_type(fn_ty);
                    if let Ty::Fn(_) = self.arena.get(inst_fn) {
                        let (params, return_type) = self.arena.fn_parts(inst_fn);
                        let params: Vec<TypeHandle> = params.to_vec();
                        // 第一个参数是 self/接收者：unify recv 与 params[0]
                        // 这样自由函数泛型参数能从接收者类型推断（如 iter<T> 从 arr: T[] 推断 T）
                        if !params.is_empty() {
                            self.unify_or_constrain(params[0], recv_ty);
                        }
                        // 其余参数从 args 推断
                        let n = params.len().min(args.len() + 1);
                        for i in 1..n {
                            let arg_ty = self.infer_expr(args[i - 1], ast, env, Some(params[i]));
                            self.unify_or_constrain(params[i], arg_ty);
                        }
                        return return_type;
                    }
                }

                // await 是通用挂起语义：不产生值，仅挂起帧等待事件。
                // IR 层通过 infer_event_source_kind 根据 recv 类型决定事件源种类
                // （AsyncJoin/Channel/Timer），Sema 层统一返回 void。
                if *method == "await" && args.is_empty() {
                    return self.make_builtin(Ty::Void);
                }

                // 兜底：推断参数，返回 fresh var
                // 对已确定类型的接收者（非 TypeVar/Unknown/Never）报"方法不存在"，
                // 帮助用户定位问题；对 TypeVar 接收者静默返回 fresh var（推断未决，延迟到 solver）
                let span = ast.expr(expr).span;
                let recv_resolved = self.arena.resolve(recv_ty);
                match self.arena.get(recv_resolved) {
                    Ty::TypeVar(_) | Ty::Unknown | Ty::Never => {
                        // 接收者类型未决，静默返回 fresh var
                    }
                    Ty::Void => {
                        // void 接收者：IR 层处理（void 方法调用）
                    }
                    ct => {
                        // 接收者类型已确定但方法查找失败：报错
                        let recv_name = self.arena.type_name(recv_resolved)
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("{:?}", ct));
                        self.add_error_at(
                            &format!("no method '{}' on type '{}'", method, recv_name),
                            span.line,
                            span.column,
                        );
                    }
                }
                for &a in args.iter() {
                    let _ = self.infer_expr(a, ast, env, None);
                }
                self.arena.fresh_type_var()
            }

            // ── 字段访问 ──
            Expr::FieldAccess { recv, field } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                // 检测 ModuleRef 接收者：Math.PI 这样的跨模块常量访问。
                // 命中时把 recv 的 expr key → mangled 名（module_path.field）记入
                // module_const_recv_exprs，供 IR 编译时跳过 recv 直接发 global_load。
                let recv_resolved = self.arena.resolve(recv_ty);
                if let Ty::ModuleRef(_) = self.arena.get(recv_resolved) {
                    let (path, module_env) = self.arena.module_ref_parts(recv_resolved);
                    if self.env.lookup_local(module_env, field).is_some() {
                        let mangled = format!("{}.{}", path, field);
                        let recv_key = crate::sema::Sema::module_expr_key(
                            &self.current_module_name,
                            recv.0 as u64,
                        );
                        self.sema_result.module_const_recv_exprs.insert(recv_key, mangled);
                    }
                }
                let span = ast.expr(expr).span;
                self.lookup_field_type(recv_ty, field, span.line, span.column)
            }
            Expr::SafeAccess { recv, field } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let resolved = self.arena.resolve(recv_ty);
                // SafeAccess `?.` 仅对 Nullable/Ref 有意义；对其他类型退化为普通字段访问
                let is_nullable = matches!(self.arena.get(resolved), Ty::Nullable(_));
                let inner = self.unwrap_ref(recv_ty);
                let span = ast.expr(expr).span;
                let field_ty = self.lookup_field_type(inner, field, span.line, span.column);
                // Nullable 接收者的字段访问结果也应是 Nullable（传播 None 语义）
                if is_nullable {
                    self.arena.make_nullable(field_ty)
                } else {
                    field_ty
                }
            }

            // ── 索引/切片 ──
            Expr::Index { recv, index } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let _ = self.infer_expr(*index, ast, env, None);
                let resolved = self.arena.resolve(recv_ty);
                match self.arena.get(resolved) {
                    Ty::Array(_) => self.arena.array_parts(resolved).0,
                    // Str 索引返回 Char（stdlib 中 normalized[0] == '/' 等用法）
                    Ty::Str => self.arena.make(Ty::Char),
                    // Unknown/TypeVar/Generic/Adt 等不报错：
                    // sema v2 对部分变量类型推断不精确（如 u8[] 可能被 unify 为 Unknown），
                    // 在 sema 类型推断完善前保守放行避免级联误报
                    _ => self.arena.fresh_type_var(),
                }
            }
            Expr::Slice { recv, start, end, .. } => {
                let recv_ty = self.infer_expr(*recv, ast, env, None);
                let _ = self.infer_expr(*start, ast, env, None);
                let _ = self.infer_expr(*end, ast, env, None);
                recv_ty // 切片返回同类型
            }

            // ── 传播 ──
            Expr::Propagate(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(inner_ty);
                let span = ast.expr(expr).span;
                self.check_propagate(resolved, inner_ty, self.expected_return, span.line, span.column)
            }
            Expr::NonNullAssert(operand) => {
                let operand_ty = self.infer_expr(*operand, ast, env, None);
                let resolved = self.arena.resolve(operand_ty);
                match self.arena.get(resolved) {
                    Ty::Nullable(_) => self.arena.nullable_inner(resolved),
                    _ => operand_ty,
                }
            }
            Expr::Elvis { lhs, rhs } => {
                let left_ty = self.infer_expr(*lhs, ast, env, None);
                let right_ty = self.infer_expr(*rhs, ast, env, None);
                let rl = self.arena.resolve(left_ty);
                if let Ty::Nullable(_) = self.arena.get(rl) {
                    let inner = self.arena.nullable_inner(rl);
                    if let Err(e) = self.try_widen_unify(inner, right_ty) {
                        self.add_error(&format!("?? default value incompatible with Nullable inner type: {}", e));
                    }
                    inner
                } else if let Ty::Throw(_) = self.arena.get(rl) {
                    // Throw<T,E> ?? rhs → 返回 T，与 Nullable 对称（Bug #28）
                    let value_ty = self.arena.throw_parts(rl).0;
                    if let Err(e) = self.try_widen_unify(value_ty, right_ty) {
                        self.add_error(&format!("?? default value incompatible with Throw value type: {}", e));
                    }
                    value_ty
                } else {
                    left_ty
                }
            }

            // ── 数组字面量 ──
            Expr::ArrayLit { elements, .. } => {
                // 从 expected 提取元素类型，使字面量元素能按注解提升
                // （例如 `val data: u8[] = [72, 101]` 中 72 应提升为 u8 而非默认 i32）
                let expected_elem = expected.and_then(|exp| {
                    let r = self.arena.resolve(exp);
                    match self.arena.get(r) {
                        Ty::Array(_) => Some(self.arena.array_parts(r).0),
                        _ => None,
                    }
                });
                if elements.is_empty() {
                    let elem_ty = expected_elem.unwrap_or_else(|| self.arena.fresh_type_var());
                    return self.arena.make_array(elem_ty, None);
                }
                let first_ty = self.infer_expr(elements[0], ast, env, expected_elem);
                for &e in elements.iter().skip(1) {
                    let elem_ty = self.infer_expr(e, ast, env, expected_elem);
                    if let Err(e_err) = self.try_widen_unify(first_ty, elem_ty) {
                        self.add_error(&format!("array element type mismatch: {}", e_err));
                    }
                }
                self.arena.make_array(first_ty, Some(elements.len() as u64))
            }

            // ── 记录字面量 ──
            Expr::RecordLit(fields) => {
                let field_types: Vec<FieldType> = fields
                    .iter()
                    .map(|f| FieldType {
                        name: Some(f.name.into()),
                        ty: self.infer_expr(f.value, ast, env, None),
                    })
                    .collect();
                self.arena.make_record(field_types.into_boxed_slice(), None)
            }
            Expr::RecordExtend { base, updates } => {
                let base_ty = self.infer_expr(*base, ast, env, None);
                let resolved = self.arena.resolve(base_ty);
                match self.arena.get(resolved) {
                    Ty::Record(_) => {
                        let base_fields = self.arena.record_fields(resolved);
                        let name = self.arena.record_name(resolved).map(|s| s.into());
                        let mut all_fields: Vec<FieldType> = base_fields.to_vec();
                        for update in updates.iter() {
                            let update_ty = self.infer_expr(update.value, ast, env, None);
                            let mut found = false;
                            for f in all_fields.iter_mut() {
                                if f.name.as_deref() == Some(update.name) {
                                    f.ty = update_ty;
                                    found = true;
                                    break;
                                }
                            }
                            if !found {
                                all_fields.push(FieldType {
                                    name: Some(update.name.into()),
                                    ty: update_ty,
                                });
                            }
                        }
                        self.arena.make_record(all_fields.into_boxed_slice(), name)
                    }
                    _ => {
                        let span = ast.expr(expr).span;
                        self.add_error_at("record extend requires record type", span.line, span.column);
                        self.arena.fresh_type_var()
                    }
                }
            }

            // ── Lambda ──
            Expr::Lambda { params, body, is_async, return_type } => {
                let child_env = self.env.child(env);
                let param_types: Vec<TypeHandle> = params
                    .iter()
                    .map(|p| {
                        let param_ty = match p.type_annotation {
                            Some(ta) => self.type_from_ast(ta, ast),
                            None => self.arena.fresh_type_var(),
                        };
                        self.env.define(child_env, p.name, param_ty);
                        param_ty
                    })
                    .collect();
                let body_ty = match body {
                    LambdaBody::Block(b) => self.infer_expr(*b, ast, child_env, None),
                    LambdaBody::Expression(e) => self.infer_expr(*e, ast, child_env, None),
                };
                let effective_body_ty = if let Some(rt) = return_type {
                    let annot_ty = self.type_from_ast(*rt, ast);
                    if let Err(e) = self.try_widen_unify(annot_ty, body_ty) {
                        self.add_error(&format!("lambda body type incompatible with declared return type: {}", e));
                    }
                    annot_ty
                } else {
                    body_ty
                };
                let ret_ty = if *is_async {
                    self.arena.make_async(effective_body_ty)
                } else {
                    effective_body_ty
                };
                self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
            }

            // ── if 表达式 ──
            Expr::If { cond, then_branch, else_branch } => {
                let cond_ty = self.infer_expr(*cond, ast, env, None);
                let bool_ty = self.make_builtin(Ty::Bool);
                self.unify_or_constrain(cond_ty, bool_ty);

                // sema v2: 提取 flow facts（nullable narrowing）
                let (then_facts, else_facts) = analyze_null_check_facts(
                    self.arena,
                    ast,
                    *cond,
                    env,
                    &self.env,
                );

                let then_env = self.env.child(env);
                // 进入 then scope，应用 then facts
                self.flow_ctx.push_scope();
                for fact in &then_facts {
                    self.flow_ctx.add_fact(fact.clone());
                }
                let then_ty = self.infer_expr(*then_branch, ast, then_env, expected);
                self.flow_ctx.pop_scope();

                if let Some(else_br) = else_branch {
                    let else_env = self.env.child(env);
                    // 进入 else scope，应用 else facts
                    self.flow_ctx.push_scope();
                    for fact in &else_facts {
                        self.flow_ctx.add_fact(fact.clone());
                    }
                    let else_ty = self.infer_expr(*else_br, ast, else_env, expected);
                    self.flow_ctx.pop_scope();

                    // v2 收敛：只用 peer_type 统一分支类型（消除 try_widen_unify 双轨制）
                    // peer_type 已内化 Never/Void 过滤、数值宽化、nullable/throw 传播
                    peer_type(self.arena, &[then_ty, else_ty])
                } else {
                    then_ty
                }
            }

            // ── 块表达式 ──
            Expr::Block { stmts, trailing } => {
                let child_env = self.env.child(env);
                let mut diverges = false;
                for &stmt in stmts.iter() {
                    let _ = self.infer_stmt(stmt, ast, child_env);
                    match &ast.stmt(stmt).node {
                        Stmt::Return { .. } | Stmt::Throw { .. } => diverges = true,
                        _ => {}
                    }
                }
                if let Some(te) = trailing {
                    self.infer_expr(*te, ast, child_env, expected)
                } else if diverges {
                    self.make_builtin(Ty::Never)
                } else {
                    self.make_builtin(Ty::Void)
                }
            }

            // ── match 表达式 ──
            Expr::Match { scrutinee, arms } => {
                let scrutinee_ty = self.infer_expr(*scrutinee, ast, env, None);
                let resolved_scrutinee = self.arena.resolve(scrutinee_ty);

                // Throw 类型 widening：当 match 同时包含 Ok 和 Error/错误构造器模式时，
                // scrutinee 应为 Throw<T, E>。若 scrutinee 是 Err 实现者（如 Error ADT）
                // 而非 Throw，将其 widen 为 Throw<fresh, scrutinee_ty>（scrutinee 作为 error_type），
                // 使 Ok(v) 匹配值变体、Error(e) 匹配错误变体并绑定整个错误值（而非构造器字段）。
                let resolved_scrutinee = {
                    let is_throw = matches!(
                        self.arena.get(resolved_scrutinee),
                        Ty::Throw(_)
                    );
                    let has_ok_arm = arms.iter().any(|arm| {
                        match &ast.pattern(arm.pattern).node {
                            Pattern::Constructor { name, .. } => *name == crate::ir::Compute::CTOR_OK,
                            Pattern::Variable { name } => *name == crate::ir::Compute::CTOR_OK,
                            _ => false,
                        }
                    });
                    if !is_throw && has_ok_arm {
                        // 检查 scrutinee 是否实现 Err trait（通过 witness table）
                        let implements_err = self.arena.type_name(resolved_scrutinee)
                            .and_then(|tn| self.sema_result.type_def_index.get(tn).copied())
                            .map(|idx| self.witness_table.implements("Err", dynamic_type_id(idx)))
                            .unwrap_or(false);
                        let widened = if implements_err {
                            // scrutinee 是错误类型 → Throw<fresh_val, scrutinee>
                            let fresh_val = self.arena.fresh_type_var();
                            self.arena.make_throw(fresh_val, resolved_scrutinee)
                        } else {
                            // scrutinee 是值类型 → Throw<scrutinee, fresh_err>
                            let fresh_err = self.arena.fresh_type_var();
                            self.arena.make_throw(resolved_scrutinee, fresh_err)
                        };
                        self.unify_or_constrain(scrutinee_ty, widened);
                        self.arena.resolve(widened)
                    } else {
                        resolved_scrutinee
                    }
                };

                // sema v2: 提取 scrutinee 的路径（用于 ConstructorMatch narrowing）
                let scrutinee_path = expr_path(ast, *scrutinee);

                let mut arm_tys: Vec<TypeHandle> = Vec::new();
                for arm in arms.iter() {
                    let child_env = self.env.child(env);

                    // sema v2: 进入 match arm scope，应用 ConstructorMatch narrowing
                    self.flow_ctx.push_scope();
                    if let Some(ref path) = scrutinee_path {
                        // 检查是否为构造器模式，若是则添加 ConstructorMatch fact
                        if let Some((ctor_name, bound_vars)) =
                            extract_constructor_pattern(&ast.pattern(arm.pattern).node, ast)
                        {
                            // 构造器匹配：scrutinee 被窄化为该构造器类型
                            let narrowed_ty = self.arena.make_adt(
                                ctor_name.into(),
                                Box::new([]),
                            );
                            self.flow_ctx.add_fact(FlowFact {
                                path: path.clone().into(),
                                narrowed_ty,
                                kind: NarrowKind::ConstructorMatch {
                                    ctor_name: ctor_name.into(),
                                    bound_vars: bound_vars.into(),
                                },
                            });
                        }
                    }

                    self.infer_pattern(arm.pattern, ast, resolved_scrutinee, child_env);
                    if let Some(guard) = arm.guard {
                        let _ = self.infer_expr(guard, ast, child_env, None);
                    }
                    // 将 match 的 expected 类型传播给 arm body，
                    // 使 NullLit 等依赖 expected 约束的表达式能正确推导
                    let body_ty = self.infer_expr(arm.body, ast, child_env, expected);
                    self.flow_ctx.pop_scope();

                    arm_tys.push(body_ty);
                }

                // v2 收敛：只用 peer_type 统一所有 arm 类型（消除逐个 widen 双轨制）
                // peer_type 处理单 arm（直接返回）、多 arm（join）、全 Never/Void（返回 Never/Void）
                let result_ty = if arm_tys.is_empty() {
                    self.make_builtin(Ty::Void)
                } else {
                    peer_type(self.arena, &arm_tys)
                };
                // match 结果与外层 expected 约束，使 match 作为 let RHS 时
                // 能反向求解结果类型中的未决 TypeVar
                if let Some(exp) = expected {
                    self.unify_or_constrain(result_ty, exp);
                }
                result_ty
            }

            // ── Atomic / Lazy ──
            Expr::Atomic(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_atomic(inner_ty)
            }
            Expr::Lazy(operand) => {
                let inner_ty = self.infer_expr(*operand, ast, env, None);
                self.arena.make_lazy(inner_ty)
            }

            // ── select 表达式：Go 风格 channel 多路复用 ──
            //
            // 遍历所有 arms：
            //   receive 分支：创建子 env，从 channel_expr 推断 Channel<T>，
            //                 提取元素类型 T 给 binding（若有），推断 body 类型
            //   timeout 分支：直接推断 body 类型
            // 用 peer_type join 所有 body 类型（与 Match 一致，比 Zig 侧只取首个更健壮）
            Expr::Select(arms) => {
                let mut arm_tys: Vec<TypeHandle> = Vec::new();
                for arm in arms.iter() {
                    let child_env = self.env.child(env);
                    self.flow_ctx.push_scope();
                    match arm {
                        crate::ast::Ast::SelectArm::Receive { channel_expr, binding, body } => {
                            // 推断 channel 表达式类型，提取元素类型给 binding
                            let chan_ty = self.infer_expr(*channel_expr, ast, child_env, None);
                            let resolved = self.arena.resolve(chan_ty);
                            let elem_ty = match self.arena.get(resolved) {
                                // Nullable(Channel<T>) → 取 Channel 的 T
                                Ty::Nullable(_) => {
                                    let inner = self.arena.nullable_inner(resolved);
                                    let inner_resolved = self.arena.resolve(inner);
                                    match self.arena.get(inner_resolved) {
                                        Ty::Channel(_) => self.arena.channel_elem(inner_resolved),
                                        _ => chan_ty,
                                    }
                                }
                                // Channel<T> → 取 T
                                Ty::Channel(_) => self.arena.channel_elem(resolved),
                                _ => chan_ty,
                            };
                            if let Some(name) = binding {
                                let _ = self.env.define(child_env, name, elem_ty);
                            }
                            let body_ty = self.infer_expr(*body, ast, child_env, None);
                            arm_tys.push(body_ty);
                        }
                        crate::ast::Ast::SelectArm::Timeout { body, .. } => {
                            let body_ty = self.infer_expr(*body, ast, child_env, None);
                            arm_tys.push(body_ty);
                        }
                    }
                    self.flow_ctx.pop_scope();
                }
                if arm_tys.is_empty() {
                    self.make_builtin(Ty::Void)
                } else {
                    peer_type(self.arena, &arm_tys)
                }
            }

            // ── inline_trait 值：构造 TraitObject 类型 ──
            //
            // 从 expected type（val_decl 的类型注解）获取 trait 名，
            // 验证方法完备性，产出 TraitObject { trait_name, method_sigs }。
            // 若无 expected type，报错并返回 fresh_type_var（不允许无注解的 inline_trait）。
            Expr::InlineTrait(methods) => {
                // 从 expected type 获取 trait 名
                let trait_name: Option<Box<str>> = if let Some(exp) = expected {
                    let resolved = self.arena.resolve(exp);
                    match self.arena.get(resolved) {
                        Ty::Trait(_) => {
                            let (name, _) = self.arena.trait_parts(resolved);
                            Some(name.into())
                        }
                        Ty::TraitObject(_) => {
                            let (trait_name, _) = self.arena.trait_object_parts(resolved);
                            Some(trait_name.into())
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                // 收集 inline_trait 的方法签名
                let method_sigs: Vec<TraitMethodSig> = methods
                    .iter()
                    .map(|m| {
                        let return_type = match m.return_type {
                            Some(rt) => self.type_from_ast(rt, ast),
                            None => self.arena.make(Ty::Void),
                        };
                        TraitMethodSig {
                            name: m.name.into(),
                            param_count: m.params.len() as u8,
                            return_type,
                            is_async: m.is_async,
                            has_body: m.body.is_some(),
                        }
                    })
                    .collect();

                if let Some(tname) = trait_name {
                    // 验证方法完备性：trait_def 的 required methods（无 body）必须全部出现在 inline_trait 中
                    let missing: Vec<String> = if let Some(trait_def) = self.sema_result.get_trait_def(&tname) {
                        trait_def
                            .methods
                            .iter()
                            .filter(|req| !req.has_body)
                            .filter(|req| {
                                !method_sigs
                                    .iter()
                                    .any(|m| m.name == req.name && m.param_count == req.param_count)
                            })
                            .map(|req| {
                                format!(
                                    "inline_trait missing required method {} of trait {} (param count {})",
                                    tname, req.name, req.param_count
                                )
                            })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let span = ast.expr(expr).span;
                    for msg in missing {
                        self.sema_result.errors.push(SemaError::new(&msg, span.line, span.column));
                    }

                    // 类型检查各方法体：为参数绑定类型（有注解则用注解，无注解则 fresh_type_var），
                    // 设置 expected_return，调用 infer_expr 填充 body 内各子表达式的 expr_types。
                    // 这是 IR 编译期类型查询（如 str + str → concat）的数据来源。
                    for m in methods.iter() {
                        if let Some(body) = m.body {
                            let method_env = self.env.child(env);
                            for param in m.params.iter() {
                                let param_ty = match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                };
                                self.env.define(method_env, param.name, param_ty);
                            }
                            let prev_return = self.expected_return;
                            self.expected_return =
                                m.return_type.map(|rt| self.type_from_ast(rt, ast));
                            let _ = self.infer_expr(body, ast, method_env, self.expected_return);
                            self.expected_return = prev_return;
                        }
                    }

                    self.arena.make_trait_object(
                        tname,
                        method_sigs.into_boxed_slice(),
                    )
                } else {
                    let span = ast.expr(expr).span;
                    self.sema_result.errors.push(SemaError::new(
                        "inline_trait cannot infer trait name: explicit type annotation required",
                        span.line,
                        span.column,
                    ));
                    self.arena.fresh_type_var()
                }
            }
        }
    }

    /// 整数后缀 → 对应整型 TypeHandle（派生自 `BUILTIN_TABLE`，未命中返回 `None`）。
    fn int_suffix_to_type(&mut self, suffix: &str) -> Option<TypeHandle> {
        let tag = crate::types::ValueTag::from_name(suffix)?;
        if tag.is_int() {
            Some(self.arena.from_scalar_name(suffix))
        } else {
            None
        }
    }

    /// 浮点后缀 → 对应浮点 TypeHandle（派生自 `BUILTIN_TABLE`，未命中返回 `None`）。
    fn float_suffix_to_type(&mut self, suffix: &str) -> Option<TypeHandle> {
        let tag = crate::types::ValueTag::from_name(suffix)?;
        if tag.is_float() {
            Some(self.arena.from_scalar_name(suffix))
        } else {
            None
        }
    }

    /// 从 MethodSigInfo 的 owned 数据构造 Ty::Fn 类型。
    /// 参数和返回类型均通过 type_repr_to_handle 从 TypeRepr 完整解析，
    /// 正确处理嵌套泛型（如 Async<Throw<T, E>>）、数组、Nullable 等复合类型，
    /// 克服 type_name 仅存顶层名的限制。
    fn build_fn_type_from_sig(
        &mut self,
        param_type_reprs: Vec<TypeRepr>,
        return_type_repr: Option<TypeRepr>,
        _recv_ty: TypeHandle,
    ) -> TypeHandle {
        // SelfType 由 type_repr_to_handle 通过 current_self_type() 解析，
        // 调用方（lookup_method_type）已 push recv_ty 作为 self_type。
        let params: Vec<TypeHandle> = param_type_reprs
            .iter()
            .map(|repr| self.type_repr_to_handle(repr))
            .collect();
        let return_type = match return_type_repr {
            Some(repr) => self.type_repr_to_handle(&repr),
            None => self.arena.fresh_type_var(),
        };
        self.arena.make_fn(params.into_boxed_slice(), return_type)
    }

    /// 从自包含的 TypeRepr 构造 TypeHandle（不依赖 AstArena 引用）。
    /// 与 type_from_ast_with_params 逻辑镜像，但读取 TypeRepr 而非 AST TypeNode。
    /// 用于跨模块方法返回类型还原（MethodSigInfo.return_type_repr）。
    fn type_repr_to_handle(&mut self, repr: &TypeRepr) -> TypeHandle {
        match repr {
            TypeRepr::Named(name) => {
                let empty_map: FxHashMap<String, TypeHandle> = FxHashMap::default();
                let mut visiting = FxHashSet::default();
                self.resolve_name_to_type(name.as_ref(), &empty_map, &mut visiting)
            }
            TypeRepr::SelfType => match self.current_self_type() {
                Some(ty) => ty,
                None => self.arena.fresh_type_var(),
            },
            TypeRepr::Generic(name, args) => {
                let new_args: Vec<TypeHandle> =
                    args.iter().map(|a| self.type_repr_to_handle(a)).collect();
                let args_box: Box<[TypeHandle]> = new_args.into_boxed_slice();

                // 内置泛型类型（Throw/Atomic/Async/Channel 等）构造专用 Ty 变体
                if is_builtin_generic_type(name) {
                    return self.make_builtin_generic(name.clone(), args_box);
                }
                // trait 定义 → Trait 类型
                if self.sema_result.get_trait_def(name).is_some() {
                    return self.arena.make_trait(name.clone(), args_box);
                }
                // 用户自定义泛型 ADT
                let has_type_params = self
                    .sema_result
                    .get_type_def(name)
                    .map(|d| !d.type_params.is_empty())
                    .unwrap_or(false);
                if has_type_params {
                    return self.arena.make_adt(name.clone(), args_box);
                }
                // 兜底：构造 Generic（可能未定义或前向引用，后续使用时报错）
                self.arena.make_generic(name.clone(), args_box)
            }
            TypeRepr::Nullable(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_nullable(inner_ty)
            }
            TypeRepr::Ref(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_ref(inner_ty, false)
            }
            TypeRepr::RawPtr(inner) => {
                let inner_ty = self.type_repr_to_handle(inner);
                self.arena.make_ref(inner_ty, true)
            }
            TypeRepr::Function(params, return_type) => {
                let p: Vec<TypeHandle> =
                    params.iter().map(|a| self.type_repr_to_handle(a)).collect();
                let r = self.type_repr_to_handle(return_type);
                self.arena.make_fn(p.into_boxed_slice(), r)
            }
            TypeRepr::Array(elem, _) => {
                let elem_ty = self.type_repr_to_handle(elem);
                self.arena.make_array(elem_ty, None)
            }
        }
    }

    /// 查找对象类型的方法签名（返回函数类型，第一个参数为 self）。
    fn lookup_method_type(
        &mut self,
        recv_ty: TypeHandle,
        method: &str,
    ) -> Option<TypeHandle> {
        let resolved = self.arena.resolve(recv_ty);

        // ── 接收者规范化 ──
        // 包装类型（Nullable/Ref）递归转发到 inner 类型的方法查找，
        // 使 s?.len() / (&arr).len() 等调用自动解包到正确的方法表。
        // Nullable 自有方法（is_null）由统一 TypeDefInfo 路径处理，不转发。
        match self.arena.get(resolved) {
            Ty::Nullable(_) => {
                // Nullable 自有方法（is_null）走 TypeDefInfo("nullable") 路径，
                // 其他方法递归转发到 inner 类型。
                if method != "is_null" {
                    let inner = self.arena.nullable_inner(resolved);
                    return self.lookup_method_type(inner, method);
                }
            }
            Ty::Ref(_) => {
                // Ref 自动解引用：&T 的方法查找转发到 T
                let inner = self.arena.ref_parts(resolved).0;
                return self.lookup_method_type(inner, method);
            }
            _ => {}
        }

        // 将 recv_ty 作为 Self 类型压栈，使 build_fn_type_from_sig 中
        // type_repr_to_handle(SelfType) 能正确解析为接收者类型，
        // 无需对第一个参数做位置特判。
        self.push_self_type(resolved);

        // 泛型类型参数绑定：将类型定义的类型参数名（如 "T"）绑定到接收者
        // 类型中的具体类型参数，使方法签名中的 T（如 `pub fun next(&self): T?`）
        // 能通过 type_binding_stack 解析为接收者中对应的类型参数，
        // 而非生成孤立的 fresh_type_var。
        //
        // 统一处理 Adt（用户自定义泛型）和内置类型（Array/Nullable/Throw/Generic），
        // 使内置类型方法签名中的泛型参数也能正确绑定。
        let mut pushed_bindings = false;
        let builtin_bindings: Option<(Box<str>, Vec<TypeHandle>)> = match self.arena.get(resolved) {
            Ty::Adt(_) => {
                let (name, type_args) = self.arena.adt_parts(resolved);
                Some((name.into(), type_args.to_vec()))
            }
            Ty::Array(_) => {
                let (element_type, _) = self.arena.array_parts(resolved);
                Some(("array".into(), vec![element_type]))
            }
            Ty::Nullable(_) => {
                let inner = self.arena.nullable_inner(resolved);
                Some(("nullable".into(), vec![inner]))
            }
            Ty::Throw(_) => {
                let (value_type, error_type) = self.arena.throw_parts(resolved);
                Some(("Throw".into(), vec![value_type, error_type]))
            }
            // 内置泛型专用变体：提取元素/值类型作为 type_args 绑定
            Ty::Channel(_) => Some(("Channel".into(), vec![self.arena.channel_elem(resolved)])),
            Ty::Async(_) => Some(("Async".into(), vec![self.arena.async_value(resolved)])),
            Ty::Lazy(_) => Some(("Lazy".into(), vec![self.arena.lazy_value(resolved)])),
            Ty::Atomic(_) => Some(("Atomic".into(), vec![self.arena.atomic_elem(resolved)])),
            Ty::Sender(_) => Some(("Sender".into(), vec![self.arena.sender_elem(resolved)])),
            Ty::Receiver(_) => Some(("Receiver".into(), vec![self.arena.receiver_elem(resolved)])),
            Ty::Generic(_) => {
                let (name, args) = self.arena.generic_parts(resolved);
                Some((name.into(), args.to_vec()))
            }
            _ => None,
        };
        if let Some((type_name, actual_args)) = builtin_bindings {
            if let Some(def) = self.sema_result.get_type_def(type_name.as_ref()) {
                if !def.type_params.is_empty() && def.type_params.len() == actual_args.len() {
                    self.type_binding_stack.push();
                    for (pname, &arg) in def.type_params.iter().zip(actual_args.iter()) {
                        self.type_binding_stack.insert_top(pname.as_ref(), arg);
                    }
                    pushed_bindings = true;
                }
            }
        }

        let result = self.lookup_method_type_inner(resolved, method);
        if pushed_bindings {
            self.pop_type_bindings();
        }
        self.pop_self_type();
        result
    }

    fn lookup_method_type_inner(
        &mut self,
        resolved: TypeHandle,
        method: &str,
    ) -> Option<TypeHandle> {
        match self.arena.get(resolved) {
            Ty::Trait(_) => {
                let (name, _) = self.arena.trait_parts(resolved);
                // trait 类型（如 l: Logger）直接查 trait_def.methods 还原方法签名，
                // 参数用 fresh_type_var（trait 方法的精确参数类型由实现类型决定）
                if let Some(td) = self.sema_result.get_trait_def(name) {
                    if let Some(sig) = td.methods.iter().find(|m| m.name.as_ref() == method) {
                        // params[0] 是 self，绑定到接收者类型（resolved）避免产生孤立 TypeVar；
                        // 其余参数仍用 fresh_type_var（精确类型由实现类型决定）。
                        let params: Vec<TypeHandle> = (0..sig.param_count)
                            .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                            .collect();
                        let return_type = sig.return_type;
                        return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                    }
                }
            }
            _ => {}
        }

        // 内置类型名映射：Array/Nullable/Throw 是结构变体，arena.type_name 返回 None
        // 或递归到 inner。方法查找需要统一的类型名来查 type_def_index，
        // 此处映射到注册时的合成 TypeDefInfo 名称。
        let type_name: Option<String> = match self.arena.get(resolved) {
            Ty::Array(_) => Some("array".to_string()),
            Ty::Str => Some("str".to_string()),
            Ty::Nullable(_) => Some("nullable".to_string()),
            Ty::Throw(_) => Some("Throw".to_string()),
            _ => self.arena.type_name(resolved).map(|s| s.to_string()),
        };

        // v2 收敛：路径 1 — 查 witness_table（trait 方法分派，type_id 索引）
        if let Some(ref name) = type_name {
            let type_id = self
                .sema_result
                .type_def_index
                .get(name.as_str())
                .map(|&idx| dynamic_type_id(idx));
            if let Some(tid) = type_id {
                for entry in self.witness_table.entries().iter() {
                    if entry.type_id != tid {
                        continue;
                    }
                    // 从 TypeDefInfo.methods 获取签名（按 method_name 查找）
                    // 提取 owned 数据以释放 sema_result 借用
                    let sig_data: Option<(Vec<TypeRepr>, Option<TypeRepr>)> =
                        if let Some(&type_idx) = self.sema_result.type_def_index.get(name.as_str()) {
                            self.sema_result.type_defs[type_idx as usize]
                                .methods
                                .iter()
                                .find(|m| m.name.as_ref() == method)
                                .map(|sig| (sig.param_type_reprs.to_vec(), sig.return_type_repr.clone()))
                        } else {
                            None
                        };
                    if let Some((param_type_reprs, return_type_repr)) = sig_data {
                        return Some(self.build_fn_type_from_sig(param_type_reprs, return_type_repr, resolved));
                    }
                    // TypeDefInfo.methods 未命中 → 查 trait_def.methods（trait 默认方法）
                    // 类型通过 `type T: Trait = ...` 实现 trait 但未 override 方法时，
                    // method_slots 为空，方法签名从 trait_def 获取。
                    let trait_sig_data: Option<(u8, TypeHandle)> =
                        self.sema_result
                            .get_trait_def(entry.trait_name.as_ref())
                            .and_then(|td| {
                                td.methods
                                    .iter()
                                    .find(|m| m.name.as_ref() == method)
                                    .map(|m| (m.param_count, m.return_type))
                            });
                    if let Some((param_count, return_type)) = trait_sig_data {
                        // params[0] 是 self，绑定到接收者类型（resolved）避免产生孤立 TypeVar；
                        // 其余参数用 fresh_type_var（精确类型由实现类型决定）。
                        let params: Vec<TypeHandle> = (0..param_count)
                            .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                            .collect();
                        return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
                    }
                    // 当前 trait 未找到该方法，继续检查其他 trait 实现
                }
            }
        }

        // v2: 路径 1.5 — TraitObject 接收者，从 method_sigs 还原真实签名
        // 先提取 sig 数据（param_count + return_type）到 owned 变量，
        // 释放 arena.types 借用后再构造 Fn 类型
        let trait_sig_data: Option<(u8, TypeHandle)> =
            if let Ty::TraitObject(_) = self.arena.get(resolved) {
                let (_, method_sigs) = self.arena.trait_object_parts(resolved);
                method_sigs
                    .iter()
                    .find(|m| m.name.as_ref() == method)
                    .map(|sig| (sig.param_count, sig.return_type))
            } else {
                None
            };
        if let Some((param_count, return_type)) = trait_sig_data {
            // params[0] 是 self，绑定到接收者类型（resolved）避免产生孤立 TypeVar；
            // 其余参数用 fresh_type_var（精确类型由实现类型决定）。
            let params: Vec<TypeHandle> = (0..param_count)
                .map(|i| if i == 0 { resolved } else { self.arena.fresh_type_var() })
                .collect();
            return Some(self.arena.make_fn(params.into_boxed_slice(), return_type));
        }

        // v2 收敛：路径 2 — 查 TypeDefInfo.methods（类型自有方法，按 method_idx 索引）
        if let Some(ref name) = type_name {
            let sig_data: Option<(Vec<TypeRepr>, Option<TypeRepr>)> =
                if let Some(&type_idx) = self.sema_result.type_def_index.get(name.as_str()) {
                    self.sema_result.type_defs[type_idx as usize]
                        .methods
                        .iter()
                        .find(|m| m.name.as_ref() == method)
                        .map(|sig| (sig.param_type_reprs.to_vec(), sig.return_type_repr.clone()))
                } else {
                    None
                };
            if let Some((param_type_reprs, return_type_repr)) = sig_data {
                return Some(self.build_fn_type_from_sig(param_type_reprs, return_type_repr, resolved));
            }
        }

        None
    }

    /// 查找对象类型的字段类型。
    /// line/column 用于字段不存在时的错误定位（由调用方传入 AST span）。
    fn lookup_field_type(&mut self, recv_ty: TypeHandle, field: &str, line: u32, column: u32) -> TypeHandle {
        let resolved = self.arena.resolve(recv_ty);

        // Ref 自动解引用：&T 的字段访问转发到 T。
        // 对 &Record / &Adt 等引用类型，先剥除 Ref 再走正常的字段查找路径，
        // 避免 type_name 间接路径在 inner 为 TypeVar 时返回 None 而静默失败。
        if let Ty::Ref(_) = self.arena.get(resolved) {
            let inner = self.arena.ref_parts(resolved).0;
            return self.lookup_field_type(inner, field, line, column);
        }

        // ModuleRef 字段访问：在 ModuleRef 携带的模块 env 中按裸名查找 field。
        //
        // 使用 lookup_local（不穿透父 env 链）统一处理：
        // - 子模块：ensure_module_env 创建层级 env 时已将子模块短名注册到父 env
        // - 模块内符号：predeclare_declarations 已将函数/构造器注册到 module_env
        // 查不到即报错，无需字符串拼接或前缀校验。
        if let Ty::ModuleRef(_) = self.arena.get(resolved) {
            let (path, module_env) = self.arena.module_ref_parts(resolved);
            if let Some(sym_ty) = self.env.lookup_local(module_env, field) {
                return sym_ty;
            }
            self.add_error_at(
                &format!("no module or symbol '{}.{}'", path, field),
                line,
                column,
            );
            return self.arena.make(Ty::Unknown);
        }

        let type_name = self.arena.type_name(resolved).map(|s| s.to_string());
        if let Some(name) = type_name {
            if let Some(field_id) = self.sema_result.lookup_field_id(&name, field) {
                if let Some(ctor) = self.sema_result.get_ctor_def(&name) {
                    let idx = match self.sema_result.get_type_def(&name) {
                        Some(def) if def.kind == TypeDefKind::Record => field_id as usize,
                        _ => (field_id as usize).saturating_sub(1),
                    };
                    // 使用 field_type_reprs 通过 type_repr_to_handle 完整解析字段类型，
                    // 正确处理数组（T[]）、Nullable、Ref 等复合类型，
                    // 克服 field_type_names 仅存顶层名的限制。
                    // 先克隆 TypeRepr 以释放 sema_result 的不可变借用，再调用可变方法。
                    if let Some(repr) = ctor.field_type_reprs.get(idx).cloned() {
                        return self.type_repr_to_handle(&repr);
                    }
                    return self.arena.fresh_type_var();
                }
            }
        }
        // record 结构字段
        let ct = self.arena.get(resolved);
        if let Ty::Record(_) = ct {
            let fields = self.arena.record_fields(resolved);
            for f in fields.iter() {
                if f.name.as_deref() == Some(field) {
                    return f.ty;
                }
            }
        }
        // Channel<T>.sender / .receiver 字段：返回 Sender<T> / Receiver<T>
        // （运行时 Value.rs 已支持，Sema 层补全类型签名）
        if let Ty::Channel(_) = ct {
            let elem = self.arena.channel_elem(resolved);
            match field {
                "sender" => return self.arena.make_sender(elem),
                "receiver" => return self.arena.make_receiver(elem),
                _ => {}
            }
        }
        // 字段未找到：对已确定类型报"字段不存在"错误（与方法调用兜底一致）；
        // 未决类型（TypeVar/Unknown/Never/Void）静默返回 fresh var，延迟到 solver 全局诊断
        match ct {
            Ty::Record(_) => {
                self.add_error_at(&format!("no such field '{}' on this type", field), line, column);
                self.arena.fresh_type_var()
            }
            Ty::Adt(_) => {
                let (name, _) = self.arena.adt_parts(resolved);
                // 对已注册的 Adt 类型报字段不存在错误；未注册的保守放行
                if self.sema_result.get_type_def(name).is_some() {
                    self.add_error_at(
                        &format!("no such field '{}' on type '{}'", field, name),
                        line,
                        column,
                    );
                }
                self.arena.fresh_type_var()
            }
            // 未决类型：静默返回 fresh var（推断未决，延迟到 solver 全局诊断）
            Ty::TypeVar(_) | Ty::Unknown
            | Ty::Never | Ty::Void => {
                self.arena.fresh_type_var()
            }
            // 已确定类型但字段查找失败：报错
            ct_other => {
                let recv_name = self.arena.type_name(resolved)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| format!("{:?}", ct_other));
                self.add_error_at(
                    &format!("no such field '{}' on type '{}'", field, recv_name),
                    line,
                    column,
                );
                self.arena.fresh_type_var()
            }
        }
    }

    // ── infer_stmt ──

    /// 推断语句类型。返回 `Some(ty)` 表示语句产生值（表达式语句）。
    pub fn infer_stmt(
        &mut self,
        stmt: StmtId,
        ast: &AstArena<'_>,
        env: EnvId,
    ) -> Option<TypeHandle> {
        let node = &ast.stmt(stmt).node;
        match node {
            Stmt::ValDecl { name, type_annotation, value, .. } | Stmt::VarDecl { name, type_annotation, value, .. } => {
                // kind_check 类型注解
                if let Some(ta) = type_annotation {
                    let mut errors = Vec::new();
                    check_type_node(self.sema_result, ast, *ta, &[], &mut errors);
                    for e in errors {
                        self.sema_result.add_error(e);
                    }
                }
                let expected_ty = type_annotation.map(|ta| self.type_from_ast(ta, ast));
                let val_ty = self.infer_expr(*value, ast, env, expected_ty);
                let bind_ty = if let Some(ta) = type_annotation {
                    let annot_ty = self.type_from_ast(*ta, ast);
                    if self.try_widen_unify(annot_ty, val_ty).is_err() {
                        let annot_str = format!("{}", self.arena.display(annot_ty));
                        let val_str = format!("{}", self.arena.display(val_ty));
                        let span = ast.ty(*ta).span;
                        self.add_error_at(
                            &format!(
                                "type annotation mismatch: expected '{}', found '{}'",
                                annot_str, val_str
                            ),
                            span.line,
                            span.column,
                        );
                    }
                    annot_ty
                } else {
                    val_ty
                };
                self.env.define(env, name, bind_ty);
                None
            }
            Stmt::Assignment { target, value } => {
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
                let _ = self.infer_expr(*target, ast, env, None);
                let _ = self.infer_expr(*value, ast, env, None);
                None
            }
            Stmt::Expression { expr } => {
                let ty = self.infer_expr(*expr, ast, env, None);
                Some(ty)
            }
            Stmt::Return { value } => {
                if let Some(v) = value {
                    // 传播 expected_return 给 infer_expr，使 NullLit / match arm 等
                    // 依赖 expected 约束的表达式能正确推导，避免创建孤儿 TypeVar。
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
                    }
                    Some(val_ty)
                } else {
                    Some(self.make_builtin(Ty::Void))
                }
            }
            Stmt::Defer { expr } => {
                let _ = self.infer_expr(*expr, ast, env, None);
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
                let child_env = self.env.child(env);
                // 从迭代器类型结构化提取元素类型，建立约束而非使用孤立 fresh_type_var。
                // 覆盖：Array<T>、ArrayIter<T>、RangeIterator、Str→char、Map<K,V>→Entry<K,V> 等。
                // 提取失败时回退到 fresh_type_var 并通过约束让不动点求解。
                let item_ty = {
                    let resolved = self.arena.resolve(iterable_ty);
                    let ct = self.arena.get(resolved);
                    // 检查 iterable 是否为非迭代器类型（Array/Str/基元）
                    let is_non_iterator = match ct {
                        Ty::Array(_) => true,
                        ct if ct.is_scalar() => true,
                        _ => false,
                    };
                    if is_non_iterator {
                        let type_name = match ct {
                            Ty::Array(_) => "array",
                            _ => ct.name(),
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
                    // 结构化元素类型提取
                    self.extract_iterator_element(resolved).unwrap_or_else(|| {
                        // 提取失败：用 fresh_type_var 并注册约束供不动点求解
                        let fv = self.arena.fresh_type_var();
                        // 构造 ArrayIter<fv> 作为期望迭代器类型，与实际 iterable_ty 约束
                        let expected_iter = self.arena.make_generic(
                            "ArrayIter".into(),
                            vec![fv].into_boxed_slice(),
                        );
                        self.unify_or_constrain(iterable_ty, expected_iter);
                        fv
                    })
                };
                self.env.define(child_env, name, item_ty);
                let _ = self.infer_expr(*body, ast, child_env, None);
                None
            }
            Stmt::While { condition, body } => {
                let cond_ty = self.infer_expr(*condition, ast, env, None);
                let bool_ty = self.make_builtin(Ty::Bool);
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
                // 统一走 check_decl：函数、类型、trait 嵌套声明共享同一处理路径
                // LocalDecl 的 Box<Decl> 无 span，由所属 Stmt 提供
                self.check_decl(decl.as_ref(), ast.stmt(stmt).span, ast, env);
                None
            }

        }
    }

    // ── infer_pattern ──

    /// 推断模式类型，将绑定变量加入环境。
    pub fn infer_pattern(
        &mut self,
        pat: PatternId,
        ast: &AstArena<'_>,
        expected_ty: TypeHandle,
        env: EnvId,
    ) {
        let node = &ast.pattern(pat).node;
        match node {
            Pattern::Wildcard => {}
            Pattern::Literal(lit) => {
                let lit_ty = match lit {
                    PatternLiteral::Int(_) => Some(self.make_builtin(Ty::I32)),
                    PatternLiteral::Float(_) => Some(self.make_builtin(Ty::F64)),
                    PatternLiteral::Bool(_) => Some(self.make_builtin(Ty::Bool)),
                    PatternLiteral::Char(_) => Some(self.make_builtin(Ty::Char)),
                    PatternLiteral::String(_) => Some(self.make_builtin(Ty::Str)),
                    PatternLiteral::Null => None,
                };
                if let Some(lt) = lit_ty {
                    let resolved = self.arena.resolve(expected_ty);
                    let ct = self.arena.get(resolved).clone();
                    let is_int_expected = ct.is_int();
                    let is_int_lit = matches!(lit, PatternLiteral::Int(_));
                    if !(is_int_lit && is_int_expected) {
                        self.unify_or_constrain(lt, expected_ty);
                    }
                }
            }
            Pattern::Variable { name } => {
                // 大写开头 → 零参构造器
                if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    let sub_pats: Vec<PatternRef> = Vec::new();
                    self.refine_constructor_pattern(name, &sub_pats, expected_ty, ast, env);
                } else {
                    self.env.define(env, name, expected_ty);
                }
            }
            Pattern::Constructor { name, patterns } => {
                if !self.refine_constructor_pattern(name, patterns, expected_ty, ast, env) {
                    // 常规构造器 fallback：使用 field_type_reprs（自包含 TypeRepr）
                    // 替代 field_type_nodes（AST 引用），避免跨模块 AST arena 不匹配。
                    let field_type_reprs: Box<[TypeRepr]> = self
                        .sema_result
                        .get_ctor_def(name)
                        .map(|c| c.field_type_reprs.clone())
                        .unwrap_or_else(|| Box::new([]));
                    for (i, &sub_pat) in patterns.iter().enumerate() {
                        let sub_ty = if i < field_type_reprs.len() {
                            self.type_repr_to_handle(&field_type_reprs[i])
                        } else {
                            self.arena.fresh_type_var()
                        };
                        self.infer_pattern(sub_pat, ast, sub_ty, env);
                    }
                }
            }
            Pattern::Record { fields } => {
                for field in fields.iter() {
                    let field_ty = self.arena.fresh_type_var();
                    self.infer_pattern(field.pattern, ast, field_ty, env);
                }
            }
            Pattern::OrPattern { left, right } => {
                self.infer_pattern(*left, ast, expected_ty, env);
                self.infer_pattern(*right, ast, expected_ty, env);
            }
            Pattern::Guard { pattern, condition } => {
                self.infer_pattern(*pattern, ast, expected_ty, env);
                let cond_ty = self.infer_expr(*condition, ast, env, None);
                let bool_ty = self.make_builtin(Ty::Bool);
                self.unify_or_constrain(cond_ty, bool_ty);
            }
        }
    }

    // ── register_builtins ──

    /// 注册内置函数到环境。
    pub fn register_builtins(&mut self, env: EnvId) {
        // Panic: (str) -> void
        let str_ty = self.make_builtin(Ty::Str);
        let void_ty = self.make_builtin(Ty::Void);
        let panic_fn = self.arena.make_fn(
            vec![str_ty].into_boxed_slice(),
            void_ty,
        );
        self.env.define(env, "Panic", panic_fn);

        // type/type_name 已改为 kuzo wrapper（见 Reflect.kz::type_name）
        // Sema 不再注册 type builtin

        // Ok: ∀T,E. (T) -> Throw<T, E>
        // 用 rigid var 注册（泛型参数），调用时由 instantiate_fn_type 实例化为 fresh non-rigid var
        let val_ty = self.arena.fresh_rigid_var();
        let err_ty = self.arena.fresh_rigid_var();
        let throw_ty = self.arena.make_throw(val_ty, err_ty);
        let ok_fn = self.arena.make_fn(
            vec![val_ty].into_boxed_slice(),
            throw_ty,
        );
        self.env.define(env, "Ok", ok_fn);

        // 数值类型构造器：i8/i16/.../f64 等作为 ∀T. (T) -> Self
        // 用 rigid var 注册，调用时由 instantiate_fn_type 实例化
        for (name, ct) in numeric_builtin_names() {
            let param = self.arena.fresh_rigid_var();
            let ret_ty = self.make_builtin(ct);
            let fn_ty = self.arena.make_fn(
                vec![param].into_boxed_slice(),
                ret_ty,
            );
            self.env.define(env, name, fn_ty);
        }

        // channel<T>(capacity: usize) -> Channel<T>
        // 内置 channel 构造器：创建容量为 capacity 的 Channel<T>
        let usize_ty = self.make_builtin(Ty::Usize);
        let t_var3 = self.arena.fresh_rigid_var();
        let chan_ret = self.arena.make_channel(t_var3);
        let chan_fn = self.arena.make_fn(
            vec![usize_ty].into_boxed_slice(),
            chan_ret,
        );
        self.env.define(env, "channel", chan_fn);

        // Value：builtin opaque 类型（ValueHandle, u32）
        // 反射原语接收 Value，内部查 ValueArena 拿 HeapObj 直接 match。
        // 对 Sema 是 opaque 类型（不暴露内部结构），大小 4B。
        let value_ty = self.arena.make_generic(
            "Value".into(),
            Box::new([]),
        );
        self.env.define(env, "Value", value_ty);
    }

    // ── check_module ──

    /// 获取或创建模块路径对应的专属 EnvId。
    ///
    /// 按路径段逐级创建 env，形成层级结构：
    ///   "std.io.File" → 创建 env_std (parent=root_env)
    ///                  → env_std_io (parent=env_std)
    ///                  → env_std_io_file (parent=env_std_io)
    ///
    /// 每一级 env 中会注册子模块短名 → ModuleRef，使逐级字段访问能通过 env 链结构化查找。
    /// 已存在的路径 env 会被复用（幂等）。
    ///
    /// 返回该路径对应的 EnvId。
    fn ensure_module_env(&mut self, full_path: &str, root_env: EnvId) -> EnvId {
        // 已缓存：直接返回
        if let Some(&eid) = self.module_envs.get(full_path) {
            return eid;
        }
        let segments: Vec<&str> = full_path.split('.').collect();
        let mut current_path = String::new();
        let mut parent_env = root_env;
        for (i, seg) in segments.iter().enumerate() {
            if i > 0 {
                current_path.push('.');
            }
            current_path.push_str(seg);
            // 当前路径段的 env：已存在则复用，否则创建
            let env_id = if let Some(&eid) = self.module_envs.get(&current_path) {
                eid
            } else {
                let eid = self.env.child(parent_env);
                self.module_envs.insert(current_path.clone(), eid);
                eid
            };
            // 在父 env 中注册当前段短名 → ModuleRef（使逐级字段访问可查到）
            // 首段注册到 root_env，其余段注册到父路径 env
            let mod_ref_ty = self.arena.make_module_ref(
                current_path.clone().into_boxed_str(),
                env_id,
            );
            // 不覆盖已存在的绑定（用户显式 import / 构造器优先）
            self.env.define(parent_env, seg, mod_ref_ty);
            parent_env = env_id;
        }
        parent_env
    }

    /// 目录模块语义：在兄弟模块的 env 中查找函数。
    ///
    /// 当 `Math.sqrt` 中 `sqrt` 定义在 `Power.kz`（而非 `Math.kz`）时，
    /// 从 `mod_path`（如 "std.math.Math"）推导目录前缀（"std.math"），
    /// 遍历同目录下所有兄弟模块（"std.math.Power", "std.math.Trig", ...）的 env，
    /// 按 `method` 裸名查找函数。跳过自身 env（已由调用方 lookup_local 查过）。
    fn lookup_sibling_module_fn(
        &self,
        mod_path: &str,
        self_env: EnvId,
        method: &str,
    ) -> Option<TypeHandle> {
        // 推导目录前缀：mod_path 的最后一个 '.' 之前部分
        let dot_pos = mod_path.rfind('.')?;
        let dir_prefix = &mod_path[..dot_pos]; // 如 "std.math"
        let sibling_prefix = format!("{}.", dir_prefix); // "std.math."

        // 遍历 module_envs 中以 "std.math." 开头的兄弟模块
        for (path, &env_id) in self.module_envs.iter() {
            if !path.starts_with(&sibling_prefix) {
                continue;
            }
            if path == mod_path {
                continue; // 跳过自身
            }
            if env_id == self_env {
                continue; // 跳过自身 env
            }
            if let Some(ty) = self.env.lookup_local(env_id, method) {
                return Some(ty);
            }
        }
        None
    }

    /// 注册模块路径别名到环境（用于同包模块符号可见性）。
    ///
    /// 为每个模块路径创建层级 env，并在 root_env 中注册末段短名 → ModuleRef，
    /// 使同包模块可直接通过短名访问（如 `Calendar` → `ModuleRef("std.time.Calendar", env)`）。
    /// 已存在的绑定不会被覆盖（用户显式 import 优先）。
    pub fn register_module_aliases(&mut self, root_env: EnvId, module_paths: &[String]) {
        for path in module_paths {
            if path.is_empty() {
                continue;
            }
            // 确保模块层级 env 存在（含中间路径前缀）
            let module_env = self.ensure_module_env(path, root_env);
            // 在 root_env 注册末段短名（同包短名访问）
            if let Some(last_seg) = path.rsplit('.').next() {
                if !last_seg.is_empty() && path.contains('.') {
                    // 不覆盖已存在的绑定
                    if self.env.lookup(root_env, last_seg).is_none() {
                        let mod_ref_ty = self.arena.make_module_ref(
                            path.clone().into_boxed_str(),
                            module_env,
                        );
                        self.env.define(root_env, last_seg, mod_ref_ty);
                    }
                }
            }
        }
    }

    /// 模块检查入口：编排 populate → 预声明 → 推断 → kind_check → monomorph。
    ///
    /// 返回 true 表示无错误。步骤：
    /// 1. populate_module 填充 SemaResult 定义表
    /// 2. 创建根环境，注册内置函数
    /// 3. 预声明函数和类型构造器
    /// 4. 推导表达式声明和函数体
    /// 5. 运行 kind_check
    /// 6. 收集单态化实例
    pub fn check_module(&mut self, module: &Module<'_>) -> bool {
        // 单模块检查：创建新 root_env，注册 builtins，检查模块
        self.reset_state();
        let root_env = self.env.root();
        self.register_builtins(root_env);
        let all_modules = [module];
        self.check_module_with_env(module, root_env, &all_modules)
    }

    /// 多模块共享 env 检查入口。
    ///
    /// 接受外部共享的 `root_env`（已注册 builtins 和前置模块的符号），
    /// 在此基础上处理 import、预声明、检查当前模块。
    /// 跨模块符号通过共享 env 链查找。
    pub fn check_module_with_env<'m>(
        &mut self,
        module: &'m Module<'m>,
        root_env: EnvId,
        all_modules: &[&'m Module<'m>],
    ) -> bool {
        // 1. 填充定义表（若尚未填充）
        populate_module(self.arena, self.sema_result, module);

        // 2. 重置状态（不重置 env，保留共享 root_env）
        self.reset_state();
        // 快照当前 type_vars/types 长度：arena 跨模块共享不重置，诊断时只统计本模块新增的 TypeVar
        let type_vars_baseline = self.arena.type_vars_len();
        let types_baseline = self.arena.len();
        self.current_module_logical_path = module_logical_path(module.name);
        self.current_module_name = module.name.to_string();

        // 3. 处理 import 声明：注册模块引用别名 + import 别名
        self.process_import_decls(module, root_env);

        // 4. 预声明函数和类型构造器（含 extern 函数）
        self.predeclare_declarations(module, root_env);

        // 5. 填充 witness table（遍历 trait impl）
        self.populate_witness_table(module);

        // 6. 推导声明
        // 使用 module_env 作为基环境（而非 root_env），使函数体可通过 env 链
        // 查找同模块函数（predeclare_declarations 注册于 module_env），
        // 同时仍可通过父链访问 root_env 的全局 builtins 和构造器。
        let check_env = self.current_module_env.unwrap_or(root_env);
        for decl in module.declarations.iter() {
            self.check_decl(&decl.node, decl.span, &module.arena, check_env);
        }

        // 7. kind_check 所有类型注解
        self.run_kind_checks(module);

        // 8. 收集单态化实例（泛型函数实例，不依赖 witness_table）
        crate::sema::Monomorph::collect_monomorph_instances(module, all_modules, self.sema_result, self.arena);

        // 9. 求解延迟约束（带 witness table 支持 trait bound 求解）
        // 分离借用 self 的不同字段：arena 可变借用，witness_table 只读借用
        let InferContext { arena, solver, witness_table, type_trace, .. } = self;
        solver.solve_with_witness(arena, Some(witness_table));

        // 9.4 默认化未绑定的非刚性 TypeVar 为 void（根因F）
        //
        // 约束求解后仍有未绑定的非刚性 TypeVar 是正常的类型推断现象：
        // - Throw<T, E> 中 E 未约束（函数不抛出错误时 E 无信息来源）
        // - ArrayIter<T> 中 T 未约束（迭代器未被消费）
        // - 泛型函数实例化产生的 TypeVar 未被调用点约束
        //
        // 这些 TypeVar 没有任何候选类型（不是歧义），void 作为单位类型是安全的默认值：
        // 未约束意味着任何类型都满足，void 不引入额外约束。
        // 这是标准类型推断默认化技术（ML 的 defaulting rule），不是错误回退。
        // 真正的类型错误由 unify 失败和歧义检测捕获，不依赖此处诊断。
        let void_ty = arena.make(Ty::Void);
        for i in type_vars_baseline..arena.type_vars.len() {
            let tv = &mut arena.type_vars[i];
            if !tv.is_rigid && tv.bound.is_none() {
                tv.bound = Some(void_ty);
            }
        }

        // 9.5 全局残留 TypeVar 诊断：求解后仍有未绑定的非 rigid TypeVar 表示类型推断失败
        // 只统计本模块新增的 TypeVar（arena 跨模块共享，baseline 之前的属于前置模块）
        let unresolved: Vec<u32> = arena.type_vars.iter().enumerate()
            .skip(type_vars_baseline)
            .filter(|(_, tv)| !tv.is_rigid && tv.bound.is_none())
            .map(|(i, _)| i as u32)
            .collect();

        // 详细日志（环境变量 KUZO_SEMA_TRACE 控制）：打印未解析 TypeVar 详情，便于定位
        if !unresolved.is_empty() && std::env::var("KUZO_SEMA_TRACE").is_ok() {
            let unresolved_set: FxHashSet<u32> = unresolved.iter().copied().collect();
            eprintln!(
                "[sema] {} unresolved type variable(s) after constraint solving:",
                unresolved.len()
            );
            for &idx in unresolved.iter().take(50) {
                let tv = &arena.type_vars[idx as usize];
                eprintln!("  TypeVar({}) kind={:?}", idx, tv.kind);
            }
            if unresolved.len() > 50 {
                eprintln!("  ... and {} more", unresolved.len() - 50);
            }
            // 打印包含未解析 TypeVar 的类型槽位样本（最多 30 个）
            // 只遍历本模块新增的类型槽位（baseline 之前的属于前置模块）
            eprintln!("  sample referencing types (baseline={}):", types_baseline);
            let mut shown = 0u32;
            for i in types_baseline..arena.types.len() {
                let h = TypeHandle(i as u32);
                let s = format!("{}", arena.display(h));
                if s.contains("'_") {
                    eprintln!("    types[{}] = {}", i, s);
                    shown += 1;
                    if shown >= 30 { break; }
                }
            }
            // 反向定位：遍历 type_trace，找到引用未解析 TypeVar 的表达式 span
            eprintln!("  referencing expression spans:");
            let mut span_shown = 0u32;
            for &(ty, span) in type_trace.iter() {
                if type_contains_any_unresolved(ty, arena, &unresolved_set) {
                    let s = format!("{}", arena.display(ty));
                    eprintln!("    {}:{}  {}", span.line, span.column, s);
                    span_shown += 1;
                    if span_shown >= 50 { break; }
                }
            }
            if span_shown == 0 {
                eprintln!("    (no direct expression references found — TypeVar may be inside fn signature)");
            }
        }

        // 10. 镜像 witness_table 到 sema_result（供 IR 层访问 trait 方法分派信息）
        // witness_table 跨模块累积，每次 check 完成后同步最新状态。
        self.sema_result.witness_table = witness_table.clone();

        // 10a. 收集 trait 默认方法单态化实例（依赖已镜像的 witness_table）
        crate::sema::Monomorph::collect_trait_default_instances(module, self.sema_result);

        // 11. 报告全局残留 TypeVar 诊断
        if !unresolved.is_empty() {
            self.add_error_at(
                &format!("{} unresolved type variable(s) after constraint solving", unresolved.len()),
                0, 0,
            );
        }

        !self.sema_result.has_error
    }

    /// 重置检查状态（env 不重置，保留共享 root_env）。
    pub fn reset_state(&mut self) {
        self.expected_return = None;
        self.type_binding_stack = TypeBindingStack::new();
        self.self_binding_stack = SelfBindingStack::new();
        self.solver.reset();
        self.flow_ctx.reset();
        self.type_trace.clear();
        // witness_table 不重置（跨模块累积，支持多模块 trait 实现）
    }

    /// 处理模块中的 ImportDecl：
    /// - 整路径导入 `import std.io.File` → 确保模块层级 env 存在，首段注册为 ModuleRef
    ///   （字段访问逐级构建路径：std → std.io → std.io.File，通过 env 链查找）
    /// - selective import `import std.io.File { open }` → 从目标模块 env 查找符号并注册别名
    fn process_import_decls(&mut self, module: &Module<'_>, env: EnvId) {
        // 注册当前模块自身的模块路径前缀（如 std/io/Path.kz → std.io.Path）
        // 使模块内自引用（如 std.io.Path.last_index_of）可解析
        if let Some(logical_path) = module_logical_path(module.name) {
            // ensure_module_env 会创建层级 env 并在父 env 注册首段 ModuleRef
            self.ensure_module_env(&logical_path, env);
        }

        for decl in module.declarations.iter() {
            if let Decl::ImportDecl { module_path, items, .. } = &decl.node {
                if module_path.is_empty() {
                    continue;
                }
                let full_path = module_path.join(".");
                // 确保导入模块的层级 env 存在（含中间路径前缀和首段 ModuleRef 注册）
                let module_env = self.ensure_module_env(&full_path, env);

                // selective import：从目标模块 env 查找符号并注册到当前 env
                if let Some(items) = items {
                    for item in items.iter() {
                        // 在模块 env 中按裸名查找符号（不穿透父 env，避免导入全局符号）
                        if let Some(sym_ty) = self.env.lookup_local(module_env, item.name) {
                            let local_name = item.alias.unwrap_or(item.name);
                            self.env.define(env, local_name, sym_ty);
                        }
                    }
                }
            }
        }
    }

    /// 填充 witness table：遍历模块中的 trait impl，注册到 witness table。
    ///
    /// 对于每个 `impl Trait for Type`，提取 trait_name 和 type_name，
    /// 查询 type_def 获取 type_id，将方法注册到 witness table。
    fn populate_witness_table(&mut self, module: &Module<'_>) {
        type TraitImplInfo = (String, String, Vec<(String, u16)>);
        // 收集 trait impl 信息，避免在遍历时借用 module 同时 &mut self
        let mut impls: Vec<TraitImplInfo> = Vec::new();

        for decl in module.declarations.iter() {
            if let Decl::TypeDecl { name, implemented_traits, methods, .. } = &decl.node {
                // 查询 type_id（用 type_def_index + FIRST_DYNAMIC_TYPE_ID 偏移）
                let type_id = self
                    .sema_result
                    .type_def_index
                    .get(*name)
                    .map(|&idx| dynamic_type_id(idx));

                if let Some(tid) = type_id {
                    // 为每个实现的 trait 注册 witness entry
                    for impl_trait in implemented_traits.iter() {
                        let trait_name = impl_trait.trait_name.to_string();
                        // 收集方法槽位：method_name → method_idx（在 TypeDefInfo.methods 中的位置）
                        let method_slots: Vec<(String, u16)> = methods
                            .iter()
                            .enumerate()
                            .map(|(i, m)| (m.name.to_string(), i as u16))
                            .collect();
                        impls.push((trait_name, name.to_string(), method_slots));
                        let _ = tid; // tid 在下面的循环中使用
                    }
                }
            }
        }

        // 注册到 witness table
        for (trait_name, type_name, method_slots_vec) in impls {
            // 重新查询 type_id（因为上面的借用已释放）
            let type_id = self
                .sema_result
                .type_def_index
                .get(type_name.as_str())
                .map(|&idx| dynamic_type_id(idx));
            if let Some(tid) = type_id {
                let mut slots = FxHashMap::default();
                for (method_name, method_idx) in method_slots_vec {
                    slots.insert(method_name.into_boxed_str(), method_idx);
                }
                self.witness_table
                    .register(&trait_name, tid, &type_name, slots);
            }
        }
    }

    /// 将返回类型按需包装为 `Async<T>`。
    ///
    /// async 函数/方法声明的返回类型 `X` 实际表示 `Async<X>`（与 Lambda/Zig 实现对齐）。
    /// 若用户已显式写 `Async<X>`，不二次包装；否则包装为 `Async<ret_ty>`。
    /// 统一用于 predeclare_declarations、check_decl(FunDecl)、type 块方法、trait 块方法，
    /// 避免各处重复内联导致遗漏（根因D：type/trait 块方法曾遗漏包装）。
    fn wrap_async_return(&mut self, ret_ty: TypeHandle, is_async: bool) -> TypeHandle {
        if !is_async {
            return ret_ty;
        }
        let resolved = self.arena.resolve(ret_ty);
        let already_async = matches!(self.arena.get(resolved), Ty::Async(_));
        if already_async {
            ret_ty
        } else {
            self.arena.make_async(ret_ty)
        }
    }

    /// 预声明模块中的函数和类型构造器到环境。
    ///
    /// 函数和类型构造器注册到模块专属 env（module_env），而非 root_env。
    /// 模块 env 的父环境指向 root_env（或父路径 env），使模块内可访问全局 builtins。
    /// 调用方通过 ModuleRef 携带的 env 引用直接在模块 env 中按裸名查找，无需 mangled name。
    pub fn predeclare_declarations(&mut self, module: &Module<'_>, root_env: EnvId) {
        let module_path = module_logical_path(module.name);
        // 获取或创建模块专属 env（幂等：ensure_module_env 会复用已存在的 env）
        let module_env = match &module_path {
            Some(mp) => self.ensure_module_env(mp, root_env),
            None => root_env,
        };
        // 记录当前模块 env，供 check_decl 中的 let 绑定等使用
        self.current_module_env = Some(module_env);
        for decl in module.declarations.iter() {
            match &decl.node {
                Decl::FunDecl { name, type_params, params, return_type, is_async, .. } => {
                    // 顶层函数不允许 self 参数（通过 SelfType 类型节点判断，不依赖参数名）
                    if !params.is_empty() && self.is_self_param(params[0].type_annotation, &module.arena) {
                        self.add_error_at(
                            "self parameter is not allowed in top-level function",
                            decl.span.line,
                            decl.span.column,
                        );
                    }
                    // 泛型函数：push type_bindings 使 type_from_ast 能解析类型参数为 rigid var，
                    // 预声明类型与 check_decl 阶段一致（避免泛型参数被误解析为 Adt）
                    if !type_params.is_empty() {
                        self.push_type_bindings(
                            &type_params.iter().map(|tp| {
                                (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                            }).collect::<Vec<_>>(),
                        );
                    }
                    // 所有函数都预声明（含泛型）：泛型函数用 rigid var 作为类型参数，
                    // 解决前向引用问题（函数体内可引用后续定义的同模块函数）
                    let param_types: Vec<TypeHandle> = params
                        .iter()
                        .map(|p| match p.type_annotation {
                            Some(ta) => self.type_from_ast(ta, &module.arena),
                            None => self.arena.fresh_type_var(),
                        })
                        .collect();
                    // async 函数：用户声明的返回类型 X 实际表示 Async<X>（与 check_decl/Lambda 对齐）
                    // - 若用户已显式写 Async<X>，不二次包装
                    // - 否则包装为 Async<ret_ty_raw>
                    let ret_ty_raw = match return_type {
                        Some(rt) => self.type_from_ast(*rt, &module.arena),
                        None => self.arena.fresh_type_var(),
                    };
                    let ret_ty = self.wrap_async_return(ret_ty_raw, *is_async);
                    let fn_ty = self.arena.make_fn(
                        param_types.into_boxed_slice(),
                        ret_ty,
                    );
                    // 注册到模块专属 env（裸名），ModuleRef 查找时通过 lookup_local 在此 env 中查找
                    // 同时注册到 root_env 使其全局可见（跨模块裸名引用兼容）：
                    //   define 不覆盖已存在绑定，同名函数首次注册生效
                    self.env.define(module_env, name, fn_ty);
                    self.env.define(root_env, name, fn_ty);
                    // 泛型函数：弹出 type_bindings（与 check_decl 对称）
                    if !type_params.is_empty() {
                        self.pop_type_bindings();
                    }
                }
                Decl::TypeDecl { name, type_params, def, .. } => {
                    // 预声明类型构造器
                    let self_ty = if type_params.is_empty() {
                        self.arena.make_adt((*name).into(), Box::new([]))
                    } else {
                        // 泛型类型：用 rigid var 预声明
                        self.arena.fresh_rigid_var()
                    };
                    // 构造器注册到 root_env（而非 module_env）：
                    // 构造器是类型的伴生符号，与类型在同一命名层级，
                    // 需通过 redefine 覆盖 register_module_aliases 先注册的 ModuleRef 别名，
                    // 使 `DateTime(...)` 解析为构造器而非 ModuleRef。
                    match def {
                        crate::ast::Ast::TypeDef::Adt { constructors } => {
                            for ctor in constructors.iter() {
                                let ctor_fn_ty = self.build_ctor_fn_type(ctor, name, &module.arena);
                                self.env.redefine(root_env, ctor.name, ctor_fn_ty);
                                // 记录构造器短名 → 模块 env（Zig @This 语义），
                                // 使 `TypeName.free_func(args)` 能回退查找模块内自由函数
                                self.ctor_module_envs.insert(ctor.name.to_string(), module_env);
                            }
                        }
                        crate::ast::Ast::TypeDef::Newtype { name: ctor_name, inner } => {
                            // newtype 构造器：(inner) -> Self
                            let inner_ty = self.type_from_ast(*inner, &module.arena);
                            let ctor_fn_ty = self.arena.make_fn(
                                vec![inner_ty].into_boxed_slice(),
                                self_ty,
                            );
                            self.env.redefine(root_env, ctor_name, ctor_fn_ty);
                            // 记录构造器短名 → 模块 env（Zig @This 语义），
                            // 使 `TypeName.free_func(args)` 能回退查找模块内自由函数
                            self.ctor_module_envs.insert(ctor_name.to_string(), module_env);
                        }
                        _ => {}
                    }
                    let _ = self_ty;
                }
                _ => {}
            }
        }
    }

    /// 构造构造器的函数类型。
    fn build_ctor_fn_type(
        &mut self,
        ctor: &crate::ast::Ast::ConstructorDef<'_>,
        type_name: &str,
        ast: &AstArena<'_>,
    ) -> TypeHandle {
        let param_types: Vec<TypeHandle> = ctor
            .fields
            .iter()
            .map(|f| self.type_from_ast(f.ty, ast))
            .collect();
        let ret_ty = match ctor.return_type {
            Some(rt) => self.type_from_ast(rt, ast),
            None => self.arena.make_adt(type_name.into(), Box::new([])),
        };
        // 零参数变体是值，不是函数：Leaf 的类型应为 Tree 而非 () -> Tree
        if param_types.is_empty() {
            return ret_ty;
        }
        self.arena.make_fn(param_types.into_boxed_slice(), ret_ty)
    }

    /// 检查单个声明（推导函数体/表达式）。
    ///
    /// 接受 `&Decl` 与 `decl_span` 分开参数：顶层声明从 `Spanned<Decl>` 取 span+node，
    /// 嵌套 `LocalDecl` 的 `Box<Decl>` 无 span，由调用方从所属 Stmt 提供。
    fn check_decl(&mut self, decl: &Decl<'_>, decl_span: crate::ast::Ast::Span, ast: &AstArena<'_>, env: EnvId) {
        match decl {
            Decl::FunDecl { name, type_params, params, return_type, body, extern_c_body, is_async, .. } => {
                // 顶层函数不允许 self 参数（通过 SelfType 类型节点判断，不依赖参数名；
                // self 只能在 type/trait 块内方法中使用）
                if !params.is_empty() && self.is_self_param(params[0].type_annotation, ast) {
                    self.add_error_at(
                        "self parameter is not allowed in top-level function",
                        decl_span.line,
                        decl_span.column,
                    );
                }
                // 为函数创建子环境
                let fn_env = self.env.child(env);
                // 类型参数绑定
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                // @extern("C") 函数：注册签名但跳过函数体类型检查（body 为 C 代码，非 Kuzo 表达式）
                if extern_c_body.is_some() {
                    if !type_params.is_empty() {
                        self.pop_type_bindings();
                    }
                    let _ = name;
                    return;
                }
                // 参数绑定（同时收集参数类型用于构造函数类型）
                let param_types: Vec<TypeHandle> = params.iter().map(|p| {
                    let param_ty = match p.type_annotation {
                        Some(ta) => self.type_from_ast(ta, ast),
                        None => self.arena.fresh_type_var(),
                    };
                    self.env.define(fn_env, p.name, param_ty);
                    param_ty
                }).collect();
                // 返回类型（未标注时用 fresh_type_var，后续与函数体类型统一）
                // async 函数：用户声明的返回类型 X 实际表示 Async<X>（与 Lambda/Zig 实现对齐）
                // - 若用户已显式写 Async<X>，不二次包装
                // - 否则包装为 Async<ret_ty_raw>
                let ret_ty_raw = match return_type {
                    Some(rt) => self.type_from_ast(*rt, ast),
                    None => self.arena.fresh_type_var(),
                };
                let ret_ty = self.wrap_async_return(ret_ty_raw, *is_async);
                // 构造函数类型并注册到 fn_env（支持递归自引用）和 env（支持后续引用）
                // 顶层函数已由 predeclare_declarations 预注册，define 返回 false 不覆盖
                let fn_ty = self.arena.make_fn(
                    param_types.into_boxed_slice(),
                    ret_ty,
                );
                self.env.define(fn_env, *name, fn_ty);
                self.env.define(env, *name, fn_ty);
                // 设置返回类型
                let prev_return = self.expected_return;
                self.expected_return = Some(ret_ty);
                // 推导函数体
                let body_ty = self.infer_expr(*body, ast, fn_env, self.expected_return);
                // 恢复
                self.expected_return = prev_return;
                // 返回类型与函数体类型统一：
                // - 无标注返回类型：ret_ty 为 fresh TypeVar，用 unify_or_constrain 绑定
                // - 有标注返回类型：用 unify_return_type 统一，处理 async 穿透
                //   （声明 Async<Throw<T, E>>，body 直接返回 Throw<T', E'>，
                //    需穿透 Async 层统一内层 Throw，使 E' 中的 TypeVar 被求解）
                //   失败时注册 Equality 约束供 solver 延迟重试
                if return_type.is_none() {
                    self.unify_or_constrain(ret_ty, body_ty);
                } else if self.unify_return_type(ret_ty, body_ty).is_err() {
                    self.solver.add_equality(ret_ty, body_ty);
                }
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
                let _ = name;
            }
            Decl::ExprDecl { expr, stmt, .. } => {
                if let Some(s) = stmt {
                    let _ = self.infer_stmt(*s, ast, env);
                } else {
                    let _ = self.infer_expr(*expr, ast, env, None);
                }
            }
            Decl::TypeDecl { name, type_params, def, methods, .. } => {
                // 注册嵌套类型定义到 sema_result（使构造器调用可被类型检查识别）
                ast_type_decl_to_type_def(self.arena, self.sema_result, *name, type_params, def, ast);
                // 类型参数绑定（含 kind 注册）：使类型块内部引用泛型参数 T 时可从 type_binding_stack 解析
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                // 构造 ADT 类型 handle
                let self_ty = if type_params.is_empty() {
                    self.arena.make_adt((*name).into(), Box::new([]))
                } else {
                    // 泛型类型：构造 Adt { name, type_args: [rigid_T, ...] }
                    // 使用 type_binding_stack 中的 rigid var 作为 type_args，
                    // 避免 fresh_type_var 作为 self_ty 产生未解析 TypeVar
                    let type_args: Vec<TypeHandle> = type_params.iter()
                        .map(|tp| self.lookup_type_binding(tp.name)
                            .unwrap_or_else(|| self.arena.fresh_type_var()))
                        .collect();
                    self.arena.make_adt((*name).into(), type_args.into_boxed_slice())
                };
                // 将构造器函数类型注册到当前环境（使 Call 表达式能查找到构造器）
                match def {
                    crate::ast::Ast::TypeDef::Record { fields } => {
                        let param_types: Vec<TypeHandle> = fields.iter().map(|f| {
                            self.type_from_ast(f.ty, ast)
                        }).collect();
                        let fn_ty = self.arena.make_fn(
                            param_types.into_boxed_slice(),
                            self_ty,
                        );
                        self.env.define(env, *name, fn_ty);
                    }
                    crate::ast::Ast::TypeDef::Adt { constructors } => {
                        for ctor in constructors {
                            let param_types: Vec<TypeHandle> = ctor.fields.iter().map(|f| {
                                self.type_from_ast(f.ty, ast)
                            }).collect();
                            let fn_ty = if param_types.is_empty() {
                                // 零参数变体是值，不是函数
                                self_ty
                            } else {
                                self.arena.make_fn(
                                    param_types.into_boxed_slice(),
                                    self_ty,
                                )
                            };
                            self.env.define(env, ctor.name, fn_ty);
                        }
                    }
                    crate::ast::Ast::TypeDef::Alias { .. } | crate::ast::Ast::TypeDef::Newtype { .. } => {}
                }
                // 类型方法检查
                self.push_self_type(self_ty);
                // 先注册所有方法为函数到 env（支持裸名方法调用 method(recv, args) 语法糖），
                // 再检查方法体（避免前向引用问题）
                for method in methods.iter() {
                    let m_param_types: Vec<TypeHandle> = method.params.iter().map(|p| {
                        if self.is_self_param(p.type_annotation, ast) {
                            self_ty
                        } else {
                            match p.type_annotation {
                                Some(ta) => self.type_from_ast(ta, ast),
                                None => self.arena.fresh_type_var(),
                            }
                        }
                    }).collect();
                    let m_ret_ty_raw = match method.return_type {
                        Some(rt) => self.type_from_ast(rt, ast),
                        None => self.arena.fresh_type_var(),
                    };
                    // async 方法：返回类型包装为 Async<T>（与顶层 FunDecl/predeclare 对齐，根因D）
                    let m_ret_ty = self.wrap_async_return(m_ret_ty_raw, method.is_async);
                    let m_fn_ty = self.arena.make_fn(
                        m_param_types.into_boxed_slice(),
                        m_ret_ty,
                    );
                    self.env.define(env, method.name, m_fn_ty);
                }
                for method in methods.iter() {
                    if let Some(body) = method.body {
                        let method_env = self.env.child(env);
                        for param in method.params.iter() {
                            let param_ty = if self.is_self_param(param.type_annotation, ast) {
                                self.infer_self_param(param.type_annotation, ast)
                            } else {
                                match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                }
                            };
                            self.env.define(method_env, param.name, param_ty);
                        }
                        let prev_return = self.expected_return;
                        let ret_ty_raw = method.return_type.map(|rt| self.type_from_ast(rt, ast));
                        // async 方法：返回类型包装为 Async<T>（与 FunDecl 一致，根因D），
                        // unify_return_type 会穿透 Async 层统一内层类型与函数体类型。
                        let ret_ty = ret_ty_raw.map(|t| self.wrap_async_return(t, method.is_async));
                        self.expected_return = ret_ty;
                        let body_ty = self.infer_expr(body, ast, method_env, ret_ty);
                        self.expected_return = prev_return;
                        // 统一方法体类型与声明返回类型（与 FunDecl 一致）：
                        // - 无标注返回类型：ret_ty 为 None → fresh_type_var，用 unify_or_constrain 绑定
                        // - 有标注返回类型：用 unify_return_type 统一（处理 async 穿透），
                        //   失败时注册 Equality 约束供 solver 延迟重试
                        // 这使 FieldAccess 等不依赖 expected 的表达式产生的 fresh var
                        // 能被 ret_ty 约束求解，避免成为孤儿 TypeVar。
                        let ret = ret_ty.unwrap_or_else(|| self.arena.fresh_type_var());
                        if method.return_type.is_none() {
                            self.unify_or_constrain(ret, body_ty);
                        } else if self.unify_return_type(ret, body_ty).is_err() {
                            self.solver.add_equality(ret, body_ty);
                        }
                    }
                }
                self.pop_self_type();
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
            }
            Decl::TraitDecl { name, type_params, methods, .. } => {
                // 注册嵌套 trait 定义到 sema_result（使 trait 类型标注可被识别）
                ast_trait_decl_to_trait_def(self.arena, self.sema_result, name, methods, ast);
                // 类型参数绑定（含 kind 注册）：使 trait 块内部引用泛型参数时可从 type_binding_stack 解析
                if !type_params.is_empty() {
                    self.push_type_bindings(
                        &type_params.iter().map(|tp| {
                            (tp.name, tp.kind.as_ref().map(|k| SemKind::from_ast(k)))
                        }).collect::<Vec<_>>(),
                    );
                }
                let self_var = self.push_self_type_var();
                for method in methods.iter() {
                    if let Some(body) = method.body {
                        let method_env = self.env.child(env);
                        for param in method.params.iter() {
                            let param_ty = if self.is_self_param(param.type_annotation, ast) {
                                self.infer_self_param(param.type_annotation, ast)
                            } else {
                                match param.type_annotation {
                                    Some(ta) => self.type_from_ast(ta, ast),
                                    None => self.arena.fresh_type_var(),
                                }
                            };
                            self.env.define(method_env, param.name, param_ty);
                        }
                        let prev_return = self.expected_return;
                        let ret_ty_raw = method.return_type.map(|rt| self.type_from_ast(rt, ast));
                        // async 默认方法：返回类型包装为 Async<T>（与 FunDecl 一致，根因D），
                        // unify_return_type 会穿透 Async 层统一内层类型与函数体类型。
                        let ret_ty = ret_ty_raw.map(|t| self.wrap_async_return(t, method.is_async));
                        self.expected_return = ret_ty;
                        let body_ty = self.infer_expr(body, ast, method_env, ret_ty);
                        self.expected_return = prev_return;
                        // 统一方法体类型与声明返回类型（与 FunDecl 一致）：
                        // - 无标注返回类型：ret_ty 为 None → fresh_type_var，用 unify_or_constrain 绑定
                        // - 有标注返回类型：用 unify_return_type 统一（处理 async 穿透），
                        //   失败时注册 Equality 约束供 solver 延迟重试
                        // 这使 FieldAccess 等不依赖 expected 的表达式产生的 fresh var
                        // 能被 ret_ty 约束求解，避免成为孤儿 TypeVar。
                        let ret = ret_ty.unwrap_or_else(|| self.arena.fresh_type_var());
                        if method.return_type.is_none() {
                            self.unify_or_constrain(ret, body_ty);
                        } else if self.unify_return_type(ret, body_ty).is_err() {
                            self.solver.add_equality(ret, body_ty);
                        }
                    }
                }
                self.pop_self_type();
                if !type_params.is_empty() {
                    self.pop_type_bindings();
                }
                let _ = (name, self_var);
            }
            _ => {}
        }
    }

    /// 对模块中所有类型注解运行 kind_check。
    fn run_kind_checks(&mut self, module: &Module<'_>) {
        let mut errors = Vec::new();
        for decl in module.declarations.iter() {
            match &decl.node {
                Decl::FunDecl { params, return_type, .. } => {
                    for p in params.iter() {
                        if let Some(ta) = p.type_annotation {
                            check_type_node(self.sema_result, &module.arena, ta, &[], &mut errors);
                        }
                    }
                    if let Some(rt) = return_type {
                        check_type_node(self.sema_result, &module.arena, *rt, &[], &mut errors);
                    }
                }
                Decl::TypeDecl { def: crate::ast::Ast::TypeDef::Adt { constructors }, .. } => {
                    for ctor in constructors.iter() {
                        for f in ctor.fields.iter() {
                            check_type_node(
                                self.sema_result,
                                &module.arena,
                                f.ty,
                                &[],
                                &mut errors,
                            );
                        }
                        if let Some(rt) = ctor.return_type {
                            check_type_node(
                                self.sema_result,
                                &module.arena,
                                rt,
                                &[],
                                &mut errors,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        for e in errors {
            self.sema_result.add_error(e);
        }
    }
}

/// 内置标量名 → Ty（单一派生点，替代原三处重复 match）。
///
/// 派生自 `Type::BUILTIN_TABLE`：按 name 查 ValueTag，再按 ValueTag 分派 Ty。
/// name→ValueTag 映射来自单一真相源。
///
/// 类型名统一为小写（与 .kz 源码语法一致）：null/void/bool/char/str 及各数值类型。
fn name_to_concrete(name: &str) -> Option<Ty> {
    use crate::types::{builtin_info_by_name, ValueTag};
    let info = builtin_info_by_name(name)?;
    let ct = match info.value_tag {
        ValueTag::I8 => Ty::I8,
        ValueTag::I16 => Ty::I16,
        ValueTag::I32 => Ty::I32,
        ValueTag::I64 => Ty::I64,
        ValueTag::I128 => Ty::I128,
        ValueTag::U8 => Ty::U8,
        ValueTag::U16 => Ty::U16,
        ValueTag::U32 => Ty::U32,
        ValueTag::U64 => Ty::U64,
        ValueTag::U128 => Ty::U128,
        ValueTag::Isize => Ty::Isize,
        ValueTag::Usize => Ty::Usize,
        ValueTag::F16 => Ty::F16,
        ValueTag::F32 => Ty::F32,
        ValueTag::F64 => Ty::F64,
        ValueTag::F128 => Ty::F128,
        ValueTag::Bool => Ty::Bool,
        ValueTag::Char => Ty::Char,
        ValueTag::Ref => Ty::Str,   // str 的 value_tag 是 Ref
        ValueTag::Null => Ty::Null,
        ValueTag::Void => Ty::Void,
    };
    Some(ct)
}

/// 返回所有数值内置类型名 + Ty（派生自 BUILTIN_TABLE）。
///
/// 替代原静态 `NUMERIC_BUILTIN_NAMES` 表，自动同步 BUILTIN_TABLE 变更。
/// 包含所有标量（含 bool/char，与原表一致），排除 str/null/void。
fn numeric_builtin_names() -> Vec<(&'static str, Ty)> {
    use crate::types::{BUILTIN_TABLE, ValueTag};
    BUILTIN_TABLE.iter()
        .filter(|s| !matches!(s.value_tag, ValueTag::Ref | ValueTag::Null | ValueTag::Void))
        .filter_map(|s| {
            let ct = name_to_concrete(s.name)?;
            Some((s.name, ct))
        })
        .collect()
}

// =========================================================================
// sema v2: Constraint Solver — 统一约束求解引擎
//
// 设计理念（原创，非照搬 GHC/rustc/Swift）：
// - 所有类型关系（相等、子类型、trait bound、narrowing）统一为 Constraint
// - snapshot/rollback 支持尝试性推断（match 分支、重载选择）
// - 批量求解：函数体结束时统一求解，而非立即 unify
// - DOD：约束用 Vec，snapshot 用长度索引，subst 用 FxHashMap
//
// 与现有 TypeArena::unify 的关系：
// solver 调用 unify 实现 Equality 约束，但增加延迟和回滚能力。
// 现有的立即 unify 调用保持兼容，新代码可选用 solver。
// =========================================================================

/// 约束种类：统一所有类型关系为约束。
///
/// 设计要点：
/// - Equality：最常见，直接调用 TypeArena::unify
/// - Subtype：调用 is_subtype，失败时记录错误但不立即中断
/// - TraitBound：ty 是否实现某 trait（延迟到 witness table 查询）
/// - Narrow：path-sensitive 窄化（flow narrowing 使用）
#[derive(Debug, Clone)]
pub enum Constraint {
    /// 类型相等约束：`t1 = t2`
    Equality(TypeHandle, TypeHandle),
    /// 子类型约束：`sub <: sup`（方向性，非对称）
    Subtype(TypeHandle, TypeHandle),
    /// Trait bound 约束：`ty` 实现 trait `trait_name<type_args>`
    TraitBound {
        ty: TypeHandle,
        trait_name: Box<str>,
        type_args: Box<[TypeHandle]>,
    },
    /// 窄化约束：在某路径上 `original` 被窄化为 `narrowed`
    /// 用于 flow-sensitive narrowing（NonNull/IsCheck/ConstructorMatch)
    Narrow {
        path: Box<str>,
        original: TypeHandle,
        narrowed: TypeHandle,
    },
}

/// 从约束提取关联的 span 信息（用于错误定位）。
/// Constraint 自身不携带 span，由约束生成处的上下文单独传入。
/// 保留 line/column 字段以兼容 ConstraintError，求解器填 0 表示"无 span"。
impl Constraint {
    /// 约束类型的可读名称。
    pub fn kind_str(&self) -> &'static str {
        match self {
            Constraint::Equality(..) => "Equality",
            Constraint::Subtype(..) => "Subtype",
            Constraint::TraitBound { .. } => "TraitBound",
            Constraint::Narrow { .. } => "Narrow",
        }
    }
}

/// 约束求解错误：记录求解失败的原因，不中断推断（错误恢复）。
#[derive(Debug, Clone)]
pub struct ConstraintError {
    pub constraint: Constraint,
    pub reason: Box<str>,
    /// span 信息：约束生成处可传入，求解器内部生成的错误填 0,0。
    pub line: u32,
    pub column: u32,
}

/// 约束求解器：收集约束、批量求解。
///
/// 设计：
/// - `pending`：待求解约束队列（FIFO）
/// - `subst`：已求解的 TypeVar → TypeHandle 映射（求解结果）
/// - `errors`：求解失败记录（不中断，错误恢复）
pub struct ConstraintSolver {
    pending: Vec<Constraint>,
    subst: FxHashMap<u32, TypeHandle>,
    errors: Vec<ConstraintError>,
    /// 每个 TypeVar 在不动点迭代中收到的所有候选绑定（多值记录）。
    ///
    /// key = TypeVar idx，value = 该 TypeVar 被要求绑定的所有目标类型 handle 列表。
    /// 不动点收敛后由 `finalize_solution` 去重并检测歧义：
    /// - 唯一候选 → 写入 subst
    /// - 多个不同候选 → 标记歧义错误（仍选 arena 的实际解写入 subst 以避免级联误报）
    candidates: FxHashMap<u32, Vec<TypeHandle>>,
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintSolver {
    pub fn new() -> Self {
        ConstraintSolver {
            pending: Vec::new(),
            subst: FxHashMap::default(),
            errors: Vec::new(),
            candidates: FxHashMap::default(),
        }
    }

    /// 添加约束到待求解队列。
    #[inline]
    pub fn add(&mut self, c: Constraint) {
        self.pending.push(c);
    }

    /// 添加相等约束的便捷方法。
    #[inline]
    pub fn add_equality(&mut self, t1: TypeHandle, t2: TypeHandle) {
        self.add(Constraint::Equality(t1, t2));
    }

    /// 添加子类型约束的便捷方法。
    #[inline]
    pub fn add_subtype(&mut self, sub: TypeHandle, sup: TypeHandle) {
        self.add(Constraint::Subtype(sub, sup));
    }

    /// 添加 trait bound 约束的便捷方法。
    #[inline]
    pub fn add_trait_bound(
        &mut self,
        ty: TypeHandle,
        trait_name: &str,
        type_args: &[TypeHandle],
    ) {
        self.add(Constraint::TraitBound {
            ty,
            trait_name: trait_name.into(),
            type_args: type_args.to_vec().into_boxed_slice(),
        });
    }

    /// 批量求解所有 pending 约束。
    ///
    /// 求解策略：
    /// 1. Equality → TypeArena::unify，成功则更新 subst
    /// 2. Subtype → is_subtype 检查，失败记录错误
    /// 3. TraitBound → 通过 witness table 查询（需传入 witness_table）
    /// 4. Narrow → 更新 flow fact table（phase 3 实现）
    ///
    /// 求解后 pending 清空，结果存入 subst 和 errors。
    pub fn solve(&mut self, arena: &mut TypeArena) {
        self.solve_with_witness(arena, None)
    }

    /// 批量求解所有 pending 约束（带 witness table 支持）。
    ///
    /// 不动点迭代：重复扫描约束队列，直到一轮无新绑定产生。
    /// 约束间存在依赖关系（约束 A 依赖约束 B 先绑定某 TypeVar），
    /// 单遍 FIFO 会因时序问题漏解；不动点迭代通过重试消除时序依赖。
    ///
    /// - Equality：两边仍含 TypeVar 时重新入队等待下一轮；两边都是具体类型时记入 errors
    /// - TraitBound：ty 仍是 TypeVar 时重新入队；否则查 witness table 判定
    /// - Subtype/Narrow：单遍处理（不涉及 TypeVar 绑定传播）
    pub fn solve_with_witness(&mut self, arena: &mut TypeArena, witness: Option<&WitnessTable>) {
        const MAX_ITERATIONS: usize = 1000;
        let mut pending = std::mem::take(&mut self.pending);

        for _iteration in 0..MAX_ITERATIONS {
            if pending.is_empty() {
                break;
            }

            // 取出当前所有约束，本轮处理
            let current = std::mem::take(&mut pending);
            let mut changed = false;

            for c in current {
                match c {
                    Constraint::Equality(t1, t2) => {
                        // 在 resolve/unify 之前记录候选（多值记录）。
                        // arena.get 返回原始 Ty，即使 TypeVar 已被
                        // 之前的 unify 绑定，get 仍返回 TypeVar(idx)，
                        // 因此能捕捉到所有约束路径对该 TypeVar 的绑定要求。
                        self.record_candidate(arena, t1, t2);

                        let r1 = arena.resolve(t1);
                        let r2 = arena.resolve(t2);

                        // 两边都已解析为同一类型，无需处理
                        if r1 == r2 {
                            continue;
                        }

                        match arena.unify(r1, r2) {
                            Ok(()) => {
                                changed = true;
                            }
                            Err(_) => {
                                // unify 失败：若两边仍含 TypeVar，重新入队等待下一轮
                                // （其他约束可能在本轮绑定这些 TypeVar）
                                let r1_has_var = Self::resolve_has_type_var(arena, r1);
                                let r2_has_var = Self::resolve_has_type_var(arena, r2);
                                if r1_has_var || r2_has_var {
                                    pending.push(Constraint::Equality(t1, t2));
                                } else {
                                    // 两边都是具体类型且不匹配：真错误
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::Equality(t1, t2),
                                        reason: "type mismatch".into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                        }
                    }
                    Constraint::Subtype(sub, sup) => {
                        if !is_subtype(arena, sub, sup) {
                            self.errors.push(ConstraintError {
                                constraint: Constraint::Subtype(sub, sup),
                                reason: "not a subtype".into(),
                                line: 0,
                                column: 0,
                            });
                        }
                    }
                    Constraint::TraitBound { ty, trait_name, type_args } => {
                        let resolved = arena.resolve(ty);
                        // ty 仍是 TypeVar：重新入队等待下一轮
                        if matches!(arena.get(resolved), Ty::TypeVar(_)) {
                            pending.push(Constraint::TraitBound {
                                ty,
                                trait_name,
                                type_args,
                            });
                            continue;
                        }

                        // ty 已解析：查 witness table 判定
                        if let Some(wt) = witness {
                            let ct = arena.get(resolved);
                            let type_id = match ct {
                                Ty::Adt(_) | Ty::Generic(_) => {
                                    // 用户类型：type_id 由外部注册
                                    // 此处无法访问 sema_result，跳过（由 check_module 统一处理）
                                    None
                                }
                                _ => ct.type_id(),
                            };
                            if let Some(tid) = type_id {
                                if !wt.implements(&trait_name, tid) {
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::TraitBound {
                                            ty,
                                            trait_name: trait_name.clone(),
                                            type_args: type_args.clone(),
                                        },
                                        reason: format!(
                                            "type does not implement trait '{}'",
                                            trait_name
                                        )
                                        .into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                            // type_id 为 None 时延迟到 check_module 处理
                        }
                    }
                    Constraint::Narrow { original, narrowed, .. } => {
                        // 窄化约束：在特定路径上 original 被窄化为 narrowed。
                        // 求解策略：若 original 是未绑定的 TypeVar，绑定到 narrowed；
                        // 若 original 已绑定，尝试 unify（窄化类型必须与原类型兼容）。
                        let r_orig = arena.resolve(original);
                        let r_narrow = arena.resolve(narrowed);
                        if let Ty::TypeVar(idx) = arena.get(r_orig).clone() {
                            // TypeVar 未绑定：直接绑定到窄化类型
                            arena.type_vars[idx as usize].bound = Some(r_narrow);
                            changed = true;
                        } else if r_orig != r_narrow {
                            // 已绑定：尝试 unify（窄化类型必须是原类型的子类型）
                            match arena.unify(r_orig, r_narrow) {
                                Ok(()) => { changed = true; }
                                Err(_) => {
                                    // 窄化与原类型冲突：记录但不中断
                                    self.errors.push(ConstraintError {
                                        constraint: Constraint::Narrow {
                                            path: String::new().into_boxed_str(),
                                            original,
                                            narrowed,
                                        },
                                        reason: "narrowed type conflicts with original".into(),
                                        line: 0,
                                        column: 0,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            // 不动点：一轮无新绑定且无重新入队的约束，结束
            if !changed {
                break;
            }
        }

        // 超过 MAX_ITERATIONS 仍未收敛的约束：记录但不报错（防御性）
        // 这些通常是 TypeVar ↔ TypeVar 的循环依赖，不影响正确性

        // 不动点收敛后：从 candidates 构建 subst，检测歧义
        self.finalize_solution(arena);
    }

    /// 判断 resolve 后的 TypeHandle 是否仍含未绑定 TypeVar。
    /// 用于不动点迭代中决定是否重新入队约束。
    fn resolve_has_type_var(arena: &TypeArena, ty: TypeHandle) -> bool {
        let resolved = arena.resolve(ty);
        match arena.get(resolved) {
            Ty::TypeVar(_) => true,
            Ty::Fn(_) => {
                let (params, return_type) = arena.fn_parts(resolved);
                params.iter().any(|&p| Self::resolve_has_type_var(arena, p))
                    || Self::resolve_has_type_var(arena, return_type)
            }
            Ty::Nullable(_) => {
                Self::resolve_has_type_var(arena, arena.nullable_inner(resolved))
            }
            Ty::Ref(_) => {
                let (inner, _) = arena.ref_parts(resolved);
                Self::resolve_has_type_var(arena, inner)
            }
            Ty::Adt(_) => {
                let (_, type_args) = arena.adt_parts(resolved);
                type_args.iter().any(|&a| Self::resolve_has_type_var(arena, a))
            }
            Ty::Throw(_) => {
                let (value_type, error_type) = arena.throw_parts(resolved);
                Self::resolve_has_type_var(arena, value_type)
                    || Self::resolve_has_type_var(arena, error_type)
            }
            Ty::Generic(_) => {
                let (_, args) = arena.generic_parts(resolved);
                args.iter().any(|&a| Self::resolve_has_type_var(arena, a))
            }
            Ty::Trait(_) => {
                let (_, type_args) = arena.trait_parts(resolved);
                type_args.iter().any(|&a| Self::resolve_has_type_var(arena, a))
            }
            Ty::Array(_) => {
                let (element_type, _) = arena.array_parts(resolved);
                Self::resolve_has_type_var(arena, element_type)
            }
            _ => false,
        }
    }

    /// 记录 TypeVar 的候选绑定到 candidates（多值记录）。
    ///
    /// 在 unify **之前**调用，用 `arena.get`（原始 Ty，不 resolve）判断 TypeVar。
    /// 即使 TypeVar 已被先前 unify 绑定到具体类型，`get` 仍返回 `TypeVar(idx)`，
    /// 因此能捕捉到所有约束路径对该 TypeVar 的绑定要求，用于后续歧义检测。
    ///
    /// - 若 t1 是 TypeVar 且 t2 不是 → candidates[t1.idx].push(t2)
    /// - 若 t2 是 TypeVar 且 t1 不是 → candidates[t2.idx].push(t1)
    /// - 两边都是 TypeVar → 不记录（var-var 绑定由 unify 直接处理）
    fn record_candidate(&mut self, arena: &TypeArena, t1: TypeHandle, t2: TypeHandle) {
        match (arena.get(t1), arena.get(t2)) {
            (Ty::TypeVar(_), Ty::TypeVar(_)) => {
                // 两边都是 TypeVar：由 unify 处理 var-var 绑定，不记录候选
            }
            (Ty::TypeVar(idx), _) => {
                self.candidates.entry(idx).or_default().push(t2);
            }
            (_, Ty::TypeVar(idx)) => {
                self.candidates.entry(idx).or_default().push(t1);
            }
            _ => {}
        }
    }

    /// 不动点收敛后从 candidates 构建最终 subst，并检测歧义。
    ///
    /// 对每个 TypeVar 的候选集：
    /// 1. 基于 resolve 后的 TypeHandle 相等性去重
    /// 2. 唯一候选 → 写入 subst
    /// 3. 多个不同候选 → 标记歧义错误，仍选 arena 实际解写入 subst（避免级联误报）
    fn finalize_solution(&mut self, arena: &mut TypeArena) {
        let candidates = std::mem::take(&mut self.candidates);
        for (idx, cands) in candidates {
            // 去重：基于 resolve 后的 TypeHandle 相等性
            let mut unique: Vec<TypeHandle> = Vec::new();
            for c in &cands {
                let r = arena.resolve(*c);
                if !unique.iter().any(|&u| arena.resolve(u) == r) {
                    unique.push(r);
                }
            }

            match unique.len() {
                0 => {} // 不可能（cands 非空才会迭代）
                1 => {
                    // 唯一候选：写入 subst 并回写 arena.type_vars[idx].bound
                    // 回写是关键：诊断检查的是 arena.type_vars[idx].bound，
                    // 若不回写，已求解的 TypeVar 仍被判为 unresolved
                    let resolved = arena.resolve(unique[0]);
                    self.subst.insert(idx, resolved);
                    arena.type_vars[idx as usize].bound = Some(resolved);
                }
                _ => {
                    // 多个不同候选：歧义
                    // 选 arena 实际解（unify 已选第一个成功的）写入 subst，避免级联误报
                    let resolved = arena.resolve(cands[0]);
                    self.subst.insert(idx, resolved);
                    arena.type_vars[idx as usize].bound = Some(resolved);
                    // 记录歧义错误
                    self.errors.push(ConstraintError {
                        constraint: Constraint::Equality(unique[0], unique[1]),
                        reason: format!(
                            "ambiguous inference for TypeVar{}: {} distinct candidates",
                            idx,
                            unique.len()
                        )
                        .into(),
                        line: 0,
                        column: 0,
                    });
                }
            }
        }
    }

    /// 查询 TypeVar 的求解结果。
    #[inline]
    pub fn lookup_subst(&self, var_idx: u32) -> Option<TypeHandle> {
        self.subst.get(&var_idx).copied()
    }

    /// 获取所有求解错误。
    #[inline]
    pub fn errors(&self) -> &[ConstraintError] {
        &self.errors
    }

    /// 是否有求解错误。
    #[inline]
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// 待求解约束数量。
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// 清空所有状态（模块切换时调用）。
    pub fn reset(&mut self) {
        self.pending.clear();
        self.subst.clear();
        self.errors.clear();
        self.candidates.clear();
    }
}


// =========================================================================
// sema v2: Flow-Sensitive Narrowing — 通用 flow fact 系统
//
// 设计理念（原创，非照搬 Kotlin/TS）：
// - 把 Zig 版的 nullable narrowing 泛化为通用 flow fact 系统
// - 支持 NonNull / IsCheck / ConstructorMatch 三种窄化
// - DOD：flow facts 用 arena 索引，scope 用栈式管理
// - 查询：lookup_narrowed(path) -> Option<TypeHandle>
//
// 与 constraint solver 的关系：
// Narrowing 约束通过 FlowContext 管理，不直接进 solver 队列。
// FlowContext 是 path-sensitive 的，solver 是 path-insensitive 的。
// =========================================================================

/// 窄化种类：覆盖 Kuzo 的所有 flow-sensitive 类型精化场景。
#[derive(Debug, Clone)]
pub enum NarrowKind {
    /// 非空窄化：`if x != null` → x 从 `Nullable<T>` 窄化为 `T`
    NonNull,
    /// 类型判断窄化：`if x is Type` → x 窄化为 Type
    /// （Kuzo 的 `is` 表达式，类似 Kotlin 的 smart cast）
    IsCheck(TypeHandle),
    /// ADT 构造器匹配窄化：`match x { Some(v) => ... }` → x 窄化为 `Some<T>`
    /// （GADT 类型精化，构造器匹配后类型变量获得具体信息）
    ConstructorMatch {
        /// 构造器名（如 "Some"、"None"、"Ok"、"Err"）
        ctor_name: Box<str>,
        /// 绑定的子模式变量名（用于子模式类型精化）
        bound_vars: Box<[Box<str>]>,
    },
}

/// Flow fact：在某程序点对某路径的类型窄化断言。
///
/// `path` 是表达式的规范化路径（如 "x"、"obj.field"、"a.b.c"），
/// 用于在不同位置引用同一表达式。
#[derive(Debug, Clone)]
pub struct FlowFact {
    /// 表达式路径（规范化为字符串）
    pub path: Box<str>,
    /// 窄化后的类型
    pub narrowed_ty: TypeHandle,
    /// 窄化条件
    pub kind: NarrowKind,
}

/// Flow fact 表：存储当前 scope 内的所有 flow facts。
///
/// DOD：facts 用 Vec，by_path 用 FxHashMap 索引。
#[derive(Default)]
pub struct FlowFactTable {
    facts: Vec<FlowFact>,
    /// 按路径索引：path → fact indices
    by_path: FxHashMap<Box<str>, Vec<u32>>,
}

impl FlowFactTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加 flow fact。
    pub fn add(&mut self, fact: FlowFact) {
        let idx = self.facts.len() as u32;
        self.by_path
            .entry(fact.path.clone())
            .or_default()
            .push(idx);
        self.facts.push(fact);
    }

    /// 查询某路径的最新窄化类型。
    ///
    /// 返回该路径最后一次窄化的类型（同一路径可能有多次窄化，
    /// 取最新的——facts 是追加顺序，最后添加的最新）。
    pub fn lookup(&self, path: &str) -> Option<TypeHandle> {
        self.by_path
            .get(path)
            .and_then(|indices| indices.last())
            .and_then(|&idx| self.facts.get(idx as usize))
            .map(|f| f.narrowed_ty)
    }

    /// 查询某路径的最新 flow fact（含 kind）。
    pub fn lookup_fact(&self, path: &str) -> Option<&FlowFact> {
        self.by_path
            .get(path)
            .and_then(|indices| indices.last())
            .and_then(|&idx| self.facts.get(idx as usize))
    }

    /// 当前 scope 的 fact 数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// Flow context：栈式管理 flow fact scopes。
///
/// 进入 if/match 分支时 push 新 scope，离开时 pop。
/// 查询时从栈顶向下查找（内层 scope 覆盖外层）。
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
            scopes: vec![FlowFactTable::new()], // 根 scope
        }
    }

    /// 进入新 scope（if/match 分支）。
    pub fn push_scope(&mut self) {
        self.scopes.push(FlowFactTable::new());
    }

    /// 离开 scope。
    ///
    /// 不会弹出根 scope（保持至少一层）。
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// 在当前（栈顶）scope 添加 flow fact。
    pub fn add_fact(&mut self, fact: FlowFact) {
        if let Some(top) = self.scopes.last_mut() {
            top.add(fact);
        }
    }

    /// 查询某路径的窄化类型：从栈顶向下查找。
    ///
    /// 内层 scope 的窄化覆盖外层（path-sensitive）。
    ///
    /// Bug #35: ConstructorMatch 的 fact 表示 scrutinee 变量被窄化为 ADT 类型。
    /// 如果该 fact 的 bound_vars 包含 path，说明 path 已被模式变量遮蔽
    /// （如 `match w { W3(w) => w * w }` 中模式变量 w 遮蔽参数 w）。
    /// 此时 scrutinee 的窄化类型不适用于模式变量，应跳过此 fact，
    /// 让 infer_expr 走 env 查询获取模式变量的正确字段类型。
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

    /// 查询某路径的 flow fact（含 kind）：从栈顶向下查找。
    pub fn lookup_fact(&self, path: &str) -> Option<&FlowFact> {
        for scope in self.scopes.iter().rev() {
            if let Some(fact) = scope.lookup_fact(path) {
                return Some(fact);
            }
        }
        None
    }

    /// 当前 scope 深度。
    #[inline]
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// 重置到根 scope（函数切换时调用）。
    pub fn reset(&mut self) {
        self.scopes.truncate(1);
        self.scopes[0] = FlowFactTable::new();
    }
}

/// 从 `if cond { ... }` 的条件表达式提取 flow facts。
///
/// 返回 (then_facts, else_facts)：
/// - then_facts：then 分支成立的窄化
/// - else_facts：else 分支成立的窄化（条件取反）
///
/// 支持：
/// - `x != null` → then: NonNull(x), else: 无
/// - `x == null` → then: 无, else: NonNull(x)
/// - `x is Type` → then: IsCheck(x, Type), else: 无
///
/// （ConstructorMatch 由 match 表达式处理，不在此函数）
pub fn analyze_null_check_facts(
    arena: &TypeArena,
    ast: &AstArena<'_>,
    cond: ExprId,
    env: EnvId,
    env_arena: &EnvArena,
) -> (Vec<FlowFact>, Vec<FlowFact>) {
    let mut then_facts = Vec::new();
    let mut else_facts = Vec::new();

    // 若 `path_expr` 是 nullable 变量路径且 `null_expr` 是 null 字面量，
    // 则向 `facts` 推入 NonNull 窄化事实。
    let push_nonnull = |path_expr: ExprId, null_expr: ExprId, facts: &mut Vec<FlowFact>| {
        if let Some(path) = expr_path(ast, path_expr) {
            if matches!(ast.expr(null_expr).node, Expr::NullLit) {
                if let Some(ty) = env_arena.lookup(env, &path) {
                    let resolved = arena.resolve(ty);
                    if let Ty::Nullable(_) = arena.get(resolved) {
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
                // `x != null` / `null != x` → then: NonNull(x)
                push_nonnull(*lhs, *rhs, &mut then_facts);
                push_nonnull(*rhs, *lhs, &mut then_facts);
            }
            crate::ast::Ast::BinaryOp::Eq => {
                // `x == null` / `null == x` → else: NonNull(x)
                push_nonnull(*lhs, *rhs, &mut else_facts);
                push_nonnull(*rhs, *lhs, &mut else_facts);
            }
            _ => {}
        }
    }

    (then_facts, else_facts)
}

/// 提取表达式的规范化路径（用于 flow narrowing 标识）。
///
/// 支持：
/// - `Ident(name)` → `name`
/// - `FieldAccess(recv, field)` → `{recv_path}.{field}`
/// - 其他 → None（不可窄化）
fn expr_path(ast: &AstArena<'_>, expr: ExprId) -> Option<String> {
    match &ast.expr(expr).node {
        Expr::Ident(name) => Some((*name).to_string()),
        Expr::FieldAccess { recv, field } => {
            let recv_path = expr_path(ast, *recv)?;
            Some(format!("{}.{}", recv_path, field))
        }
        _ => None,
    }
}

/// 从模式节点提取构造器名和绑定的变量名（用于 ConstructorMatch narrowing）。
///
/// 仅处理 `Constructor { name, patterns }` 模式，提取构造器名和
/// 子模式中所有 `Variable` 绑定的变量名。
///
/// 其他模式（Wildcard/Literal/Variable/Record/Or/Guard）返回 None。
fn extract_constructor_pattern<'a>(
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

/// 递归收集模式中绑定的变量名（Variable 模式的 name）。
fn collect_pattern_binds<'a>(
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

