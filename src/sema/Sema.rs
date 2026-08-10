//! Sema.rs — 语义分析核心数据结构
//!
//! 类型系统单一真相源为 `crate::types`（`Ty` / `TypeArena` / `TypeOps`）。
//! 不再依赖 `ConcreteType` / `TypeDescriptor`——干净切除，无兼容层。
//!
//! 关键设计：
//! - **arena + 索引**：递归子类型用 `TypeHandle(u32)` 索引 `TypeArena`，
//!   `unify`/`occurs`/`resolve` 为 `TypeArena` 方法。
//! - **`TypeVar` 身份 = 其在 `type_vars` 中的下标**。
//! - **`Box<str>` / `Box<[...]>`** 持有复合类型自身数据，所有权清晰。
//! - **`Result<(), UnifyError>`** 替代 Zig error union；`EnvArena` + `EnvId` 索引。
//!
//! 依赖关系：单向依赖 `crate::types`（`Ty` / `TypeArena` / `TypeOps` / `DynamicOpsRegistry`）
//! 以及 `crate::Ast`（`TypeRef`，仅 `CtorDefInfo` 的 GADT 回溯字段引用）。

use crate::ast::Ast::{
    AstArena, Decl, TypeNode, TypeRef as AstTypeRef,
};
use crate::types::{
    FIRST_DYNAMIC_TYPE_ID, type_def_index_of,
};
use rustc_hash::{FxHashMap, FxHashSet};

// 从 Type 模块 re-export 所有类型系统符号（打破 Type↔sema 循环依赖）。
// sema 子模块（Inference.rs / Relations.rs / Monomorph.rs）通过
// `use crate::sema::Sema::*;` glob import 获取这些符号。
pub use crate::types::{
    TypeHandle, Ty, TypeFamily, DetailId, EnvId, FieldType, TraitMethodSig,
    SemKind, TypeVar, UnifyError,
    TypeArena, TypeDetail, TypeDisplay,
    TypeOps, ops_of, ops_by_type_id,
    DynamicOpsRegistry, DynamicOpsEntry,
    RefOps, HeapRefOps,
    STR_TYPE_ID, NULL_TYPE_ID, VOID_TYPE_ID,
    FIRST_INT_TYPE_ID, LAST_INT_TYPE_ID,
    FIRST_FLOAT_TYPE_ID, LAST_FLOAT_TYPE_ID,
    dynamic_type_id,
};

// =========================================================================
// ConcreteEnv / EnvArena — 类型环境（替代旧 TypeEnv，无 TypeScheme）
// =========================================================================

/// 类型环境节点：自身绑定 + 可选父环境（通过索引共享）。
struct EnvNode {
    bindings: FxHashMap<String, TypeHandle>,
    parent: Option<EnvId>,
}

/// 类型环境 arena：以索引管理环境节点，支持父环境共享，无 `Rc`/`RefCell`。
pub struct EnvArena {
    envs: Vec<EnvNode>,
}

impl Default for EnvArena {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvArena {
    pub fn new() -> Self {
        EnvArena { envs: Vec::new() }
    }

    /// 创建顶层环境（无父环境）。
    pub fn root(&mut self) -> EnvId {
        let id = EnvId(self.envs.len() as u32);
        self.envs.push(EnvNode {
            bindings: FxHashMap::default(),
            parent: None,
        });
        id
    }

    /// 创建子环境，父环境为 `parent`。
    pub fn child(&mut self, parent: EnvId) -> EnvId {
        let id = EnvId(self.envs.len() as u32);
        self.envs.push(EnvNode {
            bindings: FxHashMap::default(),
            parent: Some(parent),
        });
        id
    }

    /// 在 `env` 中定义绑定；已存在同名绑定返回 `false`。
    pub fn define(&mut self, env: EnvId, name: &str, ty: TypeHandle) -> bool {
        let node = &mut self.envs[env.0 as usize];
        if node.bindings.contains_key(name) {
            return false;
        }
        node.bindings.insert(name.to_string(), ty);
        true
    }

    /// 在 `env` 中强制定义绑定（覆盖已存在的同名绑定）。
    ///
    /// 用于构造器注册：`register_module_aliases` 先注册模块路径别名（如 "DateTime" → ModuleRef），
    /// 随后 `predeclare_declarations` 注册构造器时需覆盖别名，使 `DateTime(...)` 解析为构造器而非 ModuleRef。
    pub fn redefine(&mut self, env: EnvId, name: &str, ty: TypeHandle) {
        let node = &mut self.envs[env.0 as usize];
        node.bindings.insert(name.to_string(), ty);
    }

    /// 自 `env` 向上查找名字（含父环境链）；未找到返回 `None`。
    pub fn lookup(&self, mut env: EnvId, name: &str) -> Option<TypeHandle> {
        loop {
            let node = &self.envs[env.0 as usize];
            if let Some(&ty) = node.bindings.get(name) {
                return Some(ty);
            }
            {
                let p = node.parent?;
                env = p
            }
        }
    }

    /// 仅在 `env` 自身查找名字（不含父环境链）；未找到返回 `None`。
    ///
    /// 用于模块限定访问（ModuleRef.field）：只搜索该模块自己的符号，
    /// 不穿透到父 env（避免 `std.io.File.println` 错误地找到全局 `println`）。
    pub fn lookup_local(&self, env: EnvId, name: &str) -> Option<TypeHandle> {
        self.envs[env.0 as usize].bindings.get(name).copied()
    }

    /// 自 `env` 向上查找名为 `name` 且满足 `pred` 的绑定（跳过不满足的同名绑定）。
    /// 用于方法调用 `recv.method(args)` → `method(recv, args)` 路径，
    /// 避免局部变量遮蔽同名自由函数。
    pub fn lookup_with_pred(
        &self,
        mut env: EnvId,
        name: &str,
        pred: impl Fn(TypeHandle) -> bool,
    ) -> Option<TypeHandle> {
        loop {
            let node = &self.envs[env.0 as usize];
            if let Some(&ty) = node.bindings.get(name) {
                if pred(ty) {
                    return Some(ty);
                }
            }
            {
                let p = node.parent?;
                env = p
            }
        }
    }
}

// =========================================================================
// ConstVal — 编译期常量值
// =========================================================================

/// 编译期常量值（对应 ir/meta.zig 的 `ConstVal`）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstVal {
    /// 整数字面量
    Int(i128),
    /// 浮点字面量的位模式（按目标 float 类型解释）
    Float(u128),
    /// 布尔字面量
    Bool(bool),
    /// 字符字面量（Unicode scalar value）
    Char(u32),
}

// =========================================================================
// SemaResult 辅助结构
// =========================================================================

/// 单个表达式的语义信息。
#[derive(Debug, Clone)]
pub struct ExprInfo {
    /// 表达式的类型句柄（决定通道宽度与读写 vtable）
    pub ty: TypeHandle,
    /// 编译期常量值（若表达式是常量）
    pub const_val: Option<ConstVal>,
    /// 表达式的 AST 句柄地址（用作 key）
    pub expr_id: u64,
    /// 表达式的类型名（adt/generic 等场景，消除 IR 侧 AST 回溯）
    pub type_name: Option<Box<str>>,
    /// 是否为 trait 对象（Ty::TraitObject）：IR 层据此走 vtable 动态分派，
    /// 而非按字符串值匹配 trait 名。适用于任何 trait（Iterator/Stream/Iterable 等）。
    pub is_trait_object: bool,
    /// 是否为 `&T` / `*T` 引用类型（运行时保持引用语义不深拷贝）
    pub is_ref_type: bool,
    /// 区分 `&T`(false) 与 `*T`(true)；仅 `is_ref_type=true` 时有效
    pub is_raw_ref: bool,
}

impl ExprInfo {
    /// 以给定 `ty` 构造最小 `ExprInfo`（其余字段为默认值）。
    pub fn new(ty: TypeHandle, expr_id: u64) -> Self {
        ExprInfo {
            ty,
            const_val: None,
            expr_id,
            type_name: None,
            is_trait_object: false,
            is_ref_type: false,
            is_raw_ref: false,
        }
    }
}

/// 类型定义种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeDefKind {
    /// 代数数据类型
    Adt,
    /// 记录类型
    Record,
    /// 类型别名
    Alias,
    /// newtype 包装
    Newtype,
}

/// 构造器定义信息（压平后的 sema AdtInfo 构造器）。
#[derive(Debug, Clone)]
pub struct CtorDefInfo {
    pub name: Box<str>,
    pub type_name: Box<str>,
    pub field_names: Box<[Option<Box<str>>]>,
    pub field_types: Box<[TypeHandle]>,
    pub is_newtype: bool,
    /// GADT 构造器返回类型名（仅 GADT 有效）
    pub return_type_name: Option<Box<str>>,
    /// GADT 构造器返回类型 TypeNode（消除 IR 侧 AST 回退）
    pub return_type_node: Option<AstTypeRef>,
    /// 字段类型的自包含表示（不依赖 AST 引用），用于跨模块完整还原字段类型
    /// （包括数组、Nullable、Ref 等复合类型）。
    /// 长度与 `field_names` 一致。
    pub field_type_reprs: Box<[TypeRepr]>,
}

/// 类型方法的签名信息，按 method_idx（在 type 块 methods 数组中的位置）索引。
///
/// 自包含的类型表示（不依赖 AST 引用），用于跨模块传递方法返回类型信息。
/// 在 build_method_sig_info 阶段从 AST TypeNode 转换，在 lookup_method_type 中
/// 通过 type_repr_to_handle 还原为 TypeHandle。
#[derive(Debug, Clone)]
pub enum TypeRepr {
    Named(Box<str>),
    SelfType,
    Generic(Box<str>, Box<[TypeRepr]>),
    Nullable(Box<TypeRepr>),
    Ref(Box<TypeRepr>),
    RawPtr(Box<TypeRepr>),
    Function(Box<[TypeRepr]>, Box<TypeRepr>),
    Array(Box<TypeRepr>, Option<u64>),
}

/// 内置 intrinsic 方法的降级策略。
///
/// 存储在 `MethodSigInfo.intrinsic` 中，IR 层通过 (type_id, method_idx) 查到
/// 方法签名后，根据此字段选择节点 kind 和 compute_fn，消除按方法名查表的特判。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntrinsicKind {
    /// 单节点一元运算（无参数）：len/close/bytes/cancel → compute_fn(idx)
    UnOp(u32),
    /// 挂起等待事件源（无参数）：await → Await 节点（无条件降级）
    Await,
    /// Channel 接收（无参数）：recv → Await 节点（仅 Channel/Receiver 类型）
    ChannelAwait,
    /// 二元运算（recv + 1 参数）：send(value) → compute_fn(idx)
    BinOp(u32),
}

/// 替代旧的 func_sigs mangled name（"TypeName.method"）注册方式，
/// 使方法分派通过 (type_id, method_idx) 结构化键驱动。
#[derive(Debug, Clone)]
pub struct MethodSigInfo {
    pub name: Box<str>,
    pub param_is_ref: Box<[bool]>,
    pub return_is_ref: bool,
    pub is_async: bool,
    pub is_throwing: bool,
    /// 参数类型的自包含表示（不依赖 AST 引用），用于跨模块完整还原参数类型
    /// （包括数组、Nullable、Ref 等复合类型）。
    pub param_type_reprs: Box<[TypeRepr]>,
    /// 返回类型的自包含表示（不依赖 AST 引用），用于跨模块完整解析嵌套泛型类型
    /// （如 Async<Throw<T, E>>）。
    pub return_type_repr: Option<TypeRepr>,
    /// intrinsic 降级策略：None 表示普通方法（有方法体或 trait 方法），
    /// Some 表示内置 intrinsic 方法（无方法体，IR 层直接降级为 compute_fn 节点）。
    pub intrinsic: Option<IntrinsicKind>,
}

/// 类型定义信息（替代 IRBuilder 的 type_table + ctor_table）。
#[derive(Debug, Clone)]
pub struct TypeDefInfo {
    pub name: Box<str>,
    pub kind: TypeDefKind,
    /// adt/newtype/error_newtype：构造器列表
    /// record：`constructors[0]` 存字段（name == type_name）
    /// alias：空切片
    pub constructors: Box<[CtorDefInfo]>,
    pub type_params: Box<[Box<str>]>,
    /// 仅 alias/newtype：目标类型名
    pub target_type_name: Option<Box<str>>,
    /// 仅 alias/newtype：目标类型描述符
    pub target_type: Option<TypeHandle>,
    /// 类型块内方法签名表，按 method_idx 索引（AST 声明顺序）。
    /// 空切片表示该类型无方法（alias / 无方法的 record/adt）。
    pub methods: Box<[MethodSigInfo]>,
}

/// Trait 定义信息（替代 IRBuilder 的 trait_table 签名部分）。
#[derive(Debug, Clone)]
pub struct TraitDefInfo {
    pub name: Box<str>,
    pub methods: Box<[TraitMethodSig]>,
}

/// 函数签名引用（嵌入 `ExprInfo`，仅对 callee 表达式有效）。
#[derive(Debug, Clone)]
pub struct FnSigRef {
    pub param_types: Box<[TypeHandle]>,
    pub return_type: TypeHandle,
    pub is_async: bool,
    pub is_throwing: bool,
}

/// 函数签名信息（替代 IRBuilder 的 func_generic_info）。
#[derive(Debug, Clone)]
pub struct FuncSigInfo {
    /// 函数名或 mangled 名（TypeName.method）
    pub name: Box<str>,
    pub type_params: Box<[Box<str>]>,
    pub return_type: TypeHandle,
    /// 每个参数是否为 `&T` 引用语义
    pub param_is_ref: Box<[bool]>,
    pub return_is_ref: bool,
    pub is_async: bool,
    pub is_throwing: bool,
}

/// Import 别名目标（区分模块引用和符号引用）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AliasTarget {
    /// 模块短名 → 完整模块路径
    /// `import std.time.Calendar` → "Calendar" → `AliasTarget::Module("std.time.Calendar")`
    Module(Box<str>),
    /// 函数/常量短名 → mangled 名
    /// `import std.time.Calendar { is_leap_year }` → "is_leap_year" → `AliasTarget::Symbol("std.time.Calendar.is_leap_year")`
    Symbol(Box<str>),
}

/// 通道布局（单态化实例的通道分配方案）。
#[derive(Debug, Clone)]
pub struct ChanLayout {
    pub local_chan_count: u16,
    pub return_channel: u16,
    pub local_offsets: Box<[u32]>,
    pub chan_types: Box<[TypeHandle]>,
    pub chan_total_bytes: u32,
}

impl ChanLayout {
    /// 空 layout（未计算通道分配时的占位）。
    pub fn empty() -> Self {
        ChanLayout {
            local_chan_count: 0,
            return_channel: 0,
            local_offsets: Box::new([]),
            chan_types: Box::new([]),
            chan_total_bytes: 0,
        }
    }
}

/// 字段访问元信息（运行时分派 field_id 查找）。
#[derive(Debug, Clone)]
pub struct FieldAccessInfo {
    pub obj_type: TypeHandle,
    pub field_idx: u16,
    pub field_type: TypeHandle,
}

/// 方法分派元信息（trait 方法 → 具体 impl 函数）。
#[derive(Debug, Clone, Copy)]
pub struct DispatchInfo {
    pub trait_id: u16,
    pub method_idx: u16,
    pub impl_fn_idx: u16,
    /// 泛型方法调用的单态化实例 ID（非泛型为 0）
    pub instance_id: u32,
    /// 语言级 intrinsic 标记（await/recv 等由 sema 统一识别，不依赖类型注册）。
    /// 用户自定义类型（Timer 等）未注册 MethodSigInfo.intrinsic，
    /// 但 await/recv 是通用语义，由 sema 在 MethodCall 推断时标记此处。
    pub intrinsic: Option<IntrinsicKind>,
}

/// reflect 已解析元信息。
#[derive(Debug, Clone, Copy)]
pub struct ReflectMeta {
    pub ty: TypeHandle,
}

/// 单态化实例（一个泛型函数 + 一组 type_args → 一个实例）。
#[derive(Debug, Clone)]
pub struct MonomorphInstance {
    pub instance_id: u32,
    pub func_name: Box<str>,
    /// 函数所在模块名（用于 expr_types 复合 key，确保跨模块单态化时 key 一致）
    pub module_name: Box<str>,
    pub type_args: Box<[TypeHandle]>,
    pub chan_layout: ChanLayout,
    pub return_type: TypeHandle,
    pub is_async: bool,
    /// 实例本地表达式类型表（key = module_expr_key(module_name, expr_id)）
    pub expr_types: FxHashMap<u64, ExprInfo>,
    /// 字段访问元信息（key = AST field_access Expr 句柄地址）
    pub field_accesses: FxHashMap<u64, FieldAccessInfo>,
}

/// trait 默认方法单态化实例。
///
/// 每个实现 trait 但未显式覆写该方法的类型，对应一个特化实例。
/// 由 `Monomorph::collect_trait_default_instances` 在 Sema 后阶段收集，
/// 供 IR 层（IrBuilder）预注册并编译特化子图。
///
/// 键语义：`(type_id, trait_idx, method_idx)` 与 `Ir.trait_default_subgraphs` 的键一致。
#[derive(Debug, Clone)]
pub struct TraitDefaultInstance {
    /// 实现类型的 type_id（与 Ty 的 type_id 对应）
    pub type_id: u16,
    /// 实现类型名（如 "Lt"、"Ordering"）
    pub type_name: Box<str>,
    /// trait 在 `trait_defs` 中的索引
    pub trait_idx: u16,
    /// trait 名（如 "Greet"、"Show"）
    pub trait_name: Box<str>,
    /// 方法在 trait methods 中的索引（有 body 的默认方法）
    pub method_idx: u16,
}

/// 协程元数据（async 函数状态机变换产物）。
///
/// Sema 输出的最小元数据：func_idx 定位函数，segment_count 描述状态段数。
/// 完整的状态机变换（段/帧/defer/catch/loop 表）由 IR 层基于此元数据构建，
/// 不在 Sema 层维护，保持 Sema 与 IR 的职责分离。
#[derive(Debug, Clone)]
pub struct CoroutineMeta {
    /// async 函数索引（functions 表中的索引）
    pub func_idx: u16,
    /// 状态段数
    pub segment_count: u16,
}

// =========================================================================
// SemaError — 语义错误
// =========================================================================

/// 语义错误。
#[derive(Debug, Clone)]
pub struct SemaError {
    pub message: Box<str>,
    pub line: u32,
    pub column: u32,
}

impl SemaError {
    pub fn new(message: &str, line: u32, column: u32) -> Self {
        SemaError {
            message: message.into(),
            line,
            column,
        }
    }
}

// =========================================================================
// SemaResult — sema 产出的图构建元信息
// =========================================================================

/// sema 产出的图构建元信息。
///
/// 从"检查器"升级为"图构建驱动器"，输出图构建所需的全部元信息。
/// 所有字段均为自有数据（`Box<str>` / `Vec` / `FxHashMap`），无需额外 arena 所有权。
pub struct SemaResult {
    /// 表达式 → 类型信息（决定通道宽度），key = AST 表达式句柄地址
    pub expr_types: FxHashMap<u64, ExprInfo>,
    /// 编译期错误
    pub errors: Vec<SemaError>,
    /// 是否有错误
    pub has_error: bool,
    /// 类型定义表（替代 IRBuilder 的 type_table + ctor_table）
    pub type_defs: Vec<TypeDefInfo>,
    /// 类型名 → type_defs 索引
    pub type_def_index: FxHashMap<String, u16>,
    /// Trait 定义表
    pub trait_defs: Vec<TraitDefInfo>,
    /// Trait 名 → trait_defs 索引
    pub trait_def_index: FxHashMap<String, u16>,
    /// 函数签名表
    pub func_sigs: Vec<FuncSigInfo>,
    /// 函数名 → func_sigs 索引
    pub func_sig_index: FxHashMap<String, u16>,
    /// 协程元数据表
    pub coroutine_metas: Vec<CoroutineMeta>,
    /// 构造器名 → (type_def_index << 16 | ctor_index)
    pub ctor_def_index: FxHashMap<String, u32>,
    /// import 别名表：短名 → 别名目标
    pub import_aliases: FxHashMap<String, AliasTarget>,
    /// 单态化实例表
    pub monomorph_instances: Vec<MonomorphInstance>,
    /// 单态化实例名 → monomorph_instances 索引
    pub monomorph_index: FxHashMap<u64, u32>,
    /// trait 默认方法单态化实例表（Sema 后阶段由 Monomorph 模块收集）
    pub trait_default_instances: Vec<TraitDefaultInstance>,
    /// 动态 ops 注册表（用户类型 ops，替代 TypeDescriptorPool）
    pub dynamic_ops: DynamicOpsRegistry,
    /// 调用点 → 实例映射
    pub call_instantiations: FxHashMap<u64, u32>,
    /// 字段访问元信息（全局，key = AST field_access Expr 句柄地址）
    pub field_accesses: FxHashMap<u64, FieldAccessInfo>,
    /// 方法分派元信息（key = AST call Expr 句柄地址）
    pub method_dispatches: FxHashMap<u64, DispatchInfo>,
    /// reflect 已解析元信息
    pub reflect_metas: FxHashMap<u64, ReflectMeta>,
    /// 已解析类型句柄（key = AST Expr 句柄地址）
    pub resolved_types: FxHashMap<u64, TypeHandle>,
    /// 字段 ID 映射（key = "type_name\x00field_name" → field_id）
    /// ADT/newtype/error_newtype: `__tag=0`，字段从 1 开始
    /// Record: 字段按声明顺序 0..N-1
    pub field_id_map: FxHashMap<String, u16>,
    /// witness table（trait 实现的静态分派表）。
    ///
    /// sema 检查期间由 InferContext 维护并跨模块累积，check 完成后
    /// 镜像到此字段供 IR 层（IrBuilder）访问 trait 方法分派信息。
    pub witness_table: WitnessTable,
    /// 模块函数调用的 recv ExprId key 集合（Zig @This 语义）。
    ///
    /// 当 `import std.time.Duration` 且模块内定义 `pub type Duration` 时，
    /// predefine 用 redefine 将 ModuleRef 覆盖为构造器 Fn。sema MethodCall 路径 0b
    /// 检测到此情况后，在此集合标记 recv 的 expr key，使 IR 编译时不传 recv
    /// （`Duration.from_millis(100)` → `from_millis(100)` 而非 `from_millis(Duration, 100)`）。
    pub module_func_recv_exprs: FxHashSet<u64>,
    /// 模块常量访问的 recv ExprId key → mangled 名（module_path.field）。
    ///
    /// 当 `Math.PI` 这样的 FieldAccess 的 recv 是 ModuleRef 且 field 是模块内常量时，
    /// sema 在此映射记录 recv 的 expr key → 全局变量 mangled 名（如 "std.math.Math.PI"）。
    /// IR 编译时据此跳过 recv 编译，直接发 compile_global_load 读取全局变量 slot，
    /// 与本地全局变量访问同路径，避免把模块名编译成僵尸 Const 节点。
    pub module_const_recv_exprs: FxHashMap<u64, String>,
}

impl Default for SemaResult {
    fn default() -> Self {
        Self::new()
    }
}


/// 生成 "表 + 索引 + put/get" 三件套的标准注册函数。
/// `$put`/`$get` 为方法名，`$field` 为表字段名，`$index` 为索引字段名，`$ty` 为元素类型。
macro_rules! define_table_registry {
    ($put:ident, $get:ident, $field:ident, $index:ident, $ty:ty) => {
        /// 添加元素并注册索引；重复名返回 `false`。
        pub fn $put(&mut self, def: $ty) -> bool {
            if self.$index.contains_key(def.name.as_ref()) {
                return false;
            }
            let idx: u16 = self.$field.len() as u16;
            self.$index.insert(def.name.to_string(), idx);
            self.$field.push(def);
            true
        }
        /// 按名查询元素。
        pub fn $get(&self, name: &str) -> Option<&$ty> {
            let idx = *self.$index.get(name)?;
            self.$field.get(idx as usize)
        }
    };
}

impl SemaResult {
    pub fn new() -> Self {
        SemaResult {
            expr_types: FxHashMap::default(),
            errors: Vec::new(),
            has_error: false,
            type_defs: Vec::new(),
            type_def_index: FxHashMap::default(),
            trait_defs: Vec::new(),
            trait_def_index: FxHashMap::default(),
            func_sigs: Vec::new(),
            func_sig_index: FxHashMap::default(),
            coroutine_metas: Vec::new(),
            ctor_def_index: FxHashMap::default(),
            import_aliases: FxHashMap::default(),
            monomorph_instances: Vec::new(),
            monomorph_index: FxHashMap::default(),
            trait_default_instances: Vec::new(),
            dynamic_ops: DynamicOpsRegistry::new(),
            call_instantiations: FxHashMap::default(),
            field_accesses: FxHashMap::default(),
            method_dispatches: FxHashMap::default(),
            reflect_metas: FxHashMap::default(),
            resolved_types: FxHashMap::default(),
            field_id_map: FxHashMap::default(),
            witness_table: WitnessTable::new(),
            module_func_recv_exprs: FxHashSet::default(),
            module_const_recv_exprs: FxHashMap::default(),
        }
    }

    // ── 表达式 ──

    /// 记录表达式类型。
    pub fn put_expr(&mut self, expr_id: u64, info: ExprInfo) {
        self.expr_types.insert(expr_id, info);
    }

    /// 查询表达式类型。
    pub fn get_expr(&self, expr_id: u64) -> Option<&ExprInfo> {
        self.expr_types.get(&expr_id)
    }

    // ── import 别名 ──

    /// 注册 import 别名；重复短名返回 `false`。
    pub fn put_import_alias(&mut self, short_name: &str, target: AliasTarget) -> bool {
        if self.import_aliases.contains_key(short_name) {
            return false;
        }
        self.import_aliases.insert(short_name.to_string(), target);
        true
    }

    /// 查询 import 别名。
    pub fn get_import_alias(&self, short_name: &str) -> Option<&AliasTarget> {
        self.import_aliases.get(short_name)
    }

    // ── 错误 ──

    /// 记录错误。
    pub fn add_error(&mut self, err: SemaError) {
        self.has_error = true;
        self.errors.push(err);
    }

    // ── 类型定义 ──

    /// 添加类型定义并注册 `type_def_index` / `ctor_def_index`，同时自动填充
    /// `field_id_map`。
    ///
    /// 类型名冲突时返回 `false`（同名类型不能重复定义）。
    /// 构造器名冲突时跳过该构造器（不注册到 `ctor_def_index`），但继续注册
    /// 其余构造器和类型定义本身，返回 `true`。
    /// 这处理类型名与构造器名共享命名空间的场景（如 `File` 既是 newtype
    /// 类型名又是 `FileKind` ADT 变体名），确保非冲突变体（如 `Directory`）
    /// 能被正常注册。
    pub fn put_type_def(&mut self, def: TypeDefInfo) -> bool {
        // u16 索引溢出检查（与 TypeDesc.rs 的 register 对齐）
        assert!(
            self.type_defs.len() < u16::MAX as usize,
            "type_def index overflow: too many type definitions"
        );
        let idx: u16 = self.type_defs.len() as u16;
        // 类型名冲突：拒绝（同名类型不能重复定义）
        if self.type_def_index.contains_key(def.name.as_ref()) {
            return false;
        }
        // 构造器名冲突：跳过该构造器，继续注册其余构造器
        self.populate_field_ids(&def);
        for (ci, ctor) in def.constructors.iter().enumerate() {
            if self.ctor_def_index.contains_key(ctor.name.as_ref()) {
                continue;
            }
            let packed_idx: u32 = ((idx as u32) << 16) | (ci as u32);
            self.ctor_def_index
                .insert(ctor.name.to_string(), packed_idx);
        }
        self.type_def_index.insert(def.name.to_string(), idx);
        self.type_defs.push(def);
        true
    }

    /// 按 type_def 的 kind 规则填充 `field_id_map`。
    /// - adt/newtype/error_newtype: `__tag=0`，字段从 1 开始
    /// - record: 字段按声明顺序 0..N-1
    /// - alias: 无字段
    fn populate_field_ids(&mut self, def: &TypeDefInfo) {
        match def.kind {
            TypeDefKind::Adt => {
                for ctor in def.constructors.iter() {
                    for (fi, fname) in ctor.field_names.iter().enumerate() {
                        if let Some(name) = fname {
                            let field_id = (fi + 1) as u16;
                            self.put_field_id(&def.name, name, field_id);
                        }
                    }
                }
                self.put_field_id(&def.name, "__tag", 0);
            }
            TypeDefKind::Newtype => {
                for (fi, fname) in def.constructors.iter().flat_map(|c| c.field_names.iter()).enumerate() {
                    let field_id = (fi + 1) as u16;
                    match fname {
                        Some(name) => self.put_field_id(&def.name, name, field_id),
                        None => {
                            let positional = format!("_{}", fi);
                            self.put_field_id(&def.name, &positional, field_id);
                        }
                    }
                }
                self.put_field_id(&def.name, "__tag", 0);
            }
            TypeDefKind::Record => {
                if let Some(ctor) = def.constructors.first() {
                    for (fi, fname) in ctor.field_names.iter().enumerate() {
                        if let Some(name) = fname {
                            let field_id = fi as u16;
                            self.put_field_id(&def.name, name, field_id);
                        }
                    }
                }
            }
            TypeDefKind::Alias => {}
        }
    }

    /// 构造 `field_id_map` 的 key：`"type_name\x00field_name"`。
    fn make_field_key(type_name: &str, field_name: &str) -> String {
        format!("{}\0{}", type_name, field_name)
    }

    /// 构造 `field_id_map` 的 key 并插入（已存在则覆盖）。
    fn put_field_id(&mut self, type_name: &str, field_name: &str, field_id: u16) {
        let key = Self::make_field_key(type_name, field_name);
        self.field_id_map.insert(key, field_id);
    }

    /// 查询 field_id（找不到返回 `None`）。
    /// key = "type_name\x00field_name"
    pub fn lookup_field_id(&self, type_name: &str, field_name: &str) -> Option<u16> {
        let key = Self::make_field_key(type_name, field_name);
        self.field_id_map.get(&key).copied()
    }

    /// 按名查询类型定义。
    pub fn get_type_def(&self, name: &str) -> Option<&TypeDefInfo> {
        let idx = *self.type_def_index.get(name)?;
        self.type_defs.get(idx as usize)
    }

    /// 按构造器名查询构造器定义。
    pub fn get_ctor_def(&self, name: &str) -> Option<&CtorDefInfo> {
        let packed_idx = *self.ctor_def_index.get(name)?;
        let type_idx = (packed_idx >> 16) as u16;
        let ctor_idx = (packed_idx & 0xFFFF) as u16;
        let def = self.type_defs.get(type_idx as usize)?;
        def.constructors.get(ctor_idx as usize)
    }

    /// 解析记录/ADT 字段类型描述符。
    /// 按 `TypeDefKind` 区分 Record（field_id 从 0）与 ADT（field_id 从 1）。
    /// 返回 `(field_id, field_type_desc)`，找不到返回 `None`。
    pub fn resolve_field_td(
        &self,
        type_name: &str,
        field: &str,
    ) -> Option<(u16, TypeHandle)> {
        let field_id = self.lookup_field_id(type_name, field)?;
        let ctor = self.get_ctor_def(type_name)?;
        let idx = match self.get_type_def(type_name) {
            Some(def) if def.kind == TypeDefKind::Record => field_id as usize,
            _ => (field_id as usize).saturating_sub(1),
        };
        let &field_td = ctor.field_types.get(idx)?;
        Some((field_id, field_td))
    }

    // ── Trait 定义 ──
    define_table_registry!(put_trait_def, get_trait_def, trait_defs, trait_def_index, TraitDefInfo);

    // ── 函数签名 ──
    define_table_registry!(put_func_sig, get_func_sig, func_sigs, func_sig_index, FuncSigInfo);

    // ── 方法签名（Ty 驱动） ──

    /// 按类型名和方法名查找 method_idx（在 TypeDefInfo.methods 中的位置）。
    ///
    /// IR 层用 (type_id, method_idx) 查 method_subgraphs 获取子图。
    /// 返回 None 表示该类型无此方法（可能是 trait 默认方法，需查 witness_table）。
    pub fn lookup_method_idx(&self, type_name: &str, method_name: &str) -> Option<u16> {
        let &type_idx = self.type_def_index.get(type_name)?;
        let type_def = &self.type_defs[type_idx as usize];
        type_def
            .methods
            .iter()
            .position(|m| m.name.as_ref() == method_name)
            .map(|i| i as u16)
    }

    /// 按 type_id 和 method_idx 获取方法签名。
    pub fn get_method_sig(&self, type_id: u16, method_idx: u16) -> Option<&MethodSigInfo> {
        if type_id < FIRST_DYNAMIC_TYPE_ID {
            return None;
        }
        let type_idx = type_def_index_of(type_id) as usize;
        let type_def = self.type_defs.get(type_idx)?;
        type_def.methods.get(method_idx as usize)
    }

    // ── 协程元数据 ──

    /// 添加协程元数据。
    pub fn put_coroutine_meta(&mut self, meta: CoroutineMeta) {
        self.coroutine_metas.push(meta);
    }

    /// 按 func_idx 查询协程元数据。
    pub fn get_coroutine_meta_by_func_idx(&self, func_idx: u16) -> Option<&CoroutineMeta> {
        self.coroutine_metas.iter().find(|m| m.func_idx == func_idx)
    }
}

// =========================================================================
// builtin_types — 内置类型注册表
//
// 对 `src/sema/builtin_types.zig` 的 Rust 移植。
// 统一标量名 → Ty 映射，以及内置泛型类型 arity 表。
// 数据源：Type.rs 的 BUILTIN_TABLE（type_id 1..=21），单一真相。
// =========================================================================

/// 内置泛型类型条目（高阶类型，固定 arity）。
#[derive(Debug, Clone, Copy)]
pub struct BuiltinGenericEntry {
    pub name: &'static str,
    pub arity: u8,
}

/// 内置类型声明宏：一处声明，生成两个产物。
///
/// - **产物 1**：`BUILTIN_GENERIC_TYPES` 静态 arity 表（仅 `generic` 组）。
///   kind_check（step 7）在 `register_builtin_method_sigs`（step 8）之前执行，
///   故 arity 表必须为静态常量。
/// - **产物 2**：`register_builtin_method_sigs` 函数体（`generic` + `nongeneric` 组）。
///   运行时注册合成 `TypeDefInfo`（含方法签名表），使内置类型方法查找走与
///   用户自定义类型统一的 `(type_id, method_idx)` 路径。
///
/// `generic` 组的类型使用 `TypeNode::Generic { name, .. }` AST 节点，需在
/// `BUILTIN_GENERIC_TYPES` 中有条目以供 kind_check 查询 arity。
/// `nongeneric` 组有专用 `Ty`/`TypeNode` 变体（如 Array/Nullable/Str），
/// 不需要 arity 表条目。
///
/// 声明语法：
/// ```ignore
/// define_builtin_types! {
///     generic {
///         "TypeName" : ["T", "E"] = [ sig(...), sig(...), ... ],
///         ...
///     }
///     nongeneric {
///         "TypeName" : ["T"] = [ sig(...), ... ],
///         ...
///     }
/// }
/// ```
macro_rules! define_builtin_types {
    (
        generic { $($gname:literal : [$($gp:literal),*] = [$($gmethod:expr),* $(,)?]),* $(,)? }
        nongeneric { $($nname:literal : [$($np:literal),*] = [$($nmethod:expr),* $(,)?]),* $(,)? }
    ) => {
        /// 内置泛型类型构造器表（由 `define_builtin_types!` 宏从 `generic` 组派生）。
        pub const BUILTIN_GENERIC_TYPES: &[BuiltinGenericEntry] = &[
            $( BuiltinGenericEntry {
                name: $gname,
                arity: <[&'static str]>::len(&[$($gp),*]) as u8,
            } ),*
        ];

        /// 为内置类型注册合成 TypeDefInfo（含方法签名表），使内置类型的方法查找走与
        /// 用户自定义类型统一的 (type_id, method_idx) 路径，消除 lookup_builtin_method
        /// 的 match 分支特判。
        ///
        /// 方法签名中：
        /// - param_type_reprs[0] = SelfType（self 参数，与用户 type 块一致）
        /// - 泛型参数用 Named("T")/Named("E")，由 type_binding_stack 解析
        /// - 标量返回类型用 Named("usize")/Named("bool")/Named("void")/Named("str")
        /// - build_fn_type_from_sig 通过 type_repr_to_handle 还原完整 Ty::Fn
        pub fn register_builtin_method_sigs(sema_result: &mut SemaResult) {
            /// 构建单条内置方法签名。type 字段用 Ty::Void 占位（不影响类型检查，
            /// build_fn_type_from_sig 只读 param_type_reprs / return_type_repr）。
            /// intrinsic 参数标注降级策略，None 表示普通方法（有方法体或 trait 方法）。
            fn sig(
                name: &str,
                param_reprs: Vec<TypeRepr>,
                return_repr: Option<TypeRepr>,
                intrinsic: Option<IntrinsicKind>,
            ) -> MethodSigInfo {
                let n = param_reprs.len();
                MethodSigInfo {
                    name: name.into(),
                    param_is_ref: vec![false; n].into_boxed_slice(),
                    return_is_ref: false,
                    is_async: false,
                    is_throwing: false,
                    param_type_reprs: param_reprs.into_boxed_slice(),
                    return_type_repr: return_repr,
                    intrinsic,
                }
            }

            /// 为一个内置类型注册合成 TypeDefInfo。
            fn register(
                sema_result: &mut SemaResult,
                type_name: &str,
                type_params: &[&str],
                methods: Vec<MethodSigInfo>,
            ) {
                if sema_result.type_def_index.contains_key(type_name) {
                    return; // 已注册（如用户 stdlib 已声明同名 type 块）
                }
                let def = TypeDefInfo {
                    name: type_name.into(),
                    kind: TypeDefKind::Alias,
                    constructors: Box::new([]),
                    type_params: type_params.iter().map(|t| (*t).into()).collect(),
                    target_type_name: None,
                    target_type: None,
                    methods: methods.into_boxed_slice(),
                };
                sema_result.put_type_def(def);
            }

            // ── generic 组：进入 BUILTIN_GENERIC_TYPES + 方法注册 ──
            $(
                register(sema_result, $gname, &[$($gp),*], vec![$($gmethod),*]);
            )*
            // ── nongeneric 组：仅方法注册（有专用 Ty 变体）──
            $(
                register(sema_result, $nname, &[$($np),*], vec![$($nmethod),*]);
            )*
        }
    };
}

define_builtin_types! {
    generic {
        "Throw" : ["T", "E"] = [
            sig("is_ok", vec![TypeRepr::SelfType], Some(TypeRepr::Named("bool".into())), None),
        ],
        "Channel" : ["T"] = [
            sig("send", vec![TypeRepr::SelfType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(284))),
            sig("recv", vec![TypeRepr::SelfType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::ChannelAwait)),
            sig("close", vec![TypeRepr::SelfType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(285))),
        ],
        "Atomic" : ["T"] = [
            sig("swap", vec![TypeRepr::SelfType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("T".into())), None),
            sig("cas", vec![TypeRepr::SelfType, TypeRepr::Named("T".into()), TypeRepr::Named("T".into())], Some(TypeRepr::Named("bool".into())), None),
            sig("load", vec![TypeRepr::SelfType], Some(TypeRepr::Named("T".into())), None),
            sig("store", vec![TypeRepr::SelfType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), None),
        ],
        "Async" : ["T"] = [
            sig("status", vec![TypeRepr::SelfType], Some(TypeRepr::Named("str".into())), None),
            sig("await", vec![TypeRepr::SelfType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::Await)),
            sig("cancel", vec![TypeRepr::SelfType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(42))),
        ],
        "Sender" : ["T"] = [
            sig("send", vec![TypeRepr::SelfType, TypeRepr::Named("T".into())], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::BinOp(284))),
            sig("close", vec![TypeRepr::SelfType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(285))),
        ],
        "Receiver" : ["T"] = [
            sig("recv", vec![TypeRepr::SelfType], Some(TypeRepr::Named("T".into())), Some(IntrinsicKind::ChannelAwait)),
            sig("close", vec![TypeRepr::SelfType], Some(TypeRepr::Named("void".into())), Some(IntrinsicKind::UnOp(285))),
        ],
        "Lazy" : ["T"] = [],
    }
    nongeneric {
        "array" : ["T"] = [
            sig("len", vec![TypeRepr::SelfType], Some(TypeRepr::Named("usize".into())), Some(IntrinsicKind::UnOp(35))),
            sig("is_empty", vec![TypeRepr::SelfType], Some(TypeRepr::Named("bool".into())), None),
        ],
        "str" : [] = [
            sig("len", vec![TypeRepr::SelfType], Some(TypeRepr::Named("usize".into())), Some(IntrinsicKind::UnOp(35))),
            sig("is_empty", vec![TypeRepr::SelfType], Some(TypeRepr::Named("bool".into())), None),
            sig("bytes", vec![TypeRepr::SelfType], Some(TypeRepr::Array(Box::new(TypeRepr::Named("u8".into())), None)), Some(IntrinsicKind::UnOp(287))),
        ],
        "nullable" : ["T"] = [
            sig("is_null", vec![TypeRepr::SelfType], Some(TypeRepr::Named("bool".into())), None),
        ],
    }
}

/// 内置泛型类型名 → arity（未匹配返回 `None`）。
pub fn generic_type_arity(name: &str) -> Option<u8> {
    BUILTIN_GENERIC_TYPES
        .iter()
        .find(|e| e.name == name)
        .map(|e| e.arity)
}

/// 判断 name 是否为内置泛型类型构造器。
#[inline]
pub fn is_builtin_generic_type(name: &str) -> bool {
    generic_type_arity(name).is_some()
}

// =========================================================================
// type_resolver — 类型解析器
//
// 职责：将 AST 类型节点 + type_args 绑定上下文解析为 TypeHandle。
// 所有函数为自由函数（非方法），因解析需要 `&AstArena` + `&mut TypeArena` +
// `&mut SemaResult` 多输入，无单一 self 持有状态。
// =========================================================================

/// 从 TypeNode 提取类型名（用于变量绑定的类型推断）。
///
/// 对 `&T` / `*T` 递归到 inner，对泛型返回基类名，对命名类型返回其名。
/// 其他类型节点返回 `None`。
pub fn type_name_from_node<'a>(
    type_ref: Option<AstTypeRef>,
    ast: &AstArena<'a>,
) -> Option<&'a str> {
    let type_ref = type_ref?;
    let tn = &ast.ty(type_ref).node;
    let effective = match tn {
        TypeNode::RefType { inner } | TypeNode::RawPtr { inner } => &ast.ty(*inner).node,
        _ => tn,
    };
    match effective {
        TypeNode::Named { name } => Some(name),
        TypeNode::Generic { name, .. } => Some(name),
        _ => None,
    }
}

/// Check if a TypeHandle corresponds to a type with the given name.
fn type_handle_name_matches(arena: &TypeArena, h: TypeHandle, name: &str) -> bool {
    match arena.get(h) {
        Ty::Adt(_) => arena.adt_parts(h).0 == name,
        Ty::Generic(_) => arena.generic_parts(h).0 == name,
        Ty::Trait(_) => arena.trait_parts(h).0 == name,
        // 其余类型（含内置泛型 Throw/Channel/Async/Lazy/Atomic/Sender/Receiver/Timer
        // 及标量/str/void 等）统一走 ty.name()，单一真相源
        ty => ty.name() == name,
    }
}

/// 类型解析递归深度上限：防止极深 alias/newtype 链导致栈溢出。
/// visiting.len() 即当前递归深度，达到上限时停止递归。
const MAX_TYPE_RECURSION_DEPTH: usize = 256;

/// 按名解析类型（resolved 版本，含 alias/newtype 链展开）。
///
/// 优先级：type_args 绑定 → 内置标量/str/void → type_defs alias/newtype 递归 →
/// 用户自定义类型（arena.make_adt）。
/// 提取为独立函数以便 `resolve_type_node_resolved` 的 alias 递归调用，无需构造临时 TypeNode。
///
/// `visiting` 用于循环 alias 检测：若 name 已在集合中，说明出现循环 alias 链，
/// 返回 arena.make_adt(name) 而非继续递归（避免无限递归栈溢出）。
fn resolve_named_type_resolved(
    arena: &mut TypeArena,
    name: &str,
    type_args: &[TypeHandle],
    sema_result: &mut SemaResult,
    visiting: &mut FxHashSet<String>,
) -> TypeHandle {
    // 1. 优先查 type_args 绑定（泛型类型参数）
    for &ta in type_args {
        if type_handle_name_matches(arena, ta, name) {
            return ta;
        }
    }
    // 2. 内置标量/str/null/void
    if let Some(ty) = Ty::from_type_name(name) {
        return arena.make(ty);
    }
    // 循环 alias 检测：name 已在 visiting 中说明出现循环，停止递归
    if visiting.contains(name) {
        return arena.make_adt(name.into(), Box::new([]));
    }
    // 递归深度上限：visiting.len() 即当前递归深度，超限停止递归防止栈溢出
    if visiting.len() >= MAX_TYPE_RECURSION_DEPTH {
        return arena.make_adt(name.into(), Box::new([]));
    }
    visiting.insert(name.to_string());
    // 3. 查 type_defs 解析 alias/newtype 链
    //    提取所需信息（owned）以释放不可变借用，允许后续 &mut 调用。
    let (target_ty, target_name): (Option<TypeHandle>, Option<String>) =
        match sema_result.get_type_def(name) {
            Some(td) => (
                td.target_type,
                td.target_type_name.as_deref().map(String::from),
            ),
            None => (None, None),
        };
    if let Some(inner_ty) = target_ty {
        // alias/newtype 有目标 TypeHandle：直接返回
        visiting.remove(name);
        return inner_ty;
    }
    if let Some(ttn) = target_name {
        // target_type_name 已知：递归解析到最终具体类型
        let result = resolve_named_type_resolved(arena, &ttn, type_args, sema_result, visiting);
        visiting.remove(name);
        return result;
    }
    // 4. 其他用户自定义类型 → 创建具名 Adt
    visiting.remove(name);
    arena.make_adt(name.into(), Box::new([]))
}

/// 解析 TypeNode 为 TypeHandle（resolved 版本，含 alias/newtype 链展开）。
///
/// 与 `concretize_type` 的差异：Named 分支查询 `sema_result.type_defs`，
/// 若为 alias/newtype 且 target_type 已知，递归解析到具体标量类型。
/// 用于需要穿透 alias 链获取最终标量通道类型的场景（如 field_value 标量单态化）。
pub fn resolve_type_node_resolved<'a>(
    arena: &mut TypeArena,
    type_ref: Option<AstTypeRef>,
    type_args: &[TypeHandle],
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> Option<TypeHandle> {
    let type_ref = type_ref?;
    let tn = &ast.ty(type_ref).node;
    let mut visiting: FxHashSet<String> = FxHashSet::default();
    Some(match tn {
        TypeNode::Named { name } => resolve_named_type_resolved(arena, name, type_args, sema_result, &mut visiting),
        TypeNode::Generic { name, args } => {
            // Lazy<T>：递归解析内部类型
            if Ty::from_type_name(name).is_some_and(|t| t.family() == TypeFamily::Lazy)
                && !args.is_empty() {
                if let Some(inner_ty) =
                    resolve_type_node_resolved(arena, Some(args[0]), type_args, ast, sema_result)
                {
                    return Some(inner_ty);
                }
            }
            arena.make_generic((*name).into(), Box::new([]))
        }
        TypeNode::Nullable { inner } => {
            return resolve_type_node_resolved(arena, Some(*inner), type_args, ast, sema_result);
        }
        TypeNode::RefType { inner } => {
            let inner_name = type_name_from_node(Some(*inner), ast).unwrap_or("ref");
            arena.make_adt(inner_name.into(), Box::new([]))
        }
        TypeNode::RawPtr { inner } => {
            let inner_name = type_name_from_node(Some(*inner), ast).unwrap_or("ptr");
            arena.make_adt(inner_name.into(), Box::new([]))
        }
        TypeNode::Record { .. } => arena.make_record(Vec::<FieldType>::new().into_boxed_slice(), None),
        TypeNode::Function { .. } => {
            let ret = arena.make(Ty::Unknown);
            arena.make_fn(Vec::<TypeHandle>::new().into_boxed_slice(), ret)
        }
        TypeNode::Array { .. } => arena.make_adt("array".into(), Box::new([])),
        TypeNode::SelfType => {
            for &ta in type_args {
                if type_handle_name_matches(arena, ta, "Self") {
                    return Some(ta);
                }
            }
            arena.make_adt("Self".into(), Box::new([]))
        }
        TypeNode::KindAnnotated { inner, .. } => {
            return resolve_type_node_resolved(arena, Some(*inner), type_args, ast, sema_result);
        }
    })
}

// =========================================================================
// inference — 类型推导核心
//
// 对 `src/sema/inference.zig` 的 Rust 移植。
// 职责：泛型实参反推、self 参数绑定、字面量提升、GADT 推断。
//
// 与 Zig 原版的差异（有意改进，用户确认）：
// - **self 参数强制 scope 绑定**：不再允许顶层 extension fun 的 `self: TypeName`。
//   self 只能在 type/trait 块内使用，且不允许类型注解。
//   Zig 原版的 3 层 fallback（scope→标注→fresh var）简化为 scope→error。
// - **字面量提升**：保留 Zig 语义，字面量与变量运算时提升到变量类型。
// - **泛型延迟求解**：保留 Zig 语义，未求解的 TypeVar 留待后续 unify。
//
// 绑定栈架构：
// - TypeBindingStack：泛型参数名 → TypeHandle（rigid var）
// - SelfBindingStack：Self → TypeHandle（scope 类型）
// 两栈同步 push/pop：进入 impl Type<T> 块时，T 入 TypeBindingStack，
// Type<T> 入 SelfBindingStack；离开时同步弹出。
// =========================================================================

/// 类型绑定栈帧：泛型参数名 → TypeHandle（通常为 rigid TypeVar）
#[derive(Debug, Default)]
pub struct TypeBindingFrame {
    bindings: FxHashMap<Box<str>, TypeHandle>,
}

impl TypeBindingFrame {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: &str, ty: TypeHandle) {
        self.bindings.insert(name.into(), ty);
    }

    pub fn get(&self, name: &str) -> Option<TypeHandle> {
        self.bindings.get(name).copied()
    }
}

/// 类型绑定栈：管理泛型实例化期间的类型参数绑定。
///
/// 进入 `impl Type<T>` 或 `fn method<U>` 时 push 一帧，离开时 pop。
/// `lookup` 从栈顶向下查找，内层绑定优先（shadowing 语义）。
///
/// 注意：此栈持有 `TypeHandle`（Ty 索引），类型解析通过 `InferContext::lookup_type_binding`
/// 完成，不走独立的 trait 抽象以避免类型混淆。
#[derive(Debug, Default)]
pub struct TypeBindingStack {
    frames: Vec<TypeBindingFrame>,
}

impl TypeBindingStack {
    pub fn new() -> Self {
        Self::default()
    }

    /// 压入空帧，后续通过 `insert` 添加绑定。
    pub fn push(&mut self) {
        self.frames.push(TypeBindingFrame::new());
    }

    /// 压入预构造帧。
    pub fn push_frame(&mut self, frame: TypeBindingFrame) {
        self.frames.push(frame);
    }

    /// 弹出栈顶帧。
    pub fn pop(&mut self) -> Option<TypeBindingFrame> {
        self.frames.pop()
    }

    /// 当前栈深度。
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// 从栈顶向下查找类型参数绑定（内层优先）。
    pub fn lookup(&self, name: &str) -> Option<TypeHandle> {
        for frame in self.frames.iter().rev() {
            if let Some(ty) = frame.get(name) {
                return Some(ty);
            }
        }
        None
    }

    /// 在栈顶帧添加绑定（仅当栈非空）。
    pub fn insert_top(&mut self, name: &str, ty: TypeHandle) {
        if let Some(frame) = self.frames.last_mut() {
            frame.insert(name, ty);
        }
    }
}

/// Self 绑定栈：管理 type/trait 块的 Self 类型绑定。
///
/// 进入 `type T { ... }` 块时 push `T` 的 TypeHandle；
/// 进入 `trait Foo<T> { default methods }` 时 push fresh_type_var；
/// 离开时 pop。`lookup` 返回栈顶（内层优先）。
#[derive(Debug, Default)]
pub struct SelfBindingStack {
    stack: Vec<TypeHandle>,
}

impl SelfBindingStack {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, self_ty: TypeHandle) {
        self.stack.push(self_ty);
    }

    pub fn pop(&mut self) -> Option<TypeHandle> {
        self.stack.pop()
    }

    pub fn current(&self) -> Option<TypeHandle> {
        self.stack.last().copied()
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

// =========================================================================
// populate — 从 AST 填充 SemaResult 定义表
//
// 对 `src/sema/populate.zig` 的 Rust 移植。
// 职责：遍历模块声明，分发到对应的转换函数，填充 type_defs/func_sigs/trait_defs。
//
// 与 Zig 原版的差异：
// - 去掉 arena_alloc 参数：Rust 用 Box<[T]> / Vec<T> 自有数据
// - anytype 参数 → 具体类型（Decl 变体解构）
// - *const TypeNode → TypeRef + &AstArena 解引用
// - orelse ... catch unreachable → unwrap_or_else
// - 无 _force_analysis（Rust 无懒分析）
//
// 依赖：单向依赖 crate::Ast（Module/Decl/TypeNode 等）+ 已有的 SemaResult put 方法
// =========================================================================

use crate::ast::Ast::{
    ConstructorDef, RecordFieldType, TypeDef as AstTypeDef,
};

/// populate 主入口：遍历模块声明，填充 SemaResult 的定义表。
///
/// 遍历 `module.declarations`，按声明类型分发：
/// - `Decl::FunDecl` → `ast_fun_decl_to_func_sig`
/// - `Decl::TypeDecl` → `ast_type_decl_to_type_def`
/// - `Decl::TraitDecl` → `ast_trait_decl_to_trait_def`
/// - 其他（ImportDecl/PackDecl/ExprDecl）→ 跳过
///
/// 返回 false 表示有重复定义错误（put 方法返回 false 时记录）。
pub fn populate_sema_result_from_ast<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    decl: &'a crate::ast::Ast::Spanned<Decl<'a>>,
    ast: &AstArena<'a>,
) -> bool {
    match &decl.node {
        Decl::FunDecl { name, type_params, params, return_type, is_async, .. } => {
            ast_fun_decl_to_func_sig(arena, sema_result, name, type_params, params, *return_type, *is_async, ast)
        }
        Decl::TypeDecl { name, type_params, def, methods, .. } => {
            ast_type_decl_to_type_def(arena, sema_result, name, type_params, def, ast);
            // 注册 type 块内方法到 TypeDefInfo.methods（按 method_idx 索引）
            for method in methods.iter() {
                ast_method_to_func_sig(arena, sema_result, name, method, ast);
            }
            true
        }
        Decl::TraitDecl { name, methods, .. } => {
            ast_trait_decl_to_trait_def(arena, sema_result, name, methods, ast)
        }
        _ => true, // ImportDecl/PackDecl/ExprDecl 跳过
    }
}

/// 遍历模块的所有声明，批量填充 SemaResult。
///
/// 便捷封装：对 `module.declarations` 中每个声明调用 `populate_sema_result_from_ast`。
/// 任一声明填充失败（返回 false）则整体返回 false。
pub fn populate_module<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    module: &'a crate::ast::Ast::Module<'a>,
) -> bool {
    let mut ok = true;
    for decl in &module.declarations {
        if !populate_sema_result_from_ast(arena, sema_result, decl, &module.arena) {
            ok = false;
        }
    }
    ok
}

// ── 私有转换函数 ──

/// 将模块文件路径转换为逻辑模块路径。
///
/// `std/io/Path.kz` → `std.io.Path`
/// `stdlib/std/io/Path.kz` → `std.io.Path`（去掉 stdlib/ 前缀）
/// `builtin/error/Err.kz` → `builtin.error.Err`
/// 无 .kz 后缀或为空返回 None。
pub fn module_logical_path(name: &str) -> Option<String> {
    let path = name.strip_suffix(".kz")?;
    // 去掉 stdlib/ 前缀（如果存在）
    let path = path.strip_prefix("stdlib/").unwrap_or(path);
    if path.is_empty() {
        return None;
    }
    Some(path.replace('/', "."))
}

/// 计算模块感知的表达式 key：组合模块名哈希 + ExprId。
///
/// ExprId 是模块特定的（每个模块的 AST arena 独立编号），
/// 直接用 ExprId 作为全局 key 会导致跨模块冲突。
/// 此函数将模块名与 ExprId 组合为全局唯一的 u64 key。
pub fn module_expr_key(module_name: &str, expr_id: u64) -> u64 {
    use rustc_hash::FxHasher;
    use std::hash::Hasher;
    let mut hasher = FxHasher::default();
    hasher.write(module_name.as_bytes());
    hasher.write_u64(expr_id);
    hasher.finish()
}

/// fun_decl → FuncSigInfo，注册到 sema_result.func_sigs。
///
/// 顶层函数以裸名注册。type 块内方法用 `ast_method_to_func_sig` 注册为 mangled 名 `TypeName.method`。
fn ast_fun_decl_to_func_sig<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: &'a str,
    type_params: &[crate::ast::Ast::TypeParam<'a>],
    params: &[crate::ast::Ast::Param<'a>],
    return_type: Option<AstTypeRef>,
    is_async: bool,
    ast: &AstArena<'a>,
) -> bool {
    let name: Box<str> = name.into();
    ast_fun_decl_to_func_sig_inner(arena, sema_result, name, type_params, params, return_type, is_async, ast)
}

/// 从 AST MethodDecl 构造 MethodSigInfo（不注册到 func_sigs）。
///
/// 复用 `resolve_param_type` / `concretize_type` 进行类型解析，
/// 产出按 method_idx 索引的方法签名，存入 TypeDefInfo.methods。
fn build_method_sig_info<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    method: &crate::ast::Ast::MethodDecl<'a>,
    ast: &AstArena<'a>,
) -> MethodSigInfo {
    let mut param_is_ref: Vec<bool> = Vec::with_capacity(method.params.len());
    let mut param_type_reprs: Vec<TypeRepr> = Vec::with_capacity(method.params.len());

    for param in &method.params {
        let (_, is_ref, _, repr) = resolve_param_type(arena, param, ast, sema_result);
        param_is_ref.push(is_ref);
        param_type_reprs.push(repr);
    }

    let (_, return_type_repr, is_throwing) = match method.return_type {
        Some(rt) => {
            // return type 的自包含表示（TypeRepr）由 type_node_to_repr 直接从 AST 构造，
            // 不需要在此解析为 TypeHandle（旧 concretize_type 调用结果被丢弃，无副作用，已移除）。
            let repr = type_node_to_repr(&ast.ty(rt).node, ast);
            ((), Some(repr), is_throw_type(&ast.ty(rt).node))
        }
        None => ((), None, false),
    };

    let return_is_ref = match method.return_type {
        Some(rt) => matches!(ast.ty(rt).node, TypeNode::RefType { .. }),
        None => false,
    };

    MethodSigInfo {
        name: method.name.into(),
        param_is_ref: param_is_ref.into_boxed_slice(),
        return_is_ref,
        is_async: method.is_async,
        is_throwing,
        param_type_reprs: param_type_reprs.into_boxed_slice(),
        return_type_repr,
        intrinsic: None,
    }
}

/// type 块内方法 → MethodSigInfo，存入 TypeDefInfo.methods（按 method_idx 索引）。
///
/// method_idx = 方法在 type 块 methods 数组中的位置（AST 声明顺序）。
/// IR 阶段通过 (type_id, method_idx) 查 method_subgraphs 获取子图。
fn ast_method_to_func_sig<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    type_name: &str,
    method: &crate::ast::Ast::MethodDecl<'a>,
    ast: &AstArena<'a>,
) -> bool {
    let sig = build_method_sig_info(arena, sema_result, method, ast);
    if let Some(&type_idx) = sema_result.type_def_index.get(type_name) {
        let type_def = &mut sema_result.type_defs[type_idx as usize];
        let mut methods_vec: Vec<MethodSigInfo> = type_def.methods.to_vec();
        methods_vec.push(sig);
        type_def.methods = methods_vec.into_boxed_slice();
        true
    } else {
        false
    }
}

fn ast_fun_decl_to_func_sig_inner<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: Box<str>,
    type_params: &[crate::ast::Ast::TypeParam<'a>],
    params: &[crate::ast::Ast::Param<'a>],
    return_type: Option<AstTypeRef>,
    is_async: bool,
    ast: &AstArena<'a>,
) -> bool {

    // type_params：取每个 TypeParam 的 name
    let type_params: Box<[Box<str>]> = type_params.iter().map(|tp| tp.name.into()).collect();

    // param_is_ref：解析每个参数是否为引用类型
    let mut param_is_ref: Vec<bool> = Vec::with_capacity(params.len());

    for param in params {
        let (_, is_ref, _, _) = resolve_param_type(arena, param, ast, sema_result);
        param_is_ref.push(is_ref);
    }

    // return_type + is_throwing
    let (return_ty, is_throwing) = match return_type {
        Some(rt) => {
            let ty = concretize_type(arena, rt, &[], ast, sema_result);
            (ty, is_throw_type(&ast.ty(rt).node))
        }
        None => (arena.make(Ty::Void), false),
    };

    // return_is_ref：返回类型为 RefType 即为 true
    let return_is_ref = match return_type {
        Some(rt) => matches!(ast.ty(rt).node, TypeNode::RefType { .. }),
        None => false,
    };

    let sig = FuncSigInfo {
        name,
        type_params,
        return_type: return_ty,
        param_is_ref: param_is_ref.into_boxed_slice(),
        return_is_ref,
        is_async,
        is_throwing,
    };

    sema_result.put_func_sig(sig)
}

/// trait_decl → TraitDefInfo，注册到 sema_result.trait_defs。
pub(crate) fn ast_trait_decl_to_trait_def<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: &'a str,
    methods: &[crate::ast::Ast::MethodDecl<'a>],
    ast: &AstArena<'a>,
) -> bool {
    let name: Box<str> = name.into();

    let methods: Vec<TraitMethodSig> = methods
        .iter()
        .map(|m| {
            let return_type = match m.return_type {
                Some(rt) => concretize_type(arena, rt, &[], ast, sema_result),
                None => arena.make(Ty::Void),
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

    let trait_def = TraitDefInfo {
        name,
        methods: methods.into_boxed_slice(),
    };

    sema_result.put_trait_def(trait_def)
}

/// type_decl → TypeDefInfo，按 5 种 def 变体分发，注册到 sema_result.type_defs。
pub(crate) fn ast_type_decl_to_type_def<'a>(
    arena: &mut TypeArena,
    sema_result: &mut SemaResult,
    name: &'a str,
    type_params: &[crate::ast::Ast::TypeParam<'a>],
    def: &AstTypeDef<'a>,
    ast: &AstArena<'a>,
) -> bool {
    let name: Box<str> = name.into();
    let type_params: Box<[Box<str>]> = type_params.iter().map(|tp| tp.name.into()).collect();

    let (kind, constructors, target_type_name, target_type) = match def {
        AstTypeDef::Adt { constructors: ctor_defs } => {
            let ctors: Vec<CtorDefInfo> = ctor_defs
                .iter()
                .map(|c| constructor_def_to_ctor_info(arena, c, name.as_ref(), ast, sema_result))
                .collect();
            (TypeDefKind::Adt, ctors, None, None)
        }
        AstTypeDef::Record { fields } => {
            let ctor = record_fields_to_ctor_info(arena, fields, name.as_ref(), ast, sema_result);
            (TypeDefKind::Record, vec![ctor], None, None)
        }
        AstTypeDef::Alias { target } => {
            let target_ty = concretize_type(arena, *target, &[], ast, sema_result);
            let target_name = type_name_from_node(Some(*target), ast);
            (
                TypeDefKind::Alias,
                Vec::new(),
                target_name.map(|n| n.into()),
                Some(target_ty),
            )
        }
        AstTypeDef::Newtype { name: nt_name, inner } => {
            let target_ty = concretize_type(arena, *inner, &[], ast, sema_result);
            let target_name = type_name_from_node(Some(*inner), ast);
            let target_repr = type_node_to_repr(&ast.ty(*inner).node, ast);
            let ctor = CtorDefInfo {
                name: (*nt_name).into(),
                type_name: name.clone(),
                field_names: Box::new([Some("_0".into())]),
                field_types: Box::new([target_ty]),
                is_newtype: true,
                return_type_name: None,
                return_type_node: None,
                field_type_reprs: Box::new([target_repr]),
            };
            (
                TypeDefKind::Newtype,
                vec![ctor],
                target_name.map(|n| n.into()),
                Some(target_ty),
            )
        }
    };

    let type_def = TypeDefInfo {
        name,
        kind,
        constructors: constructors.into_boxed_slice(),
        type_params,
        target_type_name,
        target_type,
        methods: Box::new([]),
    };

    sema_result.put_type_def(type_def)
}

// ── 辅助函数 ──

/// 解析参数类型：返回 (TypeHandle, is_ref, type_name, type_repr)
fn resolve_param_type<'a>(
    arena: &mut TypeArena,
    param: &crate::ast::Ast::Param<'a>,
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> (TypeHandle, bool, Option<Box<str>>, TypeRepr) {
    match param.type_annotation {
        Some(tr) => {
            let node = &ast.ty(tr).node;
            let is_ref = matches!(node, TypeNode::RefType { .. });
            let ty = concretize_type(arena, tr, &[], ast, sema_result);
            let name = type_name_from_node(Some(tr), ast).map(|n| n.into());
            let repr = type_node_to_repr(node, ast);
            (ty, is_ref, name, repr)
        }
        None => (
            arena.make_adt("param".into(), Box::new([])),
            false,
            None,
            TypeRepr::Named("unknown".into()),
        ),
    }
}

/// 单一类型具体化入口（registration 阶段）：将 AST TypeNode 解析为 TypeHandle。
///
/// 统一 registration 阶段的类型解析，结构保留（make_ref/make_array/make_nullable/
/// make_fn/make_record）+ Named 别名/newtype 链展开（含循环检测与深度上限）。
///
/// 与 `resolve_type_node_resolved` 的差异：后者为标量通道宽度计算专用，将
/// Ref/Array 投影为 Adt(name)；本函数保留结构，是通用类型具体化入口。
/// 推断阶段（含 type_binding_stack/self_binding_stack 上下文）使用 InferContext
/// 的 `type_from_ast_with_params`，其上下文无法由本自由函数提供，故为独立阶段入口。
pub(crate) fn concretize_type<'a>(
    arena: &mut TypeArena,
    type_ref: AstTypeRef,
    type_args: &[TypeHandle],
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> TypeHandle {
    let tn = &ast.ty(type_ref).node;
    match tn {
        TypeNode::Named { name } => {
            // 委托 resolve_named_type_resolved：type_args 绑定 → 内置标量 →
            // alias/newtype 链展开（visiting 循环检测 + MAX_TYPE_RECURSION_DEPTH 深度上限）
            // → 用户自定义 Adt。消除旧 Named 分支仅 Ty::from_type_name/make_adt 的精度缺失。
            let mut visiting = FxHashSet::default();
            resolve_named_type_resolved(arena, name, type_args, sema_result, &mut visiting)
        }
        TypeNode::Generic { name, .. } => {
            if let Some(ty) = Ty::from_type_name(name) {
                arena.make(ty)
            } else {
                arena.make_generic((*name).into(), Box::new([]))
            }
        }
        TypeNode::Nullable { inner } => {
            let inner = concretize_type(arena, *inner, type_args, ast, sema_result);
            arena.make_nullable(inner)
        }
        TypeNode::RefType { inner } => {
            let inner = concretize_type(arena, *inner, type_args, ast, sema_result);
            arena.make_ref(inner, false)
        }
        TypeNode::RawPtr { inner } => {
            let inner = concretize_type(arena, *inner, type_args, ast, sema_result);
            arena.make_ref(inner, true)
        }
        TypeNode::Record { .. } => arena.make_record(Vec::<FieldType>::new().into_boxed_slice(), None),
        TypeNode::Function { .. } => {
            let ret = arena.make(Ty::Unknown);
            arena.make_fn(Vec::<TypeHandle>::new().into_boxed_slice(), ret)
        }
        TypeNode::Array { element_type, size } => {
            let elem = concretize_type(arena, *element_type, type_args, ast, sema_result);
            arena.make_array(elem, *size)
        }
        TypeNode::SelfType => {
            for &ta in type_args {
                if arena.get(ta).name() == "Self" {
                    return ta;
                }
            }
            arena.make_adt("Self".into(), Box::new([]))
        }
        TypeNode::KindAnnotated { inner, .. } => {
            concretize_type(arena, *inner, type_args, ast, sema_result)
        }
    }
}

/// 判断 TypeNode 是否为 `Throw<T, E>` 类型。
///
/// Throw 在 TypeNode 中表示为 `Generic { name: "Throw", args: [V, E] }`。
fn is_throw_type(tn: &TypeNode) -> bool {
    matches!(tn, TypeNode::Generic { name, .. }
        if Ty::from_type_name(name).is_some_and(|t| t.family() == TypeFamily::Throw))
}

/// 将 AST TypeNode 递归转换为自包含的 TypeRepr（不依赖 AstArena 引用）。
/// 用于在 sema 阶段将方法返回类型信息序列化存储，供后续跨模块 lookup_method_type 使用。
fn type_node_to_repr<'a>(tn: &TypeNode<'a>, ast: &AstArena<'a>) -> TypeRepr {
    match tn {
        TypeNode::Named { name } => TypeRepr::Named((*name).into()),
        TypeNode::SelfType => TypeRepr::SelfType,
        TypeNode::Generic { name, args } => {
            let repr_args: Vec<TypeRepr> = args
                .iter()
                .map(|&a| type_node_to_repr(&ast.ty(a).node, ast))
                .collect();
            TypeRepr::Generic((*name).into(), repr_args.into_boxed_slice())
        }
        TypeNode::Nullable { inner } => {
            TypeRepr::Nullable(Box::new(type_node_to_repr(&ast.ty(*inner).node, ast)))
        }
        TypeNode::RefType { inner } => {
            TypeRepr::Ref(Box::new(type_node_to_repr(&ast.ty(*inner).node, ast)))
        }
        TypeNode::RawPtr { inner } => {
            TypeRepr::RawPtr(Box::new(type_node_to_repr(&ast.ty(*inner).node, ast)))
        }
        TypeNode::Function {
            params,
            return_type,
        } => {
            let p: Vec<TypeRepr> = params
                .iter()
                .map(|&a| type_node_to_repr(&ast.ty(a).node, ast))
                .collect();
            let r = type_node_to_repr(&ast.ty(*return_type).node, ast);
            TypeRepr::Function(p.into_boxed_slice(), Box::new(r))
        }
        TypeNode::Record { .. } => TypeRepr::Named("record".into()),
        TypeNode::Array {
            element_type,
            size,
        } => TypeRepr::Array(
            Box::new(type_node_to_repr(&ast.ty(*element_type).node, ast)),
            *size,
        ),
        TypeNode::KindAnnotated { inner, .. } => {
            type_node_to_repr(&ast.ty(*inner).node, ast)
        }
    }
}

/// 将 ConstructorDef 转为 CtorDefInfo。
fn constructor_def_to_ctor_info<'a>(
    arena: &mut TypeArena,
    c: &ConstructorDef<'a>,
    type_name: &str,
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> CtorDefInfo {
    let mut field_names: Vec<Option<Box<str>>> = Vec::with_capacity(c.fields.len());
    let mut field_types: Vec<TypeHandle> = Vec::with_capacity(c.fields.len());
    let mut field_type_reprs: Vec<TypeRepr> = Vec::with_capacity(c.fields.len());

    for f in &c.fields {
        field_names.push(f.name.map(|n| n.into()));
        let ty = concretize_type(arena, f.ty, &[], ast, sema_result);
        field_types.push(ty);
        field_type_reprs.push(type_node_to_repr(&ast.ty(f.ty).node, ast));
    }

    CtorDefInfo {
        name: c.name.into(),
        type_name: type_name.into(),
        field_names: field_names.into_boxed_slice(),
        field_types: field_types.into_boxed_slice(),
        is_newtype: false,
        return_type_name: None,
        return_type_node: c.return_type,
        field_type_reprs: field_type_reprs.into_boxed_slice(),
    }
}

/// 将 RecordFieldType 列表转为单构造器 CtorDefInfo（record 类型）。
fn record_fields_to_ctor_info<'a>(
    arena: &mut TypeArena,
    fields: &[RecordFieldType<'a>],
    type_name: &str,
    ast: &AstArena<'a>,
    sema_result: &mut SemaResult,
) -> CtorDefInfo {
    let mut field_names: Vec<Option<Box<str>>> = Vec::with_capacity(fields.len());
    let mut field_types: Vec<TypeHandle> = Vec::with_capacity(fields.len());
    let mut field_type_reprs: Vec<TypeRepr> = Vec::with_capacity(fields.len());

    for f in fields {
        field_names.push(Some(f.name.into()));
        let ty = concretize_type(arena, f.ty, &[], ast, sema_result);
        field_types.push(ty);
        field_type_reprs.push(type_node_to_repr(&ast.ty(f.ty).node, ast));
    }

    CtorDefInfo {
        name: type_name.into(), // record 构造器名 = 类型名
        type_name: type_name.into(),
        field_names: field_names.into_boxed_slice(),
        field_types: field_types.into_boxed_slice(),
        is_newtype: false,
        return_type_name: None,
        return_type_node: None,
        field_type_reprs: field_type_reprs.into_boxed_slice(),
    }
}

// =========================================================================
// sema v2: Witness Table — trait 实现的静态分派表
//
// 设计理念（原创，非照搬 Swift/Haskell）：
// - trait 实现编译期物化为 WitnessEntry（函数指针表）
// - 通过 Ty 的 type_id 索引分派，O(1)
// - 替代当前 mangled name ("TypeName.method") 查找
// - 与 Kuzo 的 type_id/反射机制天然契合
//
// 数据结构：
// - WitnessEntry { trait_name, type_id, method_slots }
// - WitnessTable 用 Vec<WitnessEntry> + FxHashMap<(trait_name, type_id), idx> 索引
//
// 分派流程：
// 1. 推断接收者类型 → resolve → 取 type_id（标量直接有，ADT 查 type_def）
// 2. 构造 key = (trait_name, type_id)
// 3. 查 witness table → 取 method_slots
// 4. method_slots[method_name] → method slot index
// 5. slot index 指向 MonomorphInstance（已编译的方法体）
// =========================================================================

/// Witness table 条目：一个 trait 在一个类型上的实现。
#[derive(Debug, Clone)]
pub struct WitnessEntry {
    /// trait 名（如 "Show"、"Eq"、"Error"）
    pub trait_name: Box<str>,
    /// 实现类型的 type_id（与 Ty 的 type_id 对应）
    pub type_id: u16,
    /// 方法槽位：method_name → method_idx（在 TypeDefInfo.methods 中的位置）
    pub method_slots: FxHashMap<Box<str>, u16>,
    /// 实现类型的名字（用于错误信息）
    pub type_name: Box<str>,
}

/// Witness table：所有 trait 实现的索引表。
///
/// 通过 (trait_name, type_id) 索引到 WitnessEntry，
/// 再通过 method_name 索引到 method slot。
#[derive(Default, Clone)]
pub struct WitnessTable {
    entries: Vec<WitnessEntry>,
    /// 索引：(trait_name, type_id) → entries 下标
    index: FxHashMap<(Box<str>, u16), u32>,
}

impl WitnessTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个 trait 实现。
    ///
    /// 若 (trait_name, type_id) 已存在，覆盖旧实现（允许重定义）。
    pub fn register(
        &mut self,
        trait_name: &str,
        type_id: u16,
        type_name: &str,
        method_slots: FxHashMap<Box<str>, u16>,
    ) {
        let key = (trait_name.into(), type_id);
        if let Some(&idx) = self.index.get(&key) {
            // 覆盖已有实现
            self.entries[idx as usize] = WitnessEntry {
                trait_name: trait_name.into(),
                type_id,
                method_slots,
                type_name: type_name.into(),
            };
        } else {
            let idx = self.entries.len() as u32;
            self.entries.push(WitnessEntry {
                trait_name: trait_name.into(),
                type_id,
                method_slots,
                type_name: type_name.into(),
            });
            self.index.insert(key, idx);
        }
    }

    /// 查询某类型是否实现了某 trait。
    #[inline]
    pub fn implements(&self, trait_name: &str, type_id: u16) -> bool {
        self.index.contains_key(&(trait_name.into(), type_id))
    }

    /// 查询某 trait 实现的某方法的 method_idx。
    ///
    /// 返回方法在 TypeDefInfo.methods 中的位置索引。
    /// IR 层用 (type_id, method_idx) 查 method_subgraphs 获取子图。
    pub fn resolve_method(
        &self,
        trait_name: &str,
        type_id: u16,
        method_name: &str,
    ) -> Option<u16> {
        let key = (trait_name.into(), type_id);
        let &idx = self.index.get(&key)?;
        let entry = &self.entries[idx as usize];
        entry.method_slots.get(method_name).copied()
    }

    /// 获取某 trait 实现的所有方法名。
    pub fn trait_methods(&self, trait_name: &str, type_id: u16) -> Vec<&str> {
        let key = (trait_name.into(), type_id);
        match self.index.get(&key) {
            Some(&idx) => self.entries[idx as usize]
                .method_slots
                .keys()
                .map(|k| k.as_ref())
                .collect(),
            None => Vec::new(),
        }
    }

    /// 获取所有条目（用于反射/诊断）。
    #[inline]
    pub fn entries(&self) -> &[WitnessEntry] {
        &self.entries
    }

    /// 条目数量。
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
