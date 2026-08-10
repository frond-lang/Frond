//! Ir.rs — 数据流就绪调度执行模型的 IR 核心数据结构
//!
//! 基于 Sema.rs 的 SemaResult 产出，定义：
//! - Node（固定 16B，存拓扑引用，不存值）
//! - InputsPool（独立连续输入池）
//! - ValueTable / Frame（运行时值表，SoA 布局）
//! - SubGraph（函数=子图）
//! - EventSource（channel/async/timer/子图完成事件源声明）
//! - DataFlowGraph（全局图容器）
//! - ComputeFn（构建期绑定的计算函数索引，消除 dispatch）
//!
//! 设计原则（见 docs/superpowers/specs/2026-07-31-dataflow-engine-design.md）：
//! - 节点固定 16B，只存拓扑引用，output 隐含 = 节点自身 id
//! - kind 只有 6 种，仅用于调度器就绪判定，不用于运算分派
//! - compute_fn 是构建期按类型特化绑定的函数索引，运行时数组索引取出调用
//! - 值表槽使用 Value.rs 的 Value enum（含标量与 Arc<HeapObj> 引用）
//! - 独立输入池连续存储所有节点输入，缓存友好

use crate::value::Value;
use std::sync::Arc;

// =========================================================================
// 索引 newtype — 保证类型安全的句柄
// =========================================================================

/// 节点 id（全局连续，值表按此索引）
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// 子图 id（函数=子图）
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubGraphId(pub u32);

/// 函数 id（与 SubGraphId 一一对应，语义别名）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FuncId(pub u32);

/// 子图实例 id（运行时，每次调用一个实例）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubgraphInstanceId(pub u32);

/// 帧 id（运行时）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameId(pub u32);

/// 计算函数索引（指向 COMPUTE_FN_TABLE）
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeFnId(pub u32);

/// ComputeFnId 命名常量生成宏：与 build_compute_fn_table() 索引一一对应。
macro_rules! compute_fn_ids {
    ( $( $idx:literal => $name:ident ),* $(,)? ) => {
        $(
            pub const $name: ComputeFnId = ComputeFnId($idx);
        )*
    };
}

compute_fn_ids! {
    0 => CF_NOOP,
    1 => CF_ADD_I32,
    2 => CF_ADD_F64,
    3 => CF_MUL_I32,
    4 => CF_LE_I32,
    5 => CF_SUB_I32,
    6 => CF_DIV_I32,
    7 => CF_MOD_I32,
    8 => CF_EQ_I32,
    9 => CF_NE_I32,
    10 => CF_LT_I32,
    11 => CF_GT_I32,
    12 => CF_GE_I32,
    13 => CF_SUB_F64,
    14 => CF_MUL_F64,
    15 => CF_DIV_F64,
    16 => CF_EQ_F64,
    17 => CF_NE_F64,
    18 => CF_LT_F64,
    19 => CF_GT_F64,
    20 => CF_LE_F64,
    21 => CF_GE_F64,
    22 => CF_AND_BOOL,
    23 => CF_OR_BOOL,
    24 => CF_NOT_BOOL,
    25 => CF_NEG_I32,
    26 => CF_NEG_F64,
    27 => CF_EQ_BOOL,
    28 => CF_THROW_WRAP_ERR,
    29 => CF_RECORD_CONSTRUCT,
    30 => CF_RECORD_FIELD_GET,
    31 => CF_ARRAY_CONSTRUCT,
    32 => CF_ARRAY_INDEX,
    33 => CF_RECORD_FIELD_SET,
    34 => CF_IS_NULL,
    35 => CF_ARRAY_LEN,
    36 => CF_CALL_LAUNCH,
    37 => CF_GATE_LAUNCH,
    38 => CF_AWAIT,
    39 => CF_ASYNC_CALL_LAUNCH,
    40 => CF_CLOSURE_CONSTRUCT,
    41 => CF_CLOSURE_CALL,
    42 => CF_CANCEL_ASYNC_HANDLE,
    43 => CF_SELECT_GATE,
    44 => CF_THROW_OK,
    45 => CF_THROW_ERR,
    46 => CF_FFI_CALL,
    47 => CF_PROPAGATE,
    48 => CF_SEQ,
    49 => CF_WRITEBACK,
    // i64 算术与比较（50-63）
    50 => CF_ADD_I64,
    51 => CF_SUB_I64,
    52 => CF_MUL_I64,
    53 => CF_DIV_I64,
    54 => CF_MOD_I64,
    55 => CF_EQ_I64,
    56 => CF_NE_I64,
    57 => CF_LT_I64,
    58 => CF_GT_I64,
    59 => CF_LE_I64,
    60 => CF_GE_I64,
    61 => CF_NEG_I64,
    62 => CF_BITNOT_I32,
    63 => CF_BITNOT_I64,
    // i128 算术与比较（64-77）
    64 => CF_ADD_I128,
    65 => CF_SUB_I128,
    66 => CF_MUL_I128,
    67 => CF_DIV_I128,
    68 => CF_MOD_I128,
    69 => CF_EQ_I128,
    70 => CF_NE_I128,
    71 => CF_LT_I128,
    72 => CF_GT_I128,
    73 => CF_LE_I128,
    74 => CF_GE_I128,
    75 => CF_NEG_I128,
    76 => CF_BITNOT_I128,
    // 整数位运算（77-91）
    77 => CF_BITAND_I32,
    78 => CF_BITOR_I32,
    79 => CF_BITXOR_I32,
    80 => CF_BITAND_I64,
    81 => CF_BITOR_I64,
    82 => CF_BITXOR_I64,
    83 => CF_BITAND_I128,
    84 => CF_BITOR_I128,
    85 => CF_BITXOR_I128,
    86 => CF_SHL_I32,
    87 => CF_SHR_I32,
    88 => CF_SHL_I64,
    89 => CF_SHR_I64,
    90 => CF_SHL_I128,
    91 => CF_SHR_I128,
    // 全基本类型 compute_fn（92-259）
    // i8: 92-103
    92 => CF_ADD_I8,
    93 => CF_SUB_I8,
    94 => CF_MUL_I8,
    95 => CF_DIV_I8,
    96 => CF_MOD_I8,
    97 => CF_BITAND_I8,
    98 => CF_BITOR_I8,
    99 => CF_BITXOR_I8,
    100 => CF_SHL_I8,
    101 => CF_SHR_I8,
    102 => CF_NEG_I8,
    103 => CF_BITNOT_I8,
    // i16: 104-115
    104 => CF_ADD_I16,
    105 => CF_SUB_I16,
    106 => CF_MUL_I16,
    107 => CF_DIV_I16,
    108 => CF_MOD_I16,
    109 => CF_BITAND_I16,
    110 => CF_BITOR_I16,
    111 => CF_BITXOR_I16,
    112 => CF_SHL_I16,
    113 => CF_SHR_I16,
    114 => CF_NEG_I16,
    115 => CF_BITNOT_I16,
    // i32: 116-127
    116 => CF_ADD_I32_FULL,
    117 => CF_SUB_I32_FULL,
    118 => CF_MUL_I32_FULL,
    119 => CF_DIV_I32_FULL,
    120 => CF_MOD_I32_FULL,
    121 => CF_BITAND_I32_FULL,
    122 => CF_BITOR_I32_FULL,
    123 => CF_BITXOR_I32_FULL,
    124 => CF_SHL_I32_FULL,
    125 => CF_SHR_I32_FULL,
    126 => CF_NEG_I32_FULL,
    127 => CF_BITNOT_I32_FULL,
    // i64: 128-139
    128 => CF_ADD_I64_FULL,
    129 => CF_SUB_I64_FULL,
    130 => CF_MUL_I64_FULL,
    131 => CF_DIV_I64_FULL,
    132 => CF_MOD_I64_FULL,
    133 => CF_BITAND_I64_FULL,
    134 => CF_BITOR_I64_FULL,
    135 => CF_BITXOR_I64_FULL,
    136 => CF_SHL_I64_FULL,
    137 => CF_SHR_I64_FULL,
    138 => CF_NEG_I64_FULL,
    139 => CF_BITNOT_I64_FULL,
    // i128: 140-151
    140 => CF_ADD_I128_FULL,
    141 => CF_SUB_I128_FULL,
    142 => CF_MUL_I128_FULL,
    143 => CF_DIV_I128_FULL,
    144 => CF_MOD_I128_FULL,
    145 => CF_BITAND_I128_FULL,
    146 => CF_BITOR_I128_FULL,
    147 => CF_BITXOR_I128_FULL,
    148 => CF_SHL_I128_FULL,
    149 => CF_SHR_I128_FULL,
    150 => CF_NEG_I128_FULL,
    151 => CF_BITNOT_I128_FULL,
    // u8: 152-163
    152 => CF_ADD_U8,
    153 => CF_SUB_U8,
    154 => CF_MUL_U8,
    155 => CF_DIV_U8,
    156 => CF_MOD_U8,
    157 => CF_BITAND_U8,
    158 => CF_BITOR_U8,
    159 => CF_BITXOR_U8,
    160 => CF_SHL_U8,
    161 => CF_SHR_U8,
    162 => CF_NEG_U8,
    163 => CF_BITNOT_U8,
    // u16: 164-175
    164 => CF_ADD_U16,
    165 => CF_SUB_U16,
    166 => CF_MUL_U16,
    167 => CF_DIV_U16,
    168 => CF_MOD_U16,
    169 => CF_BITAND_U16,
    170 => CF_BITOR_U16,
    171 => CF_BITXOR_U16,
    172 => CF_SHL_U16,
    173 => CF_SHR_U16,
    174 => CF_NEG_U16,
    175 => CF_BITNOT_U16,
    // u32: 176-187
    176 => CF_ADD_U32,
    177 => CF_SUB_U32,
    178 => CF_MUL_U32,
    179 => CF_DIV_U32,
    180 => CF_MOD_U32,
    181 => CF_BITAND_U32,
    182 => CF_BITOR_U32,
    183 => CF_BITXOR_U32,
    184 => CF_SHL_U32,
    185 => CF_SHR_U32,
    186 => CF_NEG_U32,
    187 => CF_BITNOT_U32,
    // u64: 188-199
    188 => CF_ADD_U64,
    189 => CF_SUB_U64,
    190 => CF_MUL_U64,
    191 => CF_DIV_U64,
    192 => CF_MOD_U64,
    193 => CF_BITAND_U64,
    194 => CF_BITOR_U64,
    195 => CF_BITXOR_U64,
    196 => CF_SHL_U64,
    197 => CF_SHR_U64,
    198 => CF_NEG_U64,
    199 => CF_BITNOT_U64,
    // u128: 200-211
    200 => CF_ADD_U128,
    201 => CF_SUB_U128,
    202 => CF_MUL_U128,
    203 => CF_DIV_U128,
    204 => CF_MOD_U128,
    205 => CF_BITAND_U128,
    206 => CF_BITOR_U128,
    207 => CF_BITXOR_U128,
    208 => CF_SHL_U128,
    209 => CF_SHR_U128,
    210 => CF_NEG_U128,
    211 => CF_BITNOT_U128,
    // isize: 212-223
    212 => CF_ADD_ISIZE,
    213 => CF_SUB_ISIZE,
    214 => CF_MUL_ISIZE,
    215 => CF_DIV_ISIZE,
    216 => CF_MOD_ISIZE,
    217 => CF_BITAND_ISIZE,
    218 => CF_BITOR_ISIZE,
    219 => CF_BITXOR_ISIZE,
    220 => CF_SHL_ISIZE,
    221 => CF_SHR_ISIZE,
    222 => CF_NEG_ISIZE,
    223 => CF_BITNOT_ISIZE,
    // usize: 224-235
    224 => CF_ADD_USIZE,
    225 => CF_SUB_USIZE,
    226 => CF_MUL_USIZE,
    227 => CF_DIV_USIZE,
    228 => CF_MOD_USIZE,
    229 => CF_BITAND_USIZE,
    230 => CF_BITOR_USIZE,
    231 => CF_BITXOR_USIZE,
    232 => CF_SHL_USIZE,
    233 => CF_SHR_USIZE,
    234 => CF_NEG_USIZE,
    235 => CF_BITNOT_USIZE,
    // 浮点 4 类型 × 6 运算
    // f16: 236-241
    236 => CF_ADD_F16,
    237 => CF_SUB_F16,
    238 => CF_MUL_F16,
    239 => CF_DIV_F16,
    240 => CF_MOD_F16,
    241 => CF_NEG_F16,
    // f32: 242-247
    242 => CF_ADD_F32,
    243 => CF_SUB_F32,
    244 => CF_MUL_F32,
    245 => CF_DIV_F32,
    246 => CF_MOD_F32,
    247 => CF_NEG_F32,
    // f64: 248-253
    248 => CF_ADD_F64_FULL,
    249 => CF_SUB_F64_FULL,
    250 => CF_MUL_F64_FULL,
    251 => CF_DIV_F64_FULL,
    252 => CF_MOD_F64_FULL,
    253 => CF_NEG_F64_FULL,
    // f128: 254-259
    254 => CF_ADD_F128,
    255 => CF_SUB_F128,
    256 => CF_MUL_F128,
    257 => CF_DIV_F128,
    258 => CF_MOD_F128,
    259 => CF_NEG_F128,
    // 语义运算（260-265）
    260 => CF_REF_EQ,
    261 => CF_REF_NEQ,
    262 => CF_CONCAT_LIST,
    263 => CF_RANGE,
    264 => CF_RANGE_INCLUSIVE,
    265 => CF_ELVIS,
    // inline_trait / lazy 构造（266-267）
    266 => CF_TRAIT_CONSTRUCT,
    267 => CF_LAZY_CONSTRUCT,
    268 => CF_SLICE,
    269 => CF_STR_CONCAT,
    // 全局变量读写（270-271）
    270 => CF_GLOBAL_LOAD,
    271 => CF_GLOBAL_STORE,
    // 记录扩展 / 原子构造（272-273）
    272 => CF_RECORD_EXTEND,
    273 => CF_ATOMIC_CONSTRUCT,
    // 模式匹配（274-276）
    274 => CF_PATTERN_CTOR_MATCH,
    275 => CF_PATTERN_ADT_FIELD_GET,
    276 => CF_PATTERN_STR_EQ,
    // 通用类型转换（277-278）
    277 => CF_CAST_TO_STR,
    278 => CF_CAST_SCALAR,
    // 引用语义与非空断言（279-282）
    279 => CF_NON_NULL_ASSERT,
    280 => CF_REF_OF,
    281 => CF_DEREF_READ,
    282 => CF_DEREF_WRITE,
    // channel 操作（283-285）
    283 => CF_CHANNEL_CREATE,
    284 => CF_CHANNEL_SEND,
    285 => CF_CHANNEL_CLOSE,
    // 偏应用构造（286）
    286 => CF_PARTIAL_CONSTRUCT,
    // str.bytes() → u8[]（287）
    287 => CF_STR_BYTES,
    // 栈分配版构造（288-289）
    288 => CF_RECORD_CONSTRUCT_STACK,
    289 => CF_ARRAY_CONSTRUCT_STACK,
    // reflect 独立 compute_fn（290-291）：从 compute_ffi_call 拆分，
    // 避免 lazy force 逻辑与 FFI 调用耦合
    290 => CF_REFLECT_FORMAT,
    291 => CF_REFLECT_SCALAR_TO_STR,
    // str 比较（292-297）：按 Unicode 码点序列字典序比较，
    // 不走 i32 路径（str 无 as_i32 语义，走 i32 会恒为 0 导致结果错误）
    292 => CF_EQ_STR,
    293 => CF_NE_STR,
    294 => CF_LT_STR,
    295 => CF_GT_STR,
    296 => CF_LE_STR,
    297 => CF_GE_STR,
    // 复合类型语义相等/不等（298-299）：record/adt/newtype/array/closure/throw 等
    298 => CF_EQ_OBJ,
    299 => CF_NE_OBJ,
    // bool 不等（300）：与 CF_EQ_BOOL(27) 对称，as_i32 对 bool 恒为 0 故不能走 CF_NE_I32
    300 => CF_NE_BOOL,
    // 数组索引存储（301）：arr[i] = x，原地修改 Array 堆对象
    301 => CF_ARRAY_STORE,
    // f128 比较（302-307）：f128 经 to_f64 会丢 60 位精度，需专用 bit-pattern 比较
    302 => CF_EQ_F128,
    303 => CF_NE_F128,
    304 => CF_LT_F128,
    305 => CF_GT_F128,
    306 => CF_LE_F128,
    307 => CF_GE_F128,
    // 记忆化缓存（308-309）：memo_check 查缓存返回 record(hit,value)，memo_store 写缓存透传值
    308 => CF_MEMO_CHECK,
    309 => CF_MEMO_STORE,
    // 尾递归 WriteBack（310）：compute_writeback + 设置 Continue 信号
    310 => CF_TAILREC_WRITEBACK,
    // 控制流 compute_fn（311-313）：替代 control_signal_nodes 表，
    // compute_fn 直接返回 NodeResult::Return/Break/Continue
    311 => CF_RETURN,
    312 => CF_BREAK,
    313 => CF_CONTINUE,
}

// =========================================================================
// NodeKind — 节点种类（非 op，仅 8 种用于就绪判定）
// =========================================================================

/// 节点种类：仅用于调度器判断"如何就绪判定"，不用于运算分派。
///
/// 与传统 IR 的 op（100+ 操作码）根本区别：kind 不参与 dispatch。
/// 具体运算（加减乘除等）由 compute_fn 构建期绑定决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum NodeKind {
    /// 纯计算：常量
    Const = 0,
    /// 纯计算：二元运算（输入就绪即执行）
    BinOp = 1,
    /// 纯计算：一元运算
    UnOp = 2,
    /// 纯计算：字段访问
    FieldAccess = 3,
    /// 函数调用：启动子图 + 等完成事件
    Call = 4,
    /// 事件源消费：等待事件（channel/async/timer）
    Await = 5,
    /// 控制流：条件选择，激活选中子图
    Gate = 6,
    /// 事件源声明：不执行计算，声明外部事件接入点
    EventSource = 7,
}

// =========================================================================
// Node — 固定大小节点（只存拓扑引用，不存值）
// =========================================================================

/// 数据流图节点：固定大小，只存拓扑引用。
///
/// - `kind`：节点种类（仅就绪判定用，不参与运算分派）
/// - `input_count`：输入数量（任意，实际输入在 InputsPool）
/// - `inputs_offset`：在 InputsPool.data 中的起始位置
/// - `compute_fn`：计算函数索引（构建期绑定，运行时数组索引调用）
///
/// output 隐含 = 节点自身 NodeId（值表按 NodeId 索引）。
/// 具体运算由 compute_fn 决定，调度器不关心。
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub kind: NodeKind,
    pub input_count: u8,
    pub inputs_offset: u32,
    pub compute_fn: ComputeFnId,
}
// 布局：kind(1) + input_count(1) + pad(2) + inputs_offset(4) + compute_fn(4) = 12B
// align(16) 强制对齐 16B，size 向上取整为 16B（尾部 pad 4 字节）
const _: () = assert!(std::mem::size_of::<Node>() == 16);

// =========================================================================
// BatchInfo — 编译期 SIMD/并行批量化标记（per-Node，仿 tail_call_flags）
// =========================================================================

/// 批量化运算类型：映射到 Value.rs 的 SIMD/rayon 批算函数。
///
/// 编译期由 compile_binary/compile_unary 设置，运行期 run_ready_nodes 按
/// (ValueTag, BatchOp) 分组就绪节点，复用 Value.rs 的 batch_binop/batch_cmp/
/// batch_unaryop 做 SIMD 向量化 + rayon 并行批算，避免逐节点 compute_fn 开销。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BatchOp {
    /// 二元算术/位运算（返回同类型标量）
    Bin(crate::value::BinOp),
    /// 比较运算（返回 bool）
    Cmp(crate::value::CmpOp),
    /// 一元运算（返回同类型标量）
    Unary(crate::value::UnaryOp),
}

/// 编译期批量化信息（per-Node，按 NodeId 索引）。
///
/// 仅 BinOp/UnOp 且标量类型的节点有 BatchInfo；Call/Gate/Await/record/array/
/// field 等节点为 None（不可批量化，走原有 compute_fn 顺序路径）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchInfo {
    /// 输入/输出的标量类型（决定 SIMD lane 宽度）
    pub tag: crate::value::ValueTag,
    /// 运算类型
    pub op: BatchOp,
}

// =========================================================================
// InputsPool — 独立连续输入池
// =========================================================================

/// 独立输入池：连续存储所有节点的输入 NodeId。
///
/// 节点 N 的输入 = `data[N.inputs_offset .. N.inputs_offset + N.input_count]`。
/// 连续存储保证缓存友好，可批量 SIMD 扫描就绪状态。
pub struct InputsPool {
    pub data: Vec<NodeId>,
}

impl InputsPool {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// 推入一组输入，返回起始 offset。
    pub fn push(&mut self, inputs: &[NodeId]) -> u32 {
        let offset = self.data.len() as u32;
        self.data.extend_from_slice(inputs);
        offset
    }

    /// 读取指定位置的输入切片。
    pub fn get(&self, offset: u32, count: u8) -> &[NodeId] {
        let start = offset as usize;
        let end = start + count as usize;
        &self.data[start..end]
    }

    /// 当前池长度。
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for InputsPool {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// ValueTable — 值表（SoA 布局，运行时，每帧一个）
// =========================================================================

/// 值表（SoA 布局）：values / ready / refcounts 分离存储。
///
/// 相比 AoS（Vec<ValueSlot>），SoA 让 Value 连续排布（stride = sizeof(Value)），
/// 消除 bool/u16 交错，提升 SIMD 批量提取的缓存密度与向量化效率。
///
/// - `values`：节点产出值（按帧内局部 NodeId 索引）
/// - `ready`：是否已产出
/// - `refcounts`：槽级 RC（剩余下游消费者数，0 可回收）
///
/// 槽级 RC：节点产出时设 refcount = 下游数量，每个下游消费时 -1，归零可清槽。
/// 帧级兜底：帧结束时所有未归零槽统一回收（堆对象 Arc Drop 自动 decref）。
#[derive(Clone)]
pub struct ValueTable {
    pub values: Vec<Value>,
    /// 就绪位图（bitmap，每 bit 代表一个节点的就绪状态）。
    /// 替代原 Vec<bool>，8x 压缩（N 节点：N B → N/8 B）。
    pub ready: Vec<u8>,
    pub refcounts: Vec<u16>,
}

impl ValueTable {
    /// 创建空表。
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            ready: Vec::new(),
            refcounts: Vec::new(),
        }
    }

    /// 创建指定容量、全部未就绪的表。
    pub fn with_unready(n: usize) -> Self {
        Self {
            values: vec![Value::NULL; n],
            ready: vec![0u8; (n + 7) / 8],
            refcounts: vec![0; n],
        }
    }

    /// 节点数。
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// 调整尺寸（新槽为未就绪）。
    pub fn resize(&mut self, n: usize) {
        self.values.resize(n, Value::NULL);
        self.ready.resize((n + 7) / 8, 0);
        self.refcounts.resize(n, 0);
    }

    /// 检查节点 idx 是否就绪。
    #[inline]
    pub fn is_ready(&self, idx: usize) -> bool {
        self.ready[idx >> 3] & (1 << (idx & 7)) != 0
    }

    /// 标记节点 idx 为就绪。
    #[inline]
    pub fn set_ready(&mut self, idx: usize) {
        self.ready[idx >> 3] |= 1 << (idx & 7);
    }

    /// 标记节点 idx 为未就绪。
    #[inline]
    pub fn clear_ready(&mut self, idx: usize) {
        self.ready[idx >> 3] &= !(1 << (idx & 7));
    }

    /// 设置产出值 + 下游消费者数量（局部索引）。
    pub fn set_value(&mut self, idx: usize, value: Value, consumer_count: u16) {
        self.values[idx] = value;
        self.set_ready(idx);
        self.refcounts[idx] = consumer_count;
    }

    /// 获取产出值（克隆）。
    pub fn get_value(&self, idx: usize) -> Value {
        self.values[idx].clone()
    }

    /// 获取产出值可变引用（用于 &self 语义直接修改底层 HeapObj）。
    pub fn get_value_mut(&mut self, idx: usize) -> Option<&mut Value> {
        self.values.get_mut(idx)
    }

    /// 消费一次（下游读取）。返回 true 表示 refcount 仍 >0（未归零），
    /// 返回 false 表示已归零可回收。
    pub fn consume(&mut self, idx: usize) -> bool {
        if self.refcounts[idx] > 0 {
            self.refcounts[idx] -= 1;
        }
        self.refcounts[idx] > 0
    }

    /// 是否已被所有消费者消费完（refcount 归零）。
    pub fn is_consumed(&self, idx: usize) -> bool {
        self.is_ready(idx) && self.refcounts[idx] == 0
    }

    /// 重置单个槽为未就绪（堆对象 Arc Drop 自动 decref）。
    pub fn reset_slot(&mut self, idx: usize) {
        self.values[idx] = Value::NULL;
        self.clear_ready(idx);
        self.refcounts[idx] = 0;
    }

    /// 重置所有槽为未就绪（堆对象 Arc Drop 自动 decref）。
    pub fn reset_all(&mut self) {
        for v in self.values.iter_mut() {
            *v = Value::NULL;
        }
        for r in self.ready.iter_mut() {
            *r = 0;
        }
        for rc in self.refcounts.iter_mut() {
            *rc = 0;
        }
    }
}

impl Default for ValueTable {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ValueTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValueTable")
            .field("len", &self.values.len())
            .field("ready_count", &(0..self.values.len()).filter(|&i| self.is_ready(i)).count())
            .finish()
    }
}

// =========================================================================
// ConstValue — 编译期常量原始值（IrBuilder 存储，Engine 分配 ValueHandle）
// =========================================================================

/// 编译期常量原始值（IrBuilder 存储，Engine 分配 ValueHandle）。
///
/// IrBuilder 在编译 Const 节点时将原始值存入 graph.const_values[NodeId]，
/// Engine 在帧初始化时用 ValueArena 分配 ValueHandle 并预填充到 value_table。
///
/// `Str` 变体存 (offset, len) 引用，指向 `DataFlowGraph.string_pool` 中的字节。
/// 访问时通过 `to_value(&pool)` 传入 string pool 切片实时构造 KuzoStr。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConstValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    Isize(isize),
    Usize(usize),
    F32(f32),
    F64(f64),
    F16(u16),
    F128([u8; 16]),
    Bool(bool),
    Char(u32),
    /// 字符串引用：(offset, len) 指向 DataFlowGraph.string_pool
    Str { offset: u32, len: u32 },
    Null,
    Void,
}

impl ConstValue {
    /// 转为 Value（用于优化器/Engine 读取常量值）。
    /// `pool` = DataFlowGraph.string_pool 的字节切片，Str 变体需要从中读取字符串。
    pub fn to_value(&self, pool: &[u8]) -> crate::value::Value {
        match self {
            ConstValue::I8(v) => crate::value::Value::i8(*v),
            ConstValue::I16(v) => crate::value::Value::i16(*v),
            ConstValue::I32(v) => crate::value::Value::i32(*v),
            ConstValue::I64(v) => crate::value::Value::i64(*v),
            ConstValue::I128(v) => crate::value::Value::i128(*v),
            ConstValue::U8(v) => crate::value::Value::u8(*v),
            ConstValue::U16(v) => crate::value::Value::u16(*v),
            ConstValue::U32(v) => crate::value::Value::u32(*v),
            ConstValue::U64(v) => crate::value::Value::u64(*v),
            ConstValue::U128(v) => crate::value::Value::u128(*v),
            ConstValue::Isize(v) => crate::value::Value::isize_val(*v),
            ConstValue::Usize(v) => crate::value::Value::usize_val(*v),
            ConstValue::F32(v) => crate::value::Value::f32(*v),
            ConstValue::F64(v) => crate::value::Value::f64(*v),
            ConstValue::F16(bits) => crate::value::Value::f16(crate::value::F16(*bits)),
            ConstValue::F128(bytes) => crate::value::Value::f128(crate::value::F128(*bytes)),
            ConstValue::Bool(v) => crate::value::Value::bool_val(*v),
            ConstValue::Char(cp) => crate::value::Value::char_val(
                char::from_u32(*cp).unwrap_or('\0'),
            ),
            ConstValue::Str { offset, len } => {
                let off = *offset as usize;
                let end = off + *len as usize;
                let s = std::str::from_utf8(&pool[off..end]).unwrap_or("");
                crate::value::Value::ref_val(
                    crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(s)),
                )
            }
            ConstValue::Null => crate::value::Value::NULL,
            ConstValue::Void => crate::value::Value::VOID,
        }
    }

    /// 从 Value 构造 ConstValue（用于 ConstFold 生成新常量）。
    pub fn from_value(v: &crate::value::Value) -> Option<ConstValue> {
        use crate::value::{ValueTag, Value};
        match v {
            Value::Scalar(sv, tag) => match tag {
                ValueTag::I8 => Some(ConstValue::I8(unsafe { sv.i8_val })),
                ValueTag::I16 => Some(ConstValue::I16(unsafe { sv.i16_val })),
                ValueTag::I32 => Some(ConstValue::I32(unsafe { sv.i32_val })),
                ValueTag::I64 => Some(ConstValue::I64(unsafe { sv.i64_val })),
                ValueTag::I128 => {
                    let bits = unsafe { (sv.i128_val[0] as u128) | ((sv.i128_val[1] as u128) << 64) };
                    Some(ConstValue::I128(bits as i128))
                }
                ValueTag::U8 => Some(ConstValue::U8(unsafe { sv.u8_val })),
                ValueTag::U16 => Some(ConstValue::U16(unsafe { sv.u16_val })),
                ValueTag::U32 => Some(ConstValue::U32(unsafe { sv.u32_val })),
                ValueTag::U64 => Some(ConstValue::U64(unsafe { sv.u64_val })),
                ValueTag::U128 => {
                    let bits = unsafe { (sv.u128_val[0] as u128) | ((sv.u128_val[1] as u128) << 64) };
                    Some(ConstValue::U128(bits))
                }
                ValueTag::Isize => Some(ConstValue::Isize(unsafe { sv.isize_val })),
                ValueTag::Usize => Some(ConstValue::Usize(unsafe { sv.usize_val })),
                ValueTag::F32 => Some(ConstValue::F32(unsafe { sv.f32_val })),
                ValueTag::F64 => Some(ConstValue::F64(unsafe { sv.f64_val })),
                ValueTag::F16 => Some(ConstValue::F16(unsafe { sv.f16_val })),
                ValueTag::F128 => {
                    let lo = unsafe { sv.f128_val[0] };
                    let hi = unsafe { sv.f128_val[1] };
                    let bits: u128 = (lo as u128) | ((hi as u128) << 64);
                    Some(ConstValue::F128(bits.to_le_bytes()))
                }
                ValueTag::Bool => Some(ConstValue::Bool(unsafe { sv.bool_val })),
                ValueTag::Char => Some(ConstValue::Char(unsafe { sv.u32_val })),
                _ => None,
            },
            Value::Null => Some(ConstValue::Null),
            Value::Void => Some(ConstValue::Void),
            _ => None,
        }
    }
}

/// Gate 节点的分支信息。
///
/// Gate 节点根据条件值选择激活哪个分支子图。
/// `condition_input` 是条件值的 NodeId（全局）。
/// `branches` 是分支列表，每个分支携带自己的 inputs（全局 NodeId，值从父帧读取）。
/// 不同分支可有不同数量的 inputs（对应不同 param_count 的子图）。
#[derive(Debug, Clone)]
pub struct GateBranches {
    /// 条件输入节点（全局 NodeId）
    pub condition_input: NodeId,
    /// 分支列表：(条件值, 子图id, 参数节点列表)
    pub branches: Vec<(bool, SubGraphId, Vec<NodeId>)>,
}

/// select 表达式分支信息（按 Gate 节点 NodeId 索引到 select_infos）。
#[derive(Debug, Clone)]
pub struct SelectInfo {
    /// 每个 Receive/Timeout 分支的信息
    pub branches: Vec<SelectBranch>,
}

/// select 表达式的单个分支信息。
#[derive(Debug, Clone)]
pub struct SelectBranch {
    /// 分支子图 id（执行分支 body）
    pub subgraph_id: SubGraphId,
    /// 事件源类型（Channel 或 Timer）
    pub event_kind: EventSourceKind,
    /// 事件源值节点（channel handle 或 timer handle 的 NodeId，全局）
    pub event_source_node: NodeId,
}

// =========================================================================
// ControlSignal — 控制信号（非局部跳转的统一表达）
// =========================================================================

/// 控制信号：非局部跳转的统一表达。
///
/// run_ready_nodes 每次循环检查此字段，非 None 则停止处理。
/// 由控制流 compute_fn（CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR）
/// 返回 NodeResult::Return/Break/Continue 触发。
#[derive(Debug, Clone, Default)]
pub enum ControlSignal {
    /// 无信号，正常执行
    #[default]
    None,
    /// return 语句触发：子图提前返回该值
    Return(Value),
    /// break 语句触发：循环跳出
    Break,
    /// continue 语句触发：循环下一轮
    Continue,
}

/// 检查节点是否为控制流节点（Return/Break/Continue/Throw）。
///
/// 替代旧的 control_signal_nodes 表检查：控制流语义现在通过 compute_fn
///（CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR）直接返回
/// NodeResult::Return/Break/Continue 表达。
pub fn is_control_flow_compute_fn(cf: ComputeFnId) -> bool {
    cf == CF_RETURN || cf == CF_BREAK || cf == CF_CONTINUE || cf == CF_THROW_WRAP_ERR
}

// =========================================================================
// FrameState — 帧状态
// =========================================================================

/// 帧状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameState {
    /// 就绪可执行（有就绪节点）
    Ready,
    /// 正在执行
    Running,
    /// 挂起等事件（async，阶段5实现）
    Suspended,
    /// 取消中（阶段5实现）
    Cancelling,
    /// 完成
    Completed,
    /// 失败
    Failed,
}

// =========================================================================
// SuspendState — 帧挂起状态（偏差 2：call/await 统一挂起模型）
// =========================================================================

/// 帧挂起状态。
///
/// 帧执行到 call/await 节点时挂起，等待事件恢复：
/// - `NotSuspended`：正常运行
/// - `WaitingSubgraph(FrameId)`：等待子图帧完成（sync call 节点用）
/// - `WaitingEvent(NodeId)`：等待 channel/timer/async 事件（await 节点用，NodeId 是 await 节点）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuspendState {
    NotSuspended,
    WaitingSubgraph(FrameId),
    WaitingEvent(NodeId),
}

// =========================================================================
// RuntimeEvent — 运行时事件（子图完成等）
// =========================================================================

/// 运行时事件：驱动挂起帧恢复执行。
///
/// spec 4.4 on_event_arrived 统一处理所有事件源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeEvent {
    /// 子图执行完成（sync call 节点等待的事件）
    SubgraphComplete(FrameId),
    /// channel 有数据可读（await channel.recv() 等待的事件）
    ChannelReady(ChannelId),
    /// timer 到期触发（await timer.sleep() 等待的事件）
    TimerFired(TimerId),
    /// async 调用完成（await async_handle.await() 等待的事件）
    AsyncJoin(AsyncHandleId),
}

// =========================================================================
// PendingCall — 待发起的子图调用（call 节点执行时构造）
// =========================================================================

/// 待发起的子图调用。
///
/// call 节点 compute_fn 执行时构造，调度器消费后启动子图帧。
/// - `is_async=false`：sync call，当前帧挂起等 SubgraphComplete
/// - `is_async=true`：async call，当前帧不挂起，call 节点写 AsyncHandle + 通知下游
#[derive(Debug, Clone)]
pub struct PendingCall {
    /// 目标子图 id
    pub target_sg: SubGraphId,
    /// 调用参数（值列表）
    pub args: Vec<Value>,
    /// 发起调用的节点（帧内局部 NodeId，子图完成后回写返回值）
    pub call_node_local: NodeId,
    /// async call 标记：true=不挂起当前帧，返回 AsyncHandle
    pub is_async: bool,
    /// 逃逸闭包调用时存储 Closure 值，用于子帧完成后回写 upvalue 到 Closure。
    /// None = 普通函数调用或 same_function 闭包调用（无需回写）。
    pub closure_val: Option<Value>,
}

// =========================================================================
// PendingAwait — 待处理的 await 挂起（await 节点执行时构造）
// =========================================================================

/// 待处理的 await 挂起。
///
/// await 节点 compute_fn 执行时构造，核心循环消费后检查事件源就绪状态：
/// 就绪→注入值继续执行，未就绪→注册 event_waiters + 帧挂起。
#[derive(Debug, Clone)]
pub struct PendingAwait {
    /// await 节点（帧内局部 NodeId，事件到达时回写值）
    pub await_node_local: NodeId,
    /// 事件对象值（AsyncHandle/Channel/Timer 的 Value）
    pub event_obj: Value,
    /// 事件种类（决定如何检查就绪 + 如何解析事件源 id）
    pub event_kind: EventSourceKind,
}

// =========================================================================
// Pending — 统一挂起动作枚举
// =========================================================================

// Pending enum 已删除：副作用通过 NodeResult 返回值显式传递。
// 保留 PendingCall/PendingAwait 结构体供 NodeResult::Call/Await 使用。

// =========================================================================
// NodeResult — compute_fn 统一返回值（显式传递所有副作用）
// =========================================================================

/// compute_fn 的统一返回值。
///
/// 所有副作用通过返回值显式传递，消除 frame.pending 隐式副作用。
/// engine 热循环 match NodeResult 分派处理。
#[derive(Debug, Clone)]
pub enum NodeResult {
    /// 正常值计算完成
    Value(Value),
    /// 批量计算完成（多个节点同时产出值）
    Batch(Vec<(NodeId, Value)>),
    /// 函数调用（同步/异步/尾调用，由 PendingCall.is_async 区分）
    Call(PendingCall),
    /// Await 挂起（等待 channel/timer/async 事件）
    Await(PendingAwait),
    /// Channel 通知（Send 操作触发 ChannelReady 唤醒等待帧）
    ChannelNotify(ChannelId),
    /// 取消异步操作
    Cancel(AsyncHandleId),
    /// Select 等待（Gate 无就绪分支时挂起）
    SelectWait(NodeId),
    /// 控制流：return（值作为函数返回值）
    Return(Value),
    /// 控制流：break
    Break,
    /// 控制流：continue
    Continue,
}

// =========================================================================
// EvalContext — compute_fn 执行上下文（提供批处理决策支持）
// =========================================================================

/// compute_fn 执行上下文。
///
/// 不借用 frame 数据（避免与 &mut Frame 借用冲突）。
/// collect_batch_candidates 通过参数接收 &Frame 访问 ready_queue。
pub struct EvalContext {
    /// 子图节点起始偏移（局部 NodeId → 全局 NodeId 转换用）
    pub node_start: u32,
}

impl EvalContext {
    /// 从当前节点之后扫描 ready_queue，收集与当前节点同类型的节点。
    ///
    /// compute_fn 用此方法决定是否做 SIMD 批处理。
    /// 返回全局 NodeId 列表。
    pub fn collect_batch_candidates(
        &self,
        frame: &Frame,
        _current: NodeId,
        predicate: impl Fn(NodeId) -> bool,
    ) -> Vec<NodeId> {
        let mut result = Vec::new();
        for &local_nid in frame.ready_queue.iter() {
            let gid = NodeId(local_nid.0 + self.node_start);
            if predicate(gid) {
                result.push(gid);
            }
        }
        result
    }

    /// ready_queue 长度
    pub fn queue_len(&self, frame: &Frame) -> usize {
        frame.ready_queue.len()
    }
}

// =========================================================================
// Frame — 执行帧（一次函数调用的运行时状态）
// =========================================================================

/// 执行帧：一次函数调用的运行时状态。
///
/// - `value_table`：按 NodeId 索引的值表（SoA 布局，每节点一个槽）
/// - `pending_inputs`：每节点剩余未就绪输入数
/// - `ready_queue`：就绪待执行节点队列
/// - `state`：帧状态
/// - `subgraph_id`：所属子图
/// - `caller`：调用方帧+call节点（子图完成时回写返回值）
///
/// 帧级回收：帧结束时整个 value_table 释放，堆对象走 Arc Drop RC。
pub struct Frame {
    /// 数据流图（只读共享，compute_fn 通过 frame.graph 访问）
    pub graph: std::sync::Arc<DataFlowGraph>,
    /// 值表（SoA 布局，按帧内局部 NodeId 索引，从 0 开始）
    pub value_table: ValueTable,
    /// 每节点剩余未就绪输入数
    pub pending_inputs: Vec<u16>,
    /// 就绪待执行节点队列
    pub ready_queue: std::collections::VecDeque<NodeId>,
    /// 帧状态
    pub state: FrameState,
    /// 所属子图 id
    pub subgraph_id: SubGraphId,
    /// 调用方帧+call节点（None = 顶层帧）
    pub caller: Option<(FrameId, NodeId)>,
    /// 帧 id
    pub id: FrameId,
    /// 子图节点起始偏移（全局 NodeId = 局部 NodeId + node_offset）
    pub node_offset: u32,
    /// 控制信号（return/break/continue 触发）
    pub control_signal: ControlSignal,
    /// 挂起状态（call/await 节点挂起时设置）
    pub suspend_state: SuspendState,
    /// defer 栈（运行时，帧释放时 LIFO 执行）
    pub defer_stack: Vec<DeferEntry>,
    /// 挂起事件（子图完成等，驱动帧恢复）
    pub suspend_event: Option<RuntimeEvent>,
    /// select 中已启动的 timer（branch_idx, timer_id），Timer 分支首次检查时启动
    pub select_timers: Vec<(usize, crate::ir::Ir::TimerId)>,
    /// 指向函数根帧。同函数子图继承，跨函数调用设为 null，async 子帧设为 null。
    /// 安全性由 Box<Frame> 地址稳定 + 同步循环单 worker 保证。
    pub root_frame_ptr: *mut Frame,
    /// 指向直接调用方帧（caller frame）。用于 get_value_by_global 遍历中间帧
    /// （如循环体帧中声明的变量），弥补 root_frame_ptr 只能直达根帧的不足。
    pub parent_frame_ptr: *mut Frame,
    /// 通用缓存子帧 ID（循环体帧复用：while_sg/loop_sg/for_sg/tailrec 帧缓存 body_sg 子帧）。
    pub cached_child_frame: Option<FrameId>,
    /// 逃逸闭包调用时存储 Closure 值，用于子帧完成后回写 upvalue 到 Closure。
    /// None = 普通函数调用或 same_function 闭包调用。
    pub closure_val: Option<Value>,
}

impl Frame {
    /// 创建新帧，值表和 pending_inputs 按子图节点数初始化。
    pub fn new(id: FrameId, subgraph_id: SubGraphId, node_count: usize, graph: std::sync::Arc<DataFlowGraph>) -> Self {
        Self {
            graph,
            value_table: ValueTable::with_unready(node_count),
            pending_inputs: vec![0; node_count],
            ready_queue: std::collections::VecDeque::new(),
            state: FrameState::Ready,
            subgraph_id,
            caller: None,
            id,
            node_offset: 0,
            control_signal: ControlSignal::None,
            suspend_state: SuspendState::NotSuspended,
            defer_stack: Vec::new(),
            suspend_event: None,
            select_timers: Vec::new(),
            root_frame_ptr: std::ptr::null_mut(),
            parent_frame_ptr: std::ptr::null_mut(),
            cached_child_frame: None,
            closure_val: None,
        }
    }

    /// 设置节点的产出值（局部 NodeId）。
    pub fn set_value(&mut self, node: NodeId, value: Value, consumer_count: u16) {
        self.value_table.set_value(node.0 as usize, value, consumer_count);
    }

    /// 获取节点的产出值（局部 NodeId，克隆返回）。
    pub fn get_value(&self, node: NodeId) -> Value {
        self.value_table.get_value(node.0 as usize)
    }

    /// 获取节点的产出值（全局 NodeId，自动转换为局部索引，克隆返回）。
    /// compute_fn 读取输入时使用此方法（inputs_pool 存全局 NodeId）。
    /// 越界时通过 parent_frame_ptr 遍历调用链（中间帧变量），
    /// 再回退到 root_frame_ptr（函数根帧）。
    pub fn get_value_by_global(&self, global_node: NodeId) -> Value {
        let local = global_node.0.wrapping_sub(self.node_offset);
        if (local as usize) < self.value_table.len() {
            if self.value_table.is_ready(local as usize) {
                self.value_table.get_value(local as usize)
            } else if self.pending_inputs[local as usize] > 0 {
                // 节点在当前帧范围内但永不会就绪（嵌套子图节点 pending_inputs=MAX，
                // 或依赖嵌套节点的节点 pending_inputs>0 且永不归零）。
                // 向上查找父帧获取值。
                if !self.parent_frame_ptr.is_null() {
                    unsafe { (*self.parent_frame_ptr).get_value_by_global(global_node) }
                } else if !self.root_frame_ptr.is_null() {
                    unsafe { (*self.root_frame_ptr).get_value_by_global(global_node) }
                } else {
                    Value::NULL
                }
            } else {
                self.value_table.get_value(local as usize)
            }
        } else if !self.parent_frame_ptr.is_null() {
            unsafe { (*self.parent_frame_ptr).get_value_by_global(global_node) }
        } else if !self.root_frame_ptr.is_null() {
            unsafe { (*self.root_frame_ptr).get_value_by_global(global_node) }
        } else {
            Value::NULL
        }
    }

    /// 检查节点是否就绪（所有输入已产出）。
    pub fn is_node_ready(&self, node: NodeId) -> bool {
        self.pending_inputs[node.0 as usize] == 0
    }

    /// 入就绪队列。
    pub fn push_ready(&mut self, node: NodeId) {
        self.ready_queue.push_back(node);
    }

    /// 弹出就绪节点。
    pub fn pop_ready(&mut self) -> Option<NodeId> {
        self.ready_queue.pop_front()
    }
}

// =========================================================================
// EventSource — 事件源（图外运行时对象，产出值注入 await 节点输入边）
// =========================================================================

/// Channel id（运行时）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u64);

/// Timer id（运行时）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimerId(pub u32);

/// Async handle id（运行时，async 调用完成事件）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncHandleId(pub u32);

/// 事件源：图外运行时对象，产出值注入到 await 节点的输入边。
///
/// - await 节点的某个输入边指向 EventSource 声明节点
/// - EventSource 声明节点在运行时绑定到具体 EventSource 实例
/// - 事件到达时，事件源把值写到 await 节点对应输入的值表槽
///
/// call 节点等"子图完成事件"，await 节点等"channel/timer/async 事件"——
/// 两者执行引擎无差别处理，这就是 call 和 await 的统一。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventSource {
    /// channel 有数据可读/可写
    Channel(ChannelId),
    /// async 调用完成
    AsyncJoin(AsyncHandleId),
    /// 定时器到期
    Timer(TimerId),
    /// 子图执行完成（用于 call 节点）
    SubgraphComplete(SubgraphInstanceId),
}

// =========================================================================
// DeferEntry — defer 块（帧 Drop 语义）
// =========================================================================

/// defer 块定义：编译为独立子图，帧释放时按 LIFO 执行。
///
/// 解决 Zig 痛点：defer 挂在帧上，任何帧释放路径都执行
/// （正常返回、错误传播、取消），统一无特例。
#[derive(Debug, Clone)]
pub struct DeferEntry {
    /// defer 注册点（触发节点）
    pub trigger_node: NodeId,
    /// defer 块体子图
    pub body_subgraph: SubGraphId,
    /// 捕获的变量（注册时快照的 NodeId 列表）
    pub captured_inputs: Vec<NodeId>,
    /// 是否已注册到帧的 defer_stack（运行时标记，避免重复执行）
    pub registered: bool,
}

// =========================================================================
// RecordLitInfo — 记录构造信息（RecordLit 节点用）
// =========================================================================

/// 构造种类：区分 Record / ADT / Newtype，驱动 compute_record_construct 构造不同的 HeapObj。
#[derive(Debug, Clone, Copy, PartialEq, Eq, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum RecordLitKind {
    Record = 0,
    Adt = 1,
    Newtype = 2,
}

/// 类型字段信息（注册到 type_scope_stack，按构造器名或类型名索引）。
///
/// field_names: 构造器的字段名列表（Newtype 为空，走单独路径）
/// type_name: 所属类型名（多构造器 ADT 的构造器名 != 类型名，需存储类型名）
/// kind: 构造种类（Record / Adt / Newtype）
#[derive(Debug, Clone)]
pub struct TypeFieldInfo {
    pub field_names: Vec<String>,
    pub type_name: String,
    pub kind: RecordLitKind,
}

/// 记录构造信息（RecordLit 节点用）。
///
/// RecordLit 编译为构造节点，compute_fn 从输入收集字段值构造 HeapObj。
/// 根据 kind 构造 RecordValue / AdtValue / NewtypeValue。
/// type_name 存所属类型名（非构造器名），constructor 存构造器名（ADT 用）。
#[derive(Debug, Clone)]
pub struct RecordLitInfo {
    pub type_name: String,
    pub field_names: Vec<Option<String>>,
    pub constructor: String,
    pub kind: RecordLitKind,
}

/// 闭包构造节点的信息（按 NodeId 索引，非闭包构造节点为 None）。
///
/// 闭包构造节点（compute_fn = 40）运行时从 closure_infos 取子图 id + arity，
/// 合并 inputs（捕获值）构造 Closure 堆对象。
#[derive(Debug, Clone, Copy)]
pub struct ClosureInfo {
    /// 闭包子图 id
    pub subgraph_id: SubGraphId,
    /// lambda 参数数（不含捕获的 upvalues）
    pub arity: u8,
    /// 自身引用 upvalue 的索引（递归嵌套函数用，-1 表示无自身引用）
    pub self_upvalue_idx: i32,
}

/// 偏应用构造节点信息（compute_fn = 286）。
///
/// compile_call 检测到实参数 < 目标函数形参数时生成 partial_construct 节点，
/// 运行时从 partial_infos 取子图 id + bound_count，合并 inputs（已绑定参数值）
/// 构造 HeapObj::Partial。remaining_arity 由 subgraph.param_count - bound_count 推导。
#[derive(Debug, Clone, Copy)]
pub struct PartialInfo {
    /// 目标函数子图 id
    pub subgraph_id: SubGraphId,
    /// 已绑定参数数（= 节点 inputs 数）
    pub bound_count: u8,
}

/// inline_trait 构造节点信息（按 NodeId 索引）。
///
/// compute_trait_construct（compute_fn=266）运行时从此信息取每个方法的
/// 子图 id + arity + upvalue 数量，合并节点 inputs（各方法 upvalues 依次拼接）
/// 构造多个 Closure，打包成 TraitValue 堆对象。
#[derive(Debug, Clone)]
pub struct TraitConstructInfo {
    /// trait 名（运行时填入 TraitValue.trait_name）
    pub trait_name: String,
    /// 方法名列表（与 methods 一一对应，填入 TraitValue.method_names）
    pub method_names: Vec<String>,
    /// 每个方法的子图信息（与 method_names 一一对应）
    pub methods: Vec<TraitMethodEntry>,
}

/// inline_trait 单个方法的子图信息。
#[derive(Debug, Clone, Copy)]
pub struct TraitMethodEntry {
    pub subgraph_id: SubGraphId,
    pub arity: u8,         // 方法参数数（不含 upvalues）
    pub upvalue_count: u8, // 该方法的 upvalue 数（从 inputs 中按顺序切分）
}

/// lazy 构造节点信息（按 NodeId 索引）。
///
/// compute_lazy_construct（compute_fn=267）运行时从此信息取 thunk 子图 id，
/// 构造 LazyValue 堆对象（thunk 未求值，首次 force 时启动子图计算并缓存）。
#[derive(Debug, Clone, Copy)]
pub struct LazyConstructInfo {
    /// thunk 子图 id（无参数，返回值为 lazy 表达式的值）
    pub thunk_sg: SubGraphId,
}

/// 记录扩展节点信息（按 NodeId 索引）。
///
/// compute_record_extend（compute_fn=272）运行时从此信息取更新字段名列表，
/// 从 base RecordValue 克隆字段，按更新字段名替换/追加，构造新 RecordValue。
/// inputs[0] = base record，inputs[1..] = 更新字段值（顺序对应 update_names）。
#[derive(Debug, Clone)]
pub struct RecordExtendInfo {
    /// 更新字段名列表（长度 = input_count - 1，对应 inputs[1..]）
    pub update_names: Vec<String>,
}

/// 记忆化缓存节点元数据：memo_check / memo_store 共用。
/// memo_check: inputs[0..param_count] = 参数值，table_index 索引缓存表
/// memo_store: inputs[0..param_count] = 参数值, inputs[param_count] = 结果值
#[derive(Debug, Clone)]
pub struct MemoInfo {
    /// 缓存表索引（graph.memo_tables 中的位置）
    pub table_index: u32,
    /// 参数个数（inputs 前 param_count 个为缓存 key 组成部分）
    pub param_count: u8,
}

// =========================================================================
// EventSourceDecl — 事件源声明（静态，编译期）
// =========================================================================

/// 事件源声明：在子图中声明外部事件接入点。
///
/// await 节点的某个输入边指向 EventSource 声明节点，
/// 运行时绑定到具体 EventSource 实例。
#[derive(Debug, Clone)]
pub struct EventSourceDecl {
    /// 声明所在节点
    pub node: NodeId,
    /// 事件源种类（运行时绑定实例）
    pub kind: EventSourceKind,
}

/// 事件源种类（静态声明，运行时绑定实例）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum EventSourceKind {
    /// channel 事件
    Channel,
    /// async join 事件
    AsyncJoin,
    /// timer 事件
    Timer,
    /// 子图完成事件
    SubgraphComplete,
}

// =========================================================================
// SubGraph — 函数子图（静态，编译期生成）
// =========================================================================

/// 函数子图：每个函数（含单态化实例）编译为一个 SubGraph。
///
/// - `node_range`：节点 id 范围 [start, end)
/// - `entry_node`：入口节点（接收参数）
/// - `return_node`：返回节点（产出返回值）
/// - `has_suspend`：是否有挂起点（async=true）
///
/// sync 函数 = 子图无挂起点，同步跑完立即产完成事件
/// async 函数 = 子图有挂起点（await 节点连事件源）
/// 区别仅在子图是否有 await 节点，执行引擎无差别处理。
#[derive(Debug, Clone)]
pub struct SubGraph {
    /// 子图 id
    pub id: SubGraphId,
    /// 节点 id 范围 [start, end)
    pub node_range: (NodeId, NodeId),
    /// 参数数（入口节点的输入数）
    pub param_count: u8,
    /// 入口节点（接收参数）
    pub entry_node: NodeId,
    /// 返回节点（产出返回值）
    pub return_node: NodeId,
    /// 是否有挂起点（async=true）
    pub has_suspend: bool,
    /// 声明的事件源（channel/timer 等）
    pub event_source_decls: Vec<EventSourceDecl>,
    /// defer 块子图定义
    pub defer_table: Vec<DeferEntry>,
    /// 循环种类（普通子图=None，while_sg=While，loop_sg=Loop，for_sg=For，body_sg=LoopBody）
    pub loop_kind: LoopKind,
    /// body_sg 指向父循环子图（while_sg/loop_sg/for_sg）
    pub loop_parent_sg: Option<SubGraphId>,
    /// 循环条件节点（While/For 用，循环重置时需重置）
    pub cond_node: Option<NodeId>,
    /// 所属函数 ID（顶层函数子图=自身 SubGraphId.0，循环/分支子图=父函数的 function_id）
    pub function_id: u32,
    /// For 循环迭代器推进节点（reset_loop_iteration 时重置）
    pub iter_next_node: Option<NodeId>,
    /// upvalue 数量（lambda 捕获变量数，含 self 递归引用）
    /// param_count = 实际参数数 + upvalue_count
    pub upvalue_count: u8,
    /// 每个 upvalue 对应的外层节点 ID（用于 same_function 调用时注入当前父帧值）
    pub upvalue_outer_nodes: Vec<NodeId>,
    /// 直接嵌套子图的 node_range 列表（构建期预计算，运行时 O(len) 查询而非全图扫描）。
    /// 仅包含直接嵌套的子图，不含孙子图（孙子图由递归的 prepare 逻辑处理）。
    pub nested_ranges: Vec<(u32, u32)>,
    /// 帧复用重置计划（编译期生成，替代运行时 LoopKind 分支判断）。
    /// 仅循环子图（while_sg/loop_sg/for_sg）有此计划，普通子图为 None。
    pub reset_plan: Option<ResetPlan>,
}

/// 子图帧复用时的重置计划（编译期由 Builder 计算，存入 SubGraph）。
///
/// 将 For vs While/Loop 的重置差异编码为数据，engine 不再分支判断 LoopKind。
#[derive(Debug, Clone, Default)]
pub struct ResetPlan {
    /// 重置为 pending=0 并入队的节点（For 的 iter_next_node）
    pub reset_to_zero: Vec<NodeId>,
    /// 重置为 pending=1 的节点（For 的 cond_node，输入来自 iter_next）
    pub reset_to_one: Vec<NodeId>,
    /// 需递归重置的条件树根节点（While/Loop 的 cond_node）
    pub reset_condition_tree: Vec<NodeId>,
}

/// 循环子图种类
#[derive(Debug, Clone, Copy, PartialEq, Eq, num_enum::TryFromPrimitive)]
#[repr(u8)]
pub enum LoopKind {
    /// 普通子图
    None,
    /// while_sg（含 cond + Gate）
    While,
    /// loop_sg（无 cond，靠 break 终止）
    Loop,
    /// for_sg（含迭代器 + cond）
    For,
    /// body_sg（循环体，不尾递归）
    LoopBody,
    /// 尾递归转迭代循环（cond-based Gate + Continue 信号退出机制）
    /// WriteBack 设置 Continue → 循环继续；body_sg 无信号完成 → 命中 base case → 循环退出
    TailRec,
}

// =========================================================================
// ComputeFn — 计算函数（构建期绑定，消除 dispatch）
// =========================================================================

/// 计算函数签名：接收帧 + 节点 id + 执行上下文，返回 NodeResult。
///
/// frame 持有 graph（Arc<DataFlowGraph>），compute_fn 通过 frame.graph 访问图数据。
/// 构建期绑定索引（ComputeFnId），运行时通过计算函数表索引调用。
/// 每种运算+类型组合一个特化函数，运行时无类型检查、无 op 查表。
/// 所有副作用通过 NodeResult 返回值显式传递。
pub type ComputeFn = fn(frame: &mut Frame, node: NodeId, ctx: &EvalContext) -> NodeResult;

/// wrapper 宏：将旧签名 `fn(&mut Frame, NodeId) -> Value` 包装为新签名
/// `fn(&mut Frame, NodeId, &EvalContext) -> NodeResult`。
///
/// 对于有 BatchInfo 的节点（BinOp/UnOp/Cmp），通过 EvalContext 检查 ready_queue
/// 中是否有同类型就绪节点。若有 ≥2 个（含当前节点），做 SIMD 批量计算并返回
/// NodeResult::Batch；否则回退到单节点计算。
/// 对于无 BatchInfo 的节点（Call/Gate/Await/record/array 等），直接走单节点路径。
macro_rules! wrap_fn {
    ($f:expr) => {{
        fn wrapper(frame: &mut Frame, node: NodeId, ctx: &EvalContext) -> NodeResult {
            // safe_op 短路：?. 标记的节点在接收者（inputs[0]）为 null 时返回 Null，
            // 不执行后续计算（字段访问/方法调用/intrinsic 等）。
            // 这是数据驱动的统一短路逻辑，由编译期 set_safe_op 标记触发。
            if frame.graph.safe_op_flag(node.0 as usize) {
                let n = frame.graph.node(node.0 as usize);
                if n.input_count > 0 {
                    let inputs = frame.graph.inputs(n.inputs_offset, n.input_count);
                    let recv = frame.get_value_by_global(inputs[0]);
                    if recv.is_null() {
                        return NodeResult::Value(Value::Null);
                    }
                }
            }
            // SIMD 批处理决策：检查 batch_infos，若有同类型就绪节点则批量计算
            if let Some(info) = frame.graph.batch_info(node.0 as usize) {
                let graph = frame.graph.clone();
                let candidates = ctx.collect_batch_candidates(frame, node, |gid| {
                    graph.batch_info(gid.0 as usize) == Some(info)
                });
                if !candidates.is_empty() {
                    let mut all_locals = Vec::with_capacity(candidates.len() + 1);
                    all_locals.push(NodeId(node.0.wrapping_sub(ctx.node_start)));
                    for &gid in &candidates {
                        all_locals.push(NodeId(gid.0.wrapping_sub(ctx.node_start)));
                    }
                    if let Some(results) = super::Compute::do_simd_batch(
                        frame, &all_locals, info, ctx.node_start,
                    ) {
                        return NodeResult::Batch(results);
                    }
                }
            }
            NodeResult::Value($f(frame, node))
        }
        wrapper as ComputeFn
    }};
}

/// 计算函数表注册宏。
///
/// 接收 `idx => fn_path` 对列表，展开为带运行时索引断言的 Vec 构造。
/// 每项 push 后立即断言 `table.len() == idx + 1`，确保索引与实际位置一致。
/// 若删除某项但忘记更新后续索引，断言会立即失败，防止 ComputeFnId 错位。
/// 过渡期自动用 wrap_fn! 包装每个条目。
macro_rules! compute_fn_table {
    ( $( $idx:literal => $f:expr ),* $(,)? ) => {{
        let mut table: Vec<ComputeFn> = Vec::new();
        $(
            table.push(wrap_fn!($f));
            assert_eq!(table.len(), ($idx as usize) + 1,
                concat!("compute_fn_table: index ", stringify!($idx), " mismatch"));
        )*
        table
    }};
}

/// 构建真实计算函数表（引用 Engine 模块的 compute_* 函数）。
///
/// 索引与 ComputeFnId 一一对应，IrBuilder::build() 末尾填充到 graph.compute_fns。
/// 使用 `compute_fn_table!` 宏：每项 `idx => fn_path` 自动生成运行时断言，
/// 确保索引与实际位置一致——若删除某项但忘记更新后续索引，断言会立即失败。
pub fn build_compute_fn_table() -> Vec<ComputeFn> {
    let mut table = compute_fn_table! {
        0   => super::Compute::noop_compute_real,
        1   => super::Compute::compute_add_i32,
        2   => super::Compute::compute_add_f64,
        3   => super::Compute::compute_mul_i32,
        4   => super::Compute::compute_le_i32,
        5   => super::Compute::compute_sub_i32,
        6   => super::Compute::compute_div_i32,
        7   => super::Compute::compute_mod_i32,
        8   => super::Compute::compute_eq_i32,
        9   => super::Compute::compute_ne_i32,
        10  => super::Compute::compute_lt_i32,
        11  => super::Compute::compute_gt_i32,
        12  => super::Compute::compute_ge_i32,
        13  => super::Compute::compute_sub_f64,
        14  => super::Compute::compute_mul_f64,
        15  => super::Compute::compute_div_f64,
        16  => super::Compute::compute_eq_f64,
        17  => super::Compute::compute_ne_f64,
        18  => super::Compute::compute_lt_f64,
        19  => super::Compute::compute_gt_f64,
        20  => super::Compute::compute_le_f64,
        21  => super::Compute::compute_ge_f64,
        22  => super::Compute::compute_and_bool,
        23  => super::Compute::compute_or_bool,
        24  => super::Compute::compute_not_bool,
        25  => super::Compute::compute_neg_i32,
        26  => super::Compute::compute_neg_f64,
        27  => super::Compute::compute_eq_bool,
        28  => super::Compute::noop_compute_real, // compute_throw_wrap_err — 新签名，table override
        29  => super::Compute::compute_record_construct,
        30  => super::Compute::compute_record_field_get,
        31  => super::Compute::compute_array_construct,
        32  => super::Compute::compute_array_index,
        33  => super::Compute::compute_record_field_set,
        34  => super::Compute::compute_is_null,
        35  => super::Compute::compute_array_len,
        36  => super::Compute::noop_compute_real, // compute_call_launch — 新签名，table override
        37  => super::Compute::noop_compute_real, // compute_gate_launch — 新签名，table override
        38  => super::Compute::noop_compute_real, // compute_await — 新签名，table override
        39  => super::Compute::noop_compute_real, // compute_call_launch alias — 新签名，table override
        40  => super::Compute::compute_closure_construct,
        41  => super::Compute::noop_compute_real, // compute_closure_call — 新签名，table override
        42  => super::Compute::noop_compute_real, // compute_cancel_async_handle — 新签名，table override
        43  => super::Compute::noop_compute_real, // compute_select_gate — 新签名，table override
        44  => super::Compute::compute_throw_ok,
        45  => super::Compute::compute_throw_err,
        46  => super::Compute::compute_ffi_call,
        47  => super::Compute::noop_compute_real, // compute_propagate — 新签名，table override
        48  => super::Compute::compute_seq,
        49  => super::Compute::noop_compute_real, // compute_writeback — 新签名，table override
        // i64 算术与比较（50-63）
        50  => super::Compute::compute_add_i64,
        51  => super::Compute::compute_sub_i64,
        52  => super::Compute::compute_mul_i64,
        53  => super::Compute::compute_div_i64,
        54  => super::Compute::compute_mod_i64,
        55  => super::Compute::compute_eq_i64,
        56  => super::Compute::compute_ne_i64,
        57  => super::Compute::compute_lt_i64,
        58  => super::Compute::compute_gt_i64,
        59  => super::Compute::compute_le_i64,
        60  => super::Compute::compute_ge_i64,
        61  => super::Compute::compute_neg_i64,
        62  => super::Compute::compute_bitnot_i32,
        63  => super::Compute::compute_bitnot_i64,
        // i128 算术与比较（64-77）
        64  => super::Compute::compute_add_i128,
        65  => super::Compute::compute_sub_i128,
        66  => super::Compute::compute_mul_i128,
        67  => super::Compute::compute_div_i128,
        68  => super::Compute::compute_mod_i128,
        69  => super::Compute::compute_eq_i128,
        70  => super::Compute::compute_ne_i128,
        71  => super::Compute::compute_lt_i128,
        72  => super::Compute::compute_gt_i128,
        73  => super::Compute::compute_le_i128,
        74  => super::Compute::compute_ge_i128,
        75  => super::Compute::compute_neg_i128,
        76  => super::Compute::compute_bitnot_i128,
        // 整数位运算（77-92）：BitAnd/BitOr/BitXor/Shl/Shr × i32/i64/i128
        77  => super::Compute::compute_bitand_i32,
        78  => super::Compute::compute_bitor_i32,
        79  => super::Compute::compute_bitxor_i32,
        80  => super::Compute::compute_bitand_i64,
        81  => super::Compute::compute_bitor_i64,
        82  => super::Compute::compute_bitxor_i64,
        83  => super::Compute::compute_bitand_i128,
        84  => super::Compute::compute_bitor_i128,
        85  => super::Compute::compute_bitxor_i128,
        86  => super::Compute::compute_shl_i32,
        87  => super::Compute::compute_shr_i32,
        88  => super::Compute::compute_shl_i64,
        89  => super::Compute::compute_shr_i64,
        90  => super::Compute::compute_shl_i128,
        91  => super::Compute::compute_shr_i128,
        // ---- 全基本类型 compute_fn（92-259）----
        // 整数 12 类型 × 12 运算（add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot）
        // i8: 92-103
        92  => super::Compute::compute_add_i8,
        93  => super::Compute::compute_sub_i8,
        94  => super::Compute::compute_mul_i8,
        95  => super::Compute::compute_div_i8,
        96  => super::Compute::compute_mod_i8,
        97  => super::Compute::compute_bitand_i8,
        98  => super::Compute::compute_bitor_i8,
        99  => super::Compute::compute_bitxor_i8,
        100 => super::Compute::compute_shl_i8,
        101 => super::Compute::compute_shr_i8,
        102 => super::Compute::compute_neg_i8,
        103 => super::Compute::compute_bitnot_i8,
        // i16: 104-115
        104 => super::Compute::compute_add_i16,
        105 => super::Compute::compute_sub_i16,
        106 => super::Compute::compute_mul_i16,
        107 => super::Compute::compute_div_i16,
        108 => super::Compute::compute_mod_i16,
        109 => super::Compute::compute_bitand_i16,
        110 => super::Compute::compute_bitor_i16,
        111 => super::Compute::compute_bitxor_i16,
        112 => super::Compute::compute_shl_i16,
        113 => super::Compute::compute_shr_i16,
        114 => super::Compute::compute_neg_i16,
        115 => super::Compute::compute_bitnot_i16,
        // i32: 116-127
        116 => super::Compute::compute_add_i32,
        117 => super::Compute::compute_sub_i32,
        118 => super::Compute::compute_mul_i32,
        119 => super::Compute::compute_div_i32,
        120 => super::Compute::compute_mod_i32,
        121 => super::Compute::compute_bitand_i32,
        122 => super::Compute::compute_bitor_i32,
        123 => super::Compute::compute_bitxor_i32,
        124 => super::Compute::compute_shl_i32,
        125 => super::Compute::compute_shr_i32,
        126 => super::Compute::compute_neg_i32,
        127 => super::Compute::compute_bitnot_i32,
        // i64: 128-139
        128 => super::Compute::compute_add_i64,
        129 => super::Compute::compute_sub_i64,
        130 => super::Compute::compute_mul_i64,
        131 => super::Compute::compute_div_i64,
        132 => super::Compute::compute_mod_i64,
        133 => super::Compute::compute_bitand_i64,
        134 => super::Compute::compute_bitor_i64,
        135 => super::Compute::compute_bitxor_i64,
        136 => super::Compute::compute_shl_i64,
        137 => super::Compute::compute_shr_i64,
        138 => super::Compute::compute_neg_i64,
        139 => super::Compute::compute_bitnot_i64,
        // i128: 140-151
        140 => super::Compute::compute_add_i128,
        141 => super::Compute::compute_sub_i128,
        142 => super::Compute::compute_mul_i128,
        143 => super::Compute::compute_div_i128,
        144 => super::Compute::compute_mod_i128,
        145 => super::Compute::compute_bitand_i128,
        146 => super::Compute::compute_bitor_i128,
        147 => super::Compute::compute_bitxor_i128,
        148 => super::Compute::compute_shl_i128,
        149 => super::Compute::compute_shr_i128,
        150 => super::Compute::compute_neg_i128,
        151 => super::Compute::compute_bitnot_i128,
        // u8: 152-163
        152 => super::Compute::compute_add_u8,
        153 => super::Compute::compute_sub_u8,
        154 => super::Compute::compute_mul_u8,
        155 => super::Compute::compute_div_u8,
        156 => super::Compute::compute_mod_u8,
        157 => super::Compute::compute_bitand_u8,
        158 => super::Compute::compute_bitor_u8,
        159 => super::Compute::compute_bitxor_u8,
        160 => super::Compute::compute_shl_u8,
        161 => super::Compute::compute_shr_u8,
        162 => super::Compute::compute_neg_u8,
        163 => super::Compute::compute_bitnot_u8,
        // u16: 164-175
        164 => super::Compute::compute_add_u16,
        165 => super::Compute::compute_sub_u16,
        166 => super::Compute::compute_mul_u16,
        167 => super::Compute::compute_div_u16,
        168 => super::Compute::compute_mod_u16,
        169 => super::Compute::compute_bitand_u16,
        170 => super::Compute::compute_bitor_u16,
        171 => super::Compute::compute_bitxor_u16,
        172 => super::Compute::compute_shl_u16,
        173 => super::Compute::compute_shr_u16,
        174 => super::Compute::compute_neg_u16,
        175 => super::Compute::compute_bitnot_u16,
        // u32: 176-187
        176 => super::Compute::compute_add_u32,
        177 => super::Compute::compute_sub_u32,
        178 => super::Compute::compute_mul_u32,
        179 => super::Compute::compute_div_u32,
        180 => super::Compute::compute_mod_u32,
        181 => super::Compute::compute_bitand_u32,
        182 => super::Compute::compute_bitor_u32,
        183 => super::Compute::compute_bitxor_u32,
        184 => super::Compute::compute_shl_u32,
        185 => super::Compute::compute_shr_u32,
        186 => super::Compute::compute_neg_u32,
        187 => super::Compute::compute_bitnot_u32,
        // u64: 188-199
        188 => super::Compute::compute_add_u64,
        189 => super::Compute::compute_sub_u64,
        190 => super::Compute::compute_mul_u64,
        191 => super::Compute::compute_div_u64,
        192 => super::Compute::compute_mod_u64,
        193 => super::Compute::compute_bitand_u64,
        194 => super::Compute::compute_bitor_u64,
        195 => super::Compute::compute_bitxor_u64,
        196 => super::Compute::compute_shl_u64,
        197 => super::Compute::compute_shr_u64,
        198 => super::Compute::compute_neg_u64,
        199 => super::Compute::compute_bitnot_u64,
        // u128: 200-211
        200 => super::Compute::compute_add_u128,
        201 => super::Compute::compute_sub_u128,
        202 => super::Compute::compute_mul_u128,
        203 => super::Compute::compute_div_u128,
        204 => super::Compute::compute_mod_u128,
        205 => super::Compute::compute_bitand_u128,
        206 => super::Compute::compute_bitor_u128,
        207 => super::Compute::compute_bitxor_u128,
        208 => super::Compute::compute_shl_u128,
        209 => super::Compute::compute_shr_u128,
        210 => super::Compute::compute_neg_u128,
        211 => super::Compute::compute_bitnot_u128,
        // isize: 212-223
        212 => super::Compute::compute_add_isize,
        213 => super::Compute::compute_sub_isize,
        214 => super::Compute::compute_mul_isize,
        215 => super::Compute::compute_div_isize,
        216 => super::Compute::compute_mod_isize,
        217 => super::Compute::compute_bitand_isize,
        218 => super::Compute::compute_bitor_isize,
        219 => super::Compute::compute_bitxor_isize,
        220 => super::Compute::compute_shl_isize,
        221 => super::Compute::compute_shr_isize,
        222 => super::Compute::compute_neg_isize,
        223 => super::Compute::compute_bitnot_isize,
        // usize: 224-235
        224 => super::Compute::compute_add_usize,
        225 => super::Compute::compute_sub_usize,
        226 => super::Compute::compute_mul_usize,
        227 => super::Compute::compute_div_usize,
        228 => super::Compute::compute_mod_usize,
        229 => super::Compute::compute_bitand_usize,
        230 => super::Compute::compute_bitor_usize,
        231 => super::Compute::compute_bitxor_usize,
        232 => super::Compute::compute_shl_usize,
        233 => super::Compute::compute_shr_usize,
        234 => super::Compute::compute_neg_usize,
        235 => super::Compute::compute_bitnot_usize,
        // 浮点 4 类型 × 6 运算（add/sub/mul/div/mod/neg）
        // f16: 236-241
        236 => super::Compute::compute_add_f16,
        237 => super::Compute::compute_sub_f16,
        238 => super::Compute::compute_mul_f16,
        239 => super::Compute::compute_div_f16,
        240 => super::Compute::compute_mod_f16,
        241 => super::Compute::compute_neg_f16,
        // f32: 242-247
        242 => super::Compute::compute_add_f32,
        243 => super::Compute::compute_sub_f32,
        244 => super::Compute::compute_mul_f32,
        245 => super::Compute::compute_div_f32,
        246 => super::Compute::compute_mod_f32,
        247 => super::Compute::compute_neg_f32,
        // f64: 248-253
        248 => super::Compute::compute_add_f64,
        249 => super::Compute::compute_sub_f64,
        250 => super::Compute::compute_mul_f64,
        251 => super::Compute::compute_div_f64,
        252 => super::Compute::compute_mod_f64,
        253 => super::Compute::compute_neg_f64,
        // f128: 254-259
        254 => super::Compute::compute_add_f128,
        255 => super::Compute::compute_sub_f128,
        256 => super::Compute::compute_mul_f128,
        257 => super::Compute::compute_div_f128,
        258 => super::Compute::compute_mod_f128,
        259 => super::Compute::compute_neg_f128,
        // 语义运算（260-265）：RefEq/RefNeq/ConcatList/Range/RangeInclusive/Elvis
        260 => super::Compute::compute_ref_eq,
        261 => super::Compute::compute_ref_neq,
        262 => super::Compute::compute_concat_list,
        263 => super::Compute::compute_range,
        264 => super::Compute::compute_range_inclusive,
        265 => super::Compute::compute_elvis,
        // inline_trait / lazy 构造（266-267）
        266 => super::Compute::compute_trait_construct,
        267 => super::Compute::compute_lazy_construct,
        268 => super::Compute::compute_slice,
        269 => super::Compute::compute_str_concat,
        // 全局变量读写（270-271）
        270 => super::Compute::compute_global_load,
        271 => super::Compute::compute_global_store,
        // 记录扩展 / 原子构造（272-273）
        272 => super::Compute::compute_record_extend,
        273 => super::Compute::compute_atomic_construct,
        // 模式匹配（274-276）
        274 => super::Compute::compute_pattern_ctor_match,
        275 => super::Compute::compute_pattern_adt_field_get,
        276 => super::Compute::compute_pattern_str_eq,
        // 通用类型转换（277-278）
        277 => super::Compute::compute_cast_to_str,
        278 => super::Compute::compute_cast_scalar,
        // 引用语义与非空断言（279-282）
        279 => super::Compute::compute_non_null_assert,
        280 => super::Compute::compute_ref_of,
        281 => super::Compute::compute_deref_read,
        282 => super::Compute::compute_deref_write,
        // channel 操作（283-285）
        283 => super::Compute::compute_channel_create,
        284 => super::Compute::noop_compute_real, // compute_channel_send — 新签名，table override
        285 => super::Compute::compute_channel_close,
        // 偏应用构造（286）
        286 => super::Compute::compute_partial_construct,
        // str.bytes() → u8[]（287）
        287 => super::Compute::compute_str_bytes,
        // 栈分配版构造（288-289）：分析器标记的不逃逸分配点使用
        288 => super::Compute::compute_record_construct_stack,
        289 => super::Compute::compute_array_construct_stack,
        // reflect 独立 compute_fn（290-291）：lazy force + Reflect::format_value
        290 => super::Compute::compute_reflect_format,
        291 => super::Compute::compute_reflect_scalar_to_str,
        // str 比较（292-297）：按 Unicode 码点序列字典序比较
        292 => super::Compute::compute_eq_str,
        293 => super::Compute::compute_ne_str,
        294 => super::Compute::compute_lt_str,
        295 => super::Compute::compute_gt_str,
        296 => super::Compute::compute_le_str,
        297 => super::Compute::compute_ge_str,
        // 复合类型语义相等/不等（298-299）
        298 => super::Compute::compute_eq_obj,
        299 => super::Compute::compute_ne_obj,
        // bool 不等（300）
        300 => super::Compute::compute_ne_bool,
        // 数组索引存储（301）
        301 => super::Compute::compute_array_store,
        // f128 比较（302-307）
        302 => super::Compute::compute_eq_f128,
        303 => super::Compute::compute_ne_f128,
        304 => super::Compute::compute_lt_f128,
        305 => super::Compute::compute_gt_f128,
        306 => super::Compute::compute_le_f128,
        307 => super::Compute::compute_ge_f128,
        // 记忆化缓存（308-309）
        308 => super::Compute::compute_memo_check,
        309 => super::Compute::compute_memo_store,
        // 尾递归 WriteBack（310）
        310 => super::Compute::noop_compute_real, // compute_tailrec_writeback — 新签名，table override
        // 控制流 compute_fn（311-313）— 新签名，table override
        311 => super::Compute::noop_compute_real, // compute_return
        312 => super::Compute::noop_compute_real, // compute_break
        313 => super::Compute::noop_compute_real, // compute_continue
    };
    // 替换 index 0 为 compute_const（不包装，直接使用新签名）
    // Const 节点使用 CF_NOOP(0)，通过 compute_const 从 const_values 物化值
    table[0] = super::Compute::compute_const;
    // 已迁移到新签名的 compute_fn（不通过 wrap_fn! 包装，直接使用新签名）
    table[28] = super::Compute::compute_throw_wrap_err;
    table[36] = super::Compute::compute_call_launch;
    table[37] = super::Compute::compute_gate_launch;
    table[38] = super::Compute::compute_await;
    table[39] = super::Compute::compute_call_launch; // CF_ASYNC_CALL_LAUNCH 别名
    table[41] = super::Compute::compute_closure_call;
    table[42] = super::Compute::compute_cancel_async_handle;
    table[43] = super::Compute::compute_select_gate;
    table[47] = super::Compute::compute_propagate;
    table[284] = super::Compute::compute_channel_send;
    table[310] = super::Compute::compute_tailrec_writeback;
    table[311] = super::Compute::compute_return;
    table[312] = super::Compute::compute_break;
    table[313] = super::Compute::compute_continue;
    table[49] = super::Compute::compute_writeback;
    table
}

/// 纯 compute_fn 集合（无副作用，可 CSE/DCE）。
///
/// 包含：所有算术与比较（1-27, 50-91, 92-259）、纯读取（30/32/34/35）、
/// 纯语义运算（260/261/265/274-276/278/279/287）、栈分配构造（288-289）。
/// 不包含：call/gate/await（36-49）、堆分配（29/31）、mutation（33/271/282）、
/// channel（283-285）、global_store（271）、throw（28/47）、ffi（46）。
pub fn pure_compute_fn_set() -> rustc_hash::FxHashSet<ComputeFnId> {
    let mut s = rustc_hash::FxHashSet::default();
    // ── Legacy i32/f64/bool 算术与比较（1-27）──
    for id in 1..=27u32 { s.insert(ComputeFnId(id)); }
    // ── i64/i128 算术比较 + 位运算（50-91）──
    for id in 50..=91u32 { s.insert(ComputeFnId(id)); }
    // ── 全基本类型算术（92-259：12 整数类型×12 运算 + 4 浮点类型×6 运算）──
    for id in 92..=259u32 { s.insert(ComputeFnId(id)); }
    // ── 纯读取与查询 ──
    s.insert(CF_RECORD_FIELD_GET); // record_field_get
    s.insert(CF_ARRAY_INDEX); // array_index
    s.insert(CF_IS_NULL); // is_null
    s.insert(CF_ARRAY_LEN); // array_len
    // ── 纯语义运算 ──
    s.insert(CF_REF_EQ); // ref_eq
    s.insert(CF_REF_NEQ); // ref_neq
    s.insert(CF_ELVIS); // elvis
    s.insert(CF_PATTERN_CTOR_MATCH); // pattern_ctor_match
    s.insert(CF_PATTERN_ADT_FIELD_GET); // pattern_adt_field_get
    s.insert(CF_PATTERN_STR_EQ); // pattern_str_eq
    s.insert(CF_CAST_SCALAR); // cast_scalar
    s.insert(CF_NON_NULL_ASSERT); // non_null_assert
    s.insert(CF_STR_BYTES); // str_bytes
    // ── str 比较（纯函数，可 CSE）──
    s.insert(CF_EQ_STR);
    s.insert(CF_NE_STR);
    s.insert(CF_LT_STR);
    s.insert(CF_GT_STR);
    s.insert(CF_LE_STR);
    s.insert(CF_GE_STR);
    // ── 复合类型语义相等/不等（纯函数，深度比较，可 CSE）──
    s.insert(CF_EQ_OBJ);
    s.insert(CF_NE_OBJ);
    // ── bool 不等（纯比较，与 CF_EQ_BOOL 对称）──
    s.insert(CF_NE_BOOL);
    // ── f128 比较（纯比较，专用 bit-pattern 路径）──
    s.insert(CF_EQ_F128);
    s.insert(CF_NE_F128);
    s.insert(CF_LT_F128);
    s.insert(CF_GT_F128);
    s.insert(CF_LE_F128);
    s.insert(CF_GE_F128);
    // 注意：CF_RECORD_CONSTRUCT_STACK / CF_ARRAY_CONSTRUCT_STACK 不加入 pure_set。
    // 虽然它们无外部可观察副作用，但每次执行产生独立对象（不同内存地址）。
    // 若被 LICM 外提或 CSE 消除，循环迭代会共享同一对象，导致状态污染。
    s
}

// =========================================================================
// 节点元数据宏：消除 NodeId 索引字段的声明/new/add_node/setter 四件套重复
// =========================================================================
//
// 中心定义宏 `node_metadata!($callback)` 列出所有按 NodeId 索引的元数据字段，
// 通过不同 callback 宏展开为：
//   - add_node() push（metadata_push!）   ← 自动同步
//   - setter 方法（metadata_setters!）     ← 自动同步
//   - struct 字段声明                      ← 手写（Rust 不允许宏在此位置展开）
//   - new() 初始化                         ← 手写（同上）
//
// 三种字段类别：
//   opt(field, Type, setter)   → Vec<Option<Type>>, set_setter(node, v: Type)
//   bool_flag(field, setter)   → Vec<bool>, set_setter(node) { = true }
//   bool_val(field, setter)    → Vec<bool>, set_setter(node, v: bool) { = v }
//
// 新增元数据字段：在 node_metadata! 追加一行（push+setter 自动同步），
// 再在 struct 定义和 new() 各补一行（手写）。

/// 中心定义：所有 NodeId 索引的元数据字段。
macro_rules! node_metadata {
    ($callback:ident) => {
        $callback! {
            opt(call_targets, SubGraphId, set_call_target)
            opt(gate_branches, GateBranches, set_gate_branches)
            opt(field_access_infos, u16, set_field_access_info)
            opt(record_lit_infos, RecordLitInfo, set_record_lit_info)
            opt(ffi_call_names, String, set_ffi_call_name)
            opt(field_set_names, String, set_field_set_name)
            opt(vtable_call_methods, u16, set_vtable_call)
            opt(await_event_sources, NodeId, set_await_event_source)
            opt(closure_infos, ClosureInfo, set_closure_info)
            opt(partial_infos, PartialInfo, set_partial_info)
            opt(closure_call_arg_counts, u8, set_closure_call_arg_count)
            opt(select_infos, SelectInfo, set_select_info)
            opt(writeback_targets, NodeId, set_writeback_target)
            opt(batch_infos, BatchInfo, set_batch_info)
            opt(trait_construct_infos, TraitConstructInfo, set_trait_construct_info)
            opt(lazy_construct_infos, LazyConstructInfo, set_lazy_construct_info)
            opt(record_extend_infos, RecordExtendInfo, set_record_extend_info)
            opt(global_load_slots, u32, set_global_load_slot)
            opt(global_store_slots, u32, set_global_store_slot)
            opt(pattern_ctor_names, String, set_pattern_ctor_name)
            opt(pattern_field_indices, u16, set_pattern_field_index)
            opt(cast_target_types, String, set_cast_target_type)
            opt(memo_infos, MemoInfo, set_memo_info)
            ;
            bool_flag(tail_call_flags, set_tail_call)
            bool_flag(safe_op_flags, set_safe_op)
            bool_flag(hoisted_node, set_hoisted)
            ;
            bool_val(slice_inclusive, set_slice_inclusive)
        }
    };
    ($callback:ident, $self:ident $(, $extra:ident)*) => {
        $callback! {
            $self $(, $extra)* ;
            opt(call_targets, SubGraphId, set_call_target)
            opt(gate_branches, GateBranches, set_gate_branches)
            opt(field_access_infos, u16, set_field_access_info)
            opt(record_lit_infos, RecordLitInfo, set_record_lit_info)
            opt(ffi_call_names, String, set_ffi_call_name)
            opt(field_set_names, String, set_field_set_name)
            opt(vtable_call_methods, u16, set_vtable_call)
            opt(await_event_sources, NodeId, set_await_event_source)
            opt(closure_infos, ClosureInfo, set_closure_info)
            opt(partial_infos, PartialInfo, set_partial_info)
            opt(closure_call_arg_counts, u8, set_closure_call_arg_count)
            opt(select_infos, SelectInfo, set_select_info)
            opt(writeback_targets, NodeId, set_writeback_target)
            opt(batch_infos, BatchInfo, set_batch_info)
            opt(trait_construct_infos, TraitConstructInfo, set_trait_construct_info)
            opt(lazy_construct_infos, LazyConstructInfo, set_lazy_construct_info)
            opt(record_extend_infos, RecordExtendInfo, set_record_extend_info)
            opt(global_load_slots, u32, set_global_load_slot)
            opt(global_store_slots, u32, set_global_store_slot)
            opt(pattern_ctor_names, String, set_pattern_ctor_name)
            opt(pattern_field_indices, u16, set_pattern_field_index)
            opt(cast_target_types, String, set_cast_target_type)
            opt(memo_infos, MemoInfo, set_memo_info)
            ;
            bool_flag(tail_call_flags, set_tail_call)
            bool_flag(safe_op_flags, set_safe_op)
            bool_flag(hoisted_node, set_hoisted)
            ;
            bool_val(slice_inclusive, set_slice_inclusive)
        }
    };
}

// 注：struct 字段声明与 new() 初始化器无法用 macro_rules! 宏化——
// Rust 不允许宏在 struct 字段声明位置和 struct 初始化器字段位置展开。
// 这两处必须手写（见 DataFlowGraph 定义和 new()）。新增字段时：
//   1. 在 node_metadata! 追加一行（自动同步 push + setter）
//   2. 在 struct 定义补一行字段
//   3. 在 new() 补一行 Vec::new()

/// 展开为 add_node() 中的 push 语句。
macro_rules! metadata_push {
    ( $self:ident ; $( opt($f:ident, $t:ty, $_s:ident) )* ; $( bool_flag($bf:ident, $_bs:ident) )* ; $( bool_val($vf:ident, $_vs:ident) )* ) => {
        $( $self.$f.push(None); )*
        $( $self.$bf.push(false); )*
        $( $self.$vf.push(false); )*
    };
}

/// 展开为 impl DataFlowGraph 的 setter 方法。
macro_rules! metadata_setters {
    ( $( opt($f:ident, $t:ty, $s:ident) )* ; $( bool_flag($bf:ident, $bs:ident) )* ; $( bool_val($vf:ident, $vs:ident) )* ) => {
        $(
            pub fn $s(&mut self, node: NodeId, v: $t) {
                self.$f[node.0 as usize] = Some(v);
            }
        )*
        $(
            pub fn $bs(&mut self, node: NodeId) {
                self.$bf[node.0 as usize] = true;
            }
        )*
        $(
            pub fn $vs(&mut self, node: NodeId, v: bool) {
                self.$vf[node.0 as usize] = v;
            }
        )*
    };
}

/// 展开为 clone_node_metadata() 中的逐字段克隆语句。
/// opt 字段统一用 `.clone()`（Copy 类型上的 clone 等价于拷贝），
/// bool_flag/bool_val 为 Copy 直接赋值。
macro_rules! metadata_clone {
    ( $self:ident, $src:ident, $dst:ident ;
      $( opt($f:ident, $t:ty, $_s:ident) )* ;
      $( bool_flag($bf:ident, $_bs:ident) )* ;
      $( bool_val($vf:ident, $_vs:ident) )* ) => {
        $( $self.$f[$dst] = $self.$f[$src].clone(); )*
        $( $self.$bf[$dst] = $self.$bf[$src]; )*
        $( $self.$vf[$dst] = $self.$vf[$src]; )*
    };
}

// =========================================================================
// DataFlowGraph — 全局图容器
// =========================================================================

/// 数据流图：全局只读容器，所有 worker 共享。
///
/// - `nodes`：所有节点（全局连续）
/// - `inputs_pool`：所有输入（连续存储）
/// - `subgraphs`：所有函数子图
/// - `entry_subgraph`：程序入口子图
/// - `downstreams`：每节点的下游列表（fan-out 统计，用于槽级 RC）
pub struct DataFlowGraph {
    /// 所有节点（全局连续，按 NodeId 索引）
    pub nodes: Vec<Node>,
    /// 独立输入池
    pub inputs_pool: InputsPool,
    /// 所有函数子图
    pub subgraphs: Vec<SubGraph>,
    /// 程序入口子图
    pub entry_subgraph: Option<SubGraphId>,
    /// 计算函数表（构建期填充，运行时按 ComputeFnId 索引调用）
    pub compute_fns: Vec<ComputeFn>,
    /// 每节点的下游列表（fan-out 统计，downstreams[n] = 以节点 n 为输入的下游节点列表）
    pub downstreams: Vec<Vec<NodeId>>,
    /// 常量节点的原始值（按 NodeId 索引，非 Const 节点为 None）
    pub const_values: Vec<Option<ConstValue>>,
    /// Call 节点的目标子图（按 NodeId 索引，非 Call 节点为 None）
    pub call_targets: Vec<Option<SubGraphId>>,
    /// Gate 节点的分支信息（按 NodeId 索引，非 Gate 节点为 None）
    pub gate_branches: Vec<Option<GateBranches>>,
    /// 字段访问信息（按 NodeId 索引，存 field_idx）
    pub field_access_infos: Vec<Option<u16>>,
    /// 记录构造信息（按 NodeId 索引）
    pub record_lit_infos: Vec<Option<RecordLitInfo>>,
    /// FFI 调用节点对应的 @extern("C") 函数名（用于 compute_ffi_call 分派）
    pub ffi_call_names: Vec<Option<String>>,
    /// 字段赋值信息（按 NodeId 索引，存字段名，用于 compute_record_field_set）
    pub field_set_names: Vec<Option<String>>,
    /// vtable 动态分派 Call 节点的方法 idx（按 NodeId 索引，None=静态调用）
    /// method_idx = 方法在 TraitDefInfo.methods 中的位置（与 TraitValue.method_values 索引一致）
    pub vtable_call_methods: Vec<Option<u16>>,
    /// Await 节点对应的 EventSource 声明节点（按 NodeId 索引，非 Await 节点为 None）
    pub await_event_sources: Vec<Option<NodeId>>,
    /// 闭包构造节点信息（按 NodeId 索引，非闭包构造节点为 None）
    pub closure_infos: Vec<Option<ClosureInfo>>,
    /// 偏应用构造节点信息（按 NodeId 索引，非 partial_construct 节点为 None）
    pub partial_infos: Vec<Option<PartialInfo>>,
    /// 闭包调用节点的实参数（不含闭包值和 effect，用于链式偏应用判断）
    pub closure_call_arg_counts: Vec<Option<u8>>,
    /// select 表达式分支信息（按 NodeId 索引，非 select gate 节点为 None）
    pub select_infos: Vec<Option<SelectInfo>>,
    /// WriteBack 节点的目标外层 NodeId（按 NodeId 索引，非 WriteBack 节点为 None）
    pub writeback_targets: Vec<Option<NodeId>>,
    /// Call 节点的尾调用标记（按 NodeId 索引，true=尾调用帧复用）
    pub tail_call_flags: Vec<bool>,
    /// 安全操作标记（按 NodeId 索引，true=inputs[0] 为 Null 时短路返回 Null）
    /// 用于 ?.field / ?.method() / cast(x).to(T)?
    pub safe_op_flags: Vec<bool>,
    /// 外提/展开/内联产生的节点标记（按 NodeId 索引）
    /// true = 由 pass 层追加的节点，需被所属函数子图的帧初始化
    pub hoisted_node: Vec<bool>,
    /// hoisted 节点的归属函数子图（按 NodeId 索引，仅 hoisted_node=true 时有效）
    /// rebuild 按函数级子图分组重排时，将 hoisted 节点排到 owner 子图范围内
    pub hoisted_owners: Vec<SubGraphId>,
    /// 编译期 SIMD/并行批量化标记（按 NodeId 索引，None=不可批量化）
    pub batch_infos: Vec<Option<BatchInfo>>,
    /// IR 编译期错误（未实现的特性、找不到函数等），build() 末尾从 IrBuilder.errors 移入
    pub ir_errors: Vec<String>,
    /// inline_trait 构造节点信息（按 NodeId 索引，非 trait construct 节点为 None）
    pub trait_construct_infos: Vec<Option<TraitConstructInfo>>,
    /// lazy 构造节点信息（按 NodeId 索引，非 lazy construct 节点为 None）
    pub lazy_construct_infos: Vec<Option<LazyConstructInfo>>,
    /// 记录扩展节点信息（按 NodeId 索引，非 record extend 节点为 None）
    pub record_extend_infos: Vec<Option<RecordExtendInfo>>,
    /// 切片节点的 inclusive 标志（按 NodeId 索引，true = `[start..=end]`，false = `[start..end]`）
    pub slice_inclusive: Vec<bool>,
    /// 全局变量运行时存储（顶层 var/val 声明，跨函数共享，不依赖帧链）
    pub global_var_storage: Arc<Vec<std::sync::Mutex<Option<crate::value::Value>>>>,
    /// global_load 节点的 slot index（按 NodeId 索引，非 global_load 节点为 None）
    pub global_load_slots: Vec<Option<u32>>,
    /// global_store 节点的 slot index（按 NodeId 索引，非 global_store 节点为 None）
    pub global_store_slots: Vec<Option<u32>>,
    /// 模式匹配：构造器名判别节点存储的构造器名（按 NodeId 索引）
    pub pattern_ctor_names: Vec<Option<String>>,
    /// 模式匹配：ADT 按位置提取字段节点的索引（按 NodeId 索引）
    pub pattern_field_indices: Vec<Option<u16>>,
    /// 通用 cast 节点的目标类型名（按 NodeId 索引，非 cast 节点为 None）
    pub cast_target_types: Vec<Option<String>>,
    /// memo_check / memo_store 节点的缓存元信息（按 NodeId 索引，None=非 memo 节点）
    pub memo_infos: Vec<Option<MemoInfo>>,
    /// 记忆化缓存表运行时存储（每个 memoized 函数一个 HashMap<u64, Value>）
    pub memo_tables: Arc<Vec<std::sync::Mutex<rustc_hash::FxHashMap<u64, Value>>>>,
    /// 字符串池：ConstValue::Str { offset, len } 引用此池
    /// 构建期由 IrBuilder 维护 intern，load 期从 .resin StringPool section 填充
    pub string_pool: Arc<[u8]>,
    /// GraphMemory（加载路径）：mmap 或 owned bytes 的二进制 backing。
    /// 构建路径为 None（直接访问 owned Vec 字段）；
    /// 加载路径为 Some(GraphMemory)，zerocopy 表通过 accessor 方法从此 backing 读取。
    pub mem: Option<crate::resin::Spec::GraphMemory>,
    /// SubGraph upvalue_outer_nodes 的 CSR offset 表（加载路径）。
    /// 每个元素 = (byte_offset_into_SgUpvalueNodes, count)。
    /// 构建路径为空（accessor 回退到 SubGraph.upvalue_outer_nodes Vec）。
    /// 加载路径（zerocopy）填充，SubGraph.upvalue_outer_nodes Vec 设为空。
    pub sg_uv_offsets: Vec<(u32, u32)>,
    /// SubGraph nested_ranges 的 CSR offset 表（加载路径）。
    /// 每个元素 = (byte_offset_into_SgNestedRanges, count)。
    /// 构建路径为空（accessor 回退到 SubGraph.nested_ranges Vec）。
    /// 加载路径（zerocopy）填充，SubGraph.nested_ranges Vec 设为空。
    pub sg_nr_offsets: Vec<(u32, u32)>,
    /// 5 个复杂变长表的 per-Node 字节偏移表（加载路径）。
    /// u32::MAX = None（该节点无此表数据），其他值 = section 内字节偏移。
    /// 构建路径为空（accessor 回退到 owned Vec 字段）。
    /// 加载路径（zerocopy）填充，owned Vec 字段设为空。
    pub gate_branch_offsets: Vec<u32>,
    pub record_lit_info_offsets: Vec<u32>,
    pub select_info_offsets: Vec<u32>,
    pub trait_construct_info_offsets: Vec<u32>,
    pub record_extend_info_offsets: Vec<u32>,
}

impl DataFlowGraph {
    /// 创建空图。
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs_pool: InputsPool::new(),
            subgraphs: Vec::new(),
            entry_subgraph: None,
            compute_fns: build_compute_fn_table(),
            downstreams: Vec::new(),
            const_values: Vec::new(),
            // 元数据字段初始化（Rust 不允许 struct 初始化器内展开宏，故手写）
            call_targets: Vec::new(),
            gate_branches: Vec::new(),
            field_access_infos: Vec::new(),
            record_lit_infos: Vec::new(),
            ffi_call_names: Vec::new(),
            field_set_names: Vec::new(),
            vtable_call_methods: Vec::new(),
            await_event_sources: Vec::new(),
            closure_infos: Vec::new(),
            partial_infos: Vec::new(),
            closure_call_arg_counts: Vec::new(),
            select_infos: Vec::new(),
            writeback_targets: Vec::new(),
            tail_call_flags: Vec::new(),
            safe_op_flags: Vec::new(),
            hoisted_node: Vec::new(),
            hoisted_owners: Vec::new(),
            batch_infos: Vec::new(),
            trait_construct_infos: Vec::new(),
            lazy_construct_infos: Vec::new(),
            record_extend_infos: Vec::new(),
            slice_inclusive: Vec::new(),
            global_load_slots: Vec::new(),
            global_store_slots: Vec::new(),
            pattern_ctor_names: Vec::new(),
            pattern_field_indices: Vec::new(),
            cast_target_types: Vec::new(),
            ir_errors: Vec::new(),
            global_var_storage: Arc::new(Vec::new()),
            memo_infos: Vec::new(),
            memo_tables: Arc::new(Vec::new()),
            string_pool: Arc::from(Vec::new()),
            mem: None,
            sg_uv_offsets: Vec::new(),
            sg_nr_offsets: Vec::new(),
            gate_branch_offsets: Vec::new(),
            record_lit_info_offsets: Vec::new(),
            select_info_offsets: Vec::new(),
            trait_construct_info_offsets: Vec::new(),
            record_extend_info_offsets: Vec::new(),
        }
    }

    /// 添加节点，返回其 NodeId。
    pub fn add_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.downstreams.push(Vec::new());
        self.const_values.push(None);
        // 元数据字段 push（由 node_metadata! 宏统一生成）
        node_metadata!(metadata_push, self);
        self.hoisted_owners.push(SubGraphId(u32::MAX));
        id
    }

    // ---- 节点元数据 setter（由 node_metadata! 宏统一生成）----
    node_metadata!(metadata_setters);

    /// 克隆源节点的所有元数据到目标节点（用于 pass 层节点克隆）。
    pub fn clone_node_metadata(&mut self, src_idx: usize, dst_idx: usize) {
        self.const_values[dst_idx] = self.const_values[src_idx].clone();
        node_metadata!(metadata_clone, self, src_idx, dst_idx);
        self.hoisted_owners[dst_idx] = self.hoisted_owners[src_idx];
    }

    /// 直接添加节点（不经过 Builder），用于 pass 层变换。
    /// 自动同步元数据 push（与 add_node 相同）。
    pub fn add_node_raw(
        &mut self,
        kind: NodeKind,
        inputs: &[NodeId],
        compute_fn: ComputeFnId,
    ) -> NodeId {
        let inputs_offset = self.inputs_pool.push(inputs);
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            kind,
            input_count: inputs.len() as u8,
            inputs_offset,
            compute_fn,
        });
        self.downstreams.push(Vec::new());
        self.const_values.push(None);
        node_metadata!(metadata_push, self);
        self.hoisted_owners.push(SubGraphId(u32::MAX));
        id
    }

    /// 找到包含给定节点的函数子图（最外层，loop_kind=None 且 loop_parent_sg=None）。
    /// 用于 pass 层确定新追加节点应归属的子图。
    pub fn find_function_sg_for_node(&self, node: NodeId) -> Option<SubGraphId> {
        let mut best_sg: Option<SubGraphId> = None;
        let mut best_range_size = u32::MAX;
        for (idx, sg) in self.subgraphs.iter().enumerate() {
            let (start, end) = sg.node_range;
            if node.0 >= start.0 && node.0 < end.0 {
                let size = end.0 - start.0;
                if size < best_range_size {
                    best_range_size = size;
                    best_sg = Some(SubGraphId(idx as u32));
                }
            }
        }
        // 从最内层子图向上找到函数子图
        if let Some(inner_sg_id) = best_sg {
            let mut cur = inner_sg_id;
            loop {
                let sg = &self.subgraphs[cur.0 as usize];
                if sg.loop_kind == LoopKind::None && sg.loop_parent_sg.is_none() {
                    return Some(cur);
                }
                // 找包含 cur 的更外层子图
                let (cs, ce) = sg.node_range;
                let mut parent: Option<SubGraphId> = None;
                let mut parent_size = u32::MAX;
                for (idx, psg) in self.subgraphs.iter().enumerate() {
                    if idx == cur.0 as usize {
                        continue;
                    }
                    let (ps, pe) = psg.node_range;
                    if cs.0 >= ps.0 && ce.0 <= pe.0 {
                        let size = pe.0 - ps.0;
                        if size < parent_size {
                            parent_size = size;
                            parent = Some(SubGraphId(idx as u32));
                        }
                    }
                }
                match parent {
                    Some(p) => cur = p,
                    None => return Some(cur), // 已到最外层
                }
            }
        }
        None
    }

    /// 找到包含给定节点的最内层子图。
    /// 用于 pass 层判断节点是否直接在函数级子图中（而非嵌套在 Gate 分支/循环体内）。
    pub fn find_innermost_sg_for_node(&self, node: NodeId) -> Option<SubGraphId> {
        let mut best_sg: Option<SubGraphId> = None;
        let mut best_range_size = u32::MAX;
        for (idx, sg) in self.subgraphs.iter().enumerate() {
            let (start, end) = sg.node_range;
            if node.0 >= start.0 && node.0 < end.0 {
                let size = end.0 - start.0;
                if size < best_range_size {
                    best_range_size = size;
                    best_sg = Some(SubGraphId(idx as u32));
                }
            }
        }
        best_sg
    }

    /// 找到包含给定子图的最小外层子图（immediate parent）。
    /// 用于 LICM：不变量应外提到 loop_sg 的 immediate parent，
    /// 而非总是提到 function sg（嵌套循环时 outer loop 的 body_sg 才是正确目标）。
    pub fn find_immediate_parent_sg(&self, sg_id: SubGraphId) -> Option<SubGraphId> {
        let (cs, ce) = self.subgraphs[sg_id.0 as usize].node_range;
        let mut best: Option<SubGraphId> = None;
        let mut best_size = u32::MAX;
        for (idx, psg) in self.subgraphs.iter().enumerate() {
            if idx == sg_id.0 as usize {
                continue;
            }
            let (ps, pe) = psg.node_range;
            // 必须严格包含 sg 的范围
            if ps.0 <= cs.0 && pe.0 >= ce.0 && (ps.0 < cs.0 || pe.0 > ce.0) {
                let size = pe.0 - ps.0;
                if size < best_size {
                    best_size = size;
                    best = Some(SubGraphId(idx as u32));
                }
            }
        }
        best
    }

    /// 扩展子图的 node_range 以包含新追加的节点，并递归扩展所有祖先子图。
    /// 在 pass 层添加节点后调用，确保 rebuild 时新节点被包含在子图范围内，
    /// 同时保持子图嵌套结构一致性（祖先范围必须包含后代范围）。
    pub fn extend_function_sg_range(&mut self, sg_id: SubGraphId, new_node_end: NodeId) {
        // 扩展目标子图
        let sg = &mut self.subgraphs[sg_id.0 as usize];
        if sg.node_range.1 < new_node_end {
            sg.node_range.1 = new_node_end;
        }
        // 递归扩展所有包含目标子图的祖先
        let (cs, ce) = self.subgraphs[sg_id.0 as usize].node_range;
        for (idx, psg) in self.subgraphs.iter_mut().enumerate() {
            if idx == sg_id.0 as usize {
                continue;
            }
            let (ps, pe) = psg.node_range;
            // 如果祖先包含目标子图的范围，且新节点超出祖先范围，则扩展祖先
            if ps.0 <= cs.0 && pe.0 >= ce.0 && pe.0 < new_node_end.0 {
                psg.node_range.1 = new_node_end;
            }
        }
    }

    /// 添加子图，返回其 SubGraphId。
    pub fn add_subgraph(&mut self, sg: SubGraph) -> SubGraphId {
        let id = SubGraphId(self.subgraphs.len() as u32);
        self.subgraphs.push(sg);
        id
    }

    /// 设置程序入口子图。
    pub fn set_entry_subgraph(&mut self, id: SubGraphId) {
        self.entry_subgraph = Some(id);
    }

    /// 计算节点的元数据哈希，用于 CSE 去重键。
    ///
    /// 通用方法：将所有 per-node 元数据字段哈希为 u64。
    /// 两个 (compute_fn, inputs) 相同但元数据不同的节点不会被 CSE 合并。
    /// 例如 pattern_adt_field_get 的 field_index 不同、pattern_ctor_match 的 ctor_name 不同。
    /// 对不实现 Hash 的类型使用 Debug 字符串哈希。
    pub fn cse_metadata_hash(&self, idx: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        let mut h = DefaultHasher::new();

        macro_rules! hash_opt {
            ($field:ident) => {
                match self.$field.get(idx).and_then(|o| o.as_ref()) {
                    None => 0u8.hash(&mut h),
                    Some(v) => { 1u8.hash(&mut h); format!("{:?}", v).hash(&mut h); }
                }
            };
        }

        hash_opt!(call_targets);
        hash_opt!(gate_branches);
        hash_opt!(field_access_infos);
        hash_opt!(record_lit_infos);
        hash_opt!(ffi_call_names);
        hash_opt!(field_set_names);
        hash_opt!(vtable_call_methods);
        hash_opt!(await_event_sources);
        hash_opt!(closure_infos);
        hash_opt!(partial_infos);
        hash_opt!(closure_call_arg_counts);
        hash_opt!(select_infos);
        hash_opt!(writeback_targets);
        hash_opt!(batch_infos);
        hash_opt!(trait_construct_infos);
        hash_opt!(lazy_construct_infos);
        hash_opt!(record_extend_infos);
        hash_opt!(global_load_slots);
        hash_opt!(global_store_slots);
        hash_opt!(pattern_ctor_names);
        hash_opt!(pattern_field_indices);
        hash_opt!(cast_target_types);

        self.tail_call_flags.get(idx).copied().unwrap_or(false).hash(&mut h);
        self.safe_op_flags.get(idx).copied().unwrap_or(false).hash(&mut h);
        self.slice_inclusive.get(idx).copied().unwrap_or(false).hash(&mut h);

        h.finish()
    }

    /// 计算所有节点的下游列表（fan-out 统计）。
    ///
    /// 遍历每个节点的输入，把该节点注册到各输入节点的 downstreams 列表。
    /// 用于槽级 RC：节点产出时 refcount = downstreams[n].len()。
    pub fn compute_downstreams(&mut self) {
        // 先清空
        for ds in &mut self.downstreams {
            ds.clear();
        }
        // 遍历节点，注册下游关系（inputs_pool 中的输入边）
        for nid in 0..self.nodes.len() {
            let node = self.nodes[nid];
            let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
            for &input in inputs {
                self.downstreams[input.0 as usize].push(NodeId(nid as u32));
            }
        }
        // Gate 节点的 condition_input → Gate 边（Gate 就绪依赖条件值被计算）
        // 避免重复：如果 condition_input 已在 Gate 的 inputs 中（input_count=1 的 if/while/for/match Gate），
        // 第一个循环已注册下游关系，跳过。仅对 condition_input 不在 inputs 中的 Gate（如 select gate）注册。
        for nid in 0..self.nodes.len() {
            if let Some(gb) = &self.gate_branches[nid] {
                let node = self.nodes[nid];
                let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
                if !inputs.contains(&gb.condition_input) {
                    self.downstreams[gb.condition_input.0 as usize].push(NodeId(nid as u32));
                }
            }
        }
    }

    /// 晚期压缩重建：根据 dead 集与 redirect 映射重建图。
    /// 压缩 nodes/inputs_pool，重映射所有 NodeId 引用与 per-NodeId 元数据向量。
    /// 重建后所有 NodeId 引用更新为新连续编号。
    /// 返回 old_to_new 映射（供外部同步 expr_node_map 等）。
    pub fn rebuild(
        &mut self,
        dead: &rustc_hash::FxHashSet<NodeId>,
        redirect: &rustc_hash::FxHashMap<NodeId, NodeId>,
    ) -> Vec<Option<NodeId>> {
        // ── 递归解析重定向 ──
        let resolve = |id: NodeId| -> NodeId {
            let mut cur = id;
            while let Some(&next) = redirect.get(&cur) { cur = next; }
            cur
        };

        // ── 1. 按函数级子图分组排列存活节点 ──
        // pass 层（LICM/inline）追加的 hoisted 节点在 graph.nodes 末尾，不在 caller
        // 子图的 node_range 内。如果按 0..total 顺序压缩，hoisted 节点的 new_id 在末尾，
        // 不在 caller 的 node_range 内 → caller 帧不执行它们 → 变换无效。
        //
        // 改为按函数级子图分组排列：每个函数级子图的原生存活节点 + 属于它的 hoisted
        // 存活节点，使 hoisted 节点的 new_id 紧跟在 caller 原生节点后面，保证连续。
        let total = self.nodes.len();
        let mut old_to_new: Vec<Option<NodeId>> = vec![None; total];
        let mut new_to_old: Vec<usize> = Vec::with_capacity(total);
        let mut new_nodes: Vec<Node> = Vec::with_capacity(total);

        // 1a. 计算每个节点的归属函数级子图（保存副本供步骤 5 使用，避免被步骤 3b 压缩破坏）
        let mut node_owner: Vec<u32> = vec![u32::MAX; total];
        for sg in &self.subgraphs {
            if sg.loop_kind != LoopKind::None || sg.loop_parent_sg.is_some() {
                continue;
            }
            let start = sg.node_range.0.0 as usize;
            let end = (sg.node_range.1.0 as usize).min(total);
            for idx in start..end {
                node_owner[idx] = sg.id.0;
            }
        }
        // hoisted 节点的归属（不在任何 node_range 内，通过 hoisted_owners 确定）
        for idx in 0..total {
            if self.hoisted_node[idx] && node_owner[idx] == u32::MAX {
                node_owner[idx] = self.hoisted_owners[idx].0;
            }
        }
        // 保存旧索引的 node_owner 副本（步骤 5 在步骤 3b 压缩后仍需按旧索引访问）
        let node_owner_old = node_owner.clone();

        // 1b. 收集函数级子图列表（按 node_range.0 排序，保持原有顺序）
        let mut func_sgs: Vec<u32> = self
            .subgraphs
            .iter()
            .filter(|sg| sg.loop_kind == LoopKind::None && sg.loop_parent_sg.is_none())
            .map(|sg| sg.id.0)
            .collect();
        func_sgs.sort_by_key(|&sg_id| self.subgraphs[sg_id as usize].node_range.0);

        // 1c. 按函数级子图顺序分配 new_id
        for &sg_id in &func_sgs {
            let sg = &self.subgraphs[sg_id as usize];
            let start = sg.node_range.0.0 as usize;
            let end = (sg.node_range.1.0 as usize).min(total);

            // 原生存活节点（含嵌套子图节点，跳过 hoisted）
            for old_idx in start..end {
                if self.hoisted_node[old_idx] {
                    continue;
                }
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                // 防重复：如果节点已在其他子图循环中被分配（node_range 重叠），跳过
                if old_to_new[old_idx].is_some() {
                    continue;
                }
                let new_id = NodeId(new_nodes.len() as u32);
                old_to_new[old_idx] = Some(new_id);
                new_to_old.push(old_idx);
                new_nodes.push(self.nodes[old_idx]);
            }

            // hoisted 存活节点（owner == sg_id）
            for old_idx in 0..total {
                if !self.hoisted_node[old_idx] {
                    continue;
                }
                if self.hoisted_owners[old_idx].0 != sg_id {
                    continue;
                }
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                if old_to_new[old_idx].is_some() {
                    continue;
                }
                let new_id = NodeId(new_nodes.len() as u32);
                old_to_new[old_idx] = Some(new_id);
                new_to_old.push(old_idx);
                new_nodes.push(self.nodes[old_idx]);
            }
        }

        // 1d. 未归属的存活节点（不应存在，安全起见排到最后）
        for old_idx in 0..total {
            if old_to_new[old_idx].is_some() {
                continue;
            }
            let old_id = NodeId(old_idx as u32);
            if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                continue;
            }
            let new_id = NodeId(new_nodes.len() as u32);
            old_to_new[old_idx] = Some(new_id);
            new_to_old.push(old_idx);
            new_nodes.push(self.nodes[old_idx]);
        }

        // ── 2. 重建 inputs_pool（resolve + remap）──
        let mut new_inputs: Vec<NodeId> = Vec::new();
        for node in &mut new_nodes {
            let old_inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
            let new_offset = new_inputs.len() as u32;
            for &old_in in old_inputs {
                let resolved = resolve(old_in);
                let new_in = old_to_new[resolved.0 as usize]
                    .expect("rebuild: input node not live");
                new_inputs.push(new_in);
            }
            node.inputs_offset = new_offset;
        }

        self.nodes = new_nodes;
        self.inputs_pool.data = new_inputs;

        // ── 3. 压缩 per-NodeId 元数据向量 ──
        // 用 new_to_old 映射按新编号顺序收集元数据
        let remap_n = |id: NodeId| -> NodeId {
            let r = resolve(id);
            old_to_new[r.0 as usize].expect("rebuild: ref node not live")
        };

        // 3a. 压缩 Vec<Option<T: Clone>>（无内部 NodeId）
        macro_rules! compress_opt {
            ($field:ident) => {{
                let mut v: Vec<_> = Vec::with_capacity(new_to_old.len());
                for &old_idx in &new_to_old {
                    v.push(self.$field[old_idx].clone());
                }
                self.$field = v;
            }};
        }
        compress_opt!(const_values);
        compress_opt!(call_targets);
        compress_opt!(field_access_infos);
        compress_opt!(record_lit_infos);
        compress_opt!(ffi_call_names);
        compress_opt!(field_set_names);
        compress_opt!(vtable_call_methods);
        compress_opt!(closure_infos);
        compress_opt!(partial_infos);
        compress_opt!(closure_call_arg_counts);
        compress_opt!(batch_infos);
        compress_opt!(trait_construct_infos);
        compress_opt!(lazy_construct_infos);
        compress_opt!(record_extend_infos);
        compress_opt!(global_load_slots);
        compress_opt!(global_store_slots);
        compress_opt!(pattern_ctor_names);
        compress_opt!(pattern_field_indices);
        compress_opt!(cast_target_types);
        compress_opt!(memo_infos);

        // 3b. 压缩 Vec<bool>
        macro_rules! compress_bool {
            ($field:ident) => {{
                let mut v: Vec<_> = Vec::with_capacity(new_to_old.len());
                for &old_idx in &new_to_old {
                    v.push(self.$field[old_idx]);
                }
                self.$field = v;
            }};
        }
        compress_bool!(tail_call_flags);
        compress_bool!(safe_op_flags);
        compress_bool!(hoisted_node);
        compress_bool!(slice_inclusive);

        // 3b2. 压缩 hoisted_owners: Vec<SubGraphId>
        {
            let mut v: Vec<SubGraphId> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.hoisted_owners[old_idx]);
            }
            self.hoisted_owners = v;
        }

        // 3c. 压缩含 NodeId 的向量
        // await_event_sources: Vec<Option<NodeId>>
        {
            let mut v: Vec<Option<NodeId>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.await_event_sources[old_idx].map(&remap_n));
            }
            self.await_event_sources = v;
        }
        // writeback_targets: Vec<Option<NodeId>>
        {
            let mut v: Vec<Option<NodeId>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                v.push(self.writeback_targets[old_idx].map(&remap_n));
            }
            self.writeback_targets = v;
        }
        // gate_branches: Vec<Option<GateBranches>> — 内部 NodeId 需 remap
        {
            let mut v: Vec<Option<GateBranches>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                let opt = &self.gate_branches[old_idx];
                v.push(opt.as_ref().map(|gb| GateBranches {
                    condition_input: remap_n(gb.condition_input),
                    branches: gb.branches.iter().map(|(b, sg, params)| {
                        (*b, *sg, params.iter().map(|&n| remap_n(n)).collect())
                    }).collect(),
                }));
            }
            self.gate_branches = v;
        }
        // select_infos: Vec<Option<SelectInfo>> — SelectBranch.event_source_node 需 remap
        {
            let mut v: Vec<Option<SelectInfo>> = Vec::with_capacity(new_to_old.len());
            for &old_idx in &new_to_old {
                let opt = &self.select_infos[old_idx];
                v.push(opt.as_ref().map(|si| SelectInfo {
                    branches: si.branches.iter().map(|sb| SelectBranch {
                        subgraph_id: sb.subgraph_id,
                        event_kind: sb.event_kind,
                        event_source_node: remap_n(sb.event_source_node),
                    }).collect(),
                }));
            }
            self.select_infos = v;
        }

        // ── 4. 重建 downstreams（含 Gate condition_input 边）──
        let n = self.nodes.len();
        self.downstreams = vec![Vec::new(); n];
        for node_idx in 0..n {
            let node = self.nodes[node_idx];
            let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
            for &input in inputs {
                self.downstreams[input.0 as usize].push(NodeId(node_idx as u32));
            }
        }
        // Gate condition_input → Gate 边（与 compute_downstreams 对齐）
        for nid in 0..n {
            if let Some(gb) = &self.gate_branches[nid] {
                let node = self.nodes[nid];
                let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
                if !inputs.contains(&gb.condition_input) {
                    self.downstreams[gb.condition_input.0 as usize].push(NodeId(nid as u32));
                }
            }
        }

        // ── 5. 重映射 subgraphs 内的 NodeId 引用 ──
        // node_range 通过扫描旧范围内的存活节点 + 属于该子图的 hoisted 节点重新计算。
        // 步骤 1 已按函数级子图分组排列，hoisted 节点紧跟在原生节点后面，
        // 因此新 node_range 自然连续（原生存活节点 new_id + hoisted 存活节点 new_id）。
        for sg in self.subgraphs.iter_mut() {
            let old_start = sg.node_range.0.0 as usize;
            let old_end = (sg.node_range.1.0 as usize).min(total);
            let sg_id = sg.id;
            let mut new_start: Option<u32> = None;
            let mut new_end: u32 = 0;

            // 辅助：更新 new_start/new_end
            let mut update_range = |nid: NodeId| {
                if new_start.is_none() {
                    new_start = Some(nid.0);
                }
                if nid.0 + 1 > new_end {
                    new_end = nid.0 + 1;
                }
            };

            // 原生范围内的存活节点
            for old_idx in old_start..old_end {
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                update_range(old_to_new[old_idx].unwrap());
            }

            // 属于该子图的 hoisted 存活节点（使用 node_owner_old，因为 self.hoisted_owners
            // 已在步骤 3b2 被压缩为新数组，不能再用旧索引访问）
            for old_idx in 0..total {
                if node_owner_old[old_idx] != sg_id.0 {
                    continue;
                }
                // 跳过原生范围内的节点（已在上面处理）
                if old_idx >= old_start && old_idx < old_end {
                    continue;
                }
                let old_id = NodeId(old_idx as u32);
                if dead.contains(&old_id) || redirect.contains_key(&old_id) {
                    continue;
                }
                update_range(old_to_new[old_idx].unwrap());
            }

            sg.node_range = match new_start {
                Some(ns) => (NodeId(ns), NodeId(new_end)),
                None => (NodeId(0), NodeId(0)), // 全部 dead，范围坍缩
            };
            sg.entry_node = remap_n(sg.entry_node);
            sg.return_node = remap_n(sg.return_node);
            if let Some(c) = sg.cond_node { sg.cond_node = Some(remap_n(c)); }
            if let Some(n) = sg.iter_next_node { sg.iter_next_node = Some(remap_n(n)); }
            // event_source_decls: EventSourceDecl.node
            for decl in &mut sg.event_source_decls {
                decl.node = remap_n(decl.node);
            }
            // defer_table: DeferEntry.trigger_node + captured_inputs
            for entry in &mut sg.defer_table {
                entry.trigger_node = remap_n(entry.trigger_node);
                entry.captured_inputs = entry.captured_inputs.iter().map(|&n| remap_n(n)).collect();
            }
            // upvalue_outer_nodes: 捕获变量外层节点需重映射
            sg.upvalue_outer_nodes = sg.upvalue_outer_nodes.iter().map(|&n| remap_n(n)).collect();
            // nested_ranges: 子图 node_range 需重映射
            sg.nested_ranges = sg.nested_ranges.iter().map(|&(s, e)| {
                (remap_n(NodeId(s)).0, remap_n(NodeId(e)).0)
            }).collect();
            // reset_plan: ResetPlan 中的 NodeId 需重映射（与 cond_node/iter_next_node 同步）
            if let Some(ref mut plan) = sg.reset_plan {
                plan.reset_to_zero = plan.reset_to_zero.iter().map(|&n| remap_n(n)).collect();
                plan.reset_to_one = plan.reset_to_one.iter().map(|&n| remap_n(n)).collect();
                plan.reset_condition_tree = plan.reset_condition_tree.iter().map(|&n| remap_n(n)).collect();
            }
        }

        // 验证：检查是否有悬空引用
        if std::env::var("KUZO_VERIFY_GRAPH").is_ok() {
            let total = self.nodes.len();
            for (idx, node) in self.nodes.iter().enumerate() {
                let inputs = self.inputs_pool.get(node.inputs_offset, node.input_count);
                for &inp in inputs {
                    if inp.0 as usize >= total {
                        eprintln!("[VERIFY] node={} has dangling input {} (total={})", idx, inp.0, total);
                    }
                }
            }
            for (sg_idx, sg) in self.subgraphs.iter().enumerate() {
                if sg.entry_node.0 as usize >= total {
                    eprintln!("[VERIFY] sg={} has dangling entry_node {} (total={})", sg_idx, sg.entry_node.0, total);
                }
                if sg.return_node.0 as usize >= total {
                    eprintln!("[VERIFY] sg={} has dangling return_node {} (total={})", sg_idx, sg.return_node.0, total);
                }
                let (s, e) = sg.node_range;
                if e.0 < s.0 || e.0 as usize > total {
                    eprintln!("[VERIFY] sg={} has invalid node_range [{},{}) (total={})", sg_idx, s.0, e.0, total);
                }
            }
        }

        old_to_new
    }

    /// 计算所有子图的 `nested_ranges`：对每个子图，收集直接嵌套在其
    /// `node_range` 内的子图范围。构建期调用一次，运行时 O(len) 查询替代全图扫描。
    pub fn compute_nested_ranges(&mut self) {
        let subgraph_count = self.subgraphs.len();
        // 预收集所有子图范围，避免在循环中反复借用 self.subgraphs
        let ranges: Vec<(SubGraphId, u32, u32)> = self.subgraphs.iter()
            .map(|sg| (sg.id, sg.node_range.0 .0, sg.node_range.1 .0))
            .collect();
        for sg in &mut self.subgraphs {
            let (sg_id, branch_start, branch_end) = (sg.id, sg.node_range.0 .0, sg.node_range.1 .0);
            sg.nested_ranges = ranges.iter()
                .filter(|(id, s, e)| {
                    *id != sg_id && *s >= branch_start && *e <= branch_end
                })
                .map(|(_, s, e)| (*s, *e))
                .collect();
        }
        // 避免 unused 警告
        let _ = subgraph_count;
    }
}

impl Default for DataFlowGraph {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// expr_id_to_key — ExprId → expr_types 的 key 转换
// =========================================================================

/// ExprId → expr_types 的 key（u64）。
///
/// Sema 的 expr_types key 是 "AST Expr 句柄地址"，
/// 经研究 key = ExprId.0 as u64（AstArena.exprs 下标）。
#[inline]
pub fn expr_id_to_key(id: crate::ast::Ast::ExprId) -> u64 {
    id.0 as u64
}

// =========================================================================
// LoopContext — 循环上下文（continue 跳转目标 + For 循环迭代器节点）
// =========================================================================

/// 循环上下文：压入 loop_stack 供 continue/break 语义使用。
///
/// - `sg`：递归子图 id（continue 跳转目标）
/// - `iter_node`：For 循环 body_sg 中的迭代器参数节点（continue 需传递给尾递归；
///   None = While/Loop，param_count=0 无需传参）
/// body_node_start: 循环体子图的起始节点 ID，用于判断捕获变量是否定义在循环体内。
///   循环体内定义的变量在循环体帧销毁后不可访问，捕获此类变量的闭包必须走 Cell 路径。
#[derive(Debug, Clone, Copy)]
pub struct LoopContext {
    pub sg: SubGraphId,
    pub iter_node: Option<NodeId>,
    pub body_node_start: u32,
}

// =========================================================================
// IrBuilder — 从 SemaResult + Module 构建 DataFlowGraph
// =========================================================================

/// IR 构建器：遍历 AST，生成 Node + InputsPool + SubGraph。
///
/// 以函数为单位编译子图：
/// 1. 注册所有函数为 SubGraph

// =========================================================================
// SCALAR_META — 标量类型算术元信息（以 ValueTag 为键，name 从 Value.rs 单点派生）
// =========================================================================
//
// 集中存储每个标量类型的：
//   - arith_base:  算术 compute_fn 基址（与 compute_fn_table! 索引一致）
//   - family:      比较运算分派族（"i32"/"i64"/"i128"/"float"/"bool"）
//   - is_float:    是否浮点（决定位运算可用性、neg 偏移量）
//
// 类型名 ↔ ValueTag 的映射由 `Value::ValueTag::from_name`/`type_name` 单点维护，
// 本表不再重复 name 字段。arith_base 必须与 compute_fn_table! 中的索引严格一致。

/// 标量类型算术元信息。
///
/// `family` 为 `TypeFamily` 枚举（统一类型族，调用方用 `|` 合并整数变体按位宽分派）。
pub struct ScalarMeta {
    pub arith_base: u32,
    pub family: crate::types::TypeFamily,
    pub is_float: bool,
}

/// 按 ValueTag 查询算术元信息（const fn，编译期可求值）。
///
/// `family` 派生自 `ValueTag::family()`（单点维护，不再手写 18 个分支）。
pub const fn scalar_meta(tag: crate::value::ValueTag) -> Option<ScalarMeta> {
    use crate::value::ValueTag;
    // family 由 ValueTag::family() 派生（保持单一真相源）
    let family = tag.family();
    Some(match tag {
        // 整数 12 类型（arith_base 从 92 开始，每 12 个索引）
        ValueTag::I8    => ScalarMeta { arith_base: 92,  family, is_float: false },
        ValueTag::I16   => ScalarMeta { arith_base: 104, family, is_float: false },
        ValueTag::I32   => ScalarMeta { arith_base: 116, family, is_float: false },
        ValueTag::I64   => ScalarMeta { arith_base: 128, family, is_float: false },
        ValueTag::I128  => ScalarMeta { arith_base: 140, family, is_float: false },
        ValueTag::U8    => ScalarMeta { arith_base: 152, family, is_float: false },
        ValueTag::U16   => ScalarMeta { arith_base: 164, family, is_float: false },
        ValueTag::U32   => ScalarMeta { arith_base: 176, family, is_float: false },
        ValueTag::U64   => ScalarMeta { arith_base: 188, family, is_float: false },
        ValueTag::U128  => ScalarMeta { arith_base: 200, family, is_float: false },
        ValueTag::Isize => ScalarMeta { arith_base: 212, family, is_float: false },
        ValueTag::Usize => ScalarMeta { arith_base: 224, family, is_float: false },
        // 浮点 4 类型（arith_base 从 236 开始，每 6 个索引）
        ValueTag::F16   => ScalarMeta { arith_base: 236, family, is_float: true },
        ValueTag::F32   => ScalarMeta { arith_base: 242, family, is_float: true },
        ValueTag::F64   => ScalarMeta { arith_base: 248, family, is_float: true },
        ValueTag::F128  => ScalarMeta { arith_base: 254, family, is_float: true },
        // 非算术标量类型（bool/char，无 arith_base）
        ValueTag::Bool  => ScalarMeta { arith_base: 0,   family, is_float: false },
        ValueTag::Char  => ScalarMeta { arith_base: 0,   family, is_float: false },
        // 非标量 tag：无算术元信息
        _ => return None,
    })
}
