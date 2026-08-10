//! Compute.rs — compute_fn 计算函数表模块
//!
//! 从 Engine.rs 拆分而来，集中存放所有 compute_fn（构建期绑定的节点计算函数），
//! 包括：
//! - 哨兵常量（THUNK_FRAME_ID / IO / CTOR / TYPE_NAME 等）
//! - reflect 辅助函数 + utf8 解码工具
//! - compute_fn 生成宏（read_node_inputs / impl_cmp_compute / impl_int_ops / impl_float_ops）
//! - 全部 compute_fn（算术 / 比较 / 记录 / 数组 / 字符串 / channel / async / 闭包 等）
//! - 同步执行辅助：force_lazy_value_sync / run_frame_sync / run_defers_sync / unwrap_cell
//!
//! 调度器（Engine.rs）通过 graph.compute_fns[idx] 间接调用这些函数，
//! ir/Ir.rs 的 build_compute_fn_table 通过 super::Compute:: 引用。

use super::Ir::*;
use crate::value::Value;
use crate::engine::{prepare_frame_nodes, switch_subgraph, notify_downstream};
use std::sync::OnceLock;

/// 缓存环境变量布尔标志，避免热路径每次调用 std::env::var（getenv 系统调用 + String 分配）。
/// 首次调用读取 env，后续直接返回缓存的 bool。
#[inline]
fn env_flag(name: &str) -> bool {
    static FLAG_CALL: OnceLock<bool> = OnceLock::new();
    static FLAG_GATE: OnceLock<bool> = OnceLock::new();
    static FLAG_STALL: OnceLock<bool> = OnceLock::new();
    static FLAG_WB: OnceLock<bool> = OnceLock::new();
    match name {
        "KUZO_DEBUG_CALL" => *FLAG_CALL.get_or_init(|| std::env::var("KUZO_DEBUG_CALL").is_ok()),
        "KUZO_DEBUG_GATE" => *FLAG_GATE.get_or_init(|| std::env::var("KUZO_DEBUG_GATE").is_ok()),
        "KUZO_DEBUG_STALL" => *FLAG_STALL.get_or_init(|| std::env::var("KUZO_DEBUG_STALL").is_ok()),
        "KUZO_DEBUG_WB" => *FLAG_WB.get_or_init(|| std::env::var("KUZO_DEBUG_WB").is_ok()),
        _ => std::env::var(name).is_ok(),
    }
}

// =========================================================================
// 哨兵常量 — 集中定义，避免散落魔数
// =========================================================================

/// Thunk 帧使用的哨兵 FrameId（不参与正常分配，避免与 alloc_frame_id 冲突）。
const THUNK_FRAME_ID: FrameId = FrameId(u32::MAX);
/// LoopBody 回退子帧使用的哨兵 FrameId（不参与正常分配）。
const LOOPBODY_FALLBACK_FRAME_ID: FrameId = FrameId(u32::MAX - 1);

/// IO 写入成功返回值（i32）。仅在 `#[cfg(not(has_extern_c))]` 的 fallback 路径使用。
#[cfg(not(has_extern_c))]
const IO_OK: i32 = 0;
/// IO 写入失败返回值（i32）。仅在 `#[cfg(not(has_extern_c))]` 的 fallback 路径使用。
#[cfg(not(has_extern_c))]
const IO_ERR: i32 = -1;
/// UTF-8 解码失败/越界返回值（i64）。
const UTF8_DECODE_ERR: i64 = -1;

/// Result 变体构造器名（与 stdlib 的 Result 类型定义保持同步）。
pub(crate) const CTOR_OK: &str = "Ok";
pub(crate) const CTOR_ERR: &str = "Error";
pub(crate) const CTOR_ERR_ALT: &str = "Err";

/// reflect 类型名常量（单点维护，供 __reflect_type_name / compute_cast_to_str 共用）。
const TYPE_NAME_NULL: &str = "null";
const TYPE_NAME_VOID: &str = "void";
const TYPE_NAME_STR: &str = "str";
const TYPE_NAME_ARRAY: &str = "array";
const TYPE_NAME_UNKNOWN: &str = "unknown";

// =========================================================================
// 运行时错误构造 — 统一使用 ErrorVal（与 Arena::alloc_error_val 同构）
// =========================================================================

/// 构造运行时错误值：用 ErrorValue（专用错误类型）包装在 ThrowVal::Err 中。
///
/// 与 `ValueArena::alloc_error_val` 使用相同的 `HeapObj::ErrorVal` 表示，
/// 消除各 compute_fn 中手构造 RecordValue 的重复模式。
/// compute_fn 无 Arena 访问权，直接构造 `Value::ref_val`。
fn make_error_throw(type_name: &str, msg: &str) -> Value {
    use crate::value::{HeapObj, ErrorValue, ThrowValue, ThrowPayload};
    let err_val = Value::ref_val(HeapObj::ErrorVal(ErrorValue {
        type_name: type_name.to_string(),
        message: msg.to_string(),
        is_error_subtype: true,
    }));
    Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Err(err_val) }))
}

// =========================================================================
// reflect 辅助函数 — 消除 FFI/fallback 双路径重复
// =========================================================================

/// 返回 Value 的 reflect kind 编号（ABI 协议：0-12）。
/// 单一权威来源，FFI 与 fallback 路径共用，确保一致。
fn reflect_kind(v: &Value) -> u8 {
    match v {
        Value::Null => 0,
        Value::Void => 1,
        Value::Scalar(_, _) => 2,
        Value::Ref(r) => match &**r {
            crate::value::HeapObj::Str(_) => 3,
            crate::value::HeapObj::Array(_) => 4,
            crate::value::HeapObj::Record(_) => 5,
            crate::value::HeapObj::Adt(_) => 6,
            crate::value::HeapObj::Closure(_) => 7,
            crate::value::HeapObj::TraitVal(_) => 8,
            crate::value::HeapObj::ThrowVal(_) => 9,
            crate::value::HeapObj::ChannelVal(_) => 10,
            crate::value::HeapObj::AsyncVal(_) => 11,
            _ => 12,
        },
    }
}

/// 返回 Value 的 reflect kind 显示名（单点维护，FFI 与 fallback 共用）。
fn reflect_kind_str(v: &Value) -> &'static str {
    match v {
        Value::Null => "Null",
        Value::Void => "Void",
        Value::Scalar(_, _) => "Primitive",
        Value::Ref(r) => match &**r {
            crate::value::HeapObj::Str(_) => "Str",
            crate::value::HeapObj::Array(_) => "Array",
            crate::value::HeapObj::Record(_) => "Record",
            crate::value::HeapObj::Adt(_) => "Adt",
            crate::value::HeapObj::Newtype(_) => "Newtype",
            crate::value::HeapObj::Closure(_) => "Closure",
            crate::value::HeapObj::TraitVal(_) => "Trait",
            _ => "Ref",
        },
    }
}

/// 返回 Value 的类型名（单点维护，FFI 与 fallback / cast_to_str 共用）。
fn reflect_type_name(v: &Value) -> String {
    match v {
        Value::Null => TYPE_NAME_NULL.to_string(),
        Value::Void => TYPE_NAME_VOID.to_string(),
        Value::Scalar(_, tag) => tag.type_name().to_string(),
        Value::Ref(r) => match &**r {
            crate::value::HeapObj::Str(_) => TYPE_NAME_STR.to_string(),
            crate::value::HeapObj::Array(_) => TYPE_NAME_ARRAY.to_string(),
            crate::value::HeapObj::Record(rec) => rec.type_name.clone(),
            crate::value::HeapObj::Adt(a) => a.type_name.clone(),
            crate::value::HeapObj::Newtype(n) => n.type_name.clone(),
            _ => TYPE_NAME_UNKNOWN.to_string(),
        },
    }
}

/// UTF-8 解码：从 bytes[offset] 起解码一个 codepoint。
/// 成功返回 (codepoint, consumed_bytes)，失败（越界/非法首字节）返回 None。
/// 单一实现，消除 FFI/fallback 双路径重复。
fn utf8_decode_at(bytes: &[u8], offset: usize) -> Option<(u32, usize)> {
    if offset >= bytes.len() {
        return None;
    }
    let c = bytes[offset];
    // ASCII（1 字节）
    if c < 0x80 {
        return Some((c as u32, 1));
    }
    // 2 字节序列：110xxxxx 10xxxxxx
    if (c & 0xE0) == 0xC0 {
        if offset + 1 >= bytes.len() {
            return None;
        }
        let cp = ((c as u32 & 0x1F) << 6) | (bytes[offset + 1] as u32 & 0x3F);
        return Some((cp, 2));
    }
    // 3 字节序列：1110xxxx 10xxxxxx 10xxxxxx
    if (c & 0xF0) == 0xE0 {
        if offset + 2 >= bytes.len() {
            return None;
        }
        let cp = ((c as u32 & 0x0F) << 12)
            | ((bytes[offset + 1] as u32 & 0x3F) << 6)
            | (bytes[offset + 2] as u32 & 0x3F);
        return Some((cp, 3));
    }
    // 4 字节序列：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    if (c & 0xF8) == 0xF0 {
        if offset + 3 >= bytes.len() {
            return None;
        }
        let cp = ((c as u32 & 0x07) << 18)
            | ((bytes[offset + 1] as u32 & 0x3F) << 12)
            | ((bytes[offset + 2] as u32 & 0x3F) << 6)
            | (bytes[offset + 3] as u32 & 0x3F);
        return Some((cp, 4));
    }
    // 非法首字节
    None
}

/// 将 u32 codepoint 转为 char，非法 codepoint 回退为 U+0000。
/// 单点统一，消除 3 处重复的 `char::from_u32(x).unwrap_or('\0')`。
#[inline]
pub fn char_from_u32_or_nul(u: u32) -> char {
    char::from_u32(u).unwrap_or('\0')
}

// =========================================================================
// compute_fn 生成宏 — 批量生成类型特化计算函数
// =========================================================================

/// 读取节点输入的样板宏。
///
/// 每个 compute_fn 开头都需要从 frame.graph 取出 node 和 inputs 切片，
/// 这 3 行代码在 100+ 个 compute_fn 中完全重复。本宏消除该重复。
///
/// 用法（在 compute_fn 函数体内）：
/// ```ignore
/// pub fn compute_foo(frame: &mut Frame, node: NodeId) -> Value {
///     read_node_inputs!(frame, node, graph, n, inputs);
///     let a = frame.get_value_by_global(inputs[0]).as_i32();
///     ...
/// }
/// ```
/// 展开后 `graph`、`n`、`inputs` 三个绑定在当前作用域可用。
/// `inputs` 的生命周期绑定到 `graph`（frame.graph 的 Arc clone）。
macro_rules! read_node_inputs {
    ($frame:ident, $node:ident, $graph:ident, $n:ident, $inputs:ident) => {
        let $graph = $frame.graph.clone();
        let $n = $graph.node($node.0 as usize);
        let $inputs = $graph.inputs($n.inputs_offset, $n.input_count);
    };
}

/// 批量生成比较 compute_fn（返回 bool）。
macro_rules! impl_cmp_compute {
    ($($name:ident: $op:tt for $acc:ident);* $(;)?) => {
        $(
            pub fn $name(frame: &mut Frame, node: NodeId) -> Value {
                read_node_inputs!(frame, node, graph, n, inputs);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::bool_val(a $op b)
            }
        )*
    };
}

// =========================================================================
// SIMD 批处理 — compute_fn 内部批算（通过 EvalContext 自主决策）
// =========================================================================

/// 批量提取二元运算输入 → SIMD/rayon 批算 → 返回 (local NodeId, Value) 列表。
/// 不写 frame.value_table、不通知下游——由 engine 热循环通过 NodeResult::Batch 处理。
macro_rules! compute_bin_batch_results {
    ($frame:expr, $graph:expr, $locals:expr, $ns:expr, $rust:ty, $ctor:ident, $acc:ident, $batch_fn:ident, $op:expr) => {{
        let n = $locals.len();
        let mut a: Vec<$rust> = Vec::with_capacity(n);
        let mut b: Vec<$rust> = Vec::with_capacity(n);
        for &lid in $locals.iter() {
            let gid = NodeId(lid.0 + $ns);
            let node = $graph.node(gid.0 as usize);
            let inp = $graph.inputs(node.inputs_offset, node.input_count);
            a.push($frame.get_value_by_global(inp[0]).$acc());
            b.push($frame.get_value_by_global(inp[1]).$acc());
        }
        let mut dst = vec![0 as $rust; n];
        crate::value::$batch_fn(&mut dst, &a, &b, $op);
        $locals.iter().zip(dst.iter())
            .map(|(&lid, &v)| (lid, Value::$ctor(v)))
            .collect::<Vec<_>>()
    }};
}

/// 批量提取比较运算输入 → SIMD/rayon 批算 → 返回 (local NodeId, bool Value) 列表。
macro_rules! compute_cmp_batch_results {
    ($frame:expr, $graph:expr, $locals:expr, $ns:expr, $rust:ty, $acc:ident, $batch_fn:ident, $op:expr) => {{
        let n = $locals.len();
        let mut a: Vec<$rust> = Vec::with_capacity(n);
        let mut b: Vec<$rust> = Vec::with_capacity(n);
        for &lid in $locals.iter() {
            let gid = NodeId(lid.0 + $ns);
            let node = $graph.node(gid.0 as usize);
            let inp = $graph.inputs(node.inputs_offset, node.input_count);
            a.push($frame.get_value_by_global(inp[0]).$acc());
            b.push($frame.get_value_by_global(inp[1]).$acc());
        }
        let mut mask = vec![0u8; n];
        crate::value::$batch_fn(&mut mask, &a, &b, $op);
        $locals.iter().zip(mask.iter())
            .map(|(&lid, &m)| (lid, Value::bool_val(m != 0)))
            .collect::<Vec<_>>()
    }};
}

/// 批量提取一元运算输入 → SIMD/rayon 批算 → 返回 (local NodeId, Value) 列表。
macro_rules! compute_unary_batch_results {
    ($frame:expr, $graph:expr, $locals:expr, $ns:expr, $rust:ty, $ctor:ident, $acc:ident, $op:expr) => {{
        let n = $locals.len();
        let mut a: Vec<$rust> = Vec::with_capacity(n);
        for &lid in $locals.iter() {
            let gid = NodeId(lid.0 + $ns);
            let node = $graph.node(gid.0 as usize);
            let inp = $graph.inputs(node.inputs_offset, node.input_count);
            a.push($frame.get_value_by_global(inp[0]).$acc());
        }
        let mut dst = vec![0 as $rust; n];
        crate::value::batch_unaryop(&mut dst, &a, $op);
        $locals.iter().zip(dst.iter())
            .map(|(&lid, &v)| (lid, Value::$ctor(v)))
            .collect::<Vec<_>>()
    }};
}

/// SIMD 批处理：对一组同类型同操作的节点做批量计算。
///
/// 从 frame 读取输入，调用 Value.rs 的 SIMD 批算函数，返回 (local NodeId, Value) 列表。
/// 不支持类型返回 None，调用方（wrap_fn! 宏）回退到单节点计算。
/// 不写 frame.value_table、不通知下游——由 engine 热循环通过 NodeResult::Batch 处理。
pub fn do_simd_batch(
    frame: &Frame,
    locals: &[NodeId],
    info: BatchInfo,
    node_start: u32,
) -> Option<Vec<(NodeId, Value)>> {
    use crate::value::{ValueTag, BinOp, CmpOp, UnaryOp};
    let _ = (BinOp::Add, CmpOp::Eq, UnaryOp::Neg); // 抑制 unused import
    let graph = &frame.graph;

    if locals.is_empty() { return None; }

    match info {
        BatchInfo { tag, op: BatchOp::Bin(op) } => {
            match tag {
                ValueTag::I32 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i32, i32, as_i32, batch_binop_i32, op)),
                ValueTag::I64 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i64, i64, as_i64, batch_binop_i64, op)),
                ValueTag::F32 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, f32, f32, as_f32, batch_binop_f32, op)),
                ValueTag::F64 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, f64, f64, as_f64, batch_binop_f64, op)),
                ValueTag::I8 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i8, i8, as_i8, batch_binop, op)),
                ValueTag::I16 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i16, i16, as_i16, batch_binop, op)),
                ValueTag::U8 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u8, u8, as_u8, batch_binop, op)),
                ValueTag::U16 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u16, u16, as_u16, batch_binop, op)),
                ValueTag::U32 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u32, u32, as_u32, batch_binop, op)),
                ValueTag::U64 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u64, u64, as_u64, batch_binop, op)),
                ValueTag::I128 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, i128, i128, as_i128, batch_binop, op)),
                ValueTag::U128 => Some(compute_bin_batch_results!(frame, graph, locals, node_start, u128, u128, as_u128, batch_binop, op)),
                ValueTag::Isize => Some(compute_bin_batch_results!(frame, graph, locals, node_start, isize, isize_val, as_isize, batch_binop, op)),
                ValueTag::Usize => Some(compute_bin_batch_results!(frame, graph, locals, node_start, usize, usize_val, as_usize, batch_binop, op)),
                _ => None, // F16/F128/Bool/Char → 不支持，回退到单节点路径
            }
        }
        BatchInfo { tag, op: BatchOp::Cmp(op) } => {
            match tag {
                ValueTag::F32 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, f32, as_f32, batch_cmp_f32, op)),
                ValueTag::F64 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, f64, as_f64, batch_cmp_f64, op)),
                ValueTag::I32 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i32, as_i32, batch_cmp, op)),
                ValueTag::I64 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i64, as_i64, batch_cmp, op)),
                ValueTag::I8 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i8, as_i8, batch_cmp, op)),
                ValueTag::I16 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i16, as_i16, batch_cmp, op)),
                ValueTag::U8 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u8, as_u8, batch_cmp, op)),
                ValueTag::U16 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u16, as_u16, batch_cmp, op)),
                ValueTag::U32 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u32, as_u32, batch_cmp, op)),
                ValueTag::U64 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u64, as_u64, batch_cmp, op)),
                ValueTag::I128 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, i128, as_i128, batch_cmp, op)),
                ValueTag::U128 => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, u128, as_u128, batch_cmp, op)),
                ValueTag::Isize => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, isize, as_isize, batch_cmp, op)),
                ValueTag::Usize => Some(compute_cmp_batch_results!(frame, graph, locals, node_start, usize, as_usize, batch_cmp, op)),
                _ => None, // F16/F128/Bool/Char → 不支持
            }
        }
        BatchInfo { tag, op: BatchOp::Unary(op) } => {
            match tag {
                ValueTag::I32 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i32, i32, as_i32, op)),
                ValueTag::I64 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i64, i64, as_i64, op)),
                ValueTag::I8 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i8, i8, as_i8, op)),
                ValueTag::I16 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i16, i16, as_i16, op)),
                ValueTag::U8 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u8, u8, as_u8, op)),
                ValueTag::U16 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u16, u16, as_u16, op)),
                ValueTag::U32 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u32, u32, as_u32, op)),
                ValueTag::U64 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u64, u64, as_u64, op)),
                ValueTag::I128 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, i128, i128, as_i128, op)),
                ValueTag::U128 => Some(compute_unary_batch_results!(frame, graph, locals, node_start, u128, u128, as_u128, op)),
                ValueTag::Isize => Some(compute_unary_batch_results!(frame, graph, locals, node_start, isize, isize_val, as_isize, op)),
                ValueTag::Usize => Some(compute_unary_batch_results!(frame, graph, locals, node_start, usize, usize_val, as_usize, op)),
                _ => None, // F16/F128/F32/F64/Bool/Char → 不支持
            }
        }
    }
}

// =========================================================================
// compute_fns — 真实计算函数（构建期绑定的函数索引）
// =========================================================================

/// compute_fn: i32 小于等于比较 (<=)
pub fn compute_le_i32(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_i32();
    let b = frame.get_value_by_global(inputs[1]).as_i32();
    Value::bool_val(a <= b)
}

// ---- i32 比较（索引 8-12, 25；算术/位运算/一元由宏生成）----

impl_cmp_compute! {
    compute_eq_i32: == for as_i32;
    compute_ne_i32: != for as_i32;
    compute_lt_i32: < for as_i32;
    compute_gt_i32: > for as_i32;
    compute_ge_i32: >= for as_i32;
}

// ---- i64 比较（索引 55-60；算术/位运算/一元由宏生成）----

impl_cmp_compute! {
    compute_eq_i64: == for as_i64;
    compute_ne_i64: != for as_i64;
    compute_lt_i64: < for as_i64;
    compute_gt_i64: > for as_i64;
    compute_le_i64: <= for as_i64;
    compute_ge_i64: >= for as_i64;
}

// ---- i128 比较（索引 69-74；算术/位运算/一元由宏生成）----
// i128 路径覆盖 i128/u128 类型，并通过 as_int_i128 支持所有整数类型输入

impl_cmp_compute! {
    compute_eq_i128: == for as_int_i128;
    compute_ne_i128: != for as_int_i128;
    compute_lt_i128: < for as_int_i128;
    compute_gt_i128: > for as_int_i128;
    compute_le_i128: <= for as_int_i128;
    compute_ge_i128: >= for as_int_i128;
}

// ---- 整数位运算（索引 78-92）----
// BitAnd/BitOr/BitXor 对 i32/i64/i128 三族，Shl/Shr 对 i32/i64/i128 三族
// 通过 as_int_i128 通用读取，结果按目标类型构造
// 注：具体位运算 compute_fn 由下方 impl_int_ops 宏按类型生成

// =========================================================================
// 全基本类型 compute_fn（索引 92-）：用 paste 宏为每个类型生成全套运算
// =========================================================================
// 整数 12 类型 × 12 运算 = 144；浮点 4 类型 × 6 运算 = 24；合计 168。
// 比较运算沿用按族共用的版本（结果为 bool，输入用 as_int_i128/as_float_f64 跨类型读取）。
// 算术/位运算/一元按具体类型生成，结果天然带正确 tag 并按类型宽度截断/回绕。
//
// 类型规格表：(类型名, Rust 类型, Value ctor, accessor, 是否整数)
// 索引从 92 开始分配。

/// 为指定整数类型生成全套 compute_fn（add/sub/mul/div/mod/bitand/bitor/bitxor/shl/shr/neg/bitnot）
///
/// 算术逻辑复用 Value.rs 的纯算术核心（`arith_*` 函数），runtime 与编译期 ConstFold 共用。
/// compute_fn 仅负责 Frame 取值与 Value 包装，算术本身无 Frame 依赖。
macro_rules! impl_int_ops {
    ($ty:ident, $rust:ty, $ctor:ident, $acc:ident) => {
        pastey::paste! {
            pub fn [<compute_add_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_add_$ty>](a, b))
            }
            pub fn [<compute_sub_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_sub_$ty>](a, b))
            }
            pub fn [<compute_mul_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mul_$ty>](a, b))
            }
            pub fn [<compute_div_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                // 整数除零返回 0（checked 语义，由 arith_div_$ty 实现）
                Value::$ctor(crate::value::[<arith_div_$ty>](a, b))
            }
            pub fn [<compute_mod_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mod_$ty>](a, b))
            }
            pub fn [<compute_bitand_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_bitand_$ty>](a, b))
            }
            pub fn [<compute_bitor_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_bitor_$ty>](a, b))
            }
            pub fn [<compute_bitxor_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_bitxor_$ty>](a, b))
            }
            pub fn [<compute_shl_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                // 移位量按 i32 读取（与原语义一致），纯函数内部 cast u32
                let shift = frame.get_value_by_global(inputs[1]).as_i32();
                Value::$ctor(crate::value::[<arith_shl_$ty>](a, shift))
            }
            pub fn [<compute_shr_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let shift = frame.get_value_by_global(inputs[1]).as_i32();
                Value::$ctor(crate::value::[<arith_shr_$ty>](a, shift))
            }
            pub fn [<compute_neg_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                Value::$ctor(crate::value::[<arith_neg_$ty>](a))
            }
            pub fn [<compute_bitnot_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                Value::$ctor(crate::value::[<arith_bitnot_$ty>](a))
            }
        }
    };
}

/// 为指定浮点类型生成全套 compute_fn（add/sub/mul/div/mod/neg）
///
/// 算术逻辑复用 Value.rs 的纯算术核心（`arith_*` 函数）。
macro_rules! impl_float_ops {
    ($ty:ident, $rust:ty, $ctor:ident, $acc:ident) => {
        pastey::paste! {
            pub fn [<compute_add_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_add_$ty>](a, b))
            }
            pub fn [<compute_sub_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_sub_$ty>](a, b))
            }
            pub fn [<compute_mul_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mul_$ty>](a, b))
            }
            pub fn [<compute_div_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_div_$ty>](a, b))
            }
            pub fn [<compute_mod_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                let b = frame.get_value_by_global(inputs[1]).$acc();
                Value::$ctor(crate::value::[<arith_mod_$ty>](a, b))
            }
            pub fn [<compute_neg_$ty>](frame: &mut Frame, node: NodeId) -> Value {
                let graph = frame.graph.clone();
                let n = graph.node(node.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let a = frame.get_value_by_global(inputs[0]).$acc();
                Value::$ctor(crate::value::[<arith_neg_$ty>](a))
            }
        }
    };
}

// 整数类型展开（12 类型 × 12 运算 = 144 函数）
impl_int_ops!(i8,    i8,    i8,    as_i8);
impl_int_ops!(i16,   i16,   i16,   as_i16);
impl_int_ops!(i32,   i32,   i32,   as_i32);
impl_int_ops!(i64,   i64,   i64,   as_i64);
impl_int_ops!(i128,  i128,  i128,  as_i128);
impl_int_ops!(u8,    u8,    u8,    as_u8);
impl_int_ops!(u16,   u16,   u16,   as_u16);
impl_int_ops!(u32,   u32,   u32,   as_u32);
impl_int_ops!(u64,   u64,   u64,   as_u64);
impl_int_ops!(u128,  u128,  u128,  as_u128);
impl_int_ops!(isize, isize, isize_val, as_isize);
impl_int_ops!(usize, usize, usize_val, as_usize);

// 浮点类型展开（4 类型 × 6 运算 = 24 函数）
impl_float_ops!(f16, F16, f16, as_f16);
impl_float_ops!(f32, f32, f32, as_f32);
impl_float_ops!(f64, f64, f64, as_f64);
impl_float_ops!(f128, F128, f128, as_f128);

// ---- f64 比较（索引 16-21；算术/一元由宏生成）----

impl_cmp_compute! {
    compute_eq_f64: == for as_f64;
    compute_ne_f64: != for as_f64;
    compute_lt_f64: < for as_f64;
    compute_gt_f64: > for as_f64;
    compute_le_f64: <= for as_f64;
    compute_ge_f64: >= for as_f64;
}

// ---- f128 比较（索引 302-307）：IEEE 754 语义，不经 to_f64 丢精度 ----
// F128 的 derive PartialEq 是 bit-pattern 比较（NaN==NaN 为 true），
// 不能直接用于 IEEE 语义。这里手动实现：
//   - NaN 与任何值比较：eq/lt/gt/le/ge → false，ne → true
//   - -0 == +0（bit pattern 仅符号位不同时视为相等）
//   - 其余用 totalOrder 排序键（sign-aware bit-pattern）

/// F128 NaN 判定
#[inline]
fn f128_is_nan(bits: u128) -> bool {
    (bits >> 112) & 0x7FFF == 0x7FFF && (bits & ((1u128 << 112) - 1)) != 0
}

/// F128 totalOrder 排序键（非 NaN 值）
#[inline]
fn f128_sort_key(bits: u128) -> u128 {
    // 负数（sign=1）：翻转所有位 → 映射到 [0, 0x7FFF...FFF]
    // 正数（sign=0）：置符号位为 1 → 映射到 [0x8000...000, 0xFFFF...FFF]
    // 这样 -0 < +0（totalOrder 语义），-Inf < +Inf 等
    if (bits >> 127) != 0 { !bits } else { bits | (1u128 << 127) }
}

pub fn compute_eq_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let eq = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else {
        ab == bb || (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
    };
    Value::bool_val(eq)
}

pub fn compute_ne_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let ne = if f128_is_nan(ab) || f128_is_nan(bb) {
        true
    } else {
        ab != bb && ((ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF != 0)
    };
    Value::bool_val(ne)
}

pub fn compute_lt_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let lt = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else if (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0 {
        false // -0 == +0，不小于
    } else {
        f128_sort_key(ab) < f128_sort_key(bb)
    };
    Value::bool_val(lt)
}

pub fn compute_gt_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let gt = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else if (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0 {
        false // -0 == +0，不大于
    } else {
        f128_sort_key(ab) > f128_sort_key(bb)
    };
    Value::bool_val(gt)
}

pub fn compute_le_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let le = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else {
        // -0 == +0 → le=true；否则 totalOrder less-or-equal
        (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
            || f128_sort_key(ab) < f128_sort_key(bb)
    };
    Value::bool_val(le)
}

pub fn compute_ge_f128(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_f128();
    let b = frame.get_value_by_global(inputs[1]).as_f128();
    let ab = u128::from_le_bytes(a.0);
    let bb = u128::from_le_bytes(b.0);
    let ge = if f128_is_nan(ab) || f128_is_nan(bb) {
        false
    } else {
        (ab | bb) & 0x7FFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF == 0
            || f128_sort_key(ab) > f128_sort_key(bb)
    };
    Value::bool_val(ge)
}

// ---- bool 逻辑（索引 22-24, 27）----

/// compute_fn: bool 与（复用纯算术核心）
pub fn compute_and_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(crate::value::arith_and_bool(a, b))
}

/// compute_fn: bool 或（复用纯算术核心）
pub fn compute_or_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(crate::value::arith_or_bool(a, b))
}

/// compute_fn: bool 非（一元，复用纯算术核心）
pub fn compute_not_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    Value::bool_val(crate::value::arith_not_bool(a))
}

/// compute_fn: bool 相等
pub fn compute_eq_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(a == b)
}

/// compute_fn: bool 不等（与 eq_bool 对称）
pub fn compute_ne_bool(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let a = frame.get_value_by_global(inputs[0]).as_bool();
    let b = frame.get_value_by_global(inputs[1]).as_bool();
    Value::bool_val(a != b)
}

// ---- throw 包装（索引 28，无 try-catch）----

/// compute_fn: 将值包装为 ThrowVal(Err)（throw 语句用）。
///
/// Kuzo 无 try-catch，throw 产 ThrowVal(Err) + Return 信号，逐层透传至顶层。
/// Err payload 直接持有 thrown 值本身（Bug #27 修复前曾把原始类型包装为
/// Error(value:v) record，导致需要 Error(Error(v)) 嵌套解构）。
/// - 输入为 ThrowVal（已是 throw 值）→ 直接返回（幂等）
/// - 其他值（标量/Str/Record/Adt/Array）→ 直接作为 ThrowVal(Err(v))
pub fn compute_throw_wrap_err(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    use crate::value::{HeapObj, ThrowValue, ThrowPayload};
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    // 已是 ThrowVal → 直接 re-throw（幂等，支持 re-throw）
    if let Some(HeapObj::ThrowVal(_)) = v.heap_obj() {
        return NodeResult::Return(v);
    }
    // 任意值直接作为 Err payload（原始类型不再包装为 Error record）
    let throw_val = Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Err(v) }));
    NodeResult::Return(throw_val)
}

/// compute_fn: 将值包装为 ThrowVal(Ok(val))（Ok 构造器用）。
pub fn compute_throw_ok(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ThrowValue, ThrowPayload};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Ok(val) }))
}

/// compute_fn: 将值包装为 ThrowVal(Err(v))（Err 构造器用）。
///
/// 输入通常为 record_construct 节点的结果（Record/Adt），但 Err 构造器对任意
/// 值类型一视同仁：直接作为 ThrowVal(Err(v))。与 compute_throw_wrap_err 一致，
/// 不再对原始类型做 Error(value:v) 包装（Bug #27）。
pub fn compute_throw_err(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ThrowValue, ThrowPayload};
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    Value::ref_val(HeapObj::ThrowVal(ThrowValue { payload: ThrowPayload::Err(v) }))
}

/// compute_fn (idx 47): `?` 运算符（Propagate）。
///
/// 输入为 ThrowVal：
/// - Ok(val) → 返回 NodeResult::Value(val)（解包）
/// - Err(err) → 返回 NodeResult::Return(ThrowVal(Err))，函数提前返回错误
///
/// 输入为 Nullable 值：
/// - null → 返回 NodeResult::Return(null)，函数提前返回 null
/// - 非 null → 返回 NodeResult::Value(v)（nullable 值与非空值表示同构，直接透传）
pub fn compute_propagate(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);

    if let Some(crate::value::HeapObj::ThrowVal(tv)) = v.heap_obj() {
        match &tv.payload {
            crate::value::ThrowPayload::Ok(val) => NodeResult::Value(val.clone()),
            crate::value::ThrowPayload::Err(_) => {
                // 错误传播：返回 Return，携带原始 ThrowVal(Err) 逐层透传
                NodeResult::Return(v.clone())
            }
        }
    } else if v.is_null() {
        // Nullable 传播：值为 null 时，返回 Return 携带 null 提前返回
        NodeResult::Return(v.clone())
    } else {
        // 非 null 的 Nullable 值：直接透传
        NodeResult::Value(v)
    }
}

/// compute_fn (idx 46): @extern("C") FFI 调用。
///
/// 根据节点的 ffi_call_names 元数据获取函数名，从输入收集参数值，
/// 分发到对应的 Ffi::wrapper 函数，返回结果 Value。
/// FFI 调用是同步的，不设 pending_call，不挂起帧。
#[cfg(has_extern_c)]
pub fn compute_ffi_call(frame: &mut Frame, node: NodeId) -> Value {
    use crate::ffi::Ffi::wrapper;
    read_node_inputs!(frame, node, graph, n, inputs);
    let fn_name = graph.ffi_call_name(node.0 as usize)
        .expect("compute_ffi_call: no ffi_call_name");

    // 从 Value 提取 str 参数（HeapObj::Str → owned String，避免临时 Value 生命周期问题）
    fn extract_str(v: &Value) -> String {
        match v.heap_obj() {
            Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
            _ => panic!("FFI str arg expected, got non-str value"),
        }
    }

    // 从 Value 提取 u8[] 参数（统一走 ArrayValue::collect_u8_bytes）
    fn extract_u8_buf(v: &Value) -> Vec<u8> {
        match v.heap_obj() {
            Some(crate::value::HeapObj::Array(arr)) => arr.collect_u8_bytes(),
            _ => panic!("FFI u8[] arg expected"),
        }
    }

    // 将 FFI 读取的数据写回 Kuzo u8[] 数组（原地修改底层 HeapObj，&self 语义）。
    //
    // extract_u8_buf 返回 clone 的 Vec，FFI 写入局部 Vec 后需写回原数组，
    // 否则 Kuzo 侧读取的数据为零（H-2 修复）。
    // 仅写回 buf[0..n]，n 由 FFI 返回值决定（调用方传入）。
    fn writeback_u8_buf(buf_val: &Value, data: &[u8], n: usize) {
        if let Value::Ref(arc) = buf_val {
            // Safety: 引擎单线程执行，caller 帧在 callee 执行期间 Suspended，
            // 不会有并发访问同一 HeapObj 的路径（与 compute_record_field_set 一致）。
            let ptr = std::sync::Arc::as_ptr(arc) as *mut crate::value::HeapObj;
            unsafe {
                if let crate::value::HeapObj::Array(arr) = &mut *ptr {
                    let len = n.min(data.len()).min(arr.elements.len());
                    // SOA 快路径：U8 连续存储直接 memcpy
                    if let Some(crate::value::ScalarSoA::U8(ref mut soa_data)) = arr.scalar_soa {
                        let len = len.min(soa_data.len());
                        soa_data[..len].copy_from_slice(&data[..len]);
                    } else {
                        for i in 0..len {
                            arr.elements[i] = Value::u8(data[i]);
                        }
                    }
                }
            }
        }
    }


    match fn_name {
        // ── IO: stdout/stderr ──
        "__stdout_write_raw" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let rc = unsafe { wrapper::__stdout_write_raw(&s) };
            Value::i32(rc)
        }
        "__stderr_write_raw" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let rc = unsafe { wrapper::__stderr_write_raw(&s) };
            Value::i32(rc)
        }

        // ── IO: file ops ──
        "__file_open_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let flags = frame.get_value_by_global(inputs[1]).as_i32();
            let mode = frame.get_value_by_global(inputs[2]).as_i32();
            let fd = unsafe { wrapper::__file_open_raw(&path, flags, mode) };
            Value::i64(fd)
        }
        "__file_close_raw" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let rc = unsafe { wrapper::__file_close_raw(fd) };
            Value::i32(rc)
        }
        "__file_seek_raw" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let offset = frame.get_value_by_global(inputs[1]).as_i64();
            let whence = frame.get_value_by_global(inputs[2]).as_i32();
            let pos = unsafe { wrapper::__file_seek_raw(fd, offset, whence) };
            Value::i64(pos)
        }
        "__file_remove_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let rc = unsafe { wrapper::__file_remove_raw(&path) };
            Value::i32(rc)
        }
        "__file_rename_raw" => {
            let old = extract_str(&frame.get_value_by_global(inputs[0]));
            let new = extract_str(&frame.get_value_by_global(inputs[1]));
            let rc = unsafe { wrapper::__file_rename_raw(&old, &new) };
            Value::i32(rc)
        }
        "__file_chmod_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let mode = frame.get_value_by_global(inputs[1]).as_i32();
            let rc = unsafe { wrapper::__file_chmod_raw(&path, mode) };
            Value::i32(rc)
        }

        // ── IO: file read/write (u8[] + len) ──
        "__file_read_into" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__file_read_into(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }
        "__file_write" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[1]));
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__file_write(fd, &buf, len) };
            Value::i64(n)
        }

        // ── IO: stdin ──
        "__stdin_readln_into" => {
            let buf_val = frame.get_value_by_global(inputs[0]);
            let mut buf = extract_u8_buf(&buf_val);
            let n = unsafe { wrapper::__stdin_readln_into(&mut buf) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }

        // ── IO: stat/fstat ──
        "__file_stat_into" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let rc = unsafe { wrapper::__file_stat_into(&path, &mut buf) };
            if rc == 0 { writeback_u8_buf(&buf_val, &buf, buf.len()); }
            Value::i32(rc)
        }
        "__file_fstat_into" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let rc = unsafe { wrapper::__file_fstat_into(fd, &mut buf) };
            if rc == 0 { writeback_u8_buf(&buf_val, &buf, buf.len()); }
            Value::i32(rc)
        }

        // ── IO: dir ops ──
        "__dir_create_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let recursive = frame.get_value_by_global(inputs[1]).as_bool();
            let rc = unsafe { wrapper::__dir_create_raw(&path, recursive) };
            Value::i32(rc)
        }
        "__dir_remove_raw" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let recursive = frame.get_value_by_global(inputs[1]).as_bool();
            let rc = unsafe { wrapper::__dir_remove_raw(&path, recursive) };
            Value::i32(rc)
        }

        // ── IO: dir list ──
        "__dir_list_into" => {
            let path = extract_str(&frame.get_value_by_global(inputs[0]));
            let names_val = frame.get_value_by_global(inputs[1]);
            let offsets_val = frame.get_value_by_global(inputs[2]);
            let kinds_val = frame.get_value_by_global(inputs[3]);
            let mut names_buf = extract_u8_buf(&names_val);
            let mut name_offsets = extract_u8_buf(&offsets_val);
            let mut kinds_buf = extract_u8_buf(&kinds_val);
            let max_count = frame.get_value_by_global(inputs[4]).as_usize();
            let count = unsafe {
                wrapper::__dir_list_into(&path, &mut names_buf, &mut name_offsets, &mut kinds_buf, max_count)
            };
            if count > 0 {
                writeback_u8_buf(&names_val, &names_buf, names_buf.len());
                writeback_u8_buf(&offsets_val, &name_offsets, name_offsets.len());
                writeback_u8_buf(&kinds_val, &kinds_buf, kinds_buf.len());
            }
            Value::i64(count)
        }

        // ── net: tcp ──
        "__net_tcp_connect_v4" => {
            let ip_bits = frame.get_value_by_global(inputs[0]).as_u32();
            let port = frame.get_value_by_global(inputs[1]).as_u16();
            let timeout_ns = frame.get_value_by_global(inputs[2]).as_i64();
            let fd = unsafe { wrapper::__net_tcp_connect_v4(ip_bits, port, timeout_ns) };
            Value::i64(fd)
        }
        "__net_tcp_listen_v4" => {
            let ip_bits = frame.get_value_by_global(inputs[0]).as_u32();
            let port = frame.get_value_by_global(inputs[1]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[2]).as_bool();
            let fd = unsafe {
                wrapper::__net_tcp_listen_v4(ip_bits, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_tcp_accept" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let conn_fd = unsafe { wrapper::__net_tcp_accept(fd) };
            Value::i64(conn_fd)
        }
        "__net_tcp_read" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_tcp_read(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }
        "__net_tcp_write" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[1]));
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_tcp_write(fd, &buf, len) };
            Value::i64(n)
        }
        "__net_tcp_close_raw" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let rc = unsafe { wrapper::__net_tcp_close_raw(fd) };
            Value::i32(rc)
        }

        // ── net: udp v4 ──
        "__net_udp_bind_v4" => {
            let ip_bits = frame.get_value_by_global(inputs[0]).as_u32();
            let port = frame.get_value_by_global(inputs[1]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[2]).as_bool();
            let fd = unsafe {
                wrapper::__net_udp_bind_v4(ip_bits, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_udp_send_to_v4" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let ip_bits = frame.get_value_by_global(inputs[1]).as_u32();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[3]));
            let len = frame.get_value_by_global(inputs[4]).as_usize();
            let n = unsafe { wrapper::__net_udp_send_to_v4(fd, ip_bits, port, &buf, len) };
            Value::i64(n)
        }
        "__net_udp_recv_from_v4" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_udp_recv_from_v4(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }

        // ── net: resolve ──
        "__net_resolve_into" => {
            let host = extract_str(&frame.get_value_by_global(inputs[0]));
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut out_buf = extract_u8_buf(&buf_val);
            let max_count = frame.get_value_by_global(inputs[2]).as_usize();
            let count = unsafe { wrapper::__net_resolve_into(&host, &mut out_buf, max_count) };
            if count > 0 { writeback_u8_buf(&buf_val, &out_buf, out_buf.len()); }
            Value::i64(count)
        }

        // ── net: tcp/udp v6 ──
        "__net_tcp_connect_v6" => {
            let ip_hi = frame.get_value_by_global(inputs[0]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[1]).as_u64();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let timeout_ns = frame.get_value_by_global(inputs[3]).as_i64();
            let fd = unsafe { wrapper::__net_tcp_connect_v6(ip_hi, ip_lo, port, timeout_ns) };
            Value::i64(fd)
        }
        "__net_tcp_listen_v6" => {
            let ip_hi = frame.get_value_by_global(inputs[0]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[1]).as_u64();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[3]).as_bool();
            let fd = unsafe {
                wrapper::__net_tcp_listen_v6(ip_hi, ip_lo, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_udp_bind_v6" => {
            let ip_hi = frame.get_value_by_global(inputs[0]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[1]).as_u64();
            let port = frame.get_value_by_global(inputs[2]).as_u16();
            let reuse_addr = frame.get_value_by_global(inputs[3]).as_bool();
            let fd = unsafe {
                wrapper::__net_udp_bind_v6(ip_hi, ip_lo, port, reuse_addr)
            };
            Value::i64(fd)
        }
        "__net_udp_send_to_v6" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let ip_hi = frame.get_value_by_global(inputs[1]).as_u64();
            let ip_lo = frame.get_value_by_global(inputs[2]).as_u64();
            let port = frame.get_value_by_global(inputs[3]).as_u16();
            let buf = extract_u8_buf(&frame.get_value_by_global(inputs[4]));
            let len = frame.get_value_by_global(inputs[5]).as_usize();
            let n = unsafe { wrapper::__net_udp_send_to_v6(fd, ip_hi, ip_lo, port, &buf, len) };
            Value::i64(n)
        }
        "__net_udp_recv_from_v6" => {
            let fd = frame.get_value_by_global(inputs[0]).as_i64();
            let buf_val = frame.get_value_by_global(inputs[1]);
            let mut buf = extract_u8_buf(&buf_val);
            let len = frame.get_value_by_global(inputs[2]).as_usize();
            let n = unsafe { wrapper::__net_udp_recv_from_v6(fd, &mut buf, len) };
            if n > 0 { writeback_u8_buf(&buf_val, &buf, n as usize); }
            Value::i64(n)
        }

        // ── time ──
        "__instant_now_ns" => {
            let ns = unsafe { wrapper::__instant_now_ns() };
            Value::i64(ns)
        }
        "__systemtime_now_ns" => {
            let ns = unsafe { wrapper::__systemtime_now_ns() };
            Value::i64(ns)
        }
        "__sleep_ns" => {
            let ns = frame.get_value_by_global(inputs[0]).as_i64();
            unsafe { wrapper::__sleep_ns(ns) };
            Value::VOID
        }
        "__localtime_offset_minutes" => {
            let minutes = unsafe { wrapper::__localtime_offset_minutes() };
            Value::i32(minutes)
        }

        // ── cast: widening to i128 ──
        "__cast_i8_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i8();
            let r = unsafe { wrapper::__cast_i8_to_i128(x) };
            Value::i128(r)
        }
        "__cast_i16_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i16();
            let r = unsafe { wrapper::__cast_i16_to_i128(x) };
            Value::i128(r)
        }
        "__cast_i32_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i32();
            let r = unsafe { wrapper::__cast_i32_to_i128(x) };
            Value::i128(r)
        }
        "__cast_i64_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_i64();
            let r = unsafe { wrapper::__cast_i64_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u8_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u8();
            let r = unsafe { wrapper::__cast_u8_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u16_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u16();
            let r = unsafe { wrapper::__cast_u16_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u32_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u32();
            let r = unsafe { wrapper::__cast_u32_to_i128(x) };
            Value::i128(r)
        }
        "__cast_u64_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_u64();
            let r = unsafe { wrapper::__cast_u64_to_i128(x) };
            Value::i128(r)
        }
        "__cast_usize_to_i128" => {
            let x = frame.get_value_by_global(inputs[0]).as_usize();
            let r = unsafe { wrapper::__cast_usize_to_i128(x) };
            Value::i128(r)
        }

        // ── cast: narrowing from i128 ──
        "__cast_i128_to_i8" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i8(v) };
            Value::i8(r)
        }
        "__cast_i128_to_i16" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i16(v) };
            Value::i16(r)
        }
        "__cast_i128_to_i32" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i32(v) };
            Value::i32(r)
        }
        "__cast_i128_to_i64" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_i64(v) };
            Value::i64(r)
        }
        "__cast_i128_to_u8" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u8(v) };
            Value::u8(r)
        }
        "__cast_i128_to_u16" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u16(v) };
            Value::u16(r)
        }
        "__cast_i128_to_u32" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u32(v) };
            Value::u32(r)
        }
        "__cast_i128_to_u64" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_u64(v) };
            Value::u64(r)
        }
        "__cast_i128_to_usize" => {
            let v = frame.get_value_by_global(inputs[0]).as_i128();
            let r = unsafe { wrapper::__cast_i128_to_usize(v) };
            Value::usize_val(r)
        }

        // ── cast: char ──
        "__cast_char_to_u8" => {
            let x = frame.get_value_by_global(inputs[0]).as_u32();
            let r = unsafe { wrapper::__cast_char_to_u8(char_from_u32_or_nul(x)) };
            Value::u8(r)
        }

        // ── reflect: __reflect_format/__reflect_scalar_to_str 已拆分为独立
        // compute_fn（CF_REFLECT_FORMAT/CF_REFLECT_SCALAR_TO_STR，idx 290/291），
        // 不再走 FFI 分派路径 ──
        "__reflect_kind" => {
            let v = frame.get_value_by_global(inputs[0]);
            Value::u8(reflect_kind(&v))
        }
        "__reflect_type_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = reflect_type_name(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_array_len" => {
            let v = frame.get_value_by_global(inputs[0]);
            match v.heap_obj() {
                Some(crate::value::HeapObj::Array(arr)) => Value::usize_val(arr.elements.len()),
                _ => Value::usize_val(0),
            }
        }
        "__reflect_field_count" => {
            let v = frame.get_value_by_global(inputs[0]);
            let count: u16 = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => rec.fields.len() as u16,
                Some(crate::value::HeapObj::Adt(a)) => a.fields.len() as u16,
                _ => 0,
            };
            Value::u16(count)
        }
        "__reflect_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u8 = match &v {
                Value::Scalar(_, tag) => tag.byte_width() as u8,
                _ => 0,
            };
            Value::u8(size)
        }
        "__reflect_field_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => {
                    rec.field_names.get(i as usize)
                        .and_then(|n| n.as_ref())
                        .cloned()
                        .unwrap_or_default()
                }
                Some(crate::value::HeapObj::Adt(a)) => {
                    a.fields.get(i as usize)
                        .and_then(|f| f.name.as_ref().cloned())
                        .unwrap_or_default()
                }
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_field_value" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => {
                    rec.fields.get(i as usize).cloned().unwrap_or(Value::NULL)
                }
                Some(crate::value::HeapObj::Adt(a)) => {
                    a.fields.get(i as usize).map(|f| f.value.clone()).unwrap_or(Value::NULL)
                }
                _ => Value::NULL,
            }
        }
        "__reflect_adt_constructor" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Adt(a)) => a.constructor.clone(),
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_kind_str" => {
            let v = frame.get_value_by_global(inputs[0]);
            let kind = reflect_kind_str(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(kind)))
        }
        "__reflect_layout_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u32 = crate::value::reflect_layout_size(&v);
            Value::u32(size)
        }
        "__reflect_layout_alignment" => {
            let v = frame.get_value_by_global(inputs[0]);
            let align: u32 = crate::value::reflect_layout_alignment(&v);
            Value::u32(align)
        }

        // ── str: UTF-8 逐字符解码（纯 Rust 位运算，与 C 实现语义一致）──
        "__str_utf8_decode_at" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            match utf8_decode_at(bytes, offset) {
                Some((cp, _)) => Value::i64(cp as i64),
                None => Value::i64(UTF8_DECODE_ERR),
            }
        }
        "__str_utf8_char_len_at" => {
            let s = extract_str(&frame.get_value_by_global(inputs[0]));
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            if offset >= bytes.len() {
                Value::usize_val(0)
            } else {
                let c = bytes[offset];
                let len = if c < 0x80 { 1 }
                    else if (c & 0xE0) == 0xC0 { 2 }
                    else if (c & 0xF0) == 0xE0 { 3 }
                    else if (c & 0xF8) == 0xF0 { 4 }
                    else { 1 };
                Value::usize_val(len)
            }
        }

        // ── 未实现的 FFI 函数 ──
        other => panic!("compute_ffi_call: unimplemented FFI function '{}'", other),
    }
}

/// compute_fn (idx 46) fallback：has_extern_c 未设置时（无 C 编译器），
/// 对纯 Rust 可实现的 FFI 函数（cast）用 Rust 直接计算，其余返回默认值。
#[cfg(not(has_extern_c))]
pub fn compute_ffi_call(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let fn_name = graph.ffi_call_name(node.0 as usize)
        .expect("compute_ffi_call: no ffi_call_name");

    match fn_name {
        // ── IO: 用 Rust std 直接实现 ──
        "__stdout_write_raw" => {
            if let Some(crate::value::HeapObj::Str(s)) = frame.get_value_by_global(inputs[0]).heap_obj() {
                use std::io::Write;
                let _ = std::io::stdout().write_all(s.bytes().as_bytes());
                let _ = std::io::stdout().flush();
                Value::i32(IO_OK)
            } else {
                Value::i32(IO_ERR)
            }
        }
        "__stderr_write_raw" => {
            if let Some(crate::value::HeapObj::Str(s)) = frame.get_value_by_global(inputs[0]).heap_obj() {
                use std::io::Write;
                let _ = std::io::stderr().write_all(s.bytes().as_bytes());
                Value::i32(IO_OK)
            } else {
                Value::i32(IO_ERR)
            }
        }

        // ── time: 用 Rust std 直接实现 ──
        "__instant_now_ns" => {
            let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
            Value::i64(ns as i64)
        }
        "__systemtime_now_ns" => {
            let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
            Value::i64(ns as i64)
        }
        "__sleep_ns" => {
            let ns = frame.get_value_by_global(inputs[0]).as_i64();
            std::thread::sleep(std::time::Duration::from_nanos(ns as u64));
            Value::VOID
        }
        "__localtime_offset_minutes" => {
            Value::i32(0)
        }

        // ── cast: widening to i128（纯 Rust 计算）──
        "__cast_i8_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i8() as i128),
        "__cast_i16_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i16() as i128),
        "__cast_i32_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i32() as i128),
        "__cast_i64_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_i64() as i128),
        "__cast_u8_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u8() as i128),
        "__cast_u16_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u16() as i128),
        "__cast_u32_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u32() as i128),
        "__cast_u64_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_u64() as i128),
        "__cast_usize_to_i128" => Value::i128(frame.get_value_by_global(inputs[0]).as_usize() as i128),

        // ── cast: narrowing from i128（纯 Rust 计算）──
        "__cast_i128_to_i8" => Value::i8(frame.get_value_by_global(inputs[0]).as_i128() as i8),
        "__cast_i128_to_i16" => Value::i16(frame.get_value_by_global(inputs[0]).as_i128() as i16),
        "__cast_i128_to_i32" => Value::i32(frame.get_value_by_global(inputs[0]).as_i128() as i32),
        "__cast_i128_to_i64" => Value::i64(frame.get_value_by_global(inputs[0]).as_i128() as i64),
        "__cast_i128_to_u8" => Value::u8(frame.get_value_by_global(inputs[0]).as_i128() as u8),
        "__cast_i128_to_u16" => Value::u16(frame.get_value_by_global(inputs[0]).as_i128() as u16),
        "__cast_i128_to_u32" => Value::u32(frame.get_value_by_global(inputs[0]).as_i128() as u32),
        "__cast_i128_to_u64" => Value::u64(frame.get_value_by_global(inputs[0]).as_i128() as u64),
        "__cast_i128_to_usize" => Value::usize_val(frame.get_value_by_global(inputs[0]).as_i128() as usize),

        // ── cast: char ──
        "__cast_char_to_u8" => Value::u8(frame.get_value_by_global(inputs[0]).as_u32() as u8),

        // ── reflect: __reflect_format/__reflect_scalar_to_str 已拆分为独立
        // compute_fn（CF_REFLECT_FORMAT/CF_REFLECT_SCALAR_TO_STR，idx 290/291），
        // 不再走 FFI 分派路径 ──
        "__reflect_kind" => {
            let v = frame.get_value_by_global(inputs[0]);
            Value::u8(reflect_kind(&v))
        }
        "__reflect_type_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = reflect_type_name(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_array_len" => {
            let v = frame.get_value_by_global(inputs[0]);
            match v.heap_obj() {
                Some(crate::value::HeapObj::Array(arr)) => Value::usize_val(arr.elements.len()),
                _ => Value::usize_val(0),
            }
        }
        "__reflect_field_count" => {
            let v = frame.get_value_by_global(inputs[0]);
            let count: u16 = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => rec.fields.len() as u16,
                Some(crate::value::HeapObj::Adt(a)) => a.fields.len() as u16,
                _ => 0,
            };
            Value::u16(count)
        }
        "__reflect_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u8 = match &v {
                Value::Scalar(_, tag) => tag.byte_width() as u8,
                _ => 0,
            };
            Value::u8(size)
        }
        "__reflect_field_name" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => {
                    rec.field_names.get(i as usize).and_then(|n| n.as_ref()).cloned().unwrap_or_default()
                }
                Some(crate::value::HeapObj::Adt(a)) => {
                    a.fields.get(i as usize).and_then(|f| f.name.as_ref().cloned()).unwrap_or_default()
                }
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_field_value" => {
            let v = frame.get_value_by_global(inputs[0]);
            let i = frame.get_value_by_global(inputs[1]).as_u16();
            match v.heap_obj() {
                Some(crate::value::HeapObj::Record(rec)) => rec.fields.get(i as usize).cloned().unwrap_or(Value::NULL),
                Some(crate::value::HeapObj::Adt(a)) => a.fields.get(i as usize).map(|f| f.value.clone()).unwrap_or(Value::NULL),
                _ => Value::NULL,
            }
        }
        "__reflect_adt_constructor" => {
            let v = frame.get_value_by_global(inputs[0]);
            let name = match v.heap_obj() {
                Some(crate::value::HeapObj::Adt(a)) => a.constructor.clone(),
                _ => String::new(),
            };
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&name)))
        }
        "__reflect_kind_str" => {
            let v = frame.get_value_by_global(inputs[0]);
            let kind = reflect_kind_str(&v);
            Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(kind)))
        }
        "__reflect_layout_size" => {
            let v = frame.get_value_by_global(inputs[0]);
            let size: u32 = crate::value::reflect_layout_size(&v);
            Value::u32(size)
        }
        "__reflect_layout_alignment" => {
            let v = frame.get_value_by_global(inputs[0]);
            let align: u32 = crate::value::reflect_layout_alignment(&v);
            Value::u32(align)
        }

        // ── str: UTF-8 逐字符解码（纯 Rust 位运算，与 C 实现语义一致）──
        "__str_utf8_decode_at" => {
            let s = match frame.get_value_by_global(inputs[0]).heap_obj() {
                Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
                _ => String::new(),
            };
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            match utf8_decode_at(bytes, offset) {
                Some((cp, _)) => Value::i64(cp as i64),
                None => Value::i64(UTF8_DECODE_ERR),
            }
        }
        "__str_utf8_char_len_at" => {
            let s = match frame.get_value_by_global(inputs[0]).heap_obj() {
                Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
                _ => String::new(),
            };
            let offset = frame.get_value_by_global(inputs[1]).as_usize();
            let bytes = s.as_bytes();
            if offset >= bytes.len() {
                Value::usize_val(0)
            } else {
                let c = bytes[offset];
                let len = if c < 0x80 { 1 }
                    else if (c & 0xE0) == 0xC0 { 2 }
                    else if (c & 0xF0) == 0xE0 { 3 }
                    else if (c & 0xF8) == 0xF0 { 4 }
                    else { 1 };
                Value::usize_val(len)
            }
        }

        // ── 未实现的 FFI 函数（无 C 编译器时返回默认值）──
        _ => Value::i32(0),
    }
}

// =========================================================================
// reflect 独立 compute_fn（290-291）
//
// 从 compute_ffi_call 拆分出来，避免 lazy force 逻辑与 FFI 调用耦合。
// 这两个函数是唯一涉及 LazyValue 强制求值的 reflect 操作，独立后：
//   - 不再依赖 ffi_call_name 元数据
//   - 不走 FFI 分派路径
//   - lazy force 逻辑与 reflect 格式化逻辑内聚
// =========================================================================

/// compute_fn (idx 290): `__reflect_format` — 任意值 → str
///
/// 格式化前先强制求值 LazyValue（若输入是 lazy），再调用 Reflect::format_value。
/// 不依赖 ffi_call_name，直接读取 inputs[0]。
pub fn compute_reflect_format(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, _n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    let v = force_lazy_value_sync(frame, &v);
    let s = crate::value::format_value(&v, 0);
    Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&s)))
}

/// compute_fn (idx 291): `__reflect_scalar_to_str` — 标量值 → str
///
/// 语义与 compute_reflect_format 一致（均走 format_value），独立保留以
/// 对应 Raw.kz 中的两个不同 @extern("C") 原语声明。
pub fn compute_reflect_scalar_to_str(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, _n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    let v = force_lazy_value_sync(frame, &v);
    let s = crate::value::format_value(&v, 0);
    Value::ref_val(crate::value::HeapObj::Str(crate::value::KuzoStr::from_rust_str(&s)))
}

/// compute_fn: 类型构造（从输入收集字段值，根据 kind 构造 Record/Adt/Newtype HeapObj）
pub fn compute_record_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::ir::Ir::{RecordLitKind, RecordLitInfo};
    use crate::value::{AdtField, AdtValue, HeapObj, NewtypeValue, RecordValue, ValueArena};
    read_node_inputs!(frame, node, graph, n, inputs);
    let fields: Vec<Value> = inputs
        .iter()
        .map(|&input_node| frame.get_value_by_global(input_node))
        .collect();
    let info = graph.record_lit_info_at(node.0 as usize);
    let info: &RecordLitInfo = info
        .as_ref()
        .expect("record construct node has no RecordLitInfo");
    match info.kind {
        RecordLitKind::Record => {
            Value::ref_val(HeapObj::Record(RecordValue {
                type_name: info.type_name.clone(),
                fields,
                field_names: info.field_names.clone(),
                field_ref_bits: 0,
            }))
        }
        RecordLitKind::Adt => {
            let adt_fields: Vec<AdtField> = fields
                .into_iter()
                .enumerate()
                .map(|(i, v)| AdtField {
                    name: info.field_names.get(i).and_then(|n| n.clone()),
                    value: v,
                })
                .collect();
            Value::ref_val(HeapObj::Adt(AdtValue {
                type_name: info.type_name.clone(),
                constructor: info.constructor.clone(),
                fields: adt_fields,
                field_ref_bits: 0,
            }))
        }
        RecordLitKind::Newtype => {
            // Newtype：单字段，将 inner Value 存入全局 arena 得到 ValueHandle
            let inner_val = fields.into_iter().next().unwrap_or(Value::VOID);
            let inner = ValueArena::with_global_mut(|a| a.alloc_value(&inner_val));
            Value::ref_val(HeapObj::Newtype(NewtypeValue {
                type_name: info.type_name.clone(),
                inner,
            }))
        }
    }
}

/// compute_fn: 记录字段访问（按 field 名称从 Record/Adt 取字段值）
///
/// 统一机制：Record 与 Adt 均通过 `find_field(name)` 按名取值，
/// 不依赖编译期 field_idx，消除 idx fallback 与 Record/Adt 双路径差异。
pub fn compute_record_field_get(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let record_val = frame.get_value_by_global(inputs[0]);
    let name = graph.field_set_name(node.0 as usize);
    let make_err = |msg: &str| make_error_throw("FieldError", msg);
    let Some(h) = record_val.heap_obj() else {
        return make_err("field access on non-record value");
    };
    let Some(name) = name else {
        return make_err("field_get node has no field name");
    };
    h.field_get(name).unwrap_or_else(|| {
        make_err(&format!("no such field '{}' on record", name))
    })
}

/// compute_fn: 数组构造（从输入收集元素构造 ArrayValue）
pub fn compute_array_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let elements: Vec<Value> = inputs
        .iter()
        .map(|&input_node| frame.get_value_by_global(input_node))
        .collect();
    Value::ref_val(HeapObj::Array(ArrayValue::new(elements)))
}

/// compute_fn: 栈分配版记录构造（288）
///
/// 分析器标记为不逃逸的分配点使用此 compute_fn。
/// 当前实现等同 compute_record_construct（Value 模型限制下 Arc 是唯一引用方式），
/// 预留分离点：未来 Value 模型支持帧局部分配后，此函数切换为真正的栈分配。
pub fn compute_record_construct_stack(frame: &mut Frame, node: NodeId) -> Value {
    compute_record_construct(frame, node)
}

/// compute_fn: 栈分配版数组构造（289）
///
/// 分析器标记为不逃逸的分配点使用此 compute_fn。
/// 当前实现等同 compute_array_construct，预留分离点。
pub fn compute_array_construct_stack(frame: &mut Frame, node: NodeId) -> Value {
    compute_array_construct(frame, node)
}

/// compute_fn: 数组索引（从 ArrayValue 按 i32 索引取元素）
/// 索引越界时返回 ThrowVal(Err) 错误值，逐层透传至顶层。
pub fn compute_array_index(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let recv_val = frame.get_value_by_global(inputs[0]);
    let idx = frame.get_value_by_global(inputs[1]).as_i32() as usize;
    let make_err = |msg: &str| make_error_throw("IndexError", msg);
    match recv_val.heap_obj() {
        Some(crate::value::HeapObj::Array(arr)) => {
            arr.get(idx).cloned().unwrap_or_else(|| {
                make_err(&format!("index {} out of bounds (len {})", idx, arr.len()))
            })
        }
        Some(crate::value::HeapObj::Str(s)) => {
            s.char_at(idx).map(|c| Value::char_val(c)).unwrap_or_else(|| {
                make_err(&format!("index {} out of bounds (len {})", idx, s.codepoint_count()))
            })
        }
        _ => make_err("index on non-indexable type"),
    }
}

/// compute_fn: 切片 `recv[start..end]` / `recv[start..=end]`。
///
/// 三输入：recv, start, end。inclusive 标志从 graph.slice_inclusive[node] 读取。
/// - str：按码点索引切片，返回新 str
/// - array：按元素索引切片，返回新 array
/// 越界时 clamp 到 [0, len]，与 Rust 切片语义一致（不 panic）。
pub fn compute_slice(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue, KuzoStr};
    read_node_inputs!(frame, node, graph, n, inputs);
    let recv_val = frame.get_value_by_global(inputs[0]);
    let start = frame.get_value_by_global(inputs[1]).as_usize();
    let mut end = frame.get_value_by_global(inputs[2]).as_usize();
    let inclusive = graph.slice_inclusive(node.0 as usize);
    if inclusive {
        end = end.saturating_add(1);
    }
    let make_err = |msg: &str| make_error_throw("SliceError", msg);
    match recv_val.heap_obj() {
        Some(crate::value::HeapObj::Array(arr)) => {
            let len = arr.len();
            let s = start.min(len);
            let e = end.min(len);
            if s > e {
                return make_err(&format!("slice start {} > end {}", s, e));
            }
            let sliced: Vec<Value> = arr.elements[s..e].to_vec();
            Value::ref_val(HeapObj::Array(ArrayValue {
                elements: sliced,
                fixed_size: None,
                elem_is_ref: arr.elem_is_ref,
                scalar_soa: None,
            }))
        }
        Some(crate::value::HeapObj::Str(s)) => {
            // 按码点索引切片：collect chars in [start, end)，重组为 str
            let chars: Vec<char> = s.bytes().chars().collect();
            let len = chars.len();
            let st = start.min(len);
            let en = end.min(len);
            if st > en {
                return make_err(&format!("slice start {} > end {}", st, en));
            }
            let mut buf = String::with_capacity(en - st);
            for c in &chars[st..en] {
                buf.push(*c);
            }
            Value::ref_val(HeapObj::Str(KuzoStr::new(buf)))
        }
        _ => make_err("slice on non-sliceable type"),
    }
}

/// compute_fn: 字符串拼接 `lhs + rhs`（两侧均为 str）。
///
/// 两输入：lhs, rhs。任一非 str 时返回错误值。
pub fn compute_str_concat(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::HeapObj;
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let make_err = |msg: &str| make_error_throw("TypeError", msg);
    match (lhs.heap_obj(), rhs.heap_obj()) {
        (Some(HeapObj::Str(a)), Some(HeapObj::Str(b))) => {
            Value::ref_val(HeapObj::Str(a.concat(b)))
        }
        _ => make_err("str concat on non-str operand"),
    }
}

/// compute_fn (idx 270): 全局变量读取。
///
/// 无输入，从 graph.global_var_storage[slot] 读取值。
/// slot index 从 graph.global_load_slots[node] 获取。
/// 全局变量不依赖帧链，任何函数都能正确读取。
pub fn compute_global_load(frame: &mut Frame, node: NodeId) -> Value {
    let slot = frame.graph.global_load_slot(node.0 as usize)
        .expect("global_load node has no slot");
    let storage = &frame.graph.global_var_storage;
    let guard = storage[slot as usize].lock().unwrap();
    let val = guard.clone().unwrap_or(Value::NULL);
    val
}

/// compute_fn (idx 271): 全局变量写入。
///
/// inputs[0] = 值来源节点，写入 graph.global_var_storage[slot]。
/// slot index 从 graph.global_store_slots[node] 获取。
/// 返回写入的值（供下游链式使用）。
pub fn compute_global_store(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let slot = graph.global_store_slot(node.0 as usize)
        .expect("global_store node has no slot");
    let storage = &frame.graph.global_var_storage;
    *storage[slot as usize].lock().unwrap() = Some(val.clone());
    val
}

/// compute_fn (idx 308): 记忆化缓存查询。
///
/// inputs[0..param_count] = 参数值（用作缓存 key）。
/// MemoInfo.table_index 索引 graph.memo_tables 中的哈希表。
/// 返回 Record `{hit: bool, value: Value}`：
/// - 命中：hit=true, value=缓存值
/// - 未命中：hit=false, value=Void
pub fn compute_memo_check(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, RecordValue};
    use std::hash::{Hash, Hasher};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.memo_info(node.0 as usize)
        .expect("memo_check node has no MemoInfo");
    let param_count = info.param_count as usize;
    // 构造缓存 key：将参数值哈希为 u64
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let param_vals: Vec<Value> = inputs[..param_count].iter()
        .map(|&inp| frame.get_value_by_global(inp))
        .collect();
    if std::env::var("KUZO_DEBUG_MEMO").is_ok() {
        eprintln!("[MEMO_CHECK] table={} params={:?}", info.table_index, param_vals);
    }
    for val in &param_vals {
        val.hash(&mut hasher);
    }
    let key = hasher.finish();
    // 查缓存表
    let table = &frame.graph.memo_tables;
    let hit_val = {
        let guard = table[info.table_index as usize].lock().unwrap();
        guard.get(&key).cloned()
    };
    if std::env::var("KUZO_DEBUG_MEMO").is_ok() {
        eprintln!("[MEMO_CHECK] key={} hit={}", key, hit_val.is_some());
    }
    match hit_val {
        Some(cached) => {
            // 命中：返回 record(hit=true, value=cached)
            Value::ref_val(HeapObj::Record(RecordValue {
                type_name: String::new(),
                fields: vec![Value::bool_val(true), cached],
                field_names: vec![Some("hit".into()), Some("value".into())],
                field_ref_bits: 0,
            }))
        }
        None => {
            // 未命中：返回 record(hit=false, value=void)
            Value::ref_val(HeapObj::Record(RecordValue {
                type_name: String::new(),
                fields: vec![Value::bool_val(false), Value::VOID],
                field_names: vec![Some("hit".into()), Some("value".into())],
                field_ref_bits: 0,
            }))
        }
    }
}

/// compute_fn (idx 309): 记忆化缓存写入。
///
/// inputs[0..param_count] = 参数值（用作缓存 key），
/// inputs[param_count] = 结果值。
/// 写入缓存表后透传结果值（供下游使用）。
pub fn compute_memo_store(frame: &mut Frame, node: NodeId) -> Value {
    use std::hash::{Hash, Hasher};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.memo_info(node.0 as usize)
        .expect("memo_store node has no MemoInfo");
    let param_count = info.param_count as usize;
    let result_val = frame.get_value_by_global(inputs[param_count]);
    // 构造缓存 key
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let param_vals: Vec<Value> = inputs[..param_count].iter()
        .map(|&inp| frame.get_value_by_global(inp))
        .collect();
    for val in &param_vals {
        val.hash(&mut hasher);
    }
    let key = hasher.finish();
    if std::env::var("KUZO_DEBUG_MEMO").is_ok() {
        eprintln!("[MEMO_STORE] table={} key={} params={:?} result={:?}",
            info.table_index, key, param_vals, result_val);
    }
    // 写缓存表
    let table = &frame.graph.memo_tables;
    {
        let mut guard = table[info.table_index as usize].lock().unwrap();
        guard.insert(key, result_val.clone());
    }
    result_val
}

/// compute_fn (idx 272): 记录扩展。
///
/// inputs[0] = base RecordValue，inputs[1..] = 更新字段值。
/// RecordExtendInfo.update_names 给出 inputs[1..] 对应的字段名。
/// 从 base 克隆字段与字段名，按 update_names 替换同名字段或追加新字段，
/// 构造新 RecordValue（保留 base 的 type_name）。
pub fn compute_record_extend(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, RecordValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.record_extend_info_at(node.0 as usize);
    let info = info
        .as_ref()
        .expect("record extend node has no RecordExtendInfo");

    // 取 base RecordValue
    let base_val = frame.get_value_by_global(inputs[0]);
    let base_record: RecordValue = match base_val.heap_obj() {
        Some(HeapObj::Record(r)) => r.clone(),
        _ => {
            // base 非 record：退化为空记录，所有 update 字段作为新字段追加
            RecordValue::new(String::new(), Vec::new(), Vec::new())
        }
    };

    // 收集 update 值（inputs[1..]，按 update_names 顺序）
    let update_values: Vec<Value> = inputs[1..]
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();

    // 克隆 base 字段与字段名，按 update_names 替换/追加
    let mut fields: Vec<Value> = base_record.fields.clone();
    let mut field_names: Vec<Option<String>> = base_record.field_names.clone();
    for (i, update_name) in info.update_names.iter().enumerate() {
        let update_val = update_values[i].clone();
        // 查找同名字段位置
        let pos = field_names.iter().position(|n| n.as_deref() == Some(update_name));
        match pos {
            Some(idx) => {
                // 替换已有字段值
                fields[idx] = update_val;
            }
            None => {
                // 追加新字段
                fields.push(update_val);
                field_names.push(Some(update_name.clone()));
            }
        }
    }

    Value::ref_val(HeapObj::Record(RecordValue {
        type_name: base_record.type_name.clone(),
        fields,
        field_names,
        field_ref_bits: 0,
    }))
}

/// compute_fn (idx 273): 原子构造。
///
/// inputs[0] = 初始值节点，包装为 AtomicValue（共享底层内存的原子容器）。
/// AtomicValue.data 为 Value，compute_fn 上下文无需 arena 即可构造。
pub fn compute_atomic_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, AtomicValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    Value::ref_val(HeapObj::AtomicVal(AtomicValue::new(val)))
}

/// compute_fn: 模式匹配 — 构造器名判别（idx 274）。
///
/// 输入：scrutinee。元数据：构造器名（graph.pattern_ctor_names）。
/// 检查 scrutinee 是否为 ADT 且 constructor 匹配，或 Record 且 type_name 匹配，
/// 或 ThrowVal 且构造器名为 "Ok"/"Error" 匹配对应 payload 变体。
/// 返回 bool。
pub fn compute_pattern_ctor_match(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let ctor_name = graph.pattern_ctor_name(node.0 as usize)
        .expect("pattern ctor match node has no ctor name");
    let matched = match val.heap_obj() {
        Some(crate::value::HeapObj::Adt(a)) => a.constructor == ctor_name,
        Some(crate::value::HeapObj::Record(r)) => r.type_name == ctor_name,
        // Newtype：构造器名 == 类型名，匹配 NewtypeValue.type_name
        Some(crate::value::HeapObj::Newtype(n)) => n.type_name == ctor_name,
        Some(crate::value::HeapObj::ThrowVal(tv)) => match &tv.payload {
            crate::value::ThrowPayload::Ok(_) => ctor_name == CTOR_OK,
            crate::value::ThrowPayload::Err(_) => ctor_name == CTOR_ERR || ctor_name == CTOR_ERR_ALT,
        },
        _ => false,
    };
    Value::bool_val(matched)
}

/// compute_fn: 模式匹配 — ADT/Record/ThrowVal 按位置提取字段（idx 275）。
///
/// 输入：scrutinee。元数据：字段索引（graph.pattern_field_indices）。
/// 从 ADT 按位置取字段值，或从 Record 按位置取字段值，
/// 或从 ThrowVal 取内部值（索引 0：Ok 的 val 或 Err 的 record）。
/// 返回字段值（越界返回 Void）。
pub fn compute_pattern_adt_field_get(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let idx = graph.pattern_field_index(node.0 as usize)
        .expect("pattern adt field get node has no field index")
        as usize;
    match val.heap_obj() {
        Some(crate::value::HeapObj::Adt(a)) => {
            a.fields.get(idx).map(|f| f.value.clone()).unwrap_or(Value::VOID)
        }
        Some(crate::value::HeapObj::Record(r)) => {
            r.fields.get(idx).cloned().unwrap_or(Value::VOID)
        }
        // Newtype：单字段，idx 0 取 inner 值（通过 ValueArena 全局句柄解引用）
        Some(crate::value::HeapObj::Newtype(n)) => {
            if idx == 0 {
                crate::value::ValueArena::with_global(|a| a.get_value(n.inner))
            } else {
                Value::VOID
            }
        }
        Some(crate::value::HeapObj::ThrowVal(tv)) => {
            if idx == 0 {
                match &tv.payload {
                    crate::value::ThrowPayload::Ok(v) => v.clone(),
                    // Err 直接持有 thrown 值本身（Bug #27），match 模式 `Error(v)`
                    // 的 v 直接绑定到 throw 的值，无需 Error(Error(v)) 嵌套解构
                    crate::value::ThrowPayload::Err(v) => v.clone(),
                }
            } else {
                Value::VOID
            }
        }
        _ => Value::VOID,
    }
}

/// compute_fn: 模式匹配 — 字符串相等判别（idx 276）。
///
/// 输入：scrutinee, str_const。比较两个值是否为相等字符串。
/// 返回 bool。
pub fn compute_pattern_str_eq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let lhs_str = match lhs.heap_obj() {
        Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
        _ => return Value::bool_val(false),
    };
    let rhs_str = match rhs.heap_obj() {
        Some(crate::value::HeapObj::Str(s)) => s.bytes().to_string(),
        _ => return Value::bool_val(false),
    };
    Value::bool_val(lhs_str == rhs_str)
}

/// compute_fn: str 比较（292-297）
///
/// 按 Unicode 码点序列字典序比较（Rust str 的 Ord 语义，UTF-8 字节序与码点序一致）。
/// 操作数非 str 时返回 false（Eq/Le/Ge）或按 Ord 语义不 panic 地返回 false。
/// 使用 KuzoStr.compare（Ordering）避免重复分配。
fn str_compare_operands(frame: &mut Frame, node: NodeId) -> Option<std::cmp::Ordering> {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    match (lhs.heap_obj(), rhs.heap_obj()) {
        (Some(crate::value::HeapObj::Str(a)), Some(crate::value::HeapObj::Str(b))) => {
            Some(a.compare(b))
        }
        _ => None,
    }
}

pub fn compute_eq_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) == Some(std::cmp::Ordering::Equal))
}

pub fn compute_ne_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) != Some(std::cmp::Ordering::Equal))
}

pub fn compute_lt_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) == Some(std::cmp::Ordering::Less))
}

pub fn compute_gt_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(str_compare_operands(frame, node) == Some(std::cmp::Ordering::Greater))
}

pub fn compute_le_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(matches!(str_compare_operands(frame, node), Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)))
}

pub fn compute_ge_str(frame: &mut Frame, node: NodeId) -> Value {
    Value::bool_val(matches!(str_compare_operands(frame, node), Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)))
}

/// compute_fn: 通用类型转换 — 任意值 → str（idx 277）。
///
/// 输入：源值节点。按 Value 变体分派格式化为 KuzoStr：
///   - 标量整数 → as_int_i128().to_string()
///   - 标量浮点 → as_float_f64().to_string()
///   - bool → "true"/"false"
///   - char → String::from(char)
///   - Str → clone（identity）
///   - Null → "null"
///   - Void → "void"
///   - 其他 Ref → "<non-scalar>"
pub fn compute_cast_to_str(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, KuzoStr, ValueTag};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);

    let s: String = match &val {
        Value::Null => TYPE_NAME_NULL.to_string(),
        Value::Void => TYPE_NAME_VOID.to_string(),
        Value::Scalar(_, tag) => {
            match tag {
                ValueTag::Bool => val.as_bool().to_string(),
                ValueTag::Char => {
                    let c = val.as_char();
                    let mut buf = [0u8; 4];
                    c.encode_utf8(&mut buf).to_string()
                }
                ValueTag::F16 | ValueTag::F32 | ValueTag::F64 | ValueTag::F128 => {
                    val.as_float_f64().to_string()
                }
                // 所有整数类型
                _ => val.as_int_i128().to_string(),
            }
        }
        Value::Ref(r) => match r.as_ref() {
            HeapObj::Str(kuzo_str) => kuzo_str.bytes().to_string(),
            _ => "<non-scalar>".to_string(),
        },
    };
    Value::ref_val(HeapObj::Str(KuzoStr::new(s)))
}

/// compute_fn: 通用类型转换 — 标量 → 标量（idx 278）。
///
/// 输入：源值节点。元数据：目标类型名（graph.cast_target_types）。
/// 覆盖所有标量互转：int↔int（截断/扩展）、int↔float、float↔float、bool→int、char→int。
/// 目标类型从 cast_target_types 元数据读取，按 ValueTag 分派构造对应 Value。
pub fn compute_cast_scalar(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::ValueTag;
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let target_ty = graph.cast_target_type(node.0 as usize)
        .expect("cast_scalar node has no target type");

    let target_tag = match ValueTag::from_name(target_ty) {
        Some(tag) => tag,
        // 未知目标类型：safe cast 返回 Null，否则返回 Void
        None => {
            return if graph.safe_op_flag(node.0 as usize) {
                Value::Null
            } else {
                Value::VOID
            };
        }
    };

    // 源值是否为浮点
    let src_is_float = matches!(
        &val,
        Value::Scalar(_, ValueTag::F16 | ValueTag::F32 | ValueTag::F64 | ValueTag::F128)
    );
    // 统一读取源值为 f64：浮点用 as_float_f64，整数用 as_int_i128 as f64
    let src_f64 = if src_is_float { val.as_float_f64() } else { val.as_int_i128() as f64 };

    match target_tag {
        ValueTag::I8 => Value::i8(if src_is_float { src_f64 as i8 } else { val.as_i8() }),
        ValueTag::I16 => Value::i16(if src_is_float { src_f64 as i16 } else { val.as_i16() }),
        ValueTag::I32 => Value::i32(if src_is_float { src_f64 as i32 } else { val.as_i32() }),
        ValueTag::I64 => Value::i64(if src_is_float { src_f64 as i64 } else { val.as_i64() }),
        ValueTag::I128 => Value::i128(if src_is_float { src_f64 as i128 } else { val.as_i128() }),
        ValueTag::U8 => Value::u8(if src_is_float { src_f64 as u8 } else { val.as_u8() }),
        ValueTag::U16 => Value::u16(if src_is_float { src_f64 as u16 } else { val.as_u16() }),
        ValueTag::U32 => Value::u32(if src_is_float { src_f64 as u32 } else { val.as_u32() }),
        ValueTag::U64 => Value::u64(if src_is_float { src_f64 as u64 } else { val.as_u64() }),
        ValueTag::U128 => Value::u128(if src_is_float { src_f64 as u128 } else { val.as_u128() }),
        ValueTag::Isize => Value::isize_val(if src_is_float { src_f64 as isize } else { val.as_isize() }),
        ValueTag::Usize => Value::usize_val(if src_is_float { src_f64 as usize } else { val.as_usize() }),
        ValueTag::F16 => Value::f16(crate::value::F16::from_f64(src_f64)),
        ValueTag::F32 => Value::f32(src_f64 as f32),
        ValueTag::F64 => Value::f64(src_f64),
        // 用 as_f128() 精确访问器：整数源走 from_i128/from_u128，浮点源走 to_f64（已精确舍入）
        ValueTag::F128 => Value::f128(val.as_f128()),
        ValueTag::Bool => Value::bool_val(if src_is_float { src_f64 != 0.0 } else { val.as_int_i128() != 0 }),
        ValueTag::Char => Value::char_val(char_from_u32_or_nul(if src_is_float { src_f64 as u32 } else { val.as_int_i128() as u32 })),
        _ => unreachable!("non-scalar target_tag {:?} in cast", target_tag),
    }
}

/// compute_fn (idx 279): 非空断言 `expr!`。
///
/// 输入为 nullable 值：Null → panic（编程错误，非可恢复流程）；
/// 非 Null → 原样返回（Scalar/Ref 透传，即解包 nullable）。
pub fn compute_non_null_assert(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    if v.is_null() {
        panic!("non-null assertion failed: value is null");
    }
    v
}

/// compute_fn (idx 280): 取引用 `&expr`（RefOf）。
///
/// 将输入值包装进 `Arc<HeapObj::Cell>`，返回 `Value::Ref(arc)`。
/// 多个引用共享同一 Cell（通过 Arc clone），写入对所有人可见。
/// 对于已是 Ref 的值（record 等），直接共享同一 Arc（无需二次包装）。
pub fn compute_ref_of(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    match &v {
        // 标量/Null/Void → 包装进 Cell
        Value::Scalar(_, _) | Value::Null | Value::Void => {
            let cell = crate::value::Cell::new(v.clone());
            Value::ref_val(crate::value::HeapObj::Cell(cell))
        }
        // 已是堆引用：直接共享 Arc（引用语义，不深拷贝）
        Value::Ref(_) => v,
    }
}

/// compute_fn (idx 281): 解引用读取 `*ref`（Deref）。
///
/// 输入为 `Arc<HeapObj::Cell>`：返回 Cell 内部值。
/// 输入为其他 Ref（record/array 等）：原样返回（`&rec` 共享 Arc，`*r` 即 rec 本身）。
pub fn compute_deref_read(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    match v.heap_obj() {
        Some(crate::value::HeapObj::Cell(c)) => c.get(),
        _ => v,
    }
}

/// compute_fn (idx 282): 解引用写入 `*ref = value`（DerefAssign）。
///
/// inputs[0] = 引用（Cell），inputs[1] = 新值。
/// 将新值写入 Cell，返回写入的值（供链式使用）。
/// 对非 Cell 引用（record 共享 Arc）不做处理（record 字段写入走 record_field_set）。
pub fn compute_deref_write(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let ref_val = frame.get_value_by_global(inputs[0]);
    let new_val = frame.get_value_by_global(inputs[1]);
    if let Some(crate::value::HeapObj::Cell(c)) = ref_val.heap_obj() {
        c.set(new_val.clone());
    }
    new_val
}


/// compute_fn: 记录字段赋值（就地修改 RecordValue 的字段，返回 void）
///
/// inputs[0] = 记录值节点，inputs[1] = 新值。
/// 字段名从 graph.field_set_names[node] 获取，通过 Arc::make_mut 就地修改。
/// 修改后写回值表槽，使变更对其他节点可见。
pub fn compute_record_field_set(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let new_value = frame.get_value_by_global(inputs[1]);
    let field_name = graph.field_set_name(node.0 as usize)
        .expect("field set node has no field name");
    let record_node_local = NodeId(inputs[0].0.wrapping_sub(frame.node_offset));
    // &self 语义：直接修改 Arc 底层 HeapObj，确保修改对所有持有者可见。
    // 这对迭代器模式（next() 修改 self.pos）等场景至关重要：
    // for 循环通过尾递归传递迭代器引用，若 COW 则 pos 永不更新 → 死循环。
    //
    // Arc::make_mut 在 refcount>1 时 COW，破坏 &self 引用语义。
    // 此处通过 Arc::as_ptr 获取可变指针直接修改，绕过 Rust 别名规则。
    //
    // Safety: 引擎单线程执行（LockStrategy::Single 无锁，Multi 在帧级别互斥），
    // caller 帧在 callee 执行期间处于 Suspended 状态，不会有并发访问同一 HeapObj。
    // Arc 的引用计数不变（不 clone 也不 drop），仅修改堆数据。
    if let Some(val) = frame.value_table.get_value_mut(record_node_local.0 as usize) {
        if let Value::Ref(arc) = val {
            let ptr = std::sync::Arc::as_ptr(arc) as *mut crate::value::HeapObj;
            unsafe {
                match &mut *ptr {
                    crate::value::HeapObj::Record(r) => {
                        if let Some(idx) = r.field_names.iter().position(|n| n.as_deref() == Some(field_name)) {
                            if idx < r.fields.len() {
                                r.fields[idx] = new_value.clone();
                            }
                        }
                    }
                    crate::value::HeapObj::Adt(a) => {
                        if let Some(idx) = a.fields.iter().position(|f| f.name.as_deref() == Some(field_name)) {
                            a.fields[idx].value = new_value.clone();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    Value::VOID
}

/// compute_fn (idx 301): 数组索引存储 `arr[i] = x`。
///
/// 三输入：arr, index, value。原地修改 Array 堆对象的 elements 向量。
/// 与 record_field_set 同语义：通过 Arc::as_ptr 直接修改堆数据，
/// 确保 &self 引用语义（修改对所有持有者可见）。
///
/// Safety: 引擎单线程执行，caller 帧在 callee 执行期间 Suspended，无并发访问。
/// 越界索引扩展数组到 idx+1（补 Void），与动态数组语义一致。
pub fn compute_array_store(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let idx = frame.get_value_by_global(inputs[1]).as_usize();
    let new_value = frame.get_value_by_global(inputs[2]);

    let arr_node_local = NodeId(inputs[0].0.wrapping_sub(frame.node_offset));
    if let Some(val) = frame.value_table.get_value_mut(arr_node_local.0 as usize) {
        if let Value::Ref(arc) = val {
            let ptr = std::sync::Arc::as_ptr(arc) as *mut crate::value::HeapObj;
            unsafe {
                if let crate::value::HeapObj::Array(arr) = &mut *ptr {
                    if idx >= arr.elements.len() {
                        arr.elements.resize(idx + 1, Value::VOID);
                        // SOA 布局在 resize 后需重建（新增元素填充 Void，SOA 无法简单扩展）
                        arr.scalar_soa = None;
                    }
                    arr.elements[idx] = new_value.clone();
                    // 同步更新 SOA：若类型匹配则就地写入，否则失效 SOA 缓存
                    if let Some(ref mut soa) = arr.scalar_soa {
                        if !soa.try_store(idx, &new_value) {
                            arr.scalar_soa = None;
                        }
                    }
                }
            }
        }
    }
    Value::VOID
}

/// compute_fn: null 检查（检查值是否为 null，返回 bool）
pub fn compute_is_null(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let is_null = val.is_null();
    Value::bool_val(is_null)
}

/// compute_fn: 长度（返回 i32，与默认整数运算类型一致）
/// - Array：元素个数
/// - Str：Unicode 码点数（与 str[i] 索引语义一致，均按码点计数）
pub fn compute_array_len(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let len = match val.heap_obj() {
        Some(crate::value::HeapObj::Array(arr)) => arr.len() as i32,
        Some(crate::value::HeapObj::Str(s)) => s.codepoint_count() as i32,
        _ => 0,
    };
    Value::i32(len)
}

/// compute_fn: 引用相等比较（===），比较两个 Ref 的 Arc 指针是否指向同一对象。
/// 返回 bool。两边均为 Ref 时用 Arc::ptr_eq；否则返回 false。
pub fn compute_ref_eq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let eq = match (&lhs, &rhs) {
        (Value::Ref(a), Value::Ref(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    };
    Value::bool_val(eq)
}

/// compute_fn: 引用不等比较（!==），RefEq 的否定。
pub fn compute_ref_neq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let neq = match (&lhs, &rhs) {
        (Value::Ref(a), Value::Ref(b)) => !std::sync::Arc::ptr_eq(a, b),
        _ => true,
    };
    Value::bool_val(neq)
}

/// compute_fn: 复合类型（record/adt/newtype/array/closure/throw 等）语义相等。
/// 对 Ref 走 heap_equals 深度比较；对标量/Null/Void 回退到 value_equals。
pub fn compute_eq_obj(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let eq = crate::value::ValueArena::with_global(|arena| {
        crate::value::value_equals_with_arena(&lhs, &rhs, arena)
    });
    Value::bool_val(eq)
}

/// compute_fn: 复合类型语义不等，compute_eq_obj 的否定。
pub fn compute_ne_obj(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    let neq = crate::value::ValueArena::with_global(|arena| {
        !crate::value::value_equals_with_arena(&lhs, &rhs, arena)
    });
    Value::bool_val(neq)
}

/// compute_fn: 列表拼接（ConcatList），两个 Array 拼接为新 Array。
pub fn compute_concat_list(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    let rhs = frame.get_value_by_global(inputs[1]);
    match (lhs.heap_obj(), rhs.heap_obj()) {
        (Some(HeapObj::Array(a)), Some(HeapObj::Array(b))) => {
            let mut elements = Vec::with_capacity(a.len() + b.len());
            elements.extend(a.elements.iter().cloned());
            elements.extend(b.elements.iter().cloned());
            Value::ref_val(HeapObj::Array(ArrayValue::new(elements)))
        }
        _ => Value::VOID,
    }
}

/// compute_fn: 范围生成（Range，a..b，左闭右开）。
pub fn compute_range(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Range};
    read_node_inputs!(frame, node, graph, n, inputs);
    let start = frame.get_value_by_global(inputs[0]).as_i64();
    let end = frame.get_value_by_global(inputs[1]).as_i64();
    Value::ref_val(HeapObj::Range(Range::new(start, end, false)))
}

/// compute_fn: 范围生成（RangeInclusive，a..=b，左闭右闭）。
pub fn compute_range_inclusive(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Range};
    read_node_inputs!(frame, node, graph, n, inputs);
    let start = frame.get_value_by_global(inputs[0]).as_i64();
    let end = frame.get_value_by_global(inputs[1]).as_i64();
    Value::ref_val(HeapObj::Range(Range::new(start, end, true)))
}

/// compute_fn: Elvis 运算（lhs ?: rhs）。
///
/// 统一处理 Nullable 与 Throw 两种"可能缺失值"的类型（Bug #28）：
/// - ThrowVal(Ok(v)) → 返回 v（解包成功值）
/// - ThrowVal(Err(_)) → 返回 rhs（错误时用默认值）
/// - null（Nullable）→ 返回 rhs
/// - 其他非空值 → 返回 lhs
pub fn compute_elvis(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let lhs = frame.get_value_by_global(inputs[0]);
    // Throw 类型：Ok 解包，Err 用默认值
    if let Some(crate::value::HeapObj::ThrowVal(tv)) = lhs.heap_obj() {
        return match &tv.payload {
            crate::value::ThrowPayload::Ok(v) => v.clone(),
            crate::value::ThrowPayload::Err(_) => frame.get_value_by_global(inputs[1]),
        };
    }
    // Nullable 类型：null 用默认值，非空返回 lhs
    if lhs.is_null() {
        frame.get_value_by_global(inputs[1])
    } else {
        lhs
    }
}

/// compute_fn: Call 节点启动子图（参数收集 + 标记 frame.pending_call）。
///
/// 统一 sync/async 调用路径：从 target_sg.has_suspend 推导 is_async，
/// 核心循环检测 pending_call 后据此决定是否启动子帧 + 挂起当前帧。
/// 不直接 start_subgraph（compute_fn 无 Engine 引用）。
pub fn compute_call_launch(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    let call_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));

    // safe_op 短路：?.method(args) 在接收者为 null 时返回 Null，不发起调用
    if graph.safe_op_flag(node.0 as usize) {
        let n = graph.node(node.0 as usize);
        if n.input_count > 0 {
            let inputs = graph.inputs(n.inputs_offset, n.input_count);
            let recv = frame.get_value_by_global(inputs[0]);
            if recv.is_null() {
                return NodeResult::Value(Value::Null);
            }
        }
    }

    // 静态绑定：有 call_target → 收集参数 + 返回 NodeResult::Call
    if let Some(target_sg) = graph.call_target(node.0 as usize) {
        if env_flag("KUZO_DEBUG_CALL") {
            eprintln!("[CALL] node={:?} target_sg={} frame.sg={} frame.offset={}",
                node, target_sg.0, frame.subgraph_id.0, frame.node_offset);
        }
        let is_async = graph.subgraphs[target_sg.0 as usize].has_suspend;
        let param_count = graph.subgraphs[target_sg.0 as usize].param_count as usize;
        let n = graph.node(node.0 as usize);
        let inputs = graph.inputs(n.inputs_offset, n.input_count);
        let args: Vec<Value> = inputs
            .iter()
            .take(param_count)
            .map(|&in_node| frame.get_value_by_global(in_node))
            .collect();

        return NodeResult::Call(PendingCall {
            target_sg,
            args,
            call_node_local,
            is_async,
            closure_val: None,
        });
    }

    // 动态分派：vtable_call_methods（从 TraitVal 运行时查询方法子图）
    if let Some(method_idx) = graph.vtable_call_method(node.0 as usize) {
        let n = graph.node(node.0 as usize);
        let inputs = graph.inputs(n.inputs_offset, n.input_count);
        let recv_val = frame.get_value_by_global(inputs[0]);

        let (target_sg, upvalues): (SubGraphId, Vec<Value>) = match recv_val.heap_obj() {
            Some(crate::value::HeapObj::TraitVal(tv)) => {
                let idx = method_idx as usize;
                match tv.method_values.get(idx).and_then(|v| v.heap_obj()) {
                    Some(crate::value::HeapObj::Closure(c)) => {
                        (SubGraphId(c.func_id), c.upvalues.clone())
                    }
                    _ => panic!("vtable method_idx {} is not a Closure", method_idx),
                }
            }
            _ => panic!("vtable call on non-trait value"),
        };

        let is_async = graph.subgraphs[target_sg.0 as usize].has_suspend;
        let arity = (graph.subgraphs[target_sg.0 as usize].param_count as usize)
            .saturating_sub(upvalues.len());
        let mut args: Vec<Value> = Vec::with_capacity(arity + upvalues.len());
        for &in_node in inputs.iter().skip(1).take(arity) {
            args.push(frame.get_value_by_global(in_node));
        }
        args.extend(upvalues);

        return NodeResult::Call(PendingCall {
            target_sg,
            args,
            call_node_local,
            is_async,
            closure_val: None,
        });
    }

    // 两者都无：编译器保证 Call 节点必有其一；此处不 panic，保持容错
    NodeResult::Value(Value::VOID)
}

/// compute_fn: Gate 节点选择分支 + 返回 NodeResult::Call。
pub fn compute_gate_launch(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    let branches = graph.gate_branches_at(node.0 as usize);
    let branches = branches
        .as_ref()
        .expect("Gate node has no branches");

    // 读条件值
    let cond_raw = frame.get_value_by_global(branches.condition_input);
    let cond = cond_raw.as_bool();

    if env_flag("KUZO_DEBUG_GATE") {
        let sg = &graph.subgraphs[frame.subgraph_id.0 as usize];
        eprintln!("[GATE] node={:?} cond_raw={:?} cond={} frame.sg={} frame.offset={} sg.range=[{},{}) branches={:?}",
            node, cond_raw, cond, frame.subgraph_id.0, frame.node_offset,
            sg.node_range.0 .0, sg.node_range.1 .0,
            branches.branches.iter().map(|(c, sg, _)| (*c, sg.0)).collect::<Vec<_>>());
    }

    // 选分支
    let (target_sg, branch_inputs) = branches
        .branches
        .iter()
        .find(|(c, _, _)| *c == cond)
        .map(|(_, sg, inputs)| (*sg, inputs.clone()))
        .expect("no matching gate branch");

    // 收集参数
    let param_count = graph.subgraphs[target_sg.0 as usize].param_count as usize;
    let args: Vec<Value> = branch_inputs
        .iter()
        .take(param_count)
        .map(|&n| frame.get_value_by_global(n))
        .collect();

    if env_flag("KUZO_DEBUG_STALL") {
        let (ns, ne) = graph.subgraphs[target_sg.0 as usize].node_range;
        eprintln!("[GATE] node={} cond={} target_sg={} sg_range=[{},{}) params={} branch_inputs={:?} args={}",
            node.0, cond, target_sg.0, ns.0, ne.0, param_count, branch_inputs, args.len());
        for gid in ns.0..ne.0 {
            let n = graph.node(gid as usize);
            let cv = graph.const_value(gid as usize);
            eprintln!("  [GATE-NODE] gid={} kind={:?} cf={} const_values={:?} inputs_count={}",
                gid, n.kind, n.compute_fn.0, cv.is_some(), n.input_count);
        }
    }

    let gate_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));

    NodeResult::Call(PendingCall {
        target_sg,
        args,
        call_node_local: gate_node_local,
        is_async: false,
        closure_val: None,
    })
}

/// compute_await（idx 38）：await 节点返回 NodeResult::Await。
///
/// spec 4.4：事件源未就绪 → await 未就绪 → 帧无更多就绪节点 → 挂起。
/// 核心循环收到 NodeResult::Await 后解析事件源 → 检查就绪 → 就绪则注入值继续 → 未就绪则挂起。
pub fn compute_await(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    use crate::ir::Ir::PendingAwait;

    read_node_inputs!(frame, node, graph, n, inputs);
    // inputs[0] = 事件对象节点（AsyncHandle/Channel/Timer）
    let event_obj = frame.get_value_by_global(inputs[0]);
    let await_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));

    // EventSource 节点从 await_event_sources 表读取（元数据引用，非数据依赖）
    let es_node = graph.await_event_source(node.0 as usize);
    let event_kind = match es_node {
        Some(es) => graph
            .subgraphs
            .get(frame.subgraph_id.0 as usize)
            .and_then(|sg| {
                sg.event_source_decls
                    .iter()
                    .find(|d| d.node == es)
                    .map(|d| d.kind)
            })
            .unwrap_or(crate::ir::Ir::EventSourceKind::AsyncJoin),
        None => crate::ir::Ir::EventSourceKind::AsyncJoin,
    };

    NodeResult::Await(PendingAwait {
        await_node_local,
        event_obj,
        event_kind,
    })
}

/// compute_channel_create（idx 283）：创建 ChannelValue 堆对象。
///
/// 输入：inputs[0] = capacity (usize)
/// 输出：Value::ref_val(HeapObj::ChannelVal(Arc<ChannelValue>))
pub fn compute_channel_create(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let capacity = frame.get_value_by_global(inputs[0]).as_usize();
    Value::ref_val(crate::value::HeapObj::ChannelVal(
        std::sync::Arc::new(crate::value::ChannelValue::new(capacity)),
    ))
}

/// compute_channel_send（idx 284）：非阻塞发送 + 返回 NodeResult::ChannelNotify。
///
/// 输入：inputs[0] = channel ref, inputs[1] = value
/// 发送后返回 NodeResult::ChannelNotify，核心循环消费时触发 ChannelReady 事件
/// 唤醒等待该 channel 的挂起帧（内联触发，零延迟）。
pub fn compute_channel_send(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    // safe_op 短路：?.send(v) 在接收者为 null 时返回 Null
    if graph.safe_op_flag(node.0 as usize) {
        let ch_val = frame.get_value_by_global(inputs[0]);
        if ch_val.is_null() {
            return NodeResult::Value(Value::Null);
        }
    }
    let ch_val = frame.get_value_by_global(inputs[0]);
    let val = frame.get_value_by_global(inputs[1]);
    let make_err = |msg: &str| make_error_throw("ChannelError", msg);
    let ch = match ch_val.heap_obj().and_then(|h| h.channel()) {
        Some(ch) => ch,
        None => return NodeResult::Value(make_err("send on non-channel value")),
    };
    match ch.send(val) {
        Ok(()) => {
            let ch_id = crate::ir::Ir::ChannelId(ch.id());
            NodeResult::ChannelNotify(ch_id)
        }
        Err(e) => NodeResult::Value(make_err(e.message())),
    }
}

/// compute_channel_close（idx 285）：关闭 channel。
///
/// 输入：inputs[0] = channel ref
pub fn compute_channel_close(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    let ch_val = frame.get_value_by_global(inputs[0]);
    let ch = ch_val.heap_obj().and_then(|h| h.channel())
        .expect("close on non-channel value");
    ch.close();
    Value::VOID
}

/// compute_fn: 闭包构造（idx 40）。
///
/// 从 graph.closure_infos 取子图 id + arity，合并 inputs（捕获值）构造 Closure 堆对象。
/// 节点的 inputs 即捕获的 upvalues（按 compile_lambda 中 captured 顺序）。
pub fn compute_closure_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Closure, Cell};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.closure_info(node.0 as usize)
        .expect("closure construct node has no ClosureInfo");
    // 用 Cell 包装每个 upvalue，使逃逸闭包（跨函数调用）能通过 Cell
    // 的 interior mutability 持久化 upvalue 修改。
    // same_function 调用不使用 Cell（直接从父帧读最新值）。
    let upvalues: Vec<Value> = inputs
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .map(|v| Value::ref_val(HeapObj::Cell(Cell::new(v))))
        .collect();
    let cell_bits = if upvalues.len() >= 8 { 0xFF } else { (1u8 << upvalues.len()) - 1 };
    Value::ref_val(HeapObj::Closure(Closure {
        func_id: info.subgraph_id.0,
        arity: info.arity,
        upvalues,
        bound_args: Vec::new(),
        self_upvalue_idx: info.self_upvalue_idx,
        upvalue_ref_bits: 0,
        cell_upvalues: cell_bits,
    }))
}

/// compute_fn: inline_trait 构造（idx 266）。
///
/// 从 graph.trait_construct_infos 取 trait 名 + 方法列表，
/// 合并节点 inputs（各方法 upvalues 依次拼接）构造多个 Closure，
/// 打包成 TraitValue 堆对象。
pub fn compute_trait_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, Closure, TraitValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.trait_construct_info_at(node.0 as usize);
    let info = info
        .as_ref()
        .expect("trait construct node has no TraitConstructInfo");

    // 从 inputs 按各方法 upvalue_count 依次切分，构造每个方法的 Closure
    let mut method_values: Vec<Value> = Vec::with_capacity(info.methods.len());
    let mut input_cursor = 0usize;
    for m in &info.methods {
        let upvalue_count = m.upvalue_count as usize;
        let upvalues: Vec<Value> = inputs[input_cursor..input_cursor + upvalue_count]
            .iter()
            .map(|&in_node| frame.get_value_by_global(in_node))
            .collect();
        input_cursor += upvalue_count;
        method_values.push(Value::ref_val(HeapObj::Closure(Closure {
            func_id: m.subgraph_id.0,
            arity: m.arity,
            upvalues,
            bound_args: Vec::new(),
            self_upvalue_idx: -1,
            upvalue_ref_bits: 0,
            cell_upvalues: 0,
        })));
    }

    Value::ref_val(HeapObj::TraitVal(TraitValue {
        trait_name: info.trait_name.clone(),
        method_names: info.method_names.clone(),
        method_values,
        data: None,
        owned: true,
    }))
}

/// compute_fn: lazy 构造（idx 267）。
///
/// 从 graph.lazy_construct_infos 取 thunk 子图 id，
/// 合并节点 inputs（upvalues）构造 LazyValue 堆对象。
/// thunk 未求值，首次 force 时启动子图计算并缓存结果。
pub fn compute_lazy_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, LazyValue, Closure};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.lazy_construct_info(node.0 as usize)
        .expect("lazy construct node has no LazyConstructInfo");

    // upvalues 从 inputs 收集，存入 Closure（thunk 首次 force 时用）
    let upvalues: Vec<Value> = inputs
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();

    // 用 Closure 包装 thunk 子图（func_id = thunk_sg），存为 LazyValue.data
    // force 时从 data 取 Closure，启动子图计算，结果缓存到 cached
    let thunk_closure = Value::ref_val(HeapObj::Closure(Closure {
        func_id: info.thunk_sg.0,
        arity: 0,
        upvalues,
        bound_args: Vec::new(),
        self_upvalue_idx: -1,
        upvalue_ref_bits: 0,
        cell_upvalues: 0,
    }));

    Value::ref_val(HeapObj::LazyVal(LazyValue {
        cached: std::sync::Mutex::new(None),
        forced: std::sync::atomic::AtomicBool::new(false),
        data: Some(thunk_closure),
    }))
}

// =========================================================================
// LazyValue force 机制：同步执行 thunk 子图，缓存结果
// =========================================================================

/// 强制求值 LazyValue：同步执行 thunk 子图，返回计算结果。
///
/// 若已 forced，直接返回 cached 值；否则创建 thunk 帧，同步运行至完成，
/// 将结果缓存到 LazyValue（通过 Arc::make_mut 原地更新），返回结果。
///
/// 此函数在 compute_reflect_format / compute_reflect_scalar_to_str 中调用，
/// 用于在格式化前强制求值 lazy 值。
pub fn force_lazy_value_sync(caller_frame: &mut Frame, lazy_val: &Value) -> Value {
    use crate::value::HeapObj;

    // 提取 LazyValue 引用
    let arc = match lazy_val {
        Value::Ref(r) => r,
        _ => return lazy_val.clone(), // 非 LazyValue，直接返回
    };

    // 检查是否已 forced
    {
        if let HeapObj::LazyVal(lazy) = &**arc {
            if lazy.forced.load(std::sync::atomic::Ordering::Relaxed) {
                return lazy.cached.lock().unwrap().clone().unwrap_or(Value::NULL);
            }
        } else {
            return lazy_val.clone(); // 非 LazyVal，直接返回
        }
    }

    // 取 thunk Closure
    let closure = {
        let HeapObj::LazyVal(lazy) = &**arc else { return lazy_val.clone() };
        match &lazy.data {
            Some(v) => match v.heap_obj() {
                Some(HeapObj::Closure(c)) => c.clone(),
                _ => return Value::NULL,
            },
            None => return Value::NULL,
        }
    };

    let graph = caller_frame.graph.clone();
    let thunk_sg = SubGraphId(closure.func_id);

    // 创建 thunk 帧
    let (node_start, node_end) = graph.subgraphs[thunk_sg.0 as usize].node_range;
    let node_count = (node_end.0 - node_start.0) as usize;
    let mut thunk_frame = Frame::new(THUNK_FRAME_ID, thunk_sg, node_count, graph.clone());
    prepare_frame_nodes(&mut thunk_frame, &graph);

    // 注入 upvalues 作为参数
    let offset = node_start.0 as usize;
    let param_count = graph.subgraphs[thunk_sg.0 as usize].param_count as usize;
    for (i, arg) in closure.upvalues.iter().enumerate().take(param_count) {
        let local_id = NodeId(i as u32);
        let consumer_count = graph.downstream_slice(offset + i).len() as u16;
        thunk_frame.set_value(local_id, arg.clone(), consumer_count);
        thunk_frame.push_ready(local_id);
    }

    // thunk 帧的 upvalues 已作为参数注入（上方循环），不需通过 parent_frame_ptr 访问外层变量。
    // 设为 null 避免 caller_frame 的 &mut 借用与裸指针解引用构成别名 UB。
    thunk_frame.parent_frame_ptr = std::ptr::null_mut();

    // 同步执行 thunk 帧
    let result = run_frame_sync(&mut thunk_frame, &graph);

    // 缓存结果到 LazyValue（通过 Mutex/AtomicBool 的 interior mutability 更新）
    if let HeapObj::LazyVal(lazy) = &**arc {
        lazy.forced.store(true, std::sync::atomic::Ordering::Relaxed);
        *lazy.cached.lock().unwrap() = Some(result.clone());
    }

    result
}

/// 同步路径循环迭代重置：LoopBody 完成 Continue/None 后重置循环帧的
/// cond/gate/iter_next，使其重新进入下一迭代。
///
/// 与 Engine::reset_loop_iteration 对应，但不处理 body_frame 复用
/// （同步路径每次迭代都新建 child_frame，不复用）。
/// 同步路径不通过帧队列驱动，而是 run_frame_sync_inner 主循环直接从
/// ready_queue pop 节点执行，因此 reset 后 cond/iter_next 入队即可
/// 被主循环重新拾取执行。
fn reset_loop_frame_for_next_iteration(frame: &mut Frame, graph: &DataFlowGraph) {
    let loop_sg_id = frame.subgraph_id;
    let (loop_kind, cond_node, return_node, iter_next_node) = {
        let sg = &graph.subgraphs[loop_sg_id.0 as usize];
        (sg.loop_kind, sg.cond_node, sg.return_node, sg.iter_next_node)
    };
    let loop_offset = frame.node_offset;

    // 0. 清空 ready_queue（必须在 push cond/iter_next 之前）
    // 若不清空，旧就绪条目残留，会先于 cond/iter_next 执行，引用过时值
    frame.ready_queue.clear();

    // 1. For 循环：重置 iter_next_node
    if loop_kind == LoopKind::For {
        if let Some(next_node) = iter_next_node {
            let next_local = NodeId(next_node.0.wrapping_sub(loop_offset));
            let i = next_local.0 as usize;
            if i < frame.pending_inputs.len() {
                frame.pending_inputs[i] = 0;
            }
            if i < frame.value_table.len() {
                frame.value_table.reset_slot(i);
            }
            frame.push_ready(next_local);
        }
    }

    // 2. 重置 cond_node
    if let Some(cond_node) = cond_node {
        let cond_local = NodeId(cond_node.0.wrapping_sub(loop_offset));
        let i = cond_local.0 as usize;
        if loop_kind == LoopKind::For {
            // For 循环 cond 依赖 iter_next，pending=1
            if i < frame.pending_inputs.len() {
                frame.pending_inputs[i] = 1;
            }
            if i < frame.value_table.len() {
                frame.value_table.reset_slot(i);
            }
        } else {
            // While/Loop cond 无输入依赖，pending=0
            if i < frame.pending_inputs.len() {
                frame.pending_inputs[i] = 0;
            }
            if i < frame.value_table.len() {
                frame.value_table.reset_slot(i);
            }
            // Const cond_node 重新预填充
            if graph.node(cond_node.0 as usize).kind == NodeKind::Const {
                if let Some(cv) = graph.const_value(cond_node.0 as usize) {
                    let handle = cv.to_value(graph.string_pool_slice());
                    let consumer_count =
                        graph.downstream_slice(cond_node.0 as usize).len() as u16;
                    frame.set_value(cond_local, handle, consumer_count);
                }
            }
            frame.push_ready(cond_local);
        }
    }

    // 3. 重置 Gate 节点（= return_node，pending=1，等 cond notify）
    let gate_local = NodeId(return_node.0.wrapping_sub(loop_offset));
    let gi = gate_local.0 as usize;
    if gi < frame.pending_inputs.len() {
        frame.pending_inputs[gi] = 1;
    }
    if gi < frame.value_table.len() {
        frame.value_table.reset_slot(gi);
    }

    // 4. 重置循环帧状态
    frame.control_signal = ControlSignal::None;
    frame.state = FrameState::Ready;
}

/// 同步执行帧至完成，处理嵌套函数调用、控制信号、vtable 分派。
///
/// 这是 Engine 异步执行模型的同步简化版：
/// - 帧内节点按就绪队列调度执行
/// - 遇到 Call 节点时递归调用 run_frame_sync 执行子帧
/// - 控制信号（return/break/continue）终止循环
///
/// defer 执行：帧完成后（任何终止路径），按 LIFO 顺序执行 defer_table 中的
/// defer body 子图。defer body 通过递归 run_frame_sync 执行（支持嵌套 defer）。
fn run_frame_sync(frame: &mut Frame, graph: &DataFlowGraph) -> Value {
    let result = run_frame_sync_inner(frame, graph);
    // 执行 defer（LIFO）：任何终止路径都执行 defer
    run_defers_sync(frame, graph);
    result
}

/// 执行帧的 defer_table 中的 defer body（LIFO 顺序）。
/// defer body 是独立子图，创建新帧并通过 run_frame_sync 同步执行。
fn run_defers_sync(frame: &mut Frame, graph: &DataFlowGraph) {
    let sg_id = frame.subgraph_id;
    let defer_entries: Vec<crate::ir::Ir::DeferEntry> =
        graph.subgraphs[sg_id.0 as usize].defer_table.clone();
    for entry in defer_entries.iter().rev() {
        let (dn_start, dn_end) = graph.subgraphs[entry.body_subgraph.0 as usize].node_range;
        let dn_count = (dn_end.0 - dn_start.0) as usize;
        let mut defer_frame = Frame::new(
            FrameId(u32::MAX),
            entry.body_subgraph,
            dn_count,
            frame.graph.clone(),
        );
        prepare_frame_nodes(&mut defer_frame, graph);
        let _ = run_frame_sync(&mut defer_frame, graph);
    }
}

/// run_frame_sync 的内部实现（不执行 defer）。
///
/// 统一热循环：pop → compute_fn → match NodeResult
/// - Call: 递归创建子帧 + 同步执行 + 注入返回值
/// - Return/Break/Continue: 设置 control_signal 终止循环
/// - Await/ChannelNotify/Cancel/SelectWait: 同步路径不支持，返回 NULL
///
/// 不支持：async/await、channel/timer 事件、select、循环体复用。
/// 适用于 thunk 子图（纯计算 + 同步函数调用）。
fn run_frame_sync_inner(frame: &mut Frame, graph: &DataFlowGraph) -> Value {
    use crate::ir::Ir::{ControlSignal, LoopKind, NodeKind};

    let mut iter_guard: u64 = 0;
    loop {
        iter_guard += 1;
        if iter_guard > 100000 {
            return Value::NULL;
        }
        // 1. 检查控制信号
        let cs = frame.control_signal.clone();
        match cs {
            ControlSignal::Return(v) => return v,
            ControlSignal::Break | ControlSignal::Continue => return Value::VOID,
            ControlSignal::None => {}
        }

        // 2. POP
        let local_id = match frame.pop_ready() {
            Some(n) => n,
            None => {
                let sg = &graph.subgraphs[frame.subgraph_id.0 as usize];
                let return_local = sg.return_node.0.wrapping_sub(frame.node_offset);
                if (return_local as usize) < frame.value_table.len()
                    && !frame.value_table.is_ready(return_local as usize)
                {
                    return Value::NULL;
                }
                return frame.get_value_by_global(sg.return_node);
            }
        };

        let node_start = frame.node_offset;
        let graph_node_id = NodeId(local_id.0 + node_start);
        let node = graph.node(graph_node_id.0 as usize);
        let ctx = EvalContext { node_start };

        // 3. COMPUTE
        let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, graph_node_id, &ctx);

        // 4. MATCH NodeResult
        match result {
            NodeResult::Value(v) => {
                let cc = graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
                frame.set_value(local_id, v, cc);
                notify_downstream(frame, graph, local_id, graph_node_id, NodeId(node_start));
            }
            NodeResult::Batch(results) => {
                for &(lid, ref v) in &results {
                    let gid = NodeId(lid.0 + node_start);
                    let cc = graph.downstream_slice(gid.0 as usize).len() as u16;
                    frame.set_value(lid, v.clone(), cc);
                }
                for &(lid, _) in &results {
                    frame.ready_queue.retain(|n| *n != lid);
                }
                for &(lid, _) in &results {
                    let gid = NodeId(lid.0 + node_start);
                    notify_downstream(frame, graph, lid, gid, NodeId(node_start));
                }
            }
            NodeResult::Call(pending) => {
                // 尾调用：复用当前帧
                if graph.tail_call_flag(graph_node_id.0 as usize) {
                    switch_subgraph(frame, graph, pending.target_sg, &pending.args);
                    continue;
                }

                let target_loop_kind = graph.subgraphs[pending.target_sg.0 as usize].loop_kind;

                // LoopBody：不支持循环体复用（thunk 不应有循环），回退为普通调用
                let (child_start, child_end) = graph.subgraphs[pending.target_sg.0 as usize].node_range;
                let child_count = (child_end.0 - child_start.0) as usize;
                let mut child_frame = Frame::new(
                    LOOPBODY_FALLBACK_FRAME_ID,
                    pending.target_sg,
                    child_count,
                    frame.graph.clone(),
                );
                prepare_frame_nodes(&mut child_frame, graph);

                // 注入参数
                let child_offset = child_start.0 as usize;
                let child_param_count = graph.subgraphs[pending.target_sg.0 as usize].param_count as usize;
                for (i, arg) in pending.args.iter().enumerate().take(child_param_count) {
                    let lid = NodeId(i as u32);
                    let cc = graph.downstream_slice(child_offset + i).len() as u16;
                    child_frame.set_value(lid, arg.clone(), cc);
                    child_frame.push_ready(lid);
                }

                // 设置帧链指针
                let same_function = graph.subgraphs[frame.subgraph_id.0 as usize].function_id
                    == graph.subgraphs[pending.target_sg.0 as usize].function_id;
                child_frame.parent_frame_ptr = if same_function {
                    frame as *mut Frame
                } else {
                    std::ptr::null_mut()
                };
                child_frame.root_frame_ptr = if same_function {
                    if frame.root_frame_ptr.is_null() {
                        frame as *mut Frame
                    } else {
                        frame.root_frame_ptr
                    }
                } else {
                    std::ptr::null_mut()
                };
                child_frame.closure_val = pending.closure_val.clone();

                // 同步执行子帧
                let child_result = run_frame_sync(&mut child_frame, graph);
                let child_signal = child_frame.control_signal.clone();

                // 注入返回值到当前帧
                let consumer_count = graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
                frame.set_value(pending.call_node_local, child_result.clone(), consumer_count);

                // throw 传播
                let is_throw_err = matches!(
                    child_result.heap_obj(),
                    Some(crate::value::HeapObj::ThrowVal(t)) if matches!(t.payload, crate::value::ThrowPayload::Err(_))
                );
                if is_throw_err {
                    frame.control_signal = ControlSignal::Return(child_result);
                    continue;
                }

                // Gate 分支控制信号传播
                let is_gate = graph.node(graph_node_id.0 as usize).kind == NodeKind::Gate;
                if is_gate && !matches!(child_signal, ControlSignal::None) {
                    frame.control_signal = child_signal;
                    continue;
                }

                // LoopBody 完成处理
                if target_loop_kind == LoopKind::LoopBody {
                    match child_signal {
                        ControlSignal::Break | ControlSignal::Return(_) => {
                            frame.control_signal = child_signal;
                            continue;
                        }
                        ControlSignal::Continue => {
                            reset_loop_frame_for_next_iteration(frame, graph);
                            continue;
                        }
                        ControlSignal::None => {
                            let loop_kind = graph.subgraphs[frame.subgraph_id.0 as usize].loop_kind;
                            if loop_kind == LoopKind::TailRec {
                                frame.control_signal = ControlSignal::Return(child_result);
                                continue;
                            } else {
                                reset_loop_frame_for_next_iteration(frame, graph);
                                continue;
                            }
                        }
                    }
                }

                notify_downstream(frame, graph, pending.call_node_local, graph_node_id, NodeId(node_start));
            }
            NodeResult::Return(v) => {
                frame.control_signal = ControlSignal::Return(v);
                continue;
            }
            NodeResult::Break => {
                frame.control_signal = ControlSignal::Break;
                continue;
            }
            NodeResult::Continue => {
                frame.control_signal = ControlSignal::Continue;
                continue;
            }
            // 同步路径不支持：async/await、channel/timer、select
            NodeResult::Await(_)
            | NodeResult::ChannelNotify(_)
            | NodeResult::Cancel(_)
            | NodeResult::SelectWait(_) => {
                return Value::NULL;
            }
        }
    }
}

/// compute_fn: 偏应用构造（idx 286）。
///
/// 从 partial_infos 取子图 id + bound_count，合并 inputs（已绑定参数值）
/// 构造 HeapObj::Partial。remaining_arity = subgraph.param_count - bound_count。
/// 顶层函数偏应用时 upvalues 为空，self_upvalue_idx = -1。
pub fn compute_partial_construct(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, PartialApplication};
    read_node_inputs!(frame, node, graph, n, inputs);
    let info = graph.partial_info(node.0 as usize)
        .expect("partial construct node has no PartialInfo");
    let bound_args: Vec<Value> = inputs
        .iter()
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();
    let param_count = graph.subgraphs[info.subgraph_id.0 as usize].param_count as usize;
    let remaining_arity = param_count.saturating_sub(bound_args.len()) as u8;
    Value::ref_val(HeapObj::Partial(PartialApplication {
        func_id: info.subgraph_id.0,
        upvalues: Vec::new(),
        bound_args,
        remaining_arity,
        self_upvalue_idx: -1,
    }))
}

/// compute_str_bytes（idx 287）：str.bytes() → u8[]
/// 将 KuzoStr 的 UTF-8 字节序列构造为 u8 数组。
pub fn compute_str_bytes(frame: &mut Frame, node: NodeId) -> Value {
    use crate::value::{HeapObj, ArrayValue};
    read_node_inputs!(frame, node, graph, n, inputs);
    let val = frame.get_value_by_global(inputs[0]);
    let bytes: Vec<Value> = match val.heap_obj() {
        Some(HeapObj::Str(s)) => s.bytes().as_bytes()
            .iter()
            .map(|&b| Value::u8(b))
            .collect(),
        _ => Vec::new(),
    };
    Value::ref_val(HeapObj::Array(ArrayValue::new(bytes)))
}

/// compute_fn: 可调用值调用（idx 41）— 统一处理 Closure | Partial。
///
/// inputs[0] = 可调用值节点，inputs[1..1+arg_count] = 调用参数节点（arg_count 从
/// closure_call_arg_counts 元数据读取，不含闭包值和 effect 依赖）。
///
/// 统一调用语义：
/// - Closure: needed_arity = subgraph.param_count - upvalues.len()
/// - Partial: needed_arity = remaining_arity
///
/// 当新参数数 < needed_arity → 产出新的 Partial（链式偏应用）；
/// 当新参数数 >= needed_arity → 合并 bound_args + 新参数 + upvalues，设 pending_call。
/// 解包 Cell 包装的 upvalue：若值是 Cell 则返回内部值的克隆，否则原样克隆。
/// 用于 compute_closure_call 将 Cell upvalues 转为原始值注入子帧参数。
fn unwrap_cell(v: &Value) -> Value {
    match v.heap_obj() {
        Some(crate::value::HeapObj::Cell(cell)) => cell.get(),
        _ => v.clone(),
    }
}

pub fn compute_closure_call(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    use crate::value::{HeapObj, PartialApplication};
    read_node_inputs!(frame, node, graph, n, inputs);
    let callable_val = frame.get_value_by_global(inputs[0]);
    // safe_op 短路：?.method(args) 在接收者为 null 时返回 Null
    if graph.safe_op_flag(node.0 as usize) && callable_val.is_null() {
        return NodeResult::Value(Value::Null);
    }

    // 从元数据读取实参数（不含闭包值和 effect 依赖）
    let arg_count = graph.closure_call_arg_count(node.0 as usize)
        .expect("closure_call node has no arg_count") as usize;
    let new_args: Vec<Value> = inputs
        .iter()
        .skip(1)
        .take(arg_count)
        .map(|&in_node| frame.get_value_by_global(in_node))
        .collect();

    // 统一提取可调用值的启动信息
    let (func_id, upvalues, bound_args, needed_arity, self_upvalue_idx) = match callable_val.heap_obj() {
        Some(HeapObj::Closure(c)) => {
            let total_params = graph.subgraphs[c.func_id as usize].param_count as usize;
            let needed = total_params.saturating_sub(c.upvalues.len());
            // 解包 Cell upvalues 为原始值注入参数（Cell 用于逃逸闭包的持久化回写）
            let upvalues: Vec<Value> = c.upvalues.iter().map(|v| unwrap_cell(v)).collect();
            (c.func_id, upvalues, Vec::new(), needed, c.self_upvalue_idx)
        }
        Some(HeapObj::Partial(p)) => {
            let upvalues: Vec<Value> = p.upvalues.iter().map(|v| unwrap_cell(v)).collect();
            (p.func_id, upvalues, p.bound_args.clone(), p.remaining_arity as usize, p.self_upvalue_idx)
        }
        _ => panic!("compute_closure_call: input is not callable (Closure or Partial)"),
    };

    // 链式偏应用：新参数不足 → 产出新 Partial
    if new_args.len() < needed_arity {
        let provided = new_args.len();
        let mut extended = bound_args;
        extended.extend(new_args);
        let new_remaining = needed_arity - provided;
        return NodeResult::Value(Value::ref_val(HeapObj::Partial(PartialApplication {
            func_id,
            upvalues,
            bound_args: extended,
            remaining_arity: new_remaining as u8,
            self_upvalue_idx,
        })));
    }

    // 满 arity：合并 bound_args + new_args[..needed] + upvalues，返回 NodeResult::Call
    let target_sg = SubGraphId(func_id);
    let call_node_local = NodeId(node.0.wrapping_sub(frame.node_offset));
    let upvalues_len = upvalues.len();
    let mut args: Vec<Value> = Vec::with_capacity(bound_args.len() + needed_arity + upvalues_len);
    args.extend(bound_args);
    args.extend(new_args.iter().take(needed_arity).cloned());
    args.extend(upvalues);

    // 递归闭包：将自身引用注入到 self_upvalue_idx 对应的 upvalue slot
    // 边界检查：防止 usize 下溢与数组越界（self_upvalue_idx 必须落在 upvalues 区间内）
    if self_upvalue_idx >= 0 {
        assert!(upvalues_len <= args.len(), "upvalues_len exceeds args.len()");
        let self_upvalue_idx = self_upvalue_idx as usize;
        assert!(self_upvalue_idx < upvalues_len, "self_upvalue_idx out of bounds");
        let upvalues_start = args.len() - upvalues_len;
        let self_idx = upvalues_start + self_upvalue_idx;
        assert!(self_idx < args.len(), "self_idx out of bounds");
        args[self_idx] = callable_val.clone();
    }

    NodeResult::Call(PendingCall {
        target_sg,
        args,
        call_node_local,
        is_async: false,
        closure_val: Some(callable_val.clone()),
    })
}

/// compute_fn: 取消 async handle 对应的子帧。
///
/// inputs[0] = async handle 值（i32 标量，值为 async_id）。
/// 返回 NodeResult::Cancel，核心循环从 AsyncJoinRuntime 查 async_id → child_fid 执行取消。
pub fn compute_cancel_async_handle(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    let handle_val = frame.get_value_by_global(inputs[0]);
    // safe_op 短路：?.cancel() 在接收者为 null 时返回 Null
    if graph.safe_op_flag(node.0 as usize) && handle_val.is_null() {
        return NodeResult::Value(Value::Null);
    }
    // async handle 是 i32 标量，值即 async_id
    let async_id = crate::ir::Ir::AsyncHandleId(handle_val.as_i32() as u32);
    NodeResult::Cancel(async_id)
}

/// compute_fn: select 门控节点（idx 43）— 返回 NodeResult::SelectWait。
///
/// 核心循环收到后检查所有分支事件源的就绪状态（能访问 Engine 全部状态）。
pub fn compute_select_gate(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    // 校验 gate 节点确实绑定了 SelectInfo
    let info = graph.select_info_at(node.0 as usize);
    let _ = info
        .as_ref()
        .expect("select gate node has no SelectInfo");
    let gate_local = NodeId(node.0.wrapping_sub(frame.node_offset));
    NodeResult::SelectWait(gate_local)
}


/// noop compute_fn（匹配真实签名）。
pub fn noop_compute_real(_frame: &mut Frame, _node: NodeId) -> Value {
    Value::VOID
}

/// compute_fn for Const nodes (新签名，不通过 wrap_fn! 包装)。
/// 从 const_values 表物化值并返回。
/// 非 Const 节点（也使用 CF_NOOP）返回 Value::VOID（兼容 noop_compute_real）。
pub fn compute_const(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    if let Some(cv) = frame.graph.const_value(node.0 as usize) {
        NodeResult::Value(crate::engine::alloc_const_value(cv, frame.graph.string_pool_slice()))
    } else {
        NodeResult::Value(Value::VOID)
    }
}

/// compute_return (idx 311): 提取输入值并返回 NodeResult::Return。
///
/// inputs[0] = 返回值。可选的 inputs[1] = 前序副作用依赖（仅用于就绪判定，值被忽略）。
/// 替代旧的 control_signal_nodes[SignalKind::Return] 表检查。
pub fn compute_return(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    read_node_inputs!(frame, node, graph, n, inputs);
    let v = frame.get_value_by_global(inputs[0]);
    NodeResult::Return(v)
}

/// compute_break (idx 312): 返回 NodeResult::Break。
///
/// 可选 inputs[0] = 前序副作用依赖（仅用于就绪判定，值被忽略）。
/// 替代旧的 control_signal_nodes[SignalKind::Break] 表检查。
pub fn compute_break(_frame: &mut Frame, _node: NodeId, _ctx: &EvalContext) -> NodeResult {
    NodeResult::Break
}

/// compute_continue (idx 313): 返回 NodeResult::Continue。
///
/// 可选 inputs[0] = 前序副作用依赖（仅用于就绪判定，值被忽略）。
/// 替代旧的 control_signal_nodes[SignalKind::Continue] 表检查。
pub fn compute_continue(_frame: &mut Frame, _node: NodeId, _ctx: &EvalContext) -> NodeResult {
    NodeResult::Continue
}

/// compute_fn (idx 48): 序列节点 — 等待所有输入就绪后返回最后一个输入的值。
///
/// 用于语句顺序链接：inputs = [prev_effect, current_value]，返回 current_value。
/// prev_effect 仅作数据依赖边（顺序约束），确保前一个语句完成后才执行当前语句。
pub fn compute_seq(frame: &mut Frame, node: NodeId) -> Value {
    read_node_inputs!(frame, node, graph, n, inputs);
    if n.input_count == 0 {
        return Value::VOID;
    }
    let last_input = inputs[n.input_count as usize - 1];
    frame.get_value_by_global(last_input)
}

/// compute_writeback（idx 49）：赋值外层变量，通过 root_frame_ptr 写回函数根帧。
///
/// inputs[0] = 值来源（当前帧内节点），writeback_targets[node] = 外层全局 NodeId。
/// 非阻塞：compute_fn 内直接完成写入，无 pending、无 Engine 层消费。
///
/// 三条回写路径（按优先级）：
/// 1. parent_frame_ptr 链：同函数闭包调用，写入最近的包含 target 的父帧
/// 2. root_frame_ptr：同函数闭包调用，写入函数根帧（使其他 same_function 调用可见）
/// 3. closure_val Cell：逃逸闭包（跨函数调用，帧链为 null），通过 Cell 的 interior
///    mutability 更新闭包 upvalues，使下次调用能读到最新值
pub fn compute_writeback(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    let graph = frame.graph.clone();
    let n = graph.node(node.0 as usize);
    if n.input_count == 0 {
        return NodeResult::Value(Value::VOID);
    }
    let val_node = graph.inputs(n.inputs_offset, n.input_count)[0];
    let val = frame.get_value_by_global(val_node);
    let target = graph.writeback_target(node.0 as usize)
        .expect("WriteBack node missing target");
    let consumer_count = graph.downstream_slice(target.0 as usize).len() as u16;

    if env_flag("KUZO_DEBUG_WB") {
        let sg = &graph.subgraphs[frame.subgraph_id.0 as usize];
        eprintln!("[WB] node={:?} target={:?} val={:?} val_node={:?} frame.sg={} frame.offset={} sg.range=[{},{}) sg.func_id={} vt_len={}",
            node, target, val, val_node, frame.subgraph_id.0, frame.node_offset,
            sg.node_range.0 .0, sg.node_range.1 .0, sg.function_id, frame.value_table.len());
    }

    // 路径 0：写入当前帧（same_function 闭包调用场景）。
    // same_function 帧的值表扩展到父帧大小，target 可能在当前帧范围内。
    // 若不写当前帧：a() 修改 log 后 WriteBack 只写父帧链（main 帧），
    // a 子帧自身的 log 仍为旧值；后续 b() 从 a 子帧（parent_frame）读取
    // upvalue 时得到陈旧值，导致闭包链共享可变捕获失效（Bug #31）。
    let cur_local = target.0.wrapping_sub(frame.node_offset);
    if (cur_local as usize) < frame.value_table.len() {
        frame.set_value(NodeId(cur_local), val.clone(), consumer_count);
    }

    // 路径 1：遍历 parent_frame_ptr 链，写入所有包含 target 的祖先帧。
    // 不能只写最近父帧就 break：嵌套 same_function 子图（如 if 分支 → 循环体 →
    // 循环帧 → main）中，中间帧（循环帧）也需要更新，否则下一迭代的 body
    // 从循环帧拷贝时会读到旧值。
    // SAFETY: parent_frame_ptr 指向同函数帧（setup_frame_chain 设置），
    // caller 帧在 callee 执行期间处于 Suspended 状态，无并发访问。
    let mut written_parent = false;
    let mut ptr = frame.parent_frame_ptr;
    while !ptr.is_null() {
        let f = unsafe { &mut *ptr };
        let local = target.0.wrapping_sub(f.node_offset);
        if (local as usize) < f.value_table.len() {
            f.set_value(NodeId(local), val.clone(), consumer_count);
            written_parent = true;
        }
        ptr = f.parent_frame_ptr;
    }

    // 路径 2：写入 root_frame_ptr（函数根帧），使同函数闭包调用能从根帧读到最新值。
    if !frame.root_frame_ptr.is_null() {
        let root = unsafe { &mut *frame.root_frame_ptr };
        let local = target.0.wrapping_sub(root.node_offset);
        if (local as usize) < root.value_table.len() {
            root.set_value(NodeId(local), val.clone(), consumer_count);
        } else {
            return NodeResult::Return(make_error_throw("InternalError",
                &format!("writeback target {:?} out of root frame range", target)));
        }
    } else if !written_parent {
        // 路径 3：逃逸闭包（帧链为 null）— 通过 closure_val 的 Cell 回写 upvalue。
        // 逃逸闭包跨函数调用时，parent/root 均为 null，无法通过帧链回写。
        // closure_val 中的 upvalues 以 Cell 包装（compute_closure_construct），
        // 通过 Cell::set 持久化修改，使下次调用能读到最新值。
        let mut written_cell = false;
        if let Some(ref closure_val) = frame.closure_val {
            if let Value::Ref(arc) = closure_val {
                let upvalues: &[Value] = match arc.as_ref() {
                    crate::value::HeapObj::Closure(c) => &c.upvalues,
                    crate::value::HeapObj::Partial(p) => &p.upvalues,
                    _ => &[],
                };
                if !upvalues.is_empty() {
                    let sg_idx = frame.subgraph_id.0 as usize;
                    for (i, &outer_node) in frame.graph.sg_upvalue_outer_nodes(sg_idx).iter().enumerate() {
                        if outer_node == target && i < upvalues.len() {
                            if let Some(crate::value::HeapObj::Cell(cell)) = upvalues[i].heap_obj() {
                                cell.set(val.clone());
                                written_cell = true;
                            }
                            break;
                        }
                    }
                }
            }
        }
        // 路径 4：非逃逸闭包的根帧场景（顶层函数内的赋值），写入当前帧
        if !written_cell {
            let local = target.0.wrapping_sub(frame.node_offset);
            if (local as usize) < frame.value_table.len() {
                frame.set_value(NodeId(local), val.clone(), consumer_count);
            } else {
                return NodeResult::Return(make_error_throw("InternalError",
                    &format!("writeback target {:?} out of current frame range", target)));
            }
        }
    }
    NodeResult::Value(val)
}

/// compute_tailrec_writeback（idx 310）：尾递归转迭代专用 WriteBack。
///
/// 与 compute_writeback 相同的回写逻辑，额外返回 NodeResult::Continue。
/// 在 TailRec 循环中，body_sg 完成时：
/// - Continue（rec arm 的 WriteBack 返回）→ reset_loop_iteration（循环继续）
/// - None（base arm 无 WriteBack）→ 循环退出，返回 body_sg 的返回值
pub fn compute_tailrec_writeback(frame: &mut Frame, node: NodeId, _ctx: &EvalContext) -> NodeResult {
    // 正常回写 → Continue（循环继续）；越界等错误（NodeResult::Return）→ 向上传播（非静默）
    match compute_writeback(frame, node, _ctx) {
        NodeResult::Value(_) => NodeResult::Continue,
        other => other,
    }
}
