//! Optimizer.rs — IR 后优化器
//!
//! 对 IrBuilder 生成的 DataFlowGraph 做固定点迭代的图级优化。
//! Pass 管线（每轮）：Inline → ConstFold → StrengthRed → CSE → CopyProp → DCE → DSE。
//! 结构变换 pass（LICM/Unroll/Inline）在传统优化前运行，产出 redirect/dead 由
//! 晚期 rebuild 统一压缩。Engine 侧零改动。
//! 详见 docs/superpowers/plans/2026-08-08-loop-opts-inline.md

use crate::ir::Ir::{
    CF_ARRAY_STORE, CF_CALL_LAUNCH, CF_GLOBAL_LOAD, CF_GLOBAL_STORE, CF_NOOP,
    CF_RECORD_FIELD_SET, CF_SEQ, CF_WRITEBACK, ConstValue, ComputeFnId, DataFlowGraph,
    Node, NodeId, NodeKind, SubGraphId,
};
use crate::pass::Analyzer::{AnalysisReport, UnrollInfo};
use pastey::paste;
use rustc_hash::{FxHashMap, FxHashSet};

// =========================================================================
// ConstValue 提取器 — 类型安全地从 ConstValue 提取原始值
// =========================================================================

macro_rules! impl_cv_extract {
    ($cv:ident, $rust:ty, $name:ident) => {
        fn $name(cv: &ConstValue) -> Option<$rust> {
            match cv { ConstValue::$cv(v) => Some(*v), _ => None }
        }
    };
}
impl_cv_extract!(I8, i8, cv_i8);
impl_cv_extract!(I16, i16, cv_i16);
impl_cv_extract!(I32, i32, cv_i32);
impl_cv_extract!(I64, i64, cv_i64);
impl_cv_extract!(I128, i128, cv_i128);
impl_cv_extract!(U8, u8, cv_u8);
impl_cv_extract!(U16, u16, cv_u16);
impl_cv_extract!(U32, u32, cv_u32);
impl_cv_extract!(U64, u64, cv_u64);
impl_cv_extract!(U128, u128, cv_u128);
impl_cv_extract!(Isize, isize, cv_isize);
impl_cv_extract!(Usize, usize, cv_usize);
impl_cv_extract!(F32, f32, cv_f32);
impl_cv_extract!(F64, f64, cv_f64);
impl_cv_extract!(Bool, bool, cv_bool);

/// 从 args 提取两个同类型值。
fn two<T>(args: &[ConstValue], extract: fn(&ConstValue) -> Option<T>) -> Option<(T, T)> {
    Some((extract(args.get(0)?)?, extract(args.get(1)?)?))
}

// =========================================================================
// try_fold — 常量折叠分派
// =========================================================================

/// 尝试对给定 compute_fn 和常量参数执行编译期求值。
/// 返回 None 表示无法折叠（类型不匹配或非可折叠 op）。
pub fn try_fold(cf: ComputeFnId, args: &[ConstValue]) -> Option<ConstValue> {
    use crate::value as V;
    match cf.0 {
        // ── Legacy i32 算术 (1,3,5,6,7) ──
        1  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_add_i32(a, b))) }
        3  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_mul_i32(a, b))) }
        5  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_sub_i32(a, b))) }
        6  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_div_i32(a, b))) }
        7  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_mod_i32(a, b))) }
        // ── Legacy i32 比较 (4,8,9,10,11,12) → bool ──
        4  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a <= b)) }
        8  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a == b)) }
        9  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a != b)) }
        10 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a < b)) }
        11 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a > b)) }
        12 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a >= b)) }
        // ── Legacy f64 算术 (2,13,14,15) ──
        2  => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_add_f64(a, b))) }
        13 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_sub_f64(a, b))) }
        14 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_mul_f64(a, b))) }
        15 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_div_f64(a, b))) }
        // ── Legacy f64 比较 (16-21) → bool ──
        16 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::Bool(a == b)) }
        17 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::Bool(a != b)) }
        18 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::Bool(a < b)) }
        19 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::Bool(a > b)) }
        20 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::Bool(a <= b)) }
        21 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::Bool(a >= b)) }
        // ── Legacy bool (22,23,24,27) ──
        22 => { let (a, b) = two(args, cv_bool)?; Some(ConstValue::Bool(V::arith_and_bool(a, b))) }
        23 => { let (a, b) = two(args, cv_bool)?; Some(ConstValue::Bool(V::arith_or_bool(a, b))) }
        24 => { let a = cv_bool(args.get(0)?)?; Some(ConstValue::Bool(V::arith_not_bool(a))) }
        27 => { let (a, b) = two(args, cv_bool)?; Some(ConstValue::Bool(a == b)) }
        // ── Legacy neg (25,26) ──
        25 => { let a = cv_i32(args.get(0)?)?; Some(ConstValue::I32(V::arith_neg_i32(a))) }
        26 => { let a = cv_f64(args.get(0)?)?; Some(ConstValue::F64(V::arith_neg_f64(a))) }

        // ── i64 算术 + 比较 (50-61) ──
        50 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_add_i64(a, b))) }
        51 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_sub_i64(a, b))) }
        52 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_mul_i64(a, b))) }
        53 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_div_i64(a, b))) }
        54 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_mod_i64(a, b))) }
        55 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::Bool(a == b)) }
        56 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::Bool(a != b)) }
        57 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::Bool(a < b)) }
        58 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::Bool(a > b)) }
        59 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::Bool(a <= b)) }
        60 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::Bool(a >= b)) }
        61 => { let a = cv_i64(args.get(0)?)?; Some(ConstValue::I64(V::arith_neg_i64(a))) }
        // ── bitnot (62-63, 76) ──
        62 => { let a = cv_i32(args.get(0)?)?; Some(ConstValue::I32(V::arith_bitnot_i32(a))) }
        63 => { let a = cv_i64(args.get(0)?)?; Some(ConstValue::I64(V::arith_bitnot_i64(a))) }
        76 => { let a = cv_i128(args.get(0)?)?; Some(ConstValue::I128(V::arith_bitnot_i128(a))) }

        // ── i128 算术 + 比较 (64-75) ──
        64 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_add_i128(a, b))) }
        65 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_sub_i128(a, b))) }
        66 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_mul_i128(a, b))) }
        67 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_div_i128(a, b))) }
        68 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_mod_i128(a, b))) }
        69 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a == b)) }
        70 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a != b)) }
        71 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a < b)) }
        72 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a > b)) }
        73 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a <= b)) }
        74 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a >= b)) }
        75 => { let a = cv_i128(args.get(0)?)?; Some(ConstValue::I128(V::arith_neg_i128(a))) }

        // ── 位运算 i32 (77-79) ──
        77 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_bitand_i32(a, b))) }
        78 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_bitor_i32(a, b))) }
        79 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_bitxor_i32(a, b))) }
        // ── 位运算 i64 (80-82) ──
        80 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_bitand_i64(a, b))) }
        81 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_bitor_i64(a, b))) }
        82 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_bitxor_i64(a, b))) }
        // ── 位运算 i128 (83-85) ──
        83 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_bitand_i128(a, b))) }
        84 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_bitor_i128(a, b))) }
        85 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_bitxor_i128(a, b))) }
        // ── 移位 i32 (86-87)：移位量为 i32 ──
        86 => { let a = cv_i32(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; Some(ConstValue::I32(V::arith_shl_i32(a, s))) }
        87 => { let a = cv_i32(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; Some(ConstValue::I32(V::arith_shr_i32(a, s))) }
        // ── 移位 i64 (88-89) ──
        88 => { let a = cv_i64(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; Some(ConstValue::I64(V::arith_shl_i64(a, s))) }
        89 => { let a = cv_i64(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; Some(ConstValue::I64(V::arith_shr_i64(a, s))) }
        // ── 移位 i128 (90-91) ──
        90 => { let a = cv_i128(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; Some(ConstValue::I128(V::arith_shl_i128(a, s))) }
        91 => { let a = cv_i128(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; Some(ConstValue::I128(V::arith_shr_i128(a, s))) }

        // ── 全基本类型算术（92-259）──
        id if id >= 92 && id <= 259 => fold_basic_range(id, args),

        _ => None,
    }
}

/// 整数类型 12 运算折叠宏。
macro_rules! fold_int_arith {
    ($args:expr, $op:expr, $cv:ident, $ext:ident, $ty:ident) => { paste! {
        match $op {
            0 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_add_$ty>](a, b))) }
            1 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_sub_$ty>](a, b))) }
            2 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_mul_$ty>](a, b))) }
            3 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_div_$ty>](a, b))) }
            4 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_mod_$ty>](a, b))) }
            5 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_bitand_$ty>](a, b))) }
            6 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_bitor_$ty>](a, b))) }
            7 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_bitxor_$ty>](a, b))) }
            8 => { let a = $ext($args.get(0)?)?; let s = cv_i32($args.get(1)?)?; Some(ConstValue::$cv(crate::value::[<arith_shl_$ty>](a, s))) }
            9 => { let a = $ext($args.get(0)?)?; let s = cv_i32($args.get(1)?)?; Some(ConstValue::$cv(crate::value::[<arith_shr_$ty>](a, s))) }
            10 => { let a = $ext($args.get(0)?)?; Some(ConstValue::$cv(crate::value::[<arith_neg_$ty>](a))) }
            11 => { let a = $ext($args.get(0)?)?; Some(ConstValue::$cv(crate::value::[<arith_bitnot_$ty>](a))) }
            _ => None,
        }
    }};
}

/// 浮点类型 6 运算折叠宏。
macro_rules! fold_float_arith {
    ($args:expr, $op:expr, $cv:ident, $ext:ident, $ty:ident) => { paste! {
        match $op {
            0 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_add_$ty>](a, b))) }
            1 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_sub_$ty>](a, b))) }
            2 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_mul_$ty>](a, b))) }
            3 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_div_$ty>](a, b))) }
            4 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_mod_$ty>](a, b))) }
            5 => { let a = $ext($args.get(0)?)?; Some(ConstValue::$cv(crate::value::[<arith_neg_$ty>](a))) }
            _ => None,
        }
    }};
}

/// 基本类型算术折叠（92-259）。
/// 整数 12 类型 × 12 运算（92-235），浮点 4 类型 × 6 运算（236-259）。
/// f16/f128 无 ConstValue 变体，跳过（返回 None）。
fn fold_basic_range(id: u32, args: &[ConstValue]) -> Option<ConstValue> {
    if id <= 235 {
        // 整数：12 类型 × 12 运算（92-235）
        let offset = id - 92;
        let type_idx = (offset / 12) as usize;
        let op_idx = (offset % 12) as usize;
        // op: 0=add 1=sub 2=mul 3=div 4=mod 5=bitand 6=bitor 7=bitxor 8=shl 9=shr 10=neg 11=bitnot
        match type_idx {
            0 => return fold_int_arith!(args, op_idx, I8, cv_i8, i8),
            1 => return fold_int_arith!(args, op_idx, I16, cv_i16, i16),
            2 => return fold_int_arith!(args, op_idx, I32, cv_i32, i32),
            3 => return fold_int_arith!(args, op_idx, I64, cv_i64, i64),
            4 => return fold_int_arith!(args, op_idx, I128, cv_i128, i128),
            5 => return fold_int_arith!(args, op_idx, U8, cv_u8, u8),
            6 => return fold_int_arith!(args, op_idx, U16, cv_u16, u16),
            7 => return fold_int_arith!(args, op_idx, U32, cv_u32, u32),
            8 => return fold_int_arith!(args, op_idx, U64, cv_u64, u64),
            9 => return fold_int_arith!(args, op_idx, U128, cv_u128, u128),
            10 => return fold_int_arith!(args, op_idx, Isize, cv_isize, isize),
            11 => return fold_int_arith!(args, op_idx, Usize, cv_usize, usize),
            _ => return None,
        }
    } else {
        // 浮点：4 类型 × 6 运算（236-259）
        let offset = id - 236;
        let type_idx = (offset / 6) as usize;
        let op_idx = (offset % 6) as usize;
        // op: 0=add 1=sub 2=mul 3=div 4=mod 5=neg
        // f16(type_idx=0) 和 f128(type_idx=3) 无 ConstValue 变体
        match type_idx {
            1 => return fold_float_arith!(args, op_idx, F32, cv_f32, f32),
            2 => return fold_float_arith!(args, op_idx, F64, cv_f64, f64),
            _ => return None,
        }
    }
}

// =========================================================================
// OptimizerContext — 优化期变换记录
// =========================================================================

/// 优化期累积的变换：dead 集与 redirect 映射。
/// 固定点收敛后由 DataFlowGraph::rebuild 消费，一次性重建图。
#[derive(Default)]
pub struct OptimizerContext {
    /// 死节点集（DCE 标记）
    pub dead: FxHashSet<NodeId>,
    /// 重定向映射：old_node_id → new_node_id（CSE/CopyProp 产生）
    pub redirect: FxHashMap<NodeId, NodeId>,
    /// ConstFold 是否修改了节点（直接修改原节点，不产生 redirect）
    pub mutated: bool,
    /// ConstFold 本轮折叠的节点数（调试用）
    pub cf_folded_count: usize,
}

impl OptimizerContext {
    /// 递归解析重定向到最终目标。
    #[inline]
    pub fn resolve(&self, id: NodeId) -> NodeId {
        let mut cur = id;
        while let Some(&next) = self.redirect.get(&cur) {
            cur = next;
        }
        cur
    }

    /// 节点是否存活（未死且未被重定向消除）。
    #[inline]
    pub fn is_live(&self, id: NodeId) -> bool {
        !self.dead.contains(&id) && !self.redirect.contains_key(&id)
    }

    /// 本轮是否有变换。
    #[inline]
    pub fn has_changes(&self) -> bool {
        self.mutated || !self.dead.is_empty() || !self.redirect.is_empty()
    }
}

/// 检查节点是否有副作用（不可被 CSE/CopyProp/DCE 消除或重定向）。
fn has_side_effect(graph: &DataFlowGraph, idx: usize) -> bool {
    graph.writeback_targets.get(idx).map_or(false, |o| o.is_some())
    || graph.field_set_names.get(idx).map_or(false, |o| o.is_some())
    || graph.global_store_slots.get(idx).map_or(false, |o| o.is_some())
    || crate::ir::Ir::is_control_flow_compute_fn(graph.nodes[idx].compute_fn)
    || graph.ffi_call_names.get(idx).map_or(false, |o| o.is_some())
    || graph.tail_call_flags.get(idx).copied().unwrap_or(false)
}

/// 收集所有 writeback 目标节点 ID。
/// 这些节点的运行时值会被 writeback 覆盖，不可作为 CSE 合并目标或 ConstFold 常量源。
fn collect_writeback_targets(graph: &DataFlowGraph) -> FxHashSet<NodeId> {
    let mut set = FxHashSet::default();
    for opt_wt in &graph.writeback_targets {
        if let Some(wt) = opt_wt {
            set.insert(*wt);
        }
    }
    set
}

// =========================================================================
// compute_live_set — 反向可达性分析
// =========================================================================

/// 计算活跃节点集：从所有子图的 return_node/cond_node/iter_next_node 反向遍历 inputs。
/// 使用 ctx.resolve 解析重定向，确保 redirected 节点的 inputs 不被遍历。
/// 同时遍历 per-node 元数据中的 NodeId 引用（gate_branches/select_infos/writeback_targets）。
pub fn compute_live_set(graph: &DataFlowGraph, ctx: &OptimizerContext) -> FxHashSet<NodeId> {
    let mut live: FxHashSet<NodeId> = FxHashSet::default();
    let mut stack: Vec<NodeId> = Vec::new();

    // 辅助：resolve + insert + push
    let add = |id: NodeId, live: &mut FxHashSet<NodeId>, stack: &mut Vec<NodeId>| {
        let r = ctx.resolve(id);
        if live.insert(r) { stack.push(r); }
    };

    // 种子：所有子图的 return_node + entry_node + cond_node + iter_next_node
    for sg in &graph.subgraphs {
        for &raw in &[sg.return_node, sg.entry_node] {
            add(raw, &mut live, &mut stack);
        }
        if let Some(c) = sg.cond_node { add(c, &mut live, &mut stack); }
        if let Some(n) = sg.iter_next_node { add(n, &mut live, &mut stack); }
        // defer_table: trigger_node + captured_inputs
        for entry in &sg.defer_table {
            add(entry.trigger_node, &mut live, &mut stack);
            for &cap in &entry.captured_inputs { add(cap, &mut live, &mut stack); }
        }
        // event_source_decls: node
        for decl in &sg.event_source_decls {
            add(decl.node, &mut live, &mut stack);
        }
    }
    // 事件源声明节点保留
    for opt_n in &graph.await_event_sources {
        if let Some(n) = opt_n { add(*n, &mut live, &mut stack); }
    }
    // 副作用节点保留（不可被 DCE 删除）
    for idx in 0..graph.nodes.len() {
        if has_side_effect(graph, idx) {
            add(NodeId(idx as u32), &mut live, &mut stack);
        }
    }

    while let Some(n) = stack.pop() {
        let idx = n.0 as usize;
        let node = graph.nodes[idx];

        // 1. 遍历 node.inputs
        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        for &input in inputs {
            add(input, &mut live, &mut stack);
        }

        // 2. 遍历 per-node 元数据中的 NodeId 引用
        // gate_branches: condition_input + branches params
        if let Some(gb) = graph.gate_branches.get(idx).and_then(|o| o.as_ref()) {
            add(gb.condition_input, &mut live, &mut stack);
            for (_, _, params) in &gb.branches {
                for &p in params { add(p, &mut live, &mut stack); }
            }
        }
        // select_infos: event_source_node
        if let Some(si) = graph.select_infos.get(idx).and_then(|o| o.as_ref()) {
            for sb in &si.branches {
                add(sb.event_source_node, &mut live, &mut stack);
            }
        }
        // writeback_targets
        if let Some(Some(wt)) = graph.writeback_targets.get(idx).map(|o| o.as_ref()) {
            add(*wt, &mut live, &mut stack);
        }
    }
    live
}

// =========================================================================
// Pass: ConstFold — 常量折叠
// =========================================================================

/// ConstFold pass：BinOp/UnOp 全 Const 输入 → 折叠为 Const。
/// 直接修改原节点为 Const（不创建新节点，保持 NodeId 不变，确保在 node_range 内）。
/// 单轮内反复扫描直到无新折叠（链式折叠：A→Const 后 B 依赖 A 也可折叠）。
pub fn pass_const_fold(graph: &mut DataFlowGraph, ctx: &mut OptimizerContext) {
    let node_count = graph.nodes.len();
    let wb_targets = collect_writeback_targets(graph);
    let mut total_folded = 0usize;

    loop {
        let mut folded_this_round: Vec<(usize, ConstValue)> = Vec::new();

        for idx in 0..node_count {
            let id = NodeId(idx as u32);
            if !ctx.is_live(id) { continue; }
            let node = graph.nodes[idx];
            if node.kind == NodeKind::Const { continue; }
            // 副作用节点不可折叠：writeback 目标的输入在运行时会变，
            // 不能用初始常量值替代运行时计算。
            if has_side_effect(graph, idx) { continue; }

            let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
            let mut arg_values: Vec<ConstValue> = Vec::with_capacity(inputs.len());
            let mut all_const = true;
            for &input in inputs {
                let resolved = ctx.resolve(input);
                // writeback 目标的运行时值会变，不可作为常量源
                if wb_targets.contains(&resolved) { all_const = false; break; }
                let ridx = resolved.0 as usize;
                match graph.const_values.get(ridx).and_then(|o| o.as_ref()) {
                    Some(cv) => arg_values.push(cv.clone()),
                    None => { all_const = false; break; }
                }
            }
            if !all_const || arg_values.is_empty() { continue; }

            if let Some(result) = try_fold(node.compute_fn, &arg_values) {
                folded_this_round.push((idx, result));
            }
        }

        if folded_this_round.is_empty() { break; }

        // 直接修改原节点为 Const
        for (idx, cv) in folded_this_round {
            let new_offset = graph.inputs_pool.push(&[]);
            graph.nodes[idx] = Node {
                kind: NodeKind::Const,
                input_count: 0,
                inputs_offset: new_offset,
                compute_fn: CF_NOOP,
            };
            graph.const_values[idx] = Some(cv);
        }
        total_folded += 1;
    }

    // 折叠了节点就标记 mutated，让外层固定点继续跑下一轮：
    // 新常量可能给 CSE/DCE 提供新机会（如 Const 节点可被消除）。
    // 不会不收敛——每轮 ConstFold 至少折叠一个节点，节点总数有限，
    // 最终 folded_this_round 为空退出循环。
    if total_folded > 0 {
        ctx.mutated = true;
        ctx.cf_folded_count += total_folded;
    }
}

// =========================================================================
// Pass: CSE — 公共子表达式消除
// =========================================================================

/// 为每个节点预计算其所属的最内层子图的起始 NodeId。
/// CSE 只在同一最内层子图内合并节点，防止跨 if-else/match 分支子图合并
/// 导致分支帧无法正确计算被合并的节点（Bug #45）。
fn compute_innermost_sg_starts(graph: &DataFlowGraph) -> Vec<u32> {
    let n = graph.nodes.len();
    // 默认 0 = 函数体子图起始（或无子图）
    let mut starts: Vec<u32> = vec![0; n];
    // 按子图范围大小降序排序：大范围先填，小范围后覆盖
    let mut sgs: Vec<(u32, u32)> = graph
        .subgraphs
        .iter()
        .map(|sg| (sg.node_range.0 .0, sg.node_range.1 .0))
        .collect();
    sgs.sort_by_key(|&(_, end)| std::cmp::Reverse(end));
    // 降序不够：需要按范围大小降序，确保小范围覆盖大范围
    sgs.sort_by_key(|&(start, end)| std::cmp::Reverse(end - start));
    for (start, end) in &sgs {
        for i in *start..*end {
            if (i as usize) < n {
                starts[i as usize] = *start;
            }
        }
    }
    starts
}

/// CSE pass：纯节点 (compute_fn, resolved_inputs, metadata_hash, innermost_sg) 相同 → 合并。
/// 首个出现者保留，后续 redirect 到首个。
/// key 包含最内层子图起始 ID，确保跨 if-else/match 分支的相同计算不会被合并
/// （分支子图互斥执行，合并会导致未执行分支的节点引用丢失，Bug #45）。
/// 元数据哈希确保 pattern_field_indices/pattern_ctor_names/field_access_infos 等
/// per-node 元数据不同的节点不会被错误合并。
pub fn pass_cse(graph: &DataFlowGraph, ctx: &mut OptimizerContext, pure_set: &FxHashSet<ComputeFnId>) {
    let mut seen: FxHashMap<(ComputeFnId, Vec<NodeId>, u64, u32), NodeId> = FxHashMap::default();
    let wb_targets = collect_writeback_targets(graph);
    let sg_starts = compute_innermost_sg_starts(graph);

    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(idx as u32);
        if !ctx.is_live(id) { continue; }
        // 跳过已被 redirect 的节点（避免反复产生相同 redirect）
        if ctx.redirect.contains_key(&id) { continue; }
        if !pure_set.contains(&node.compute_fn) { continue; }
        if node.kind == NodeKind::Gate { continue; }
        // 副作用节点不可重定向（writeback/field_set/global_store 等）
        if has_side_effect(graph, idx) { continue; }
        // writeback 目标的运行时值会变，不可作为 CSE 合并目标或源
        if wb_targets.contains(&id) { continue; }

        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        let resolved: Vec<NodeId> = inputs.iter().map(|&i| ctx.resolve(i)).collect();
        let meta_hash = graph.cse_metadata_hash(idx);
        let sg_start = sg_starts[idx];
        let key = (node.compute_fn, resolved, meta_hash, sg_start);
        if let Some(&existing) = seen.get(&key) {
            ctx.redirect.insert(id, existing);
        } else {
            seen.insert(key, id);
        }
    }
}

// =========================================================================
// Pass: CopyProp — 拷贝传播
// =========================================================================

/// 透传 compute_fn 集合：单输入、输出=输入。
/// noop_compute_real(0) 是纯透传。
fn passthrough_set() -> FxHashSet<ComputeFnId> {
    let mut s = FxHashSet::default();
    s.insert(CF_NOOP); // noop_compute_real
    s
}

/// CopyProp pass：透传节点 redirect 到其唯一 input。
pub fn pass_copy_prop(graph: &DataFlowGraph, ctx: &mut OptimizerContext) {
    let passthrough = passthrough_set();
    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(idx as u32);
        if !ctx.is_live(id) { continue; }
        // 跳过已被 redirect 的节点（避免反复产生相同 redirect）
        if ctx.redirect.contains_key(&id) { continue; }
        if node.input_count != 1 { continue; }
        if !passthrough.contains(&node.compute_fn) { continue; }
        // 副作用节点不可重定向（writeback/field_set/global_store 等）
        if has_side_effect(graph, idx) { continue; }
        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        let src = ctx.resolve(inputs[0]);
        // 避免自环
        if src != id {
            ctx.redirect.insert(id, src);
        }
    }
}

// =========================================================================
// Pass: DCE — 死代码消除
// =========================================================================

/// 收集节点的所有 inputs 和元数据中的 NodeId 引用（resolve 后）。
fn collect_refs(graph: &DataFlowGraph, ctx: &OptimizerContext, idx: usize, out: &mut Vec<NodeId>) {
    let node = graph.nodes[idx];
    let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
    for &input in inputs {
        out.push(ctx.resolve(input));
    }
    if let Some(gb) = graph.gate_branches.get(idx).and_then(|o| o.as_ref()) {
        out.push(ctx.resolve(gb.condition_input));
        for (_, _, params) in &gb.branches {
            for &p in params { out.push(ctx.resolve(p)); }
        }
    }
    if let Some(si) = graph.select_infos.get(idx).and_then(|o| o.as_ref()) {
        for sb in &si.branches {
            out.push(ctx.resolve(sb.event_source_node));
        }
    }
    if let Some(Some(wt)) = graph.writeback_targets.get(idx).map(|o| o.as_ref()) {
        out.push(ctx.resolve(*wt));
    }
}

/// DCE pass：标记不可达的纯计算节点为 dead。
/// 三步策略：
/// 1. 计算live set，标记不在live set中的纯计算节点为dead候选
/// 2. 保留传播：从所有保留节点（非dead、非redirect key）的inputs反向遍历，
///    把被保留节点依赖的dead候选从dead集中移除
/// 3. 处理redirect目标为dead的情况：redirect目标dead则redirect key也dead
pub fn pass_dce(graph: &DataFlowGraph, ctx: &mut OptimizerContext, pure_set: &FxHashSet<ComputeFnId>) {
    let live = compute_live_set(graph, ctx);

    // Step 1: 标记不在live set中的纯计算节点为dead候选
    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(idx as u32);
        if live.contains(&id) { continue; }
        if !ctx.is_live(id) { continue; }
        let is_pure_calc = match node.kind {
            NodeKind::BinOp | NodeKind::UnOp | NodeKind::FieldAccess => {
                pure_set.contains(&node.compute_fn)
            }
            NodeKind::Const | NodeKind::Call | NodeKind::Gate
            | NodeKind::Await | NodeKind::EventSource => false,
        };
        if is_pure_calc {
            ctx.dead.insert(id);
        }
    }

    // Step 2: 保留传播 — 从所有保留节点的引用反向遍历，移除可达的dead候选
    // 保留节点 = 非dead、非redirect key 的节点（这些节点会留在graph中）
    // 它们的inputs必须保留，否则rebuild时panic
    let mut preserve_stack: Vec<NodeId> = Vec::new();
    let mut refs_buf: Vec<NodeId> = Vec::new();
    for idx in 0..graph.nodes.len() {
        let id = NodeId(idx as u32);
        if ctx.dead.contains(&id) || ctx.redirect.contains_key(&id) { continue; }
        refs_buf.clear();
        collect_refs(graph, ctx, idx, &mut refs_buf);
        for r in &refs_buf {
            if ctx.dead.remove(r) { preserve_stack.push(*r); }
        }
    }
    while let Some(n) = preserve_stack.pop() {
        refs_buf.clear();
        collect_refs(graph, ctx, n.0 as usize, &mut refs_buf);
        for r in &refs_buf {
            if ctx.dead.remove(r) { preserve_stack.push(*r); }
        }
    }

    // Step 3: 处理redirect目标为dead的情况
    // 如果redirect的resolve目标是dead，redirect key也应加入dead集
    // （否则rebuild时resolve(redirect_key)=dead_target，old_to_new[dead_target]=None → panic）
    loop {
        let mut changed = false;
        let keys: Vec<NodeId> = ctx.redirect.keys().copied().collect();
        for key in keys {
            if ctx.dead.contains(&key) { continue; }
            let target = ctx.resolve(key);
            if ctx.dead.contains(&target) {
                ctx.dead.insert(key);
                changed = true;
            }
        }
        if !changed { break; }
    }
}

// =========================================================================
// Pass: Strength Reduction — 强度归约
// =========================================================================

/// 判断 u128 值是否为 2 的幂，返回 log2（0 表示 2^0=1）。
fn power_of_two(v: u128) -> Option<u32> {
    if v == 0 { return None; }
    let n = v.trailing_zeros();
    if (1u128 << n) == v { Some(n) } else { None }
}

/// 从 ConstValue 提取无符号 u128 值（用于 2 的幂判定）。
fn cv_to_u128(cv: &ConstValue) -> Option<u128> {
    match cv {
        ConstValue::I8(v)   => Some((*v as i128) as u128),
        ConstValue::I16(v)  => Some((*v as i128) as u128),
        ConstValue::I32(v)  => Some((*v as i128) as u128),
        ConstValue::I64(v)  => Some((*v as i128) as u128),
        ConstValue::I128(v) => Some(*v as u128),
        ConstValue::U8(v)   => Some(*v as u128),
        ConstValue::U16(v)  => Some(*v as u128),
        ConstValue::U32(v)  => Some(*v as u128),
        ConstValue::U64(v)  => Some(*v as u128),
        ConstValue::U128(v) => Some(*v),
        ConstValue::Isize(v) => Some((*v as i128) as u128),
        ConstValue::Usize(v) => Some(*v as u128),
        _ => None,
    }
}

/// 将 ConstValue 转换为指定移位量的 i32 ConstValue（移位量始终用 i32 存储）。
fn make_shift_const(n: u32) -> ConstValue {
    ConstValue::I32(n as i32)
}

/// 从 mul compute_fn 推导对应类型的 shl compute_fn。
/// 整数全范围（92-235）：mul(offset 2) → shl(offset 8)。
/// 浮点 mul 不支持强度归约，返回 None。
fn mul_to_shl(cf: ComputeFnId) -> Option<ComputeFnId> {
    let id = cf.0;
    if id >= 92 && id <= 235 {
        let offset = id - 92;
        let op = offset % 12;
        if op == 2 { // mul
            let type_base = id - op;
            return Some(ComputeFnId(type_base + 8)); // shl
        }
    }
    None
}

/// 从无符号 div compute_fn 推导对应类型的 shr compute_fn。
/// 整数全范围（92-235）：div(offset 3) → shr(offset 9)。
/// 有符号除法不可安全转为右移（负数截断语义不同），仅无符号类型适用。
/// type_idx >= 5 表示 u8 起始的无符号类型区间。
fn div_to_shr(cf: ComputeFnId) -> Option<ComputeFnId> {
    let id = cf.0;
    if id >= 92 && id <= 235 {
        let offset = id - 92;
        let op = offset % 12;
        let type_idx = offset / 12;
        if op == 3 && type_idx >= 5 { // div 且无符号类型（u8 起始）
            let type_base = id - op;
            return Some(ComputeFnId(type_base + 9)); // shr
        }
    }
    None
}

/// 从无符号 mod compute_fn 推导对应类型的 bitand compute_fn。
/// 整数全范围（92-235）：mod(offset 4) → bitand(offset 5)。
/// `x % 2^n` → `x & (2^n - 1)`，仅无符号类型安全（有符号模数符号与被除数一致）。
fn mod_to_bitand(cf: ComputeFnId) -> Option<ComputeFnId> {
    let id = cf.0;
    if id >= 92 && id <= 235 {
        let offset = id - 92;
        let op = offset % 12;
        let type_idx = offset / 12;
        if op == 4 && type_idx >= 5 { // mod 且无符号类型（u8 起始）
            let type_base = id - op;
            return Some(ComputeFnId(type_base + 5)); // bitand
        }
    }
    None
}

/// 将 ConstValue 的值替换为新的 u128 值，保持原始类型不变。
/// 用于 mod→bitand 变换：将常量从 `2^n` 改为 `2^n - 1`（同类型）。
fn cv_set_u128(cv: &ConstValue, val: u128) -> Option<ConstValue> {
    match cv {
        ConstValue::I8(_)   => val.try_into().ok().map(ConstValue::I8),
        ConstValue::I16(_)  => val.try_into().ok().map(ConstValue::I16),
        ConstValue::I32(_)  => val.try_into().ok().map(ConstValue::I32),
        ConstValue::I64(_)  => val.try_into().ok().map(ConstValue::I64),
        ConstValue::I128(_) => Some(ConstValue::I128(val as i128)),
        ConstValue::U8(_)   => val.try_into().ok().map(ConstValue::U8),
        ConstValue::U16(_)  => val.try_into().ok().map(ConstValue::U16),
        ConstValue::U32(_)  => val.try_into().ok().map(ConstValue::U32),
        ConstValue::U64(_)  => val.try_into().ok().map(ConstValue::U64),
        ConstValue::U128(_) => Some(ConstValue::U128(val)),
        ConstValue::Isize(_) => val.try_into().ok().map(ConstValue::Isize),
        ConstValue::Usize(_) => val.try_into().ok().map(ConstValue::Usize),
        _ => None,
    }
}

/// Strength Reduction pass：将乘除模 2 的幂转换为移位/位运算。
///
/// 变换模式：
/// - `x * 2^n` → `x << n`（乘法 → 左移，所有整数类型）
/// - `x / 2^n`（无符号）→ `x >> n`（无符号除法 → 逻辑右移）
/// - `x % 2^n`（无符号）→ `x & (2^n - 1)`（无符号模运算 → 位掩码）
///
/// 有符号除法/模运算不归约：负数截断/取模语义与移位/位运算不同，
/// 需要额外的舍入修正序列，复杂度高于收益。
///
/// 变换方式：原位改写 compute_fn + 复用已有常量节点（改其值为移位量/掩码）。
/// 不创建新节点，避免 hoisted 节点跨子图范围问题。
/// 通过 ctx.mutated 标记触发下一轮固定点迭代。
pub fn pass_strength_reduction(graph: &mut DataFlowGraph, ctx: &mut OptimizerContext) {
    let wb_targets = collect_writeback_targets(graph);
    let node_count = graph.nodes.len();
    let mut changed = false;

    for idx in 0..node_count {
        let id = NodeId(idx as u32);
        if !ctx.is_live(id) { continue; }
        let node = graph.nodes[idx];
        if node.kind != NodeKind::BinOp { continue; }
        if has_side_effect(graph, idx) { continue; }

        let cf = node.compute_fn;

        // 尝试乘法 → 左移
        if let Some(shl_cf) = mul_to_shl(cf) {
            let inputs: Vec<NodeId> = graph.inputs_pool.get(node.inputs_offset, node.input_count).to_vec();
            if inputs.len() != 2 { continue; }

            // 检查某个输入是否为 2 的幂常量
            for which in 0..2 {
                let other = 1 - which;
                let resolved = ctx.resolve(inputs[which]);
                if wb_targets.contains(&resolved) { continue; }
                let ridx = resolved.0 as usize;
                let Some(cv) = graph.const_values.get(ridx).and_then(|o| o.as_ref()) else { continue; };
                let Some(val) = cv_to_u128(cv) else { continue; };
                let Some(n) = power_of_two(val) else { continue; };
                if n == 0 { continue; } // x*1 应由 ConstFold 处理

                // 安全检查：常量节点不能被其他节点引用（否则改值会影响其他使用者）
                // downstreams 为 1 表示只有当前乘法节点引用它
                if graph.downstreams[ridx].len() != 1 { continue; }

                // 原位改写：compute_fn → shl
                graph.nodes[idx].compute_fn = shl_cf;

                // 清除 batch_infos：原 mul 节点有 BatchInfo::Bin(Mul)，
                // 改为 shl 后批处理路径会用 as_i64 读移位量（i32），导致错误结果
                if idx < graph.batch_infos.len() {
                    graph.batch_infos[idx] = None;
                }

                // 复用已有常量节点：将其值改为移位量（i32）
                // 该节点已在正确的子图范围内，无需 hoisted
                graph.const_values[ridx] = Some(make_shift_const(n));

                // 重排输入：[变量, 移位量常量]
                let var_input = inputs[other];
                let const_input = resolved;
                let new_inputs = [var_input, const_input];
                let new_offset = graph.inputs_pool.push(&new_inputs);
                graph.nodes[idx].inputs_offset = new_offset;
                graph.nodes[idx].input_count = 2;

                if std::env::var("KUZO_STRENGTH_DBG").is_ok() {
                    eprintln!("[STRENGTH] mul→shl node={} var={} const_node={} const_val={}→{} cf={} downstreams={} const_kind={:?} const_cf={}",
                        idx, var_input.0, resolved.0, val, n, shl_cf.0, graph.downstreams[ridx].len(),
                        graph.nodes[ridx].kind, graph.nodes[ridx].compute_fn.0);
                }
                changed = true;
                break; // 已处理此节点，跳出 which 循环
            }
            continue;
        }

        // 尝试无符号除法 → 逻辑右移
        if let Some(shr_cf) = div_to_shr(cf) {
            let inputs: Vec<NodeId> = graph.inputs_pool.get(node.inputs_offset, node.input_count).to_vec();
            if inputs.len() != 2 { continue; }

            // 除数必须是第二个输入（除法不可交换）
            let divisor_resolved = ctx.resolve(inputs[1]);
            if wb_targets.contains(&divisor_resolved) { continue; }
            let didx = divisor_resolved.0 as usize;
            let Some(dcv) = graph.const_values.get(didx).and_then(|o| o.as_ref()) else { continue; };
            let Some(dval) = cv_to_u128(dcv) else { continue; };
            let Some(n) = power_of_two(dval) else { continue; };
            if n == 0 { continue; } // x/1 应由 ConstFold 处理

            // 安全检查：常量节点不能被其他节点引用
            if graph.downstreams[didx].len() != 1 { continue; }

            // 原位改写：compute_fn → shr
            graph.nodes[idx].compute_fn = shr_cf;

            // 清除 batch_infos：同 mul→shl 理由
            if idx < graph.batch_infos.len() {
                graph.batch_infos[idx] = None;
            }

            // 复用已有常量节点：将其值改为移位量（i32）
            graph.const_values[didx] = Some(make_shift_const(n));

            // 重排输入：[被除数, 移位量常量]
            let dividend_input = inputs[0];
            let new_inputs = [dividend_input, divisor_resolved];
            let new_offset = graph.inputs_pool.push(&new_inputs);
            graph.nodes[idx].inputs_offset = new_offset;
            graph.nodes[idx].input_count = 2;

            if std::env::var("KUZO_STRENGTH_DBG").is_ok() {
                eprintln!("[STRENGTH] udiv→shr node={} dividend={} divisor={}→{} cf={}",
                    idx, dividend_input.0, dval, n, shr_cf.0);
            }
            changed = true;
        }

        // 尝试无符号模运算 → 位掩码
        if let Some(bitand_cf) = mod_to_bitand(cf) {
            let inputs: Vec<NodeId> = graph.inputs_pool.get(node.inputs_offset, node.input_count).to_vec();
            if inputs.len() != 2 { continue; }

            // 模运算的除数是第二个输入（不可交换）
            let divisor_resolved = ctx.resolve(inputs[1]);
            if wb_targets.contains(&divisor_resolved) { continue; }
            let didx = divisor_resolved.0 as usize;
            let Some(dcv) = graph.const_values.get(didx).and_then(|o| o.as_ref()) else { continue; };
            let Some(dval) = cv_to_u128(dcv) else { continue; };
            let Some(n) = power_of_two(dval) else { continue; };
            if n == 0 { continue; } // x%1 应由 ConstFold 处理

            // 安全检查：常量节点不能被其他节点引用
            if graph.downstreams[didx].len() != 1 { continue; }

            // 原位改写：compute_fn mod → bitand
            graph.nodes[idx].compute_fn = bitand_cf;

            // 清除 batch_infos：原 mod 节点有 BatchInfo::Bin(Mod)，
            // 改为 bitand 后批处理路径会用错误的 op
            if idx < graph.batch_infos.len() {
                graph.batch_infos[idx] = None;
            }

            // 复用已有常量节点：将其值从 2^n 改为 2^n - 1（保持原始类型）
            let mask = dval - 1;
            graph.const_values[didx] = cv_set_u128(dcv, mask);

            // 输入顺序不变：[被除数, 掩码常量]
            // 无需重排 inputs，mod 和 bitand 都是 [x, const]

            if std::env::var("KUZO_STRENGTH_DBG").is_ok() {
                eprintln!("[STRENGTH] umod→bitand node={} dividend={} divisor={}→mask={} cf={}",
                    idx, inputs[0].0, dval, mask, bitand_cf.0);
            }
            changed = true;
        }
    }

    if changed { ctx.mutated = true; }
}

// =========================================================================
// Pass: Dead Store Elimination — 死存储消除
// =========================================================================

/// 判断节点是否为存储类副作用节点（WriteBack/FieldSet/ArrayStore/GlobalStore）。
/// 统一使用 compute_fn 判定，与节点实际运算语义一致。
fn is_store_node(graph: &DataFlowGraph, idx: usize) -> bool {
    let cf = graph.nodes[idx].compute_fn;
    cf == CF_WRITEBACK
    || cf == CF_RECORD_FIELD_SET
    || cf == CF_ARRAY_STORE
    || cf == CF_GLOBAL_STORE
}

/// DSE pass：消除结果无人消费的存储节点。
///
/// 存储节点（CF_WRITEBACK/CF_RECORD_FIELD_SET/CF_ARRAY_STORE/global_store）
/// 的返回值通常是 VOID，不被下游节点消费。但如果存储节点的 downstreams 为空，
/// 且该节点不被任何活跃节点的元数据引用（非 cond/return/defer/event 触发），
/// 则该存储是死存储，可安全消除。
///
/// 安全约束：
/// - 不消除控制流节点（CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR）
/// - 不消除 tail_call_flags 节点
/// - 不消除 defer_table/event_source_decls 引用的节点
/// - 不消除被其他活跃节点 inputs 引用的节点（存储值可能被读取）
///
/// 消除方式：加入 ctx.dead 集，由 rebuild 统一清理。
pub fn pass_dse(graph: &DataFlowGraph, ctx: &mut OptimizerContext) {
    let live = compute_live_set(graph, ctx);

    // 构建两个引用集合：
    // - all_refs：所有非 dead 节点的 inputs 引用（包括存储类节点）。
    //   用于 rebuild 安全检查：rebuild 保留所有不在 dead 集中的节点，
    //   如果存储节点被任何保留节点的 inputs 引用，消除它会导致 rebuild panic。
    //   注意：必须遍历所有非 dead 节点（不仅仅是 live 集合），因为有些节点
    //   可能不在 live 中但也不在 dead 中（如 Call/Gate 节点），rebuild 仍保留它们。
    // - read_refs：仅非存储类活跃节点的 inputs 引用。
    //   存储类节点的 inputs 是"写入"语义（写入值/被修改对象），不算"读取"。
    //   read_refs 精确表示"被读取"的节点，用于判断存储副作用是否可观测。
    let mut all_refs: FxHashSet<NodeId> = FxHashSet::default();
    let mut read_refs: FxHashSet<NodeId> = FxHashSet::default();
    // all_refs：遍历所有非 dead、非 redirect-key 节点
    for idx in 0..graph.nodes.len() {
        let id = NodeId(idx as u32);
        if ctx.dead.contains(&id) { continue; }
        if ctx.redirect.contains_key(&id) { continue; }
        let node = graph.nodes[idx];
        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        for &inp in inputs {
            all_refs.insert(ctx.resolve(inp));
        }
        // gate_branches / select_infos 元数据引用
        if let Some(gb) = graph.gate_branches.get(idx).and_then(|o| o.as_ref()) {
            all_refs.insert(ctx.resolve(gb.condition_input));
            for (_, _, params) in &gb.branches {
                for &p in params { all_refs.insert(ctx.resolve(p)); }
            }
        }
        if let Some(si) = graph.select_infos.get(idx).and_then(|o| o.as_ref()) {
            for sb in &si.branches {
                all_refs.insert(ctx.resolve(sb.event_source_node));
            }
        }
    }
    // read_refs：仅从 live 集合中的非存储类节点构建
    for &nid in &live {
        let idx = nid.0 as usize;
        if is_store_node(graph, idx) { continue; }
        let node = graph.nodes[idx];
        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        for &inp in inputs {
            read_refs.insert(ctx.resolve(inp));
        }
        if let Some(gb) = graph.gate_branches.get(idx).and_then(|o| o.as_ref()) {
            read_refs.insert(ctx.resolve(gb.condition_input));
            for (_, _, params) in &gb.branches {
                for &p in params { read_refs.insert(ctx.resolve(p)); }
            }
        }
        if let Some(si) = graph.select_infos.get(idx).and_then(|o| o.as_ref()) {
            for sb in &si.branches {
                read_refs.insert(ctx.resolve(sb.event_source_node));
            }
        }
    }

    // 收集子图结构引用的节点（entry/return/cond/iter_next/defer/event）
    let mut struct_refs: FxHashSet<NodeId> = FxHashSet::default();
    for sg in &graph.subgraphs {
        struct_refs.insert(ctx.resolve(sg.return_node));
        struct_refs.insert(ctx.resolve(sg.entry_node));
        if let Some(c) = sg.cond_node { struct_refs.insert(ctx.resolve(c)); }
        if let Some(n) = sg.iter_next_node { struct_refs.insert(ctx.resolve(n)); }
        for entry in &sg.defer_table {
            struct_refs.insert(ctx.resolve(entry.trigger_node));
            for &cap in &entry.captured_inputs { struct_refs.insert(ctx.resolve(cap)); }
        }
        for decl in &sg.event_source_decls {
            struct_refs.insert(ctx.resolve(decl.node));
        }
    }

    // 收集所有活跃 global_load 读取的槽位集合。
    // 如果某个 global_store 的槽位不在该集合中，说明全图无任何活跃节点
    // 读取该全局变量，该 store 是死存储。
    let mut loaded_slots: FxHashSet<u32> = FxHashSet::default();
    for &nid in &live {
        let idx = nid.0 as usize;
        let node = graph.nodes[idx];
        if node.compute_fn == CF_GLOBAL_LOAD {
            if let Some(slot) = graph.global_load_slots.get(idx).and_then(|o| *o) {
                loaded_slots.insert(slot);
            }
        }
    }

    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(idx as u32);
        if !ctx.is_live(id) { continue; }
        if !live.contains(&id) { continue; } // 不在活跃集中的已由 DCE 处理

        // 仅处理存储类节点
        if !is_store_node(graph, idx) { continue; }

        // 安全检查：不消除控制流/尾调用节点
        if crate::ir::Ir::is_control_flow_compute_fn(graph.nodes[idx].compute_fn) { continue; }
        if graph.tail_call_flags.get(idx).copied().unwrap_or(false) { continue; }

        // 安全检查：不消除子图结构引用的节点
        if struct_refs.contains(&id) { continue; }

        // rebuild 安全检查：如果存储节点被任何保留节点的 inputs 引用，
        // 消除它会导致 rebuild panic。all_refs 涵盖了 downstreams 检查
        // （有活跃下游 ⟹ 在 all_refs 中），因此无需单独检查 downstreams。
        if all_refs.contains(&id) { continue; }

        let cf = node.compute_fn;

        // WriteBack：target 不被任何非存储活跃节点读取 → 死存储
        if cf == CF_WRITEBACK {
            if let Some(Some(wt)) = graph.writeback_targets.get(idx).map(|o| o.as_ref()) {
                let wt_resolved = ctx.resolve(*wt);
                if read_refs.contains(&wt_resolved) || struct_refs.contains(&wt_resolved) {
                    continue;
                }
            }
        }

        // FieldSet/ArrayStore：被修改的堆对象不被任何非存储活跃节点读取 → 死存储
        if cf == CF_RECORD_FIELD_SET || cf == CF_ARRAY_STORE {
            let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
            if !inputs.is_empty() {
                let obj_node = ctx.resolve(inputs[0]);
                if read_refs.contains(&obj_node) {
                    continue;
                }
            }
        }

        // GlobalStore：全图无活跃 global_load 读取该槽位 → 死存储
        if cf == CF_GLOBAL_STORE {
            if let Some(slot) = graph.global_store_slots.get(idx).and_then(|o| *o) {
                if loaded_slots.contains(&slot) {
                    continue;
                }
            }
        }

        // 死存储：加入 dead 集
        if std::env::var("KUZO_DSE_DBG").is_ok() {
            eprintln!("[DSE] eliminate dead store node={} cf={} kind={:?}",
                idx, node.compute_fn.0, node.kind);
        }
        ctx.dead.insert(id);
    }
}

// =========================================================================
// 固定点迭代驱动器
// =========================================================================

/// 优化等级。
///
/// 驱动 `optimize_with_analysis` 的 pass 选择与迭代上限：
/// - `O0`：不优化，仅 Build 产出
/// - `O1`：跳过结构变换，仅固定点迭代（Inline + 传统优化）
/// - `O2`：全量（结构变换 + 固定点迭代），标准等级
/// - `O3`：全量 + 提高迭代上限（200），激进优化
///
/// 环境变量 `KUZO_NO_*` 仍可逐 pass 禁用（调试用），优先级高于等级。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum OptLevel {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::O2
    }
}

/// 优化入口（无分析报告）：等价于 `optimize_with_analysis(graph, None, OptLevel::default())`。
pub fn optimize(graph: &mut DataFlowGraph) {
    optimize_with_analysis(graph, None, OptLevel::default());
}

/// 优化入口（带分析报告）：对 graph 执行固定点迭代优化。
///
/// 两阶段管线（按 `level` 启用）：
/// 1. 结构变换（一次性，level >= 2）：LICM → LoopUnroll — 依赖 AnalysisReport 中的 NodeId，
///    rebuild 后 NodeId 失效，故仅运行一次。
/// 2. 固定点迭代（level >= 1）：Inline → ConstFold → StrengthRed → CSE → CopyProp → DCE → DSE
///    — Inline 自收集候选（不依赖 analysis），可在每轮安全运行。
/// 无分析报告时 Phase 1 跳过，退化为纯传统优化管线。
/// `level == O0` 时整个优化器跳过。
pub fn optimize_with_analysis(
    graph: &mut DataFlowGraph,
    analysis: Option<&AnalysisReport>,
    level: OptLevel,
) {
    // O0：不优化
    if level == OptLevel::O0 {
        return;
    }

    let pure_set = crate::ir::Ir::pure_compute_fn_set();
    let no_fold = std::env::var("KUZO_NO_FOLD").is_ok();
    let no_cse = std::env::var("KUZO_NO_CSE").is_ok();
    let no_copy = std::env::var("KUZO_NO_COPY").is_ok();
    let no_dce = std::env::var("KUZO_NO_DCE").is_ok();
    let no_licm = std::env::var("KUZO_NO_LICM").is_ok();
    let no_unroll = std::env::var("KUZO_NO_UNROLL").is_ok();
    let no_strength = std::env::var("KUZO_NO_STRENGTH").is_ok();
    let no_dse = std::env::var("KUZO_NO_DSE").is_ok();
    // Inline pass 默认开启。KUZO_NO_INLINE=1 可显式关闭。
    // hoisted_owners 追踪 + rebuild 分组重排确保 body 节点正确纳入 caller 范围。
    let no_inline = std::env::var("KUZO_NO_INLINE").is_ok();

    // ── Phase 1：结构变换（一次性，依赖 analysis 的 NodeId，level >= 2）──
    if level >= OptLevel::O2 && analysis.is_some() {
        let mut ctx = OptimizerContext::default();
        if !no_licm   { pass_licm(graph, &mut ctx, analysis); }
        if !no_unroll { pass_loop_unroll(graph, &mut ctx, analysis); }
        if ctx.has_changes() {
            graph.rebuild(&ctx.dead, &ctx.redirect);
        }
    }

    // ── Phase 2：固定点迭代（Inline + 传统优化，level >= 1）──
    let dbg_iter = std::env::var("KUZO_INLINE_DBG").is_ok();
    // O3 提高迭代上限；环境变量 KUZO_OPT_MAX_ITER 优先（调试用）
    let default_max_iter = if level >= OptLevel::O3 { 200 } else { 50 };
    let mut max_iter = std::env::var("KUZO_OPT_MAX_ITER")
        .ok().and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default_max_iter);
    loop {
        let mut ctx = OptimizerContext::default();

        if !no_inline { pass_inline(graph, &mut ctx, None); }
        if !no_fold   { pass_const_fold(graph, &mut ctx); }
        if !no_strength { pass_strength_reduction(graph, &mut ctx); }
        if !no_cse    { pass_cse(graph, &mut ctx, &pure_set); }
        if !no_copy   { pass_copy_prop(graph, &mut ctx); }
        if !no_dce    { pass_dce(graph, &mut ctx, &pure_set); }
        if !no_dse    { pass_dse(graph, &mut ctx); }

        if !ctx.has_changes() { break; }

        if dbg_iter {
            eprintln!("[OPT-ITER] iter={} nodes={} before rebuild", 51 - max_iter, graph.nodes.len());
        }
        let _old_to_new = graph.rebuild(&ctx.dead, &ctx.redirect);
        if dbg_iter {
            eprintln!("[OPT-ITER] iter={} nodes={} after rebuild", 51 - max_iter, graph.nodes.len());
        }

        max_iter -= 1;
        if max_iter == 0 {
            break;
        }
    }
}

// =========================================================================
// 循环变换 pass（从 LoopOptimizer.rs 合并）— LICM + 循环展开
//
// LICM：将 body_sg 中的纯不变量节点外提到函数子图帧。
// 循环展开：对静态 trip count 的小循环，复制 body_sg 节点到父帧。
// 详见 docs/superpowers/specs/2026-08-08-loop-opts-inline-design.md
// =========================================================================

/// 运行 LICM pass。
pub fn pass_licm(
    graph: &mut DataFlowGraph,
    ctx: &mut OptimizerContext,
    analysis: Option<&AnalysisReport>,
) {
    let Some(analysis) = analysis else {
        return;
    };
    let loop_analysis = &analysis.loop_analysis;
    if loop_analysis.invariants.is_empty() {
        return;
    }

    // 收集 (body_sg_id, invariants) 快照（避免持有 analysis 借用）
    let body_sgs: Vec<(SubGraphId, Vec<NodeId>)> = loop_analysis
        .invariants
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();

    let mut hoisted_count = 0;

    for (body_sg_id, invariants) in &body_sgs {
        let body_sg = &graph.subgraphs[body_sg_id.0 as usize];

        // 找到 body_sg 的 loop_parent_sg（即 loop_sg）
        let Some(loop_sg_id) = body_sg.loop_parent_sg else {
            continue;
        };

        // 外提目标：函数级子图（loop_kind == None）。
        // 不外提到循环 body_sg——body_sg 帧在 reset_loop_iteration 时帧链被
        // 设为 null，外提节点若依赖帧链访问的变量会拿到错误值。
        // 函数级子图的帧只创建一次，不涉及循环帧重置，安全。
        let func_sg_id = SubGraphId(graph.subgraphs[loop_sg_id.0 as usize].function_id);

        // 克隆不变量节点到 graph.nodes 末尾
        let mut node_map: FxHashMap<u32, NodeId> = FxHashMap::default();

        for &inv_node_id in invariants {
            let src_idx = inv_node_id.0 as usize;
            let src_node = graph.nodes[src_idx];
            let old_inputs = graph.inputs_pool.get(src_node.inputs_offset, src_node.input_count);

            // 计算新 inputs：body_sg 内引用的不变量 → 克隆节点，外部引用 → 保持原 NodeId
            let mut new_inputs: Vec<NodeId> = Vec::with_capacity(old_inputs.len());
            for &old_in in old_inputs {
                if let Some(&mapped) = node_map.get(&old_in.0) {
                    new_inputs.push(mapped);
                } else {
                    new_inputs.push(old_in); // 外部引用保持原 NodeId
                }
            }

            let new_id = graph.add_node_raw(src_node.kind, &new_inputs, src_node.compute_fn);
            let new_idx = new_id.0 as usize;

            // 克隆元数据
            graph.clone_node_metadata(src_idx, new_idx);

            // 标记为 hoisted + 设置归属子图
            graph.hoisted_node[new_idx] = true;
            graph.hoisted_owners[new_idx] = func_sg_id;

            // 克隆 const_values（如果是不变量 Const 节点）
            if let Some(cv) = &graph.const_values[src_idx] {
                graph.const_values[new_idx] = Some(*cv);
            }

            node_map.insert(inv_node_id.0, new_id);
            hoisted_count += 1;
        }

        // body_sg 内原不变量节点 → redirect 到克隆节点
        for (&old_id, &new_id) in &node_map {
            ctx.redirect.insert(NodeId(old_id), new_id);
        }

        // 不扩展 node_range：hoisted_owners 已记录归属，rebuild 按函数级子图
        // 分组重排时会将 hoisted 节点排到 func_sg 范围内。
        // 扩展 node_range 会覆盖中间其他函数的节点，导致 rebuild 后 node_range
        // 包含不属于该子图的节点 → 执行错误。
    }

    if hoisted_count > 0 {
        ctx.mutated = true;
    }
}

/// 运行循环展开 pass。
pub fn pass_loop_unroll(
    graph: &mut DataFlowGraph,
    ctx: &mut OptimizerContext,
    analysis: Option<&AnalysisReport>,
) {
    let Some(analysis) = analysis else {
        return;
    };
    let loop_analysis = &analysis.loop_analysis;
    if loop_analysis.unrollable.is_empty() {
        return;
    }

    let unrollable: Vec<(SubGraphId, UnrollInfo)> = loop_analysis
        .unrollable
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();

    let mut unroll_count = 0;

    for (loop_sg_id, unroll_info) in &unrollable {
        let loop_sg = &graph.subgraphs[loop_sg_id.0 as usize];

        // 找到 loop_sg 的 immediate parent（展开 body 放置目标）
        let Some(parent_sg_id) = graph.find_immediate_parent_sg(*loop_sg_id) else {
            continue;
        };
        // 函数级子图（hoisted_owners 归属目标）
        let func_sg_id = SubGraphId(graph.subgraphs[parent_sg_id.0 as usize].function_id);

        let body_sg = &graph.subgraphs[unroll_info.body_sg.0 as usize];
        let (body_start, body_end) = body_sg.node_range;
        let (loop_start, loop_end) = loop_sg.node_range;

        // body_sg 结构：param_0 = 迭代器, param_1 = 当前值（循环变量）
        let param_0_node = NodeId(body_start.0);
        let loop_var_node = unroll_info.loop_var_node;

        // 检查 body 是否引用 param_0（迭代器）— 如果引用则跳过展开
        let mut body_uses_iter = false;
        for idx in ((body_start.0 + 2) as usize)..(body_end.0 as usize) {
            let node = graph.nodes[idx];
            let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
            if inputs.contains(&param_0_node) {
                body_uses_iter = true;
                break;
            }
        }
        if body_uses_iter {
            continue;
        }

        // 对每次迭代克隆 body（跳过 param_0 和 param_1 参数节点）
        let body_content_start = (body_start.0 + 2) as usize;
        let body_content_end = body_end.0 as usize;
        let mut last_body_last_node: Option<NodeId> = None;

        for i in 0..unroll_info.trip_count {
            let iter_val = unroll_info.start_value + (unroll_info.step * i as i128);

            // 创建 Const 节点持有迭代值（保持原始类型）
            let const_cv = make_const_value(&unroll_info.start_const, iter_val);
            let const_node = graph.add_node_raw(NodeKind::Const, &[], CF_NOOP);
            graph.const_values[const_node.0 as usize] = Some(const_cv);
            graph.hoisted_node[const_node.0 as usize] = true;
            graph.hoisted_owners[const_node.0 as usize] = func_sg_id;

            // 克隆 body_sg 的内容节点
            let mut node_map: FxHashMap<u32, NodeId> = FxHashMap::default();
            node_map.insert(loop_var_node.0, const_node);

            for bidx in body_content_start..body_content_end {
                let src_node = graph.nodes[bidx];
                let old_id = NodeId(bidx as u32);
                let old_inputs =
                    graph.inputs_pool.get(src_node.inputs_offset, src_node.input_count);

                let mut new_inputs: Vec<NodeId> = Vec::with_capacity(old_inputs.len());
                for &old_in in old_inputs {
                    if let Some(&mapped) = node_map.get(&old_in.0) {
                        new_inputs.push(mapped);
                    } else {
                        new_inputs.push(old_in);
                    }
                }

                let new_id = graph.add_node_raw(src_node.kind, &new_inputs, src_node.compute_fn);
                let new_idx = new_id.0 as usize;
                graph.clone_node_metadata(bidx, new_idx);
                graph.hoisted_node[new_idx] = true;
                graph.hoisted_owners[new_idx] = func_sg_id;

                if let Some(cv) = &graph.const_values[bidx] {
                    graph.const_values[new_idx] = Some(*cv);
                }

                node_map.insert(old_id.0, new_id);
            }

            last_body_last_node = Some(NodeId((graph.nodes.len() - 1) as u32));
            unroll_count += 1;
        }

        // 处理 loop_sg 的 Gate 节点
        let mut gate_node: Option<NodeId> = None;
        for idx in (loop_start.0 as usize)..(loop_end.0 as usize) {
            if graph.nodes[idx].kind == NodeKind::Gate {
                gate_node = Some(NodeId(idx as u32));
                break;
            }
        }

        if let Some(gate) = gate_node {
            if let Some(last) = last_body_last_node {
                ctx.redirect.insert(gate, last);
            } else {
                ctx.dead.insert(gate);
            }
        }

        // 标记 loop_sg 的所有非 redirected 节点为 dead
        for idx in (loop_start.0 as usize)..(loop_end.0 as usize) {
            let nid = NodeId(idx as u32);
            if !ctx.redirect.contains_key(&nid) {
                ctx.dead.insert(nid);
            }
        }
        // 标记 body_sg 的所有节点为 dead
        for idx in (body_start.0 as usize)..(body_end.0 as usize) {
            let nid = NodeId(idx as u32);
            if !ctx.redirect.contains_key(&nid) {
                ctx.dead.insert(nid);
            }
        }

        // 不扩展 node_range：hoisted_owners 已记录归属，rebuild 按函数级子图
        // 分组重排时会将 hoisted 节点排到 func_sg 范围内。
    }

    if unroll_count > 0 {
        ctx.mutated = true;
    }
}

/// 根据 original 的类型创建新的 ConstValue（保持类型一致）。
fn make_const_value(original: &ConstValue, val: i128) -> ConstValue {
    use crate::ir::Ir::ConstValue::*;
    match original {
        I8(_) => I8(val as i8),
        I16(_) => I16(val as i16),
        I32(_) => I32(val as i32),
        I64(_) => I64(val as i64),
        I128(_) => I128(val),
        U8(_) => U8(val as u8),
        U16(_) => U16(val as u16),
        U32(_) => U32(val as u32),
        U64(_) => U64(val as u64),
        U128(_) => U128(val as u128),
        Isize(_) => Isize(val as isize),
        Usize(_) => Usize(val as usize),
        // 非整数类型 fallback（不应出现在 Range 展开中）
        _ => I64(val as i64),
    }
}

// =========================================================================
// IR 级函数内联 pass（从 InlineOptimizer.rs 合并）
//
// 在 CSE 之后运行，克隆小纯函数子图到调用点。
// 消除调用帧分配开销，打开更多优化机会。
//
// 内联判定：callee 子图体仅含 Const/BinOp/UnOp/FieldAccess 节点
//（此条件同时保证纯度 + 无递归，无需 AnalysisReport 映射）。
//
// 实现：克隆 callee body 节点追加到 graph.nodes 末尾，将 call_node 原地
// 改写为 CF_SEQ 序列节点（inputs = [effect_input, mapped_return]，
// 等待 effect_input 就绪后转发 mapped_return）。扩展 caller 函数子图
// node_range 将 body 节点纳入范围。rebuild 自动重建 downstreams 和
// node_range，确保数据流正确传播。
// =========================================================================

/// IR 级内联的最大 callee 节点数
const MAX_INLINE_NODES: usize = 20;

/// 运行 IR 级函数内联 pass。
///
/// `_analysis` 保留用于未来扩展（当前通过 body 结构检查保证安全性）。
pub fn pass_inline(
    graph: &mut DataFlowGraph,
    ctx: &mut OptimizerContext,
    _analysis: Option<&crate::pass::Analyzer::AnalysisReport>,
) {
    let candidates = collect_inline_candidates(graph);
    if candidates.is_empty() {
        return;
    }

    let inline_limit = std::env::var("KUZO_INLINE_LIMIT")
        .ok().and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    if std::env::var("KUZO_INLINE_DBG").is_ok() {
        eprintln!("[INLINE] {} candidates", candidates.len());
    }
    for (i, candidate) in candidates.iter().enumerate().take(inline_limit) {
        if std::env::var("KUZO_INLINE_DBG").is_ok() {
            let callee_sg = &graph.subgraphs[candidate.callee_sg.0 as usize];
            let callee_size = (callee_sg.node_range.1.0 - callee_sg.node_range.0.0) as usize;
            eprintln!("[INLINE] #{} call_node={} callee_sg={} caller_sg={} (size={}, params={}, return={})",
                i, candidate.call_node.0, candidate.callee_sg.0, candidate.caller_func_sg.0,
                callee_size, callee_sg.param_count, callee_sg.return_node.0);
        }
        inline_call(graph, ctx, candidate);
    }
}

struct InlineCandidate {
    /// Call 节点的全局 NodeId
    call_node: NodeId,
    /// callee 子图 ID
    callee_sg: SubGraphId,
    /// caller 函数子图 ID（用于扩展 node_range）
    caller_func_sg: SubGraphId,
}

fn collect_inline_candidates(graph: &DataFlowGraph) -> Vec<InlineCandidate> {
    let mut candidates = Vec::new();
    let pure_set = crate::ir::Ir::pure_compute_fn_set();

    for (idx, node) in graph.nodes.iter().enumerate() {
        // 只处理 CF_CALL_LAUNCH（sync 调用）
        if node.compute_fn != CF_CALL_LAUNCH {
            continue;
        }

        let nid = NodeId(idx as u32);

        // 非尾调用（尾调用有帧复用优化，不宜内联）
        if graph.tail_call_flags.get(idx).copied().unwrap_or(false) {
            continue;
        }

        // 有 call_targets
        let Some(Some(callee_sg_id)) = graph.call_targets.get(idx) else {
            continue;
        };
        let callee_sg = &graph.subgraphs[callee_sg_id.0 as usize];

        // sync 函数
        if callee_sg.has_suspend {
            continue;
        }

        // 无 upvalue（upvalue 需帧链注入，内联后无法处理）
        if callee_sg.upvalue_count > 0 {
            continue;
        }

        // 节点数限制
        let callee_size = (callee_sg.node_range.1.0 - callee_sg.node_range.0.0) as usize;
        if callee_size > MAX_INLINE_NODES {
            continue;
        }

        // return_node 必须在 callee node_range 内，否则 inline_call 的
        // node_map 无法映射 return_node → mapped_return 回退为原始 callee 节点 id，
        // CF_SEQ 引用 callee 帧不可达的节点 → 返回错误值 → 调用方逻辑错乱。
        let ret = callee_sg.return_node;
        if ret.0 < callee_sg.node_range.0.0 || ret.0 >= callee_sg.node_range.1.0 {
            if std::env::var("KUZO_INLINE_DBG").is_ok() {
                eprintln!("[INLINE-SKIP] call_node={} callee_sg={} return={} not in range [{},{})",
                    nid.0, callee_sg_id.0, ret.0,
                    callee_sg.node_range.0.0, callee_sg.node_range.1.0);
            }
            continue;
        }

        // body 内只有 Const/BinOp/UnOp/FieldAccess 且 compute_fn 在 pure_set 中
        //（此条件同时保证：纯函数 + 无递归 + 无控制流 + 无构造副作用）
        let (cs, ce) = callee_sg.node_range;
        let mut safe_body = true;
        for cidx in (cs.0 as usize)..(ce.0 as usize) {
            let cn = &graph.nodes[cidx];
            if !matches!(
                cn.kind,
                NodeKind::Const | NodeKind::BinOp | NodeKind::UnOp | NodeKind::FieldAccess
            ) {
                safe_body = false;
                break;
            }
            // 参数占位节点（cf=CF_NOOP, const=false）跳过 pure_set 检查
            if cn.compute_fn != crate::ir::Ir::CF_NOOP
                && !pure_set.contains(&cn.compute_fn)
            {
                safe_body = false;
                break;
            }
            // 所有 inputs 必须在 callee_sg 范围内（无跨子图引用）。
            // 有外部引用的函数内联后，引用节点在 caller 帧中不可达，会导致读到错误值。
            let cn_inputs = graph.inputs_pool.get(cn.inputs_offset, cn.input_count);
            for &inp in cn_inputs {
                if inp.0 < cs.0 || inp.0 >= ce.0 {
                    safe_body = false;
                    break;
                }
            }
            if !safe_body { break; }
        }
        if !safe_body {
            continue;
        }

        // 找到 caller 的函数子图
        let Some(caller_func_sg) = graph.find_function_sg_for_node(nid) else {
            continue;
        };

        // 安全性检查：call 节点必须直接在函数级子图中（非嵌套在 Gate 分支/循环体内）。
        // 否则内联 body 会被放到函数级，导致无条件执行，绕过 Gate 条件。
        // 例如 if cond { divFunc(a, b) } 内联后 divFunc 的除法会无条件执行。
        let Some(innermost_sg) = graph.find_innermost_sg_for_node(nid) else {
            continue;
        };
        if innermost_sg != caller_func_sg {
            continue;
        }

        // 所有 call 节点输入必须在 caller_func_sg node_range 内。
        // 若 effect_input 来自外层函数子图（非逃逸 lambda / Gate 分支被误判为函数级子图），
        // inline 后 CF_SEQ 会引用 caller 帧不可达的节点，导致 pending_inputs 永不归零 → 死锁。
        let caller_range = graph.subgraphs[caller_func_sg.0 as usize].node_range;
        let call_inputs =
            graph.inputs_pool.get(node.inputs_offset, node.input_count);
        let mut inputs_in_range = true;
        for &inp in call_inputs {
            if inp.0 < caller_range.0 .0 || inp.0 >= caller_range.1 .0 {
                inputs_in_range = false;
                break;
            }
        }
        if !inputs_in_range {
            if std::env::var("KUZO_INLINE_DBG").is_ok() {
                eprintln!("[INLINE-SKIP] call_node={} has inputs outside caller_func_sg={} range=[{},{})",
                    nid.0, caller_func_sg.0, caller_range.0 .0, caller_range.1 .0);
            }
            continue;
        }

        candidates.push(InlineCandidate {
            call_node: nid,
            callee_sg: *callee_sg_id,
            caller_func_sg,
        });
    }

    candidates
}

fn inline_call(graph: &mut DataFlowGraph, ctx: &mut OptimizerContext, candidate: &InlineCandidate) {
    let callee_sg = &graph.subgraphs[candidate.callee_sg.0 as usize];
    let (callee_start, callee_end) = callee_sg.node_range;
    let call_node = candidate.call_node;
    let param_count = callee_sg.param_count as usize;
    let return_node = callee_sg.return_node;

    // 获取 Call 节点的输入（前 param_count 个是参数，末尾可能有 effect 依赖）
    let call_node_struct = graph.nodes[call_node.0 as usize];
    let call_inputs =
        graph.inputs_pool.get(call_node_struct.inputs_offset, call_node_struct.input_count);

    if std::env::var("KUZO_INLINE_DBG").is_ok() {
        let ret_node = &graph.nodes[return_node.0 as usize];
        let ret_inputs = graph.inputs_pool.get(ret_node.inputs_offset, ret_node.input_count);
        eprintln!("[INLINE-DBG] call_node={} inputs={:?} callee=[{},{}) params={} return={} ret_kind={:?} ret_cf={} ret_inputs={:?}",
            call_node.0, call_inputs, callee_start.0, callee_end.0, param_count, return_node.0,
            ret_node.kind, ret_node.compute_fn.0, ret_inputs);
    }

    // 建立 callee 内部 NodeId → caller 中 NodeId 的映射
    let mut node_map: FxHashMap<u32, NodeId> = FxHashMap::default();

    // 映射参数节点 → call 的实际参数（前 param_count 个 input）
    for i in 0..param_count {
        let param_node_id = NodeId(callee_start.0 + i as u32);
        if i < call_inputs.len() {
            node_map.insert(param_node_id.0, call_inputs[i]);
        }
    }

    // 保留 effect 依赖（call_inputs 末尾如果有非参数 input）。
    // 必须在 body 克隆循环之前拷出，以结束对 graph.inputs_pool 的不可变借用，
    // 否则后续 graph.add_node_raw 的可变借用会与之冲突。
    let effect_input = if call_inputs.len() > param_count {
        Some(call_inputs[param_count])
    } else {
        None
    };

    // 克隆 body 节点（跳过参数占位节点）— 两遍克隆确保前向引用正确映射。
    // 单遍克隆时，若节点 A 引用后面的节点 B，A 克隆时 B 未在 node_map 中，
    // new_inputs 保持 B 旧 id，rebuild 会把 B 旧 id 映射到原 callee 节点
    // （在 callee_sg 范围内），而非克隆节点 → caller 帧无法访问 → 值表错位/死锁。
    let body_start = callee_start.0 as usize + param_count;
    let body_end = callee_end.0 as usize;

    // 快照 body 节点信息（结束对 graph 的不可变借用，以便后续 add_node_raw 可变借用）
    let body_snapshots: Vec<(usize, NodeKind, ComputeFnId, Vec<NodeId>)> =
        (body_start..body_end)
            .map(|src_idx| {
                let src_node = graph.nodes[src_idx];
                let old_inputs =
                    graph.inputs_pool.get(src_node.inputs_offset, src_node.input_count).to_vec();
                (src_idx, src_node.kind, src_node.compute_fn, old_inputs)
            })
            .collect();

    // 第一遍：为所有 body 节点分配 new_id（空 inputs），建立完整 node_map
    for &(src_idx, kind, cf, _) in &body_snapshots {
        let new_id = graph.add_node_raw(kind, &[], cf);
        let new_idx = new_id.0 as usize;
        graph.clone_node_metadata(src_idx, new_idx);
        graph.hoisted_node[new_idx] = true;
        graph.hoisted_owners[new_idx] = candidate.caller_func_sg;
        if let Some(cv) = &graph.const_values[src_idx] {
            graph.const_values[new_idx] = Some(cv.clone());
        }
        node_map.insert(src_idx as u32, new_id);
    }

    // 第二遍：重映射 inputs（此时 node_map 已包含所有 body 节点，前向引用可正确解析）
    for (src_idx, _, _, old_inputs) in &body_snapshots {
        let &new_id = node_map.get(&(*src_idx as u32)).unwrap();
        let mut new_inputs: Vec<NodeId> = Vec::with_capacity(old_inputs.len());
        for &old_in in old_inputs {
            if let Some(&mapped) = node_map.get(&old_in.0) {
                new_inputs.push(mapped);
            } else {
                new_inputs.push(old_in);
            }
        }
        let new_offset = graph.inputs_pool.push(&new_inputs);
        graph.nodes[new_id.0 as usize].inputs_offset = new_offset;
        graph.nodes[new_id.0 as usize].input_count = new_inputs.len() as u8;
    }

    // 原地替换 call_node：将其变为 CF_SEQ 序列节点。
    // CF_SEQ (idx 48) 等待所有输入就绪后返回最后一个输入的值（序列语义）。
    // 与 Builder.rs 的 chain_effects 相同模式：inputs = [prev_effect, current_value]。
    // 不使用 redirect（redirect 会在 rebuild 时移除 call_node，导致 effect 链断裂）。
    //
    // inputs = [effect_input?, mapped_return]
    // - effect_input 作为数据依赖边，强制前序副作用先于 call_node 完成（顺序约束）
    // - mapped_return 是内联 body 的返回值节点（最后输入，被转发给 call_node 的 downstreams）
    // rebuild 后 downstreams 自动重建：effect_input 和 mapped_return 的 downstreams
    // 都会包含 call_node，保证 call_node 在两者就绪后才执行。
    let mapped_return = node_map.get(&return_node.0).copied().unwrap_or(return_node);

    // CF_SEQ inputs = [effect_input?, mapped_return]
    // effect_input 必须在 caller_func_sg 的 node_range 内：caller 帧的 value_table
    // 只覆盖 node_range 范围，范围外的输入永远不 ready → CF_SEQ 死锁 → 帧卡住。
    // collect_inline_candidates 保证 call_node 在函数级 sg，但 effect_input 可能来自
    // 嵌套子图（如 Gate 分支的 effect 链），此时去掉 effect_input（effect 顺序由
    // downstreams 隐式保证：mapped_return 的依赖链自然串行化副作用）。
    let caller_sg_obj = &graph.subgraphs[candidate.caller_func_sg.0 as usize];
    let caller_range = caller_sg_obj.node_range;
    let mut new_inputs = Vec::with_capacity(2);
    if let Some(eff) = effect_input {
        if eff.0 >= caller_range.0 .0 && eff.0 < caller_range.1 .0 {
            new_inputs.push(eff);
        } else if std::env::var("KUZO_INLINE_DBG").is_ok() {
            let eff_sg = graph.find_innermost_sg_for_node(eff);
            eprintln!("[INLINE-WARN] call_node={} effect_input={} outside caller_func_sg={} range=[{},{}) — dropped | caller_sg function_id={} loop_kind={:?} loop_parent={:?} param_count={} upvalue_count={} | eff_sg={:?} eff_kind={:?} eff_cf={}",
                call_node.0, eff.0, candidate.caller_func_sg.0, caller_range.0 .0, caller_range.1 .0,
                caller_sg_obj.function_id, caller_sg_obj.loop_kind, caller_sg_obj.loop_parent_sg,
                caller_sg_obj.param_count, caller_sg_obj.upvalue_count,
                eff_sg.map(|s| s.0), graph.nodes[eff.0 as usize].kind, graph.nodes[eff.0 as usize].compute_fn.0);
        }
    }
    new_inputs.push(mapped_return);
    let new_offset = graph.inputs_pool.push(&new_inputs);

    // 原地修改 call_node
    let cn = &mut graph.nodes[call_node.0 as usize];
    cn.compute_fn = CF_SEQ;
    cn.inputs_offset = new_offset;
    cn.input_count = new_inputs.len() as u8;
    cn.kind = NodeKind::BinOp; // CF_SEQ 是 BinOp kind

    // 清除 call_node 的 call_targets 元数据（不再是 Call 节点）
    graph.call_targets[call_node.0 as usize] = None;
    graph.tail_call_flags[call_node.0 as usize] = false;

    // 不扩展 node_range：hoisted_owners 已记录归属，rebuild 按函数级子图
    // 分组重排时会将 body 节点排到 caller_func_sg 范围内。

    ctx.mutated = true;
}
