//! Optimizer.rs — IR post-optimizer
//!
//! Performs fixpoint-iterated graph-level optimization on the DataFlowGraph produced by IrBuilder.
//! Pass pipeline (per round): Inline → ConstFold → StrengthRed → CSE → CopyProp → DCE → DSE.
//! Structural transformation passes (LICM/Unroll/Inline) run before traditional optimization;
//! their redirect/dead output is compacted by the late rebuild in a single pass. Zero engine-side changes.
//! See docs/superpowers/plans/2026-08-08-loop-opts-inline.md

use crate::ir::Ir::{
    CF_ARRAY_STORE, CF_CALL_LAUNCH, CF_GLOBAL_LOAD, CF_GLOBAL_STORE, CF_NOOP,
    CF_RECORD_FIELD_SET, CF_SEQ, CF_WRITEBACK, ConstValue, ComputeFnId, DataFlowGraph,
    Node, NodeId, NodeKind, SubGraphId,
};
use crate::pass::Analyzer::{AnalysisReport, UnrollInfo};
use pastey::paste;
use rustc_hash::{FxHashMap, FxHashSet};

// =========================================================================
// ConstValue extractors — type-safe extraction of raw values from ConstValue
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

/// Extracts two values of the same type from args.
fn two<T>(args: &[ConstValue], extract: fn(&ConstValue) -> Option<T>) -> Option<(T, T)> {
    Some((extract(args.get(0)?)?, extract(args.get(1)?)?))
}

// =========================================================================
// try_fold — constant folding dispatch
// =========================================================================

/// Attempts compile-time evaluation for the given compute_fn and constant arguments.
/// Returns None if folding is not possible (type mismatch or non-foldable op).
pub fn try_fold(cf: ComputeFnId, args: &[ConstValue]) -> Option<ConstValue> {
    use crate::value as V;
    match cf.0 {
        // ── Legacy i32 arithmetic (1,3,5,6,7) ──
        1  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_add_i32(a, b))) }
        3  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_mul_i32(a, b))) }
        5  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_sub_i32(a, b))) }
        6  => { let (a, b) = two(args, cv_i32)?; V::arith_div_i32(a, b).map(|v| ConstValue::I32(v)) }
        7  => { let (a, b) = two(args, cv_i32)?; V::arith_mod_i32(a, b).map(|v| ConstValue::I32(v)) }
        // ── Legacy i32 comparison (4,8,9,10,11,12) → bool ──
        4  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a <= b)) }
        8  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a == b)) }
        9  => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a != b)) }
        10 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a < b)) }
        11 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a > b)) }
        12 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::Bool(a >= b)) }
        // ── Legacy f64 arithmetic (2,13,14,15) ──
        2  => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_add_f64(a, b))) }
        13 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_sub_f64(a, b))) }
        14 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_mul_f64(a, b))) }
        15 => { let (a, b) = two(args, cv_f64)?; Some(ConstValue::F64(V::arith_div_f64(a, b))) }
        // ── Legacy f64 comparison (16-21) → bool ──
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

        // ── i64 arithmetic + comparison (50-61) ──
        50 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_add_i64(a, b))) }
        51 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_sub_i64(a, b))) }
        52 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_mul_i64(a, b))) }
        53 => { let (a, b) = two(args, cv_i64)?; V::arith_div_i64(a, b).map(|v| ConstValue::I64(v)) }
        54 => { let (a, b) = two(args, cv_i64)?; V::arith_mod_i64(a, b).map(|v| ConstValue::I64(v)) }
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

        // ── i128 arithmetic + comparison (64-75) ──
        64 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_add_i128(a, b))) }
        65 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_sub_i128(a, b))) }
        66 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_mul_i128(a, b))) }
        67 => { let (a, b) = two(args, cv_i128)?; V::arith_div_i128(a, b).map(|v| ConstValue::I128(v)) }
        68 => { let (a, b) = two(args, cv_i128)?; V::arith_mod_i128(a, b).map(|v| ConstValue::I128(v)) }
        69 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a == b)) }
        70 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a != b)) }
        71 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a < b)) }
        72 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a > b)) }
        73 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a <= b)) }
        74 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::Bool(a >= b)) }
        75 => { let a = cv_i128(args.get(0)?)?; Some(ConstValue::I128(V::arith_neg_i128(a))) }

        // ── Bitwise i32 (77-79) ──
        77 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_bitand_i32(a, b))) }
        78 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_bitor_i32(a, b))) }
        79 => { let (a, b) = two(args, cv_i32)?; Some(ConstValue::I32(V::arith_bitxor_i32(a, b))) }
        // ── Bitwise i64 (80-82) ──
        80 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_bitand_i64(a, b))) }
        81 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_bitor_i64(a, b))) }
        82 => { let (a, b) = two(args, cv_i64)?; Some(ConstValue::I64(V::arith_bitxor_i64(a, b))) }
        // ── Bitwise i128 (83-85) ──
        83 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_bitand_i128(a, b))) }
        84 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_bitor_i128(a, b))) }
        85 => { let (a, b) = two(args, cv_i128)?; Some(ConstValue::I128(V::arith_bitxor_i128(a, b))) }
        // ── Shifts i32 (86-87): shift amount is i32 ──
        86 => { let a = cv_i32(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; V::arith_shl_i32(a, s).map(|v| ConstValue::I32(v)) }
        87 => { let a = cv_i32(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; V::arith_shr_i32(a, s).map(|v| ConstValue::I32(v)) }
        // ── Shifts i64 (88-89) ──
        88 => { let a = cv_i64(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; V::arith_shl_i64(a, s).map(|v| ConstValue::I64(v)) }
        89 => { let a = cv_i64(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; V::arith_shr_i64(a, s).map(|v| ConstValue::I64(v)) }
        // ── Shifts i128 (90-91) ──
        90 => { let a = cv_i128(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; V::arith_shl_i128(a, s).map(|v| ConstValue::I128(v)) }
        91 => { let a = cv_i128(args.get(0)?)?; let s = cv_i32(args.get(1)?)?; V::arith_shr_i128(a, s).map(|v| ConstValue::I128(v)) }

        // ── All primitive type arithmetic (92-259) ──
        id if id >= 92 && id <= 259 => fold_basic_range(id, args),

        _ => None,
    }
}

/// Folding macro for 12 integer-type operations.
macro_rules! fold_int_arith {
    ($args:expr, $op:expr, $cv:ident, $ext:ident, $ty:ident) => { paste! {
        match $op {
            0 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_add_$ty>](a, b))) }
            1 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_sub_$ty>](a, b))) }
            2 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_mul_$ty>](a, b))) }
            // div/mod/shl/shr return Option; None (divide-by-zero/overflow) means no folding, leave it for runtime to return Throw
            3 => { let (a, b) = two($args, $ext)?; crate::value::[<arith_div_$ty>](a, b).map(|v| ConstValue::$cv(v)) }
            4 => { let (a, b) = two($args, $ext)?; crate::value::[<arith_mod_$ty>](a, b).map(|v| ConstValue::$cv(v)) }
            5 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_bitand_$ty>](a, b))) }
            6 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_bitor_$ty>](a, b))) }
            7 => { let (a, b) = two($args, $ext)?; Some(ConstValue::$cv(crate::value::[<arith_bitxor_$ty>](a, b))) }
            8 => { let a = $ext($args.get(0)?)?; let s = cv_i32($args.get(1)?)?; crate::value::[<arith_shl_$ty>](a, s).map(|v| ConstValue::$cv(v)) }
            9 => { let a = $ext($args.get(0)?)?; let s = cv_i32($args.get(1)?)?; crate::value::[<arith_shr_$ty>](a, s).map(|v| ConstValue::$cv(v)) }
            10 => { let a = $ext($args.get(0)?)?; Some(ConstValue::$cv(crate::value::[<arith_neg_$ty>](a))) }
            11 => { let a = $ext($args.get(0)?)?; Some(ConstValue::$cv(crate::value::[<arith_bitnot_$ty>](a))) }
            _ => None,
        }
    }};
}

/// Folding macro for 6 floating-point-type operations.
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

/// Primitive type arithmetic folding (92-259).
/// Integers: 12 types × 12 operations (92-235), floats: 4 types × 6 operations (236-259).
/// f16/f128 have no ConstValue variant; skipped (returns None).
fn fold_basic_range(id: u32, args: &[ConstValue]) -> Option<ConstValue> {
    if id <= 235 {
        // Integers: 12 types × 12 operations (92-235)
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
        // Floats: 4 types × 6 operations (236-259)
        let offset = id - 236;
        let type_idx = (offset / 6) as usize;
        let op_idx = (offset % 6) as usize;
        // op: 0=add 1=sub 2=mul 3=div 4=mod 5=neg
        // f16 (type_idx=0) and f128 (type_idx=3) have no ConstValue variant
        match type_idx {
            1 => return fold_float_arith!(args, op_idx, F32, cv_f32, f32),
            2 => return fold_float_arith!(args, op_idx, F64, cv_f64, f64),
            _ => return None,
        }
    }
}

// =========================================================================
// OptimizerContext — optimization-time transformation tracking
// =========================================================================

/// Accumulated transformations during optimization: dead set and redirect map.
/// Consumed by DataFlowGraph::rebuild after fixpoint convergence for a single graph rebuild.
#[derive(Default)]
pub struct OptimizerContext {
    /// Dead node set (DCE-marked)
    pub dead: FxHashSet<NodeId>,
    /// Redirect map: old_node_id → new_node_id (produced by CSE/CopyProp)
    pub redirect: FxHashMap<NodeId, NodeId>,
    /// Dead subgraph set (FuncDCE-marked): a function subgraph plus every nested
    /// branch/loop/defer-body subgraph of an unreachable function. Consumed by
    /// `rebuild` for subgraph compaction (SubGraphId renumbering).
    pub dead_sgs: FxHashSet<SubGraphId>,
    /// Whether ConstFold modified any node (modifies in place, produces no redirect)
    pub mutated: bool,
    /// Number of nodes folded by ConstFold this round (for debugging)
    pub cf_folded_count: usize,
}

impl OptimizerContext {
    /// Recursively resolves a redirect to its final target.
    #[inline]
    pub fn resolve(&self, id: NodeId) -> NodeId {
        let mut cur = id;
        while let Some(&next) = self.redirect.get(&cur) {
            cur = next;
        }
        cur
    }

    /// Whether a node is live (not dead and not eliminated by redirect).
    #[inline]
    pub fn is_live(&self, id: NodeId) -> bool {
        !self.dead.contains(&id) && !self.redirect.contains_key(&id)
    }

    /// Whether any transformation occurred this round.
    #[inline]
    pub fn has_changes(&self) -> bool {
        self.mutated || !self.dead.is_empty() || !self.redirect.is_empty() || !self.dead_sgs.is_empty()
    }
}

/// Checks whether a node has side effects (cannot be eliminated or redirected by CSE/CopyProp/DCE).
fn has_side_effect(graph: &DataFlowGraph, idx: usize) -> bool {
    graph.writeback_targets.get(idx).map_or(false, |o| o.is_some())
    || graph.field_set_names.get(idx).map_or(false, |o| o.is_some())
    || graph.global_store_slots.get(idx).map_or(false, |o| o.is_some())
    || crate::ir::Ir::is_control_flow_compute_fn(graph.nodes[idx].compute_fn)
    || graph.ffi_call_names.get(idx).map_or(false, |o| o.is_some())
    || graph.tail_call_flags.get(idx).copied().unwrap_or(false)
}

/// Collects all writeback target node IDs.
/// These nodes' runtime values are overwritten by writeback; they cannot serve as CSE merge
/// targets or ConstFold constant sources.
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
// Liveness — the single liveness authority
// =========================================================================

/// Node- and function-level liveness, computed from ONE enumeration of
/// out-of-band NodeId references (the NodeRef door: `for_each_node_ref` /
/// `node_meta_refs`). Consumers:
/// - `pass_dce` / `pass_dse` → node set (`compute_live_nodes`);
/// - `pass_func_dce` → function set + node→function attribution
///   (`compute_liveness`).
/// Before this struct existed, each pass hand-rolled its own reference list
/// and they had already diverged from rebuild's remap list
/// (`upvalue_outer_nodes` / `reset_plan` were remapped but never seeded →
/// `remap_n` panics), and node- vs function-level liveness were two separate
/// BFS runs that could disagree about a call site and its callee body.
pub struct Liveness {
    /// Live node ids (redirect-resolved).
    pub nodes: FxHashSet<NodeId>,
    /// Reachable function ids (owning-function attribution).
    pub funcs: FxHashSet<u32>,
    /// Node idx → owning function id (u32::MAX = unattributed).
    pub node_owner: Vec<u32>,
}

/// Phase 1 only: the node liveness closure. Seeds = every metadata NodeId
/// (via the door — subgraph anchors, defer registration, event declarations,
/// upvalues, loop reset plans, await sources, writeback targets, gate/select
/// refs) + all side-effecting nodes; closure over inputs and per-node
/// metadata refs (same door). Effect edges are inputs, so they are traversed
/// like any other edge.
pub fn compute_live_nodes(graph: &DataFlowGraph, ctx: &OptimizerContext) -> FxHashSet<NodeId> {
    let mut live: FxHashSet<NodeId> = FxHashSet::default();
    let mut stack: Vec<NodeId> = Vec::new();
    let add = |id: NodeId, live: &mut FxHashSet<NodeId>, stack: &mut Vec<NodeId>| {
        let r = ctx.resolve(id);
        if live.insert(r) { stack.push(r); }
    };

    // Seeds 1: every metadata NodeId through the door.
    let mut refs: Vec<NodeId> = Vec::new();
    graph.for_each_node_ref(|_site, _owner, id| refs.push(id));
    for id in refs { add(id, &mut live, &mut stack); }

    // Seeds 2: side-effecting nodes (never removable by node DCE).
    for idx in 0..graph.nodes.len() {
        if has_side_effect(graph, idx) {
            add(NodeId(idx as u32), &mut live, &mut stack);
        }
    }

    // Closure: inputs + per-node metadata refs (the same door).
    let mut meta: Vec<NodeId> = Vec::new();
    while let Some(n) = stack.pop() {
        let idx = n.0 as usize;
        let node = graph.nodes[idx];
        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        for &input in inputs {
            add(input, &mut live, &mut stack);
        }
        meta.clear();
        graph.node_meta_refs(idx, &mut meta);
        for &id in &meta { add(id, &mut live, &mut stack); }
    }
    live
}

/// Phase 1 + phase 2: adds the function-level reachability closure consumed
/// by `pass_func_dce`. Phase 2 roots: entry function + vtable fallback
/// targets; edges: cross-sg references of RETAINED nodes (`ctx.is_live` —
/// rebuild retains exactly those, and a Call node that is neither
/// closure-live nor dead yet still binds its callee, so retention — not the
/// phase-1 closure — is the criterion the kill set must agree with).
pub fn compute_liveness(graph: &DataFlowGraph, ctx: &OptimizerContext) -> Liveness {
    let nodes = compute_live_nodes(graph, ctx);

    // Node → owning function subgraph (same attribution rebuild step 1a uses:
    // function-level ranges cover nested branch/loop ranges; hoisted nodes
    // resolve through hoisted_owners, upcasting a branch-sg owner to its
    // function).
    let total = graph.nodes.len();
    let mut node_owner: Vec<u32> = vec![u32::MAX; total];
    for sg in &graph.subgraphs {
        if sg.loop_kind != crate::ir::Ir::LoopKind::None || sg.loop_parent_sg.is_some() {
            continue;
        }
        if sg.id.0 != sg.function_id {
            continue;
        }
        let start = sg.node_range.0.0 as usize;
        let end = (sg.node_range.1.0 as usize).min(total);
        for idx in start..end {
            node_owner[idx] = sg.function_id;
        }
    }
    for idx in 0..total {
        if graph.hoisted_node[idx] && node_owner[idx] == u32::MAX {
            let raw_owner = graph.hoisted_owners[idx].0 as usize;
            node_owner[idx] = if raw_owner < graph.subgraphs.len() {
                graph.subgraphs[raw_owner].function_id
            } else {
                graph.hoisted_owners[idx].0
            };
        }
    }

    let mut funcs: FxHashSet<u32> = FxHashSet::default();
    if let Some(entry_sg) = graph.entry_subgraph {
        let entry_func = graph.subgraphs[entry_sg.0 as usize].function_id;
        let mut func_stack: Vec<u32> = Vec::new();
        let push_func = |f: u32, funcs: &mut FxHashSet<u32>, stack: &mut Vec<u32>| {
            if funcs.insert(f) { stack.push(f); }
        };
        push_func(entry_func, &mut funcs, &mut func_stack);
        // Concrete-type virtual dispatch may reach these impls without any
        // static Call edge.
        for sg in graph.vtable_fallback_dispatch.values() {
            push_func(graph.subgraphs[sg.0 as usize].function_id, &mut funcs, &mut func_stack);
        }
        // Retained nodes grouped by owning function. Unattributed retained
        // nodes are attributed to the entry function — the conservative
        // direction that keeps callees alive.
        let mut funcs_of: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
        for (idx, &owner) in node_owner.iter().enumerate() {
            let id = NodeId(idx as u32);
            if !ctx.is_live(id) {
                continue;
            }
            let owner = if owner == u32::MAX { entry_func } else { owner };
            funcs_of.entry(owner).or_default().push(idx);
        }
        while let Some(f) = func_stack.pop() {
            let Some(nodes_of_f) = funcs_of.get(&f) else { continue };
            for &idx in nodes_of_f {
                if let Some(t) = graph.call_targets.get(idx).and_then(|o| *o) {
                    push_func(graph.subgraphs[t.0 as usize].function_id, &mut funcs, &mut func_stack);
                }
                if let Some(ci) = graph.closure_infos.get(idx).and_then(|o| o.as_ref()) {
                    push_func(graph.subgraphs[ci.subgraph_id.0 as usize].function_id, &mut funcs, &mut func_stack);
                }
                if let Some(pi) = graph.partial_infos.get(idx).and_then(|o| o.as_ref()) {
                    push_func(graph.subgraphs[pi.subgraph_id.0 as usize].function_id, &mut funcs, &mut func_stack);
                }
                if let Some(li) = graph.lazy_construct_infos.get(idx).and_then(|o| o.as_ref()) {
                    push_func(graph.subgraphs[li.thunk_sg.0 as usize].function_id, &mut funcs, &mut func_stack);
                }
                if let Some(ti) = graph.trait_construct_infos.get(idx).and_then(|o| o.as_ref()) {
                    for m in &ti.methods {
                        push_func(graph.subgraphs[m.subgraph_id.0 as usize].function_id, &mut funcs, &mut func_stack);
                    }
                }
                // Gate/select branch targets: normally same-function, but
                // build-time inlining (compile_inline_expansion) can leave a
                // live gate in one function pointing at a branch sg
                // REGISTERED under another — including analyzer-dead branches
                // whose placeholders were never compiled. The runtime really
                // dispatches into those sgs, so their owning functions (and
                // the anchor nodes those sgs resolve values through) must
                // stay alive.
                if let Some(gb) = graph.gate_branches.get(idx).and_then(|o| o.as_ref()) {
                    for (_, bsg, _) in &gb.branches {
                        push_func(graph.subgraphs[bsg.0 as usize].function_id, &mut funcs, &mut func_stack);
                    }
                }
                if let Some(si) = graph.select_infos.get(idx).and_then(|o| o.as_ref()) {
                    for sb in &si.branches {
                        push_func(graph.subgraphs[sb.subgraph_id.0 as usize].function_id, &mut funcs, &mut func_stack);
                    }
                }
            }
        }
    }
    Liveness { nodes, funcs, node_owner }
}

// =========================================================================
// Pass: ConstFold — constant folding
// =========================================================================

/// ConstFold pass: BinOp/UnOp with all-Const inputs → fold to Const.
/// Modifies the original node in place (does not create new nodes; preserves NodeId, ensuring it
/// stays within node_range).
/// Scans repeatedly within a single round until no new folds occur (chained folding: after A→Const,
/// B depending on A can also fold).
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
            // Side-effecting nodes cannot be folded: inputs of writeback targets change at runtime,
            // and the initial constant value cannot replace the runtime computation.
            if has_side_effect(graph, idx) { continue; }

            let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
            let mut arg_values: Vec<ConstValue> = Vec::with_capacity(inputs.len());
            let mut all_const = true;
            for &input in inputs {
                let resolved = ctx.resolve(input);
                // Writeback targets' runtime values change; cannot be used as constant sources
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

        // Modify the original node in place to Const
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

    // If any nodes were folded, mark mutated so the outer fixpoint continues to the next round:
    // new constants may create new CSE/DCE opportunities (e.g., Const nodes can be eliminated).
    // This converges — each round of ConstFold folds at least one node; the total node count is finite,
    // so folded_this_round eventually becomes empty and the loop exits.
    if total_folded > 0 {
        ctx.mutated = true;
        ctx.cf_folded_count += total_folded;
    }
}

// =========================================================================
// Pass: CSE — common subexpression elimination
// =========================================================================

/// CSE pass: pure nodes with the same (compute_fn, resolved_inputs, metadata_hash) → merge.
/// The first occurrence is kept; subsequent ones are redirected to it — but ONLY
/// when the kept node's region structurally dominates the duplicate's (W3:
/// region-dominance legality replacing the innermost_sg_start key from Bug #45).
/// Sibling branch sub-graphs (if-else/match arms) never dominate each other, so
/// mutually-exclusive computations are still never merged; a function-level
/// computation dominating an identical one inside a branch/loop body IS now
/// merged (previously blocked). The metadata hash ensures nodes with different
/// per-node metadata (pattern_field_indices/pattern_ctor_names/field_access_infos,
/// etc.) are not incorrectly merged.
pub fn pass_cse(graph: &DataFlowGraph, ctx: &mut OptimizerContext, pure_set: &FxHashSet<ComputeFnId>) {
    let mut seen: FxHashMap<(ComputeFnId, Vec<NodeId>, u64), NodeId> = FxHashMap::default();
    let wb_targets = collect_writeback_targets(graph);
    let regions = crate::ir::Region::RegionTree::build(graph);
    let innermost = regions.innermost_all(graph.nodes.len());

    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(idx as u32);
        if !ctx.is_live(id) { continue; }
        // Skip already-redirected nodes (avoid repeatedly producing the same redirect)
        if ctx.redirect.contains_key(&id) { continue; }
        if !pure_set.contains(&node.compute_fn) { continue; }
        if node.kind == NodeKind::Gate { continue; }
        // Side-effecting nodes cannot be redirected (writeback/field_set/global_store, etc.)
        if has_side_effect(graph, idx) { continue; }
        // Writeback targets' runtime values change; cannot serve as CSE merge targets or sources
        if wb_targets.contains(&id) { continue; }

        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        let resolved: Vec<NodeId> = inputs.iter().map(|&i| ctx.resolve(i)).collect();
        let meta_hash = graph.cse_metadata_hash(idx);
        let key = (node.compute_fn, resolved, meta_hash);
        if let Some(&existing) = seen.get(&key) {
            // W3 dominance check. Same-subgraph merges keep the historical
            // (innermost-key) behavior. Cross-subgraph merges are NEW and must
            // respect two runtime invariants the old key implicitly preserved:
            // - the loop reset machinery re-evaluates every cond-tree node each
            //   iteration, so a node inside ANY loop subgraph (While/Loop/For/
            //   TailRec/LoopBody) must never be redirected away (a redirected
            //   cond-tree node left ResetPlan pointing at dead nodes → hang);
            // - a branch subgraph's entry/return nodes are anchor points; a
            //   cross-sg redirect would move them outside the sg range.
            let ex_sg = innermost[existing.0 as usize];
            let du_sg = innermost[idx];
            let legal = match (ex_sg, du_sg) {
                (None, None) => true,
                (Some(e), Some(d)) => {
                    if e == d {
                        true // same subgraph — historical behavior
                    } else {
                        let dup_in_loop =
                            graph.subgraphs[d.0 as usize].loop_kind != crate::ir::Ir::LoopKind::None;
                        let dup_is_anchor = {
                            let sg = &graph.subgraphs[d.0 as usize];
                            sg.entry_node == id || sg.return_node == id
                        };
                        !dup_in_loop && !dup_is_anchor && regions.dominates(e, d)
                    }
                }
                _ => false,
            };
            if legal {
                ctx.redirect.insert(id, existing);
            }
        } else {
            seen.insert(key, id);
        }
    }
}

// =========================================================================
// Pass: CopyProp — copy propagation
// =========================================================================

/// Passthrough compute_fn set: single input, output = input.
/// noop_compute_real(0) is a pure passthrough.
fn passthrough_set() -> FxHashSet<ComputeFnId> {
    let mut s = FxHashSet::default();
    s.insert(CF_NOOP); // noop_compute_real
    s
}

/// CopyProp pass: redirects passthrough nodes to their sole input.
pub fn pass_copy_prop(graph: &DataFlowGraph, ctx: &mut OptimizerContext) {
    let passthrough = passthrough_set();
    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(idx as u32);
        if !ctx.is_live(id) { continue; }
        // Skip already-redirected nodes (avoid repeatedly producing the same redirect)
        if ctx.redirect.contains_key(&id) { continue; }
        if node.input_count != 1 { continue; }
        if !passthrough.contains(&node.compute_fn) { continue; }
        // Side-effecting nodes cannot be redirected (writeback/field_set/global_store, etc.)
        if has_side_effect(graph, idx) { continue; }
        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        let src = ctx.resolve(inputs[0]);
        // Avoid self-loops
        if src != id {
            ctx.redirect.insert(id, src);
        }
    }
}

// =========================================================================
// Pass: DCE — dead code elimination
// =========================================================================

/// Collects all inputs and metadata NodeId references of a node (after resolve).
fn collect_refs(graph: &DataFlowGraph, ctx: &OptimizerContext, idx: usize, out: &mut Vec<NodeId>) {
    let node = graph.nodes[idx];
    let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
    for &input in inputs {
        out.push(ctx.resolve(input));
    }
    // Per-node metadata refs through the door (single enumeration shared
    // with liveness seeding and rebuild remapping). The tail pushed raw here
    // is resolved in place to match the input handling above.
    let base = out.len();
    graph.node_meta_refs(idx, out);
    for id in out[base..].iter_mut() {
        *id = ctx.resolve(*id);
    }
}

/// DCE pass: marks unreachable pure computation nodes as dead.
/// Three-step strategy:
/// 1. Compute the live set; mark pure computation nodes not in the live set as dead candidates
/// 2. Preserve propagation: traverse inputs backwards from all retained nodes (non-dead, non-redirect key),
///    removing dead candidates that are depended upon by retained nodes from the dead set
/// 3. Handle the case where a redirect target is dead: if the redirect target is dead, the redirect key
///    is also dead
pub fn pass_dce(graph: &DataFlowGraph, ctx: &mut OptimizerContext, pure_set: &FxHashSet<ComputeFnId>) {
    let live = compute_live_nodes(graph, ctx);

    // Step 1: Mark pure computation nodes not in the live set as dead candidates
    for (idx, node) in graph.nodes.iter().enumerate() {
        let id = NodeId(idx as u32);
        if live.contains(&id) { continue; }
        if !ctx.is_live(id) { continue; }
        // W1: kind gate via is_launch_kind — launch kinds and Const are never
        // deletable; the pure-computation kinds go through the pure set.
        let is_pure_calc = if crate::ir::Ir::is_launch_kind(node.kind)
            || node.kind == NodeKind::Const
        {
            false
        } else {
            pure_set.contains(&node.compute_fn)
        };
        if is_pure_calc {
            ctx.dead.insert(id);
        }
    }

    // Step 2: Preserve propagation — traverse references of all retained nodes backwards, removing reachable dead candidates
    // Retained nodes = non-dead, non-redirect-key nodes (these nodes remain in the graph)
    // Their inputs must be preserved, otherwise rebuild will panic
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

    // Step 3: Handle the case where a redirect target is dead
    // If the redirect's resolve target is dead, the redirect key should also be added to the dead set
    // (otherwise rebuild would resolve(redirect_key)=dead_target, old_to_new[dead_target]=None → panic)
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
// Pass: FuncDCE — function-level dead code elimination
// =========================================================================

/// Function-level DCE: kills entire functions unreachable from the entry
/// function, wholesale (all their nodes + the function subgraph and every
/// nested branch/loop/defer-body subgraph). This is what removes the uncalled
/// builtin/library bulk from the final artifact: node-level DCE cannot touch
/// it because every subgraph's entry/return anchors are DCE roots.
///
/// Soundness rests on the cross-function reference surface being enumerable
/// (no runtime name resolution exists — `.kzo` carries no function-name table):
/// - `call_targets` (static Call nodes)
/// - `closure_infos` / `partial_infos` (function values & partial application)
/// - `lazy_construct_infos` (lazy thunk bodies)
/// - `trait_construct_infos` (inline-trait method tables)
/// - `vtable_fallback_dispatch` (concrete-type virtual dispatch)
/// Global initializers are compiled INTO the entry function body, so entry
/// reachability covers them. Non-escaping lambdas share the enclosing
/// function's `function_id` and die together with it. `rebuild`'s removal
/// veto (reference scan) is the final safety net for anything missed here.
pub fn pass_func_dce(graph: &DataFlowGraph, ctx: &mut OptimizerContext) {
    if graph.entry_subgraph.is_none() {
        return;
    }
    // Single liveness authority: phase 2 of `compute_liveness` derives the
    // reachable-function set from the same door-seeded node closure and the
    // same retention criterion (`ctx.is_live`) that rebuild uses, so a live
    // call site and its callee body can never disagree about who survives.
    let liv = compute_liveness(graph, ctx);

    // Kill unreachable functions wholesale: every owned node + the function
    // subgraph and all nested subgraphs (any sg whose function_id is dead).
    let mut killed = false;
    for (idx, &owner) in liv.node_owner.iter().enumerate() {
        if owner == u32::MAX || liv.funcs.contains(&owner) {
            continue;
        }
        let id = NodeId(idx as u32);
        if ctx.is_live(id) {
            ctx.dead.insert(id);
            killed = true;
        }
    }
    if killed {
        for sg in &graph.subgraphs {
            if !liv.funcs.contains(&sg.function_id) {
                ctx.dead_sgs.insert(sg.id);
            }
        }
        ctx.mutated = true;
    }
}

// =========================================================================
// Pass: Strength Reduction — strength reduction
// =========================================================================

/// Determines whether a u128 value is a power of two, returning log2 (0 means 2^0=1).
fn power_of_two(v: u128) -> Option<u32> {
    if v == 0 { return None; }
    let n = v.trailing_zeros();
    if (1u128 << n) == v { Some(n) } else { None }
}

/// Extracts an unsigned u128 value from ConstValue (for power-of-two checking).
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

/// Converts a ConstValue to an i32 ConstValue holding the shift amount (shift amounts are always stored as i32).
fn make_shift_const(n: u32) -> ConstValue {
    ConstValue::I32(n as i32)
}

/// Derives the shl compute_fn for the corresponding type from a mul compute_fn.
/// Integer full range (92-235): mul(offset 2) → shl(offset 8).
/// Float mul does not support strength reduction; returns None.
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

/// Derives the shr compute_fn for the corresponding type from an unsigned div compute_fn.
/// Integer full range (92-235): div(offset 3) → shr(offset 9).
/// Signed division cannot be safely converted to right shift (negative truncation semantics differ);
/// only unsigned types are applicable.
/// type_idx >= 5 denotes the unsigned type range starting from u8.
fn div_to_shr(cf: ComputeFnId) -> Option<ComputeFnId> {
    let id = cf.0;
    if id >= 92 && id <= 235 {
        let offset = id - 92;
        let op = offset % 12;
        let type_idx = offset / 12;
        if op == 3 && type_idx >= 5 { // div and unsigned type (starting from u8)
            let type_base = id - op;
            return Some(ComputeFnId(type_base + 9)); // shr
        }
    }
    None
}

/// Derives the bitand compute_fn for the corresponding type from an unsigned mod compute_fn.
/// Integer full range (92-235): mod(offset 4) → bitand(offset 5).
/// `x % 2^n` → `x & (2^n - 1)`, safe only for unsigned types (signed modulo sign follows the dividend).
fn mod_to_bitand(cf: ComputeFnId) -> Option<ComputeFnId> {
    let id = cf.0;
    if id >= 92 && id <= 235 {
        let offset = id - 92;
        let op = offset % 12;
        let type_idx = offset / 12;
        if op == 4 && type_idx >= 5 { // mod and unsigned type (starting from u8)
            let type_base = id - op;
            return Some(ComputeFnId(type_base + 5)); // bitand
        }
    }
    None
}

/// Replaces a ConstValue's value with a new u128 value, preserving the original type.
/// Used for the mod→bitand transformation: changes the constant from `2^n` to `2^n - 1` (same type).
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

/// Strength Reduction pass: converts multiply/divide/modulo by powers of two into shifts/bitwise ops.
///
/// Transformation patterns:
/// - `x * 2^n` → `x << n` (multiplication → left shift, all integer types)
/// - `x / 2^n` (unsigned) → `x >> n` (unsigned division → logical right shift)
/// - `x % 2^n` (unsigned) → `x & (2^n - 1)` (unsigned modulo → bitmask)
///
/// Signed division/modulo are not reduced: negative truncation/modulo semantics differ from
/// shift/bitwise operations and would require additional rounding correction sequences
/// whose complexity outweighs the benefit.
///
/// Transformation method: rewrite compute_fn in place + reuse the existing constant node
/// (change its value to the shift amount / mask). No new nodes are created, avoiding hoisted
/// node cross-sub-graph range issues.
/// Triggers the next fixpoint iteration round via the ctx.mutated flag.
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

        // Try multiplication → left shift
        if let Some(shl_cf) = mul_to_shl(cf) {
            let inputs: Vec<NodeId> = graph.inputs_pool.get(node.inputs_offset, node.input_count).to_vec();
            if inputs.len() != 2 { continue; }

            // Check whether either input is a power-of-two constant
            for which in 0..2 {
                let other = 1 - which;
                let resolved = ctx.resolve(inputs[which]);
                if wb_targets.contains(&resolved) { continue; }
                let ridx = resolved.0 as usize;
                let Some(cv) = graph.const_values.get(ridx).and_then(|o| o.as_ref()) else { continue; };
                let Some(val) = cv_to_u128(cv) else { continue; };
                let Some(n) = power_of_two(val) else { continue; };
                if n == 0 { continue; } // x*1 should be handled by ConstFold

                // Safety check: the constant node must not be referenced by other nodes
                // (otherwise changing its value would affect other users).
                // downstreams == 1 means only the current multiplication node references it.
                if graph.downstreams[ridx].len() != 1 { continue; }

                // In-place rewrite: compute_fn → shl
                graph.nodes[idx].compute_fn = shl_cf;

                // Clear batch_infos: the original mul node has BatchInfo::Bin(Mul);
                // after changing to shl, the batch path would read the shift amount (i32) via as_i64,
                // producing incorrect results
                if idx < graph.batch_infos.len() {
                    graph.batch_infos[idx] = None;
                }

                // Reuse the existing constant node: change its value to the shift amount (i32)
                // This node is already in the correct sub-graph range; no hoisting needed
                graph.const_values[ridx] = Some(make_shift_const(n));

                // Reorder inputs: [variable, shift-amount constant]
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
                break; // This node is done; exit the which loop
            }
            continue;
        }

        // Try unsigned division → logical right shift
        if let Some(shr_cf) = div_to_shr(cf) {
            let inputs: Vec<NodeId> = graph.inputs_pool.get(node.inputs_offset, node.input_count).to_vec();
            if inputs.len() != 2 { continue; }

            // The divisor must be the second input (division is not commutative)
            let divisor_resolved = ctx.resolve(inputs[1]);
            if wb_targets.contains(&divisor_resolved) { continue; }
            let didx = divisor_resolved.0 as usize;
            let Some(dcv) = graph.const_values.get(didx).and_then(|o| o.as_ref()) else { continue; };
            let Some(dval) = cv_to_u128(dcv) else { continue; };
            let Some(n) = power_of_two(dval) else { continue; };
            if n == 0 { continue; } // x/1 should be handled by ConstFold

            // Safety check: the constant node must not be referenced by other nodes
            if graph.downstreams[didx].len() != 1 { continue; }

            // In-place rewrite: compute_fn → shr
            graph.nodes[idx].compute_fn = shr_cf;

            // Clear batch_infos: same reason as mul→shl
            if idx < graph.batch_infos.len() {
                graph.batch_infos[idx] = None;
            }

            // Reuse the existing constant node: change its value to the shift amount (i32)
            graph.const_values[didx] = Some(make_shift_const(n));

            // Reorder inputs: [dividend, shift-amount constant]
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

        // Try unsigned modulo → bitmask
        if let Some(bitand_cf) = mod_to_bitand(cf) {
            let inputs: Vec<NodeId> = graph.inputs_pool.get(node.inputs_offset, node.input_count).to_vec();
            if inputs.len() != 2 { continue; }

            // The modulo divisor is the second input (not commutative)
            let divisor_resolved = ctx.resolve(inputs[1]);
            if wb_targets.contains(&divisor_resolved) { continue; }
            let didx = divisor_resolved.0 as usize;
            let Some(dcv) = graph.const_values.get(didx).and_then(|o| o.as_ref()) else { continue; };
            let Some(dval) = cv_to_u128(dcv) else { continue; };
            let Some(n) = power_of_two(dval) else { continue; };
            if n == 0 { continue; } // x%1 should be handled by ConstFold

            // Safety check: the constant node must not be referenced by other nodes
            if graph.downstreams[didx].len() != 1 { continue; }

            // In-place rewrite: compute_fn mod → bitand
            graph.nodes[idx].compute_fn = bitand_cf;

            // Clear batch_infos: the original mod node has BatchInfo::Bin(Mod);
            // after changing to bitand, the batch path would use the wrong op
            if idx < graph.batch_infos.len() {
                graph.batch_infos[idx] = None;
            }

            // Reuse the existing constant node: change its value from 2^n to 2^n - 1 (preserving the original type)
            let mask = dval - 1;
            graph.const_values[didx] = cv_set_u128(dcv, mask);

            // Input order unchanged: [dividend, mask constant]
            // No need to reorder inputs; both mod and bitand are [x, const]

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
// Pass: Dead Store Elimination — dead store elimination
// =========================================================================

/// Determines whether a node is a store-class side-effecting node (WriteBack/FieldSet/ArrayStore/GlobalStore).
/// Uses compute_fn for unified determination, consistent with the node's actual computation semantics.
///
/// W1 note: deliberately NOT derived from `Ir::effect_class` (WriteLocal|WriteMutable
/// would additionally admit TailRec writeback, deref write, atomic RMW and memo
/// stores). Eliminating those is not yet validated — TailRec writeback drives the
/// loop-continue signal and atomic stores are synchronization effects. Revisit
/// under W2 together with storage versioning.
fn is_store_node(graph: &DataFlowGraph, idx: usize) -> bool {
    let cf = graph.nodes[idx].compute_fn;
    cf == CF_WRITEBACK
        || cf == CF_RECORD_FIELD_SET
        || cf == CF_ARRAY_STORE
        || cf == CF_GLOBAL_STORE
}

/// DSE pass: eliminates store nodes whose results are not consumed.
///
/// Store nodes (CF_WRITEBACK/CF_RECORD_FIELD_SET/CF_ARRAY_STORE/global_store)
/// typically return VOID and are not consumed by downstream nodes. However, if a store node's
/// downstreams are empty and it is not referenced by any live node's metadata (not triggered by
/// cond/return/defer/event), then it is a dead store and can be safely eliminated.
///
/// Safety constraints:
/// - Does not eliminate control-flow nodes (CF_RETURN/CF_BREAK/CF_CONTINUE/CF_THROW_WRAP_ERR)
/// - Does not eliminate tail_call_flags nodes
/// - Does not eliminate nodes referenced by defer_table/event_source_decls
/// - Does not eliminate nodes referenced by other live nodes' inputs (the stored value may be read)
///
/// Elimination method: added to the ctx.dead set, cleaned up by rebuild.
pub fn pass_dse(graph: &DataFlowGraph, ctx: &mut OptimizerContext) {
    let live = compute_live_nodes(graph, ctx);

    // Build two reference sets:
    // - all_refs: inputs references of all non-dead nodes (including store-class nodes).
    //   Used for rebuild safety check: rebuild retains all nodes not in the dead set;
    //   if a store node is referenced by any retained node's inputs, eliminating it would cause
    //   rebuild to panic.
    //   Note: must traverse all non-dead nodes (not just the live set), because some nodes
    //   may not be in live but also not in dead (e.g., Call/Gate nodes); rebuild still retains them.
    // - read_refs: inputs references of non-store-class live nodes only.
    //   Store-class nodes' inputs have "write" semantics (written value / modified object), not "read".
    //   read_refs precisely represents "read" nodes, used to determine whether a store side effect is observable.
    let mut all_refs: FxHashSet<NodeId> = FxHashSet::default();
    let mut read_refs: FxHashSet<NodeId> = FxHashSet::default();
    // all_refs: traverse all non-dead, non-redirect-key nodes
    for idx in 0..graph.nodes.len() {
        let id = NodeId(idx as u32);
        if ctx.dead.contains(&id) { continue; }
        if ctx.redirect.contains_key(&id) { continue; }
        let node = graph.nodes[idx];
        let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
        for &inp in inputs {
            all_refs.insert(ctx.resolve(inp));
        }
        // gate_branches / select_infos metadata references
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
    // read_refs: built only from non-store-class nodes in the live set
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

    // Collect nodes referenced by sub-graph structure (entry/return/cond/iter_next/defer/event)
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

    // Collect the set of slots read by all live global_load nodes.
    // If a global_store's slot is not in this set, no live node in the entire graph
    // reads that global variable, so the store is a dead store.
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
        if !live.contains(&id) { continue; } // Nodes not in the live set are already handled by DCE

        // Only process store-class nodes
        if !is_store_node(graph, idx) { continue; }

        // Safety check: do not eliminate control-flow/tail-call nodes
        if crate::ir::Ir::is_control_flow_compute_fn(graph.nodes[idx].compute_fn) { continue; }
        if graph.tail_call_flags.get(idx).copied().unwrap_or(false) { continue; }

        // Safety check: do not eliminate nodes referenced by sub-graph structure
        if struct_refs.contains(&id) { continue; }

        // Rebuild safety check: if the store node is referenced by any retained node's inputs,
        // eliminating it would cause rebuild to panic. all_refs covers the downstreams check
        // (having active downstreams ⟹ in all_refs), so no separate downstreams check is needed.
        if all_refs.contains(&id) { continue; }

        let cf = node.compute_fn;

        // WriteBack: target not read by any non-store live node → dead store
        if cf == CF_WRITEBACK {
            if let Some(Some(wt)) = graph.writeback_targets.get(idx).map(|o| o.as_ref()) {
                let wt_resolved = ctx.resolve(*wt);
                if read_refs.contains(&wt_resolved) || struct_refs.contains(&wt_resolved) {
                    continue;
                }
            }
        }

        // FieldSet/ArrayStore: modified heap object not read by any non-store live node → dead store
        if cf == CF_RECORD_FIELD_SET || cf == CF_ARRAY_STORE {
            let inputs = graph.inputs_pool.get(node.inputs_offset, node.input_count);
            if !inputs.is_empty() {
                let obj_node = ctx.resolve(inputs[0]);
                if read_refs.contains(&obj_node) {
                    continue;
                }
            }
        }

        // GlobalStore: no live global_load reads this slot in the entire graph → dead store
        if cf == CF_GLOBAL_STORE {
            if let Some(slot) = graph.global_store_slots.get(idx).and_then(|o| *o) {
                if loaded_slots.contains(&slot) {
                    continue;
                }
            }
        }

        // Dead store: add to the dead set
        if std::env::var("KUZO_DSE_DBG").is_ok() {
            eprintln!("[DSE] eliminate dead store node={} cf={} kind={:?}",
                idx, node.compute_fn.0, node.kind);
        }
        ctx.dead.insert(id);
    }
}

// =========================================================================
// Fixpoint iteration driver
// =========================================================================

/// Optimization level.
///
/// Drives pass selection and the fixpoint stall window for `optimize_with_analysis`:
/// - `O0`: no optimization, only Build output
/// - `O1`: skip structural transforms, only fixpoint iteration (Inline + traditional optimization)
/// - `O2`: full (structural transforms + fixpoint iteration), standard level
/// - `O3`: full + wider fixpoint stall window (30 vs 10), aggressive optimization
///
/// The `KUZO_NO_*` environment variables can still disable individual passes (for debugging),
/// taking priority over the level.
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

/// Optimization entry point (no analysis report): equivalent to `optimize_with_analysis(graph, None, OptLevel::default())`.
pub fn optimize(graph: &mut DataFlowGraph) {
    optimize_with_analysis(graph, None, OptLevel::default());
}

/// Best-effort human-readable message from a panic payload.
pub(crate) fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Never-corrupt execution of one optimization unit (see the stability policy
/// on `optimize_with_analysis`): snapshot → run → on panic restore the
/// snapshot, report, and return `None`. `KUZO_OPT_STRICT_PANIC=1` (or the CI
/// gate `KUZO_VERIFY_STRICT=1`) re-raises the panic unchanged so invariant
/// violations stay loud in development.
fn run_guarded<F, R>(graph: &mut DataFlowGraph, label: &str, f: F) -> Option<R>
where
    F: FnOnce(&mut DataFlowGraph) -> R,
{
    let strict = std::env::var("KUZO_OPT_STRICT_PANIC").is_ok()
        || std::env::var("KUZO_VERIFY_STRICT").is_ok();
    if strict {
        return Some(f(graph));
    }
    let snapshot = graph.clone();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(graph))) {
        Ok(result) => Some(result),
        Err(payload) => {
            eprintln!(
                "[OPT] internal error in {}: {}\n[OPT] optimization stopped; graph restored to its last consistent state — the compile continues with a less-optimized (still correct) result. Set KUZO_OPT_STRICT_PANIC=1 to re-raise this.",
                label,
                panic_payload_message(&payload)
            );
            *graph = snapshot;
            None
        }
    }
}

/// Optimization entry point (with analysis report): performs fixpoint-iterated optimization on the graph.
///
/// Two-phase pipeline (enabled by `level`):
/// 1. Structural transforms (one-time, level >= 2): LICM → LoopUnroll — depend on NodeIds in AnalysisReport;
///    NodeIds become invalid after rebuild, so these run only once.
/// 2. Fixpoint iteration (level >= 1): Inline → ConstFold → StrengthRed → CSE → CopyProp → DCE → DSE
///    — Inline self-collects candidates (does not depend on analysis), so it can safely run every round.
/// Without an analysis report, Phase 1 is skipped, degrading to a pure traditional optimization pipeline.
/// When `level == O0`, the entire optimizer is skipped.
///
/// Stability policy (never-corrupt): every optimization unit (phase / fixpoint
/// round, including its `rebuild`) runs against a snapshot of the last
/// consistent graph. Any panic inside the unit — an invariant tripwire such as
/// `rebuild: ref node not live`, an unexpected index, anything — restores the
/// snapshot and stops optimizing; the compile then continues with the
/// unoptimized-but-correct graph instead of crashing the process. Set
/// `KUZO_OPT_STRICT_PANIC=1` (or the CI gate `KUZO_VERIFY_STRICT=1`) to
/// re-raise the original panic for debugging.
pub fn optimize_with_analysis(
    graph: &mut DataFlowGraph,
    analysis: Option<&AnalysisReport>,
    level: OptLevel,
) {
    // O0: no optimization
    if level == OptLevel::O0 {
        return;
    }

    // W1: single derivation point (pure CFs minus aliasing reads when the
    // graph contains in-place mutators — Bug #99).
    let pure_set = crate::ir::Ir::graph_pure_set(graph);
    let no_fold = std::env::var("KUZO_NO_FOLD").is_ok();
    let no_cse = std::env::var("KUZO_NO_CSE").is_ok();
    let no_copy = std::env::var("KUZO_NO_COPY").is_ok();
    let no_dce = std::env::var("KUZO_NO_DCE").is_ok();
    let no_licm = std::env::var("KUZO_NO_LICM").is_ok();
    let no_unroll = std::env::var("KUZO_NO_UNROLL").is_ok();
    let no_strength = std::env::var("KUZO_NO_STRENGTH").is_ok();
    let no_dse = std::env::var("KUZO_NO_DSE").is_ok();
    // Function-level DCE (uncalled-function elimination) runs inside the
    // phase-2 fixpoint so functions freed by inlining die the same round and
    // the entry-level constants they kept alive are collected by the next
    // round's node-level DCE.
    let no_funcdce = std::env::var("KUZO_NO_FUNCDCE").is_ok();
    // Inline pass is enabled by default. KUZO_NO_INLINE=1 can explicitly disable it.
    // hoisted_owners tracking + rebuild grouped reordering ensures body nodes are correctly
    // included in the caller's range.
    let no_inline = std::env::var("KUZO_NO_INLINE").is_ok();

    // ── Phase 1: structural transforms (one-time, depend on analysis NodeIds, level >= 2) ──
    if level >= OptLevel::O2 && analysis.is_some() {
        let mut ctx = OptimizerContext::default();
        run_guarded(graph, "phase1 (LICM/Unroll)", |graph| {
            if !no_licm   { pass_licm(graph, &mut ctx, analysis); }
            if !no_unroll { pass_loop_unroll(graph, &mut ctx, analysis); }
            if ctx.has_changes() {
                check_gate_in_branch(graph, "BEFORE phase1 rebuild");
                graph.rebuild(&ctx.dead, &ctx.redirect, &ctx.dead_sgs);
                check_gate_in_branch(graph, "AFTER phase1 rebuild");
                crate::pass::Verifier::verify_and_report(graph, "opt-phase1");
            }
        });
        // On failure `run_guarded` already restored the pre-phase1 graph;
        // phase 2 still runs on the (unoptimized) consistent graph.
    }

    // ── Phase 2: fixpoint iteration (Inline + traditional optimization, level >= 1) ──
    let dbg_iter = std::env::var("KUZO_INLINE_DBG").is_ok();
    // Termination guard (replaces the former fixed 50/200-round cap): the loop stops
    // after `stall_window` consecutive rounds without a strict decrease of the
    // progress measure `(call sites, nodes)`, lexicographic. Every productive round
    // strictly decreases it — Inline removes a call site (even when it grows the node
    // count), CF/CSE/CopyProp/DCE/DSE only remove or simplify nodes. Redirect-only
    // churn (CopyProp rewiring with nothing dead yet) can hold the measure flat for a
    // few rounds, hence a window rather than a per-round requirement. Because the
    // measure is well-ordered and starts finite, improvements can only happen finitely
    // often, so the loop always terminates — no round cap is needed.
    // Legitimately long optimization chains keep improving and are never cut short;
    // only pass oscillations (flat/increasing forever) hit the window.
    // O3 gets a wider window (aggressive settings stretch flat stretches).
    // KUZO_OPT_STALL_WINDOW overrides; KUZO_OPT_MAX_ITER still imposes a hard round
    // cap when set (debugging aid, off by default).
    let default_window: u32 = if level >= OptLevel::O3 { 30 } else { 10 };
    let stall_window = std::env::var("KUZO_OPT_STALL_WINDOW")
        .ok().and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default_window);
    let hard_cap: Option<u32> = std::env::var("KUZO_OPT_MAX_ITER")
        .ok().and_then(|s| s.parse::<u32>().ok());
    let mut best = fixpoint_measure(graph);
    let mut stalled: u32 = 0;
    let mut round: u32 = 0;
    loop {
        round += 1;
        let mut ctx = OptimizerContext::default();

        // Never-corrupt policy: the round (passes + rebuild + verify) runs
        // against a snapshot of the last consistent state; a panic restores it
        // and ends optimization instead of killing the compile.
        let progressed = run_guarded(graph, &format!("phase2 round {round}"), |graph| {
            if !no_inline { pass_inline(graph, &mut ctx, None); }
            if !no_fold   { pass_const_fold(graph, &mut ctx); }
            if !no_strength { pass_strength_reduction(graph, &mut ctx); }
            if !no_cse    { pass_cse(graph, &mut ctx, &pure_set); }
            if !no_copy   { pass_copy_prop(graph, &mut ctx); }
            if !no_dce    { pass_dce(graph, &mut ctx, &pure_set); }
            if !no_dse    { pass_dse(graph, &mut ctx); }
            if !no_funcdce { pass_func_dce(graph, &mut ctx); }

            if !ctx.has_changes() {
                return false; // converged
            }

            if dbg_iter {
                eprintln!("[OPT-ITER] round={} nodes={} before rebuild", round, graph.nodes.len());
            }
            check_gate_in_branch(graph, "BEFORE phase2 rebuild");
            let _old_to_new = graph.rebuild(&ctx.dead, &ctx.redirect, &ctx.dead_sgs);
            check_gate_in_branch(graph, "AFTER phase2 rebuild");
            crate::pass::Verifier::verify_and_report(graph, "opt-phase2");
            true
        });
        if progressed != Some(true) {
            // Either converged, or the round failed and the graph was restored
            // (run_guarded already reported) — stop in both cases. On failure
            // this leaves the last consistent (possibly unoptimized) graph.
            break;
        }

        let m = fixpoint_measure(graph);
        if m < best {
            best = m;
            stalled = 0;
        } else {
            stalled += 1;
        }
        if dbg_iter {
            eprintln!("[OPT-ITER] round={} calls={} nodes={} stalled={}/{}", round, m.0, m.1, stalled, stall_window);
        }
        if stalled >= stall_window {
            eprintln!(
                "[OPT] fixpoint stalled: no measure improvement in {stalled} rounds (calls={}, nodes={}) — stopping; pass oscillation suspected",
                m.0, m.1
            );
            break;
        }
        if let Some(cap) = hard_cap {
            if round >= cap { break; }
        }
    }
}

/// Progress measure for the phase-2 fixpoint loop (see `optimize_with_analysis`):
/// `(call-site count, total node count)`, compared lexicographically.
fn fixpoint_measure(graph: &DataFlowGraph) -> (usize, usize) {
    let calls = graph
        .nodes
        .iter()
        .filter(|n| n.kind == crate::ir::Ir::NodeKind::Call)
        .count();
    (calls, graph.nodes.len())
}

/// Debug helper: check if any Gate node is inside its branch subgraph's node_range.
/// This would cause infinite recursion at runtime (Gate launches a subgraph that contains itself).
fn check_gate_in_branch(graph: &DataFlowGraph, label: &str) {
    if std::env::var("KUZO_DEBUG_REBUILD").is_err() {
        return;
    }
    for (idx, gb_opt) in graph.gate_branches.iter().enumerate() {
        if let Some(gb) = gb_opt {
            let gate_node = NodeId(idx as u32);
            for (cond, branch_sg, _) in &gb.branches {
                let branch_sg_id = branch_sg.0 as usize;
                if branch_sg_id < graph.subgraphs.len() {
                    let (s, e) = graph.subgraphs[branch_sg_id].node_range;
                    if gate_node.0 >= s.0 && gate_node.0 < e.0 {
                        eprintln!("[{}] BUG: Gate node {} INSIDE branch sg={} (cond={}) range [{},{}) func_id={}",
                            label, gate_node.0, branch_sg_id, cond, s.0, e.0,
                            graph.subgraphs[branch_sg_id].function_id);
                    }
                }
            }
        }
    }
}

// =========================================================================
// Loop transformation passes (merged from LoopOptimizer.rs) — LICM + loop unrolling
//
// LICM: hoists pure invariant nodes from body_sg into the function sub-graph frame.
// Loop unrolling: for small loops with a static trip count, copies body_sg nodes into the parent frame.
// See docs/superpowers/specs/2026-08-08-loop-opts-inline-design.md
// =========================================================================

/// Runs the LICM pass.
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

    // Collect (body_sg_id, invariants) snapshots (avoids holding an analysis borrow)
    let body_sgs: Vec<(SubGraphId, Vec<NodeId>)> = loop_analysis
        .invariants
        .iter()
        .map(|(k, v)| (*k, v.clone()))
        .collect();

    let mut hoisted_count = 0;

    for (body_sg_id, invariants) in &body_sgs {
        let body_sg = &graph.subgraphs[body_sg_id.0 as usize];

        // Find body_sg's loop_parent_sg (i.e., the loop_sg)
        let Some(loop_sg_id) = body_sg.loop_parent_sg else {
            continue;
        };

        // Hoist target: the function-level sub-graph (loop_kind == None).
        // Do not hoist into the loop body_sg — the body_sg frame chain is set to null
        // during reset_loop_iteration, so hoisted nodes that depend on frame-chain-accessed
        // variables would read incorrect values.
        // Function-level sub-graph frames are created only once and are not subject to
        // loop-frame reset, so they are safe.
        let func_sg_id = SubGraphId(graph.subgraphs[loop_sg_id.0 as usize].function_id);

        // Clone invariant nodes to the end of graph.nodes
        let mut node_map: FxHashMap<u32, NodeId> = FxHashMap::default();

        for &inv_node_id in invariants {
            let src_idx = inv_node_id.0 as usize;
            let src_node = graph.nodes[src_idx];
            let old_inputs = graph.inputs_pool.get(src_node.inputs_offset, src_node.input_count);

            // Compute new inputs: invariants referenced inside body_sg → cloned node, external references → keep original NodeId
            let mut new_inputs: Vec<NodeId> = Vec::with_capacity(old_inputs.len());
            for &old_in in old_inputs {
                if let Some(&mapped) = node_map.get(&old_in.0) {
                    new_inputs.push(mapped);
                } else {
                    new_inputs.push(old_in); // External references keep their original NodeId
                }
            }

            let new_id = graph.add_node_raw(src_node.kind, &new_inputs, src_node.compute_fn);
            let new_idx = new_id.0 as usize;

            // Clone metadata
            graph.clone_node_metadata(src_idx, new_idx);

            // Mark as hoisted + set owning sub-graph
            graph.hoisted_node[new_idx] = true;
            graph.hoisted_owners[new_idx] = func_sg_id;

            // Clone const_values (if the invariant is a Const node)
            if let Some(cv) = &graph.const_values[src_idx] {
                graph.const_values[new_idx] = Some(*cv);
            }

            node_map.insert(inv_node_id.0, new_id);
            hoisted_count += 1;
        }

        // Redirect the original invariant nodes in body_sg to the cloned nodes
        for (&old_id, &new_id) in &node_map {
            ctx.redirect.insert(NodeId(old_id), new_id);
        }

        // Do not extend node_range: hoisted_owners already records ownership, and rebuild
        // groups nodes by function-level sub-graph, placing hoisted nodes within the func_sg range.
        // Extending node_range would cover nodes belonging to other functions in between, causing
        // rebuild to produce a node_range containing nodes that don't belong to this sub-graph
        // → incorrect execution.
    }

    if hoisted_count > 0 {
        ctx.mutated = true;
    }
}

/// Runs the loop unrolling pass.
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

        // Find loop_sg's immediate parent (the placement target for the unrolled body)
        let Some(parent_sg_id) = graph.find_immediate_parent_sg(*loop_sg_id) else {
            continue;
        };
        // Function-level sub-graph (the hoisted_owners ownership target)
        let func_sg_id = SubGraphId(graph.subgraphs[parent_sg_id.0 as usize].function_id);

        let body_sg = &graph.subgraphs[unroll_info.body_sg.0 as usize];
        let (body_start, body_end) = body_sg.node_range;
        let (loop_start, loop_end) = loop_sg.node_range;

        // body_sg structure: param_0 = iterator, param_1 = current value (loop variable)
        let param_0_node = NodeId(body_start.0);
        let loop_var_node = unroll_info.loop_var_node;

        // Check whether body references param_0 (the iterator) — if so, skip unrolling
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

        // Clone the body for each iteration (skipping the param_0 and param_1 parameter nodes)
        let body_content_start = (body_start.0 + 2) as usize;
        let body_content_end = body_end.0 as usize;
        let mut last_body_last_node: Option<NodeId> = None;

        for i in 0..unroll_info.trip_count {
            let iter_val = unroll_info.start_value + (unroll_info.step * i as i128);

            // Create a Const node holding the iteration value (preserving the original type)
            let const_cv = make_const_value(&unroll_info.start_const, iter_val);
            let const_node = graph.add_node_raw(NodeKind::Const, &[], CF_NOOP);
            graph.const_values[const_node.0 as usize] = Some(const_cv);
            graph.hoisted_node[const_node.0 as usize] = true;
            graph.hoisted_owners[const_node.0 as usize] = func_sg_id;

            // Clone body_sg's content nodes
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

        // Handle loop_sg's Gate node
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

        // Mark all non-redirected nodes in loop_sg as dead
        for idx in (loop_start.0 as usize)..(loop_end.0 as usize) {
            let nid = NodeId(idx as u32);
            if !ctx.redirect.contains_key(&nid) {
                ctx.dead.insert(nid);
            }
        }
        // Mark all nodes in body_sg as dead
        for idx in (body_start.0 as usize)..(body_end.0 as usize) {
            let nid = NodeId(idx as u32);
            if !ctx.redirect.contains_key(&nid) {
                ctx.dead.insert(nid);
            }
        }

        // Do not extend node_range: hoisted_owners already records ownership, and rebuild
        // groups nodes by function-level sub-graph, placing hoisted nodes within the func_sg range.
    }

    if unroll_count > 0 {
        ctx.mutated = true;
    }
}

/// Creates a new ConstValue with the same type as `original` (preserving type consistency).
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
        // Non-integer type fallback (should not occur in Range unrolling)
        _ => I64(val as i64),
    }
}

// =========================================================================
// IR-level function inlining pass (merged from InlineOptimizer.rs)
//
// Runs after CSE, cloning small pure-function sub-graphs into the call site.
// Eliminates call-frame allocation overhead and opens up further optimization opportunities.
//
// Inlining criteria: the callee sub-graph body contains only Const/BinOp/UnOp/FieldAccess nodes
// (this condition simultaneously guarantees purity + no recursion, without needing an AnalysisReport mapping).
//
// Implementation: clone callee body nodes and append them to the end of graph.nodes, then rewrite
// call_node in place as a CF_SEQ sequence node (inputs = [effect_input, mapped_return],
// forwarding mapped_return once effect_input is ready). Extend the caller function sub-graph's
// node_range to include the body nodes. rebuild automatically reconstructs downstreams and
// node_range to ensure correct data-flow propagation.
// =========================================================================

/// Maximum number of callee nodes for IR-level inlining.
const MAX_INLINE_NODES: usize = 20;

/// Runs the IR-level function inlining pass.
///
/// `_analysis` is reserved for future extension (safety is currently guaranteed via body-structure checks).
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
    /// Global NodeId of the Call node
    call_node: NodeId,
    /// Callee sub-graph ID
    callee_sg: SubGraphId,
    /// Caller function sub-graph ID (used to extend node_range)
    caller_func_sg: SubGraphId,
}

fn collect_inline_candidates(graph: &DataFlowGraph) -> Vec<InlineCandidate> {
    let mut candidates = Vec::new();
    let pure_set = crate::ir::Ir::pure_compute_fn_set();

    for (idx, node) in graph.nodes.iter().enumerate() {
        // Only handle CF_CALL_LAUNCH (sync calls)
        if node.compute_fn != CF_CALL_LAUNCH {
            continue;
        }

        let nid = NodeId(idx as u32);

        // Non-tail calls only (tail calls have frame-reuse optimization and should not be inlined)
        if graph.tail_call_flags.get(idx).copied().unwrap_or(false) {
            continue;
        }

        // Must have call_targets
        let Some(Some(callee_sg_id)) = graph.call_targets.get(idx) else {
            continue;
        };
        let callee_sg = &graph.subgraphs[callee_sg_id.0 as usize];

        // Sync function
        if callee_sg.has_suspend {
            continue;
        }

        // No upvalues (upvalues require frame-chain injection, which cannot be handled after inlining)
        if callee_sg.upvalue_count > 0 {
            continue;
        }

        // Node count limit
        let callee_size = (callee_sg.node_range.1.0 - callee_sg.node_range.0.0) as usize;
        if callee_size > MAX_INLINE_NODES {
            continue;
        }

        // return_node must be within the callee node_range; otherwise inline_call's
        // node_map cannot map return_node → mapped_return falls back to the original callee node id,
        // and CF_SEQ would reference a node unreachable from the callee frame → wrong return value
        // → caller logic corruption.
        let ret = callee_sg.return_node;
        if ret.0 < callee_sg.node_range.0.0 || ret.0 >= callee_sg.node_range.1.0 {
            if std::env::var("KUZO_INLINE_DBG").is_ok() {
                eprintln!("[INLINE-SKIP] call_node={} callee_sg={} return={} not in range [{},{})",
                    nid.0, callee_sg_id.0, ret.0,
                    callee_sg.node_range.0.0, callee_sg.node_range.1.0);
            }
            continue;
        }

        // The body must contain only pure-computation kinds (W1: !is_launch_kind,
        // i.e. Const/BinOp/TriOp/UnOp/FieldAccess) whose compute_fn is in pure_set
        // (this condition simultaneously guarantees: pure function + no recursion + no control flow + no construction side effects)
        let (cs, ce) = callee_sg.node_range;
        let mut safe_body = true;
        for cidx in (cs.0 as usize)..(ce.0 as usize) {
            let cn = &graph.nodes[cidx];
            if crate::ir::Ir::is_launch_kind(cn.kind) {
                safe_body = false;
                break;
            }
            // Parameter placeholder nodes (cf=CF_NOOP, const=false) skip the pure_set check
            if cn.compute_fn != crate::ir::Ir::CF_NOOP
                && !pure_set.contains(&cn.compute_fn)
            {
                safe_body = false;
                break;
            }
            // All inputs must be within the callee_sg range (no cross-sub-graph references).
            // Functions with external references, once inlined, would have reference nodes unreachable
            // in the caller frame, causing incorrect values to be read.
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

        // Find the caller's function sub-graph
        let Some(caller_func_sg) = graph.find_function_sg_for_node(nid) else {
            continue;
        };

        // Safety check: the call node must be directly inside a function-level sub-graph
        // (not nested within a Gate branch / loop body).
        // Otherwise the inlined body would be placed at function level, causing unconditional
        // execution that bypasses the Gate condition.
        // For example, `if cond { divFunc(a, b) }` would, after inlining, execute divFunc's
        // division unconditionally.
        let Some(innermost_sg) = graph.find_innermost_sg_for_node(nid) else {
            continue;
        };
        if innermost_sg != caller_func_sg {
            continue;
        }

        // All call node inputs must be within the caller_func_sg node_range.
        // If effect_input comes from an outer function sub-graph (a non-escaping lambda / Gate branch
        // misidentified as a function-level sub-graph), the post-inline CF_SEQ would reference nodes
        // unreachable from the caller frame, causing pending_inputs to never reach zero → deadlock.
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

    // Get the Call node's inputs (the first param_count are arguments; there may be an effect dependency at the end)
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

    // Build a mapping from callee-internal NodeId → NodeId in the caller
    let mut node_map: FxHashMap<u32, NodeId> = FxHashMap::default();

    // Map parameter nodes → the call's actual arguments (the first param_count inputs)
    for i in 0..param_count {
        let param_node_id = NodeId(callee_start.0 + i as u32);
        if i < call_inputs.len() {
            node_map.insert(param_node_id.0, call_inputs[i]);
        }
    }

    // Preserve the effect dependency (if call_inputs has a non-parameter input at the end).
    // This must be copied out before the body cloning loop to end the immutable borrow of
    // graph.inputs_pool; otherwise the subsequent mutable borrow by graph.add_node_raw would conflict.
    let effect_input = if call_inputs.len() > param_count {
        Some(call_inputs[param_count])
    } else {
        None
    };

    // Clone body nodes (skipping parameter placeholder nodes) — two-pass cloning ensures forward references map correctly.
    // With single-pass cloning, if node A references a later node B, B is not yet in node_map when A is cloned,
    // so new_inputs retains B's old id; rebuild would then map B's old id to the original callee node
    // (within the callee_sg range) rather than the cloned node → unreachable from the caller frame
    // → value-table misalignment / deadlock.
    let body_start = callee_start.0 as usize + param_count;
    let body_end = callee_end.0 as usize;

    // Snapshot body node info (ends the immutable borrow of graph so that the subsequent add_node_raw mutable borrow is allowed)
    let body_snapshots: Vec<(usize, NodeKind, ComputeFnId, Vec<NodeId>)> =
        (body_start..body_end)
            .map(|src_idx| {
                let src_node = graph.nodes[src_idx];
                let old_inputs =
                    graph.inputs_pool.get(src_node.inputs_offset, src_node.input_count).to_vec();
                (src_idx, src_node.kind, src_node.compute_fn, old_inputs)
            })
            .collect();

    // First pass: allocate a new_id for every body node (with empty inputs), building a complete node_map
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

    // Second pass: remap inputs (node_map now contains all body nodes, so forward references resolve correctly)
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

    // Replace call_node in place: turn it into a CF_SEQ sequence node.
    // CF_SEQ (idx 48) waits for all inputs to be ready, then returns the last input's value (sequence semantics).
    // Same pattern as Builder.rs's chain_effects: inputs = [prev_effect, current_value].
    // redirect is not used (redirect would remove call_node during rebuild, breaking the effect chain).
    //
    // inputs = [effect_input?, mapped_return]
    // - effect_input acts as a data-dependency edge, forcing preceding side effects to complete before call_node (ordering constraint)
    // - mapped_return is the return-value node of the inlined body (the last input, forwarded to call_node's downstreams)
    // After rebuild, downstreams are reconstructed automatically: both effect_input's and mapped_return's downstreams
    // will include call_node, ensuring call_node executes only after both are ready.
    let mapped_return = node_map.get(&return_node.0).copied().unwrap_or(return_node);

    // CF_SEQ inputs = [effect_input?, mapped_return]
    // effect_input must be within caller_func_sg's node_range: the caller frame's value_table
    // only covers the node_range; inputs outside the range are never ready → CF_SEQ deadlock → frame stalls.
    // collect_inline_candidates guarantees call_node is in a function-level sg, but effect_input may
    // come from a nested sub-graph (e.g., a Gate branch's effect chain); in that case drop effect_input
    // (effect ordering is implicitly guaranteed by downstreams: mapped_return's dependency chain
    // naturally serializes side effects).
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

    // Modify call_node in place
    let cn = &mut graph.nodes[call_node.0 as usize];
    cn.compute_fn = CF_SEQ;
    cn.inputs_offset = new_offset;
    cn.input_count = new_inputs.len() as u8;
    cn.kind = NodeKind::BinOp; // CF_SEQ is a BinOp kind

    // Clear call_node's call_targets metadata (it is no longer a Call node)
    graph.call_targets[call_node.0 as usize] = None;
    graph.tail_call_flags[call_node.0 as usize] = false;

    // Do not extend node_range: hoisted_owners already records ownership, and rebuild
    // groups nodes by function-level sub-graph, placing body nodes within the caller_func_sg range.

    ctx.mutated = true;
}

// ==================== Stability-policy tests ====================

#[cfg(test)]
mod stability_tests {
    use super::*;
    use crate::ir::Ir::*;

    fn tiny_graph() -> DataFlowGraph {
        let mut g = DataFlowGraph::new();
        // entry function sg: [const 1, const 2, unused pure add] — the unused
        // add gives the first fixpoint round something to change so rebuild
        // (and thus the injected failure) is actually reached.
        let off = g.inputs_pool.push(&[]);
        g.add_node(Node { kind: NodeKind::Const, input_count: 0, inputs_offset: off, compute_fn: CF_NOOP });
        g.const_values[0] = Some(ConstValue::I32(1));
        let off = g.inputs_pool.push(&[]);
        g.add_node(Node { kind: NodeKind::Const, input_count: 0, inputs_offset: off, compute_fn: CF_NOOP });
        g.const_values[1] = Some(ConstValue::I32(2));
        let off = g.inputs_pool.push(&[NodeId(0), NodeId(1)]);
        g.add_node(Node { kind: NodeKind::BinOp, input_count: 2, inputs_offset: off, compute_fn: ComputeFnId(1) });
        g.add_subgraph(SubGraph {
            id: SubGraphId(0),
            node_range: (NodeId(0), NodeId(3)),
            param_count: 0,
            entry_node: NodeId(0),
            return_node: NodeId(0),
            has_suspend: false,
            event_source_decls: Vec::new(),
            defer_table: Vec::new(),
            loop_kind: LoopKind::None,
            loop_parent_sg: None,
            cond_node: None,
            function_id: 0,
            iter_next_node: None,
            upvalue_count: 0,
            upvalue_outer_nodes: Vec::new(),
            nested_ranges: Vec::new(),
            reset_plan: None,
        });
        g.set_entry_subgraph(SubGraphId(0));
        g.compute_downstreams();
        g
    }

    /// The never-corrupt policy: an invariant violation inside an optimization
    /// round must roll the graph back to the last consistent state instead of
    /// panicking out of the optimizer.
    #[test]
    fn optimizer_rollback_on_rebuild_failure() {
        std::env::set_var("KUZO_TEST_INJECT_REBUILD_FAIL", "1");
        let mut g = tiny_graph();
        let nodes_before = g.nodes.len();
        optimize_with_analysis(&mut g, None, OptLevel::O1); // must not panic
        std::env::remove_var("KUZO_TEST_INJECT_REBUILD_FAIL");
        assert_eq!(g.nodes.len(), nodes_before, "graph must be restored to the pre-round state");
        assert_eq!(g.entry_subgraph, Some(SubGraphId(0)));
        assert_eq!(g.node_count(), 3);
    }
}
