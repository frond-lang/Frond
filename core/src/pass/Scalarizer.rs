#![allow(non_snake_case)]
//! Scalarizer — L2 标量化器: pure-leaf subgraph → straight-line scalar
//! program (pass pipeline: build → devirtualize/DCE → def-use lowering).
//!
//! Pass pipeline over a pure-leaf subgraph's linear plan (whitelist-gated):
//! **build** (plan → slot-based `Sop` program, mirroring the compute_fns'
//! accessor/kernel/ctor triples) → **optimize** (`optimize_sops`: cell
//! devirtualization + backward liveness DCE) → **lower** (`lower_to_def_use`:
//! operands resolve to `Op`/`Const`/`Param` references; forwarding and const
//! materialization vanish; the return resolves through its chain).
//! The resulting `ScalarProg` is stored on the [`DataFlowGraph`] and executed
//! by the engine's synchronous fast path and the offload workers alike
//! (bit-identical to the generic executor: same kernels, same order, only
//! pure forwarding removed).
//!
//! [`DataFlowGraph`]: crate::ir::Ir::DataFlowGraph

use crate::ir::Ir::{DataFlowGraph, SubGraphId};
use crate::value::Value;

//
// A pure-leaf offload whose plan consists solely of const/noop, seq, f64
// arithmetic, cell alloc and cell deref read/write executes as a compiled
// straight-line program over a slot array: the per-node work of the generic
// executor (compute_fn pointer dispatch, node()/inputs() SoA resolution,
// readiness bitmaps, pending countdowns, notify_downstream) collapses into
// pre-resolved slot-index arithmetic. The op semantics mirror the
// corresponding compute_fns exactly (Value::f64 construction, as_f64
// coercion, Cell::set/get, seq last-input pass-through, const precedence);
// any node outside the supported set keeps the whole subgraph on the generic
// path — the fast path is a specialization, never a semantic fork.

#[derive(Clone, Copy, PartialEq)]
pub enum STy {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    Isize,
    Usize,
    F16,
    F32,
    F64,
    F128,
    Bool,
}

#[derive(Clone, Copy, PartialEq)]
pub enum SOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Neg,
    BitNot,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Not,
}

#[derive(Clone)]
pub enum Sop {
    /// cf=0 with a const value: precompiled immutable Value (per-launch clone
    /// == the generic path's const_cache clone; consts never mutate).
    Const { dst: u32, val: Value },
    /// cf=0 without a const: slot = VOID (noop semantics).
    Void { dst: u32 },
    /// cf=349: real Cell allocated around slots[src] (escape semantics free).
    CellAlloc { dst: u32, src: u32 },
    /// cf=280 with a chain-local Cell input: cell.set(slots[val]);
    /// slots[dst] = slots[val] (mirrors compute_deref_write's return).
    DerefWriteCell { dst: u32, cell: u32, val: u32 },
    /// cf=279 with a chain-local Cell input: slots[dst] = cell.get().
    DerefReadCell { dst: u32, cell: u32 },
    /// Scalar arithmetic/comparison/bitwise/shift op over full-width inputs
    /// (extra effect-chain inputs tolerated). Unary ops ignore `b`.
    Scalar { dst: u32, a: u32, b: u32, ty: STy, op: SOp, unary: bool },
    /// cf=47: slots[dst] = last input's value (VOID when inputless).
    Seq { dst: u32, src: Option<u32> },
}

/// An operand reference in the def-use program.
pub enum OpRef {
    /// Result of the op at this index (written this launch, before any read).
    Op(u32),
    /// Immutable constant owned by the program (index into `consts`).
    Const(u32),
    /// Launch argument slot.
    Param(u32),
}

/// Def-use straight-line op. Forwarding (seq/copy) and const materialization
/// resolved away at compile time — an op exists only when it computes or has
/// an effect, and operands point at their ultimate definitions.
pub enum DSop {
    Scalar { a: OpRef, b: OpRef, ty: STy, op: SOp, unary: bool },
    /// Real (escaping) Cell allocation.
    CellAlloc { src: OpRef },
    /// Real Cell write: cell.set(val); the op's result IS the value.
    DerefWrite { cell: OpRef, val: OpRef },
    /// Real Cell read: result = cell.get().
    DerefRead { cell: OpRef },
}

pub(crate) struct ScalarProg {
    ops: Vec<DSop>,
    /// Immutable constant pool (owned; cloned only at use sites that need an
    /// owned Value — arith reads borrow).
    consts: Vec<Value>,
    param_count: usize,
    /// What the subgraph returns, as a reference.
    return_ref: OpRef,
}

/// Supported structural compute_fn ids.
const CF_NOOP_OR_CONST: u32 = 0;
const CF_SEQ: u32 = 47;
const CF_DEREF_READ: u32 = 279;
const CF_DEREF_WRITE: u32 = 280;
const CF_CELL_ALLOC: u32 = 349;

/// Classifies a scalar compute_fn id into (type, op). Covers the regular
/// int families (ids 90..=233, layout add,sub,mul,div,mod,and,or,xor,shl,
/// shr,neg,bitnot), the float families (234..=257: add..mod,neg), the bool
/// ops, and the legacy scattered ids (1..=27, 48..=89, 300..=305, 341..=346)
/// — everything whose compute fn is a pure accessor/kernel/ctor triple.
/// None = not a scalar op (records, strings, calls, casts, ...).
pub fn classify_scalar_cf(cf: u32) -> Option<(STy, SOp)> {
    use SOp::*;
    use STy::*;
    const INT_OPS: [SOp; 12] = [Add, Sub, Mul, Div, Mod, BitAnd, BitOr, BitXor, Shl, Shr, Neg, BitNot];
    const FLOAT_OPS: [SOp; 6] = [Add, Sub, Mul, Div, Mod, Neg];
    const CMP_OPS: [SOp; 6] = [Eq, Ne, Lt, Gt, Le, Ge];
    let int = match cf {
        90..=101 => Some((90, I8)),
        102..=113 => Some((102, I16)),
        114..=125 => Some((114, I32)),
        126..=137 => Some((126, I64)),
        138..=149 => Some((138, I128)),
        150..=161 => Some((150, U8)),
        162..=173 => Some((162, U16)),
        174..=185 => Some((174, U32)),
        186..=197 => Some((186, U64)),
        198..=209 => Some((198, U128)),
        210..=221 => Some((210, Isize)),
        222..=233 => Some((222, Usize)),
        _ => None,
    };
    if let Some((base, ty)) = int {
        return Some((ty, INT_OPS[(cf - base) as usize]));
    }
    let flt = match cf {
        234..=239 => Some((234, F16)),
        240..=245 => Some((240, F32)),
        246..=251 => Some((246, F64)),
        252..=257 => Some((252, F128)),
        _ => None,
    };
    if let Some((base, ty)) = flt {
        return Some((ty, FLOAT_OPS[(cf - base) as usize]));
    }
    if let Some(r) = (match cf {
        1 => Some((I32, Add)),
        2 => Some((F64, Add)),
        3 => Some((I32, Mul)),
        4 => Some((I32, Le)),
        5 => Some((I32, Sub)),
        6 => Some((I32, Div)),
        7 => Some((I32, Mod)),
        8 => Some((I32, Eq)),
        9 => Some((I32, Ne)),
        10 => Some((I32, Lt)),
        11 => Some((I32, Gt)),
        12 => Some((I32, Ge)),
        13 => Some((F64, Sub)),
        14 => Some((F64, Mul)),
        15 => Some((F64, Div)),
        16 => Some((F64, Eq)),
        17 => Some((F64, Ne)),
        18 => Some((F64, Lt)),
        19 => Some((F64, Gt)),
        20 => Some((F64, Le)),
        21 => Some((F64, Ge)),
        22 => Some((Bool, And)),
        23 => Some((Bool, Or)),
        24 => Some((Bool, Not)),
        25 => Some((I32, Neg)),
        26 => Some((F64, Neg)),
        27 => Some((Bool, Eq)),
        298 => Some((Bool, Ne)),
        59 => Some((I64, Neg)),
        60 => Some((I32, BitNot)),
        61 => Some((I64, BitNot)),
        73 => Some((I128, Neg)),
        74 => Some((I128, BitNot)),
        75 => Some((I32, BitAnd)),
        76 => Some((I32, BitOr)),
        77 => Some((I32, BitXor)),
        78 => Some((I64, BitAnd)),
        79 => Some((I64, BitOr)),
        80 => Some((I64, BitXor)),
        81 => Some((I128, BitAnd)),
        82 => Some((I128, BitOr)),
        83 => Some((I128, BitXor)),
        84 => Some((I32, Shl)),
        85 => Some((I32, Shr)),
        86 => Some((I64, Shl)),
        87 => Some((I64, Shr)),
        88 => Some((I128, Shl)),
        89 => Some((I128, Shr)),
        300..=305 => Some((F128, CMP_OPS[(cf - 300) as usize])),
        341..=346 => Some((U128, CMP_OPS[(cf - 341) as usize])),
        _ => None,
    }) {
        return Some(r);
    }
    // Legacy i64/i128 blocks: add,sub,mul,div,mod,eq,ne,lt,gt,le,ge.
    const BLK: [SOp; 11] = [Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Gt, Le, Ge];
    match cf {
        48..=58 => Some((STy::I64, BLK[(cf - 48) as usize])),
        62..=72 => Some((STy::I128, BLK[(cf - 62) as usize])),
        _ => None,
    }
}

/// Executes one scalar op with the exact accessor/kernel/ctor triple the
/// generic compute fn uses (same arith_* kernels — no semantic drift).
pub fn exec_scalar_op(ty: STy, op: SOp, unary: bool, a: &Value, b: &Value) -> Value {
    /// Integer family: div/mod/shifts are Option kernels (throw on zero
    /// divisor / out-of-range shift); everything else direct.
    macro_rules! int_ty {
        ($fn_name:ident, $variant:ident, $tyname:ident, $acc:ident, $ctor:ident) => { pastey::paste! {
            fn $fn_name(op: SOp, unary: bool, a: &Value, b: &Value) -> Value {
                let a = a.$acc();
                if unary {
                    return match op {
                        SOp::Neg => Value::$ctor(crate::value::[<arith_neg_ $tyname>](a)),
                        SOp::BitNot => Value::$ctor(crate::value::[<arith_bitnot_ $tyname>](a)),
                        _ => unreachable!("bad int unary op"),
                    };
                }
                let b_raw = b;
                let b = b_raw.$acc();
                match op {
                    SOp::Add => Value::$ctor(crate::value::[<arith_add_ $tyname>](a, b)),
                    SOp::Sub => Value::$ctor(crate::value::[<arith_sub_ $tyname>](a, b)),
                    SOp::Mul => Value::$ctor(crate::value::[<arith_mul_ $tyname>](a, b)),
                    SOp::Div => match crate::value::[<arith_div_ $tyname>](a, b) {
                        Some(v) => Value::$ctor(v),
                        None => crate::ir::Compute::make_arith_throw(
                            "DivideByZero", "integer divide by zero"),
                    },
                    SOp::Mod => match crate::value::[<arith_mod_ $tyname>](a, b) {
                        Some(v) => Value::$ctor(v),
                        None => crate::ir::Compute::make_arith_throw(
                            "DivideByZero", "integer modulo by zero"),
                    },
                    SOp::BitAnd => Value::$ctor(crate::value::[<arith_bitand_ $tyname>](a, b)),
                    SOp::BitOr => Value::$ctor(crate::value::[<arith_bitor_ $tyname>](a, b)),
                    SOp::BitXor => Value::$ctor(crate::value::[<arith_bitxor_ $tyname>](a, b)),
                    SOp::Shl => match crate::value::[<arith_shl_ $tyname>](a, b_raw.as_i32()) {
                        Some(v) => Value::$ctor(v),
                        None => crate::ir::Compute::make_arith_throw(
                            "ShiftOutOfRange", "shift amount out of range"),
                    },
                    SOp::Shr => match crate::value::[<arith_shr_ $tyname>](a, b_raw.as_i32()) {
                        Some(v) => Value::$ctor(v),
                        None => crate::ir::Compute::make_arith_throw(
                            "ShiftOutOfRange", "shift amount out of range"),
                    },
                    SOp::Eq => Value::bool_val(a == b),
                    SOp::Ne => Value::bool_val(a != b),
                    SOp::Lt => Value::bool_val(a < b),
                    SOp::Gt => Value::bool_val(a > b),
                    SOp::Le => Value::bool_val(a <= b),
                    SOp::Ge => Value::bool_val(a >= b),
                    _ => unreachable!("bad int binary op"),
                }
            }
        } };
    }
    macro_rules! float_ty {
        ($fn_name:ident, $variant:ident, $tyname:ident, $acc:ident, $ctor:ident) => { pastey::paste! {
            fn $fn_name(op: SOp, unary: bool, a: &Value, b: &Value) -> Value {
                let a = a.$acc();
                if unary {
                    return match op {
                        SOp::Neg => Value::$ctor(crate::value::[<arith_neg_ $tyname>](a)),
                        _ => unreachable!("bad float unary op"),
                    };
                }
                let b = b.$acc();
                match op {
                    SOp::Add => Value::$ctor(crate::value::[<arith_add_ $tyname>](a, b)),
                    SOp::Sub => Value::$ctor(crate::value::[<arith_sub_ $tyname>](a, b)),
                    SOp::Mul => Value::$ctor(crate::value::[<arith_mul_ $tyname>](a, b)),
                    SOp::Div => Value::$ctor(crate::value::[<arith_div_ $tyname>](a, b)),
                    SOp::Mod => Value::$ctor(crate::value::[<arith_mod_ $tyname>](a, b)),
                    SOp::Eq => Value::bool_val(a == b),
                    SOp::Ne => Value::bool_val(a != b),
                    SOp::Lt => Value::bool_val(a < b),
                    SOp::Gt => Value::bool_val(a > b),
                    SOp::Le => Value::bool_val(a <= b),
                    SOp::Ge => Value::bool_val(a >= b),
                    _ => unreachable!("bad float binary op"),
                }
            }
        } };
    }
    if ty == STy::Bool {
        let a = a.as_bool();
        if unary {
            return match op {
                SOp::Not => Value::bool_val(crate::value::arith_not_bool(a)),
                _ => unreachable!("bad bool unary op"),
            };
        }
        let b = b.as_bool();
        return match op {
            SOp::And => Value::bool_val(crate::value::arith_and_bool(a, b)),
            SOp::Or => Value::bool_val(crate::value::arith_or_bool(a, b)),
            SOp::Eq => Value::bool_val(a == b),
            SOp::Ne => Value::bool_val(a != b),
            _ => unreachable!("bad bool binary op"),
        };
    }
    int_ty!(exec_i8_op, I8, i8, as_i8, i8);
    int_ty!(exec_i16_op, I16, i16, as_i16, i16);
    int_ty!(exec_i32_op, I32, i32, as_i32, i32);
    int_ty!(exec_i64_op, I64, i64, as_i64, i64);
    int_ty!(exec_i128_op, I128, i128, as_i128, i128);
    int_ty!(exec_u8_op, U8, u8, as_u8, u8);
    int_ty!(exec_u16_op, U16, u16, as_u16, u16);
    int_ty!(exec_u32_op, U32, u32, as_u32, u32);
    int_ty!(exec_u64_op, U64, u64, as_u64, u64);
    int_ty!(exec_u128_op, U128, u128, as_u128, u128);
    int_ty!(exec_isize_op, Isize, isize, as_isize, isize_val);
    int_ty!(exec_usize_op, Usize, usize, as_usize, usize_val);
    float_ty!(exec_f16_op, F16, f16, as_f16, f16);
    float_ty!(exec_f32_op, F32, f32, as_f32, f32);
    float_ty!(exec_f64_op, F64, f64, as_f64, f64);
    float_ty!(exec_f128_op, F128, f128, as_f128, f128);

    // F128 comparisons route through the shared bit-pattern kernels (NaN/±0
    // semantics differ from F128's Ord); arithmetic stays on arith_* kernels.
    if ty == STy::F128 && !unary {
        use crate::ir::Compute as C;
        let ab = u128::from_le_bytes(a.as_f128().0);
        let bb = u128::from_le_bytes(b.as_f128().0);
        return match op {
            SOp::Eq => Value::bool_val(C::f128_eq_bits(ab, bb)),
            SOp::Ne => Value::bool_val(C::f128_ne_bits(ab, bb)),
            SOp::Lt => Value::bool_val(C::f128_lt_bits(ab, bb)),
            SOp::Gt => Value::bool_val(C::f128_gt_bits(ab, bb)),
            SOp::Le => Value::bool_val(C::f128_le_bits(ab, bb)),
            SOp::Ge => Value::bool_val(C::f128_ge_bits(ab, bb)),
            _ => exec_f128_op(op, unary, a, b),
        };
    }
    match ty {
        STy::I8 => exec_i8_op(op, unary, a, b),
        STy::I16 => exec_i16_op(op, unary, a, b),
        STy::I32 => exec_i32_op(op, unary, a, b),
        STy::I64 => exec_i64_op(op, unary, a, b),
        STy::I128 => exec_i128_op(op, unary, a, b),
        STy::U8 => exec_u8_op(op, unary, a, b),
        STy::U16 => exec_u16_op(op, unary, a, b),
        STy::U32 => exec_u32_op(op, unary, a, b),
        STy::U64 => exec_u64_op(op, unary, a, b),
        STy::U128 => exec_u128_op(op, unary, a, b),
        STy::Isize => exec_isize_op(op, unary, a, b),
        STy::Usize => exec_usize_op(op, unary, a, b),
        STy::F16 => exec_f16_op(op, unary, a, b),
        STy::F32 => exec_f32_op(op, unary, a, b),
        STy::F64 => exec_f64_op(op, unary, a, b),
        STy::F128 => exec_f128_op(op, unary, a, b),
        STy::Bool => unreachable!("handled above"),
    }
}

/// Compiles the sg's linear plan into a ScalarProg. Returns None for any
/// unsupported node shape — the subgraph then stays on the generic executor.
pub(crate) fn build_scalar_prog(graph: &DataFlowGraph, sg_id: SubGraphId) -> Option<std::sync::Arc<ScalarProg>> {
    let sg = &graph.subgraphs[sg_id.0 as usize];
    let plan = graph.linear_plan(sg_id.0 as usize)?;
    if plan.is_empty() {
        return None;
    }
    let (ns, ne) = sg.node_range;
    let slot_count = (ne.0 - ns.0) as usize;
    let to_local = |gid: u32| -> Option<u32> {
        let l = gid.wrapping_sub(ns.0);
        if l < slot_count as u32 {
            Some(l)
        } else {
            None
        }
    };
    let param_count = sg.param_count as usize;
    let mut ops: Vec<Sop> = Vec::with_capacity(plan.len());
    let mut cell_slots = vec![false; slot_count];
    for &gid in plan.iter() {
        let n = graph.node(gid.0 as usize);
        let dst = match to_local(gid.0) {
            Some(d) => d,
            None => {
                return None;
            }
        };
        // Param slots are injected from the launch args (the generic path
        // seeds them ready and the plan loop skips them).
        if (dst as usize) < param_count {
            continue;
        }
        let mut inputs = Vec::with_capacity(n.input_count as usize);
        for &inp in graph.inputs(n.inputs_offset, n.input_count) {
            match to_local(inp.0) {
                Some(l) => inputs.push(l),
                None => {
                    return None;
                }
            }
        }
        match n.compute_fn.0 {
            CF_NOOP_OR_CONST => {
                if !graph.const_cache.is_empty() {
                    let val = graph.const_cache.get(gid.0 as usize)?.clone();
                    ops.push(Sop::Const { dst, val });
                } else if let Some(cv) = graph.const_value(gid.0 as usize) {
                    let val = crate::engine::alloc_const_value(cv, graph.string_pool_slice());
                    ops.push(Sop::Const { dst, val });
                } else {
                    ops.push(Sop::Void { dst });
                }
            }
            CF_SEQ => {
                let src = inputs.last().copied();
                ops.push(Sop::Seq { dst, src });
            }
            cf if classify_scalar_cf(cf).is_some() => {
                let (ty, op) = classify_scalar_cf(cf).unwrap();
                let unary = matches!(op, SOp::Neg | SOp::BitNot | SOp::Not);
                // Extra inputs beyond the operands are effect-chain ordering
                // deps — satisfied by the plan's linear order.
                let arity = if unary { 1 } else { 2 };
                if inputs.len() < arity {
                    return None;
                }
                ops.push(Sop::Scalar {
                    dst,
                    a: inputs[0],
                    b: inputs.get(1).copied().unwrap_or(inputs[0]),
                    ty,
                    op,
                    unary,
                });
            }
            CF_CELL_ALLOC => {
                if inputs.is_empty() {
                    return None;
                }
                cell_slots[dst as usize] = true;
                ops.push(Sop::CellAlloc { dst, src: inputs[0] });
            }
            CF_DEREF_WRITE => {
                if inputs.len() < 2 || !cell_slots[inputs[0] as usize] {
                    return None;
                }
                ops.push(Sop::DerefWriteCell {
                    dst,
                    cell: inputs[0],
                    val: inputs[1],
                });
            }
            CF_DEREF_READ => {
                if inputs.is_empty() || !cell_slots[inputs[0] as usize] {
                    return None;
                }
                ops.push(Sop::DerefReadCell {
                    dst,
                    cell: inputs[0],
                });
            }
            _ => return None,
        }
    }
    let return_slot = to_local(sg.return_node.0)?;
    let ops = optimize_sops(ops, param_count, slot_count, return_slot);
    let (dops, consts, return_ref) = lower_to_def_use(ops, param_count, slot_count, return_slot);
    Some(std::sync::Arc::new(ScalarProg {
        ops: dops,
        consts,
        param_count,
        return_ref,
    }))
}

/// Lowers the optimized slot-based op list into the def-use form. Operand
/// slots resolve (transitively, through forwards) to their ultimate
/// definitions; consts become program-owned immediates (no slot write, no
/// per-launch clone); the return resolves through its forwarding chain.
/// Emission order = plan order, so every Op(i) reference points at an op
/// that has already executed this launch.
pub fn lower_to_def_use(
    ops: Vec<Sop>,
    param_count: usize,
    slot_count: usize,
    return_slot: u32,
) -> (Vec<DSop>, Vec<Value>, OpRef) {
    #[derive(Clone)]
    enum Binding {
        Op(u32),
        Const(u32),
        Param(u32),
    }
    let mut consts: Vec<Value> = Vec::new();
    let mut dops: Vec<DSop> = Vec::with_capacity(ops.len());
    let mut slot_def: Vec<Option<Binding>> = vec![None; slot_count];
    for i in 0..param_count.min(slot_count) {
        slot_def[i] = Some(Binding::Param(i as u32));
    }
    let as_ref = |b: &Option<Binding>| -> OpRef {
        match b {
            Some(Binding::Op(i)) => OpRef::Op(*i),
            Some(Binding::Const(c)) => OpRef::Const(*c),
            Some(Binding::Param(p)) => OpRef::Param(*p),
            // DCE guarantees defined bindings for every remaining read;
            // the MAX sentinel reads as NULL at runtime (unseeded-slot
            // semantics of the generic path).
            None => OpRef::Param(u32::MAX),
        }
    };
    for op in ops {
        match op {
            // Consts/forwards emit nothing — they rebind the slot.
            Sop::Const { dst, val } => {
                let idx = consts.len() as u32;
                consts.push(val);
                slot_def[dst as usize] = Some(Binding::Const(idx));
            }
            Sop::Void { dst } => {
                let idx = consts.len() as u32;
                consts.push(Value::VOID);
                slot_def[dst as usize] = Some(Binding::Const(idx));
            }
            Sop::Seq { dst, src: Some(src) } => {
                slot_def[dst as usize] = slot_def[src as usize].clone();
            }
            Sop::Seq { dst, src: None } => {
                let idx = consts.len() as u32;
                consts.push(Value::VOID);
                slot_def[dst as usize] = Some(Binding::Const(idx));
            }
            Sop::Scalar { dst, a, b, ty, op, unary } => {
                let ra = as_ref(&slot_def[a as usize]);
                let rb = as_ref(&slot_def[b as usize]);
                let idx = dops.len() as u32;
                dops.push(DSop::Scalar { a: ra, b: rb, ty, op, unary });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::CellAlloc { dst, src } => {
                let rsrc = as_ref(&slot_def[src as usize]);
                let idx = dops.len() as u32;
                dops.push(DSop::CellAlloc { src: rsrc });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::DerefWriteCell { dst, cell, val } => {
                let rcell = as_ref(&slot_def[cell as usize]);
                let rval = as_ref(&slot_def[val as usize]);
                let idx = dops.len() as u32;
                dops.push(DSop::DerefWrite { cell: rcell, val: rval });
                // The write op's result IS the written value.
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::DerefReadCell { dst, cell } => {
                let rcell = as_ref(&slot_def[cell as usize]);
                let idx = dops.len() as u32;
                dops.push(DSop::DerefRead { cell: rcell });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
        }
    }
    let return_ref = as_ref(&slot_def[return_slot.min(slot_count.saturating_sub(1) as u32) as usize]);
    (dops, consts, return_ref)
}

/// Compile-time optimization of the op list. Purely structural — no
/// arithmetic is rewritten, so results stay bit-identical to the generic
/// executor:
/// 1. Cell devirtualization: a CellAlloc whose slot is never read by a
///    non-deref op and is not the return slot becomes a plain VALUE slot —
///    writes degrade to copies (updating cell-slot AND dst-slot, preserving
///    later reads), reads to forwards; no Cell heap object per launch.
///    Escaping cells keep the real-object semantics.
/// 2. Liveness DCE: ops whose destination is never read (and is not the
///    return slot) are dropped — typically thousands of seq/noop chain ops.
pub fn optimize_sops(ops: Vec<Sop>, param_count: usize, slot_count: usize, return_slot: u32) -> Vec<Sop> {
    let is_cell: Vec<bool> = ops
        .iter()
        .filter_map(|op| match op {
            Sop::CellAlloc { dst, .. } => Some(*dst as usize),
            _ => None,
        })
        .fold(vec![false; slot_count], |mut v, i| {
            v[i] = true;
            v
        });
    // A cell escapes iff a non-deref op reads its slot or it is returned.
    let mut escapes = vec![false; slot_count];
    if (return_slot as usize) < slot_count {
        escapes[return_slot as usize] = true;
    }
    for op in &ops {
        let (uses, deref_use): (Vec<u32>, bool) = match op {
            Sop::Const { .. } | Sop::Void { .. } => (vec![], false),
            Sop::CellAlloc { src, .. } => (vec![*src], false),
            Sop::DerefWriteCell { cell, val, .. } => (vec![*cell, *val], true),
            Sop::DerefReadCell { cell, .. } => (vec![*cell], true),
            Sop::Scalar { a, b, unary, .. } => {
                (if *unary { vec![*a] } else { vec![*a, *b] }, false)
            }
            Sop::Seq { src, .. } => (src.iter().copied().collect(), false),
        };
        if !deref_use {
            for u in uses {
                escapes[u as usize] = true;
            }
        }
    }
    let devirt = |slot: u32| -> bool {
        (slot as usize) < slot_count && is_cell[slot as usize] && !escapes[slot as usize]
    };

    // Rewrite pass (devirtualized cells become value slots). A write updates
    // BOTH the cell slot and its own dst (later reads see the new value).
    let prog: Vec<Sop> = ops
        .into_iter()
        .flat_map(|op| match op {
            Sop::CellAlloc { dst, src } if devirt(dst) => vec![Sop::Seq { dst, src: Some(src) }],
            Sop::DerefWriteCell { dst, cell, val } if devirt(cell) => vec![
                Sop::Seq { dst: cell, src: Some(val) },
                Sop::Seq { dst, src: Some(val) },
            ],
            Sop::DerefReadCell { dst, cell } if devirt(cell) => vec![Sop::Seq { dst, src: Some(cell) }],
            other => vec![other],
        })
        .collect();

    // Backward liveness DCE.
    let mut live = vec![false; slot_count];
    for i in 0..param_count.min(slot_count) {
        live[i] = true;
    }
    if (return_slot as usize) < slot_count {
        live[return_slot as usize] = true;
    }
    let mut kept: Vec<Sop> = Vec::with_capacity(prog.len());
    for op in prog.iter().rev() {
        let dst = match op {
            Sop::Const { dst, .. }
            | Sop::Void { dst }
            | Sop::CellAlloc { dst, .. }
            | Sop::DerefWriteCell { dst, .. }
            | Sop::DerefReadCell { dst, .. }
            | Sop::Scalar { dst, .. }
            | Sop::Seq { dst, .. } => *dst,
        };
        if !live[dst as usize] {
            continue;
        }
        match op {
            Sop::Const { .. } | Sop::Void { .. } => {}
            Sop::CellAlloc { src, .. } => live[*src as usize] = true,
            Sop::DerefWriteCell { cell, val, .. } => {
                live[*cell as usize] = true;
                live[*val as usize] = true;
            }
            Sop::DerefReadCell { cell, .. } => live[*cell as usize] = true,
            Sop::Scalar { a, b, .. } => {
                live[*a as usize] = true;
                live[*b as usize] = true;
            }
            Sop::Seq { src, .. } => {
                if let Some(src) = src {
                    live[*src as usize] = true;
                }
            }
        }
        kept.push(op.clone());
    }
    kept.reverse();
    kept
}

thread_local! {
    /// Worker-local result temporaries, reused across launches. Cleared at
    /// launch start (drops the previous launch's Arc'd results); each op
    /// pushes its result in order, so temps[i] is always op i's result.
    static SCALAR_TEMPS: std::cell::RefCell<Vec<Value>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Runs the compiled def-use program: params in, straight-line ops in plan
/// order, return reference out. Borrow discipline: every Op(i) operand is
/// read before the next push (plan order guarantees the producer ran), and
/// arith reads borrow — no Value clones on the compute path.
pub(crate) fn run_scalar_prog(prog: &ScalarProg, args: &[Value]) -> Value {
    fn fetch<'a>(r: &OpRef, temps: &'a [Value], consts: &'a [Value], args: &'a [Value]) -> &'a Value {
        match r {
            OpRef::Op(i) => &temps[*i as usize],
            OpRef::Const(c) => &consts[*c as usize],
            // Param(MAX) = unseeded slot: read as NULL (generic-path parity).
            OpRef::Param(p) => args.get(*p as usize).unwrap_or(&Value::NULL),
        }
    }
    SCALAR_TEMPS.with(|cell| {
        let mut temps = cell.borrow_mut();
        temps.clear();
        let prog_consts = &prog.consts;
        for op in &prog.ops {
            let v = match op {
                DSop::Scalar { a, b, ty, op, unary } => {
                    let va = fetch(a, &temps, prog_consts, args);
                    let vb = fetch(b, &temps, prog_consts, args);
                    exec_scalar_op(*ty, *op, *unary, va, vb)
                }
                DSop::CellAlloc { src } => {
                    let init = fetch(src, &temps, prog_consts, args);
                    Value::ref_val(crate::value::HeapObj::Cell(crate::value::Cell::new(
                        init.clone(),
                    )))
                }
                DSop::DerefWrite { cell, val } => {
                    let cell_v = fetch(cell, &temps, prog_consts, args).clone();
                    let new_val = fetch(val, &temps, prog_consts, args).clone();
                    if let Some(crate::value::HeapObj::Cell(c)) = cell_v.heap_obj() {
                        c.set(new_val.clone());
                    }
                    new_val
                }
                DSop::DerefRead { cell } => {
                    let cv = fetch(cell, &temps, prog_consts, args);
                    match cv.heap_obj() {
                        Some(crate::value::HeapObj::Cell(c)) => c.get(),
                        _ => cv.clone(),
                    }
                }
            };
            temps.push(v);
        }
        fetch(&prog.return_ref, &temps, prog_consts, args).clone()
    })
}
