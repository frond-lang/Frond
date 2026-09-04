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
//! by the engine's synchronous fast path
//! (bit-identical to the generic executor: same kernels, same order, only
//! pure forwarding removed).
//!
//! [`DataFlowGraph`]: crate::ir::Ir::DataFlowGraph

use crate::ir::Ir::{DataFlowGraph, NodeId, SubGraphId};
use crate::value::Value;

//
// A pure-leaf scalar plan that consists solely of const/noop, seq, f64
// arithmetic, cell alloc and cell deref read/write executes as a compiled
// straight-line program over a slot array: the per-node work of the generic
// executor (compute_fn pointer dispatch, node()/inputs() SoA resolution,
// readiness bitmaps, pending countdowns, notify_downstream) collapses into
// pre-resolved slot-index arithmetic. The op semantics mirror the
// corresponding compute_fns exactly (Value::f64 construction, as_f64
// coercion, Cell::set/get, seq last-input pass-through, const precedence);
// any node outside the supported set keeps the whole subgraph on the generic
// path — the fast path is a specialization, never a semantic fork.

#[derive(Clone, Copy, PartialEq, Debug)]
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

#[derive(Clone, Copy, PartialEq, Debug)]
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
    /// cf=29/288: record/Adt/Newtype construction — fields from slots in
    /// input order (mirrors compute_record_construct: all inputs are fields,
    /// shape from the per-node materialized RecordShape). Allocation +
    /// registry registration are real side effects: NEVER dead-code-eliminated.
    RecordConstruct { dst: u32, shape: std::sync::Arc<crate::value::RecordShape>, srcs: Vec<u32> },
    /// cf=30: by-name field read (mirrors compute_record_field_get exactly:
    /// record_field_get first, heap_obj fallback, FieldError throw).
    FieldGet { dst: u32, src: u32, name: String },
    /// Pure selection: a match/if else-chain gate whose arms are pure value
    /// chains (no control flow, no effects) compiles to a scalar select —
    /// `cond ? a : b`. The arm bodies are INLINED into this program (their
    /// gids are nested inside the owning sg, so they reuse its slot space).
    /// Pure value selection (cond ? a : b).
    Select { dst: u32, cond: u32, a: u32, b: u32 },
    /// cf=32: array/str element read (mirrors compute_array_index exactly:
    /// str→codepoint char, Array→element clone, negative/OOB panics).
    ArrayIndex { dst: u32, arr: u32, idx: u32 },
    /// cf=299: in-place array element store (effect; mirrors
    /// compute_array_store via array_store_inplace). Result is VOID.
    ArrayStore { dst: u32, arr: u32, idx: u32, val: u32 },
}

/// An operand reference in the def-use program.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpRef {
    /// Result of the op at this index (written this launch, before any read).
    Op(u32),
    /// Immutable constant owned by the program (index into `consts`).
    Const(u32),
    /// Launch argument slot (`p < param_count` → args[p]; beyond that → the
    /// outer-value slice: `outers[p - param_count]`, read from the launching
    /// frame at launch time).
    Param(u32),
    /// Condition-program reference to a BODY-program export (index into the
    /// body prog's export list — the tight-loop driver feeds the body's
    /// per-iteration computed values to the condition program).
    Body(u32),
    /// An sg-local slot no op in this program defines (read as NULL on the
    /// plain paths; the condition builder remaps these to `Body` refs).
    Undef(u32),
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
    /// Record construction (allocation + registration are real effects).
    RecordConstruct { fields: Vec<OpRef>, shape: std::sync::Arc<crate::value::RecordShape> },
    /// By-name field read (record_field_get semantics).
    FieldGet { src: OpRef, name: String },
    /// Pure value selection (cond ? a : b).
    Select { cond: OpRef, a: OpRef, b: OpRef },
    /// Array/str element read.
    ArrayIndex { arr: OpRef, idx: OpRef },
    /// In-place array element store (effect).
    ArrayStore { arr: OpRef, idx: OpRef, val: OpRef },
}

pub(crate) struct ScalarProg {
    ops: Vec<DSop>,
    /// Immutable constant pool (owned; cloned only at use sites that need an
    /// owned Value — arith reads borrow).
    consts: Vec<Value>,
    param_count: usize,
    /// What the subgraph returns, as a reference.
    return_ref: OpRef,
    /// Outside-the-subgraph input gids (deduplicated, plan order). Their
    /// values are read from the launching frame per launch and passed as the
    /// outer slice; `OpRef::Param(param_count + i)` refers to outer i.
    pub(crate) outer_gids: Vec<u32>,
    /// Body-program exports: (sg-local slot, defining op index) pairs. The
    /// tight-loop driver collects these per iteration and feeds them to the
    /// condition program's `OpRef::Body` references. Empty for plain programs.
    pub(crate) exports: Vec<(u32, u32)>,
}


// =========================================================================
// SIMD map: `while i < n { out[i] = expr(a[i], b[i], consts...) ; i = i + 1 }`
// recognized over the tight-loop program pair and executed lane-packed via
// the `wide` crate against the arrays' contiguous SoA buffers.
// =========================================================================

/// Lane family (v1: i32x8 / f64x4).
#[derive(Clone, Copy, PartialEq)]
pub enum SimdFamily {
    I32x8,
    F64x4,
}

/// A recognized map loop. Op indices refer to the BODY program's DSop list;
/// `n_ref` to the CONDITION program's operand space.
pub struct SimdMapPlan {
    pub family: SimdFamily,
    /// Reduction: the accumulator (a real CELL read+written each iteration).
    /// `acc_read` is the body-prog op index of its DerefRead; evaluating the
    /// spine with the read as ZERO yields the per-element contribution —
    /// i32 wrapping add is associative, so lane-grouped accumulation is
    /// bit-identical to the sequential fold. f64 reductions are REJECTED
    /// (reordering changes rounding).
    pub reduction: Option<(OpRef, u32)>,
    /// The induction variable reference in the BODY program — a Param (the
    /// outer i-value slot; the devirtualized cell read) that every feed and
    /// the store index by.
    pub i_ref: OpRef,
    /// Body-prog op indices of the ArrayIndex feeds (in op order).
    pub feeds: Vec<u32>,
    /// Body-prog op indices of the pure Scalar expression chain (post-order).
    pub exprs: Vec<u32>,
    /// Body-prog op index of the single ArrayStore (None for reductions).
    pub store: Option<u32>,
    /// Condition-prog operand of the trip bound (`i < n`'s right side).
    pub n_ref: OpRef,
    /// Condition-prog operand of the i cell (the DerefRead's cell input).
    pub cond_i_cell: OpRef,
}

/// Shape analysis over the tight-loop program pair. Any deviation rejects
/// (the driver falls back to the scalar tight loop).
pub(crate) fn analyze_simd_map(
    body: &ScalarProg,
    cond: &ScalarProg,
) -> Option<std::sync::Arc<SimdMapPlan>> {
    // ── Condition: `i < n`. Two legal lowerings: [DerefRead(cell), Lt] or
    // the devirtualized single-op [Lt{a: Param(cell)}] (the cell is
    // read-only inside the condition program → forwarded to the outer slot).
    let (cond_i_cell, n_ref) = match cond.ops.as_slice() {
        [DSop::DerefRead { cell }, DSop::Scalar { a, b, ty: STy::I32, op: SOp::Lt, unary: false }] => {
            if !matches!(a, OpRef::Op(k) if *k == 0) {
                return None;
            }
            (*cell, *b)
        }
        [DSop::Scalar { a, b, ty: STy::I32, op: SOp::Lt, unary: false }] => match a {
            OpRef::Param(p) if (*p as usize) >= cond.param_count => (*a, *b),
            _ => return None,
        },
        _ => return None,
    };
    if matches!(n_ref, OpRef::Op(_)) {
        return None; // bound must be a const/param/outer scalar
    }

    // ── Body: partition ops into i-read / feeds / exprs / store / increment. ──
    // Pre-scan: the increment DerefWrite's value op (index arithmetic).
    let mut increment_val_op: Option<u32> = None;
    for op in body.ops.iter() {
        if let DSop::DerefWrite { val: OpRef::Op(v), .. } = op {
            increment_val_op = Some(*v);
        }
    }
    let mut i_ref: Option<OpRef> = None;
    let mut acc_read: Option<u32> = None;
    let mut acc_write: Option<u32> = None;
    let mut acc_cell: Option<OpRef> = None;
    let mut feeds: Vec<u32> = Vec::new();
    let mut exprs: Vec<u32> = Vec::new();
    let mut store: Option<u32> = None;
    let mut increment: Option<u32> = None;
    let mut family: Option<SimdFamily> = None;

    for (k, op) in body.ops.iter().enumerate() {
        match op {
            DSop::DerefRead { .. } => {
                // The accumulator's read (a real cell) — at most one, paired
                // with its write below. No other cells (v1).
                if acc_read.is_some() {
                    return None;
                }
                acc_read = Some(k as u32);
            }
            DSop::DerefWrite { cell, val } => {
                // Either the i increment (val = Add(i_ref, Const), i CELL) or
                // the accumulator write (spine rooted at the acc DR).
                let is_increment = match val {
                    OpRef::Op(v) => matches!(
                        &body.ops[*v as usize],
                        DSop::Scalar { a, b, ty: STy::I32, op: SOp::Add, unary: false }
                            if (Some(*a) == i_ref || Some(*a) == Some(*cell))
                                && matches!(b, OpRef::Const(_))
                    ),
                    _ => false,
                };
                if is_increment {
                    if increment.is_some() {
                        return None;
                    }
                    increment = Some(k as u32);
                } else {
                    let Some(ar) = acc_read else {
                        return None;
                    };
                    let roots_at_acc = match val {
                        OpRef::Op(v) => match &body.ops[*v as usize] {
                            DSop::DerefRead { .. } => *v as u32 == ar,
                            DSop::Scalar { a, .. } => match a {
                                OpRef::Op(v2) if *v2 as u32 == ar => true,
                                _ => spine_roots_at(body, *v, ar),
                            },
                            _ => false,
                        },
                        _ => false,
                    };
                    if !roots_at_acc || acc_write.is_some() {
                        return None;
                    }
                    acc_write = Some(k as u32);
                    acc_cell = Some(*cell);
                }
            }
            DSop::ArrayIndex { arr: _, idx } => {
                if !matches!(idx, OpRef::Param(_)) {
                    return None;
                }
                let r = *idx;
                if let Some(prev) = i_ref {
                    if prev != r {
                        return None;
                    }
                } else {
                    i_ref = Some(r);
                }
                feeds.push(k as u32);
            }
            DSop::ArrayStore { arr: _, idx, val: _ } => {
                if Some(*idx) != i_ref {
                    return None;
                }
                if store.is_some() {
                    return None;
                }
                store = Some(k as u32);
            }
            DSop::Scalar { a, b, ty, op, unary } => {
                // The increment's operand op (index arithmetic, always I32)
                // is machinery — not part of the VALUE family.
                if increment_val_op == Some(k as u32) {
                    continue;
                }
                if *unary {
                    return None;
                }
                let allowed = match ty {
                    STy::I32 => {
                        if family.is_none() {
                            family = Some(SimdFamily::I32x8);
                        }
                        matches!(
                            op,
                            SOp::Add | SOp::Sub | SOp::Mul
                                | SOp::Shl | SOp::Shr | SOp::BitAnd | SOp::BitOr | SOp::BitXor
                        )
                        // Div/Mod excluded: scalar path panics on /0; the
                        // packed wrapping_div would diverge. Shift/bitwise
                        // are lane-exact (strength reduction emits Shl for
                        // `* 2`).
                    }
                    STy::F64 => {
                        if family.is_none() {
                            family = Some(SimdFamily::F64x4);
                        }
                        matches!(op, SOp::Add | SOp::Sub | SOp::Mul | SOp::Div)
                    }
                    _ => return None,
                };
                if !allowed {
                    return None;
                }
                if family != Some(SimdFamily::I32x8) && *ty == STy::I32 {
                    return None;
                }
                if family != Some(SimdFamily::F64x4) && *ty == STy::F64 {
                    return None;
                }
                let _ = (a, b);
                exprs.push(k as u32);
            }
            _ => return None, // CellAlloc / FieldGet / Select / Record: not a map
        }
    }
    let (Some(i_ref), Some(_increment)) = (i_ref, increment) else {
        return None;
    };
    if cond_i_cell == i_ref {
        return None; // the i reference IS the cell object — undevirtualized form
    }
    // Map (single store) or reduction (acc read+write, NO store) — not both.
    let reduction: Option<(OpRef, u32)> = match (acc_read, acc_write, acc_cell, store) {
        (Some(ar), Some(_aw), Some(ac), None) => Some((ac, ar)),
        (None, None, None, Some(_)) => None,
        _ => return None,
    };
    if feeds.is_empty() {
        return None; // expr with no array feed: scalar loop is fine
    }
    Some(std::sync::Arc::new(SimdMapPlan {
        family: family?,
        reduction,
        i_ref,
        feeds,
        exprs,
        store,
        n_ref,
        cond_i_cell,
    }))
}

/// SIMD reduction: `acc = acc ⊕ contribution(i)` folded over [i0, n). The
/// accumulator's read is substituted with ZERO in the per-element
/// contribution (the identity of +); lane-grouped wrapping i32 addition is
/// associative, so the grouping is bit-identical to the sequential fold.
/// f64 reductions never reach here (the analysis rejects the family).
pub(crate) fn run_simd_reduction(
    plan: &SimdMapPlan,
    body: &ScalarProg,
    cond: &ScalarProg,
    cond_outers: &[Value],
    body_outers: &[Value],
    red: (OpRef, u32),
) -> bool {
    use crate::value::{HeapObj, ScalarSoA};
    let (acc_cell_ref, acc_read) = red;

    // n / i0 — the same resolutions as the map path.
    let n_val = match &plan.n_ref {
        OpRef::Const(c) => cond.consts[*c as usize].clone(),
        OpRef::Param(p) if (*p as usize) >= cond.param_count => {
            cond_outers.get(*p as usize - cond.param_count).cloned().unwrap_or(Value::NULL)
        }
        _ => return false,
    };
    let n = n_val.as_i64();
    let i_val = match &plan.i_ref {
        OpRef::Param(p) if (*p as usize) >= body.param_count => {
            body_outers.get(*p as usize - body.param_count).cloned().unwrap_or(Value::NULL)
        }
        _ => return false,
    };
    let i0 = i_val.as_i64();
    // The i CELL (final write).
    let i_cell_val = match &plan.cond_i_cell {
        OpRef::Param(p) if (*p as usize) >= cond.param_count => {
            cond_outers.get(*p as usize - cond.param_count).cloned().unwrap_or(Value::NULL)
        }
        _ => return false,
    };
    let i_cell_arc = match &i_cell_val {
        crate::value::Value::Ref(arc) if matches!(arc.as_ref(), HeapObj::Cell(_)) => {
            Some(std::sync::Arc::clone(arc))
        }
        _ => None,
    };
    let Some(i_cell_arc) = i_cell_arc else { return false };
    // The ACCUMULATOR cell.
    let acc_val = match &acc_cell_ref {
        OpRef::Param(p) if (*p as usize) >= body.param_count => {
            body_outers.get(*p as usize - body.param_count).cloned().unwrap_or(Value::NULL)
        }
        _ => return false,
    };
    let acc_arc = match &acc_val {
        crate::value::Value::Ref(arc) if matches!(arc.as_ref(), HeapObj::Cell(_)) => {
            Some(std::sync::Arc::clone(arc))
        }
        _ => None,
    };
    let Some(acc_arc) = acc_arc else { return false };
    let acc0 = {
        let pr = std::sync::Arc::as_ptr(&acc_arc) as *mut HeapObj;
        unsafe {
            match &(*pr) {
                HeapObj::Cell(c) => c.get().as_i64(),
                _ => return false,
            }
        }
    };

    // Feeds (i32 SoA only — the family is I32 by analysis).
    let feed_arcs: Vec<std::sync::Arc<HeapObj>> = plan
        .feeds
        .iter()
        .map(|&f| match &body.ops[f as usize] {
            DSop::ArrayIndex { arr, .. } => match arr {
                OpRef::Param(p) if (*p as usize) >= body.param_count => body_outers
                    .get(*p as usize - body.param_count)
                    .and_then(|v| match v {
                        crate::value::Value::Ref(arc) => Some(std::sync::Arc::clone(arc)),
                        _ => None,
                    }),
                _ => None,
            },
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if feed_arcs.len() != plan.feeds.len() {
        return false;
    }
    let bufs: Vec<&Vec<i32>> = feed_arcs
        .iter()
        .map(|a| match a.as_ref() {
            HeapObj::Array(arr) => match &arr.scalar_soa {
                Some(ScalarSoA::I32(v)) => Some(v),
                _ => None,
            },
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if bufs.len() != feed_arcs.len() || bufs.iter().any(|b| b.len() < n as usize) {
        return false;
    }

    // The acc read op evaluates as ZERO inside the contribution.
    let mut acc_vec = wide::i32x8::splat(acc0 as i32);
    let getlane = |r: &OpRef,
                   lane: &rustc_hash::FxHashMap<u32, wide::i32x8>|
     -> Option<wide::i32x8> {
        match r {
            OpRef::Op(k) => {
                if *k == acc_read {
                    return Some(wide::i32x8::splat(0));
                }
                lane.get(k).copied()
            }
            OpRef::Const(c) => {
                Some(wide::i32x8::splat(body.consts[*c as usize].as_i32()))
            }
            OpRef::Param(p) if (*p as usize) >= body.param_count => body_outers
                .get(*p as usize - body.param_count)
                .map(|v| wide::i32x8::splat(v.as_i32())),
            _ => None,
        }
    };

    let mut i = i0;
    while i + 8 <= n {
        let base = i as usize;
        let mut lane: rustc_hash::FxHashMap<u32, wide::i32x8> = rustc_hash::FxHashMap::default();
        for (k2, &f) in plan.feeds.iter().enumerate() {
            let b = bufs[k2];
            lane.insert(
                f,
                wide::i32x8::new([
                    b[base], b[base + 1], b[base + 2], b[base + 3], b[base + 4], b[base + 5],
                    b[base + 6], b[base + 7],
                ]),
            );
        }
        for &k2 in plan.exprs.iter() {
            if k2 == acc_read {
                continue;
            }
            if let DSop::Scalar { a, b, op, .. } = &body.ops[k2 as usize] {
                let (Some(va), Some(vb)) = (getlane(a, &lane), getlane(b, &lane)) else {
                    return false;
                };
                let v = match op {
                    SOp::Add => va + vb,
                    SOp::Sub => va - vb,
                    SOp::Mul => va * vb,
                    SOp::Shl => va << vb,
                    SOp::Shr => va >> vb,
                    SOp::BitAnd => va & vb,
                    SOp::BitOr => va | vb,
                    SOp::BitXor => va ^ vb,
                    _ => return false,
                };
                lane.insert(k2, v);
            }
        }
        // The accumulator write's value = the contribution for these 8.
        if let Some(DSop::DerefWrite { val, .. }) = body
            .ops
            .iter()
            .find(|o| matches!(o, DSop::DerefWrite { cell, .. } if Some(*cell) == Some(acc_cell_ref)))
        {
            if let Some(v) = getlane(val, &lane) {
                acc_vec = acc_vec + v;
            } else {
                return false;
            }
        } else {
            return false;
        }
        i += 8;
    }
    // Horizontal sum (wrapping i32 — order-free).
    let lanes = acc_vec.to_array();
    let mut total: i32 = 0;
    for l in lanes.iter() {
        total = total.wrapping_add(*l);
    }
    // Scalar tail — sequential kernels (bit-identical).
    while i < n {
        let base = i as usize;
        let feed_vals: Vec<Value> = bufs.iter().map(|b| Value::i32(b[base])).collect();
        let mut out_val = Value::VOID;
        // contribution: evaluate the spine with the acc read = 0.
        let mut temps: rustc_hash::FxHashMap<u32, Value> = rustc_hash::FxHashMap::default();
        temps.insert(acc_read, Value::i32(0));
        for (k2, &f) in plan.feeds.iter().enumerate() {
            temps.insert(f, feed_vals[k2].clone());
        }
        for &k2 in plan.exprs.iter() {
            if let DSop::Scalar { a, b, ty, op, unary } = &body.ops[k2 as usize] {
                let va = temps.get(match a { OpRef::Op(v) => v, _ => &u32::MAX }).cloned().unwrap_or(Value::NULL);
                let vb = temps.get(match b { OpRef::Op(v) => v, _ => &u32::MAX }).cloned().unwrap_or(Value::NULL);
                let (va, vb) = match (a, b) {
                    (OpRef::Const(c), _) => (body.consts[*c as usize].clone(), vb),
                    (_, OpRef::Const(c)) => (va, body.consts[*c as usize].clone()),
                    _ => (va, vb),
                };
                let v = exec_scalar_op(*ty, *op, *unary, &va, &vb);
                temps.insert(k2, v);
            }
        }
        if let Some(DSop::DerefWrite { val, .. }) = body
            .ops
            .iter()
            .find(|o| matches!(o, DSop::DerefWrite { cell, .. } if Some(*cell) == Some(acc_cell_ref)))
        {
            out_val = match val {
                OpRef::Op(v) => temps.get(v).cloned().unwrap_or(Value::NULL),
                OpRef::Const(c) => body.consts[*c as usize].clone(),
                _ => Value::NULL,
            };
        }
        total = total.wrapping_add(out_val.as_i32());
        i += 1;
    }

    // Write the accumulator + the induction cell.
    {
        let pr = std::sync::Arc::as_ptr(&acc_arc) as *mut HeapObj;
        unsafe {
            if let HeapObj::Cell(c) = &mut (*pr) {
                c.set(Value::i32(total));
            }
        }
    }
    {
        let pr = std::sync::Arc::as_ptr(&i_cell_arc) as *mut HeapObj;
        unsafe {
            if let HeapObj::Cell(c) = &mut (*pr) {
                c.set(Value::i32(n as i32));
            }
        }
    }
    true
}

/// True when op `k`'s LEFT-spine of Scalar ops bottoms out at `ar` (the
/// accumulator read) without touching a feed on the left side.
fn spine_roots_at(body: &ScalarProg, k: u32, ar: u32) -> bool {
    let mut cur = k;
    for _ in 0..32 {
        match &body.ops[cur as usize] {
            DSop::Scalar { a, .. } => match a {
                OpRef::Op(v2) if *v2 as u32 == ar => return true,
                OpRef::Op(v2) => cur = *v2,
                _ => return false,
            },
            _ => return false,
        }
    }
    false
}

/// Evaluates the map's expression chain for ONE index (the SIMD tail). Uses
/// the exact scalar kernels — bit-identical to the generic path.
fn simd_tail_eval(
    plan: &SimdMapPlan,
    prog: &ScalarProg,
    feed_vals: &[Value],
    i: i64,
    out_val: &mut Value,
) {
    // temps indexed by op index (only feeds/exprs are needed).
    let mut temps: rustc_hash::FxHashMap<u32, Value> = rustc_hash::FxHashMap::default();
    for &f in plan.feeds.iter() {
        temps.insert(f, feed_vals[plan.feeds.iter().position(|&x| x == f).unwrap()].clone());
    }
    let _ = i;
    for &k in plan.exprs.iter() {
        match &prog.ops[k as usize] {
            DSop::Scalar { a, b, ty, op, unary } => {
                let va = fetch_opref(a, &temps, prog);
                let vb = fetch_opref(b, &temps, prog);
                let v = exec_scalar_op(*ty, *op, *unary, &va, &vb);
                temps.insert(k, v);
            }
            _ => {}
        }
    }
    if let Some(st) = plan.store {
        if let DSop::ArrayStore { val, .. } = &prog.ops[st as usize] {
            *out_val = fetch_opref(val, &temps, prog);
        }
    }
}

fn fetch_opref(r: &OpRef, temps: &rustc_hash::FxHashMap<u32, Value>, prog: &ScalarProg) -> Value {
    match r {
        OpRef::Op(k) => temps.get(k).cloned().unwrap_or(Value::NULL),
        OpRef::Const(c) => prog.consts[*c as usize].clone(),
        OpRef::Param(_) | OpRef::Undef(_) | OpRef::Body(_) => Value::NULL,
    }
}


/// Executes a recognized map loop lane-packed. Returns false (and does
/// nothing) when any RUNTIME precondition fails — the caller falls back to
/// the scalar tight loop.
///
/// `outer_val(r)` resolves an operand reference against the condition
/// program's outer/const space; `body_outer_val(r)` against the body's.
pub(crate) fn run_simd_map(
    plan: &SimdMapPlan,
    body: &ScalarProg,
    cond: &ScalarProg,
    cond_outers: &[Value],
    body_outers: &[Value],
) -> bool {
    use crate::value::{HeapObj, ScalarSoA};
    if let Some(red) = &plan.reduction {
        return run_simd_reduction(plan, body, cond, cond_outers, body_outers, *red);
    }

    // Resolve the trip bound and the i cell.
    let n_val = match &plan.n_ref {
        OpRef::Const(c) => cond.consts[*c as usize].clone(),
        OpRef::Param(p) if (*p as usize) >= cond.param_count => {
            cond_outers.get(*p as usize - cond.param_count).cloned().unwrap_or(Value::NULL)
        }
        _ => return false,
    };
    let n = n_val.as_i64();
    let i_cell_val = match &plan.cond_i_cell {
        OpRef::Param(p) if (*p as usize) >= cond.param_count => {
            cond_outers.get(*p as usize - cond.param_count).cloned().unwrap_or(Value::NULL)
        }
        _ => return false,
    };
    let cell_arc = match &i_cell_val {
        crate::value::Value::Ref(arc) if matches!(arc.as_ref(), HeapObj::Cell(_)) => {
            Some(std::sync::Arc::clone(arc))
        }
        _ => None,
    };
    let Some(cell_arc) = cell_arc else { return false };
    let i0 = {
        let p = std::sync::Arc::as_ptr(&cell_arc) as *mut HeapObj;
        unsafe {
            match &(*p) {
                HeapObj::Cell(c) => c.get().as_i64(),
                _ => return false,
            }
        }
    };
    if i0 > n {
        return false; // loop already done — let the scalar path handle it
    }

    // Resolve every feed array + the store target.
    let feed_arrs: Vec<std::sync::Arc<HeapObj>> = plan
        .feeds
        .iter()
        .map(|&f| match &body.ops[f as usize] {
            DSop::ArrayIndex { arr, .. } => match arr {
                OpRef::Param(p) if (*p as usize) >= body.param_count => body_outers
                    .get(*p as usize - body.param_count)
                    .and_then(|v| match v {
                        crate::value::Value::Ref(arc) => Some(std::sync::Arc::clone(arc)),
                        _ => None,
                    }),
                _ => None,
            },
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if feed_arrs.len() != plan.feeds.len() {
        return false;
    }
    let Some(store_op) = plan.store else { return false };
    let out_arc = match &body.ops[store_op as usize] {
        DSop::ArrayStore { arr, .. } => match arr {
            OpRef::Param(p) if (*p as usize) >= body.param_count => body_outers
                .get(*p as usize - body.param_count)
                .and_then(|v| match v {
                    crate::value::Value::Ref(arc) => Some(std::sync::Arc::clone(arc)),
                    _ => None,
                }),
            _ => None,
        },
        _ => None,
    };
    let Some(out_arc) = out_arc else { return false };

    // SoA buffers (mutable view of the output through the Arc — the same
    // interior-mutability argument as array_store_inplace).
    fn soa_i32(a: &std::sync::Arc<HeapObj>) -> Option<&Vec<i32>> {
        match a.as_ref() {
            HeapObj::Array(arr) => match &arr.scalar_soa {
                Some(ScalarSoA::I32(v)) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }
    fn soa_i32_mut(a: &std::sync::Arc<HeapObj>) -> Option<&mut Vec<i32>> {
        let ptr = std::sync::Arc::as_ptr(a) as *mut HeapObj;
        unsafe {
            match &(*ptr) {
                HeapObj::Array(arr) => match &arr.scalar_soa {
                    Some(ScalarSoA::I32(_)) => match &mut (*ptr) {
                        HeapObj::Array(arr2) => match &mut arr2.scalar_soa {
                            Some(ScalarSoA::I32(v)) => Some(v),
                            _ => None,
                        },
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            }
        }
    }
    fn soa_f64(a: &std::sync::Arc<HeapObj>) -> Option<&Vec<f64>> {
        match a.as_ref() {
            HeapObj::Array(arr) => match &arr.scalar_soa {
                Some(ScalarSoA::F64(v)) => Some(v),
                _ => None,
            },
            _ => None,
        }
    }
    fn soa_f64_mut(a: &std::sync::Arc<HeapObj>) -> Option<&mut Vec<f64>> {
        let ptr = std::sync::Arc::as_ptr(a) as *mut HeapObj;
        unsafe {
            match &(*ptr) {
                HeapObj::Array(arr) => match &arr.scalar_soa {
                    Some(ScalarSoA::F64(_)) => match &mut (*ptr) {
                        HeapObj::Array(arr2) => match &mut arr2.scalar_soa {
                            Some(ScalarSoA::F64(v)) => Some(v),
                            _ => None,
                        },
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            }
        }
    }

    // Const/outer scalar operands for the expression chain.
    let scalar_of = |r: &OpRef| -> Option<Value> {
        match r {
            OpRef::Const(c) => Some(body.consts[*c as usize].clone()),
            OpRef::Param(p) if (*p as usize) >= body.param_count => {
                body_outers.get(*p as usize - body.param_count).cloned()
            }
            _ => None,
        }
    };

    match plan.family {
        SimdFamily::I32x8 => {
            let bufs: Vec<&Vec<i32>> = feed_arrs.iter().map(|a| soa_i32(a)).collect::<Option<Vec<_>>>().unwrap_or_default();
            if bufs.len() != feed_arrs.len() {
                return false;
            }
            let Some(out_buf) = soa_i32_mut(&out_arc) else { return false };
            if bufs.iter().any(|b| b.len() < n as usize) || out_buf.len() < n as usize {
                return false; // OOB possible — scalar path panics correctly
            }
            let mut i = i0;
            // Packs: feed k -> lane value; expr op k -> lane value.
            while i + 8 <= n {
                let base = i as usize;
                let mut lane: rustc_hash::FxHashMap<u32, wide::i32x8> =
                    rustc_hash::FxHashMap::default();
                for (k, &f) in plan.feeds.iter().enumerate() {
                    let b = bufs[k];
                    lane.insert(f, wide::i32x8::new([
                        b[base], b[base + 1], b[base + 2], b[base + 3],
                        b[base + 4], b[base + 5], b[base + 6], b[base + 7],
                    ]));
                }
                let getlane = |r: &OpRef, lane: &rustc_hash::FxHashMap<u32, wide::i32x8>| -> Option<wide::i32x8> {
                    match r {
                        OpRef::Op(k) => lane.get(k).copied(),
                        OpRef::Const(c) => Some(wide::i32x8::splat(body.consts[*c as usize].as_i32())),
                        OpRef::Param(p) if (*p as usize) >= body.param_count => body_outers
                            .get(*p as usize - body.param_count)
                            .map(|v| v.as_i32())
                            .map(wide::i32x8::splat),
                        _ => None,
                    }
                };
                let mut ok = true;
                for &k in plan.exprs.iter() {
                    if let DSop::Scalar { a, b, op, .. } = &body.ops[k as usize] {
                        let (Some(va), Some(vb)) = (getlane(a, &lane), getlane(b, &lane)) else {
                            ok = false;
                            break;
                        };
                        let v = match op {
                            SOp::Add => va + vb,
                            SOp::Sub => va - vb,
                            SOp::Mul => va * vb,
                            SOp::Shl => va << vb,
                            SOp::Shr => va >> vb,
                            SOp::BitAnd => va & vb,
                            SOp::BitOr => va | vb,
                            SOp::BitXor => va ^ vb,
                            _ => { ok = false; break }
                        };
                        lane.insert(k, v);
                    }
                }
                if !ok {
                    return false;
                }
                if let DSop::ArrayStore { val, .. } = &body.ops[store_op as usize] {
                    if let Some(v) = getlane(val, &lane) {
                        let arr = v.to_array();
                        out_buf[base..base + 8].copy_from_slice(&arr);
                    } else {
                        return false;
                    }
                }
                i += 8;
            }
            // Tail: scalar kernels per element (bit-identical semantics).
            while i < n {
                let base = i as usize;
                let feed_vals: Vec<Value> = bufs.iter().map(|b| Value::i32(b[base])).collect();
                let mut out_val = Value::VOID;
                simd_tail_eval(plan, body, &feed_vals, i, &mut out_val);
                out_buf[base] = out_val.as_i32();
                i += 1;
            }
        }
        SimdFamily::F64x4 => {
            let bufs: Vec<&Vec<f64>> = feed_arrs.iter().map(|a| soa_f64(a)).collect::<Option<Vec<_>>>().unwrap_or_default();
            if bufs.len() != feed_arrs.len() {
                return false;
            }
            let Some(out_buf) = soa_f64_mut(&out_arc) else { return false };
            if bufs.iter().any(|b| b.len() < n as usize) || out_buf.len() < n as usize {
                return false;
            }
            let mut i = i0;
            while i + 4 <= n {
                let base = i as usize;
                let mut lane: rustc_hash::FxHashMap<u32, wide::f64x4> =
                    rustc_hash::FxHashMap::default();
                for (k, &f) in plan.feeds.iter().enumerate() {
                    let b = bufs[k];
                    lane.insert(f, wide::f64x4::new([b[base], b[base + 1], b[base + 2], b[base + 3]]));
                }
                let getlane = |r: &OpRef, lane: &rustc_hash::FxHashMap<u32, wide::f64x4>| -> Option<wide::f64x4> {
                    match r {
                        OpRef::Op(k) => lane.get(k).copied(),
                        OpRef::Const(c) => Some(wide::f64x4::splat(body.consts[*c as usize].as_f64())),
                        OpRef::Param(p) if (*p as usize) >= body.param_count => body_outers
                            .get(*p as usize - body.param_count)
                            .map(|v| v.as_f64())
                            .map(wide::f64x4::splat),
                        _ => None,
                    }
                };
                let mut ok = true;
                for &k in plan.exprs.iter() {
                    if let DSop::Scalar { a, b, op, .. } = &body.ops[k as usize] {
                        let (Some(va), Some(vb)) = (getlane(a, &lane), getlane(b, &lane)) else {
                            ok = false;
                            break;
                        };
                        let v = match op {
                            SOp::Add => va + vb,
                            SOp::Sub => va - vb,
                            SOp::Mul => va * vb,
                            SOp::Div => va / vb,
                            _ => { ok = false; break }
                        };
                        lane.insert(k, v);
                    }
                }
                if !ok {
                    return false;
                }
                if let DSop::ArrayStore { val, .. } = &body.ops[store_op as usize] {
                    if let Some(v) = getlane(val, &lane) {
                        let arr = v.to_array();
                        out_buf[base..base + 4].copy_from_slice(&arr);
                    } else {
                        return false;
                    }
                }
                i += 4;
            }
            while i < n {
                let base = i as usize;
                let feed_vals: Vec<Value> = bufs.iter().map(|b| Value::f64(b[base])).collect();
                let mut out_val = Value::VOID;
                simd_tail_eval(plan, body, &feed_vals, i, &mut out_val);
                out_buf[base] = out_val.as_f64();
                i += 1;
            }
        }
    }
    // Leave the induction cell at n — exactly what the scalar loop does.
    {
        let p = std::sync::Arc::as_ptr(&cell_arc) as *mut HeapObj;
        unsafe {
            if let HeapObj::Cell(c) = &mut (*p) {
                c.set(Value::i32(n as i32));
            }
        }
    }
    let _ = scalar_of;
    true
}

/// Supported structural compute_fn ids.
const CF_NOOP_OR_CONST: u32 = 0;
const CF_SEQ: u32 = 47;
const CF_DEREF_READ: u32 = 279;
const CF_DEREF_WRITE: u32 = 280;
const CF_CELL_ALLOC: u32 = 349;
const CF_RECORD_CONSTRUCT: u32 = 29;
const CF_RECORD_CONSTRUCT_STACK: u32 = 288;
const CF_RECORD_FIELD_GET: u32 = 30;

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
    let plan: Vec<NodeId> = graph.linear_plan(sg_id.0 as usize)?.to_vec();
    if plan.is_empty() {
        return None;
    }
    let ret = graph.subgraphs[sg_id.0 as usize].return_node;
    build_scalar_prog_for_ex(graph, sg_id, &plan, ret, &[], &rustc_hash::FxHashMap::default())
}


/// Select pre-expansion: walks the plan; every Gate whose shape is a PURE
/// VALUE SELECTION gets its else-chain recursively inlined.
///
/// Shape requirements per gate (any failure reverts the WHOLE program to the
/// generic path — select-ization is all-or-nothing per sg):
/// - `capture = false`, exactly one true + one false branch;
/// - every arm sg: sync, no nested subgraphs of its own, own-plan nodes all
///   in the supported scalar/select/construct/field/deref/const set (checked
///   implicitly by the main classification loop — here we only collect);
/// - the chain TERMINAL (a false arm that is not a wrap-with-gate) must be a
///   panic/void sg AND the last gate's condition a tautology (const-true or
///   CF_MATCH_FALLBACK) — the select then degenerates to a Seq forward.
///
/// Returns (node list in topological order, gate → (cond, true_ret,
/// false_ret), param placeholder → source gid).
fn expand_selects(
    graph: &DataFlowGraph,
    sg_id: SubGraphId,
    plan: &[NodeId],
) -> Option<(Vec<NodeId>, rustc_hash::FxHashMap<u32, (u32, u32, u32)>, rustc_hash::FxHashMap<u32, u32>)> {
    use rustc_hash::FxHashMap;

    let mut has_gate = false;
    for &g in plan {
        if graph.node(g.0 as usize).kind == crate::ir::Ir::NodeKind::Gate {
            has_gate = true;
            break;
        }
    }
    if !has_gate {
        // Fast path: no gates — the caller's plan order is already valid.
        return Some((plan.to_vec(), FxHashMap::default(), FxHashMap::default()));
    }

    let mut nodes: Vec<NodeId> = plan.to_vec();
    let mut gate_info: FxHashMap<u32, (u32, u32, u32)> = FxHashMap::default();
    let mut param_src: FxHashMap<u32, u32> = FxHashMap::default();

    // Transitive placeholder resolution: a wrap's param source can itself be
    // a placeholder of an ENCLOSING wrap (the else-chain's scrutinee hops).
    // Register each placeholder against its ULTIMATE source — an outer
    // forward (SEQ chain) resolves further in the main loop's input mapping.
    let mut sg_of: FxHashMap<u32, usize> = FxHashMap::default();
    for (si, s) in graph.subgraphs.iter().enumerate() {
        let (sns, sne) = s.node_range;
        if sns.0 >= sne.0 {
            continue;
        }
        for g in sns.0..sne.0 {
            sg_of.entry(g).and_modify(|e| {
                let cur = graph.subgraphs[*e].node_range;
                let mine = graph.subgraphs[si].node_range;
                if (mine.1 .0 - mine.0 .0) < (cur.1 .0 - cur.0 .0) {
                    *e = si;
                }
            }).or_insert(si);
        }
    }
    let resolve_ph = |mut gid: u32, launcher: &FxHashMap<u32, Vec<u32>>| -> u32 {
        for _ in 0..16 {
            let n = graph.node(gid as usize);
            if n.compute_fn.0 != 0 || graph.const_value(gid as usize).is_some() {
                return gid;
            }
            let Some(&owner) = sg_of.get(&gid) else { return gid };
            let osg = &graph.subgraphs[owner];
            let pidx = gid.wrapping_sub(osg.node_range.0 .0);
            if pidx >= osg.param_count as u32 {
                return gid;
            }
            let Some(params) = launcher.get(&(owner as u32)) else { return gid };
            let Some(nx) = params.get(pidx as usize) else { return gid };
            if *nx == gid {
                return gid;
            }
            gid = *nx;
        }
        gid
    };

    // Is a sg a "value chain" compilable inline? Guards only; the main loop
    // does the per-node classification (a bail there reverts everything).
    let arm_ok = |sidx: usize| -> bool {
        let s = &graph.subgraphs[sidx];
        !s.has_suspend && s.event_source_decls.is_empty() && s.defer_table.is_empty() && s.nested_ranges.is_empty()
    };

    // Resolve a sg's return VALUE node (through SEQ forwards).
    let ret_value = |sidx: usize| -> Option<u32> {
        let mut r = graph.subgraphs[sidx].return_node.0;
        for _ in 0..16 {
            let n = graph.node(r as usize);
            if n.compute_fn.0 == 47 {
                let ins = graph.inputs(n.inputs_offset, n.input_count);
                r = ins.last()?.0;
            } else {
                return Some(r);
            }
        }
        None
    };

    // Recursively expand one gate.
    #[allow(clippy::too_many_arguments)]
    fn expand_gate(
        graph: &DataFlowGraph,
        gate: u32,
        nodes: &mut Vec<NodeId>,
        gate_info: &mut rustc_hash::FxHashMap<u32, (u32, u32, u32)>,
        param_src: &mut rustc_hash::FxHashMap<u32, u32>,
        arm_ok: &dyn Fn(usize) -> bool,
        ret_value: &dyn Fn(usize) -> Option<u32>,
        depth: u32,
        launcher: &FxHashMap<u32, Vec<u32>>,
        resolve_ph: &dyn Fn(u32, &FxHashMap<u32, Vec<u32>>) -> u32,
    ) -> Option<()> {
        if depth > 16 {
            return None;
        }
        if gate_info.contains_key(&gate) {
            return Some(()); // shared sub-chain (CSE'd else-tails)
        }
        let n = graph.node(gate as usize);
        if n.kind != crate::ir::Ir::NodeKind::Gate {
            return None;
        }
        let gb = graph.gate_branches_at(gate as usize)?;
        if gb.capture {
            return None;
        }
        let mut tbranch = None;
        let mut fbranch = None;
        for (c, tgt, params) in gb.branches.iter() {
            if *c {
                tbranch = Some((*tgt, params.clone()));
            } else {
                fbranch = Some((*tgt, params.clone()));
            }
        }
        let (Some((tsg, tparams)), Some((fsg, fparams))) = (tbranch, fbranch) else {
            return None;
        };
        if !arm_ok(tsg.0 as usize) {
            return None;
        }

        // Arm plan inliner: append the sg's own-plan nodes + register param
        // placeholders → source gids.
        let inline_arm = |sgidx: usize,
                          params: &[NodeId],
                          nodes: &mut Vec<NodeId>,
                          param_src: &mut rustc_hash::FxHashMap<u32, u32>,
                          launcher: &FxHashMap<u32, Vec<u32>>|
         -> Option<()> {
            // Nested ranges inside are fine — the collector skips them (the
            // recursion handles their content); only execution hazards gate.
            let s = &graph.subgraphs[sgidx];
            if s.has_suspend || !s.event_source_decls.is_empty() || !s.defer_table.is_empty() {
                return None;
            }
            let (sns, sne) = s.node_range;
            // Param placeholders = the sg's first param_count nodes.
            for (i, &src) in params.iter().enumerate() {
                let ph = sns.0 + i as u32;
                if (i as u32) < s.param_count as u32 {
                    param_src.insert(ph, resolve_ph(src.0, launcher));
                }
            }
            let inner_nested = graph.sg_nested_ranges(sgidx);
            for g2 in sns.0..sne.0 {
                if inner_nested.iter().any(|&(a, b)| g2 >= a && g2 < b) {
                    continue;
                }
                nodes.push(NodeId(g2));
            }
            Some(())
        };

        // TRUE arm: value chain (must be self-contained — no nesting).
        if !arm_ok(tsg.0 as usize) {
            return None;
        }
        inline_arm(tsg.0 as usize, &tparams, nodes, param_src, launcher)?;
        let tval = ret_value(tsg.0 as usize)?;

        // FALSE arm: a wrap (own node = the next gate; its nested ranges are
        // the INNER chain — the recursion collects those, so nesting is fine
        // here) or the terminal panic/void net.
        let fsg_i = fsg.0 as usize;
        let fsg_ref = &graph.subgraphs[fsg_i];
        if fsg_ref.has_suspend
            || !fsg_ref.event_source_decls.is_empty()
            || !fsg_ref.defer_table.is_empty()
        {
            return None;
        }
        let (wns, wne) = fsg_ref.node_range;
        let inner_nested = graph.sg_nested_ranges(fsg_i);
        let mut next_gate: Option<u32> = None;
        for g2 in wns.0..wne.0 {
            if inner_nested.iter().any(|&(a, b)| g2 >= a && g2 < b) {
                continue;
            }
            if graph.node(g2 as usize).kind == crate::ir::Ir::NodeKind::Gate {
                next_gate = Some(g2);
                break;
            }
        }
        let fval = match next_gate {
            Some(ng) => {
                inline_arm(fsg_i, &fparams, nodes, param_src, launcher)?;
                expand_gate(graph, ng, nodes, gate_info, param_src, arm_ok, ret_value, depth + 1, launcher, resolve_ph)?;
                ng
            }
            None => {
                // The false target is a LEAF (no inner gate): either a real
                // final ELSE arm (if-else — its return IS a value) or the
                // match panic/void net (return = CF_MATCH_FALLBACK /
                // non-value). A value arm becomes the select's b-side; the
                // panic net is unreachable for exhaustive matches — degenerate
                // to the true value (the generic executor panics there).
                match ret_value(fsg_i) {
                    Some(rv) if graph.node(rv as usize).compute_fn.0 != 311 => rv,
                    _ => tval,
                }
            }
        };
        gate_info.insert(gate, (gb.condition_input.0, tval, fval));
        Some(())
    }

    // Launcher table (target sg → branch param source gids) for the
    // transitive placeholder resolution.
    let mut launcher_of: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for g in 0..graph.node_count() {
        if let Some(gb) = graph.gate_branches_at(g) {
            for (_, tgt, params) in gb.branches.iter() {
                launcher_of
                    .entry(tgt.0)
                    .or_insert_with(|| params.iter().map(|p| p.0).collect());
            }
        }
    }

    // Expand every gate in the ORIGINAL plan (in plan order).
    let gates: Vec<u32> = plan
        .iter()
        .filter(|&g| graph.node(g.0 as usize).kind == crate::ir::Ir::NodeKind::Gate)
        .map(|g| g.0)
        .collect();
    for g in gates {
        expand_gate(
            graph,
            g,
            &mut nodes,
            &mut gate_info,
            &mut param_src,
            &arm_ok,
            &ret_value,
            0,
            &launcher_of,
            &resolve_ph,
        )?;
    }

    // Closure completion: any input that lands INSIDE the sg's range but
    // was not collected (effect-spine SEQs, M4-redirect targets, optimizer
    // rewiring) joins the list transitively — otherwise the sorter treats it
    // as external, the slot never gets a definition, and operands read as
    // Undef/NULL.
    {
        let (ns2, ne2) = graph.subgraphs[sg_id.0 as usize].node_range;
        let mut set: std::collections::HashSet<u32> = nodes.iter().map(|g| g.0).collect();
        let mut changed = true;
        while changed {
            changed = false;
            let snapshot: Vec<u32> = nodes.iter().map(|g| g.0).collect();
            for g2 in snapshot {
                let n = graph.node(g2 as usize);
                let ins = graph.inputs(n.inputs_offset, n.input_count);
                for &inp0 in ins {
                    let inp = param_src.get(&inp0.0).copied().unwrap_or(inp0.0);
                    if inp >= ns2.0 && inp < ne2.0 && set.insert(inp) {
                        nodes.push(NodeId(inp));
                        changed = true;
                    }
                }
                // Select arm/cond values arrive via gate_info — same rule.
                if let Some(&(c, a, b)) = gate_info.get(&g2) {
                    for v in [c, a, b] {
                        if v >= ns2.0 && v < ne2.0 && set.insert(v) {
                            nodes.push(NodeId(v));
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Topologically sort the collected set (inputs before uses; O(n²) —
    // bounded by the 512-node budget).
    let set: std::collections::HashSet<u32> = nodes.iter().map(|g| g.0).collect();
    let mut sorted: Vec<NodeId> = Vec::with_capacity(nodes.len());
    let mut emitted: std::collections::HashSet<u32> = std::collections::HashSet::new();
    while sorted.len() < nodes.len() {
        let mut progressed = false;
        for idx in (0..nodes.len()).rev() {
            let g = nodes[idx];
            if emitted.contains(&g.0) {
                continue;
            }
            let n = graph.node(g.0 as usize);
            let ins = graph.inputs(n.inputs_offset, n.input_count);
            // Placeholder indirection: the input mapping (below, in the main
            // classification) rewrites wrap-param placeholders to their
            // RESOLVED sources — the order dependency is on the source, not
            // the placeholder. Sorting on raw edges let consumers fire before
            // their producers (the Eq(Undef)/Mod-ordering bug).
            let ready = ins.iter().all(|&inp| {
                let r = param_src.get(&inp.0).copied().unwrap_or(inp.0);
                !set.contains(&r) || emitted.contains(&r)
            });
            // A select's arm values arrive via gate_info (not plain inputs);
            // their producers are in the set — wait for them too.
            let sel_ready = gate_info.get(&g.0).map_or(true, |&(c, a, b)| {
                (!set.contains(&c) || emitted.contains(&c))
                    && (!set.contains(&a) || emitted.contains(&a))
                    && (!set.contains(&b) || emitted.contains(&b))
            });
            if ready && sel_ready {
                emitted.insert(g.0);
                sorted.push(g);
                progressed = true;
            }
        }
        if !progressed {
            return None;
        }
    }
    Some((sorted, gate_info, param_src))
}

/// Builds a program over an EXPLICIT node list (topological order) with an
/// explicit return node — the sg's full plan for leaf/body programs, or the
/// condition-tree node set for loop-condition programs (`build_cond_prog`).
pub(crate) fn build_scalar_prog_for(
    graph: &DataFlowGraph,
    sg_id: SubGraphId,
    plan: &[NodeId],
    return_node: NodeId,
) -> Option<std::sync::Arc<ScalarProg>> {
    build_scalar_prog_for_ex(graph, sg_id, plan, return_node, &[], &rustc_hash::FxHashMap::default())
}

/// `export_slots`: sg-local slots whose per-iteration values the caller needs
/// (the tight-loop condition's body-defined operands). Seeded live; resolved
/// to defining op indices in `exports`.
pub(crate) fn build_scalar_prog_for_ex(
    graph: &DataFlowGraph,
    sg_id: SubGraphId,
    plan: &[NodeId],
    return_node: NodeId,
    export_slots: &[u32],
    undef_remap: &rustc_hash::FxHashMap<u32, OpRef>,
) -> Option<std::sync::Arc<ScalarProg>> {
    let sg = &graph.subgraphs[sg_id.0 as usize];
    let (ns, ne) = sg.node_range;
    // Select pre-expansion: gates in the plan whose arms are pure value
    // chains (else-chain wraps included) get their arm/wrap plans INLINED —
    // the arm gids are nested inside this sg's range, so they reuse its slot
    // space. Gate → Sop::Select{cond, a, b}; the recursive node list is then
    // topologically sorted (arm nodes reference the gate's own operands).
    let (node_list, gate_info, param_src) = expand_selects(graph, sg_id, plan)?;
    if node_list.len() > 512 {
        return None;
    }
    // Effect freeze for INLINED arm nodes: a select evaluates ALL arm values
    // eagerly — any side effect (cell write, record construction/alloc,
    // cell alloc) in an arm would execute UNCONDITIONALLY instead of only on
    // the selected path. Original-plan nodes are exempt (they execute anyway
    // in order; the normal classification still gates them).
    {
        let in_plan: std::collections::HashSet<u32> =
            plan.iter().map(|g| g.0).collect();
        for &g in node_list.iter() {
            if in_plan.contains(&g.0) {
                continue;
            }
            let n = graph.node(g.0 as usize);
            if n.kind == crate::ir::Ir::NodeKind::Gate {
                continue; // nested selects (their own arms get the same check)
            }
            let cf = n.compute_fn.0;
            let pure_value = cf == 0
                || cf == 47
                || cf == 279 /* deref READ */
                || cf == 30 /* field get */
                || cf == 32 /* array/str element read (pure) */
                || cf == 29 /* record construct: allocation observable only
                             through its VALUE — dead chains are DCE'd
                             (liveness-driven, below) */
                || cf == 288
                || classify_scalar_cf(cf).is_some();
            if !pure_value {
                return None;
            }
        }
    }
    let slot_count = (ne.0 - ns.0) as usize;
    // Outside-range inputs (outer-frame values: loop-carried cells, enclosing
    // locals) get pseudo-slots at [slot_count .. slot_count + outer_n); their
    // values are supplied per launch. Inputs on nodes OUTSIDE the sg's own
    // plan/nodes (params included) stay a hard bail — only plan-node inputs
    // may reach outward.
    let mut outer_gids: Vec<u32> = Vec::new();
    let mut to_local = |gid: u32| -> Option<u32> {
        let l = gid.wrapping_sub(ns.0);
        if l < slot_count as u32 {
            Some(l)
        } else {
            // Dedup by gid; pseudo-slot index = slot_count + position.
            outer_gids
                .iter()
                .position(|&g| g == gid)
                .map(|i| (slot_count as u32 + i as u32))
                .or_else(|| {
                    outer_gids.push(gid);
                    Some((slot_count + outer_gids.len() - 1) as u32)
                })
        }
    };
    let param_count = sg.param_count as usize;
    let mut ops: Vec<Sop> = Vec::with_capacity(plan.len());
    let mut cell_slots = vec![false; slot_count];
    for &gid in node_list.iter() {
        let n = graph.node(gid.0 as usize);
        let dst = match gid.0.wrapping_sub(ns.0) < slot_count as u32 {
            true => gid.0.wrapping_sub(ns.0),
            false => {
                return None;
            }
        };
        // Param slots are injected from the launch args (the generic path
        // seeds them ready and the plan loop skips them).
        if (dst as usize) < param_count {
            continue;
        }
        let mut inputs: Vec<u32> = Vec::with_capacity(n.input_count as usize);
        for &inp0 in graph.inputs(n.inputs_offset, n.input_count) {
            // Inlined arm/wrap param placeholders resolve to their ultimate
            // source BEFORE slot mapping — the source may be body-local OR
            // an outer node (the else-chain's scrutinee hops).
            let inp = match param_src.get(&inp0.0) {
                Some(&src) => NodeId(src),
                None => inp0,
            };
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
                } else if param_src.contains_key(&gid.0) {
                    // Placeholder: consumers resolved past it in the input
                    // mapping — emit nothing (its slot stays unbound, which
                    // is exactly the placeholder's own dead semantics).
                    continue;
                } else {
                    ops.push(Sop::Void { dst });
                }
            }
            cf if cf == 37 && n.kind == crate::ir::Ir::NodeKind::Gate => {
                // CF_GATE_LAUNCH on a select-shaped gate (gate_info presence
                // is the eligibility — non-select gates bailed in the
                // pre-pass and never reach the node list).
                let Some((cond_g, t_g, f_g)) = gate_info.get(&gid.0).copied() else {
                    return None;
                };
                let cslot = cond_g.wrapping_sub(ns.0);
                let aslot = t_g.wrapping_sub(ns.0);
                let bslot = f_g.wrapping_sub(ns.0);
                if cslot >= slot_count as u32 || aslot >= slot_count as u32 || bslot >= slot_count as u32 {
                    return None;
                }
                ops.push(Sop::Select { dst, cond: cslot, a: aslot, b: bslot });
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
                if inputs.len() < 2 {
                    return None;
                }
                let cell = inputs[0];
                // A chain-local cell must be a marked local alloc; an OUTER
                // slot is a real heap cell owned by the enclosing frame
                // (loop-carried vars) — always allowed, never devirtualized.
                // Local slots must be marked local cells; outer slots are
                // real heap cells (loop-carried vars) — always allowed.
                if (cell as usize) < slot_count && !cell_slots[cell as usize] {
                    return None;
                }
                ops.push(Sop::DerefWriteCell { dst, cell, val: inputs[1] });
            }
            CF_DEREF_READ => {
                if inputs.is_empty() {
                    return None;
                }
                let cell = inputs[0];
                if (cell as usize) < slot_count && !cell_slots[cell as usize] {
                    return None;
                }
                ops.push(Sop::DerefReadCell { dst, cell });
            }
            CF_RECORD_CONSTRUCT | CF_RECORD_CONSTRUCT_STACK => {
                // The generic path maps ALL inputs to fields in order.
                if graph.record_lit_info_at(gid.0 as usize).is_none() {
                    return None;
                }
                if graph.record_shapes.is_empty() {
                    return None;
                }
                let shape = graph.record_shapes.get(gid.0 as usize)?.clone();
                if inputs.is_empty() {
                    // Nullary construct: the value is metadata-determined —
                    // build it ONCE at compile time (E8's insight) and treat
                    // as an immutable constant.
                    let built = Value::Record(crate::value::RecordRef::new_from_iter(
                        shape.clone(),
                        std::iter::empty::<Value>(),
                    ));
                    ops.push(Sop::Const { dst, val: built });
                } else {
                    ops.push(Sop::RecordConstruct { dst, shape, srcs: inputs });
                }
            }
            32 /* CF_ARRAY_INDEX */ => {
                if inputs.len() < 2 {
                    return None;
                }
                ops.push(Sop::ArrayIndex { dst, arr: inputs[0], idx: inputs[1] });
            }
            299 /* CF_ARRAY_STORE */ => {
                if inputs.len() < 3 {
                    return None;
                }
                ops.push(Sop::ArrayStore { dst, arr: inputs[0], idx: inputs[1], val: inputs[2] });
            }
            CF_RECORD_FIELD_GET => {
                let Some(name) = graph.field_set_name(gid.0 as usize) else {
                    return None;
                };
                if inputs.is_empty() {
                    return None;
                }
                ops.push(Sop::FieldGet { dst, src: inputs[0], name: name.to_string() });
            }
            _ => return None,
        }
    }
    let return_slot = gid_ok(return_node.0, ns.0, slot_count as u32)?;
    let outer_count = outer_gids.len();
    let total = slot_count + outer_count;
    let ops = optimize_sops_ex(ops, param_count, total, return_slot, export_slots);
    let (dops, consts, return_ref, slot_ops) =
        lower_to_def_use_ex(ops, param_count, slot_count, outer_count, return_slot, undef_remap);
    let mut exports: Vec<(u32, u32)> = Vec::new();
    for &s in export_slots {
        match slot_ops.get((s as usize).min(total - 1)) {
            Some(Some(op_idx)) => exports.push((s, *op_idx)),
            // Exported slot resolves to a const/param — not representable as
            // an op temp; reject (the caller falls back to the reset path).
            _ => return None,
        }
    }
    let prog = std::sync::Arc::new(ScalarProg {
        ops: dops,
        consts,
        param_count,
        return_ref,
        outer_gids,
        exports,
    });
    Some(prog)
}

/// Builds the loop-CONDITION program for a While sg: the precomputed
/// condition_tree_plan node set (topological) lowered to a scalar program
/// whose return is the condition value. Eligibility (checked here, cheap):
/// While-kind with a plan-based reset, no phi carries, no For-style
/// reset_to_zero/one entries — the tight-loop driver's contract.
pub(crate) fn build_cond_prog(
    graph: &DataFlowGraph,
    sg_id: SubGraphId,
) -> Option<std::sync::Arc<ScalarProg>> {
    build_cond_with_body(graph, sg_id).map(|(c, _)| c)
}

/// Joint condition+body builder for the tight-loop driver. Optimizer-rotated
/// loops wire condition operands to the BODY's computed values (slots defined
/// by body nodes, absent from the condition tree) — those lower to
/// `OpRef::Undef(slot)`. This builder maps each such while-local slot to the
/// body sg's local space, rebuilds the BODY program with those export slots
/// live, and rebuilds the condition with `Body(i)` references. Returns
/// (condition program, export-augmented body program); None keeps the loop
/// on the per-iteration reset path.
pub(crate) fn build_cond_with_body(
    graph: &DataFlowGraph,
    sg_id: SubGraphId,
) -> Option<(std::sync::Arc<ScalarProg>, std::sync::Arc<ScalarProg>)> {
    let sg = &graph.subgraphs[sg_id.0 as usize];
    if sg.loop_kind != crate::ir::Ir::LoopKind::While {
        return None;
    }
    let plan = sg.reset_plan.as_ref();
    let Some(plan) = plan else {
        return None;
    };
    if sg.param_count != 0
        || !plan.reset_to_zero.is_empty()
        || !plan.reset_to_one.is_empty()
        || !plan.carries_value.is_empty()
        || !plan.carries_cell.is_empty()
    {
        return None;
    }
    let mut tree: Vec<NodeId> = plan.condition_tree_plan.iter().map(|&(g, _)| g).collect();
    if tree.is_empty() {
        return None;
    }
    // condition_tree_plan is DFS PREORDER (root first) — the program builder
    // needs inputs before uses. Topologically sort the collected set (cond
    // trees are tiny; O(n²) emission is fine).
    {
        let set: std::collections::HashSet<u32> = tree.iter().map(|g| g.0).collect();
        let mut sorted: Vec<NodeId> = Vec::with_capacity(tree.len());
        let mut emitted: std::collections::HashSet<u32> = std::collections::HashSet::new();
        while sorted.len() < tree.len() {
            let mut progressed = false;
            for idx in (0..tree.len()).rev() {
                let g = tree[idx];
                if emitted.contains(&g.0) {
                    continue;
                }
                let n = graph.node(g.0 as usize);
                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                let ready = inputs
                    .iter()
                    .all(|&inp| !set.contains(&inp.0) || emitted.contains(&inp.0));
                if ready {
                    emitted.insert(g.0);
                    sorted.push(g);
                    progressed = true;
                }
            }
            if !progressed {
                return None; // cyclic (defensive)
            }
        }
        tree = sorted;
    }
    let cond = sg.cond_node?;
    let probe = build_scalar_prog_for(graph, sg_id, &tree, cond)?;
    // Collect distinct Undef slots in first-seen order.
    let mut undef: Vec<u32> = Vec::new();
    {
        let note = |r: &OpRef, undef: &mut Vec<u32>| {
            if let OpRef::Undef(slot) = r {
                if !undef.contains(slot) {
                    undef.push(*slot);
                }
            }
        };
        for op in probe.ops.iter() {
            match op {
                DSop::Scalar { a, b, .. } => { note(a, &mut undef); note(b, &mut undef); }
                DSop::CellAlloc { src } => note(src, &mut undef),
                DSop::DerefWrite { cell, val } => { note(cell, &mut undef); note(val, &mut undef); }
                DSop::DerefRead { cell } => note(cell, &mut undef),
                DSop::RecordConstruct { fields, .. } => {
                    fields.iter().for_each(|f| note(f, &mut undef))
                }
                DSop::FieldGet { src, .. } => note(src, &mut undef),
                DSop::Select { cond, a, b, .. } => {
                    note(cond, &mut undef);
                    note(a, &mut undef);
                    note(b, &mut undef);
                }
                DSop::ArrayIndex { arr, idx, .. } => {
                    note(arr, &mut undef);
                    note(idx, &mut undef);
                }
                DSop::ArrayStore { arr, idx, val, .. } => {
                    note(arr, &mut undef);
                    note(idx, &mut undef);
                    note(val, &mut undef);
                },
            }
        }
        note(&probe.return_ref, &mut undef);
    }
    let body_idx = (0..graph.subgraphs.len()).find(|&i| {
        let b = &graph.subgraphs[i];
        b.loop_kind == crate::ir::Ir::LoopKind::LoopBody
            && b.loop_parent_sg == Some(sg_id)
    })?;
    let bsg = &graph.subgraphs[body_idx];
    let (wns, wne) = (sg.node_range.0 .0, sg.node_range.1 .0);
    let (bns, bne) = (bsg.node_range.0 .0, bsg.node_range.1 .0);
    let bslots = bne - bns;
    let mut body_slots: Vec<u32> = Vec::with_capacity(undef.len());
    for &s in &undef {
        let gid = NodeId(wns + s);
        let bl = gid.0.wrapping_sub(bns);
        if bl >= bslots {
            return None; // outside the body too — unsupported shape
        }
        body_slots.push(bl);
    }
    // Body program with export slots (seeded live, exported as op temps).
    let bplan: Vec<NodeId> = graph.linear_plan(body_idx)?.to_vec();
    if bplan.is_empty() {
        return None;
    }
    let body = build_scalar_prog_for_ex(
        graph,
        SubGraphId(body_idx as u32),
        &bplan,
        bsg.return_node,
        &body_slots,
        &rustc_hash::FxHashMap::default(),
    )?;
    // Condition rebuild: Undef(slot) → Body(i). PLUS: body outer inputs
    // that land INSIDE the while sg (condition-tree nodes — e.g. the loop
    // counter's deref) are exported by the CONDITION program: the tight-loop
    // driver writes them back into the frame slots so the NEXT body
    // iteration's outer re-read sees fresh values (the generic path gets
    // this for free from the per-iteration condition re-evaluation).
    let mut remap = rustc_hash::FxHashMap::default();
    for (i, &s) in undef.iter().enumerate() {
        remap.insert(s, OpRef::Body(i as u32));
    }
    let mut cond_exports: Vec<u32> = Vec::new();
    for &g in body.outer_gids.iter() {
        let local = g.wrapping_sub(wns);
        if local < (wne - wns) {
            cond_exports.push(local);
        }
    }
    let cprog = build_scalar_prog_for_ex(graph, sg_id, &tree, cond, &cond_exports, &remap)?;
    let _ = &plan;
    Some((cprog, body))
}

/// In-range check helper for the sg's own slot space (outers excluded).
fn gid_ok(gid: u32, ns: u32, slot_count: u32) -> Option<u32> {
    let l = gid.wrapping_sub(ns);
    if l < slot_count {
        Some(l)
    } else {
        None
    }
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
    outer_count: usize,
    return_slot: u32,
) -> (Vec<DSop>, Vec<Value>, OpRef) {
    let (d, c, r, _) = lower_to_def_use_ex(
        ops,
        param_count,
        slot_count,
        outer_count,
        return_slot,
        &rustc_hash::FxHashMap::default(),
    );
    (d, c, r)
}

pub fn lower_to_def_use_ex(
    ops: Vec<Sop>,
    param_count: usize,
    slot_count: usize,
    outer_count: usize,
    return_slot: u32,
    undef_remap: &rustc_hash::FxHashMap<u32, OpRef>,
) -> (Vec<DSop>, Vec<Value>, OpRef, Vec<Option<u32>>) {
    #[derive(Clone)]
    enum Binding {
        Op(u32),
        Const(u32),
        Param(u32),
    }
    let mut consts: Vec<Value> = Vec::new();
    let mut dops: Vec<DSop> = Vec::with_capacity(ops.len());
    let total = slot_count + outer_count;
    let mut slot_def: Vec<Option<Binding>> = vec![None; total];
    for i in 0..param_count.min(slot_count) {
        slot_def[i] = Some(Binding::Param(i as u32));
    }
    // Outer pseudo-slots (appended after the sg-local range) pre-bind to the
    // outer-value slice: Param(param_count + i).
    for i in 0..outer_count {
        slot_def[slot_count + i] = Some(Binding::Param((param_count + i) as u32));
    }
    let as_ref = |b: &Option<Binding>, slot: u32| -> OpRef {
        match b {
            Some(Binding::Op(i)) => OpRef::Op(*i),
            Some(Binding::Const(c)) => OpRef::Const(*c),
            Some(Binding::Param(p)) => OpRef::Param(*p),
            // An undefined local slot: NULL at runtime (unseeded-slot
            // semantics of the generic path), unless the caller remapped it
            // (condition programs remap body-defined operands to Body refs).
            None => match undef_remap.get(&slot) {
                Some(r) => r.clone(),
                None => OpRef::Undef(slot),
            },
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
                let ra = as_ref(&slot_def[a as usize], a);
                let rb = as_ref(&slot_def[b as usize], b);
                let idx = dops.len() as u32;
                dops.push(DSop::Scalar { a: ra, b: rb, ty, op, unary });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::CellAlloc { dst, src } => {
                let rsrc = as_ref(&slot_def[src as usize], src);
                let idx = dops.len() as u32;
                dops.push(DSop::CellAlloc { src: rsrc });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::DerefWriteCell { dst, cell, val } => {
                let rcell = as_ref(&slot_def[cell as usize], cell);
                let rval = as_ref(&slot_def[val as usize], val);
                let idx = dops.len() as u32;
                dops.push(DSop::DerefWrite { cell: rcell, val: rval });
                // The write op's result IS the written value.
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::DerefReadCell { dst, cell } => {
                let rcell = as_ref(&slot_def[cell as usize], cell);
                let idx = dops.len() as u32;
                dops.push(DSop::DerefRead { cell: rcell });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::RecordConstruct { dst, shape, srcs } => {
                let fields: Vec<OpRef> = srcs
                    .iter()
                    .map(|s| as_ref(&slot_def[*s as usize], *s))
                    .collect();
                let idx = dops.len() as u32;
                dops.push(DSop::RecordConstruct { fields, shape });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::FieldGet { dst, src, name } => {
                let rsrc = as_ref(&slot_def[src as usize], src);
                let idx = dops.len() as u32;
                dops.push(DSop::FieldGet { src: rsrc, name });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::Select { dst, cond, a, b } => {
                let rc = as_ref(&slot_def[cond as usize], cond);
                let ra = as_ref(&slot_def[a as usize], a);
                let rb = as_ref(&slot_def[b as usize], b);
                let idx = dops.len() as u32;
                dops.push(DSop::Select { cond: rc, a: ra, b: rb });
                slot_def[dst as usize] = Some(Binding::Op(idx));
            }
            Sop::ArrayIndex { dst, arr, idx } => {
                let ra = as_ref(&slot_def[arr as usize], arr);
                let ri = as_ref(&slot_def[idx as usize], idx);
                let k = dops.len() as u32;
                dops.push(DSop::ArrayIndex { arr: ra, idx: ri });
                slot_def[dst as usize] = Some(Binding::Op(k));
            }
            Sop::ArrayStore { dst, arr, idx, val } => {
                let ra = as_ref(&slot_def[arr as usize], arr);
                let ri = as_ref(&slot_def[idx as usize], idx);
                let rv = as_ref(&slot_def[val as usize], val);
                let k = dops.len() as u32;
                dops.push(DSop::ArrayStore { arr: ra, idx: ri, val: rv });
                slot_def[dst as usize] = Some(Binding::Op(k));
            }
        }
    }
    let return_ref = as_ref(&slot_def[return_slot.min(total.saturating_sub(1) as u32) as usize], return_slot);
    let slot_ops: Vec<Option<u32>> = slot_def
        .iter()
        .map(|b| match b {
            Some(Binding::Op(i)) => Some(*i),
            _ => None,
        })
        .collect();
    (dops, consts, return_ref, slot_ops)
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
    optimize_sops_ex(ops, param_count, slot_count, return_slot, &[])
}

pub fn optimize_sops_ex(ops: Vec<Sop>, param_count: usize, slot_count: usize, return_slot: u32, export_slots: &[u32]) -> Vec<Sop> {
    // ── Escape-analysis layer 0: field forwarding ──
    // A FieldGet whose operand resolves (through Seq forwards) to a
    // RecordConstruct IN THIS PROGRAM forwards to the construct's field
    // operand directly — the record never materializes. The pack round-trip
    // (pack_write native width → field(i) Value construction) is
    // value-identical to the operand, and the by-name lookup mirrors
    // record_field_get's position-based find_field. The construct loses its
    // last consumer and the DCE below drops it (allocation + registration
    // vanish — zero-copy layer-0 without frame plumbing).
    let ops = {
        // def map: dst slot → defining op (index), precomputed once.
        let mut def_by_slot: rustc_hash::FxHashMap<u32, usize> =
            rustc_hash::FxHashMap::default();
        for (i, op) in ops.iter().enumerate() {
            let dst = match op {
                Sop::Const { dst, .. }
                | Sop::Void { dst }
                | Sop::CellAlloc { dst, .. }
                | Sop::DerefWriteCell { dst, .. }
                | Sop::DerefReadCell { dst, .. }
                | Sop::Scalar { dst, .. }
                | Sop::Seq { dst, .. }
                | Sop::RecordConstruct { dst, .. }
                | Sop::FieldGet { dst, .. }
                | Sop::Select { dst, .. }
                | Sop::ArrayIndex { dst, .. }
                | Sop::ArrayStore { dst, .. } => *dst,
            };
            def_by_slot.insert(dst, i);
        }
        let resolve_seq = |mut slot: u32, def: &rustc_hash::FxHashMap<u32, usize>, ops: &[Sop]| -> Option<usize> {
            for _ in 0..16 {
                let i = *def.get(&slot)?;
                match &ops[i] {
                    Sop::Seq { src: Some(s), .. } => slot = *s,
                    _ => return Some(i),
                }
            }
            None
        };
        let mut rewritten: Vec<Sop> = ops;
        for wi in 0..rewritten.len() {
            let (dst, src, name) = match &rewritten[wi] {
                Sop::FieldGet { dst, src, name } => (*dst, *src, name.clone()),
                _ => continue,
            };
            let Some(di) = resolve_seq(src, &def_by_slot, &rewritten) else { continue };
            let fields = match &rewritten[di] {
                Sop::RecordConstruct { srcs, shape, .. } => match shape
                    .field_names
                    .iter()
                    .position(|n| n.as_deref() == Some(name.as_str()))
                {
                    Some(j) if j < srcs.len() => srcs[j],
                    _ => continue,
                },
                _ => continue,
            };
            rewritten[wi] = Sop::Seq { dst, src: Some(fields) };
        }
        rewritten
    };
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
            // Field values escape into the record (cells stored as fields
            // stay reachable through it).
            Sop::RecordConstruct { srcs, .. } => (srcs.clone(), false),
            Sop::FieldGet { src, .. } => (vec![*src], false),
            Sop::Select { cond, a, b, .. } => (vec![*cond, *a, *b], false),
            // The array slot is only BORROWED (in-place store through the
            // Arc) — not an escaping read.
            Sop::ArrayIndex { arr, idx, .. } => (vec![*arr, *idx], false),
            Sop::ArrayStore { arr, idx, val, .. } => (vec![*arr, *idx, *val], false),
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
    for &s in export_slots {
        if (s as usize) < slot_count {
            live[s as usize] = true;
        }
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
            | Sop::Seq { dst, .. }
            | Sop::RecordConstruct { dst, .. }
            | Sop::FieldGet { dst, .. }
            | Sop::Select { dst, .. }
            | Sop::ArrayIndex { dst, .. }
            | Sop::ArrayStore { dst, .. } => *dst,
        };
        // Effectful ops survive DCE unconditionally: a post-devirt
        // DerefWriteCell writes a REAL heap cell (an escaping or outer cell
        // — devirtualized ones became Seqs above). Killing a write whose
        // result value nobody reads would silently drop the store (statement
        // writes in loop bodies are exactly this shape). Record construction
        // is VALUE-observable only: a dead record's allocation+registration
        // has no observer (the chain never escapes this program), so dead
        // constructs drop like pure ops — inlined select arms rely on this
        // (the unselected construct chains vanish entirely).
        let effectful =
            matches!(op, Sop::DerefWriteCell { .. } | Sop::ArrayStore { .. });
        if !live[dst as usize] && !effectful {
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
            Sop::RecordConstruct { srcs, .. } => {
                for src in srcs {
                    live[*src as usize] = true;
                }
            }
            Sop::FieldGet { src, .. } => live[*src as usize] = true,
            Sop::Select { cond, a, b, .. } => {
                live[*cond as usize] = true;
                live[*a as usize] = true;
                live[*b as usize] = true;
            }
            Sop::ArrayIndex { arr, idx, .. } => {
                live[*arr as usize] = true;
                live[*idx as usize] = true;
            }
            Sop::ArrayStore { arr, idx, val, .. } => {
                live[*arr as usize] = true;
                live[*idx as usize] = true;
                live[*val as usize] = true;
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
/// order, return reference out. `outers` carries the launch-time values of
/// `prog.outer_gids` (read from the launching frame; empty for leaf calls
/// with no outward references). Borrow discipline: every Op(i) operand is
/// read before the next push (plan order guarantees the producer ran), and
/// arith reads borrow — no Value clones on the compute path.
pub(crate) fn run_scalar_prog(prog: &ScalarProg, args: &[Value], outers: &[Value]) -> Value {
    run_scalar_prog_ex(prog, args, outers, &[], None)
}

/// Extended runner: `body_exports` feeds `OpRef::Body` refs (the tight-loop
/// condition's body-defined operands); `export_out` collects the program's
/// per-launch export values (the body's gift to the condition).
pub(crate) fn run_scalar_prog_ex(
    prog: &ScalarProg,
    args: &[Value],
    outers: &[Value],
    body_exports: &[Value],
    export_out: Option<&mut Vec<Value>>,
) -> Value {
    fn fetch<'a>(
        r: &OpRef,
        temps: &'a [Value],
        consts: &'a [Value],
        args: &'a [Value],
        outers: &'a [Value],
        body_exports: &'a [Value],
        param_count: usize,
    ) -> &'a Value {
        match r {
            OpRef::Op(i) => &temps[*i as usize],
            OpRef::Const(c) => &consts[*c as usize],
            OpRef::Body(i) => body_exports.get(*i as usize).unwrap_or(&Value::NULL),
            // Unseeded slot: read as NULL (generic-path parity).
            OpRef::Undef(_) => &Value::NULL,
            // Param(p): p < param_count → launch arg; beyond → outer slice.
            OpRef::Param(p) => {
                let p = *p as usize;
                if p < param_count {
                    args.get(p).unwrap_or(&Value::NULL)
                } else {
                    outers.get(p - param_count).unwrap_or(&Value::NULL)
                }
            }
        }
    }
    let pc = prog.param_count;
    SCALAR_TEMPS.with(|cell| {
        let mut temps = cell.borrow_mut();
        temps.clear();
        let prog_consts = &prog.consts;
        for op in &prog.ops {
            let v = match op {
                DSop::Scalar { a, b, ty, op, unary } => {
                    let va = fetch(a, &temps, prog_consts, args, outers, body_exports, pc);
                    let vb = fetch(b, &temps, prog_consts, args, outers, body_exports, pc);
                    exec_scalar_op(*ty, *op, *unary, va, vb)
                }
                DSop::CellAlloc { src } => {
                    let init = fetch(src, &temps, prog_consts, args, outers, body_exports, pc);
                    Value::ref_val(crate::value::HeapObj::Cell(crate::value::Cell::new(
                        init.clone(),
                    )))
                }
                DSop::DerefWrite { cell, val } => {
                    let cell_v = fetch(cell, &temps, prog_consts, args, outers, body_exports, pc).clone();
                    let new_val = fetch(val, &temps, prog_consts, args, outers, body_exports, pc).clone();
                    if let Some(crate::value::HeapObj::Cell(c)) = cell_v.heap_obj() {
                        c.set(new_val.clone());
                    }
                    new_val
                }
                DSop::DerefRead { cell } => {
                    let cv = fetch(cell, &temps, prog_consts, args, outers, body_exports, pc);
                    match cv.heap_obj() {
                        Some(crate::value::HeapObj::Cell(c)) => c.get(),
                        _ => cv.clone(),
                    }
                }
                DSop::RecordConstruct { fields, shape } => {
                    // Mirrors compute_record_construct: fields in input
                    // order, written straight into the block tail.
                    let vals: Vec<Value> = fields
                        .iter()
                        .map(|f| fetch(f, &temps, prog_consts, args, outers, body_exports, pc).clone())
                        .collect();
                    Value::Record(crate::value::RecordRef::new_from_iter(
                        shape.clone(),
                        vals,
                    ))
                }
                DSop::Select { cond, a, b } => {
                    let c = fetch(cond, &temps, prog_consts, args, outers, body_exports, pc);
                    let pick = if c.as_bool() { a } else { b };
                    fetch(pick, &temps, prog_consts, args, outers, body_exports, pc).clone()
                }
                DSop::ArrayIndex { arr, idx } => {
                    // Mirrors compute_array_index exactly.
                    let rv = fetch(arr, &temps, prog_consts, args, outers, body_exports, pc);
                    let iv = fetch(idx, &temps, prog_consts, args, outers, body_exports, pc);
                    let idx_raw = iv.as_i32();
                    if idx_raw < 0 {
                        panic!("index {} out of bounds (negative index)", idx_raw);
                    }
                    let i = idx_raw as usize;
                    if let Some(s) = rv.as_str() {
                        s.chars().nth(i).map(Value::char_val).unwrap_or_else(|| {
                            panic!("index {} out of bounds (len {})", i, s.chars().count())
                        })
                    } else {
                        match rv.heap_obj() {
                            Some(crate::value::HeapObj::Array(a)) => {
                                a.get(i).unwrap_or_else(|| {
                                    panic!("index {} out of bounds (len {})", i, a.len())
                                })
                            }
                            _ => panic!("index on non-indexable type"),
                        }
                    }
                }
                DSop::ArrayStore { arr, idx, val } => {
                    let av =
                        fetch(arr, &temps, prog_consts, args, outers, body_exports, pc).clone();
                    let iv = fetch(idx, &temps, prog_consts, args, outers, body_exports, pc);
                    let vv =
                        fetch(val, &temps, prog_consts, args, outers, body_exports, pc).clone();
                    if let Value::Ref(arc) = av {
                        crate::ir::Compute::array_store_inplace(&arc, iv.as_usize(), &vv);
                    }
                    Value::VOID
                }
                DSop::FieldGet { src, name } => {
                    // Mirrors compute_record_field_get exactly: by-name
                    // record lookup, heap-object fallback, FieldError throw.
                    let rv = fetch(src, &temps, prog_consts, args, outers, body_exports, pc);
                    if let Some(v) = rv.record_field_get(name) {
                        v
                    } else {
                        match rv.heap_obj().and_then(|h| h.field_get(name)) {
                            Some(v) => v,
                            None => crate::ir::Compute::make_error_throw(
                                "FieldError",
                                &format!("no such field '{}' on record", name),
                            ),
                        }
                    }
                }
            };
            temps.push(v);
        }
        if let Some(out) = export_out {
            for &(_, op_idx) in &prog.exports {
                if let Some(v) = temps.get(op_idx as usize) {
                    out.push(v.clone());
                }
            }
        }
        fetch(&prog.return_ref, &temps, prog_consts, args, outers, body_exports, pc).clone()
    })
}
