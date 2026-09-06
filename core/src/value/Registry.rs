//! Registry — heap-object registry + mark-sweep cycle collector (mixed heap).
//!
//! Frond reclaims heap objects by refcount RAII (Arc for HeapObj kinds,
//! RecordRef for single-block records); a reference CYCLE never reaches zero
//! and leaks. Every potentially-cyclic allocation registers its address here
//! (drop deregisters). Record blocks register with the low bit SET to
//! disambiguate them from HeapObj pointers in the shared registry — the
//! collector walks a mixed heap.
//!
//! `collect_cycles` never force-drops an object: it cushions every dead
//! source (+1 clone), releases exactly the outgoing edges of objects proven
//! unreachable, then drops the cushion — cycle members fall to zero through
//! normal drops. Soundness rests on ROOT COMPLETENESS (see Schedule.rs call
//! site) and on the quiescent stop-the-world point.

use std::sync::{Arc, Mutex, OnceLock};

use rustc_hash::FxHashSet;

use crate::value::{HeapObj, Value};

/// A HeapObj can only join a cycle if it holds `Value` edges. Childless kinds
/// skip registration entirely. (Records register unconditionally through
/// `register_record` — they always hold fields.)
pub fn can_cycle(obj: &HeapObj) -> bool {
    !matches!(
        obj,
        HeapObj::Range(_)
            | HeapObj::OpaquePtr(_)
            | HeapObj::LibVal(_)
            | HeapObj::ForeignFnVal(_)
            | HeapObj::GlobalSlotRef { .. }
    )
}

static REGISTRY: OnceLock<Mutex<FxHashSet<usize>>> = OnceLock::new();

fn registry() -> &'static Mutex<FxHashSet<usize>> {
    REGISTRY.get_or_init(|| Mutex::new(FxHashSet::default()))
}

/// Low tag distinguishing record blocks from HeapObj pointers in the mixed
/// registry (RecordBlock is 8-aligned; HeapObj allocations likewise).
pub const RECORD_TAG: usize = 1;

/// Called at every Arc<HeapObj> allocation site (Value::ref_val /
/// register_arc funnels; the funnels gate on `can_cycle`).
pub fn register(ptr: usize) {
    registry().lock().unwrap().insert(ptr);
}

/// Record-block registration is intrusive (Value.rs list); kept for API
/// compatibility — no hash-set work.
pub fn register_record(ptr: usize) {
    let _ = ptr;
}

/// Called from HeapObj::drop (absent pointers are a no-op miss).
pub fn deregister(ptr: usize) {
    registry().lock().unwrap().remove(&ptr);
}

/// Record-block deregistration is intrusive (Value.rs list).
pub fn deregister_record(ptr: usize) {
    let _ = ptr;
}

/// Number of currently unreclaimed heap objects (alive or leaked-cyclic).
pub fn registered_count() -> usize {
    registry().lock().unwrap().len() + crate::value::record_list_count()
}

/// Visits every Value edge of a HeapObj. Records never enter here — their
/// edges are walked by `crate::value::record_walk_edges` from the Value
/// level.
pub fn for_each_child(obj: &HeapObj, f: &mut dyn FnMut(&Value)) {
    use crate::value::HeapObj;
    match obj {
        HeapObj::Array(a) => {
            for e in &a.elements {
                f(e);
            }
        }
        HeapObj::Cell(c) => {
            let v = c.get();
            f(&v);
        }
        HeapObj::AtomicVal(a) => {
            let v = a.load();
            f(&v);
        }
        HeapObj::Closure(cl) => {
            for u in &cl.upvalues {
                f(u);
            }
        }
        HeapObj::Partial(p) => {
            for u in &p.upvalues {
                f(u);
            }
            for b in &p.bound_args {
                f(b);
            }
        }
        HeapObj::TraitVal(t) => {
            for m in &t.method_values {
                f(m);
            }
            if let Some(d) = &t.data {
                f(d);
            }
        }
        HeapObj::LazyVal(l) => {
            let g = l.cached.lock().unwrap();
            if let Some(v) = g.as_ref() {
                f(v);
            }
        }
        HeapObj::ThrowVal(t) => match &t.payload {
            crate::value::ThrowPayload::Ok(v) | crate::value::ThrowPayload::Err(v) => f(v),
        },
        HeapObj::ArrayElemRef { arr, .. } => f(arr),
        HeapObj::RecordFieldRef { rec, .. } => f(rec),
        HeapObj::ChannelVal(c) => c.each_buffered(f),
        // Sender/Receiver traverse their shared ChannelValue twin's BUFFER.
        HeapObj::SenderVal(s) => s.channel.each_buffered(f),
        HeapObj::ReceiverVal(r) => r.channel.each_buffered(f),
        _ => {}
    }
}

/// Pushes every heap edge of a Value into `work` (tagged record ptrs for
/// Value::Record, plain HeapObj ptrs for Value::Ref). Str/Scalar are leaves.
fn push_value_edges(v: &Value, work: &mut Vec<usize>) {
    match v {
        Value::Ref(a) => work.push(Arc::as_ptr(a) as usize),
        Value::Record(r) => work.push(crate::value::record_tagged_ptr(r) | RECORD_TAG),
        _ => {}
    }
}

/// Collects cyclic garbage. `roots` must enumerate every live Value.
/// Returns the number of objects whose reclamation was initiated.
pub fn collect_cycles(roots: &[Value]) -> usize {
    let reg = registry().lock().unwrap();
    let record_total = crate::value::record_list_count();
    if reg.is_empty() && record_total == 0 {
        return 0;
    }
    let mut marked: FxHashSet<usize> = FxHashSet::default();
    let mut work: Vec<usize> = Vec::new();
    for v in roots {
        push_value_edges(v, &mut work);
    }
    while let Some(p) = work.pop() {
        if !marked.insert(p) {
            continue;
        }
        if p & RECORD_TAG != 0 {
            // SAFETY: quiescent stop-the-world; every pushed record is held
            // alive by the edge/root that pushed it.
            unsafe { crate::value::record_walk_tagged(p, &mut |v: &Value| push_value_edges(v, &mut work)) };
        } else {
            // SAFETY: same argument as above for HeapObj pointers.
            let obj = unsafe { &*(p as *const HeapObj) };
            for_each_child(obj, &mut |v: &Value| push_value_edges(v, &mut work));
        }
    }
    // Sweep: dead HeapObjs = hash-set entries not marked; dead records =
    // intrusive-list members not marked (tagged pointers).
    let mut dead: Vec<usize> = reg.difference(&marked).copied().collect();
    drop(reg);
    // SAFETY: stop-the-world quiescent point (the valve runs between frames;
    // no concurrent thread/unthread — see record_list_walk).
    unsafe {
        crate::value::record_list_walk(&mut |tagged| {
            if !marked.contains(&tagged) {
                dead.push(tagged);
            }
        });
    }
    let reg_clear: Vec<usize> = dead.iter().copied().filter(|p| p & RECORD_TAG == 0).collect();
    { let mut reg = registry().lock().unwrap(); for p in &reg_clear { reg.remove(p); } }
    // Phase 1: cushion EVERY dead source (+1) — record blocks get a borrowed
    // RecordRef, HeapObj sources get their outgoing edges cloned. This
    // guarantees no cascade free can happen while releases below are running
    // (the classic double-own hazard the old from_raw scheme hit).
    let mut held: Vec<Arc<HeapObj>> = Vec::new();
    let mut held_records: Vec<crate::value::RecordRef> = Vec::new();
    for &p in &dead {
        if p & RECORD_TAG != 0 {
            held_records.push(unsafe { crate::value::record_cushion_tagged(p) });
        } else {
            // SAFETY: alive — referenced by its own/other edges.
            let obj = unsafe { &*(p as *const HeapObj) };
            for_each_child(obj, &mut |v: &Value| {
                if let Value::Ref(a) = v {
                    held.push(a.clone());
                }
            });
        }
    }
    // Phase 2: release each dead source's OWN outgoing edges.
    for &p in &dead {
        if p & RECORD_TAG != 0 {
            // Drops the inline tail in place (children decref); the block
            // itself stays alive under the cushion.
            unsafe { crate::value::record_release_edges_tagged(p) };
        } else {
            // SAFETY: stop-the-world quiescent point; cushion keeps children
            // alive through this pass.
            let old = unsafe {
                std::ptr::replace(
                    p as *mut HeapObj,
                    HeapObj::Range(crate::value::Range::new(0, 0, false)),
                )
            };
            drop(old);
        }
    }
    // Phase 3: drop the cushions — net one release per dead-source edge;
    // cycle members fall to zero through normal drops.
    drop(held);
    drop(held_records);
    dead.len()
}
