//! Registry — heap-object registry + mark-sweep cycle collector.
//!
//! Frond reclaims heap objects by Arc RAII; a reference CYCLE (arr[0] = arr,
//! two records pointing at each other) never reaches zero and leaks. Every
//! Arc<HeapObj> allocation registers its address here (HeapObj::drop
//! deregisters, so the registry holds exactly the UNRECLAIMED objects —
//! acyclic garbage leaves via normal drops, cyclic garbage stays).
//!
//! `collect_cycles` is safe by construction: it never force-drops an object,
//! it only releases the outgoing reference CLONES of objects proven
//! unreachable from the given roots (mark phase). After all dead sources'
//! edges are released, every cycle member's count falls to zero and normal
//! Rust drops take over (HeapObj::drop deregisters). A marked object losing
//! an incoming clone from a dead source merely decrements — its root-side
//! references keep it alive. Soundness rests entirely on ROOT COMPLETENESS:
//! callers must pass every live Value location (all frame value tables, and
//! any engine-owned async/pending state).

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

use crate::value::{HeapObj, Value};

static REGISTRY: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashSet<usize>> {
    REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Called at every Arc<HeapObj> allocation site (Value::ref_val funnel).
pub fn register(ptr: usize) {
    registry().lock().unwrap().insert(ptr);
}

/// Called from HeapObj::drop — normal (acyclic) reclamation leaves the
/// registry; cyclic garbage stays until a collection.
pub fn deregister(ptr: usize) {
    registry().lock().unwrap().remove(&ptr);
}

/// Number of currently unreclaimed heap objects (alive or leaked-cyclic).
pub fn registered_count() -> usize {
    registry().lock().unwrap().len()
}

/// Visits every Value edge of a heap object (the Arc-object graph). Arena
/// handle fields (Adt fields, Newtype inner, Closure bound_args) are
/// deliberately opaque: their backing arena slots are roots in their own
/// right, so treating them as non-edges cannot strand a live object — it can
/// only keep a dead one until its arena resets (bounded, frame-scoped).
pub fn for_each_child(obj: &HeapObj, f: &mut dyn FnMut(&Value)) {
    use crate::value::HeapObj;
    match obj {
        HeapObj::Array(a) => {
            for e in &a.elements {
                f(e);
            }
        }
        HeapObj::Record(r) => {
            for e in &r.fields {
                f(e);
            }
        }
        HeapObj::Cell(c) => {
            let v = c.get();
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
        // Sender/Receiver mirror their ChannelValue twin; the channel is
        // reachable through them via the shared Arc, but that twin holds no
        // back-reference, so nothing to traverse.
        HeapObj::SenderVal(_) | HeapObj::ReceiverVal(_) => {}
        _ => {}
    }
}

/// Collects cyclic garbage. `roots` must enumerate every live Value.
/// Returns the number of objects whose reclamation was initiated.
pub fn collect_cycles(roots: &[Value]) -> usize {
    if std::env::var("FROND_NO_CYCLES").is_ok() {
        return 0;
    }
    let mut reg = registry().lock().unwrap();
    if reg.is_empty() {
        return 0;
    }
    // Mark from the roots. Work list carries raw object ptrs; the mark loop
    // re-derives &HeapObj (SAFETY: quiescent stop-the-world point; every
    // pushed object is held alive by the edge/root that pushed it).
    let mut marked: HashSet<usize> = HashSet::new();
    let mut work: Vec<usize> = Vec::new();
    for v in roots {
        if let Value::Ref(a) = v {
            work.push(Arc::as_ptr(a) as usize);
        }
    }
    while let Some(p) = work.pop() {
        if !marked.insert(p) {
            continue;
        }
        let obj = unsafe { &*(p as *const HeapObj) };
        // Cell/Lazy children are clones (temporaries); extracting the inner
        // Arc ptr inside the callback is still correct — only the ptr matters.
        for_each_child(obj, &mut |v: &Value| {
            if let Value::Ref(a) = v {
                work.push(Arc::as_ptr(a) as usize);
            }
        });
    }
    // Sweep: dead = registered but unmarked.
    let dead: Vec<usize> = reg.difference(&marked).copied().collect();
    for p in &dead {
        reg.remove(p);
    }
    drop(reg);
    // Snapshot each dead source's outgoing edges BEFORE any release.
    let mut edge_list: Vec<Vec<usize>> = Vec::with_capacity(dead.len());
    for p in &dead {
        // SAFETY: alive via its own cycle edges until phase 2 below.
        let obj = unsafe { &*(*p as *const HeapObj) };
        let mut kids: Vec<usize> = Vec::new();
        for_each_child(obj, &mut |v: &Value| {
            if let Value::Ref(a) = v {
                kids.push(Arc::as_ptr(a) as usize);
            }
        });
        edge_list.push(kids);
    }
    // Phase 1: retake EVERY dead-source edge clone and hold it (each from_raw
    // +1s the child count), so no object can hit zero mid-iteration — a later
    // edge list may still carry the raw ptr of an earlier-released child.
    let mut held: Vec<Arc<HeapObj>> = Vec::new();
    for kids in &edge_list {
        for q in kids {
            // SAFETY: recreates the child Arc clone the dead source holds.
            held.push(unsafe { Arc::from_raw(*q as *const HeapObj) });
        }
    }
    // Phase 2: release all at once — cycle members fall to zero through
    // normal Rust drops; HeapObj::drop deregisters (no-op miss: removed above).
    drop(held);
    dead.len()
}
