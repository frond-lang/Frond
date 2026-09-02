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
/// handle fields (Newtype inner, Closure bound_args) are deliberately
/// opaque: their backing arena slots are roots in their own right, so
/// treating them as non-edges cannot strand a live object — it can only keep
/// a dead one until its arena resets (bounded, frame-scoped). Adt/Record
/// fields are inline Values and MUST be walked: an Adt reached through a
/// root whose fields stay unmarked gets its live children swept (observed:
/// 5万-entry IntMap judged dead at the pressure valve → UAF at teardown).
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
        HeapObj::Adt(a) => {
            for fld in &a.fields {
                f(&fld.value);
            }
        }
        HeapObj::Cell(c) => {
            let v = c.get();
            f(&v);
        }
        HeapObj::Newtype(n) => f(&n.inner),
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
        // Sender/Receiver must traverse their shared ChannelValue twin's
        // BUFFER: the twin is a separate allocation, so when a sender or
        // receiver is the only live path to the channel, skipping the buffer
        // left every buffered message unmarked → falsely swept mid-run.
        HeapObj::SenderVal(s) => s.channel.each_buffered(f),
        HeapObj::ReceiverVal(r) => r.channel.each_buffered(f),
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
    let trace = std::env::var("FROND_TRACE_CYCLES").is_ok();
    if trace {
        eprintln!(
            "[cycles] roots={} registered={} marked={} dead={}",
            roots.len(),
            reg.len(),
            marked.len(),
            dead.len()
        );
    }
    for p in &dead {
        reg.remove(p);
    }
    if trace {
        for p in &dead {
            let obj = unsafe { &*(*p as *const HeapObj) };
            let desc = match obj {
                HeapObj::Str(s) => format!("Str({s})"),
                other => format!("{other:?}").chars().take(48).collect::<String>(),
            };
            eprintln!("[cycles] dead {p:p} = {desc}");
        }
    }
    drop(reg);
    // Phase 1: CLONE every dead-source outgoing edge (a real Arc::clone,
    // +1 each). This cushion guarantees no child's count can reach zero
    // while the sources below are still releasing their own edges. (The
    // previous from_raw scheme was unsound: from_raw ADOPTS an existing
    // count instead of adding one, so the held "clones" and the sources'
    // own fields double-owned every edge — dropping both sides released
    // each count twice and detonated on any real cycle once Adt fields
    // became visible to the walk.)
    let mut held: Vec<Arc<HeapObj>> = Vec::new();
    for p in &dead {
        // SAFETY: alive — still referenced by its own/other edges, plus the
        // registry snapshot happened before any release.
        let obj = unsafe { &*(*p as *const HeapObj) };
        for_each_child(obj, &mut |v: &Value| {
            if let Value::Ref(a) = v {
                held.push(a.clone());
            }
        });
    }
    if trace {
        eprintln!("[cycles] cushion built, edges={}", held.len());
    }
    // Phase 2: release each dead source's OWN outgoing edges by replacing
    // the object in place with a benign value — dropping the old value runs
    // its normal Drop, releasing exactly the field/array Arcs it owns.
    // HeapObj::drop's deregister is a no-op miss (entries removed above).
    for p in &dead {
        // SAFETY: stop-the-world quiescent point; the object is garbage and
        // the cushion keeps every child of it alive through this pass.
        let old = unsafe {
            std::ptr::replace(
                *p as *mut HeapObj,
                HeapObj::Range(crate::value::Range::new(0, 0, false)),
            )
        };
        drop(old);
    }
    // Phase 3: drop the cushion — net effect is one release per dead-source
    // edge; cycle members fall to zero through normal Rust drops (the final
    // Arc drop re-runs HeapObj::drop on the benign replacement).
    drop(held);
    if trace {
        eprintln!("[cycles] release done");
    }
    dead.len()
}
