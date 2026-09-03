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

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use rustc_hash::FxHashSet;

use crate::value::{HeapObj, Value};

// Cycle collection is fully bypassed under FROND_NO_CYCLES: allocation-side
// registration is skipped as well.
static CYCLES_DISABLED: OnceLock<bool> = OnceLock::new();

fn cycles_disabled() -> bool {
    *CYCLES_DISABLED.get_or_init(|| std::env::var("FROND_NO_CYCLES").is_ok())
}

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
    // TEMP-PROBE (mem baseline)
    if probe::enabled() {
        let n = probe::REGISTER_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let live = n - probe::DEREGISTER_CALLS.load(Ordering::Relaxed);
        probe::track_live(live);
    }
    if cycles_disabled() {
        return;
    }
    registry().lock().unwrap().insert(ptr);
}

/// Whether record blocks should thread into the intrusive registry list
/// (乙①: the list lives in Value.rs; FROND_NO_CYCLES skips the threading).
pub fn record_registration_enabled() -> bool {
    // TEMP-PROBE (mem baseline)
    if probe::enabled() {
        let n = probe::REGISTER_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        let live = n - probe::DEREGISTER_CALLS.load(Ordering::Relaxed);
        probe::track_live(live);
    }
    !cycles_disabled()
}

/// Record-block registration is intrusive (Value.rs list); kept for API
/// compatibility — no hash-set work.
pub fn register_record(ptr: usize) {
    let _ = ptr;
}

/// Called from HeapObj::drop (absent pointers are a no-op miss).
pub fn deregister(ptr: usize) {
    // TEMP-PROBE (mem baseline)
    if probe::enabled() {
        probe::DEREGISTER_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    if cycles_disabled() {
        return;
    }
    registry().lock().unwrap().remove(&ptr);
}

/// Record-block deregistration is intrusive (Value.rs list).
pub fn deregister_record(ptr: usize) {
    let _ = ptr;
}

/// Number of currently unreclaimed heap objects (alive or leaked-cyclic).
pub fn registered_count() -> usize {
    // TEMP-PROBE (mem baseline): the pressure valve calls this per frame completion.
    if probe::enabled() {
        probe::COUNT_CHECKS.fetch_add(1, Ordering::Relaxed);
    }
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
    if std::env::var("FROND_NO_CYCLES").is_ok() {
        return 0;
    }
    let mut reg = registry().lock().unwrap();
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
    let trace = std::env::var("FROND_TRACE_CYCLES").is_ok();
    if trace {
        eprintln!(
            "[cycles] roots={} registered={} marked={} dead={}",
            roots.len(),
            registry().lock().unwrap().len() + record_total,
            marked.len(),
            dead.len()
        );
    }
    // TEMP-PROBE (mem baseline)
    if probe::enabled() {
        probe::COLLECT_FIRES.fetch_add(1, Ordering::Relaxed);
        probe::COLLECT_ROOTS.fetch_add(roots.len() as u64, Ordering::Relaxed);
        probe::COLLECT_RECLAIMED.fetch_add(dead.len() as u64, Ordering::Relaxed);
    }
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
    if trace {
        eprintln!("[cycles] release done");
    }
    dead.len()
}

// TEMP-PROBE (mem baseline) — env-gated allocation/registry counters.
// Enabled by FROND_MEM_PROBE=1; report printed via `probe::report()` at CLI exit.
pub mod probe {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static INIT: AtomicBool = AtomicBool::new(false);

    pub static REGISTER_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static DEREGISTER_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static COUNT_CHECKS: AtomicU64 = AtomicU64::new(0);
    pub static COLLECT_FIRES: AtomicU64 = AtomicU64::new(0);
    pub static COLLECT_ROOTS: AtomicU64 = AtomicU64::new(0);
    pub static COLLECT_RECLAIMED: AtomicU64 = AtomicU64::new(0);
    pub static PEAK_REGISTERED: AtomicU64 = AtomicU64::new(0);

    pub const KIND_NAMES: [&str; 22] = [
        "Array", "Cell", "Range", "Closure", "Partial",
        "Builtin", "TraitVal", "LazyVal", "ErrorVal", "ThrowVal", "ArrayElemRef",
        "RecordFieldRef", "GlobalSlotRef", "AtomicVal", "AsyncVal", "ChannelVal", "SenderVal",
        "ReceiverVal", "CoroutineFrame", "OpaquePtr", "LibVal", "ForeignFnVal",
    ];
    pub static KIND_ALLOC_COUNTS: [AtomicU64; 22] = {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: AtomicU64 = AtomicU64::new(0);
        [ZERO; 22]
    };

    #[inline]
    pub fn enabled() -> bool {
        if !INIT.load(Ordering::Relaxed) {
            ENABLED.store(std::env::var("FROND_MEM_PROBE").is_ok(), Ordering::Relaxed);
            INIT.store(true, Ordering::Relaxed);
        }
        ENABLED.load(Ordering::Relaxed)
    }

    pub fn count_kind(kind: usize) {
        if enabled() {
            if kind < 22 {
                KIND_ALLOC_COUNTS[kind].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn track_live(live: u64) {
        if enabled() {
            PEAK_REGISTERED.fetch_max(live, Ordering::Relaxed);
        }
    }

    pub fn report() {
        if !enabled() {
            return;
        }
        eprintln!("== MEM-PROBE ==");
        eprintln!(
            "process: peak_working_set={:.1} MB commit={:.1} MB",
            peak_working_set() as f64 / (1024.0 * 1024.0),
            peak_commit() as f64 / (1024.0 * 1024.0),
        );
        eprintln!(
            "registry: register={} deregister={} peak_live={} count_checks={} fires={} roots_total={} reclaimed_total={}",
            REGISTER_CALLS.load(Ordering::Relaxed),
            DEREGISTER_CALLS.load(Ordering::Relaxed),
            PEAK_REGISTERED.load(Ordering::Relaxed),
            COUNT_CHECKS.load(Ordering::Relaxed),
            COLLECT_FIRES.load(Ordering::Relaxed),
            COLLECT_ROOTS.load(Ordering::Relaxed),
            COLLECT_RECLAIMED.load(Ordering::Relaxed),
        );
        let total: u64 = KIND_ALLOC_COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        eprintln!("heap allocs by kind (total {}):", total);
        let mut idx: Vec<usize> = (0..22).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(KIND_ALLOC_COUNTS[i].load(Ordering::Relaxed)));
        for i in idx {
            let c = KIND_ALLOC_COUNTS[i].load(Ordering::Relaxed);
            if c > 0 {
                let pct = (c as f64 / total as f64 * 1000.0).round() / 10.0;
                eprintln!("  {:<16} {:>12}  {:>5}%", KIND_NAMES[i], c, pct);
            }
        }
    }

    // TEMP-PROBE (mem baseline): OS-recorded process memory high-water marks.
    #[cfg(windows)]
    mod winmem {
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_paged_pool_usage: usize,
            quota_paged_pool_usage: usize,
            quota_peak_non_paged_pool_usage: usize,
            quota_non_paged_pool_usage: usize,
            pagefile_usage: usize,
            peak_pagefile_usage: usize,
        }
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
        }
        #[link(name = "psapi")]
        extern "system" {
            fn GetProcessMemoryInfo(
                process: isize,
                ppmemcounters: *mut ProcessMemoryCounters,
                cb: u32,
            ) -> i32;
        }
        fn with_counters<R>(f: impl FnOnce(&ProcessMemoryCounters) -> R) -> Option<R> {
            let mut pmc = ProcessMemoryCounters {
                cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
                page_fault_count: 0,
                peak_working_set_size: 0,
                working_set_size: 0,
                quota_peak_paged_pool_usage: 0,
                quota_paged_pool_usage: 0,
                quota_peak_non_paged_pool_usage: 0,
                quota_non_paged_pool_usage: 0,
                pagefile_usage: 0,
                peak_pagefile_usage: 0,
            };
            let ok = unsafe {
                GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb)
            };
            if ok == 0 { None } else { Some(f(&pmc)) }
        }
        pub fn peak_working_set() -> usize {
            with_counters(|p| p.peak_working_set_size).unwrap_or(0)
        }
        pub fn peak_commit() -> usize {
            with_counters(|p| p.peak_pagefile_usage).unwrap_or(0)
        }
    }
    #[cfg(windows)]
    pub use winmem::{peak_commit, peak_working_set};
    #[cfg(not(windows))]
    pub fn peak_working_set() -> usize { 0 }
    #[cfg(not(windows))]
    pub fn peak_commit() -> usize { 0 }
}
