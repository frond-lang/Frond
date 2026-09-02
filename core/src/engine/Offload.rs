//! Offload — L2: pure-subgraph compute offload for the deterministic event
//! loop (FROND_OFFLOAD=1, gray-release).
//!
//! Model: the engine thread owns ALL mutable state and scheduling, always.
//! A synchronous call whose subgraph is classified pure
//! (`is_offload_safe_compute` over every node — see ir/Ir.rs) and heavy
//! enough (node-count threshold, FROND_OFFLOAD_MIN) executes on a rayon
//! worker instead:
//!
//!   1. LAUNCH (engine thread): arguments pass a reachability gate (no
//!      Closure/Partial — their arena-handle bound_args would dangle with
//!      the scratch arena) and are DEEP-CLONED into a private copy (copy-in).
//!      The child frame is built from the clones and shipped with a launch
//!      sequence number; the caller suspends exactly like a queue-path sync
//!      call. The deep clone is the whole soundness story: the worker shares
//!      NOTHING with the engine's live object graph, so the engine may keep
//!      running and mutating its own objects while workers compute — no
//!      aliasing analysis, no value-layer locks.
//!
//!   2. EXECUTE (worker): a lean executor drains the subgraph's linear plan.
//!      Whitelisted compute functions touch only the frame, the graph, and
//!      freshly allocated values. Any unexpected node result or a missing
//!      local input (a classifier escape) makes the worker bail with a
//!      FALLBACK marker: the engine then re-drives the child through the
//!      real machinery. Correctness never depends on the classifier being
//!      perfect — only performance does.
//!
//!   3. DELIVER (engine thread): completions are buffered by sequence number
//!      and applied strictly in LAUNCH ORDER (the sequencer), so execution
//!      stays a pure function of the program regardless of worker timing —
//!      parallel compute, deterministic semantics. Delivery feeds the caller
//!      via `pending_completions` + an idempotent queue push (the proven
//!      Bug-#78 drain path in process_frame).

use super::*;
use crate::ir::Ir::{self as ir, Frame, FrameId, NodeId, PendingCall, EvalContext, NodeResult, ControlSignal, LoopKind, RuntimeEvent, SuspendState, FrameState, SubGraphId, DataFlowGraph};
use crate::value::Value;
use parking_lot::{Condvar, Mutex as ParkingMutex};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Launch threshold below which offload is pointless (copy + delivery
/// overhead exceeds the compute). Tunable via FROND_OFFLOAD_MIN.
pub(super) const DEFAULT_OFFLOAD_MIN_NODES: usize = 2000;
// Threshold economics (measured): launch+copy+delivery overhead is ~0.5ms
// per offload; a pure straight-line leaf averages ~1µs/node, so a node-count
// gate must sit where work clearly dominates. 2000 nodes ≈ ≥1ms of compute.
// A real cost model (and loop-level / slot-provenance extensions) is the L2
// follow-up; until then this keeps the machinery sound and dormant by
// default, exercised via FROND_OFFLOAD=1 soak.

/// One completed offload, waiting for its turn in the sequencer.
struct OffloadDone {
    caller: FrameId,
    call_node: NodeId,
    value: Value,
    signal: ControlSignal,
    /// The executed child, shipped back for ENGINE-THREAD pool return.
    /// Without it the pool drains and every launch pays a fresh cold value
    /// table — the 30x slowdown root cause.
    child_frame: Option<Box<Frame>>,
    /// Classifier escape: the engine re-drives this child through the real
    /// machinery instead of applying the partial result.
    fallback_child: Option<Box<Frame>>,
    /// Restitution: the (cloned-arg) result aliased an argument object —
    /// reference semantics demand re-execution with the ORIGINAL args on
    /// the engine thread; carrying them here avoids stashing them in the
    /// engine between launch and delivery.
    fallback_args: Option<Vec<Value>>,
}

/// Shared offload runtime: held via Arc by the engine and every in-flight
/// worker. The ONLY cross-thread state in the process, fully mutex-guarded.
pub(super) struct OffloadRt {
    pub enabled: bool,
    pub min_nodes: usize,
    inner: ParkingMutex<OffloadInner>,
    /// Engine-thread wakeup: notified on every delivery.
    pub parked: Condvar,
    /// Profiling accumulators (FROND_OFFLOAD_STATS dump at exit).
    pub stats: OffloadStats,
}

/// Nanosecond buckets: engine-side launch cost (frame build + clone),
/// worker execution, and delivery apply.
#[derive(Default)]
pub struct OffloadStats {
    pub launches: std::sync::atomic::AtomicU64,
    pub launch_ns: std::sync::atomic::AtomicU64,
    pub worker_ns: std::sync::atomic::AtomicU64,
    pub deliver_ns: std::sync::atomic::AtomicU64,
}

impl OffloadStats {
    fn add(&self, slot: &std::sync::atomic::AtomicU64, d: std::time::Duration) {
        slot.fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for OffloadStats {
    fn drop(&mut self) {
        if std::env::var("FROND_OFFLOAD_STATS").is_ok() {
            let l = self.launches.load(std::sync::atomic::Ordering::Relaxed);
            let ln = self.launch_ns.load(std::sync::atomic::Ordering::Relaxed);
            let wn = self.worker_ns.load(std::sync::atomic::Ordering::Relaxed);
            let dn = self.deliver_ns.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "[OFFLOAD-STATS] launches={l} launch(engine)={ln}ns ({}/call) worker={wn}ns ({}/call) deliver={dn}ns ({}/call)",
                if l > 0 { ln / l } else { 0 },
                if l > 0 { wn / l } else { 0 },
                if l > 0 { dn / l } else { 0 },
            );
        }
    }
}

// Safety: the ONLY cross-thread component in the process. Every field is
// accessed under `inner`'s mutex; a delivered OffloadDone may carry a
// Box<Frame> whose chain raw pointers are NULL (launch contract) and whose
// table holds Arc'd values (Value: Send) — the box is moved across threads
// but never dereferenced except on the engine thread.
unsafe impl Send for OffloadRt {}
unsafe impl Sync for OffloadRt {}

struct OffloadInner {
    next_launch_seq: u64,
    next_apply_seq: u64,
    inflight: usize,
    /// Reorder buffer keyed by launch seq — completions may arrive out of
    /// order; applied strictly in seq order.
    ready: HashMap<u64, OffloadDone>,
}

impl OffloadRt {
    pub fn from_env() -> Arc<Self> {
        Arc::new(Self {
            enabled: std::env::var("FROND_OFFLOAD").is_ok(),
            min_nodes: std::env::var("FROND_OFFLOAD_MIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(DEFAULT_OFFLOAD_MIN_NODES),
            inner: ParkingMutex::new(OffloadInner {
                next_launch_seq: 0,
                next_apply_seq: 0,
                inflight: 0,
                ready: HashMap::new(),
            }),
            parked: Condvar::new(),
            stats: OffloadStats::default(),
        })
    }

    fn launch_seq(&self) -> u64 {
        let mut g = self.inner.lock();
        let seq = g.next_launch_seq;
        g.next_launch_seq += 1;
        g.inflight += 1;
        seq
    }

    fn deliver(&self, seq: u64, done: OffloadDone) {
        {
            let mut g = self.inner.lock();
            g.ready.insert(seq, done);
            g.inflight -= 1;
        }
        self.parked.notify_all();
    }

    /// Engine thread: apply completions that reached the head of the launch
    /// sequence. The callback runs WITHOUT the inner lock held (it touches
    /// engine state; OffloadRt locks are leaf-only).
    fn drain_apply(&self, f: &mut dyn FnMut(OffloadDone)) -> usize {
        let mut g = self.inner.lock();
        let mut n = 0;
        loop {
            let seq = g.next_apply_seq;
            let Some(done) = g.ready.remove(&seq) else { break; };
            g.next_apply_seq += 1;
            n += 1;
            drop(g);
            f(done);
            g = self.inner.lock();
        }
        n
    }

    fn inflight(&self) -> usize {
        self.inner.lock().inflight
    }

    /// Deadlock-dump view: sequencer position + stuck reorder-buffer keys.
    pub fn debug_state(&self) -> String {
        let g = self.inner.lock();
        let mut keys: Vec<u64> = g.ready.keys().copied().collect();
        keys.sort_unstable();
        format!(
            "launch_seq={} apply_seq={} inflight={} buffered={:?}",
            g.next_launch_seq, g.next_apply_seq, g.inflight, keys
        )
    }

    /// Is the sequencer HEAD buffered (an applyable completion waiting)?
    /// The idle branch must re-check this AFTER the inflight check: a worker
    /// can deliver between the two reads, and the loop-top apply is the only
    /// consumer — declaring deadlock with an applicable head stranded whole
    /// sync chains (the L2 chained-loss root cause).
    pub fn head_ready(&self) -> bool {
        let g = self.inner.lock();
        g.ready.contains_key(&g.next_apply_seq)
    }

    /// Engine-thread park while offloads are in flight: workers notify on
    /// delivery; `limit` (a timer deadline) bounds the wait.
    pub fn park(&self, limit: Option<std::time::Instant>) {
        let mut g = self.inner.lock();
        match limit {
            Some(deadline) => {
                let dur = deadline.saturating_duration_since(std::time::Instant::now());
                if !dur.is_zero() {
                    self.parked.wait_for(&mut g, dur);
                }
            }
            None => self.parked.wait(&mut g),
        }
    }
}

// =========================================================================
// Worker-side executor
// =========================================================================

pub(super) enum OffloadOutcome {
    Done(Value, ControlSignal),
    Fallback,
}

/// Lean executor for a pure leaf subgraph: drains the sg's linear plan
/// (topological order, precomputed at graph load) with none of the engine's
/// queue/gate machinery — the whitelist guarantees none of it is reachable.
/// Reads only the frame's own table; a missing input (an outer-scope read
/// the classifier let through) is a Fallback, never a panic.
fn diag_flag(name: &'static str) -> bool {
    static FLAGS: std::sync::OnceLock<rustc_hash::FxHashMap<&'static str, bool>> =
        std::sync::OnceLock::new();
    *FLAGS
        .get_or_init(|| {
            let mut m = rustc_hash::FxHashMap::default();
            for k in ["FROND_OFFLOAD_NOCHK", "FROND_OFFLOAD_NOCOMPUTE", "FROND_OFFLOAD_NOSTORE"] {
                m.insert(k, std::env::var(k).is_ok());
            }
            m
        })
        .get(name)
        .unwrap_or(&false)
}


// =========================================================================
// Copy-in safety gate
// =========================================================================

/// Args reachable-graph gate: Closure/Partial carry arena-handle bound_args
/// that would dangle with the deep-clone scratch arena — reject them. The
/// walk is ptr-keyed (cycles tolerated); everything else clones safely.
fn offload_args_ok(args: &[Value]) -> bool {
    let mut seen: HashSet<usize> = HashSet::new();
    args.iter().all(|v| value_offload_safe(v, &mut seen))
}

fn value_offload_safe(v: &Value, seen: &mut HashSet<usize>) -> bool {
    let Value::Ref(rc) = v else { return true };
    let key = Arc::as_ptr(rc) as usize;
    if !seen.insert(key) {
        return true; // already visited (shared/cyclic): verdict stands
    }
    use crate::value::HeapObj;
    match rc.as_ref() {
        HeapObj::Closure(_) | HeapObj::Partial(_) => false,
        HeapObj::Array(a) => a.elements.iter().all(|e| value_offload_safe(e, seen)),
        HeapObj::Record(r) => r.fields.iter().all(|e| value_offload_safe(e, seen)),
        HeapObj::Adt(a) => a.fields.iter().all(|f| value_offload_safe(&f.value, seen)),
        HeapObj::Newtype(n) => value_offload_safe(&n.inner, seen),
        HeapObj::ThrowVal(t) => match &t.payload {
            crate::value::ThrowPayload::Ok(v) | crate::value::ThrowPayload::Err(v) => {
                value_offload_safe(v, seen)
            }
        },
        HeapObj::Cell(c) => value_offload_safe(&c.get(), seen),
        _ => true,
    }
}

/// Collects every heap-object address reachable from `v` (cycles tolerated).
fn collect_ref_ptrs(v: &Value, seen: &mut HashSet<usize>) {
    let Value::Ref(rc) = v else { return };
    let key = std::sync::Arc::as_ptr(rc) as usize;
    if !seen.insert(key) {
        return;
    }
    use crate::value::HeapObj;
    match rc.as_ref() {
        HeapObj::Array(a) => a.elements.iter().for_each(|e| collect_ref_ptrs(e, seen)),
        HeapObj::Record(r) => r.fields.iter().for_each(|e| collect_ref_ptrs(e, seen)),
        HeapObj::Adt(a) => a.fields.iter().for_each(|f| collect_ref_ptrs(&f.value, seen)),
        HeapObj::Newtype(n) => collect_ref_ptrs(&n.inner, seen),
        HeapObj::ThrowVal(t) => match &t.payload {
            crate::value::ThrowPayload::Ok(v) | crate::value::ThrowPayload::Err(v) => {
                collect_ref_ptrs(v, seen)
            }
        },
        HeapObj::Cell(c) => collect_ref_ptrs(&c.get(), seen),
        _ => {}
    }
}

/// Does any heap object reachable from `v` appear in `arg_ptrs`?
fn refs_intersect(v: &Value, arg_ptrs: &HashSet<usize>, seen: &mut HashSet<usize>) -> bool {
    let Value::Ref(rc) = v else { return false };
    let key = std::sync::Arc::as_ptr(rc) as usize;
    if arg_ptrs.contains(&key) {
        return true;
    }
    if !seen.insert(key) {
        return false;
    }
    use crate::value::HeapObj;
    match rc.as_ref() {
        HeapObj::Array(a) => a.elements.iter().any(|e| refs_intersect(e, arg_ptrs, seen)),
        HeapObj::Record(r) => r.fields.iter().any(|e| refs_intersect(e, arg_ptrs, seen)),
        HeapObj::Adt(a) => a.fields.iter().any(|f| refs_intersect(&f.value, arg_ptrs, seen)),
        HeapObj::Newtype(n) => refs_intersect(&n.inner, arg_ptrs, seen),
        HeapObj::ThrowVal(t) => match &t.payload {
            crate::value::ThrowPayload::Ok(v) | crate::value::ThrowPayload::Err(v) => {
                refs_intersect(v, arg_ptrs, seen)
            }
        },
        HeapObj::Cell(c) => refs_intersect(&c.get(), arg_ptrs, seen),
        _ => false,
    }
}

// =========================================================================
// Engine-side: launch + delivery apply
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// Attempt to offload a pure, heavy, cross-function SYNC call. On
    /// success the caller frame is suspended (queue-path semantics) and the
    /// child is executing on a worker; returns true. On false nothing has
    /// been touched and the ordinary queue path proceeds.
    pub(super) fn try_launch_offload(
        &self,
        caller_fid: FrameId,
        pending: &PendingCall,
        frame: &mut Frame,
    ) -> bool {
        let Some(rt) = self.offload_rt.as_ref().filter(|r| r.enabled) else {
            return false;
        };
        if pending.is_async || pending.closure_val.is_some() {
            return false;
        }
        let sg_idx = pending.target_sg.0 as usize;
        let sg = &self.graph.subgraphs[sg_idx];
        if sg.loop_kind != LoopKind::None {
            return false;
        }
        // Only fresh cross-function leaf calls: a same-function branch
        // (if/match arm) frame reuses the PARENT's node_offset and value
        // table — the executor's local indexing would go out of bounds and
        // panic the worker (silent mid-run death).
        let parent_fn =
            self.graph.subgraphs[frame.subgraph_id.0 as usize].function_id;
        if parent_fn == sg.function_id {
            return false;
        }
        let (ns, ne) = sg.node_range;
        if ((ne.0 - ns.0) as usize) < rt.min_nodes {
            return false;
        }
        if !self.graph.offload_safe(sg_idx) {
            return false;
        }
        if !offload_args_ok(&pending.args) {
            return false;
        }
        // Reference args are now allowed (read-only by the whitelist); their
        // single semantic risk is the RESULT aliasing an argument object —
        // checked exactly on the worker (see the restitution fallback).
        let args_have_refs = pending.args.iter().any(|v| matches!(v, Value::Ref(_)));

        let __t0 = std::time::Instant::now();
        // Copy-in: private deep clones; from here on the worker shares
        // nothing with the engine's live object graph.
        let cloned: Vec<Value> = pending
            .args
            .iter()
            .map(|a| crate::value::Arena::deep_clone_isolated(a))
            .collect();
        let arg_ref_ptrs: Option<HashSet<usize>> = if args_have_refs {
            let mut set = HashSet::new();
            for v in &cloned {
                collect_ref_ptrs(v, &mut set);
            }
            Some(set)
        } else {
            None
        };
        // Originals for the restitution fallback (cheap: Arc bumps).
        let original_args: Option<Vec<Value>> =
            if args_have_refs { Some(pending.args.clone()) } else { None };

        let (child_fid, mut child) = self.start_subgraph_frame(
            caller_fid,
            pending.call_node_local,
            pending.target_sg,
            &cloned,
            frame,
            None,
        );
        // Queue-path contract: chain pointers nulled (the executor verified
        // every input is local, so parents are never walked).
        child.root_frame_ptr = std::ptr::null_mut();
        child.parent_frame_ptr = std::ptr::null_mut();
        child.state = FrameState::Ready;

        // NOTE: no event_waiters registration on the normal path — the
        // delivery applies directly via pending_completions. Registration
        // happens only in the fallback re-drive (the sole consumer of the
        // SubgraphComplete waiter); a launch-time registration left an EMPTY
        // bucket after every normal delivery, which the deadlock detector
        // then misread as a live waiter (dogfood_json ~50% false-deadlocks).
        rt.stats.launches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        rt.stats.add(&rt.stats.launch_ns, __t0.elapsed());
        let seq = rt.launch_seq();
        let graph = self.graph.clone();
        let rt2 = rt.clone();
        let call_node = pending.call_node_local;
        // Frame carries chain raw pointers (nulled above) and a table of
        // Arc'd values (Value: Send) — safe to move; pointers are never
        // dereferenced on the worker (executor contract).
        // The closure moves a Box<Frame> (raw chain pointers inside, NULLed
        // at launch) across threads; pointer widths are plain data. Route it
        // through an asserted-Send bundle so the closure type checks.
        struct SendBox<T>(T);
        unsafe impl<T> Send for SendBox<T> {}
        fn offload_worker(
            graph: &DataFlowGraph,
            mut child: Box<Frame>,
            caller_fid: FrameId,
            call_node: NodeId,
            rt: &OffloadRt,
            seq: u64,
            arg_ref_ptrs: Option<HashSet<usize>>,
            original_args: Option<Vec<Value>>,
        ) {
            let __tw = std::time::Instant::now();
            let outcome = super::Schedule::run_offloaded_subgraph(graph, &mut child);
            rt.stats.add(&rt.stats.worker_ns, __tw.elapsed());
            // Restitution check: if the result aliases any (cloned) argument
            // object, applying it would hand the caller a private clone where
            // reference semantics demand the real object — fall back to a
            // re-run with the ORIGINAL args on the engine thread.
            let aliased = match (&outcome, &arg_ref_ptrs) {
                (OffloadOutcome::Done(v, _), Some(ptrs)) => {
                    let mut seen = HashSet::new();
                    refs_intersect(v, ptrs, &mut seen)
                }
                _ => false,
            };
            let done = match outcome {
                OffloadOutcome::Done(value, signal) if !aliased => OffloadDone {
                    caller: caller_fid,
                    call_node,
                    value,
                    signal,
                    child_frame: Some(child),
                    fallback_child: None,
                    fallback_args: None,
                },
                OffloadOutcome::Done(..) => OffloadDone {
                    caller: caller_fid,
                    call_node,
                    value: Value::VOID,
                    signal: ControlSignal::None,
                    child_frame: None,
                    fallback_child: Some(child),
                    fallback_args: original_args,
                },
                OffloadOutcome::Fallback => OffloadDone {
                    caller: caller_fid,
                    call_node,
                    value: Value::VOID,
                    signal: ControlSignal::None,
                    child_frame: None,
                    fallback_child: Some(child),
                    fallback_args: original_args,
                },
            };
            if std::env::var("FROND_DEBUG_OFFLOAD").is_ok() {
                eprintln!("[OFFLOAD] deliver seq={seq} caller={caller_fid:?}");
            }
            rt.deliver(seq, done);
        }
        let payload = SendBox(child);
        rayon::spawn(move || {
            let (graph, rt2, payload) = (graph, rt2, payload);
            let payload: Box<Frame> = payload.0;
            offload_worker(
                &graph,
                payload,
                caller_fid,
                call_node,
                &rt2,
                seq,
                arg_ref_ptrs,
                original_args,
            );
        });

        if std::env::var("FROND_DEBUG_OFFLOAD").is_ok() {
            eprintln!(
                "[OFFLOAD] pre-launch sg={} nodes={} args={} caller={:?}",
                sg_idx,
                (ne.0 - ns.0),
                pending.args.len(),
                caller_fid
            );
        }
        // Caller suspension — identical fields to the queue path.
        frame.state = FrameState::Suspended;
        frame.suspend_state = SuspendState::WaitingSubgraph(child_fid);
        frame.suspend_event = Some(RuntimeEvent::SubgraphComplete(child_fid));
        true
    }

    /// Engine thread: replay sequenced completions. Normal completions go
    /// through `pending_completions` + an idempotent queue push (the proven
    /// process_frame drain path); fallback children are re-inserted into the
    /// frames map and driven by the real machinery (the launch-time waiter
    /// registration makes their completion wake the caller normally).
    pub(super) fn apply_offload_deliveries(&self, queue: &QueueHandle<'_>) {
        let Some(rt) = self.offload_rt.as_ref() else { return };
        if rt.inflight() == 0 && !rt.head_ready() {
            return;
        }
        let mut completions: Vec<(FrameId, NodeId, Value, ControlSignal)> = Vec::new();
        let mut fallbacks: Vec<(Box<Frame>, Option<Vec<Value>>)> = Vec::new();
        let mut pooled_sink: Vec<Box<Frame>> = Vec::new();
        let __td = std::time::Instant::now();
        let __n_applied = rt.drain_apply(&mut |done| {
            if std::env::var("FROND_DEBUG_OFFLOAD").is_ok() {
                eprintln!("[OFFLOAD] apply seq->caller={:?}", done.caller);
            }
            match done.fallback_child {
                Some(child) => fallbacks.push((child, done.fallback_args)),
                None => {
                    if let Some(child) = done.child_frame {
                        pooled_sink.push(child);
                    }
                    completions.push((done.caller, done.call_node, done.value, done.signal))
                }
            }
        });
        if __n_applied > 0 {
            rt.stats.add(&rt.stats.deliver_ns, __td.elapsed());
        }
        for (caller, node, value, signal) in completions {
            self.pending_completions
                .lock()
                .entry(caller)
                .or_default()
                .push((node, value, signal));
            queue.push(caller);
        }
        for child in pooled_sink {
            self.release_frame(child);
        }
        for (mut child, orig_args) in fallbacks {
            if let Some(args) = orig_args {
                // Restitution / ref-arg classifier escape: re-execute with
                // the ORIGINAL arguments on this thread. Rebuild the child
                // frame from the (Suspended, in-map) caller so reference
                // semantics survive, then drive it through the real queue
                // machinery.
                let (caller_fid, call_node) =
                    child.caller.expect("offload child has caller");
                let target_sg = child.subgraph_id;
                let caller_frame = self.frames.lock().remove(&caller_fid);
                let Some(caller_frame) = caller_frame else {
                    // The caller must be in-map Suspended waiting on this
                    // child; absence is an engine-protocol breach.
                    panic!("offload restitution: caller {caller_fid:?} not in frames map");
                };
                let (new_fid, mut new_child) = self.start_subgraph_frame(
                    caller_fid,
                    call_node,
                    target_sg,
                    &args,
                    &caller_frame,
                    None,
                );
                self.frames.lock().insert(caller_fid, caller_frame);
                new_child.root_frame_ptr = std::ptr::null_mut();
                new_child.parent_frame_ptr = std::ptr::null_mut();
                new_child.state = FrameState::Ready;
                self.event_waiters
                    .lock()
                    .entry(RuntimeEvent::SubgraphComplete(new_fid))
                    .or_default()
                    .push(caller_fid);
                self.frames.lock().insert(new_fid, new_child);
                queue.push(new_fid);
                drop(child); // the clone-built child is discarded
                continue;
            }
            let cfid = child.id;
            child.state = FrameState::Ready;
            // The re-drive completes through complete_and_wake_caller, which
            // consumes this waiter to wake the caller.
            if let Some((caller, _)) = child.caller {
                self.event_waiters
                    .lock()
                    .entry(RuntimeEvent::SubgraphComplete(cfid))
                    .or_default()
                    .push(caller);
            }
            self.frames.lock().insert(cfid, child);
            queue.push(cfid);
        }
    }

    fn stats_add_worker(&self, d: std::time::Duration) {
        if let Some(rt) = self.offload_rt.as_ref() {
            rt.stats.add(&rt.stats.worker_ns, d);
        }
    }

    /// Engine thread: in-flight offload count (idle-policy input).
    pub(super) fn offload_inflight(&self) -> usize {
        self.offload_rt.as_ref().map(|rt| rt.inflight()).unwrap_or(0)
    }
}
