//! Offload — L2: pure-subgraph compute offload for the deterministic event
//! loop (--offload, gray-release).
//!
//! Model: the engine thread owns ALL mutable state and scheduling, always.
//! A synchronous call whose subgraph is classified pure
//! (`is_offload_safe_compute` over every node — see ir/Ir.rs) and heavy
//! enough (node-count threshold, [engine] offload_min) executes on a rayon
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
use crate::pass::Scalarizer::{run_scalar_prog, ScalarProg};
use super::Schedule::relay_branch_value;
use crate::ir::Ir::{Frame, FrameId, NodeId, PendingCall, ControlSignal, LoopKind, RuntimeEvent, SuspendState, FrameState, SubGraphId, DataFlowGraph};
use crate::value::Value;
use parking_lot::{Condvar, Mutex as ParkingMutex};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Launch threshold below which offload is pointless (copy + delivery
/// overhead exceeds the compute). Tunable via [engine] offload_min.
pub(super) const DEFAULT_OFFLOAD_MIN_NODES: usize = 2000;
// Threshold economics (measured): launch+copy+delivery overhead is ~0.5ms
// per offload; a pure straight-line leaf averages ~1µs/node, so a node-count
// gate must sit where work clearly dominates. 2000 nodes ≈ ≥1ms of compute.
// A real cost model (and loop-level / slot-provenance extensions) is the L2
// follow-up; until then this keeps the machinery sound and dormant by
// default, exercised via --offload soak.

/// One completed offload, waiting for its turn in the sequencer.
struct OffloadDone {
    caller: FrameId,
    call_node: NodeId,
    value: Value,
    signal: ControlSignal,
    /// The executed child, shipped back only when the frame must return to
    /// the engine (unused on the clean path since the worker-frame cache:
    /// clean completions keep the frame worker-side for warm reuse).
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
    pub min_nodes: usize,
    inner: ParkingMutex<OffloadInner>,
    /// Engine-thread wakeup: notified on every delivery.
    pub parked: Condvar,
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

/// Outcome of the merged single-lock park.
pub enum ParkOutcome {
    /// Sequence head applicable — the loop-top drain makes progress.
    HeadReady,
    /// Nothing in flight and head not ready — caller runs the
    /// timer/reconcile/deadlock path.
    NothingInFlight,
    /// Wait bounded by a timer deadline; still work in flight.
    TimedOut,
}

impl OffloadRt {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            min_nodes: DEFAULT_OFFLOAD_MIN_NODES,
            inner: ParkingMutex::new(OffloadInner {
                next_launch_seq: 0,
                next_apply_seq: 0,
                inflight: 0,
                ready: HashMap::new(),
            }),
            parked: Condvar::new(),
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
        // Wake coalescing: notify ONLY when this delivery unblocks an apply
        // (it completed the sequence head — the engine's loop-top drain will
        // apply the whole ready chain in one pass) or when it was the last
        // completion in flight (the engine must run its idle/deadlock pass).
        // Out-of-order deliveries never wake the engine — they just buffer.
        // The engine is the sole waiter, so notify_one suffices.
        let should_wake = {
            let mut g = self.inner.lock();
            g.ready.insert(seq, done);
            g.inflight -= 1;
            seq == g.next_apply_seq || g.inflight == 0
        };
        if should_wake {
            self.parked.notify_one();
        }
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

    /// The subgraph's compiled scalar program, compiling on first use.
    fn fast_prog(&self, graph: &DataFlowGraph, sg: SubGraphId) -> Option<std::sync::Arc<ScalarProg>> {
        graph.scalar_prog(sg.0 as usize)
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

    /// Single-lock wake path (replaces the inflight()/park()/head_ready()
    /// triple acquisition): head check + inflight check + bounded wait in
    /// ONE mutex acquisition on the hot park/wake cycle.
    pub fn park_for_head(&self, limit: Option<std::time::Instant>) -> ParkOutcome {
        let mut g = self.inner.lock();
        if g.ready.contains_key(&g.next_apply_seq) {
            return ParkOutcome::HeadReady;
        }
        if g.inflight == 0 {
            return ParkOutcome::NothingInFlight;
        }
        match limit {
            Some(deadline) => {
                let dur = deadline.saturating_duration_since(std::time::Instant::now());
                if !dur.is_zero() {
                    self.parked.wait_for(&mut g, dur);
                }
            }
            None => self.parked.wait(&mut g),
        }
        if g.ready.contains_key(&g.next_apply_seq) {
            ParkOutcome::HeadReady
        } else {
            ParkOutcome::TimedOut
        }
    }

    /// Merged pending check for the apply early-exit (one lock instead of
    /// inflight() + head_ready()).
    pub fn has_pending(&self) -> bool {
        let g = self.inner.lock();
        g.inflight > 0 || g.ready.contains_key(&g.next_apply_seq)
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
    if let Value::Record(r) = v {
        let key = crate::value::record_tagged_ptr(r);
        if !seen.insert(key) {
            return true;
        }
        return (0..r.field_count()).all(|i| value_offload_safe(&r.field(i), seen));
    }
    let Value::Ref(rc) = v else { return true };
    let key = Arc::as_ptr(rc) as usize;
    if !seen.insert(key) {
        return true; // already visited (shared/cyclic): verdict stands
    }
    use crate::value::HeapObj;
    match rc.as_ref() {
        HeapObj::Closure(_) | HeapObj::Partial(_) => false,
        HeapObj::Array(a) => a.elements.iter().all(|e| value_offload_safe(e, seen)),
        HeapObj::ThrowVal(t) => match &t.payload {
            crate::value::ThrowPayload::Ok(v) | crate::value::ThrowPayload::Err(v) => {
                value_offload_safe(v, seen)
            }
        },
        HeapObj::Cell(c) => value_offload_safe(&c.get(), seen),
        _ => true,
    }
}

// =========================================================================
// Worker-side frame lifecycle
// =========================================================================

thread_local! {
    /// Per-worker reused offload execution frames, keyed by target subgraph
    /// (a cross-function frame's layout — node_offset, table sizes, seeding —
    /// is fully determined by the sg). A repeated offload of the same sg
    /// re-executes on a value table already resident in THIS core's cache:
    /// no per-launch allocation, no engine-thread-initialized memory being
    /// invalidated cross-core, no frame shipping either direction. The
    /// engine's own cached_child_frame/hot_body reuse for loop bodies is the
    /// in-thread analogue of this mechanism.
    static WORKER_FRAMES: std::cell::RefCell<HashMap<u32, Box<Frame>>> =
        std::cell::RefCell::new(HashMap::new());
}
/// Cache cap: distinct subgraphs retained per worker. Evicted frames drop
/// (Arc'd values released) on the worker thread, which owns them.
const WORKER_FRAME_CACHE_MAX: usize = 4;

/// Builds or reuses this worker's execution frame for `sg` and seeds it with
/// exactly the engine's cross-function `start_subgraph_frame` recipe — the
/// same `prepare_frame_nodes` and the same parameter-injection loop, so the
/// worker's frame is behaviorally identical to an engine-built one.
fn take_worker_frame(
    graph: &std::sync::Arc<DataFlowGraph>,
    child_fid: FrameId,
    caller_fid: FrameId,
    call_node: NodeId,
    sg_id: SubGraphId,
    args: &[Value],
) -> Box<Frame> {
    let sg = &graph.subgraphs[sg_id.0 as usize];
    let (node_start, node_end) = sg.node_range;
    let node_count = (node_end.0 - node_start.0) as usize;

    let mut child = WORKER_FRAMES
        .with(|cache| cache.borrow_mut().remove(&sg_id.0))
        .unwrap_or_else(|| Box::new(Frame::new(child_fid, sg_id, node_count, graph.clone())));
    // Re-target the reused frame exactly like the engine's acquire_frame.
    child.id = child_fid;
    child.subgraph_id = sg_id;
    child.value_table.resize(node_count);
    child.value_table.disable_dirty_tracking();
    child.pending_inputs.resize(node_count, 0);

    // Seed: mirror of Engine::prepare_frame (reset + linear-fresh + seeding).
    child.value_table.reset_all();
    child.ready_queue.clear();
    child.control_signal = ControlSignal::None;
    child.linear_fresh = true;
    super::Schedule::prepare_frame_nodes(&mut child, graph);
    // Engine-machinery residues must not survive a pooled/reused frame.
    child.defer_stack.clear();
    child.suspend_event = None;
    child.select_timers.clear();
    child.cached_child_frame = None;
    child.hot_body = None;
    child.same_fn_prep_cache = None;
    child.construct_cache.clear();
    child.branch_relays.clear();
    child.closure_val = None;
    child.caller = Some((caller_fid, call_node));
    // Queue-path contract: chain pointers nulled (the executor verified
    // every input is local, so parents are never walked).
    child.root_frame_ptr = std::ptr::null_mut();
    child.parent_frame_ptr = std::ptr::null_mut();
    child.state = FrameState::Ready;

    // Parameter injection — verbatim the engine's cross-function path.
    let offset = node_start.0 as usize;
    for (i, arg) in args.iter().enumerate().take(sg.param_count as usize) {
        let local_id = NodeId(i as u32);
        let global_id = NodeId((offset + i) as u32);
        let consumer_count = graph.downstream_count(offset + i);
        child.set_value(local_id, arg.clone(), consumer_count);
        // Do not push_ready: the parameter value is already set; notify_downstream
        // propagates it downstream.
        super::Schedule::notify_downstream(
            &mut child,
            graph,
            local_id,
            global_id,
            NodeId(node_start.0),
        );
    }
    child
}

/// Returns a cleanly-finished frame to this worker's cache for the next
/// launch of the same sg. Results are dropped here (worker-thread drop of
/// values this worker produced); the allocation and table memory stay warm.
fn store_worker_frame(sg_id: SubGraphId, mut frame: Box<Frame>) {
    frame.value_table.reset_all();
    frame.branch_relays.clear();
    frame.defer_stack.clear();
    frame.select_timers.clear();
    WORKER_FRAMES.with(|cache| {
        let mut m = cache.borrow_mut();
        if m.len() >= WORKER_FRAME_CACHE_MAX && !m.contains_key(&sg_id.0) {
            if let Some(evict) = m.keys().next().copied() {
                m.remove(&evict);
            }
        }
        m.insert(sg_id.0, frame);
    });
}

/// Worker-side job body: obtain the (reused) frame, execute, and either cache
/// it for the next launch of this sg (clean completion) or ship it back to
/// the engine for the fallback re-drive (classifier escape / restitution).
fn offload_worker(
    graph: &std::sync::Arc<DataFlowGraph>,
    child_fid: FrameId,
    caller_fid: FrameId,
    call_node: NodeId,
    target_sg: SubGraphId,
    cloned_args: Vec<Value>,
    rt: &OffloadRt,
    seq: u64,
    arg_ref_ptrs: Option<HashSet<usize>>,
    original_args: Option<Vec<Value>>,
) {
    // Fast path: scalar-args launch of a subgraph with a compiled scalar
    // program runs frame-less (params → straight-line ops → return slot).
    // Scalar results cannot alias argument objects, so the restitution
    // check is vacuous here.
    if cloned_args.iter().all(|v| matches!(v, Value::Scalar(..))) {
        if let Some(prog) = rt.fast_prog(graph, target_sg) {
            let value = run_scalar_prog(&prog, &cloned_args);
            let done = OffloadDone {
                caller: caller_fid,
                call_node,
                value,
                signal: ControlSignal::None,
                child_frame: None,
                fallback_child: None,
                fallback_args: None,
            };
            rt.deliver(seq, done);
            return;
        }
    }

    let mut child = take_worker_frame(
        graph,
        child_fid,
        caller_fid,
        call_node,
        target_sg,
        &cloned_args,
    );
    let outcome = super::Schedule::run_offloaded_subgraph(graph, &mut child);
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
        OffloadOutcome::Done(value, signal) if !aliased => {
            // Clean completion: keep the frame warm on this worker.
            store_worker_frame(target_sg, child);
            OffloadDone {
                caller: caller_fid,
                call_node,
                value,
                signal,
                child_frame: None,
                fallback_child: None,
                fallback_args: None,
            }
        }
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
    rt.deliver(seq, done);
}

/// Collects every heap-object address reachable from `v` (cycles tolerated).
fn collect_ref_ptrs(v: &Value, seen: &mut HashSet<usize>) {
    if let Value::Record(r) = v {
        let key = crate::value::record_tagged_ptr(r);
        if seen.insert(key) {
            (0..r.field_count()).for_each(|i| collect_ref_ptrs(&r.field(i), seen));
        }
        return;
    }
    let Value::Ref(rc) = v else { return };
    let key = std::sync::Arc::as_ptr(rc) as usize;
    if !seen.insert(key) {
        return;
    }
    use crate::value::HeapObj;
    match rc.as_ref() {
        HeapObj::Array(a) => a.elements.iter().for_each(|e| collect_ref_ptrs(e, seen)),
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
    if let Value::Record(r) = v {
        let key = crate::value::record_tagged_ptr(r);
        if arg_ptrs.contains(&key) {
            return true;
        }
        if !seen.insert(key) {
            return false;
        }
        return (0..r.field_count()).any(|i| refs_intersect(&r.field(i), arg_ptrs, seen));
    }
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
        let Some(rt) = self.offload_rt.as_ref() else {
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

        // Lightweight launch: only ids + cloned args cross threads. The
        // execution frame is built (and afterwards kept) on the worker —
        // repeated offloads of the same subgraph re-execute on a value table
        // that stays resident in that core's cache, mirroring the engine's
        // cached_child_frame/hot_body reuse for loop bodies.
        let child_fid = self.alloc_frame_id();

        // NOTE: no event_waiters registration on the normal path — the
        // delivery applies directly via pending_completions. Registration
        // happens only in the fallback re-drive (the sole consumer of the
        // SubgraphComplete waiter); a launch-time registration left an EMPTY
        // bucket after every normal delivery, which the deadlock detector
        // then misread as a live waiter (dogfood_json ~50% false-deadlocks).
        let seq = rt.launch_seq();
        let graph = self.graph.clone();
        let rt2 = rt.clone();
        let call_node = pending.call_node_local;
        let target_sg = pending.target_sg;
        // Payload is plain data (Arc'd graph, Send values, ids) — no frame
        // crosses threads on the launch path anymore.
        rayon::spawn(move || {
            offload_worker(
                &graph,
                child_fid,
                caller_fid,
                call_node,
                target_sg,
                cloned,
                &rt2,
                seq,
                arg_ref_ptrs,
                original_args,
            );
        });

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
        if !rt.has_pending() {
            return;
        }
        let mut completions: Vec<(FrameId, NodeId, Value, ControlSignal)> = Vec::new();
        let mut fallbacks: Vec<(Box<Frame>, Option<Vec<Value>>)> = Vec::new();
        let mut pooled_sink: Vec<Box<Frame>> = Vec::new();
        let __n_applied = rt.drain_apply(&mut |done| {
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
        for (caller, node, value, signal) in completions {
            if let Some((value, signal)) =
                self.try_inline_offload_resume(caller, node, value, signal, queue)
            {
                self.pending_completions
                    .lock()
                    .entry(caller)
                    .or_default()
                    .push((node, value, signal));
                queue.push(caller);
            }
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


    /// Offload-delivery fast resume: when the caller frame is parked in the
    /// frames map (not mid-processing), take it back, apply the completion
    /// exactly like the dispatch-time stashed-completion path (value write +
    /// signal propagation + notify + waiter cleanup), mark Ready, and drive
    /// it INLINE — collapsing the apply/run/requeue dispatch cycle into this
    /// call. Returns false (caller must stash + queue) when the frame is
    /// absent (being processed — Bug #78 race) or not parked on this child.
    fn try_inline_offload_resume(
        &self,
        caller: FrameId,
        call_node: NodeId,
        value: Value,
        signal: ControlSignal,
        queue: &QueueHandle<'_>,
    ) -> Option<(Value, ControlSignal)> {
        // Only frames suspended WaitingSubgraph-style park in the map; the
        // offload launcher set exactly that. Anything else (Ready, absent,
        // other wait kinds) falls back to the queue protocol.
        let take = {
            let mut frames = self.frames.lock();
            match frames.get_mut(&caller) {
                Some(f) if f.state == FrameState::Suspended => {
                    frames.remove(&caller)
                }
                _ => None,
            }
        };
        let Some(mut frame) = take else {
            return Some((value, signal));
        };
        // Waiter cleanup (mirror of the dispatch path).
        if let Some(e) = frame.suspend_event {
            if let Some(bucket) = self.event_waiters.lock().get_mut(&e) {
                bucket.retain(|wf| *wf != caller);
            }
        } else {
            let mut ew = self.event_waiters.lock();
            for bucket in ew.values_mut() {
                bucket.retain(|wf| *wf != caller);
            }
        }
        let caller_offset = NodeId(frame.node_offset);
        let call_graph_id = NodeId(call_node.0 + caller_offset.0);
        let consumer_count = self.graph.downstream_count(call_graph_id.0 as usize);
        frame.set_value(call_node, value, consumer_count);
        if !frame.branch_relays.is_empty() {
            relay_branch_value(&mut frame, &self.graph, call_node);
        }
        // Control-signal propagation (mirror of the dispatch path, including
        // the broadened Bug #78 rule and the capture-gate exception).
        let capture_gate = self.graph.node(call_graph_id.0 as usize).kind
            == crate::ir::Ir::NodeKind::Gate
            && self
                .graph
                .gate_branches_at(call_graph_id.0 as usize)
                .map(|gb| gb.capture)
                .unwrap_or(false);
        if !capture_gate && !matches!(signal, ControlSignal::None) {
            frame.control_signal = signal;
        }
        notify_downstream(
            &mut frame,
            &self.graph,
            call_node,
            call_graph_id,
            caller_offset,
        );
        frame.state = FrameState::Ready;
        frame.suspend_state = SuspendState::NotSuspended;
        frame.suspend_event = None;
        self.frames.lock().insert(caller, frame);
        self.process_frame(caller, queue);
        None
    }


    /// Engine thread: in-flight offload count (idle-policy input).
    pub(super) fn offload_inflight(&self) -> usize {
        self.offload_rt.as_ref().map(|rt| rt.inflight()).unwrap_or(0)
    }
}
