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
    /// Classifier escape: the engine re-drives this child through the real
    /// machinery instead of applying the partial result.
    fallback_child: Option<Box<Frame>>,
}

/// Shared offload runtime: held via Arc by the engine and every in-flight
/// worker. The ONLY cross-thread state in the process, fully mutex-guarded.
pub(super) struct OffloadRt {
    pub enabled: bool,
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

enum OffloadOutcome {
    Done(Value, ControlSignal),
    Fallback,
}

/// Lean executor for a pure leaf subgraph: drains the sg's linear plan
/// (topological order, precomputed at graph load) with none of the engine's
/// queue/gate machinery — the whitelist guarantees none of it is reachable.
/// Reads only the frame's own table; a missing input (an outer-scope read
/// the classifier let through) is a Fallback, never a panic.
pub(super) fn run_offloaded_subgraph(graph: &DataFlowGraph, frame: &mut Frame) -> OffloadOutcome {
    let plan: &[NodeId] = match graph.linear_plan(frame.subgraph_id.0 as usize) {
        Some(p) if !p.is_empty() => p,
        _ => return OffloadOutcome::Fallback,
    };
    let node_start = frame.node_offset;
    for &gid in plan {
        if !matches!(frame.control_signal, ControlSignal::None) {
            break;
        }
        let local = NodeId(gid.0.wrapping_sub(node_start));
        if local.0 as usize >= frame.value_table_len() {
            return OffloadOutcome::Fallback;
        }
        if frame.value_table.is_ready(local.0 as usize) {
            continue;
        }
        let node = graph.node(gid.0 as usize);
        // Every input must be produced inside this frame (args are injected
        // at construction); anything else is a classifier escape — bail.
        let inputs = graph.inputs(node.inputs_offset, node.input_count);
        for inp in inputs {
            let inp_local = NodeId(inp.0.wrapping_sub(node_start));
            if inp_local.0 as usize >= frame.value_table_len()
                || !frame.value_table.is_ready(inp_local.0 as usize)
            {
                return OffloadOutcome::Fallback;
            }
        }
        let ctx = EvalContext { node_start, graph };
        let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, gid, &ctx);
        match result {
            NodeResult::Value(v) => {
                let cc = graph.downstream_count(gid.0 as usize);
                frame.set_value(local, v, cc);
            }
            NodeResult::Batch(results) => {
                for &(lid, ref v) in &results {
                    let g2 = lid.0 + node_start;
                    let cc = graph.downstream_count(g2 as usize);
                    frame.set_value(lid, v.clone(), cc);
                }
            }
            NodeResult::Return(v) => {
                frame.control_signal = ControlSignal::Return(v);
                break;
            }
            NodeResult::Break => {
                frame.control_signal = ControlSignal::Break;
                break;
            }
            NodeResult::Continue => {
                frame.control_signal = ControlSignal::Continue;
                break;
            }
            // Engine-needing results cannot come from whitelisted compute
            // functions; if the classifier is ever wrong this lands here.
            _ => return OffloadOutcome::Fallback,
        }
    }
    OffloadOutcome::Done(
        super::Schedule::extract_child_return(frame, graph),
        frame.control_signal.clone(),
    )
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

        // Copy-in: private deep clones; from here on the worker shares
        // nothing with the engine's live object graph.
        let cloned: Vec<Value> = pending
            .args
            .iter()
            .map(|a| crate::value::Arena::deep_clone_isolated(a))
            .collect();

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
        ) {
            let outcome = run_offloaded_subgraph(graph, &mut child);
            let done = match outcome {
                OffloadOutcome::Done(value, signal) => OffloadDone {
                    caller: caller_fid,
                    call_node,
                    value,
                    signal,
                    fallback_child: None,
                },
                OffloadOutcome::Fallback => OffloadDone {
                    caller: caller_fid,
                    call_node,
                    value: Value::VOID,
                    signal: ControlSignal::None,
                    fallback_child: Some(child),
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
            offload_worker(&graph, payload, caller_fid, call_node, &rt2, seq);
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
        let mut fallbacks: Vec<Box<Frame>> = Vec::new();
        rt.drain_apply(&mut |done| {
            if std::env::var("FROND_DEBUG_OFFLOAD").is_ok() {
                eprintln!("[OFFLOAD] apply seq->caller={:?}", done.caller);
            }
            match done.fallback_child {
                Some(child) => fallbacks.push(child),
                None => completions.push((done.caller, done.call_node, done.value, done.signal)),
            }
        });
        for (caller, node, value, signal) in completions {
            self.pending_completions
                .lock()
                .entry(caller)
                .or_default()
                .push((node, value, signal));
            queue.push(caller);
        }
        for mut child in fallbacks {
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

    /// Engine thread: in-flight offload count (idle-policy input).
    pub(super) fn offload_inflight(&self) -> usize {
        self.offload_rt.as_ref().map(|rt| rt.inflight()).unwrap_or(0)
    }
}
