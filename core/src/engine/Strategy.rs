//! Concurrency strategies: LockStrategy / Single / Multi / QueueHandle + the
//! deterministic event loop.
//!
//! M3b (2026-09): the multi-worker pool is DELETED. Both strategy variants
//! execute graphs on ONE thread (the caller's) — Single for entries with no
//! suspension point, Multi (the async-capable variant) via the deterministic
//! event loop. The "one thread executes graphs" invariant the value layer
//! (`Arc<HeapObj>` in-place mutation, `Cell`'s UnsafeCell, the thread_local
//! GLOBAL_ARENA) and the frame-chain machinery were designed around is now an
//! architectural property, not a hope. The retired FROND_EVENTLOOP /
//! FROND_WORKERS variables are ignored.

use super::*;
use crate::ir::Ir::*;
use crate::value::{Value, ValueArena};
use std::cell::{RefCell, RefMut};
use std::ops::DerefMut;
use parking_lot::{Condvar, Mutex as ParkingMutex};
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;

// =========================================================================
// LockStrategy — compile-time lock strategy (single-threaded RefCell vs multi-threaded ParkingMutex)
// =========================================================================

/// Lock strategy: decides the field-wrapping type at compile time (single-threaded RefCell vs
/// multi-threaded ParkingMutex).
pub trait LockStrategy: 'static {
    type Mutex<T>: Lockable<T>;
}

/// Lockable trait: provides a `lock()` method that returns a guard.
pub trait Lockable<T> {
    type Guard<'a>: DerefMut<Target = T>
    where
        Self: 'a;
    fn lock(&self) -> Self::Guard<'_>;
}

// Single-threaded strategy: RefCell (borrow flag, ~2ns, no syscall).
pub struct Single;
impl LockStrategy for Single {
    type Mutex<T> = RefCell<T>;
}
impl<T> Lockable<T> for RefCell<T> {
    type Guard<'a>
        = RefMut<'a, T>
    where
        T: 'a;
    fn lock(&self) -> Self::Guard<'_> {
        self.borrow_mut()
    }
}

/// Async-capable strategy: representationally identical to [`Single`] (see
/// the module doc — the multi-worker pool was deleted in M3b).
pub struct Multi;
impl LockStrategy for Multi {
    type Mutex<T> = RefCell<T>;
}

/// Frame-queue abstraction. EventLoop pushes are IDEMPOTENT: at most one
/// pending queue entry per frame — the duplicate-entry family's root fix
/// (see `Engine::queued_dedup`).
pub enum QueueHandle<'a> {
    Single(&'a RefCell<std::collections::VecDeque<FrameId>>),
    EventLoop {
        queue: &'a RefCell<std::collections::VecDeque<FrameId>>,
        dedup: &'a ParkingMutex<HashSet<FrameId>>,
    },
}
impl QueueHandle<'_> {
    pub fn push(&self, fid: FrameId) {
        match self {
            Self::Single(q) => q.borrow_mut().push_back(fid),
            Self::EventLoop { queue, dedup } => {
                let mut set = dedup.lock();
                if set.insert(fid) {
                    queue.borrow_mut().push_back(fid);
                }
            }
        }
    }
}

// =========================================================================
// impl Engine<Single> — single-threaded mode
// =========================================================================

impl Engine<Single> {
    /// Creates a single-threaded Engine (wraps every field with RefCell).
    pub(super) fn new_single(graph: DataFlowGraph) -> Self {
        let graph = Arc::new(graph);
        Self {
            graph: graph.clone(),
            frames: RefCell::new(HashMap::new()),
            next_frame_id: RefCell::new(FrameId(0)),
            arena: RefCell::new(ValueArena::new()),
            timer_runtime: RefCell::new(TimerRuntime::new()),
            async_join_runtime: RefCell::new(AsyncJoinRuntime::new()),
            event_waiters: RefCell::new(std::collections::HashMap::new()),
            pending_completions: RefCell::new(HashMap::new()),
            pending_events: RefCell::new(HashMap::new()),
            defer_frames: RefCell::new(HashSet::new()),
            defer_waiters: RefCell::new(HashMap::new()),
            result: RefCell::new(None),
            frame_pool: RefCell::new(Vec::new()),
            ready_frames: Some(RefCell::new(std::collections::VecDeque::new())),
            queued_dedup: None,
            _strategy: std::marker::PhantomData,
        }
    }

    /// Single-threaded event loop (replaces run_event_loop + run_entry).
    ///
    /// Idle policy (queue empty + pending events present):
    /// - With a pending timer: park until the nearest deadline (Condvar.wait_for).
    /// - Without a pending timer but with event_waiters: yield_now (waiting on channel/async events).
    /// - Without pending work and without waiters: panic (deadlock detection).
    pub(super) fn run_single(&self) -> Value {
        let entry_sg = self.graph.entry_subgraph.expect("no entry subgraph");
        let fid = self.init_entry_frame(entry_sg);
        let rq = self.ready_frames.as_ref().unwrap();
        rq.borrow_mut().push_back(fid);

        // Condvar used for single-threaded parking (no external wake-up needed; used only for
        // precise wait_for timing).
        let park_mutex = ParkingMutex::new(());
        let park_cv = Condvar::new();

        // Progress-based livelock watchdog: only iterations that find the queue empty
        // (park / yield / timer waits) count toward the guard — every processed frame
        // is real progress and resets it. Legitimately long computations (including
        // user-level infinite loops that keep doing work) never trip it; a scheduler
        // livelock (queue empty forever while waiters/timers exist but are never
        // notified) still does. Hard deadlock (nothing pending at all) is caught
        // separately below.
        let mut idle_spins: u64 = 0;
        loop {
            let queue = QueueHandle::Single(rq);
            // Pop first (the RefMut is released at the end of the statement), then handle the
            // empty-queue logic.
            let fid = rq.borrow_mut().pop_front();
            let fid = match fid {
                Some(f) => f,
                None => {
                    idle_spins += 1;
                    if idle_spins > 200_000_000 {
                        panic!(
                            "event loop stuck: {idle_spins} idle iterations without a ready \
                             frame (livelock suspected: waiters/timers present but never notified)"
                        );
                    }
                    // Queue empty: check timers (check_timers -> on_event_arrived -> push needs borrow_mut).
                    self.check_timers(&queue);
                    if let Some(result) = self.result.lock().take() {
                        return result;
                    }
                    // Still no ready frame: decide the park policy.
                    if rq.borrow().is_empty() {
                        let next_deadline = self.timer_runtime.lock().next_deadline();
                        let ew_empty = self.event_waiters.lock().is_empty();
                        if ew_empty && next_deadline.is_none() {
                            panic!(
                                "event loop exhausted: no ready frames and no pending events"
                            );
                        }
                        if let Some(deadline) = next_deadline {
                            // Pending timer present: park until the deadline (precise wait, no busy polling).
                            let now = std::time::Instant::now();
                            let wait_dur = deadline.saturating_duration_since(now);
                            if !wait_dur.is_zero() {
                                let mut guard = park_mutex.lock();
                                park_cv.wait_for(&mut guard, wait_dur);
                            }
                        } else {
                            // No timer but event_waiters present: yield waiting for channel/async/subgraph events.
                            std::thread::yield_now();
                        }
                    }
                    continue;
                }
            };
            self.process_frame(fid, &queue);
            idle_spins = 0;
            if let Some(result) = self.result.lock().take() {
                return result;
            }
        }
    }
}

// =========================================================================
// impl Engine<Multi> — multi-worker mode
// =========================================================================

impl Engine<Multi> {
    /// Async-capable engine — representationally identical to
    /// [`Engine<Single>`]; see the module doc (the worker pool was deleted in
    /// M3b: it was the last home of the cross-worker UB family, and the
    /// FROND_EVENTLOOP=0 escape hatch proved ~2/3-flaky on macOS arm64's
    /// weakly-ordered memory).
    pub(super) fn new_multi(graph: DataFlowGraph) -> Self {
        let graph = Arc::new(graph);
        Self {
            graph: graph.clone(),
            frames: RefCell::new(HashMap::new()),
            next_frame_id: RefCell::new(FrameId(0)),
            arena: RefCell::new(ValueArena::new()),
            timer_runtime: RefCell::new(TimerRuntime::new()),
            async_join_runtime: RefCell::new(AsyncJoinRuntime::new()),
            event_waiters: RefCell::new(std::collections::HashMap::new()),
            pending_completions: RefCell::new(HashMap::new()),
            pending_events: RefCell::new(HashMap::new()),
            defer_frames: RefCell::new(HashSet::new()),
            defer_waiters: RefCell::new(HashMap::new()),
            result: RefCell::new(None),
            frame_pool: RefCell::new(Vec::new()),
            ready_frames: Some(RefCell::new(std::collections::VecDeque::new())),
            queued_dedup: Some(ParkingMutex::new(HashSet::new())),
            _strategy: std::marker::PhantomData,
        }
    }

    /// Async-capable entry point: the deterministic event loop, on this thread.
    pub(super) fn run_multi(self: Arc<Self>) -> Value {
        self.run_event_loop_multi()
    }
}

// =========================================================================
// Deterministic event loop (Multi) — the sole async scheduler since M3b
// =========================================================================

impl Engine<Multi> {
    /// Deterministic single-threaded event loop for async-capable graphs.
    ///
    /// Executes every frame on the caller's thread with FIFO, idempotent
    /// queueing — the value layer's "one thread executes graphs" invariant
    /// holds by construction (no cross-worker access to heap objects, cells,
    /// frame chains, or the GLOBAL_ARENA). Idle policy mirrors the proven
    /// safety nets: timer poll, stranded-frame rescue sweep (BOTH stash
    /// tables), AsyncJoin reconciler, and a provably-permanent-wait exit
    /// with a full state dump (a Frond event source is always a frame or a
    /// timer; when neither can run and nothing is deliverable, waiting is
    /// mathematically permanent).
    fn run_event_loop_multi(self: Arc<Self>) -> Value {
        let entry_sg = self.graph.entry_subgraph.expect("no entry subgraph");
        let fid = self.init_entry_frame(entry_sg);
        let rq = self.ready_frames.as_ref().unwrap();
        let dedup = self.queued_dedup.as_ref().unwrap();
        {
            let mut set = dedup.lock();
            if set.insert(fid) {
                rq.borrow_mut().push_back(fid);
            }
        }

        let park_mutex = ParkingMutex::new(());
        let park_cv = Condvar::new();
        let mut idle_spins: u64 = 0;
        loop {
            let queue = QueueHandle::EventLoop { queue: &rq, dedup };
            let fid = {
                let mut set = dedup.lock();
                let f = rq.borrow_mut().pop_front();
                if let Some(f) = &f {
                    set.remove(f);
                }
                f
            };
            let fid = match fid {
                Some(f) => f,
                None => {
                    idle_spins += 1;
                    if idle_spins > 200_000_000 {
                        panic!(
                            "event loop stuck: {idle_spins} idle iterations without a ready \
                             frame (livelock suspected)"
                        );
                    }
                    self.check_timers(&queue);
                    if let Some(result) = self.result.lock().take() {
                        return result;
                    }
                    if rq.borrow().is_empty() {
                        if idle_spins.is_multiple_of(65_536) {
                            self.rescue_stranded_frames_multi(&rq, dedup);
                        }
                        let next_deadline = self.timer_runtime.lock().next_deadline();
                        if next_deadline.is_none() && !self.reconcile_stale_joins_multi(&queue) {
                            // Provably permanent: every waiter's source is a
                            // frame (none runnable — the sweep found nothing)
                            // or a timer (none pending).
                            // Dump and exit loudly.
                            self.dump_deadlock_state_multi();
                            panic!(
                                "event loop: provably permanent wait (no runnable frames, \
                                 no timers, undeliverable waiters)"
                            );
                        }
                        if let Some(deadline) = next_deadline {
                            let wait_dur =
                                deadline.saturating_duration_since(std::time::Instant::now());
                            if !wait_dur.is_zero() {
                                let mut guard = park_mutex.lock();
                                park_cv.wait_for(&mut guard, wait_dur);
                            }
                        } else {
                            std::thread::yield_now();
                        }
                    }
                    continue;
                }
            };
            self.process_frame(fid, &queue);
            idle_spins = 0;
            if let Some(result) = self.result.lock().take() {
                return result;
            }
        }
    }

    /// Requeue frames that are provably ready-to-run but hold no queue entry
    /// (Ready-but-unqueued; Suspended carrying a stashed event OR completion —
    /// only a dispatch drains stashes). Defer-waiters are excluded.
    fn rescue_stranded_frames_multi(
        &self,
        rq: &RefCell<std::collections::VecDeque<FrameId>>,
        dedup: &ParkingMutex<HashSet<FrameId>>,
    ) {
        let mut rescued: Vec<FrameId> = Vec::new();
        {
            let mut stashed: HashSet<FrameId> = {
                let pe = self.pending_events.lock();
                pe.keys().copied().collect()
            };
            {
                let pc = self.pending_completions.lock();
                stashed.extend(pc.keys().copied());
            }
            let defer_waiters: HashSet<FrameId> = {
                let dw = self.defer_waiters.lock();
                dw.keys().copied().collect()
            };
            let frames = self.frames.lock();
            for (fid, f) in frames.iter() {
                if f.state == FrameState::Ready {
                    rescued.push(*fid);
                } else if f.state == FrameState::Suspended
                    && stashed.contains(fid)
                    && !defer_waiters.contains(fid)
                {
                    rescued.push(*fid);
                }
            }
        }
        if !rescued.is_empty() {
            let mut set = dedup.lock();
            let mut q = rq.borrow_mut();
            for fid in rescued {
                if set.insert(fid) {
                    q.push_back(fid);
                }
            }
        }
    }

    /// Re-deliver AsyncJoin results that are stored but whose delivery was
    /// lost. Returns true when something was delivered (progress possible).
    fn reconcile_stale_joins_multi(&self, queue: &QueueHandle<'_>) -> bool {
        let stale: Vec<(crate::ir::Ir::AsyncHandleId, Value)> = {
            let ew = self.event_waiters.lock();
            let mut out = Vec::new();
            for (evt, waiters) in ew.iter() {
                if waiters.is_empty() {
                    continue;
                }
                if let RuntimeEvent::AsyncJoin(id) = evt {
                    if let Some(v) = self.async_join_runtime.lock().try_get_result(*id) {
                        out.push((*id, v));
                    }
                }
            }
            out
        };
        let mut delivered = false;
        for (id, v) in stale {
            let woken = self.on_event_arrived(RuntimeEvent::AsyncJoin(id), v.clone(), queue);
            if woken > 0 {
                self.async_join_runtime.lock().remove_entry(id);
                delivered = true;
            } else {
                self.async_join_runtime.lock().set_result(id, v);
            }
        }
        delivered
    }

    /// Full engine state dump at the provable-deadlock exit (post-mortem for
    /// any future lost-wakeup report).
    fn dump_deadlock_state_multi(&self) {
        eprintln!("[DEADLOCK-EXIT] provably permanent waiters — engine state:");
        for (evt, waiters) in self.event_waiters.lock().iter() {
            if waiters.is_empty() {
                continue; // stale empty bucket (post-delivery residue)
            }
            eprintln!("  waiter evt={evt:?} frames={waiters:?}");
        }
        for (fid, f) in self.frames.lock().iter() {
            eprintln!(
                "  frame {fid:?} sg={} state={:?} suspend={:?} event={:?}",
                f.subgraph_id.0, f.state, f.suspend_state, f.suspend_event
            );
        }
        let pe = self.pending_events.lock();
        if !pe.is_empty() {
            eprintln!("  pending_events keys={:?}", pe.keys().collect::<Vec<_>>());
        }
        let pc = self.pending_completions.lock();
        if !pc.is_empty() {
            eprintln!("  pending_completions keys={:?}", pc.keys().collect::<Vec<_>>());
        }
        for line in self.async_join_runtime.lock().debug_dump() {
            eprintln!("  join: {line}");
        }
    }
}
