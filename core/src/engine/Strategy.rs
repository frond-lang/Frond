//! Concurrency strategies: LockStrategy / Single / Multi / QueueHandle + single-/multi-threaded
//! entry points + worker.

use super::*;
use crate::ir::Ir::*;
use crate::value::{Value, ValueArena};
use std::cell::{RefCell, RefMut};
use std::ops::DerefMut;
use parking_lot::{Condvar, Mutex as ParkingMutex, MutexGuard as ParkingMutexGuard};
use hashbrown::{HashMap, HashSet};
use crossbeam_deque::{Injector, Stealer, Worker as DequeWorker};
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

// Multi-threaded strategy: ParkingMutex (CAS, ~10ns when uncontended),
// wrapped by TracedMutex (debug holder registry; zero overhead when the
// FROND_DEBUG_LOCKTRACE flag is off).
pub struct Multi;
impl LockStrategy for Multi {
    type Mutex<T> = TracedMutex<T>;
}

pub(super) static LOCK_TRACE_ON: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
pub(super) static LOCK_HOLDERS: std::sync::OnceLock<parking_lot::Mutex<std::collections::HashMap<(usize, String), String>>> =
    std::sync::OnceLock::new();
fn lock_holders(
) -> &'static parking_lot::Mutex<std::collections::HashMap<(usize, String), String>> {
    LOCK_HOLDERS.get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()))
}

pub struct TracedMutex<T>(pub ParkingMutex<T>);
pub struct TracedGuard<'a, T: 'a> {
    inner: ParkingMutexGuard<'a, T>,
    key: (usize, String),
    traced: bool,
}
impl<T> std::ops::Deref for TracedGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { &self.inner }
}
impl<T> std::ops::DerefMut for TracedGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut T { &mut self.inner }
}
impl<T> Drop for TracedGuard<'_, T> {
    fn drop(&mut self) {
        if self.traced {
            lock_holders().lock().remove(&self.key);
        }
    }
}
impl<T> Lockable<T> for TracedMutex<T> {
    type Guard<'a>
        = TracedGuard<'a, T>
    where
        T: 'a;
    fn lock(&self) -> Self::Guard<'_> {
        let inner = self.0.lock();
        let traced = LOCK_TRACE_ON.load(std::sync::atomic::Ordering::Relaxed);
        let key = (
            self as *const _ as usize,
            format!("{:?}", std::thread::current().id()),
        );
        if traced {
            let bt = std::backtrace::Backtrace::force_capture().to_string();
            lock_holders().lock().insert(key.clone(), bt);
        }
        TracedGuard { inner, key, traced }
    }
}

/// Frame-queue abstraction: Single uses `RefCell<VecDeque>`, Multi uses `DequeWorker`.
///
/// Multi/EventLoop pushes are IDEMPOTENT: `dedup` tracks queue membership and
/// a second push of an already-queued frame is dropped. Root fix for the
/// duplicate-entry family (see `Engine::queued_dedup`).
pub enum QueueHandle<'a> {
    Single(&'a RefCell<std::collections::VecDeque<FrameId>>),
    Multi {
        worker: &'a DequeWorker<FrameId>,
        dedup: &'a ParkingMutex<HashSet<FrameId>>,
    },
    /// Deterministic event loop (FROND_EVENTLOOP=1): one FIFO deque + the
    /// same idempotence guarantee, on the engine's own ready queue.
    EventLoop {
        queue: &'a RefCell<std::collections::VecDeque<FrameId>>,
        dedup: &'a ParkingMutex<HashSet<FrameId>>,
    },
}
impl QueueHandle<'_> {
    pub fn push(&self, fid: FrameId) {
        match self {
            Self::Single(q) => q.borrow_mut().push_back(fid),
            Self::Multi { worker, dedup } => {
                // Insert-then-push under one lock: membership and queue
                // content cannot desynchronize (pops take the same lock).
                let mut set = dedup.lock();
                if set.insert(fid) {
                    worker.push(fid);
                }
            }
            Self::EventLoop { queue, dedup } => {
                let mut set = dedup.lock();
                if set.insert(fid) {
                    queue.borrow_mut().push_back(fid);
                }
            }
        }
    }
}

/// Pops the worker's local deque under the dedup lock (see QueueHandle::push).
#[inline]
fn pop_local_dedup(
    local_queue: &DequeWorker<FrameId>,
    dedup: &ParkingMutex<HashSet<FrameId>>,
) -> Option<FrameId> {
    let mut set = dedup.lock();
    let fid = local_queue.pop();
    if let Some(f) = &fid {
        set.remove(f);
    }
    fid
}

/// Steals one frame under the dedup lock (see QueueHandle::push).
#[inline]
fn steal_dedup(
    stealers: &[Stealer<FrameId>],
    worker_id: usize,
    seed: &mut u64,
    dedup: &ParkingMutex<HashSet<FrameId>>,
) -> Option<FrameId> {
    let mut set = dedup.lock();
    let fid = try_steal(stealers, worker_id, seed);
    if let Some(f) = &fid {
        set.remove(f);
    }
    fid
}

/// Steals from the global injector under the dedup lock (see QueueHandle::push).
#[inline]
fn steal_global_dedup(
    injector: &Injector<FrameId>,
    dedup: &ParkingMutex<HashSet<FrameId>>,
) -> Option<FrameId> {
    let mut set = dedup.lock();
    let fid = injector.steal().success();
    if let Some(f) = &fid {
        set.remove(f);
    }
    fid
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
            worker_count: 1,
            hang_progress: std::sync::atomic::AtomicU64::new(0),
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
            panic_payload: RefCell::new(None),
            frame_pool: RefCell::new(Vec::new()),
            ready_frames: Some(RefCell::new(std::collections::VecDeque::new())),
            global_queue: None,
            wakeup: None,
            active_count: None,
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
    /// Creates a multi-threaded Engine (wraps every field with ParkingMutex).
    pub(super) fn new_multi(graph: DataFlowGraph, num_workers: usize) -> Self {
        let graph = Arc::new(graph);
        Self {
            graph: graph.clone(),
            hang_progress: std::sync::atomic::AtomicU64::new(0),
            frames: TracedMutex(ParkingMutex::new(HashMap::new())),
            next_frame_id: TracedMutex(ParkingMutex::new(FrameId(0))),
            arena: TracedMutex(ParkingMutex::new(ValueArena::new())),
            timer_runtime: TracedMutex(ParkingMutex::new(TimerRuntime::new())),
            async_join_runtime: TracedMutex(ParkingMutex::new(AsyncJoinRuntime::new())),
            event_waiters: TracedMutex(ParkingMutex::new(std::collections::HashMap::new())),
            pending_completions: TracedMutex(ParkingMutex::new(HashMap::new())),
            pending_events: TracedMutex(ParkingMutex::new(HashMap::new())),
            defer_frames: TracedMutex(ParkingMutex::new(HashSet::new())),
            defer_waiters: TracedMutex(ParkingMutex::new(HashMap::new())),
            result: TracedMutex(ParkingMutex::new(None)),
            panic_payload: TracedMutex(ParkingMutex::new(None)),
            frame_pool: TracedMutex(ParkingMutex::new(Vec::new())),
            // The deterministic event loop (FROND_EVENTLOOP=1) runs on this
            // same engine struct; the worker pool ignores ready_frames.
            ready_frames: Some(RefCell::new(std::collections::VecDeque::new())),
            global_queue: Some(Injector::new()),
            wakeup: Some((ParkingMutex::new(()), Condvar::new())),
            active_count: Some(ParkingMutex::new(num_workers)),
            queued_dedup: Some(ParkingMutex::new(HashSet::new())),
            // Collection gate: sound on ONE thread (roots are complete since
            // the M1 collector fix); the loop executes on the caller thread.
            worker_count: if super::env_flag("FROND_EVENTLOOP") { 1 } else { num_workers },
            _strategy: std::marker::PhantomData,
        }
    }

    /// Multi-worker entry point that executes the entry subgraph (replaces run_multi_worker).
    pub(super) fn run_multi(self: Arc<Self>) -> Value {
        // M2 gray-release: the deterministic single-threaded event loop.
        // Same frame protocol, same queue idempotence, same stash/reconciler
        // safety nets; executed cooperatively on the caller's thread.
        if super::env_flag("FROND_EVENTLOOP") {
            return self.run_event_loop_multi();
        }
        let entry_sg = self.graph.entry_subgraph.expect("no entry subgraph");
        LOCK_TRACE_ON.store(super::env_flag("FROND_DEBUG_LOCKTRACE"), std::sync::atomic::Ordering::Relaxed);
        // Print every panic (message + location) the moment it fires, before
        // unwinding touches any engine lock: the poison path used to swallow
        // the payload, hiding the root cause of the await-loop hang.
        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic payload".to_string()
            };
            let loc = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_else(|| "<unknown>".to_string());
            eprintln!(
                "[panic] thread={:?} {} at {}",
                std::thread::current().id(),
                msg,
                loc
            );
            default_hook(info);
        }));
        let entry_fid = self.init_entry_frame(entry_sg);
        let num_workers = *self.active_count.as_ref().unwrap().lock();
        let mut local_queues: Vec<DequeWorker<FrameId>> = Vec::with_capacity(num_workers);
        let mut stealers: Vec<Stealer<FrameId>> = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let w = DequeWorker::new_lifo();
            stealers.push(w.stealer());
            local_queues.push(w);
        }

        self.global_queue.as_ref().unwrap().push(entry_fid);
        // Hang watchdog (FROND_DEBUG_HANG=1): if no frame makes progress for
        // 5s while no result, dump full engine state and abort — the await-loop
        // intermittent hang reproducer needed a post-mortem view.
        if super::env_flag("FROND_DEBUG_HANG") {
            let eng = self.clone();
            std::thread::spawn(move || {
                let mut last = 0u64;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    let cur = eng.hang_progress.load(std::sync::atomic::Ordering::Relaxed);
                    let finished = eng.result.0.try_lock().map(|g| g.is_some()).unwrap_or(false)
                        || eng
                            .panic_payload
                            .0
                            .try_lock()
                            .map(|g| g.is_some())
                            .unwrap_or(false);
                    if finished {
                        return;
                    }
                    if cur == last {
                        eprintln!("[HANG-WATCH] no progress in 5s — engine state dump:");
                        // Non-blocking probes: a wedged lock reports STUCK instead of
                        // hanging the watchdog itself (which would lose the whole dump).
                        let timeout = std::time::Duration::from_millis(250);
                        match eng.panic_payload.0.try_lock_for(timeout) {
                            Some(p) => eprintln!(
                                "  panic_payload={:?}",
                                p.as_deref().map(|s| &s[..s.len().min(200)])
                            ),
                            None => eprintln!("  panic_payload: LOCK STUCK (holder never releases)"),
                        }
                        match eng.active_count.as_ref().unwrap().try_lock_for(timeout) {
                            Some(a) => eprintln!("  active_count={}", *a),
                            None => eprintln!("  active_count: LOCK STUCK"),
                        }
                        match eng.event_waiters.0.try_lock_for(timeout) {
                            Some(ew) => eprintln!(
                                "  event_waiters={:?}",
                                ew.iter().collect::<Vec<_>>()
                            ),
                            None => eprintln!("  event_waiters: LOCK STUCK"),
                        }
                        match eng.pending_events.0.try_lock_for(timeout) {
                            Some(pe) => eprintln!("  pending_events={:?}", pe.keys().collect::<Vec<_>>()),
                            None => eprintln!("  pending_events: LOCK STUCK"),
                        }
                        match eng.pending_completions.0.try_lock_for(timeout) {
                            Some(pc) => {
                                eprintln!("  pending_completions={:?}", pc.keys().collect::<Vec<_>>())
                            }
                            None => eprintln!("  pending_completions: LOCK STUCK"),
                        }
                        match eng.async_join_runtime.0.try_lock_for(timeout) {
                            Some(jr) => {
                                let join_dump: Vec<String> = jr.debug_dump();
                                eprintln!("  join_entries={}", join_dump.join(","));
                            }
                            None => eprintln!("  async_join_runtime: LOCK STUCK"),
                        }
                        match eng.frames.0.try_lock_for(timeout) {
                            Some(frames) => {
                                for (fid, f) in frames.iter() {
                                    eprintln!(
                                        "  frame {:?} sg={:?} state={:?} suspend={:?} event={:?}",
                                        fid, f.subgraph_id, f.state, f.suspend_state, f.suspend_event
                                    );
                                }
                            }
                            None => eprintln!("  frames: LOCK STUCK"),
                        }
                        match eng.result.0.try_lock_for(timeout) {
                            Some(r) => eprintln!("  result_set={}", r.is_some()),
                            None => eprintln!("  result: LOCK STUCK"),
                        }
                        eprintln!("  [watch] lock holders (ptr, thread) -> backtrace:");
                        for (k, v) in lock_holders().lock().iter() {
                            eprintln!("   {:?}", k);
                            eprintln!("   {}", v);
                        }
                        eprintln!("[HANG-WATCH] dump complete — aborting");
                        std::process::exit(2);
                    }
                    last = cur;
                }
            });
        }

        std::thread::scope(|s| {
            for (worker_id, local_queue) in local_queues.into_iter().enumerate() {
                let shared = self.clone();
                let stealers = stealers.clone();
                s.spawn(move || {
                    worker_main(worker_id, local_queue, stealers, shared);
                });
            }
        });

        self.result
            .lock()
            .take()
            .unwrap_or_else(|| {
                // A worker panicked (poison) — re-panic with the ORIGINAL
                // message so the caller's catch_unwind reports the true cause
                // instead of a generic "no result".
                let msg = self.panic_payload.lock().take().unwrap_or_else(|| {
                    "no result produced: all workers exited without completion".to_string()
                });
                panic!("{}", msg);
            })
    }
}

// =========================================================================
// Deterministic event loop (FROND_EVENTLOOP=1) — M2 gray-release path
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
                            // or a timer (none pending). Dump and exit loudly.
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

// =========================================================================
// work-stealing worker main loop (free function, used by run_multi)
// =========================================================================

/// Worker main loop: pop_local -> try_steal -> try_global -> park.
/// Wrapped in catch_unwind so a panicking worker POISONS the engine (survivors
/// see `panic_payload` and exit) instead of parking forever: async programs
/// keep `event_waiters` non-empty, so the all-workers-idle exit never fires
/// once one worker is gone — the process used to hang with no diagnostic.
fn worker_main(
    worker_id: usize,
    local_queue: DequeWorker<FrameId>,
    stealers: Vec<Stealer<FrameId>>,
    shared: Arc<Engine<Multi>>,
) {
    let shared_for_loop = shared.clone();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        worker_loop(worker_id, local_queue, stealers, shared_for_loop)
    }));
    if let Err(payload) = outcome {
        {
            let mut p = shared.panic_payload.lock();
            if p.is_none() {
                *p = Some(crate::pass::Optimizer::panic_payload_message(&payload));
            }
        }
        // Wake every parked worker so the loop-start check observes the poison.
        let _g = shared.wakeup.as_ref().unwrap().0.lock();
        shared.wakeup.as_ref().unwrap().1.notify_all();
    }
}

fn worker_loop(
    worker_id: usize,
    local_queue: DequeWorker<FrameId>,
    stealers: Vec<Stealer<FrameId>>,
    shared: Arc<Engine<Multi>>,
) {
    let mut steal_seed: u64 = worker_id as u64 ^ GOLDEN_RATIO_64;

    loop {
        // A result has been produced — or a co-worker poisoned the engine: exit.
        if shared.result.lock().is_some() || shared.panic_payload.lock().is_some() {
            return;
        }

        // 1. pop_local (LIFO, cache-friendly).
        if let Some(fid) = pop_local_dedup(&local_queue, shared.queued_dedup.as_ref().unwrap()) {
            let queue = QueueHandle::Multi { worker: &local_queue, dedup: shared.queued_dedup.as_ref().unwrap() };
            shared.hang_progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 2. try_steal (random victim, FIFO steal).
        if let Some(fid) = steal_dedup(&stealers, worker_id, &mut steal_seed, shared.queued_dedup.as_ref().unwrap()) {
            let queue = QueueHandle::Multi { worker: &local_queue, dedup: shared.queued_dedup.as_ref().unwrap() };
            shared.hang_progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 3. try_global (global injector queue).
        if let Some(fid) = steal_global_dedup(shared.global_queue.as_ref().unwrap(), shared.queued_dedup.as_ref().unwrap()) {
            let queue = QueueHandle::Multi { worker: &local_queue, dedup: shared.queued_dedup.as_ref().unwrap() };
            shared.hang_progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 4+5. No work: decrement the active count and park within the wakeup lock.
        // Combining the active_count decrement and the park into the same wakeup-lock critical
        // section eliminates the lost-wakeup window: notify_one must acquire the wakeup lock first,
        // so a notification cannot slip between the decrement and wait_for.
        //
        // active_count bookkeeping (HANG-class fix, 2026-09): the counter means
        // "workers not parked". A worker entering this block is still ACTIVE — it
        // only decrements below, immediately before parking. The early-exit /
        // re-check arms therefore must NOT touch the counter: the three
        // `*active += 1` here were unpaired increments that permanently inflated
        // the count, after which `*active == 0` (the quiescence rescue sweep for
        // stranded Ready frames AND the all-idle deadlock exit) could never fire
        // again — lost wakeups degenerated from 10ms self-heals into permanent
        // hangs (the CI-flaky await_loop class).
        {
            let mut guard = shared.wakeup.as_ref().unwrap().0.lock();
            if shared.result.lock().is_some() || shared.panic_payload.lock().is_some() {
                return;
            }
            if !local_queue.is_empty() || !shared.global_queue.as_ref().unwrap().is_empty() {
                continue;
            }
            // Check timers before parking (a timer may have expired during park preparation).
            let queue = QueueHandle::Multi { worker: &local_queue, dedup: shared.queued_dedup.as_ref().unwrap() };
            shared.check_timers(&queue);
            // check_timers may push ready frames into local_queue; re-check to avoid a spurious park.
            if !local_queue.is_empty() || !shared.global_queue.as_ref().unwrap().is_empty() {
                continue;
            }
            // Decrement the active count (inside the wakeup lock, eliminating the lost-wakeup window).
            let mut rescued: Vec<FrameId> = Vec::new();
            let should_exit = {
                let mut active = shared.active_count.as_ref().unwrap().lock();
                *active -= 1;
                if *active == 0 {
                    // Last active worker — quiescence sweep. At this instant every
                    // other worker is parked and parked workers hold empty local
                    // queues, and this worker's own queue was just checked empty:
                    // ALL queues are empty, so any frame sitting in the map with
                    // state == Ready is a LOST WAKEUP (no queue entry will ever run
                    // it — the third await-loop hang class). Requeue every such
                    // frame and keep scheduling instead of parking. Requeuing a
                    // cached loop body is equally safe: its caller wiring routes
                    // completion through the normal LoopBody protocol, which wakes
                    // a suspended loop frame. The 792-style transient
                    // insert-before-push window cannot coincide with this sweep —
                    // that window exists only while some worker is mid-dispatch,
                    // and here every other worker is parked.
                    {
                        // Stash stranding check covers BOTH stash tables: a
                        // completion (or event) that arrived while the frame was
                        // out of the map, landing AFTER the frame's own dispatch
                        // already drained, leaves the frame Suspended in-map with
                        // no queue entry — only a re-dispatch (whose Suspended
                        // branch drains stashes) can consume it. The sweep
                        // originally covered pending_events only; a stashed
                        // COMPLETION stranded its frame identically (verified by
                        // deadlock-exit dumps: pending_completions=[child] while
                        // the parent sat Suspended forever).
                        let mut stashed: std::collections::HashSet<FrameId> = {
                            let pe = shared.pending_events.lock();
                            pe.keys().copied().collect()
                        };
                        {
                            let pc = shared.pending_completions.lock();
                            stashed.extend(pc.keys().copied());
                        }
                        let frames = shared.frames.lock();
                        for (fid, f) in frames.iter() {
                            if f.state == FrameState::Ready {
                                rescued.push(*fid);
                            } else if f.state == FrameState::Suspended
                                && stashed.contains(&fid)
                            {
                                // Suspended with an undelivered stashed event and
                                // no queue entry: its consuming drain only runs
                                // inside process_frame, which will never happen.
                                // Requeue so the Suspended branch drains the
                                // stash (a stale stash is dropped there and the
                                // frame settles — no rescue loop). Defer-waiters
                                // are excluded: their early-return path never
                                // drains and would loop.
                                let is_defer_waiter =
                                    shared.defer_waiters.lock().contains_key(fid);
                                if !is_defer_waiter {
                                    rescued.push(*fid);
                                }
                            }
                        }
                    }
                    if !rescued.is_empty() {
                        *active += 1;
                        false
                    } else {
                        // Reconciler at true quiescence: re-verify every
                        // registered waiter's event source. An AsyncJoin whose
                        // result is stored but whose delivery was lost (the
                        // completion fired while the waiter was mid-registration
                        // — woken==0 — and the registration-side repoll also
                        // missed it) is re-delivered here. At this instant every
                        // other worker is parked, so the delivery races nothing.
                        let mut redelivered = false;
                        let stale_joins: Vec<(crate::ir::Ir::AsyncHandleId, Value)> = {
                            let ew = shared.event_waiters.lock();
                            let mut out = Vec::new();
                            for (evt, waiters) in ew.iter() {
                                if waiters.is_empty() {
                                    continue;
                                }
                                if let RuntimeEvent::AsyncJoin(id) = evt {
                                    // try_get_result CONSUMES the entry; the
                                    // value is re-delivered right after (or
                                    // restored in the defensive arm below).
                                    if let Some(v) = shared
                                        .async_join_runtime
                                        .lock()
                                        .try_get_result(*id)
                                    {
                                        out.push((*id, v));
                                    }
                                }
                            }
                            out
                        };
                        for (id, v) in stale_joins {
                            let woken = shared.on_event_arrived(
                                RuntimeEvent::AsyncJoin(id),
                                v.clone(),
                                &queue,
                            );
                            if woken > 0 {
                                shared.async_join_runtime.lock().remove_entry(id);
                                redelivered = true;
                            } else {
                                // Defensive (quiescent: cannot race): restore the
                                // consumed result for a future poll.
                                shared.async_join_runtime.lock().set_result(id, v);
                            }
                        }
                        if redelivered {
                            *active += 1;
                            false
                        } else {
                            // Exit whenever no timer is pending. If waiters
                            // remain, they are PROVABLY permanent: every event
                            // source is a frame (none runnable — the sweep found
                            // nothing) or a timer (none pending). Exiting makes
                            // run_multi fail loudly with a full state dump for
                            // the post-mortem, instead of parking forever.
                            let has_pending_timer =
                                shared.timer_runtime.lock().next_deadline().is_some();
                            if !has_pending_timer {
                                eprintln!("[DEADLOCK-EXIT] provably permanent waiters — engine state:");
                                for (evt, waiters) in shared.event_waiters.lock().iter() {
                                    eprintln!("  waiter evt={evt:?} frames={waiters:?}");
                                }
                                for (fid, f) in shared.frames.lock().iter() {
                                    eprintln!(
                                        "  frame {fid:?} sg={} state={:?} suspend={:?} event={:?} caller={:?}",
                                        f.subgraph_id.0, f.state, f.suspend_state, f.suspend_event, f.caller
                                    );
                                }
                                let pe = shared.pending_events.lock();
                                if !pe.is_empty() {
                                    eprintln!("  pending_events keys={:?}", pe.keys().collect::<Vec<_>>());
                                }
                                let pc = shared.pending_completions.lock();
                                if !pc.is_empty() {
                                    eprintln!("  pending_completions keys={:?}", pc.keys().collect::<Vec<_>>());
                                }
                                let dw = shared.defer_waiters.lock();
                                if !dw.is_empty() {
                                    eprintln!("  defer_waiters={:?}", dw.iter().collect::<Vec<_>>());
                                }
                                for line in shared.async_join_runtime.lock().debug_dump() {
                                    eprintln!("  join: {line}");
                                }
                            }
                            !has_pending_timer
                        }
                    }
                } else {
                    false
                }
            };
            if !rescued.is_empty() {
                for fid in rescued {
                    local_queue.push(fid);
                }
                continue;
            }
            if should_exit {
                // No pending work: wake the other parked workers, then exit.
                shared.wakeup.as_ref().unwrap().1.notify_all();
                return;
            }
            // park timeout = nearest timer deadline (defaults to 10ms when there is no timer).
            let park_timeout = shared.timer_runtime.lock().next_deadline()
                .map(|deadline| {
                    let now = std::time::Instant::now();
                    deadline.saturating_duration_since(now)
                })
                .filter(|d| !d.is_zero())
                .unwrap_or_else(|| std::time::Duration::from_millis(10));
            shared.wakeup.as_ref().unwrap().1.wait_for(&mut guard, park_timeout);
        }
        {
            let mut active = shared.active_count.as_ref().unwrap().lock();
            *active += 1;
        }
    }
}

/// Randomly selects a victim worker to steal from.
fn try_steal(
    stealers: &[Stealer<FrameId>],
    worker_id: usize,
    seed: &mut u64,
) -> Option<FrameId> {
    let n = stealers.len();
    if n <= 1 {
        return None;
    }

    // xorshift64 pseudo-random.
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    let start = (*seed % n as u64) as usize;

    for i in 0..n {
        let idx = (start + i) % n;
        if idx == worker_id {
            continue;
        }
        if let Some(fid) = stealers[idx].steal().success() {
            return Some(fid);
        }
    }
    None
}
