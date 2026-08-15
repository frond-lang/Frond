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

// Multi-threaded strategy: ParkingMutex (CAS, ~10ns when uncontended).
pub struct Multi;
impl LockStrategy for Multi {
    type Mutex<T> = ParkingMutex<T>;
}
impl<T> Lockable<T> for ParkingMutex<T> {
    type Guard<'a>
        = ParkingMutexGuard<'a, T>
    where
        T: 'a;
    fn lock(&self) -> Self::Guard<'_> {
        self.lock()
    }
}

/// Frame-queue abstraction: Single uses `RefCell<VecDeque>`, Multi uses `DequeWorker`.
pub enum QueueHandle<'a> {
    Single(&'a RefCell<std::collections::VecDeque<FrameId>>),
    Multi(&'a DequeWorker<FrameId>),
}
impl QueueHandle<'_> {
    pub fn push(&self, fid: FrameId) {
        match self {
            Self::Single(q) => q.borrow_mut().push_back(fid),
            Self::Multi(q) => q.push(fid),
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
            event_waiters: RefCell::new(Vec::new()),
            pending_completions: RefCell::new(HashMap::new()),
            pending_events: RefCell::new(HashMap::new()),
            defer_frames: RefCell::new(HashSet::new()),
            defer_waiters: RefCell::new(HashMap::new()),
            result: RefCell::new(None),
            frame_pool: RefCell::new(Vec::new()),
            ready_frames: Some(RefCell::new(std::collections::VecDeque::new())),
            global_queue: None,
            wakeup: None,
            active_count: None,
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
            frames: ParkingMutex::new(HashMap::new()),
            next_frame_id: ParkingMutex::new(FrameId(0)),
            arena: ParkingMutex::new(ValueArena::new()),
            timer_runtime: ParkingMutex::new(TimerRuntime::new()),
            async_join_runtime: ParkingMutex::new(AsyncJoinRuntime::new()),
            event_waiters: ParkingMutex::new(Vec::new()),
            pending_completions: ParkingMutex::new(HashMap::new()),
            pending_events: ParkingMutex::new(HashMap::new()),
            defer_frames: ParkingMutex::new(HashSet::new()),
            defer_waiters: ParkingMutex::new(HashMap::new()),
            result: ParkingMutex::new(None),
            frame_pool: ParkingMutex::new(Vec::new()),
            ready_frames: None,
            global_queue: Some(Injector::new()),
            wakeup: Some((ParkingMutex::new(()), Condvar::new())),
            active_count: Some(ParkingMutex::new(num_workers)),
            _strategy: std::marker::PhantomData,
        }
    }

    /// Multi-worker entry point that executes the entry subgraph (replaces run_multi_worker).
    pub(super) fn run_multi(self: Arc<Self>) -> Value {
        let entry_sg = self.graph.entry_subgraph.expect("no entry subgraph");
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
            .expect("no result produced: all workers exited without completion")
    }
}

// =========================================================================
// work-stealing worker main loop (free function, used by run_multi)
// =========================================================================

/// Worker main loop: pop_local -> try_steal -> try_global -> park.
fn worker_main(
    worker_id: usize,
    local_queue: DequeWorker<FrameId>,
    stealers: Vec<Stealer<FrameId>>,
    shared: Arc<Engine<Multi>>,
) {
    let mut steal_seed: u64 = worker_id as u64 ^ GOLDEN_RATIO_64;

    loop {
        // A result has been produced: exit.
        if shared.result.lock().is_some() {
            return;
        }

        // 1. pop_local (LIFO, cache-friendly).
        if let Some(fid) = local_queue.pop() {
            let queue = QueueHandle::Multi(&local_queue);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 2. try_steal (random victim, FIFO steal).
        if let Some(fid) = try_steal(&stealers, worker_id, &mut steal_seed) {
            let queue = QueueHandle::Multi(&local_queue);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 3. try_global (global injector queue).
        if let Some(fid) = shared.global_queue.as_ref().unwrap().steal().success() {
            let queue = QueueHandle::Multi(&local_queue);
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
        {
            let mut guard = shared.wakeup.as_ref().unwrap().0.lock();
            if shared.result.lock().is_some() {
                let mut active = shared.active_count.as_ref().unwrap().lock();
                *active += 1;
                return;
            }
            if !local_queue.is_empty() || !shared.global_queue.as_ref().unwrap().is_empty() {
                let mut active = shared.active_count.as_ref().unwrap().lock();
                *active += 1;
                continue;
            }
            // Check timers before parking (a timer may have expired during park preparation).
            let queue = QueueHandle::Multi(&local_queue);
            shared.check_timers(&queue);
            // check_timers may push ready frames into local_queue; re-check to avoid a spurious park.
            if !local_queue.is_empty() || !shared.global_queue.as_ref().unwrap().is_empty() {
                let mut active = shared.active_count.as_ref().unwrap().lock();
                *active += 1;
                continue;
            }
            // Decrement the active count (inside the wakeup lock, eliminating the lost-wakeup window).
            let should_exit = {
                let mut active = shared.active_count.as_ref().unwrap().lock();
                *active -= 1;
                if *active == 0 {
                    // Last active worker: check whether there are pending timers or event_waiters.
                    let has_pending_timer = shared.timer_runtime.lock().next_deadline().is_some();
                    let has_event_waiters = !shared.event_waiters.lock().is_empty();
                    !has_pending_timer && !has_event_waiters
                } else {
                    false
                }
            };
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
