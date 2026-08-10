//! Async runtime + event handling: TimerRuntime / AsyncJoinRuntime + event arrival/cancel/timer
//! checks.

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::Value;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Name of the `duration` field in a Timer event Record.
const TIMER_DURATION_NS_FIELD: &str = "duration_ns";

// =========================================================================
// TimerRuntime / AsyncJoinRuntime — out-of-graph runtimes
// =========================================================================

/// Timer runtime: a min-heap managing timer deadlines + firing checks.
///
/// See spec 3.5 EventSource::Timer. The event loop checks for expired timers on each iteration.
/// The heap top is the earliest-expiring entry; `next_deadline()` lets the event loop compute the
/// park timeout.
pub struct TimerRuntime {
    /// Min-heap (Reverse makes BinaryHeap behave as a min-heap).
    heap: BinaryHeap<Reverse<TimerHeapEntry>>,
    /// Monotonically-increasing id allocator (no longer a Vec index, so entries can be popped and
    /// re-pushed).
    next_id: u32,
    /// Set of fired ids that have not yet been queried by is_fired (lazily cleaned).
    fired_set: std::collections::HashSet<crate::ir::Ir::TimerId>,
}

struct TimerHeapEntry {
    deadline: std::time::Instant,
    id: crate::ir::Ir::TimerId,
}

impl PartialEq for TimerHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline
    }
}
impl Eq for TimerHeapEntry {}
impl PartialOrd for TimerHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TimerHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Ascending by deadline (min-heap: earliest expiry at the top).
        self.deadline.cmp(&other.deadline)
    }
}

impl TimerRuntime {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_id: 0,
            fired_set: std::collections::HashSet::new(),
        }
    }
    pub fn start(&mut self, duration: std::time::Duration) -> crate::ir::Ir::TimerId {
        assert!(self.next_id < u32::MAX, "TimerId overflow: too many timers");
        let id = crate::ir::Ir::TimerId(self.next_id);
        self.next_id += 1;
        self.heap.push(Reverse(TimerHeapEntry {
            deadline: std::time::Instant::now() + duration,
            id,
        }));
        id
    }
    /// Checks for expired timers, pops them, and returns the list of fired TimerIds.
    /// Returns immediately when the heap top has not expired (O(log n) pop).
    pub fn check_and_fire(&mut self) -> Vec<crate::ir::Ir::TimerId> {
        let now = std::time::Instant::now();
        let mut fired = Vec::new();
        while let Some(Reverse(entry)) = self.heap.peek() {
            if entry.deadline > now {
                break;
            }
            let Reverse(entry) = self.heap.pop().unwrap();
            fired.push(entry.id);
        }
        // Record into fired_set for is_fired queries.
        for id in &fired {
            self.fired_set.insert(*id);
        }
        fired
    }
    /// Checks whether a timer has fired; if so, consumes (removes) the entry.
    /// Consuming reads prevent fired_set from growing unbounded.
    pub fn is_fired(&mut self, id: crate::ir::Ir::TimerId) -> bool {
        self.fired_set.remove(&id)
    }
    /// Returns the nearest timer deadline (for the event loop to compute the park timeout).
    pub fn next_deadline(&self) -> Option<std::time::Instant> {
        self.heap.peek().map(|Reverse(e)| e.deadline)
    }
    /// Clears fired_set (all fired-timer events have already been dispatched via check_timers ->
    /// on_event_arrived).
    pub fn cleanup(&mut self) {
        self.fired_set.clear();
    }
}

impl Default for TimerRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// AsyncJoin runtime: manages the async-call -> AsyncHandle mapping + completion results.
///
/// When an async function call starts a child frame, it registers async_id -> child_fid. When the
/// child frame completes, it sets the result and fires an AsyncJoin event to wake the awaiting
/// frame.
///
/// Uses dual HashMaps for O(1) bidirectional lookup: async_id -> entry + child_fid -> async_id.
/// FrameIds are monotonically increasing and never reused, so child_index has no collision risk.
pub struct AsyncJoinRuntime {
    entries: std::collections::HashMap<crate::ir::Ir::AsyncHandleId, AsyncJoinEntry>,
    child_index: std::collections::HashMap<FrameId, crate::ir::Ir::AsyncHandleId>,
    next_async_id: u32,
}
struct AsyncJoinEntry {
    child_fid: FrameId,
    result: Option<Value>,
}
impl AsyncJoinRuntime {
    pub fn new() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
            child_index: std::collections::HashMap::new(),
            next_async_id: 0,
        }
    }
    /// Atomically allocates an async_id and registers child_fid (eliminating the race window
    /// between alloc_id + register).
    pub fn alloc_and_register(&mut self, child_fid: FrameId) -> crate::ir::Ir::AsyncHandleId {
        let async_id = crate::ir::Ir::AsyncHandleId(self.next_async_id);
        self.next_async_id += 1;
        self.child_index.insert(child_fid, async_id);
        self.entries.insert(async_id, AsyncJoinEntry { child_fid, result: None });
        async_id
    }
    pub fn find_by_child(&self, child_fid: FrameId) -> Option<crate::ir::Ir::AsyncHandleId> {
        // Only return incomplete (result=None) entries: a completed old entry's child_fid mapping
        // may not have been cleaned up yet, so the result state must be double-checked.
        let async_id = self.child_index.get(&child_fid)?;
        let entry = self.entries.get(async_id)?;
        if entry.result.is_none() { Some(*async_id) } else { None }
    }
    pub fn find_child_by_async_id(&self, async_id: crate::ir::Ir::AsyncHandleId) -> Option<FrameId> {
        self.entries.get(&async_id).map(|e| e.child_fid)
    }
    /// Tries to fetch an async result. If the result is ready, consumes (removes) the entry.
    /// Consuming reads prevent entries from growing unbounded.
    pub fn try_get_result(&mut self, async_id: crate::ir::Ir::AsyncHandleId) -> Option<Value> {
        let entry = self.entries.get(&async_id)?;
        if entry.result.is_none() {
            return None;
        }
        let entry = self.entries.remove(&async_id)?;
        self.child_index.remove(&entry.child_fid);
        entry.result
    }
    pub fn set_result(&mut self, async_id: crate::ir::Ir::AsyncHandleId, value: Value) {
        if let Some(e) = self.entries.get_mut(&async_id) {
            e.result = Some(value);
        }
    }
    /// Removes the entry for the given async_id (the waiter has already been woken by
    /// on_event_arrived and the value injected).
    pub fn remove_entry(&mut self, async_id: crate::ir::Ir::AsyncHandleId) {
        if let Some(entry) = self.entries.remove(&async_id) {
            self.child_index.remove(&entry.child_fid);
        }
    }
}

impl Default for AsyncJoinRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// EventSource trait — event-source abstraction
//
// Unifies the "decode + atomically check readiness" semantics of await event sources, eliminating
// the three-way EventSourceKind branch inside resolve_check_and_register_await. Adding a new event
// source = adding a unit struct + impl + one dispatch line.
//
// Each implementation is responsible for:
// 1. Decoding source-specific data from PendingAwait.
// 2. Checking readiness inside the source-specific lock (the lock is released before returning, to
//    avoid a lock-ordering conflict with the event_waiters lock).
// 3. Returning (event, ready_value): ready_value=Some means ready; None means a waiter must be
//    registered.
//
// Waiter registration is performed uniformly by the caller, eliminating the three-way duplicated
// push.
// =========================================================================

/// Event-source trait: unifies decode + readiness check for await event sources.
trait EventSource<S: LockStrategy> {
    fn resolve(
        &self,
        engine: &Engine<S>,
        pending: &crate::ir::Ir::PendingAwait,
    ) -> (RuntimeEvent, Option<Value>);
}

struct AsyncJoinSource;
struct ChannelSource;
struct TimerSource;

impl<S: LockStrategy> EventSource<S> for AsyncJoinSource {
    fn resolve(
        &self,
        engine: &Engine<S>,
        pending: &crate::ir::Ir::PendingAwait,
    ) -> (RuntimeEvent, Option<Value>) {
        let async_id = crate::ir::Ir::AsyncHandleId(pending.event_obj.as_i32() as u32);
        let event = RuntimeEvent::AsyncJoin(async_id);
        // try_get_result is a consuming read: if the result is ready it removes the entry and
        // returns the value.
        // The async_join_runtime lock is a temporary; released at end of statement. The
        // event_waiters lock is taken separately by the caller, so there is no nesting.
        let val = engine.async_join_runtime.lock().try_get_result(async_id);
        (event, val)
    }
}

impl<S: LockStrategy> EventSource<S> for ChannelSource {
    fn resolve(
        &self,
        _engine: &Engine<S>,
        pending: &crate::ir::Ir::PendingAwait,
    ) -> (RuntimeEvent, Option<Value>) {
        let ch = pending
            .event_obj
            .heap_obj()
            .and_then(|h| h.channel())
            .expect("await on non-channel value");
        // recv failed but the channel is closed -> inject Null; otherwise None (register a waiter).
        let v = ch
            .recv()
            .or_else(|| if ch.is_closed() { Some(Value::Null) } else { None });
        let event = RuntimeEvent::ChannelReady(crate::ir::Ir::ChannelId(ch.id()));
        (event, v)
    }
}

impl<S: LockStrategy> EventSource<S> for TimerSource {
    fn resolve(
        &self,
        engine: &Engine<S>,
        pending: &crate::ir::Ir::PendingAwait,
    ) -> (RuntimeEvent, Option<Value>) {
        let duration_ns = match pending.event_obj.heap_obj() {
            Some(crate::value::HeapObj::Record(r)) => {
                r.find_field(TIMER_DURATION_NS_FIELD)
                    .map(|v| v.as_i64())
                    .expect("timer event record missing duration_ns field")
            }
            _ => pending.event_obj.as_i64(),
        };
        // start + is_fired are atomicized inside the timer_runtime lock (check_and_fire uses the
        // same lock). Explicitly drop the timer lock before the caller registers the waiter (to
        // avoid a lock-ordering conflict with the event_waiters lock).
        let mut tr = engine.timer_runtime.lock();
        let timer_id = tr.start(std::time::Duration::from_nanos(duration_ns as u64));
        let event = RuntimeEvent::TimerFired(timer_id);
        let fired = tr.is_fired(timer_id);
        drop(tr);
        let val = if fired { Some(Value::VOID) } else { None };
        (event, val)
    }
}

// =========================================================================
// impl<S: LockStrategy> Engine<S> — event-handling methods
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// Resolves the await event source, atomically checks readiness, and registers a waiter
    /// (eliminating the TOCTOU race).
    ///
    /// Returns (event, ready_value, await_node_local):
    /// - ready_value = Some(v): the event is ready; the caller injects the value and continues
    ///   execution.
    /// - ready_value = None: the event is not ready; a waiter has been registered and the caller
    ///   only needs to set the frame state and return.
    ///
    /// Event-source decode + readiness check are delegated to the EventSource trait; waiter
    /// registration is performed uniformly here. Each source-specific lock is released inside
    /// EventSource::resolve, so the event_waiters lock never nests with a source lock.
    pub(super) fn resolve_check_and_register_await(
        &self,
        pending: &crate::ir::Ir::PendingAwait,
        fid: FrameId,
    ) -> (RuntimeEvent, Option<Value>, crate::ir::Ir::NodeId) {
        use crate::ir::Ir::EventSourceKind;
        let await_node = pending.await_node_local;
        let (event, val) = match pending.event_kind {
            EventSourceKind::AsyncJoin => AsyncJoinSource.resolve(self, pending),
            EventSourceKind::Channel => ChannelSource.resolve(self, pending),
            EventSourceKind::Timer => TimerSource.resolve(self, pending),
            EventSourceKind::SubgraphComplete => {
                panic!("SubgraphComplete should not go through await path");
            }
        };
        // Uniform waiter registration: only register when not ready. Each source-specific lock has
        // already been released inside EventSource::resolve, so the event_waiters lock here does
        // not nest with a source lock, avoiding lock-ordering conflicts.
        if val.is_none() {
            self.event_waiters.lock().push((event, fid));
        }
        (event, val, await_node)
    }

    /// Injects the event value into the waiting frame and wakes it (sets Ready + pushes to the
    /// ready queue + notifies downstream). For select frames, re-pushes the gate node (no value
    /// injected); for ordinary await frames, injects the event value. Returns true on successful
    /// handling, false if the frame is not in the WaitingEvent state (already woken by another
    /// event). Shared by on_event_arrived and the pending_events consumption in process_frame.
    pub(super) fn apply_event_to_frame(&self, frame: &mut Frame, value: Value) -> bool {
        let await_node = match frame.suspend_state {
            SuspendState::WaitingEvent(node) => node,
            _ => return false,
        };
        let node_offset = frame.node_offset;
        let await_graph_id = NodeId(await_node.0 + node_offset);

        // select frame (gate node has SelectInfo): re-push the gate node, do not inject a value.
        let is_select = self.graph.has_select_info(await_graph_id.0 as usize);
        if is_select {
            frame.state = FrameState::Ready;
            frame.suspend_state = SuspendState::NotSuspended;
            frame.suspend_event = None;
            frame.push_ready(await_node);
        } else {
            // Ordinary await frame: inject the event value into the await node.
            let consumer_count =
                self.graph.downstream_slice(await_graph_id.0 as usize).len() as u16;
            frame.set_value(await_node, value, consumer_count);
            frame.state = FrameState::Ready;
            frame.suspend_state = SuspendState::NotSuspended;
            frame.suspend_event = None;
            notify_downstream(
                frame,
                &self.graph,
                await_node,
                await_graph_id,
                NodeId(node_offset),
            );
        }
        true
    }

    /// Event arrival: injects the value into the waiting frame and wakes it. Returns the number of
    /// woken waiters.
    pub(super) fn on_event_arrived(&self, event: RuntimeEvent, value: Value, queue: &QueueHandle<'_>) -> usize {
        // Find the frames waiting on this event (short critical section).
        let waiters: Vec<FrameId> = {
            let mut event_waiters = self.event_waiters.lock();
            let waiters: Vec<FrameId> = event_waiters
                .iter()
                .filter(|(e, _)| *e == event)
                .map(|(_, fid)| *fid)
                .collect();
            // Use a HashSet to avoid O(n^2) retain (Vec::contains is O(n)).
            let waiter_set: std::collections::HashSet<FrameId> = waiters.iter().copied().collect();
            event_waiters.retain(|(_, fid)| !waiter_set.contains(fid));
            waiters
        };
        let woken = waiters.len();

        for fid in waiters {
            // Take out the frame (keep it boxed to preserve address stability).
            let mut frame_box = {
                let mut frames = self.frames.lock();
                match frames.remove(&fid) {
                    Some(b) => b,
                    None => {
                        // The frame is being processed by process_frame (not in the HashMap).
                        // Stash the event; process_frame will consume it after reinserting the
                        // frame (race fallback).
                        // The waiter has already been removed from event_waiters above, so no
                        // duplicate cleanup is needed.
                        self.pending_events.lock().insert(fid, (event, value.clone()));
                        continue;
                    }
                }
            };
            let frame: &mut Frame = &mut *frame_box;

            if !self.apply_event_to_frame(frame, value.clone()) {
                // Not an event-waiting frame (already woken by another event): put it back + skip.
                self.frames.lock().insert(fid, frame_box);
                continue;
            }

            // Put the frame back + enqueue (same Box, address unchanged).
            self.frames.lock().insert(fid, frame_box);
            queue.push(fid);
        }
        woken
    }

    /// Cancels a frame: Suspended -> Cancelling + enqueue.
    pub(super) fn cancel_frame(&self, frame_id: FrameId, queue: &QueueHandle<'_>) {
        let mut frame_box = {
            let mut frames = self.frames.lock();
            match frames.remove(&frame_id) {
                Some(b) => b,
                None => return, // Frame is being processed by another worker; skip.
            }
        };
        let frame: &mut Frame = &mut *frame_box;

        if frame.state != FrameState::Suspended {
            self.frames.lock().insert(frame_id, frame_box);
            return;
        }

        // Remove the event-waiter registration.
        if let Some(event) = frame.suspend_event {
            self.event_waiters
                .lock()
                .retain(|(e, fid)| !(*e == event && *fid == frame_id));
        } else {
            // select frame: remove all event-waiter entries for this frame.
            self.event_waiters
                .lock()
                .retain(|(_, fid)| *fid != frame_id);
        }
        // Clean up pending_events (events stashed when the frame was absent from the HashMap on
        // arrival).
        self.pending_events.lock().remove(&frame_id);

        frame.state = FrameState::Cancelling;
        frame.suspend_state = SuspendState::NotSuspended;
        frame.suspend_event = None;

        self.frames.lock().insert(frame_id, frame_box);
        queue.push(frame_id);
    }

    /// Checks for timer events.
    pub(super) fn check_timers(&self, queue: &QueueHandle<'_>) {
        let fired_timers = self.timer_runtime.lock().check_and_fire();
        for tid in &fired_timers {
            self.on_event_arrived(RuntimeEvent::TimerFired(*tid), Value::VOID, queue);
        }
        // All fired-timer events have been dispatched via on_event_arrived, so the residual entries
        // in fired_set (those not consumed by is_fired) can be safely cleaned up:
        // is_fired is only called inside the same lock as start() (checking a new timer) and never
        // queries old entries.
        if !fired_timers.is_empty() {
            self.timer_runtime.lock().cleanup();
        }
    }
}
