//! 异步运行时 + 事件处理：TimerRuntime / AsyncJoinRuntime + 事件到达/取消/timer 检查。

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::Value;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// Timer 事件 Record 中 duration 字段名。
const TIMER_DURATION_NS_FIELD: &str = "duration_ns";

// =========================================================================
// TimerRuntime / AsyncJoinRuntime — 图外运行时
// =========================================================================

/// Timer 运行时：最小堆管理 timer deadline + 触发检查。
///
/// spec 3.5 EventSource::Timer。事件循环每次迭代检查到期 timer。
/// 堆顶 = 最早到期项，`next_deadline()` 供事件循环计算 park timeout。
pub struct TimerRuntime {
    /// 最小堆（Reverse 使 BinaryHeap 表现为 min-heap）
    heap: BinaryHeap<Reverse<TimerHeapEntry>>,
    /// 递增 ID 分配器（不再用 Vec 索引，允许弹出入堆）
    next_id: u32,
    /// 已触发但未被 is_fired 查询的 ID 集合（惰性清理）
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
        // 按 deadline 升序（min-heap：最早到期在堆顶）
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
    /// 检查到期 timer，弹出并返回已触发的 TimerId 列表。
    /// 堆顶未到期时立即返回（O(log n) 弹出）。
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
        // 记录到 fired_set 供 is_fired 查询
        for id in &fired {
            self.fired_set.insert(*id);
        }
        fired
    }
    /// 检查 timer 是否已触发，若是则消费（移除）该条目。
    /// 消费式读取避免 fired_set 无界增长。
    pub fn is_fired(&mut self, id: crate::ir::Ir::TimerId) -> bool {
        self.fired_set.remove(&id)
    }
    /// 返回最近到期 timer 的 deadline（供事件循环计算 park timeout）。
    pub fn next_deadline(&self) -> Option<std::time::Instant> {
        self.heap.peek().map(|Reverse(e)| e.deadline)
    }
    /// 清理 fired_set（所有已触发 timer 的事件已通过 check_timers → on_event_arrived 派发）。
    pub fn cleanup(&mut self) {
        self.fired_set.clear();
    }
}

impl Default for TimerRuntime {
    fn default() -> Self {
        Self::new()
    }
}

/// AsyncJoin 运行时：管理 async 调用 → AsyncHandle 映射 + 完成结果。
///
/// async 函数调用启动子帧时注册 async_id → child_fid。
/// 子帧完成时设置 result + 触发 AsyncJoin 事件唤醒等待的 await 帧。
///
/// 使用双 HashMap 实现 O(1) 双向查找：async_id → entry + child_fid → async_id。
/// FrameId 单调递增不复用，child_index 无冲突风险。
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
    /// 分配新的 async_id（i32 标量值）
    pub fn alloc_id(&mut self) -> crate::ir::Ir::AsyncHandleId {
        assert!(self.next_async_id < u32::MAX, "AsyncHandleId overflow: too many async calls");
        let id = crate::ir::Ir::AsyncHandleId(self.next_async_id);
        self.next_async_id += 1;
        id
    }
    pub fn register(&mut self, async_id: crate::ir::Ir::AsyncHandleId, child_fid: FrameId) {
        self.child_index.insert(child_fid, async_id);
        self.entries.insert(async_id, AsyncJoinEntry { child_fid, result: None });
    }
    /// 原子地分配 async_id 并注册 child_fid（消除 alloc_id + register 的竞态窗口）。
    pub fn alloc_and_register(&mut self, child_fid: FrameId) -> crate::ir::Ir::AsyncHandleId {
        let async_id = crate::ir::Ir::AsyncHandleId(self.next_async_id);
        self.next_async_id += 1;
        self.child_index.insert(child_fid, async_id);
        self.entries.insert(async_id, AsyncJoinEntry { child_fid, result: None });
        async_id
    }
    pub fn find_by_child(&self, child_fid: FrameId) -> Option<crate::ir::Ir::AsyncHandleId> {
        // 仅返回未完成（result=None）的 entry：已完成旧 entry 的 child_fid 映射
        // 可能尚未清理，需二次检查 result 状态。
        let async_id = self.child_index.get(&child_fid)?;
        let entry = self.entries.get(async_id)?;
        if entry.result.is_none() { Some(*async_id) } else { None }
    }
    pub fn find_child_by_async_id(&self, async_id: crate::ir::Ir::AsyncHandleId) -> Option<FrameId> {
        self.entries.get(&async_id).map(|e| e.child_fid)
    }
    /// 尝试获取 async 结果。若结果已就绪则消费（移除）该 entry。
    /// 消费式读取避免 entries 无界增长。
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
    /// 移除指定 async_id 的 entry（waiter 已被 on_event_arrived 唤醒，值已注入）。
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
// EventSource trait — 事件源抽象
//
// 统一 await 事件源的「解码 + 原子检查就绪」语义，消除 resolve_check_and_register_await
// 中按 EventSourceKind 的三路特判分支。新增事件源 = 新增一个 unit struct + impl + 一行分派。
//
// 每个实现负责：
// 1. 从 PendingAwait 解码源专属数据
// 2. 在源专属锁内检查就绪（锁在返回前释放，避免与 event_waiters 锁顺序冲突）
// 3. 返回 (event, ready_value)：ready_value=Some 表示已就绪，None 表示需注册 waiter
//
// waiter 注册由调用方统一执行，消除三路重复 push。
// =========================================================================

/// 事件源 trait：统一 await 事件源的解码 + 就绪检查。
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
        // try_get_result 消费式读取：result 已就绪则移除 entry 返回值。
        // async_join_runtime 锁为临时量，语句末释放；event_waiters 锁由调用方另取，无嵌套。
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
        // recv 失败但 channel 已关闭 → 注入 Null；否则 None（需注册 waiter）。
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
        // start + is_fired 在 timer_runtime 锁内原子化（check_and_fire 同锁），
        // 显式 drop 释放 timer 锁后再由调用方注册 waiter（避免与 event_waiters 锁顺序冲突）。
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
// impl<S: LockStrategy> Engine<S> — 事件处理方法
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// 解析 await 事件源 + 原子检查就绪并注册 waiter（消除 TOCTOU 竞态）。
    ///
    /// 返回 (event, ready_value, await_node_local)：
    /// - ready_value = Some(v)：事件已就绪，调用方直接注入值继续执行
    /// - ready_value = None：事件未就绪，waiter 已注册，调用方只需设帧状态后 return
    ///
    /// 事件源解码 + 就绪检查委托 EventSource trait；waiter 注册在此统一执行。
    /// 各源专属锁在 EventSource::resolve 内释放，event_waiters 锁不与源锁嵌套。
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
        // 统一 waiter 注册：仅未就绪时注册。源专属锁已在各 EventSource::resolve 内释放，
        // 此处 event_waiters 锁不与源锁嵌套，避免锁顺序冲突。
        if val.is_none() {
            self.event_waiters.lock().push((event, fid));
        }
        (event, val, await_node)
    }

    /// 将事件值注入等待帧并唤醒（设 Ready + 推就绪队列 + 通知下游）。
    /// select 帧重新 push gate 节点（不注入值），普通 await 帧注入事件值。
    /// 返回 true 表示成功处理，false 表示帧非 WaitingEvent 状态（已被其他事件唤醒）。
    /// 被 on_event_arrived 和 process_frame 的 pending_events 消费共用。
    pub(super) fn apply_event_to_frame(&self, frame: &mut Frame, value: Value) -> bool {
        let await_node = match frame.suspend_state {
            SuspendState::WaitingEvent(node) => node,
            _ => return false,
        };
        let node_offset = frame.node_offset;
        let await_graph_id = NodeId(await_node.0 + node_offset);

        // select 帧（gate 节点有 SelectInfo）：重新 push gate 节点，不注入值
        let is_select = self.graph.has_select_info(await_graph_id.0 as usize);
        if is_select {
            frame.state = FrameState::Ready;
            frame.suspend_state = SuspendState::NotSuspended;
            frame.suspend_event = None;
            frame.push_ready(await_node);
        } else {
            // 普通 await 帧：注入事件值到 await 节点
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

    /// 事件到达：注入值到等待帧 + 唤醒。返回被唤醒的 waiter 数量。
    pub(super) fn on_event_arrived(&self, event: RuntimeEvent, value: Value, queue: &QueueHandle<'_>) -> usize {
        // 找等待该事件的帧（短临界区）
        let waiters: Vec<FrameId> = {
            let mut event_waiters = self.event_waiters.lock();
            let waiters: Vec<FrameId> = event_waiters
                .iter()
                .filter(|(e, _)| *e == event)
                .map(|(_, fid)| *fid)
                .collect();
            // 用 HashSet 避免 O(n²) retain（Vec::contains 是 O(n)）
            let waiter_set: std::collections::HashSet<FrameId> = waiters.iter().copied().collect();
            event_waiters.retain(|(_, fid)| !waiter_set.contains(fid));
            waiters
        };
        let woken = waiters.len();

        for fid in waiters {
            // 取出帧（保持 Box 不 unbox 以维持地址稳定）
            let mut frame_box = {
                let mut frames = self.frames.lock();
                match frames.remove(&fid) {
                    Some(b) => b,
                    None => {
                        // 帧正被 process_frame 处理（不在 HashMap）。
                        // 暂存事件，process_frame insert 帧后消费（竞态兜底）。
                        // waiter 已在上方从 event_waiters 移除，无需重复清理。
                        self.pending_events.lock().insert(fid, (event, value.clone()));
                        continue;
                    }
                }
            };
            let frame: &mut Frame = &mut *frame_box;

            if !self.apply_event_to_frame(frame, value.clone()) {
                // 非事件等待帧（已被其他事件唤醒）：放回 + 跳过
                self.frames.lock().insert(fid, frame_box);
                continue;
            }

            // 放回帧 + 入队（同一个 Box，地址不变）
            self.frames.lock().insert(fid, frame_box);
            queue.push(fid);
        }
        woken
    }

    /// 取消帧：Suspended → Cancelling + 入就绪队列。
    pub(super) fn cancel_frame(&self, frame_id: FrameId, queue: &QueueHandle<'_>) {
        let mut frame_box = {
            let mut frames = self.frames.lock();
            match frames.remove(&frame_id) {
                Some(b) => b,
                None => return, // 帧正被其他 worker 处理，跳过
            }
        };
        let frame: &mut Frame = &mut *frame_box;

        if frame.state != FrameState::Suspended {
            self.frames.lock().insert(frame_id, frame_box);
            return;
        }

        // 移除事件等待注册
        if let Some(event) = frame.suspend_event {
            self.event_waiters
                .lock()
                .retain(|(e, fid)| !(*e == event && *fid == frame_id));
        } else {
            // select 帧：移除该帧所有事件等待
            self.event_waiters
                .lock()
                .retain(|(_, fid)| *fid != frame_id);
        }
        // 清理 pending_events（事件到达时帧不在 HashMap 的暂存事件）
        self.pending_events.lock().remove(&frame_id);

        frame.state = FrameState::Cancelling;
        frame.suspend_state = SuspendState::NotSuspended;
        frame.suspend_event = None;

        self.frames.lock().insert(frame_id, frame_box);
        queue.push(frame_id);
    }

    /// 检查 timer 事件
    pub(super) fn check_timers(&self, queue: &QueueHandle<'_>) {
        let fired_timers = self.timer_runtime.lock().check_and_fire();
        for tid in &fired_timers {
            self.on_event_arrived(RuntimeEvent::TimerFired(*tid), Value::VOID, queue);
        }
        // 所有已触发 timer 的事件已通过 on_event_arrived 派发，
        // fired_set 中的残余条目（is_fired 未消费的）可安全清理：
        // is_fired 仅在 start() 同锁内调用（检查新 timer），不会查询旧条目
        if !fired_timers.is_empty() {
            self.timer_runtime.lock().cleanup();
        }
    }
}
