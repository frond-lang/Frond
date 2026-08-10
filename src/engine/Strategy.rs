//! 并发策略：LockStrategy / Single / Multi / QueueHandle + 单/多线程入口 + worker。

use super::*;
use crate::ir::Ir::*;
use crate::value::{Value, ValueArena};
use std::cell::{RefCell, RefMut};
use std::ops::DerefMut;
use parking_lot::{Condvar, Mutex as ParkingMutex, MutexGuard as ParkingMutexGuard};
use hashbrown::HashMap;
use crossbeam_deque::{Injector, Stealer, Worker as DequeWorker};
use std::sync::Arc;

// =========================================================================
// LockStrategy — 编译期锁策略（单线程 RefCell vs 多线程 ParkingMutex）
// =========================================================================

/// 锁策略：编译期决定字段包装方式（单线程 RefCell vs 多线程 ParkingMutex）
pub trait LockStrategy: 'static {
    type Mutex<T>: Lockable<T>;
}

/// 可锁定 trait：提供 lock() 方法返回 guard
pub trait Lockable<T> {
    type Guard<'a>: DerefMut<Target = T>
    where
        Self: 'a;
    fn lock(&self) -> Self::Guard<'_>;
}

// 单线程策略：RefCell（borrow flag，~2ns，无系统调用）
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

// 多线程策略：ParkingMutex（CAS，无竞争时 ~10ns）
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

/// 帧队列抽象：Single 用 RefCell<VecDeque>，Multi 用 DequeWorker
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
// impl Engine<Single> — 单线程模式
// =========================================================================

impl Engine<Single> {
    /// 创建单线程 Engine（用 RefCell 包装所有字段）
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
            result: RefCell::new(None),
            frame_pool: RefCell::new(Vec::new()),
            ready_frames: Some(RefCell::new(std::collections::VecDeque::new())),
            global_queue: None,
            wakeup: None,
            active_count: None,
            _strategy: std::marker::PhantomData,
        }
    }

    /// 单线程事件循环（替代 run_event_loop + run_entry）
    ///
    /// 空闲时策略（队列空 + 有 pending events）：
    /// - 有 pending timer：park 到最近 deadline（Condvar.wait_for）
    /// - 无 pending timer 但有 event_waiters：yield_now（等 channel/async 事件）
    /// - 无 pending 且无 waiters：panic（死锁检测）
    pub(super) fn run_single(&self) -> Value {
        let entry_sg = self.graph.entry_subgraph.expect("no entry subgraph");
        let fid = self.init_frame(entry_sg);
        let rq = self.ready_frames.as_ref().unwrap();
        rq.borrow_mut().push_back(fid);

        // 单线程 park 用的 Condvar（无需被外部唤醒，仅用于 wait_for 精确等待）
        let park_mutex = ParkingMutex::new(());
        let park_cv = Condvar::new();

        let mut loop_guard: u64 = 0;
        loop {
            loop_guard += 1;
            if loop_guard > 200000000 {
                panic!("event loop stuck: guard={}", loop_guard);
            }
            let queue = QueueHandle::Single(rq);
            // 先 pop（RefMut 在语句结束时释放），再处理空队列逻辑
            let fid = rq.borrow_mut().pop_front();
            let fid = match fid {
                Some(f) => f,
                None => {
                    // 队列空：检查 timer（check_timers → on_event_arrived → push 需要 borrow_mut）
                    self.check_timers(&queue);
                    if let Some(result) = self.result.lock().take() {
                        return result;
                    }
                    // 仍然无就绪帧：决定 park 策略
                    if rq.borrow().is_empty() {
                        let next_deadline = self.timer_runtime.lock().next_deadline();
                        let ew_empty = self.event_waiters.lock().is_empty();
                        if ew_empty && next_deadline.is_none() {
                            panic!(
                                "event loop exhausted: no ready frames and no pending events"
                            );
                        }
                        if let Some(deadline) = next_deadline {
                            // 有 pending timer：park 到 deadline（精确等待，不忙轮询）
                            let now = std::time::Instant::now();
                            let wait_dur = deadline.saturating_duration_since(now);
                            if !wait_dur.is_zero() {
                                let mut guard = park_mutex.lock();
                                park_cv.wait_for(&mut guard, wait_dur);
                            }
                        } else {
                            // 无 timer 但有 event_waiters：yield 等 channel/async/subgraph 事件
                            std::thread::yield_now();
                        }
                    }
                    continue;
                }
            };
            self.process_frame(fid, &queue);
            if let Some(result) = self.result.lock().take() {
                return result;
            }
        }
    }
}

// =========================================================================
// impl Engine<Multi> — 多 worker 模式
// =========================================================================

impl Engine<Multi> {
    /// 创建多线程 Engine（用 ParkingMutex 包装所有字段）
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
            result: ParkingMutex::new(None),
            frame_pool: ParkingMutex::new(Vec::new()),
            ready_frames: None,
            global_queue: Some(Injector::new()),
            wakeup: Some((ParkingMutex::new(()), Condvar::new())),
            active_count: Some(ParkingMutex::new(num_workers)),
            _strategy: std::marker::PhantomData,
        }
    }

    /// 多 worker 模式执行入口子图（替代 run_multi_worker）
    pub(super) fn run_multi(self: Arc<Self>) -> Value {
        let entry_sg = self.graph.entry_subgraph.expect("no entry subgraph");
        let entry_fid = self.init_frame(entry_sg);
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
// work-stealing worker 主循环（自由函数，供 run_multi 使用）
// =========================================================================

/// Worker 主循环：pop_local → try_steal → try_global → park。
fn worker_main(
    worker_id: usize,
    local_queue: DequeWorker<FrameId>,
    stealers: Vec<Stealer<FrameId>>,
    shared: Arc<Engine<Multi>>,
) {
    let mut steal_seed: u64 = worker_id as u64 ^ GOLDEN_RATIO_64;

    loop {
        // 结果已产生：退出
        if shared.result.lock().is_some() {
            return;
        }

        // 1. pop_local（LIFO，缓存友好）
        if let Some(fid) = local_queue.pop() {
            let queue = QueueHandle::Multi(&local_queue);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 2. try_steal（随机 victim，FIFO 窃取）
        if let Some(fid) = try_steal(&stealers, worker_id, &mut steal_seed) {
            let queue = QueueHandle::Multi(&local_queue);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 3. try_global（全局注入队列）
        if let Some(fid) = shared.global_queue.as_ref().unwrap().steal().success() {
            let queue = QueueHandle::Multi(&local_queue);
            shared.process_frame(fid, &queue);
            {
                let _g = shared.wakeup.as_ref().unwrap().0.lock();
                shared.wakeup.as_ref().unwrap().1.notify_one();
            }
            continue;
        }

        // 4+5. 无工作：在 wakeup 锁内减少活跃计数 + park
        // 合并 active_count 减量与 park 到同一 wakeup 锁临界区，消除 lost-wakeup 窗口：
        // notify_one 必须先获取 wakeup 锁，因此无法在减量与 wait_for 之间插入通知
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
            // park 前检查 timer（可能在 park 准备期间有 timer 到期）
            let queue = QueueHandle::Multi(&local_queue);
            shared.check_timers(&queue);
            // check_timers 可能将就绪帧推入 local_queue，需重新检查避免无效 park
            if !local_queue.is_empty() || !shared.global_queue.as_ref().unwrap().is_empty() {
                let mut active = shared.active_count.as_ref().unwrap().lock();
                *active += 1;
                continue;
            }
            // 减少活跃计数（在 wakeup 锁内，消除 lost-wakeup 窗口）
            let should_exit = {
                let mut active = shared.active_count.as_ref().unwrap().lock();
                *active -= 1;
                if *active == 0 {
                    // 最后一个活跃 worker：检查是否有 pending timer 或 event_waiters
                    let has_pending_timer = shared.timer_runtime.lock().next_deadline().is_some();
                    let has_event_waiters = !shared.event_waiters.lock().is_empty();
                    !has_pending_timer && !has_event_waiters
                } else {
                    false
                }
            };
            if should_exit {
                // 无 pending 工作：唤醒其他 parked worker 后退出
                shared.wakeup.as_ref().unwrap().1.notify_all();
                return;
            }
            // park timeout = 最近 timer deadline（无 timer 则默认 10ms）
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

/// 随机选择 victim worker 进行窃取。
fn try_steal(
    stealers: &[Stealer<FrameId>],
    worker_id: usize,
    seed: &mut u64,
) -> Option<FrameId> {
    let n = stealers.len();
    if n <= 1 {
        return None;
    }

    // xorshift64 伪随机
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
