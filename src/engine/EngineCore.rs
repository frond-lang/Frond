//! Engine 核心类型定义：Engine<S> 结构体、EngineRef 工厂、Send/Sync 实现、
//! 调度器常量与 env_flag 辅助。
//!
//! 业务方法（impl<S: LockStrategy> Engine<S>）分散在子模块中：
//! - [`crate::engine::Frame`]: Frame 生命周期
//! - [`crate::engine::Subgraph`]: 子图调用与返回
//! - [`crate::engine::Schedule`]: 就绪调度核心
//! - [`crate::engine::AsyncRt`]: 事件处理
//! - [`crate::engine::Strategy`]: 单/多线程入口（new_single / new_multi / run_single / run_multi）

use super::*;
use crate::ir::Ir::*;
use crate::value::{Value, ValueArena};
use std::cell::RefCell;
use parking_lot::{Condvar, Mutex as ParkingMutex};
use hashbrown::HashMap;
use std::sync::OnceLock;
use crossbeam_deque::Injector;
use std::sync::Arc;

/// 缓存环境变量布尔标志，避免热路径每次调用 std::env::var。
#[inline]
pub(super) fn env_flag(name: &str) -> bool {
    static FLAG_STALL: OnceLock<bool> = OnceLock::new();
    match name {
        "KUZO_DEBUG_STALL" => *FLAG_STALL.get_or_init(|| std::env::var("KUZO_DEBUG_STALL").is_ok()),
        _ => std::env::var(name).is_ok(),
    }
}

// =========================================================================
// 哨兵常量 — 调度器使用
// =========================================================================

/// `pending_inputs` 槽位哨兵：标记"永不就绪/外部源"（实际入度必须 < 65535）。
pub(super) const PENDING_EXTERNAL: u16 = u16::MAX;
/// splitmix64 黄金比例散列常量（确保各 worker 的 steal 顺序互异）。
pub(super) const GOLDEN_RATIO_64: u64 = 0x9E3779B97F4A7C15;

// =========================================================================
// Engine<S> — 统一执行引擎（泛型锁策略）
// =========================================================================

/// 统一引擎：字段类型由 S 决定，业务逻辑只写一份
pub struct Engine<S: LockStrategy> {
    pub graph: Arc<DataFlowGraph>,
    pub frames: S::Mutex<HashMap<FrameId, Box<crate::ir::Ir::Frame>>>,
    pub next_frame_id: S::Mutex<FrameId>,
    pub arena: S::Mutex<ValueArena>,
    pub timer_runtime: S::Mutex<TimerRuntime>,
    pub async_join_runtime: S::Mutex<AsyncJoinRuntime>,
    pub event_waiters: S::Mutex<Vec<(crate::ir::Ir::RuntimeEvent, FrameId)>>,
    pub pending_completions:
        S::Mutex<HashMap<FrameId, Vec<(crate::ir::Ir::NodeId, Value, crate::ir::Ir::ControlSignal)>>>,
    /// 事件投递竞态兜底：事件到达时帧正被 process_frame 处理（不在 HashMap），
    /// 将事件暂存，process_frame insert 帧后消费（与 pending_completions 对称）
    pub pending_events: S::Mutex<HashMap<FrameId, (crate::ir::Ir::RuntimeEvent, Value)>>,
    pub result: S::Mutex<Option<Value>>,
    /// 帧池：回收已完成帧的 Box<Frame> 供复用，消除频繁 Vec 分配/释放
    pub frame_pool: S::Mutex<Vec<Box<crate::ir::Ir::Frame>>>,
    /// 单线程队列（Multi 模式为 None）
    pub ready_frames: Option<RefCell<std::collections::VecDeque<FrameId>>>,
    /// 多线程调度（Single 模式为 None）
    pub global_queue: Option<Injector<FrameId>>,
    pub wakeup: Option<(ParkingMutex<()>, Condvar)>,
    pub active_count: Option<ParkingMutex<usize>>,
    // pub(super)：结构体定义于 engine::EngineCore，需允许 engine 子树（Strategy 等 sibling）
    // 在构造 Engine { ... } 时写入此字段。
    pub(super) _strategy: std::marker::PhantomData<S>,
}

// Safety: Frame 含裸指针（root_frame_ptr/parent_frame_ptr），但所有可变字段都在
// ParkingMutex 保护下，同一时刻只有一个线程访问每个字段。
unsafe impl Send for Engine<Multi> {}
unsafe impl Sync for Engine<Multi> {}

// =========================================================================
// EngineRef — 统一工厂（根据 workers 数决定编译期策略）
// =========================================================================

/// 统一工厂：根据 workers 数决定编译期策略
pub enum EngineRef {
    Single(Engine<Single>),
    Multi(Arc<Engine<Multi>>),
}

impl EngineRef {
    /// 创建引擎：自动判断 worker 数。
    ///
    /// 策略：从 entry_subgraph 可达的子图中，任一 `has_suspend = true`
    /// （含 async/timer/channel 挂起点）时用多 worker（= available_parallelism），
    /// 否则单线程。
    /// 关键：可达性分析只看从 entry 出发能调用到的子图，避免被"已编译但未调用"
    /// 的 stdlib async 函数（如 open/read_file/sleep）误判为需要多 worker。
    /// 纯同步程序无挂起点，单线程数据流调度最高效（无 work-stealing 开销）；
    /// 含 async 的程序有挂起/唤醒，多 worker 并发推进多帧更高效。
    pub fn new(graph: DataFlowGraph) -> Self {
        let has_async = Self::entry_reaches_suspend(&graph);
        if has_async {
            let workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            Self::Multi(Arc::new(Engine::<Multi>::new_multi(graph, workers)))
        } else {
            Self::Single(Engine::<Single>::new_single(graph))
        }
    }

    /// 从 entry_subgraph 出发做可达性分析，判断是否有可达子图含挂起点。
    /// 遍历每个可达子图节点范围内的 Call 节点的 call_targets，BFS 扩展。
    fn entry_reaches_suspend(graph: &DataFlowGraph) -> bool {
        let entry = match graph.entry_subgraph {
            Some(sg) => sg,
            None => return false,
        };
        let mut visited = vec![false; graph.subgraphs.len()];
        let mut queue: std::collections::VecDeque<SubGraphId> = std::collections::VecDeque::new();
        if (entry.0 as usize) < visited.len() {
            visited[entry.0 as usize] = true;
            queue.push_back(entry);
        }
        while let Some(sg_id) = queue.pop_front() {
            let sg = &graph.subgraphs[sg_id.0 as usize];
            if sg.has_suspend {
                return true;
            }
            // 扫描该子图节点范围内的 Call 节点，收集 call_targets
            let (start, end) = sg.node_range;
            let start = start.0 as usize;
            let end = end.0 as usize;
            for nid in start..end {
                if nid >= graph.node_count() {
                    break;
                }
                // call_targets 是 per-Node 表，Call 节点才有 Some(target)
                if let Some(target_sg) = graph.call_target(nid) {
                    let t = target_sg.0 as usize;
                    if t < visited.len() && !visited[t] {
                        visited[t] = true;
                        queue.push_back(target_sg);
                    }
                }
            }
        }
        false
    }

    /// 运行引擎，返回结果值
    pub fn run(self) -> Value {
        match self {
            Self::Single(e) => e.run_single(),
            Self::Multi(e) => Engine::<Multi>::run_multi(e),
        }
    }
}
