//! Engine core type definitions: the `Engine<S>` struct, the `EngineRef` factory, Send/Sync
//! implementations, scheduler constants, and the `env_flag` helper.
//!
//! Business methods (`impl<S: LockStrategy> Engine<S>`) are spread across submodules:
//! - [`crate::engine::Frame`]: frame lifecycle
//! - [`crate::engine::Subgraph`]: subgraph invocation and return
//! - [`crate::engine::Schedule`]: readiness scheduling core
//! - [`crate::engine::AsyncRt`]: event handling
//! - [`crate::engine::Strategy`]: single-/multi-threaded entry points (new_single / new_multi / run_single / run_multi)

use super::*;
use crate::ir::Ir::*;
use crate::value::{Value, ValueArena};
use std::cell::RefCell;
use parking_lot::{Condvar, Mutex as ParkingMutex};
use hashbrown::HashMap;
use std::sync::OnceLock;
use crossbeam_deque::Injector;
use std::sync::Arc;

/// Caches a boolean environment-variable flag to avoid calling `std::env::var` on every hot-path invocation.
#[inline]
pub(super) fn env_flag(name: &str) -> bool {
    static FLAG_STALL: OnceLock<bool> = OnceLock::new();
    match name {
        "KUZO_DEBUG_STALL" => *FLAG_STALL.get_or_init(|| std::env::var("KUZO_DEBUG_STALL").is_ok()),
        _ => std::env::var(name).is_ok(),
    }
}

// =========================================================================
// Sentinel constants — used by the scheduler
// =========================================================================

/// Sentinel for a `pending_inputs` slot marking it as "never ready / external source" (the actual
/// in-degree must stay below 65535).
pub(super) const PENDING_EXTERNAL: u16 = u16::MAX;
/// splitmix64 golden-ratio hash constant (ensures each worker steals in a distinct order).
pub(super) const GOLDEN_RATIO_64: u64 = 0x9E3779B97F4A7C15;

// =========================================================================
// Engine<S> — unified execution engine (generic over lock strategy)
// =========================================================================

/// Unified engine: field types are determined by `S`, while the business logic is written once.
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
    /// Fallback for event-delivery races: when an event arrives while a frame is being processed by
    /// process_frame (and is therefore absent from the HashMap), the event is stashed here and
    /// consumed once process_frame inserts the frame (symmetric to pending_completions).
    pub pending_events: S::Mutex<HashMap<FrameId, (crate::ir::Ir::RuntimeEvent, Value)>>,
    pub result: S::Mutex<Option<Value>>,
    /// Frame pool: reclaims completed `Box<Frame>` for reuse, eliminating frequent Vec
    /// allocation/deallocation.
    pub frame_pool: S::Mutex<Vec<Box<crate::ir::Ir::Frame>>>,
    /// Single-threaded queue (None in Multi mode).
    pub ready_frames: Option<RefCell<std::collections::VecDeque<FrameId>>>,
    /// Multi-threaded scheduling (None in Single mode).
    pub global_queue: Option<Injector<FrameId>>,
    pub wakeup: Option<(ParkingMutex<()>, Condvar)>,
    pub active_count: Option<ParkingMutex<usize>>,
    // pub(super): the struct is defined in engine::EngineCore; sibling submodules (Strategy, etc.)
    // must be allowed to write this field when constructing `Engine { ... }`.
    pub(super) _strategy: std::marker::PhantomData<S>,
}

// Safety: Frame contains raw pointers (root_frame_ptr/parent_frame_ptr), but every mutable field
// is guarded by a ParkingMutex, so only one thread accesses each field at a time.
unsafe impl Send for Engine<Multi> {}
unsafe impl Sync for Engine<Multi> {}

// =========================================================================
// EngineRef — unified factory (picks a compile-time strategy based on the worker count)
// =========================================================================

/// Unified factory: picks a compile-time strategy based on the worker count.
pub enum EngineRef {
    Single(Engine<Single>),
    Multi(Arc<Engine<Multi>>),
}

impl EngineRef {
    /// Creates the engine, automatically deciding the worker count.
    ///
    /// Strategy: among the subgraphs reachable from `entry_subgraph`, if any has
    /// `has_suspend = true` (i.e. contains an async/timer/channel suspension point), use multiple
    /// workers (= `available_parallelism`); otherwise use a single thread.
    /// Key point: the reachability analysis only considers subgraphs callable from the entry, so
    /// "compiled-but-uncalled" stdlib async functions (such as open/read_file/sleep) are not
    /// mistaken as requiring multiple workers.
    /// A purely synchronous program has no suspension points, so single-threaded dataflow
    /// scheduling is most efficient (no work-stealing overhead); a program containing async has
    /// suspend/wake behavior, so multiple workers advancing frames concurrently is more efficient.
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

    /// Performs a reachability analysis starting from `entry_subgraph` to determine whether any
    /// reachable subgraph contains a suspension point. Traverses the `call_targets` of every Call
    /// node within each reachable subgraph's node range, expanding via BFS.
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
            // Scan the Call nodes within this subgraph's node range and collect call_targets.
            let (start, end) = sg.node_range;
            let start = start.0 as usize;
            let end = end.0 as usize;
            for nid in start..end {
                if nid >= graph.node_count() {
                    break;
                }
                // call_targets is a per-Node table; only Call nodes have Some(target).
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

    /// Runs the engine and returns the result value.
    pub fn run(self) -> Value {
        match self {
            Self::Single(e) => e.run_single(),
            Self::Multi(e) => Engine::<Multi>::run_multi(e),
        }
    }
}
