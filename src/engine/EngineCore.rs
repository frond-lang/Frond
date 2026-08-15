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
use hashbrown::{HashMap, HashSet};
use std::sync::OnceLock;
use crossbeam_deque::Injector;
use std::sync::Arc;

/// Caches boolean environment-variable flags to avoid calling `std::env::var` on every hot-path
/// invocation. All known engine-side flag names are probed once (first call) and served from the
/// map afterwards; unknown names fall back to an uncached probe (and are not memoized).
pub(super) fn env_flag(name: &str) -> bool {
    static FLAGS: OnceLock<hashbrown::HashMap<&'static str, bool>> = OnceLock::new();
    const KNOWN_FLAGS: &[&str] = &[
        "FROND_DEBUG_STALL",
        "FROND_NO_REUSECHAIN",
        "FROND_DEBUG_FORIN",
        "FROND_DEBUG_CALL",
        "FROND_DEBUG_IFELSE",
        "FROND_DEBUG_GATE",
        "FROND_DEBUG_WB",
        "FROND_DEBUG_SYNC",
        "FROND_DEBUG_MEMO",
        "FROND_VERIFY",
        "FROND_VERIFY_STRICT",
        "FROND_EXEC_COVERAGE",
    ];
    let flags = FLAGS.get_or_init(|| {
        let mut m = hashbrown::HashMap::with_capacity(KNOWN_FLAGS.len());
        for flag in KNOWN_FLAGS {
            m.insert(*flag, std::env::var(flag).is_ok());
        }
        m
    });
    match flags.get(name) {
        Some(&v) => v,
        None => std::env::var(name).is_ok(),
    }
}

// =========================================================================
// Execution coverage (FROND_EXEC_COVERAGE=1)
// =========================================================================

/// Process-global per-sg frame-start counters — the class-level detector for
/// "std paths that exist in the final graph but are NEVER executed by any
/// test". Every incident in this family (u64(x) silent void, `[0u8]*len` empty
/// array, open-flags abort, File.remove deleting nothing) lived for months in
/// exactly such paths. Instrumented at `start_subgraph_frame` (both the queue
/// and the inline sync path) and at `switch_subgraph` (tail-call frame reuse);
/// reported by `exec_cov_dump` keyed by the graph's debug name sidecar.
static EXEC_COV: OnceLock<Vec<std::sync::atomic::AtomicU32>> = OnceLock::new();

/// Bumps the execution counter for `sg`; initializes the counter table on
/// first use (sized to the graph). No-op unless FROND_EXEC_COVERAGE is set.
pub(super) fn exec_cov_bump(sg: crate::ir::Ir::SubGraphId, total_sgs: usize) {
    if !env_flag("FROND_EXEC_COVERAGE") {
        return;
    }
    let cov = EXEC_COV.get_or_init(|| {
        (0..total_sgs).map(|_| std::sync::atomic::AtomicU32::new(0)).collect()
    });
    if let Some(slot) = cov.get(sg.0 as usize) {
        slot.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// End-of-run report, keyed by the qualified debug names (`sg_debug_names`):
///   EXECCOV-INV <name>  — std.* function present in the final graph
///   EXECCOV-RUN <name>  — …and actually frame-started at least once
/// Aggregated across the whole test suite by tests/scripts/run_execcov.sh
/// (names are stable across processes; sg ids are not).
pub fn exec_cov_dump(graph: &crate::ir::Ir::DataFlowGraph) {
    if !env_flag("FROND_EXEC_COVERAGE") {
        return;
    }
    let Some(cov) = EXEC_COV.get() else { return };
    let mut inv = 0usize;
    let mut ran = 0usize;
    for (idx, name) in graph.sg_debug_names.iter().enumerate() {
        let Some(name) = name else { continue };
        if !name.starts_with("std.") {
            continue;
        }
        inv += 1;
        eprintln!("EXECCOV-INV {name}");
        if graph.subgraphs.get(idx).is_some()
            && cov.get(idx).map_or(false, |c| c.load(std::sync::atomic::Ordering::Relaxed) > 0)
        {
            ran += 1;
            eprintln!("EXECCOV-RUN {name}");
        }
    }
    eprintln!("EXECCOV-SUMMARY std_inv={inv} std_ran={ran}");
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
    /// Defer-frame tracking: the set of all currently-active defer frames (distinguishes defer
    /// frames from ordinary child frames in `process_frame`'s Completed/Failed branches).
    pub defer_frames: S::Mutex<HashSet<FrameId>>,
    /// Defer-waiter tracking: maps a frame that is suspended waiting for its defer frames to the
    /// number of defer frames still pending. When the count reaches zero the frame is resumed.
    pub defer_waiters: S::Mutex<HashMap<FrameId, u32>>,
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

/// E0 perf: one-time materialization of every node's Const value (scalars inline; strings
/// shared as one Arc for the whole run — previously every execution of a string const cost
/// two heap allocations). Runs at EngineRef::new, the single choke point after
/// build/optimize/load, so renumbering passes can never invalidate a populated cache.
fn materialize_const_cache(graph: &mut DataFlowGraph) {
    if !graph.const_cache.is_empty() {
        return;
    }
    let n = graph.node_count();
    let mut cache = Vec::with_capacity(n);
    let pool = graph.string_pool_slice();
    for idx in 0..n {
        let v = match graph.const_value(idx) {
            Some(cv) => alloc_const_value(cv, pool),
            None => Value::VOID,
        };
        cache.push(v);
    }
    graph.const_cache = cache;
}

/// E3 perf: precompute per-subgraph initial pending_inputs + ready-queue seeds for
/// cross-function frames. The derivation depends only on graph structure (nested ranges,
/// EventSource kinds, input counts, param positions) — never on runtime state — so a single
/// pass at engine start covers every later frame instantiation with a memcpy.
fn precompute_sg_templates(graph: &mut DataFlowGraph) {
    if !graph.sg_initial_pending.is_empty() {
        return;
    }
    const PENDING_EXTERNAL: u16 = u16::MAX;
    let sg_count = graph.subgraphs.len();
    let mut templates: Vec<Vec<u16>> = Vec::with_capacity(sg_count);
    let mut seeds: Vec<Vec<NodeId>> = Vec::with_capacity(sg_count);
    for s in 0..sg_count {
        let (node_start, node_end) = graph.subgraphs[s].node_range;
        let node_count = (node_end.0 - node_start.0) as usize;
        let offset = node_start.0 as usize;
        let nested_ranges: &[(u32, u32)] = graph.sg_nested_ranges(s);
        let is_nested =
            |gid: u32| -> bool { nested_ranges.iter().any(|&(a, b)| gid >= a && gid < b) };
        let param_count = graph.subgraphs[s].param_count as usize;

        let mut pending = vec![0u16; node_count];
        let mut seed = Vec::new();
        for i in 0..node_count {
            let gid = (offset + i) as u32;
            if is_nested(gid) {
                pending[i] = PENDING_EXTERNAL;
                continue;
            }
            let node = graph.node(offset + i);
            if node.kind == NodeKind::EventSource {
                pending[i] = PENDING_EXTERNAL;
                continue;
            }
            let inputs = graph.inputs(node.inputs_offset, node.input_count);
            let in_frame = inputs
                .iter()
                .filter(|&&n| (n.0.wrapping_sub(node_start.0) as usize) < node_count)
                .count() as u16;
            pending[i] = in_frame;
            if in_frame == 0 && i >= param_count {
                seed.push(NodeId(i as u32));
            }
        }
        templates.push(pending);
        seeds.push(seed);
    }
    graph.sg_initial_pending = templates;
    graph.sg_initial_seed = seeds;
}

/// E4 perf: flat per-node downstream consumer count, replacing CSR offset arithmetic on every
/// set_value / argument injection.
fn materialize_downstream_counts(graph: &mut DataFlowGraph) {
    if !graph.downstream_counts.is_empty() {
        return;
    }
    let n = graph.node_count();
    let mut counts = Vec::with_capacity(n);
    for idx in 0..n {
        counts.push(graph.downstream_slice(idx).len() as u16);
    }
    graph.downstream_counts = counts;
}

/// E5 perf: per-subgraph linearized execution plans. A plan is the topological order of the
/// sg's own nodes (node_range minus nested-subgraph ranges); run_linear executes it without the
/// readiness machinery (pending countdown / ready queue / notify). Launch nodes (Gate/Call/
/// Await/EventSource) stay IN the plan at their topological position — the linear runner bails
/// to the dataflow engine at them (control flow breaks linearity). Non-linearizable: any
/// EventSource in range, or a cyclic own-node subgraph (defensive — dataflow sgs are DAGs).
fn materialize_linear_plans(graph: &mut DataFlowGraph) {
    if !graph.linear_plans.is_empty() {
        return;
    }
    let sg_count = graph.subgraphs.len();
    let mut plans: Vec<Option<Vec<NodeId>>> = Vec::with_capacity(sg_count);
    for s in 0..sg_count {
        plans.push(compute_linear_plan(graph, s));
    }
    graph.linear_plans = plans;
}

fn compute_linear_plan(graph: &DataFlowGraph, s: usize) -> Option<Vec<NodeId>> {
    let (node_start, node_end) = graph.subgraphs[s].node_range;
    let start = node_start.0 as usize;
    let end = node_end.0 as usize;
    let n = end - start;
    if n == 0 {
        return Some(Vec::new());
    }
    let nested: &[(u32, u32)] = graph.sg_nested_ranges(s);
    let is_nested = |gid: u32| -> bool { nested.iter().any(|&(a, b)| gid >= a && gid < b) };

    let mut own = vec![false; n];
    for i in 0..n {
        let node = graph.node(start + i);
        if node.kind == NodeKind::EventSource {
            return None;
        }
        // Only fully-linear subgraphs get a plan: a launch node mid-sg would force a
        // bail + readiness rebuild every run, which costs more than pure dataflow
        // driving (measured: match_dispatch regressed ~13% with prefix-linear plans).
        // The bail machinery in run_linear remains as a safety net.
        if is_launch_kind(node.kind) {
            return None;
        }
        if is_nested((start + i) as u32) {
            continue;
        }
        own[i] = true;
    }

    // Own-node edges (producer → consumer) for Kahn's algorithm.
    let mut indeg = vec![0u32; n];
    let mut consumers: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        if !own[i] {
            continue;
        }
        let node = graph.node(start + i);
        let inputs = graph.inputs(node.inputs_offset, node.input_count);
        for &inp in inputs {
            let il = inp.0.wrapping_sub(node_start.0) as usize;
            if il < n && own[il] {
                indeg[i] += 1;
                consumers[il].push(i as u32);
            }
        }
    }

    let mut queue: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    for i in 0..n {
        if own[i] && indeg[i] == 0 {
            queue.push_back(i as u32);
        }
    }
    let own_count = own.iter().filter(|&&o| o).count();
    let mut plan: Vec<NodeId> = Vec::with_capacity(own_count);
    while let Some(i) = queue.pop_front() {
        plan.push(NodeId((start + i as usize) as u32));
        for &c in &consumers[i as usize] {
            indeg[c as usize] -= 1;
            if indeg[c as usize] == 0 {
                queue.push_back(c);
            }
        }
    }
    if plan.len() != own_count {
        return None; // cycle among own nodes — defensive
    }
    Some(plan)
}

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
        let mut graph = graph;
        materialize_const_cache(&mut graph);
        precompute_sg_templates(&mut graph);
        materialize_downstream_counts(&mut graph);
        materialize_linear_plans(&mut graph);
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
        let graph = match &self {
            Self::Single(e) => e.graph.clone(),
            Self::Multi(e) => e.graph.clone(),
        };
        let result = match self {
            Self::Single(e) => e.run_single(),
            Self::Multi(e) => Engine::<Multi>::run_multi(e),
        };
        exec_cov_dump(&graph);
        result
    }
}
