//! Engine core type definitions: the `Engine<S>` struct, the `EngineRef` factory, Send/Sync
//! implementations, and scheduler constants.
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
use parking_lot::Mutex as ParkingMutex;
use hashbrown::{HashMap, HashSet};
use std::sync::Arc;


// =========================================================================

// =========================================================================
// Sentinel constants — used by the scheduler
// =========================================================================

/// Sentinel for a `pending_inputs` slot marking it as "never ready / external source" (the actual
/// in-degree must stay below 65535).
pub(super) const PENDING_EXTERNAL: u16 = u16::MAX;

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
    /// Event-keyed waiter index: event → frames registered for it (registration
    /// order preserved within a bucket). The old Vec<(event, fid)> form scanned
    /// the WHOLE table on every event arrival / await-repoll cleanup / frame
    /// release — O(waiters) each, quadratic on deep async recursion chains
    /// (n suspended frames → n linear sweeps during unwind). Keyed buckets
    /// make arrival O(this event's waiters) and recursion linear.
    pub event_waiters: S::Mutex<std::collections::HashMap<crate::ir::Ir::RuntimeEvent, Vec<FrameId>>>,
    pub pending_completions:
        S::Mutex<HashMap<FrameId, Vec<(crate::ir::Ir::NodeId, Value, crate::ir::Ir::ControlSignal)>>>,
    /// Fallback for event-delivery races: when an event arrives while a frame is being processed by
    /// process_frame (and is therefore absent from the HashMap), the event is stashed here and
    /// consumed once process_frame inserts the frame (symmetric to pending_completions).
    /// Multi-slot per frame: several events can arrive while the frame is out of
    /// the map (being processed) — a single-slot map silently OVERWROTE the
    /// earlier event, and when the survivor was stale the frame's real wait was
    /// lost forever (the fourth await-loop hang root).
    pub pending_events:
        S::Mutex<HashMap<FrameId, Vec<(crate::ir::Ir::RuntimeEvent, Value)>>>,
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
    /// The ready-frame queue for both variants (Single direct; Multi's
    /// deterministic event loop).
    pub ready_frames: Option<RefCell<std::collections::VecDeque<FrameId>>>,
    /// Queue-membership set (Multi only): at most ONE pending ready-queue
    /// entry per frame. Root fix (2026-09) for the duplicate-entry family:
    /// the suspension handoff pushes AND the wake path pushes the same frame,
    /// and a stale second entry that survived to a relaunch double-executed
    /// the frame from its entry node (the CI-flaky await_loop corruption /
    /// hang family). push/pop are guarded by the same mutex so membership and
    /// queue content can never desynchronize.
    pub queued_dedup: Option<ParkingMutex<HashSet<FrameId>>>,
    // pub(super): the struct is defined in engine::EngineCore; sibling submodules (Strategy, etc.)
    // must be allowed to write this field when constructing `Engine { ... }`.
    pub(super) _strategy: std::marker::PhantomData<S>,
}

// Safety: Frame contains raw pointers (root_frame_ptr/parent_frame_ptr) and the
// RefCell-wrapped fields are not thread-safe. Sound by construction since M3b:
// both Single and Multi execute the graph on ONE thread (the caller's), so no
// field is accessed from another thread while the engine runs. The impls keep
// `EngineRef::Multi(Arc<Engine<Multi>>)` satisfying auto-trait bounds; moving a
// RUNNING engine across threads (or the value layer's thread_local
// GLOBAL_ARENA keying) remains unsupported.
unsafe impl Send for Engine<Multi> {}
unsafe impl Sync for Engine<Multi> {}

/// L3'' classification: which function-level subgraphs may execute as
/// same-frame callees (saved-context switch instead of a child-frame launch).
/// A leaf straight-line function: sync, no defer / event sources / converter
/// machinery / loop kinds / upvalues, whose OWN nodes contain no call launch /
/// closure call / await / select / break / continue / throw-propagation node
/// (Gates and plain CF_RETURN are fine — arms run via the E7 same-frame relay
/// inside the switched frame), and the same shape recursively for EVERY
/// descendant sg (all same-function). Nested calls are thereby excluded in
/// v1: a call inside an arm would stack a second SavedCallCtx, which the
/// runtime supports but the static shape stays leaf-only.
fn classify_same_frame_callees(graph: &mut DataFlowGraph) {
    use crate::ir::Ir::*;
    let sg_count = graph.subgraphs.len();

    // Direct-children index tree via the same sort+stack nesting walk as
    // compute_nested_ranges (which stores ranges only, without indices).
    let mut order: Vec<(u32, u32, usize)> = (0..sg_count)
        .map(|i| (graph.subgraphs[i].node_range.0 .0, graph.subgraphs[i].node_range.1 .0, i))
        .filter(|&(s, e, _)| s < e)
        .collect();
    order.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    let mut children_idx: Vec<Vec<usize>> = vec![Vec::new(); sg_count];
    let mut stack: Vec<(u32, usize)> = Vec::new();
    for &(start, end, idx) in &order {
        while let Some(&(top_end, _)) = stack.last() {
            if top_end <= start || end > top_end {
                stack.pop();
            } else {
                break;
            }
        }
        if let Some(&(_, parent)) = stack.last() {
            children_idx[parent].push(idx);
        }
        stack.push((end, idx));
    }

    // own_bad[t]: t's own nodes (range minus direct children) contain a
    // banned compute fn.
    let banned = |cf: u32| {
        cf == CF_CALL_LAUNCH.0
            || cf == CF_ASYNC_CALL_LAUNCH.0
            || cf == CF_CLOSURE_CALL.0
            || cf == CF_AWAIT.0
            || cf == CF_SELECT_GATE.0
            || cf == CF_BREAK.0
            || cf == CF_CONTINUE.0
            || cf == CF_THROW_WRAP_ERR.0
            || cf == CF_PROPAGATE.0
            || cf == CF_CANCEL_ASYNC_HANDLE.0
            || cf == CF_DEFER_REGISTER.0
            || cf == CF_BLOCK_DEFER_REGISTER.0
            || cf == CF_DEFER_RUN.0
    };
    let mut own_bad = vec![false; sg_count];
    for t in 0..sg_count {
        let (ts, te) = graph.subgraphs[t].node_range;
        if te.0 <= ts.0 {
            continue;
        }
        let children = graph.subgraphs[t].nested_ranges.clone();
        let mut ci = 0usize;
        let mut gid = ts.0;
        let mut bad = false;
        while gid < te.0 {
            while ci < children.len() && children[ci].1 <= gid {
                ci += 1;
            }
            if ci < children.len() && gid >= children[ci].0 {
                gid = children[ci].1;
                ci += 1;
                continue;
            }
            if gid as usize >= graph.node_count() {
                break;
            }
            let node = graph.node(gid as usize);
            if banned(node.compute_fn.0)
                // Gates inside a same-frame callee execute their arms via the
                // E7 branch relay in the switched frame; the relay's outer-
                // input snapshot mis-resolves a compound (multi-node) else-if
                // condition in nested wrap chains (f3(98) reproduced as arm3
                // instead of arm2). Until that path is reworked, keep
                // same-frame callees strictly straight-line.
                || node.kind == crate::ir::Ir::NodeKind::Gate
            {
                bad = true;
                break;
            }
            gid += 1;
        }
        own_bad[t] = bad;
    }

    // Subtree validity memo (0 unknown / 1 ok / 2 bad), checked against the
    // function id of the DFS root.
    let mut memo = vec![0u8; sg_count];
    fn visit(
        graph: &DataFlowGraph,
        children_idx: &Vec<Vec<usize>>,
        own_bad: &Vec<bool>,
        memo: &mut Vec<u8>,
        root_fn: usize,
        t: usize,
    ) -> bool {
        if memo[t] != 0 {
            return memo[t] == 1;
        }
        let sg = &graph.subgraphs[t];
        let ok = sg.function_id as usize == root_fn
            && !sg.has_suspend
            && sg.event_source_decls.is_empty()
            && sg.defer_table.is_empty()
            && sg.loop_kind == crate::ir::Ir::LoopKind::None
            && sg.loop_parent_sg.is_none()
            && sg.upvalue_count == 0
            && !own_bad[t]
            && children_idx[t]
                .iter()
                .all(|&c| visit(graph, children_idx, own_bad, memo, root_fn, c));
        memo[t] = if ok { 1 } else { 2 };
        ok
    }

    let mut ok = vec![false; sg_count];
    for s in 0..sg_count {
        let sg = &graph.subgraphs[s];
        if sg.function_id as usize != s
            || sg.converter_generated
            || sg.has_suspend
            || !sg.event_source_decls.is_empty()
            || !sg.defer_table.is_empty()
            || sg.loop_kind != crate::ir::Ir::LoopKind::None
            || sg.loop_parent_sg.is_some()
            || sg.upvalue_count != 0
            || own_bad[s]
        {
            continue;
        }
        ok[s] = children_idx[s]
            .iter()
            .all(|&c| visit(graph, &children_idx, &own_bad, &mut memo, s, c));
    }
    if std::env::var_os("FROND_L3PP_DEBUG").is_some() {
        let eligible: Vec<String> = (0..sg_count)
            .filter(|&i| ok[i])
            .map(|i| format!("{}#{}", graph.sg_names.get(i).map(|s| s.as_str()).unwrap_or("?"), i))
            .collect();
        if let Some(idx) = std::env::var("FROND_L3PP_SG").ok().and_then(|v| v.parse::<usize>().ok()) {
            let (ns, ne) = graph.subgraphs[idx].node_range;
            for gid in ns.0..ne.0 {
                let n = graph.node(gid as usize);
            }
        }
    }
    graph.sg_callee_same_frame = ok;
}

/// E0 perf: one-time materialization of every node's Const value (scalars inline; strings
/// shared as one Arc for the whole run — previously every execution of a string const cost
/// two heap allocations). Runs at EngineRef::new, the single choke point after
/// build/optimize/load, so renumbering passes can never invalidate a populated cache.
/// Per-node shared Record/Adt/Newtype shapes, built once from each construct
/// node's RecordLitInfo. Every runtime instance then clones one Arc instead
/// of re-allocating type_name String + field_names Vec (+ per-field name
/// Strings).
fn materialize_record_shapes(graph: &mut DataFlowGraph) {
    if !graph.record_shapes.is_empty() {
        return;
    }
    let n = graph.node_count();
    let mut shapes: Vec<std::sync::Arc<crate::value::RecordShape>> = Vec::with_capacity(n);
    for idx in 0..n {
        // .fndo-loaded graphs keep opt-metadata in packed mmap sections: the
        // `record_lit_infos` Vec is EMPTY there and only the accessor decodes
        // it. Reading the raw Vec silently produced blank shapes (constructor
        // names lost → match dispatch found no arm).
        let shape = match graph.record_lit_info_at(idx) {
            Some(info) => {
                // Packing guard: tags must be arity-exact; any mismatch
                // (literal-path sites without sema reprs, partial data)
                // degrades the WHOLE shape to generic Value slots — packed
                // storage with a short pack table would silently drop
                // fields.
                // Newtype literals carry ONE field but an empty name
                // table — pack arity must follow the actual field count.
                let arity = if info.field_names.is_empty()
                    && matches!(
                        info.kind,
                        crate::ir::Ir::RecordLitKind::Newtype
                    ) {
                    1
                } else {
                    info.field_names.len()
                };
                let generic: Vec<u8>;
                let tags: &[u8] = if info.field_tags.len() == arity {
                    &info.field_tags
                } else {
                    generic = vec![0xFF; arity];
                    &generic
                };
                let (field_packs, pack_offsets, value_region_bytes) =
                    crate::value::RecordShape::compute_layout(tags);
                let disc = crate::value::ctor_disc(&info.type_name, &info.constructor);
                std::sync::Arc::new(crate::value::RecordShape {
                    type_name: info.type_name.as_str().into(),
                    constructor: info.constructor.as_str().into(),
                    field_names: info
                        .field_names
                        .iter()
                        .map(|n| n.as_ref().map(|s| s.as_str().into()))
                        .collect(),
                    kind: match info.kind {
                        crate::ir::Ir::RecordLitKind::Record => crate::value::ShapeKind::Record,
                        crate::ir::Ir::RecordLitKind::Adt => crate::value::ShapeKind::Adt,
                        crate::ir::Ir::RecordLitKind::Newtype => crate::value::ShapeKind::Newtype,
                    },
                    field_packs,
                    pack_offsets,
                    value_region_bytes,
                    disc,
                })
            }
            None => std::sync::Arc::new(crate::value::RecordShape {
                type_name: "".into(),
                constructor: "".into(),
                field_names: Vec::new(),
                kind: crate::value::ShapeKind::Record,
                field_packs: Vec::new(),
                pack_offsets: Vec::new(),
                value_region_bytes: 0,
                disc: 0,
            }),
        };
        shapes.push(shape);
    }
    graph.record_shapes = shapes;
}

/// M1: per-pattern-node acceptable ctor discriminants. For every pattern
/// ctor-match node with a type-adjudicated ctor, the set is
/// {disc(T', ctor) : T' == pattern_type or inherits(pattern_type)} — the
/// disc equivalent of the old string-compare + type_inherits walk. Nodes
/// without adjudication (or non-Adt shapes at runtime) keep the string path.
fn materialize_pattern_discs(graph: &mut DataFlowGraph) {
    if !graph.pattern_disc_sets.is_empty() {
        return;
    }
    let n = graph.node_count();
    let mut sets: Vec<Box<[u32]>> = vec![Box::from([0u32; 0]); n];
    // Reverse-inheritance closure: all types inheriting from `tn` (incl. tn).
    let descendants = |tn: &str| -> Vec<String> {
        let mut out: Vec<String> = vec![tn.to_string()];
        let mut frontier: Vec<String> = vec![tn.to_string()];
        while let Some(c) = frontier.pop() {
            for (child, base) in &graph.inheritance_links {
                if base.as_ref() == c.as_str() && !out.iter().any(|o| o.as_str() == child.as_ref()) {
                    out.push(child.to_string());
                    frontier.push(child.to_string());
                }
            }
        }
        out
    };
    for idx in 0..n {
        let Some(ctor) = graph.pattern_ctor_name(idx) else { continue };
        let Some(tn) = graph.pattern_type_name(idx) else { continue };
        let mut set: Vec<u32> = descendants(tn)
            .iter()
            .map(|t| crate::value::ctor_disc(t, ctor.as_ref()))
            .collect();
        set.sort_unstable();
        set.dedup();
        sets[idx] = set.into_boxed_slice();
    }
    graph.pattern_disc_sets = sets;
}

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

    // Gate-in-plan is allowed only where launching is statically known to be
    // same-frame-safe: plain / LoopBody sgs whose every own Gate's BOTH branch
    // targets pass the E7 eligibility (minus the runtime arg bits). A gate
    // that would bail every iteration (loop-dispatch gates, converter state
    // machines, capture/suspending/closure arms) rejects the plan — the sg
    // keeps the pure dataflow driver (pre-E9 behavior, no regression).
    let sg_kind = graph.subgraphs[s].loop_kind;
    let sg_conv = graph.subgraphs[s].converter_generated;
    let fn_id = graph.subgraphs[s].function_id;
    let gates_ok = !sg_conv
        && matches!(
            sg_kind,
            crate::ir::Ir::LoopKind::None | crate::ir::Ir::LoopKind::LoopBody
        );
    let gate_static_ok = |gate_idx: usize| -> bool {
        let Some(gb) = graph.gate_branches_at(gate_idx) else {
            return false;
        };
        if gb.capture {
            return false;
        }
        for (_, bsg, _) in &gb.branches {
            let t = &graph.subgraphs[bsg.0 as usize];
            if t.converter_generated
                || t.has_suspend
                || !t.event_source_decls.is_empty()
                || t.loop_kind != crate::ir::Ir::LoopKind::None
                || t.function_id != fn_id
                || bsg.0 == t.function_id
            {
                return false;
            }
            // Control-signal-free own nodes (mirrors the runtime eligibility
            // scan in Schedule.rs — keep the two in sync).
            let nested_t: &[(u32, u32)] = graph.sg_nested_ranges(bsg.0 as usize);
            let (cs, ce) = t.node_range;
            for g in cs.0..ce.0 {
                if nested_t.iter().any(|&(a, b)| g >= a && g < b) {
                    continue;
                }
                if crate::ir::Ir::is_control_flow_compute_fn(
                    graph.node(g as usize).compute_fn,
                ) {
                    return false;
                }
            }
        }
        true
    };
    let mut own = vec![false; n];
    for i in 0..n {
        let node = graph.node(start + i);
        if node.kind == NodeKind::EventSource {
            return None;
        }
        // Segmented-linear (E9): Gate nodes stay IN the plan at their topo
        // position — run_linear launches the taken branch same-frame (E7) and
        // drains the injected subtree before continuing, so no bail/rebuild is
        // needed for them. Other launch kinds (Call/Await/EventSource) still
        // reject the plan: their protocols need the dataflow driver. (The
        // pre-E7 measurement — mid-sg launches regressing match_dispatch ~13%
        // — was about child-frame launches + readiness rebuilds, both gone.)
        if is_launch_kind(node.kind) {
            if node.kind != NodeKind::Gate || !gates_ok || !gate_static_ok(start + i) {
                return None;
            }
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
    /// mistaken as async-requiring.
    /// A purely synchronous program has no suspension points, so it takes the
    /// `Single` variant; a program containing async takes `Multi`. Both
    /// variants execute the same deterministic single-threaded event loop —
    /// `Multi` is the async-capable marker, not a worker pool.
    pub fn new(graph: DataFlowGraph) -> Self {
        let mut graph = graph;
        // .fndo hot-path parity: loaded (mmap-backed) graphs unpack the
        // packed Nodes section PER node() call; on interpreter hot loops
        // that indirection cost 2-3x (loop_sum: ~290ms source vs ~670ms
        // artifact). Artifacts are kilobytes — materialize the nodes Vec
        // once, up front, so both paths hit the same Vec slice in node().
        if graph.mem.is_some() && graph.nodes.is_empty() {
            let n = graph.node_count();
            let mut nodes = Vec::with_capacity(n);
            for i in 0..n {
                nodes.push(graph.node(i));
            }
            graph.nodes = nodes;
        }
        // Same treatment for the Inputs pool: inputs() does a mmap
        // section lookup + transmute per call; one bulk copy into the
        // pool removes the remaining per-node indirection.
        if graph.mem.is_some() && graph.inputs_pool.data.is_empty() {
            let total = graph
                .nodes
                .iter()
                .map(|n| n.inputs_offset as usize + n.input_count as usize)
                .max()
                .unwrap_or(0);
            let mut data = vec![NodeId(0); total];
            for nd in graph.nodes.iter() {
                let s = nd.inputs_offset as usize;
                let src = graph.inputs(nd.inputs_offset, nd.input_count);
                data[s..s + nd.input_count as usize].copy_from_slice(src);
            }
            graph.inputs_pool.data = data;
        }
        // Category B boolean tables: same story — bulk-fill the Vecs so
        // the accessors stop reading the mmap bitmaps per call.
        if graph.mem.is_some() {
            let n = graph.nodes.len();
            if graph.tail_call_flags.is_empty() {
                let v: Vec<bool> = (0..n).map(|i| graph.tail_call_flag(i)).collect();
                graph.tail_call_flags = v;
            }
            if graph.safe_op_flags.is_empty() {
                let v: Vec<bool> = (0..n).map(|i| graph.safe_op_flag(i)).collect();
                graph.safe_op_flags = v;
            }
            if graph.slice_inclusive.is_empty() {
                let v: Vec<bool> = (0..n).map(|i| graph.slice_inclusive(i)).collect();
                graph.slice_inclusive = v;
            }
        }
        materialize_record_shapes(&mut graph);
        materialize_pattern_discs(&mut graph);
        materialize_const_cache(&mut graph);
        precompute_sg_templates(&mut graph);
        materialize_downstream_counts(&mut graph);
        materialize_linear_plans(&mut graph);
        classify_same_frame_callees(&mut graph);
        // Scalar-chain programs for every subgraph (shared by the engine's
        // synchronous fast path).
        graph.scalar_progs = (0..graph.subgraphs.len())
            .map(|i| crate::pass::Scalarizer::build_scalar_prog(&graph, crate::ir::Ir::SubGraphId(i as u32)))
            .collect();
        let mut cond_progs: Vec<Option<std::sync::Arc<crate::pass::Scalarizer::ScalarProg>>> =
            (0..graph.subgraphs.len()).map(|_| None).collect();
        for i in 0..graph.subgraphs.len() {
            if let Some((c, body)) = crate::pass::Scalarizer::build_cond_with_body(
                &graph,
                crate::ir::Ir::SubGraphId(i as u32),
            ) {
                // The joint build returns an EXPORT-AUGMENTED body program —
                // overwrite the plain one so the tight loop and the plain
                // path share the same (richer) program.
                for (bi, b) in graph.subgraphs.iter().enumerate() {
                    if b.loop_kind == crate::ir::Ir::LoopKind::LoopBody
                        && b.loop_parent_sg == Some(crate::ir::Ir::SubGraphId(i as u32))
                    {
                        graph.scalar_progs[bi] = Some(body);
                        break;
                    }
                }
                cond_progs[i] = Some(c);
            }
        }
        graph.cond_progs = cond_progs;
        let mut simd_maps: Vec<Option<std::sync::Arc<crate::pass::Scalarizer::SimdMapPlan>>> =
            (0..graph.subgraphs.len()).map(|_| None).collect();
        for wi in 0..graph.subgraphs.len() {
            let (Some(cond), Some(bodyp)) = (
                graph.cond_prog(wi),
                graph.subgraphs.iter().position(|b| {
                    b.loop_kind == crate::ir::Ir::LoopKind::LoopBody
                        && b.loop_parent_sg == Some(crate::ir::Ir::SubGraphId(wi as u32))
                })
                .and_then(|bi| graph.scalar_prog(bi)),
            ) else { continue };
            if let Some(plan) = crate::pass::Scalarizer::analyze_simd_map(&bodyp, &cond) {
                simd_maps[wi] = Some(plan);
            }
        }
        graph.simd_maps = simd_maps;
        let has_async = Self::entry_reaches_suspend(&graph);
        if has_async {
            Self::Multi(Arc::new(Engine::<Multi>::new_multi(graph)))
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
        let result = match self {
            Self::Single(e) => e.run_single(),
            Self::Multi(e) => Engine::<Multi>::run_multi(e),
        };
        // Teardown cycle sweep: releases any cyclic garbage still registered
        // (roots empty — nothing runs after this).
        let _ = crate::value::Registry::collect_cycles(&[]);
        result
    }
}
