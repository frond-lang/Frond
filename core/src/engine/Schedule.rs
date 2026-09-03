//! Dataflow scheduling core: readiness-scheduling free functions, run_frame_nodes, process_frame.
//!
//! SIMD batching has been pushed down into compute_fn (via EvalContext + do_simd_batch), so the
//! engine hot loop no longer has batching-specialization checks.

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::Value;
use crate::ir::Ir::char_from_u32_or_nul;

// =========================================================================
// Frame-operation helper functions (pure, do not depend on Engine state)
// =========================================================================

/// Converts a ConstValue into a Value (constructed directly, without using the arena).
/// `pool` = a byte slice of DataFlowGraph.string_pool; the Str variant reads its string from it.
pub fn alloc_const_value(cv: ConstValue, pool: &[u8]) -> Value {
    match cv {
        ConstValue::I8(v) => Value::i8(v),
        ConstValue::I16(v) => Value::i16(v),
        ConstValue::I32(v) => Value::i32(v),
        ConstValue::I64(v) => Value::i64(v),
        ConstValue::I128(v) => Value::i128(v),
        ConstValue::U8(v) => Value::u8(v),
        ConstValue::U16(v) => Value::u16(v),
        ConstValue::U32(v) => Value::u32(v),
        ConstValue::U64(v) => Value::u64(v),
        ConstValue::U128(v) => Value::u128(v),
        ConstValue::Isize(v) => Value::isize_val(v),
        ConstValue::Usize(v) => Value::usize_val(v),
        ConstValue::F32(v) => Value::f32(v),
        ConstValue::F64(v) => Value::f64(v),
        ConstValue::F16(bits) => Value::f16(crate::value::F16(bits)),
        ConstValue::F128(bytes) => Value::f128(crate::value::F128(bytes)),
        ConstValue::Bool(v) => Value::bool_val(v),
        ConstValue::Char(c) => Value::char_val(char_from_u32_or_nul(c)),
        ConstValue::Null => Value::NULL,
        ConstValue::Void => Value::VOID,
        ConstValue::Str { offset, len } => {
            let off = offset as usize;
            let end = off + len as usize;
            let s = std::str::from_utf8(&pool[off..end]).unwrap_or("");
            Value::str_val(s)
        }
    }
}

/// Frame node initialization: sets node_offset + pending_inputs + prefills Const + enqueues Gate
/// into the ready queue.
pub fn prepare_frame_nodes(frame: &mut Frame, graph: &DataFlowGraph) {
    let sg_id = frame.subgraph_id;
    let (node_start, node_end) = graph.subgraphs[sg_id.0 as usize].node_range;
    let node_count = (node_end.0 - node_start.0) as usize;

    // Set node_offset.
    frame.node_offset = node_start.0;

    // E3: engine-precomputed per-sg template (static for cross-function frames — no parent
    // state participates). Falls back to the legacy derivation when the engine hasn't
    // populated the templates (LSP / sync-interpreter contexts).
    if !graph.sg_initial_pending.is_empty() {
        let tpl = &graph.sg_initial_pending[sg_id.0 as usize];
        frame.pending_inputs[..node_count].copy_from_slice(tpl);
        for &local in &graph.sg_initial_seed[sg_id.0 as usize] {
            if !frame.value_table.is_ready(local.0 as usize) {
                frame.push_ready(local);
            }
        }
        return;
    }

    // Use the precomputed nested_ranges (filled at build time) to avoid a runtime full-graph scan.
    let nested_ranges: &[(u32, u32)] = graph.sg_nested_ranges(sg_id.0 as usize);


    let is_nested = |global_idx: u32| -> bool {
        nested_ranges.iter().any(|&(s, e)| global_idx >= s && global_idx < e)
    };

    // 1. Initialize pending_inputs (select Gate -> 0; other nodes count actual in-frame inputs).
    for i in 0..node_count {
        if is_nested((node_start.0 as usize + i) as u32) {
            frame.pending_inputs[i] = PENDING_EXTERNAL;
        } else {
            let graph_node = graph.node(node_start.0 as usize + i);
            if graph_node.kind == NodeKind::EventSource {
                frame.pending_inputs[i] = PENDING_EXTERNAL;
            } else {
                let inputs = graph.inputs(
                    graph_node.inputs_offset,
                    graph_node.input_count,
                );
                let in_frame = inputs
                    .iter()
                    .filter(|&&n| (n.0.wrapping_sub(node_start.0) as usize) < node_count)
                    .count() as u16;
                frame.pending_inputs[i] = in_frame;
            }
        }
    }

    // 2. Enqueue 0-input nodes into the ready queue (Const nodes also take this path — compute_fn
    // returns a value).
    let param_count = graph.subgraphs[sg_id.0 as usize].param_count as usize;
    for i in 0..node_count {
        if i < param_count {
            continue;
        }
        if is_nested((node_start.0 as usize + i) as u32) {
            continue;
        }
        if frame.pending_inputs[i] == 0 && !frame.value_table.is_ready(i) {
            frame.push_ready(NodeId(i as u32));
        }
    }
}

/// E5 linear bail-out: rebuilds the dataflow readiness state (pending_inputs + ready-queue
/// seeds) for the frame's in-range nodes from the current ready bitmap, so the engine can
/// continue the remainder of the subgraph exactly as if it had been driven by the queue all
/// along. Mirrors prepare_frame_nodes / prepare_same_function_frame_sync semantics:
/// pending = count of in-sg-range inputs not ready; nested/EventSource nodes keep
/// PENDING_EXTERNAL; non-branch slots of same_function frames stay untouched.
/// E9 plan execution outcome: Done (finished or control-signaled — the frame
/// machinery takes over) vs Bailed (the remainder needs the dataflow driver).
enum PlanFlow {
    Done,
    Bailed,
}

pub fn rebuild_linear_bailout(frame: &mut Frame, graph: &DataFlowGraph) {
    let sg_id = frame.subgraph_id;
    let (start, end) = graph.subgraphs[sg_id.0 as usize].node_range;
    let start = start.0 as usize;
    let end = end.0 as usize;
    let node_start = frame.node_offset as usize;
    let table_len = frame.value_table.len();
    if table_len == 0 {
        return;
    }
    frame.ready_queue.clear();
    let nested: &[(u32, u32)] = graph.sg_nested_ranges(sg_id.0 as usize);
    let is_nested = |gid: u32| -> bool { nested.iter().any(|&(a, b)| gid >= a && gid < b) };

    for gid in start..end {
        let local = gid.wrapping_sub(node_start);
        if local >= table_len {
            break;
        }
        let node = graph.node(gid);
        if node.kind == NodeKind::EventSource {
            frame.pending_inputs[local] = PENDING_EXTERNAL;
            continue;
        }
        if is_nested(gid as u32) {
            frame.pending_inputs[local] = PENDING_EXTERNAL;
            continue;
        }
        let inputs = graph.inputs(node.inputs_offset, node.input_count);
        let mut pending: u16 = 0;
        for &inp in inputs {
            let il = inp.0 as usize;
            if il >= start && il < end {
                let l = il.wrapping_sub(node_start);
                if l < table_len && !frame.value_table.is_ready(l) {
                    pending += 1;
                }
            }
        }
        frame.pending_inputs[local] = pending;
        if pending == 0 && !frame.value_table.is_ready(local) {
            frame.push_ready(NodeId(local as u32));
        }
    }
}

/// Notifies downstream nodes: decrements pending_inputs, and enqueues them when it reaches zero
/// (with bounds checks + slot-level RC).

/// E7/E9 shared same-frame branch eligibility (graph+frame-derivable part):
/// same-function branch, non-converter, no suspension/event sources, plain
/// target sg, plain/LoopBody caller, non-body, control-signal-free own nodes,
/// non-capturing gate. The pending-dependent bits (is_async, closure_val) and
/// same-frame eligibility stay at the call sites.
/// E7 outer-value snapshot: walks the frame chain (parent first, then root —
/// mirroring `get_value_by_global`) for the nearest frame whose slot for
/// `gid` is ready, and clones that value. None when unready everywhere.
fn snapshot_outer_value(frame: &Frame, gid: NodeId) -> Option<Value> {
    let mut tried_root = false;
    let mut f: *const Frame = frame as *const Frame;
    loop {
        let fr = unsafe { &*f };
        let local = gid.0.wrapping_sub(fr.node_offset) as usize;
        if local < fr.value_table.len() && fr.value_table.is_ready(local) {
            return Some(fr.value_table.get_value(local));
        }
        if !fr.parent_frame_ptr.is_null() {
            f = fr.parent_frame_ptr;
            continue;
        }
        if !tried_root && !fr.root_frame_ptr.is_null() {
            tried_root = true;
            f = fr.root_frame_ptr;
            continue;
        }
        return None;
    }
}

/// L3'': saved caller context for a same-frame callee execution. The callee
/// runs in THIS frame via the tail-call `switch_subgraph` machinery; when it
/// completes (Return signal OR queue exhaustion — expression bodies never
/// emit Return, the value sits in the return_node slot), the caller context
/// is restored and the value written to the call node — replacing the E1
/// child-frame launch (frame acquire + prepare + release).
pub(super) struct SavedCallCtx {
    pub subgraph_id: crate::ir::Ir::SubGraphId,
    pub node_offset: u32,
    pub value_table: crate::ir::Ir::ValueTable,
    pub pending_inputs: Vec<u16>,
    pub ready_queue: std::collections::VecDeque<crate::ir::Ir::NodeId>,
    pub control_signal: crate::ir::Ir::ControlSignal,
    pub suspend_state: crate::ir::Ir::SuspendState,
    pub suspend_event: Option<crate::ir::Ir::RuntimeEvent>,
    pub defer_stack: Vec<crate::ir::Ir::RuntimeDefer>,
    pub branch_relays: Vec<(crate::ir::Ir::NodeId, crate::ir::Ir::NodeId)>,
    pub linear_fresh: bool,
    pub hot_body: Option<(crate::ir::Ir::FrameId, Box<Frame>)>,
    pub cached_child_frame: Option<crate::ir::Ir::FrameId>,
    pub same_fn_prep_cache: Option<Box<(u32, Vec<u8>, Vec<u16>, Vec<crate::ir::Ir::NodeId>)>>,
    pub select_timers: Vec<(usize, crate::ir::Ir::TimerId)>,
    /// The caller's frame chain — restored verbatim (a LoopBody caller's
    /// outer-variable reads walk these; nulling them corrupts resolution).
    pub root_frame_ptr: *mut Frame,
    pub parent_frame_ptr: *mut Frame,
    pub call_node_local: crate::ir::Ir::NodeId,
}

/// L3'' scratch: reusable callee-side table allocations, parked across calls
/// (the pooled-frame equivalent — no per-call malloc/free for the switch).
pub(super) struct L3Scratch {
    pub table: crate::ir::Ir::ValueTable,
    pub pending: Vec<u16>,
    pub queue: std::collections::VecDeque<crate::ir::Ir::NodeId>,
}

/// L3'': pushes the caller context and switches the frame to the callee.
/// Everything `switch_subgraph` resets is either saved here or intentionally
/// kept (construct_cache: gid-keyed, valid across retargets — and RETAINED
/// across calls, unlike a pooled E1 child whose cache is cleared on reuse).
pub(super) fn enter_same_frame_callee(
    frame: &mut Frame,
    graph: &DataFlowGraph,
    target_sg: crate::ir::Ir::SubGraphId,
    args: &[crate::value::Value],
    call_node_local: crate::ir::Ir::NodeId,
    scratch: &mut Vec<L3Scratch>,
) -> SavedCallCtx {
    let sc = scratch.pop().unwrap_or(L3Scratch {
        table: crate::ir::Ir::ValueTable::new(),
        pending: Vec::new(),
        queue: std::collections::VecDeque::new(),
    });
    let ctx = SavedCallCtx {
        subgraph_id: frame.subgraph_id,
        node_offset: frame.node_offset,
        value_table: std::mem::replace(&mut frame.value_table, sc.table),
        pending_inputs: std::mem::replace(&mut frame.pending_inputs, sc.pending),
        ready_queue: std::mem::replace(&mut frame.ready_queue, sc.queue),
        control_signal: frame.control_signal.clone(),
        suspend_state: frame.suspend_state.clone(),
        suspend_event: frame.suspend_event.clone(),
        defer_stack: std::mem::take(&mut frame.defer_stack),
        branch_relays: std::mem::take(&mut frame.branch_relays),
        linear_fresh: frame.linear_fresh,
        hot_body: frame.hot_body.take(),
        cached_child_frame: frame.cached_child_frame.take(),
        same_fn_prep_cache: frame.same_fn_prep_cache.take(),
        select_timers: std::mem::take(&mut frame.select_timers),
        root_frame_ptr: frame.root_frame_ptr,
        parent_frame_ptr: frame.parent_frame_ptr,
        call_node_local,
    };
    switch_subgraph(frame, graph, target_sg, args);
    ctx
}

/// L3'': restores a saved caller context after the callee completed. The
/// return value must be extracted from the callee frame BEFORE this runs.
/// The callee-side allocations are parked back into the scratch for reuse
/// (bounded: v1 callees cannot nest, so the stack stays tiny).
pub(super) fn restore_caller_ctx(
    frame: &mut Frame,
    ctx: SavedCallCtx,
    scratch: &mut Vec<L3Scratch>,
) {
    frame.subgraph_id = ctx.subgraph_id;
    frame.node_offset = ctx.node_offset;
    if scratch.len() < 4 {
        let sc = L3Scratch {
            table: std::mem::replace(&mut frame.value_table, ctx.value_table),
            pending: std::mem::replace(&mut frame.pending_inputs, ctx.pending_inputs),
            queue: std::mem::replace(&mut frame.ready_queue, ctx.ready_queue),
        };
        scratch.push(sc);
    } else {
        frame.value_table = ctx.value_table;
        frame.pending_inputs = ctx.pending_inputs;
        frame.ready_queue = ctx.ready_queue;
    }
    frame.control_signal = ctx.control_signal;
    frame.suspend_state = ctx.suspend_state;
    frame.suspend_event = ctx.suspend_event;
    frame.defer_stack = ctx.defer_stack;
    frame.branch_relays = ctx.branch_relays;
    frame.linear_fresh = ctx.linear_fresh;
    frame.hot_body = ctx.hot_body;
    frame.cached_child_frame = ctx.cached_child_frame;
    frame.same_fn_prep_cache = ctx.same_fn_prep_cache;
    frame.select_timers = ctx.select_timers;
    frame.root_frame_ptr = ctx.root_frame_ptr;
    frame.parent_frame_ptr = ctx.parent_frame_ptr;
    frame.state = FrameState::Ready;
    frame.suspend_state = crate::ir::Ir::SuspendState::NotSuspended;
    frame.suspend_event = None;
}

pub(super) fn same_frame_branch_ok(
    graph: &DataFlowGraph,
    frame: &Frame,
    gate_gid: NodeId,
    target_sg: SubGraphId,
) -> bool {
    let tsg = &graph.subgraphs[target_sg.0 as usize];
    let caller_kind = graph.subgraphs[frame.subgraph_id.0 as usize].loop_kind;
    if tsg.converter_generated
        || tsg.has_suspend
        || !tsg.event_source_decls.is_empty()
        || tsg.loop_kind != crate::ir::Ir::LoopKind::None
        || !matches!(
            caller_kind,
            crate::ir::Ir::LoopKind::None | crate::ir::Ir::LoopKind::LoopBody
        )
        || tsg.function_id != graph.subgraphs[frame.subgraph_id.0 as usize].function_id
        || target_sg.0 == tsg.function_id
    {
        return false;
    }
    let capture = graph
        .gate_branches_at(gate_gid.0 as usize)
        .map(|gb| gb.capture)
        .unwrap_or(false);
    if capture {
        return false;
    }
    let (cs, ce) = tsg.node_range;
    let nested = graph.sg_nested_ranges(target_sg.0 as usize);
    for gid in cs.0..ce.0 {
        if nested.iter().any(|&(a, b)| gid >= a && gid < b) {
            continue;
        }
        if crate::ir::Ir::is_control_flow_compute_fn(graph.node(gid as usize).compute_fn) {
            return false;
        }
    }
    true
}

/// L3'': restores a caller context saved for same-frame callee execution.
pub(super) fn relay_branch_value(
    frame: &mut Frame,
    graph: &DataFlowGraph,
    local: NodeId,
) {
    let mut cur = local;
    loop {
        let Some(pos) = frame.branch_relays.iter().position(|(r, _)| *r == cur) else {
            return;
        };
        let (_, gate_local) = frame.branch_relays.swap_remove(pos);
        let offset = frame.node_offset;
        let gate_gid = NodeId(gate_local.0 + offset);
        let cc = graph.downstream_count(gate_gid.0 as usize);
        let v = frame.get_value(cur);
        frame.set_value(gate_local, v, cc);
        notify_downstream(frame, graph, gate_local, gate_gid, NodeId(offset));
        cur = gate_local;
    }
}

pub fn notify_downstream(
    frame: &mut Frame,
    graph: &DataFlowGraph,
    producer_local: NodeId,
    producer_graph: NodeId,
    node_start: NodeId,
) {
    let downstreams: &[NodeId] = graph.downstream_slice(producer_graph.0 as usize);
    let pending_len = frame.pending_inputs.len();
    for &ds_graph_id in downstreams {
        let ds_local_id = NodeId(ds_graph_id.0.wrapping_sub(node_start.0));
        // Bounds check: skip cross-subgraph downstream.
        if ds_local_id.0 as usize >= pending_len {
            continue;
        }

        let pidx = producer_local.0 as usize;
        // Consume the producer's reference count, but do not clear the ready flag.
        // The ready flag means "this node has produced a value and need not be re-executed".
        // Clearing ready would leave the node in a pending_inputs=0 && ready=false state; when the
        // upstream is re-triggered the node would be re-pushed into ready_queue and executed
        // repeatedly, causing exponential blow-up (especially for call/closure_call nodes).
        // The value stays in the value_table until the frame ends (auto-released on frame drop)
        // or is explicitly reset by reset_node_ready/reset_node_pending (loop-body reuse case).
        let _still_has_consumers = frame.value_table.consume(pidx);

        // Skip the PENDING_EXTERNAL sentinel (nested-subgraph nodes / EventSource nodes):
        // these nodes are driven by child frames or events and must not be decremented by the
        // parent frame's notify_downstream. Decrementing would corrode the sentinel
        // (65535 -> 65534); after 65535 such decrements it would hit zero and the nested node
        // would be erroneously pushed into the parent frame for execution.
        //
        // Only push_ready when pending decrements from >0 to 0.
        // A node with pending=0 has already been enqueued by prepare_frame_nodes' 0-input
        // enqueue; pushing again would cause it to be executed multiple times (e.g. a Gate node
        // re-triggering child-frame startup, child-frame return values overwriting each other, or
        // a Gate executing early and reading null when the condition value is not ready).
        let pending = frame.pending_inputs[ds_local_id.0 as usize];
        if pending > 0 && pending != PENDING_EXTERNAL {
            let new_pending = pending - 1;
            frame.pending_inputs[ds_local_id.0 as usize] = new_pending;
            if new_pending == 0 && !frame.value_table.is_ready(ds_local_id.0 as usize) {
                frame.push_ready(ds_local_id);
            }
        }
    }
}

/// Extracts the child frame's return value: prefers the Return value carried by control_signal,
/// otherwise reads the return_node value.
pub(super) fn extract_child_return(child: &Frame, graph: &DataFlowGraph) -> Value {
    match &child.control_signal {
        ControlSignal::Return(v) => v.clone(),
        ControlSignal::Break | ControlSignal::Continue => Value::VOID,
        ControlSignal::None => {
            let sg = &graph.subgraphs[child.subgraph_id.0 as usize];
            // child.node_offset: cross-function call = subgraph node_range.0;
            // same-function branch frame = parent function node_start (see
            // Frame.rs prepare_same_function_frame).
            // Use child.node_offset (the actual value) rather than sg.node_range.0 so both cases
            // are handled correctly.
            let return_local = NodeId(sg.return_node.0.wrapping_sub(child.node_offset));
            child.get_value(return_local)
        }
    }
}

pub(super) fn run_offloaded_subgraph(graph: &DataFlowGraph, frame: &mut Frame) -> super::Offload::OffloadOutcome {
    // Delegate to the ENGINE'S OWN plan loop (exec_plan_core) — the identical
    // compiled loop the inline path uses. A textually identical copy in this
    // module compiled ~30x slower per node (measured); sharing the one
    // compiled loop guarantees engine parity by construction.
    let plan: &[NodeId] = match graph.linear_plan(frame.subgraph_id.0 as usize) {
        Some(p) if !p.is_empty() => p,
        _ => return super::Offload::OffloadOutcome::Fallback,
    };
    match exec_plan_offload(frame, plan, graph) {
        PlanFlowCore::Done => super::Offload::OffloadOutcome::Done(
            extract_child_return(frame, graph),
            frame.control_signal.clone(),
        ),
        PlanFlowCore::EngineNeeded => super::Offload::OffloadOutcome::Fallback,
    }
}

// =========================================================================
// impl<S: LockStrategy> Engine<S> — scheduling core methods
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// Native stack depth cap for E1 inline synchronous execution. Beyond it, calls fall back to
    /// the queue protocol (which has no native recursion), matching the legacy engine's ability
    /// to run arbitrarily deep recursion. Debug builds use unoptimized multi-KB stack frames, so
    /// the cap is lowered to stay well inside the 1MB Windows main-thread stack (measured: 256
    /// levels overflow in debug, pass in release).
    pub(super) const INLINE_MAX_DEPTH: u32 = if cfg!(debug_assertions) { 48 } else { 256 };

    /// Executes all ready nodes in the frame until the ready queue is empty or the frame suspends.
    ///
    /// `depth` = current inline-call nesting (E1). process_frame enters at 0; each inline
    /// synchronous child runs at depth+1.
    pub(super) fn run_frame_nodes(&self, frame: &mut Frame, fid: FrameId, queue: &QueueHandle<'_>, depth: u32) {
        let graph = frame.graph.clone();

        // L3'': stack of same-frame callee contexts. A leaf call pushes one;
        // its completion (Return or queue exhaustion) pops. Scoped to this
        // dispatch invocation — dropped when the frame finishes.
        let mut l3_saved: Vec<SavedCallCtx> = Vec::new();
        let mut l3_scratch: Vec<L3Scratch> = Vec::new();
        let mut iter_guard: u64 = 0;
        loop {
        iter_guard += 1;
        if iter_guard > 500000 {
            // Over the limit: mark Failed to prevent process_frame from re-enqueuing and causing a
            // livelock. process_frame's Failed branch wakes the caller or returns NULL.
            // L3'': a same-frame callee hitting the guard must not fail the
            // CALLER's frame — restore and continue with a NULL result
            // (mirrors E1's Completed-or-Failed writeback).
            if let Some(ctx) = l3_saved.pop() {
                let cc = graph.downstream_count(
                    (ctx.call_node_local.0 + ctx.node_offset) as usize,
                );
                let call_local = ctx.call_node_local;
                restore_caller_ctx(frame, ctx, &mut l3_scratch);
                let v = std::mem::replace(&mut frame.control_signal, crate::ir::Ir::ControlSignal::None);
                let _ = v;
                frame.set_value(call_local, crate::value::Value::NULL, cc);
                notify_downstream(frame, &graph, call_local, NodeId(call_local.0 + frame.node_offset), NodeId(frame.node_offset));
                iter_guard = 0;
                continue;
            }
            frame.state = FrameState::Failed;
            return;
        }
            // Check the control signal (return/break/continue already triggered).
            if !matches!(frame.control_signal, ControlSignal::None) {
                // L3'': a leaf callee's Return restores the caller context in
                // place and relays the value to the call node — no frame
                // completion, no scheduler round-trip. Cross-function returns
                // are data, not signals, so nothing propagates (same matrix as
                // finish_call_in_caller).
                if let ControlSignal::Return(v) = std::mem::take(&mut frame.control_signal) {
                    if let Some(ctx) = l3_saved.pop() {
                        let cc = graph.downstream_count(
                            (ctx.call_node_local.0 + ctx.node_offset) as usize,
                        );
                        let call_local = ctx.call_node_local;
                        restore_caller_ctx(frame, ctx, &mut l3_scratch);
                        frame.set_value(call_local, v, cc);
                        notify_downstream(frame, &graph, call_local, NodeId(call_local.0 + frame.node_offset), NodeId(frame.node_offset));
                        if !frame.branch_relays.is_empty() {
                            relay_branch_value(frame, &graph, call_local);
                        }
                        iter_guard = 0;
                        continue;
                    }
                    frame.control_signal = ControlSignal::Return(v);
                }
                break;
            }
            // Check whether the frame has been cancelled.
            if frame.state == FrameState::Cancelling {
                break;
            }
            // Check whether the frame has suspended.
            if frame.state == FrameState::Suspended {
                return;
            }

            // POP: pop a ready node (local id).
            let local_id = match frame.pop_ready() {
                Some(n) => n,
                None => {
                    // L3'': queue exhaustion inside a same-frame callee = its
                    // completion (expression bodies never emit Return; the
                    // value sits in the return_node slot). Extract, restore,
                    // relay to the call node.
                    if let Some(ctx) = l3_saved.pop() {
                        let ret = extract_child_return(frame, &graph);
                        let cc = graph.downstream_count(
                            (ctx.call_node_local.0 + ctx.node_offset) as usize,
                        );
                        let call_local = ctx.call_node_local;
                        restore_caller_ctx(frame, ctx, &mut l3_scratch);
                        frame.set_value(call_local, ret, cc);
                        notify_downstream(frame, &graph, call_local, NodeId(call_local.0 + frame.node_offset), NodeId(frame.node_offset));
                        if !frame.branch_relays.is_empty() {
                            relay_branch_value(frame, &graph, call_local);
                        }
                        continue;
                    }
                    break;
                }
            };

            let node_start = frame.node_offset;
            let graph_node_id = NodeId(local_id.0 + node_start);
            let node = graph.node(graph_node_id.0 as usize);
            let ctx = EvalContext { node_start, graph: &graph };

            // COMPUTE: uniformly invoke compute_fn, with no specialization checks.
            let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, graph_node_id, &ctx);

            // MATCH NodeResult: unified side-effect handling.
            match result {
                NodeResult::Value(v) => {
                    let cc = graph.downstream_count(graph_node_id.0 as usize);
                    frame.set_value(local_id, v, cc);
                    notify_downstream(frame, &graph, local_id, graph_node_id, NodeId(node_start));
                    // E7: a same-frame branch's return node just produced the arm
                    // value — relay it to the gate's slot (the dataflow equivalent
                    // of a child frame's completion writeback).
                    if !frame.branch_relays.is_empty() {
                        self.maybe_relay_branch(frame, local_id, &graph);
                    }
                }
                NodeResult::Batch(results) => {
                    for &(lid, ref v) in &results {
                        let gid = NodeId(lid.0 + node_start);
                        let cc = graph.downstream_count(gid.0 as usize);
                        frame.set_value(lid, v.clone(), cc);
                    }
                    for &(lid, _) in &results {
                        frame.ready_queue.retain(|n| *n != lid);
                    }
                    for &(lid, _) in &results {
                        let gid = NodeId(lid.0 + node_start);
                        notify_downstream(frame, &graph, lid, gid, NodeId(node_start));
                    }
                    if !frame.branch_relays.is_empty() {
                        for &(lid, _) in &results {
                            self.maybe_relay_branch(frame, lid, &graph);
                        }
                    }
                }
                NodeResult::Call(pending) => {
                    // Tail-call graph jump.
                    let graph_call_id = NodeId(pending.call_node_local.0 + frame.node_offset);
                    // Same-function branch-frame guard: when this frame IS an if/match branch
                    // (its value table lives in the parent function's layout — node_offset ≠ its
                    // own subgraph range — and its caller launch node is a Gate in that shared
                    // layout), a tail jump would swap the branch frame in place with the callee:
                    // the callee's completion then writes the Gate node directly as data and the
                    // branch's own CF_RETURN never runs — the branch's Return signal is lost and
                    // the function falls through to its tail expression (wrong result). Route
                    // such calls through the normal call path so the branch frame executes its
                    // Return node and propagates the signal properly.
                    let frame_is_branch = frame.node_offset
                        != graph.subgraphs[frame.subgraph_id.0 as usize].node_range.0 .0;
                    // Only same-function branch frames share the caller's node_offset layout,
                    // so the caller launch node is only resolvable (and only relevant) there.
                    let branch_frame_tail = frame_is_branch
                        && frame
                            .caller
                            .map(|(_, cn)| {
                                graph.node((cn.0 + frame.node_offset) as usize).kind
                                    == NodeKind::Gate
                            })
                            .unwrap_or(false);
                    if graph.tail_call_flag(graph_call_id.0 as usize) && !branch_frame_tail {
                        let caller = frame.caller;
                        let propagate_to_parent =
                            if let Some((caller_fid, call_node)) = caller {
                                let frames = self.frames.lock();
                                if let Some(caller_frame) = frames.get(&caller_fid) {
                                    let caller_sg_id = caller_frame.subgraph_id;
                                    let caller_loop_kind =
                                        graph.subgraphs[caller_sg_id.0 as usize].loop_kind;
                                    let caller_has_caller = caller_frame.caller.is_some();
                                    let caller_offset = caller_frame.node_offset;
                                    let caller_graph_node =
                                        NodeId(call_node.0 + caller_offset);
                                    let caller_is_gate = graph.node(caller_graph_node.0
                                        as usize)
                                        .kind
                                        == NodeKind::Gate;
                                    caller_is_gate
                                        && caller_loop_kind
                                            != crate::ir::Ir::LoopKind::LoopBody
                                        && caller_has_caller
                                } else {
                                    false
                                }
                            } else {
                                false
                            };

                        if propagate_to_parent {
                            let (caller_fid, _) = caller.unwrap();
                            let orig_caller = {
                                let mut frames = self.frames.lock();
                                frames.remove(&caller_fid).and_then(|cf| cf.caller)
                            };
                            {
                                let mut ew = self.event_waiters.lock();
                                for bucket in ew.values_mut() {
                                    bucket.retain(|f| *f != caller_fid);
                                }
                            }
                            self.pending_completions.lock().remove(&caller_fid);
                            frame.caller = orig_caller;
                            switch_subgraph(
                                frame,
                                &graph,
                                pending.target_sg,
                                &pending.args,
                            );
                        } else {
                            switch_subgraph(
                                frame,
                                &graph,
                                pending.target_sg,
                                &pending.args,
                            );
                        }
                        continue;
                    }

                    // E7 same-frame branch execution: a same-function branch (match
                    // arm / if arm / short-circuit RHS — no suspension points, no
                    // event sources, non-capturing gate) executes in the CALLER's
                    // frame instead of launching a child frame. The child-frame
                    // launch costs O(parent function) machinery per arm (acquire +
                    // full-table copy + prepare derivation — measured ~1.2µs per
                    // launch, ~6 launches per match_dispatch iteration); the
                    // same-frame launch is O(branch size). Nested gates inside the
                    // branch re-enter this arm and unfold cascades in-frame.
                    // Excluded: suspending/event branches (need real frames),
                    // capture gates (their Return signal must become the gate's
                    // value, not terminate this frame), LoopBody (E2 owns it), and
                    // async spawns.
                    // E7 same-frame branch execution (default on;
                    // the child-frame protocol remains below as the ineligible fallback).
                    // The outer-input snapshot at launch mirrors the child
                    // frame's launch-time copy — without it, branch reads of
                    // enclosing-frame values (loop condition chains) see slots
                    // the E6 boundary delta cleared.
                    if !pending.is_async && pending.closure_val.is_none() {
                        let gate_gid =
                            NodeId(pending.call_node_local.0 + frame.node_offset);
                        if same_frame_branch_ok(&graph, frame, gate_gid, pending.target_sg) {
                            self.launch_same_frame_branch(
                                frame,
                                pending.call_node_local,
                                pending.target_sg,
                                &pending.args,
                            );
                            continue;
                        }
                    }

                    // Scalar-chain fast call: an eligible pure-leaf subgraph with
                    // all-scalar args executes its compiled program in place —
                    // no child frame, no queue round-trip, no compute_fn
                    // dispatch. Bit-identical to the generic path (same
                    // arith kernels); the offload intercept below keeps
                    // precedence when --offload / [engine] offload is active.
                    if !pending.is_async
                        && pending.closure_val.is_none()
                        && self.offload_rt.is_none()
                    {
                        if let Some(prog) =
                            graph.scalar_prog(pending.target_sg.0 as usize)
                        {
                            if pending
                                .args
                                .iter()
                                .all(|v| matches!(v, crate::value::Value::Scalar(..)))
                            {
                                let value =
                                    crate::pass::Scalarizer::run_scalar_prog(&prog, &pending.args);
                                let node_start = frame.node_offset;
                                let graph_node_id =
                                    NodeId(pending.call_node_local.0 + node_start);
                                let consumer_count =
                                    graph.downstream_count(graph_node_id.0 as usize);
                                frame.set_value(
                                    pending.call_node_local,
                                    value,
                                    consumer_count,
                                );
                                notify_downstream(
                                    frame,
                                    &graph,
                                    pending.call_node_local,
                                    graph_node_id,
                                    NodeId(node_start),
                                );
                                if !frame.branch_relays.is_empty() {
                                    relay_branch_value(
                                        frame,
                                        &graph,
                                        pending.call_node_local,
                                    );
                                }
                                continue;
                            }
                        }
                    }

                    // LoopBody invocation.
                    let target_loop_kind =
                        graph.subgraphs[pending.target_sg.0 as usize].loop_kind;
                    let is_loop_body = target_loop_kind
                        == crate::ir::Ir::LoopKind::LoopBody;
                    // E1: non-LoopBody synchronous calls run inline (frame built by
                    // start_subgraph_frame below, never registered in the frames map unless it
                    // suspends). Async spawns and depth-exceeded calls keep the queue protocol.
                    let inline_sync = !is_loop_body
                        && !pending.is_async
                        && depth < Self::INLINE_MAX_DEPTH;
                    if is_loop_body
                        && !graph.subgraphs[pending.target_sg.0 as usize].has_suspend
                        && depth < Self::INLINE_MAX_DEPTH
                    {
                        // E2 loop hot path: drive the body on the current stack, iteration after
                        // iteration, without queue round-trips. Eligibility: body sg has no
                        // suspension point (a suspending body falls back to the queue protocol —
                        // the cached_child_frame mechanism then owns it) and inline depth headroom.
                        // Semantics mirror complete_and_wake_caller's LoopBody arms exactly:
                        // Break/Return exit (+ Bug G defer drain), TailRec base case on None,
                        // Continue/None reset via reset_loop_iteration.
                        let (body_fid, mut body) =
                            if let Some((bfid, b)) = frame.hot_body.take() {
                                (bfid, b)
                            } else if let Some(bfid) = frame.cached_child_frame {
                                match self.frames.lock().remove(&bfid) {
                                    Some(b) => (bfid, b),
                                    None => self.start_subgraph_frame(
                                        fid,
                                        pending.call_node_local,
                                        pending.target_sg,
                                        &pending.args,
                                        frame,
                                        pending.closure_val.clone(),
                                    ),
                                }
                            } else {
                                self.start_subgraph_frame(
                                    fid,
                                    pending.call_node_local,
                                    pending.target_sg,
                                    &pending.args,
                                    frame,
                                    pending.closure_val.clone(),
                                )
                            };

                        // Per-iteration argument injection + chain wiring (same as the map-reuse
                        // path: values set, notify_downstream, Bug #100 in-hand parent pointers).
                        {
                            let target_sg =
                                &graph.subgraphs[pending.target_sg.0 as usize];
                            let param_count = target_sg.param_count as usize;
                            let parent_start = body.node_offset;
                            let branch_start = target_sg.node_range.0 .0;
                            let param_local_offset =
                                (branch_start.wrapping_sub(parent_start)) as usize;
                            for (i, arg) in
                                pending.args.iter().enumerate().take(param_count)
                            {
                                let local_id =
                                    NodeId((param_local_offset + i) as u32);
                                let gid = (branch_start as usize) + i;
                                let global_id = NodeId(gid as u32);
                                let consumer_count =
                                    graph.downstream_count(gid);
                                body.set_value(local_id, arg.clone(), consumer_count);
                                notify_downstream(
                                    &mut body,
                                    &graph,
                                    local_id,
                                    global_id,
                                    NodeId(parent_start),
                                );
                            }
                            body.caller = Some((fid, pending.call_node_local));
                            {
                                let parent_ptr =
                                    frame as *const Frame as *mut Frame;
                                body.parent_frame_ptr = parent_ptr;
                                body.root_frame_ptr = if !frame.root_frame_ptr.is_null() {
                                    frame.root_frame_ptr
                                } else {
                                    parent_ptr
                                };
                            }
                            body.state = FrameState::Ready;
                        }

                        self.run_frame_dispatch(&mut body, body_fid, queue, depth + 1);

                        if body.state == FrameState::Suspended {
                            // Body awaits/selects: hand it to the queue protocol (legacy
                            // suspend), where its completion re-enters
                            // complete_and_wake_caller's LoopBody handling.
                            // Offload waits push NO queue entry: the offload
                            // delivery is the guaranteed requeuer (it applies
                            // the completion and requeues the caller). A
                            // suspend-time push would only produce a no-op
                            // dispatch while the worker is still computing.
                            // Predicate: the waited-for child never entered
                            // the frames map (offload children live on the
                            // worker; queue-protocol children are in-map).
                            let offload_wait = matches!(
                                body.suspend_state,
                                SuspendState::WaitingSubgraph(k)
                                    if !self.frames.lock().contains_key(&k)
                            );
                            self.frames.lock().insert(body_fid, body);
                            frame.cached_child_frame = Some(body_fid);
                            if !offload_wait {
                                queue.push(body_fid);
                            }
                                                        self.event_waiters
                                .lock()
                                .entry(RuntimeEvent::SubgraphComplete(body_fid))
                                .or_default()
                                .push(fid);
                            frame.state = FrameState::Suspended;
                            frame.suspend_state =
                                SuspendState::WaitingSubgraph(body_fid);
                            frame.suspend_event =
                                Some(RuntimeEvent::SubgraphComplete(body_fid));
                            return;
                        }

                        match body.control_signal.clone() {
                            ControlSignal::Break
                            | ControlSignal::Return(_) => {
                                let signal = body.control_signal.clone();
                                frame.cached_child_frame = None;
                                frame.control_signal = signal.clone();
                                // Bug G: break/return exits the loop without entering the
                                // void_sg, so CF_DEFER_RUN never drains the loop frame's
                                // defer_stack — drain here (LIFO), as complete_and_wake_caller
                                // does.
                                if !frame.defer_stack.is_empty() {
                                    let defers: Vec<crate::ir::Ir::RuntimeDefer> =
                                        frame.defer_stack.drain(..).collect();
                                    crate::ir::Compute::run_defer_entries_sync(
                                        frame,
                                        &defers,
                                        &graph,
                                    );
                                }
                                self.release_frame(body);
                                // control_signal set: the outer loop breaks and the defer tail
                                // runs; upstream propagation happens through the frame's own
                                // completion path.
                                continue;
                            }
                            ControlSignal::Continue => {
                                self.reset_loop_iteration(frame, fid, &mut body);
                                frame.hot_body = Some((body_fid, body));
                                // A finished body iteration is provable progress, not a
                                // livelock: reset the node-pop guard (long loops legitimately
                                // exceed 500k pops inside one run_frame_nodes invocation).
                                iter_guard = 0;
                                continue;
                            }
                            ControlSignal::None => {
                                let loop_kind = graph.subgraphs
                                    [frame.subgraph_id.0 as usize]
                                    .loop_kind;
                                if loop_kind == crate::ir::Ir::LoopKind::TailRec {
                                    // TailRec base case: the body's return value is the
                                    // loop's result.
                                    let return_value = extract_child_return(&body, &graph);
                                    frame.cached_child_frame = None;
                                    frame.control_signal =
                                        ControlSignal::Return(return_value);
                                    self.release_frame(body);
                                    continue;
                                }
                                self.reset_loop_iteration(frame, fid, &mut body);
                                frame.hot_body = Some((body_fid, body));
                                iter_guard = 0;
                                continue;
                            }
                        }
                    }

                    let child_fid = if is_loop_body {
                        if let Some(bfid) = frame.cached_child_frame {
                            let target_sg =
                                &graph.subgraphs[pending.target_sg.0 as usize];
                            let param_count = target_sg.param_count as usize;
                            let mut body_frame = self.frames.lock().remove(&bfid);
                            if let Some(bf) = body_frame.as_mut() {
                                let parent_start = bf.node_offset;
                                let branch_start = target_sg.node_range.0 .0;
                                let param_local_offset =
                                    (branch_start.wrapping_sub(parent_start)) as usize;
                                for (i, arg) in
                                    pending.args.iter().enumerate().take(param_count)
                                {
                                    let local_id =
                                        NodeId((param_local_offset + i) as u32);
                                    let gid = (branch_start as usize) + i;
                                    let global_id = NodeId(gid as u32);
                                    let consumer_count =
                                        graph.downstream_count(gid);
                                    bf.set_value(local_id, arg.clone(), consumer_count);
                                    // Do not push_ready: the parameter value is already set;
                                    // notify_downstream propagates it downstream.
                                    notify_downstream(bf, &graph, local_id, global_id, NodeId(parent_start));
                                }
                                bf.caller = Some((fid, pending.call_node_local));
                                // Bug #100: keep the frame chain connected on reuse.
                                // Nulling parent_frame_ptr here orphaned the body's
                                // WriteBacks (setup_frame_chain cannot restore them:
                                // the caller is mid-processing, absent from the map),
                                // so loop-variable updates never reached the loop frame
                                // and the condition re-read a stale snapshot forever.
                                // Box addresses are stable across remove/insert (see
                                // process_frame), so the in-hand frame pointer is safe.
                                {
                                    let parent_ptr = frame as *const Frame as *mut Frame;
                                    bf.parent_frame_ptr = parent_ptr;
                                    bf.root_frame_ptr = if !frame.root_frame_ptr.is_null() {
                                        frame.root_frame_ptr
                                    } else {
                                        parent_ptr
                                    };
                                }
                                bf.state = FrameState::Ready;
                            }
                            if let Some(bf) = body_frame {
                                self.frames.lock().insert(bfid, bf);
                            }
                            bfid
                        } else {
                            let bfid = self.start_subgraph(
                                fid,
                                pending.call_node_local,
                                pending.target_sg,
                                &pending.args,
                                frame,
                                pending.closure_val.clone(),
                            );
                            frame.cached_child_frame = Some(bfid);
                            bfid
                        }
                    } else if self.try_launch_offload(fid, &pending, frame) {
                        // L2: pure-heavy leaf executing on a worker with
                        // deep-copied args; the caller is suspended and the
                        // completion arrives via the offload sequencer. Takes
                        // precedence over inline execution (offload eligibility
                        // implies a heavy node count where parallelism pays).
                        return;
                    } else if inline_sync {
                        // Placeholder: the inline path below builds the frame itself (without a
                        // frames-map insert). This arm must not be reachable for is_async.
                        FrameId(u32::MAX)
                    } else {
                        self.start_subgraph(
                            fid,
                            pending.call_node_local,
                            pending.target_sg,
                            &pending.args,
                            frame,
                            pending.closure_val.clone(),
                        )
                    };

                    if pending.is_async {
                        debug_assert!(!inline_sync);
                        let async_id = self
                            .async_join_runtime
                            .lock()
                            .alloc_and_register(child_fid);
                        let async_handle = Value::i32(async_id.0 as i32);
                        queue.push(child_fid);
                        let node_start = frame.node_offset;
                        let graph_node_id =
                            NodeId(pending.call_node_local.0 + node_start);
                        let consumer_count =
                            graph.downstream_count(graph_node_id.0 as usize);
                        frame.set_value(
                            pending.call_node_local,
                            async_handle,
                            consumer_count,
                        );
                        notify_downstream(
                            frame,
                            &graph,
                            pending.call_node_local,
                            graph_node_id,
                            NodeId(node_start),
                        );
                        if !frame.branch_relays.is_empty() {
                            relay_branch_value(frame, &graph, pending.call_node_local);
                        }
                        continue;
                    } else if inline_sync
                        && pending.closure_val.is_none()
                        && graph.callee_same_frame(pending.target_sg.0 as usize)
                    {
                        // L3'': leaf straight-line callee — execute it in THIS
                        // frame (context swap via the tail-call switch machine)
                        // instead of a child-frame launch. Completion is
                        // intercepted at the Return signal or queue exhaustion.
                        l3_saved.push(enter_same_frame_callee(
                            frame,
                            &graph,
                            pending.target_sg,
                            &pending.args,
                            pending.call_node_local,
                            &mut l3_scratch,
                        ));
                        // Linear-planned callee: run its plan to completion
                        // right here (the E9 fast path the pooled E1 child
                        // gets); completion then lands in the interceptors
                        // (Return at loop top / queue-exhaustion at pop). A
                        // Bailed plan simply falls back to the queue loop.
                        if let Some(bplan) = graph.linear_plan(pending.target_sg.0 as usize) {
                            if !bplan.is_empty() {
                                let _ = self.exec_plan(frame, fid, queue, depth, bplan, &graph);
                            }
                        }
                        continue;
                    } else if inline_sync {
                        // E1: inline synchronous execution. The child frame runs to completion on
                        // the current stack — no frames-map insert, no queue round-trip, the caller
                        // never suspends. On any suspend (await/select/defer-waiter) the exact
                        // legacy queue protocol is re-established below (child registered + caller
                        // suspended on SubgraphComplete), so every wakeup path keeps working.
                        let (child_fid, mut child) = self.start_subgraph_frame(
                            fid,
                            pending.call_node_local,
                            pending.target_sg,
                            &pending.args,
                            frame,
                            pending.closure_val.clone(),
                        );
                        self.run_frame_dispatch(&mut child, child_fid, queue, depth + 1);
                        if child.state == FrameState::Suspended {
                            // Suspend fallback: hand the child to the queue exactly like the
                            // legacy suspend path (Bug #78's pending_completions race resolution
                            // in process_frame then applies verbatim).
                            self.frames.lock().insert(child_fid, child);
                            queue.push(child_fid);
                                                        self.event_waiters
                                .lock()
                                .entry(RuntimeEvent::SubgraphComplete(child_fid))
                                .or_default()
                                .push(fid);
                            frame.state = FrameState::Suspended;
                            frame.suspend_state = SuspendState::WaitingSubgraph(child_fid);
                            frame.suspend_event =
                                Some(RuntimeEvent::SubgraphComplete(child_fid));
                            return;
                        }
                        // Completed or Failed: write the result back into the caller (return
                        // value, signal propagation, downstream notify) and keep executing it.
                        super::Subgraph::finish_call_in_caller(
                            frame,
                            pending.call_node_local,
                            &child,
                            &graph,
                        );
                        self.release_frame(child);
                        continue;
                    } else {
                        queue.push(child_fid);
                        self.event_waiters
                            .lock()
                            .entry(RuntimeEvent::SubgraphComplete(child_fid))
                            .or_default()
                            .push(fid);
                        frame.state = FrameState::Suspended;
                        frame.suspend_state = SuspendState::WaitingSubgraph(child_fid);
                        frame.suspend_event =
                            Some(RuntimeEvent::SubgraphComplete(child_fid));
                        return;
                    }
                }
                NodeResult::Await(pending) => {
                    let (event, ready_value, await_node_local) =
                        self.resolve_check_and_register_await(&pending, fid);

                    if let Some(value) = ready_value {
                        let node_start = frame.node_offset;
                        let graph_node_id =
                            NodeId(await_node_local.0 + node_start);
                        let consumer_count =
                            graph.downstream_count(graph_node_id.0 as usize);
                        frame.set_value(await_node_local, value, consumer_count);
                        notify_downstream(
                            frame,
                            &graph,
                            await_node_local,
                            graph_node_id,
                            NodeId(node_start),
                        );
                        if !frame.branch_relays.is_empty() {
                            relay_branch_value(frame, &graph, await_node_local);
                        }
                        continue;
                    } else {
                        frame.state = FrameState::Suspended;
                        frame.suspend_state =
                            SuspendState::WaitingEvent(await_node_local);
                        frame.suspend_event = Some(event);
                        return;
                    }
                }
                NodeResult::ChannelNotify(ch_id) => {
                    self.on_event_arrived(
                        RuntimeEvent::ChannelReady(ch_id),
                        Value::VOID,
                        queue,
                    );
                    // After a successful send we still must set the node value + notify downstream,
                    // otherwise subsequent statements will never become ready.
                    let consumer_count =
                        graph.downstream_count(graph_node_id.0 as usize);
                    frame.set_value(local_id, Value::VOID, consumer_count);
                    notify_downstream(
                        frame,
                        &graph,
                        local_id,
                        graph_node_id,
                        NodeId(node_start),
                    );
                }
                NodeResult::Cancel(async_id) => {
                    let child_fid = self
                        .async_join_runtime
                        .lock()
                        .find_child_by_async_id(async_id);
                    if let Some(child_fid) = child_fid {
                        self.cancel_frame(child_fid, queue);
                    }
                    let consumer_count =
                        graph.downstream_count(graph_node_id.0 as usize);
                    frame.set_value(local_id, Value::VOID, consumer_count);
                    notify_downstream(
                        frame,
                        &graph,
                        local_id,
                        graph_node_id,
                        NodeId(node_start),
                    );
                }
                NodeResult::SelectWait(gate_local) => {
                    let info = graph.select_info_at(graph_node_id.0 as usize);

                    if let Some(info) = info {
                        // Tracks the selected branch: (subgraph_id, event_kind, channel value if Channel).
                        let mut ready_branch: Option<(SubGraphId, EventSourceKind, Option<Value>)> = None;
                        for (branch_idx, branch) in info.branches.iter().enumerate() {
                            let event_val =
                                frame.get_value_by_global(branch.event_source_node);
                            let is_ready = match branch.event_kind {
                                EventSourceKind::Channel => {
                                    event_val
                                        .heap_obj()
                                        .and_then(|h| h.channel())
                                        .map_or(false, |ch| ch.has_data() || ch.is_closed())
                                }
                                EventSourceKind::Timer => {
                                    let timer_id = {
                                        if let Some((_, tid)) = frame
                                            .select_timers
                                            .iter()
                                            .find(|(idx, _)| *idx == branch_idx)
                                        {
                                            *tid
                                        } else {
                                            let duration_ns = event_val.as_i64();
                                            let tid = self.timer_runtime.lock().start(
                                                std::time::Duration::from_nanos(
                                                    duration_ns as u64,
                                                ),
                                            );
                                            frame.select_timers.push((branch_idx, tid));
                                            tid
                                        }
                                    };
                                    self.timer_runtime.lock().is_fired(timer_id)
                                }
                                _ => false,
                            };
                            if is_ready {
                                // For a Channel branch, consume the value now (recv) so it is bound
                                // to the arm's binding inside the body. The branch subgraph's first
                                // param node receives this value. A closed/empty channel yields Null.
                                let recv_val = if branch.event_kind == EventSourceKind::Channel {
                                    Some(
                                        event_val
                                            .heap_obj()
                                            .and_then(|h| h.channel())
                                            .and_then(|ch| ch.recv())
                                            .unwrap_or(Value::Null),
                                    )
                                } else {
                                    None
                                };
                                ready_branch = Some((branch.subgraph_id, branch.event_kind, recv_val));
                                break;
                            }
                        }

                        if let Some((sg_id, ev_kind, recv_val)) = ready_branch {
                            // Channel branch: pass the recv'd value as arg[0] (bound to the arm's
                            // binding). Timer/other branches: no args.
                            let args: Vec<Value> = if ev_kind == EventSourceKind::Channel {
                                vec![recv_val.unwrap_or(Value::Null)]
                            } else {
                                Vec::new()
                            };
                            let child_fid =
                                self.start_subgraph(fid, gate_local, sg_id, &args, frame, None);
                            queue.push(child_fid);
                            self.event_waiters
                                .lock()
                                .entry(RuntimeEvent::SubgraphComplete(child_fid))
                                .or_default()
                                .push(fid);
                            frame.state = FrameState::Suspended;
                            frame.suspend_state =
                                SuspendState::WaitingSubgraph(child_fid);
                            frame.suspend_event =
                                Some(RuntimeEvent::SubgraphComplete(child_fid));
                            return;
                        } else {
                            for (branch_idx, branch) in info.branches.iter().enumerate() {
                                let event_val = frame
                                    .get_value_by_global(branch.event_source_node);
                                let event = match branch.event_kind {
                                    EventSourceKind::Channel => {
                                        if let Some(ch) = event_val
                                            .heap_obj()
                                            .and_then(|h| h.channel())
                                        {
                                            RuntimeEvent::ChannelReady(
                                                crate::ir::Ir::ChannelId(ch.id()),
                                            )
                                        } else {
                                            continue;
                                        }
                                    }
                                    EventSourceKind::Timer => {
                                        let timer_id = frame
                                            .select_timers
                                            .iter()
                                            .find(|(idx, _)| *idx == branch_idx)
                                            .map(|(_, tid)| *tid)
                                            .expect(
                                                "select timer should be started above",
                                            );
                                        RuntimeEvent::TimerFired(timer_id)
                                    }
                                    _ => continue,
                                };
                                self.event_waiters
                                    .lock()
                                    .entry(event)
                                    .or_default()
                                    .push(fid);
                            }
                            frame.state = FrameState::Suspended;
                            frame.suspend_state =
                                SuspendState::WaitingEvent(gate_local);
                            frame.suspend_event = None;
                            return;
                        }
                    }
                }
                NodeResult::Return(v) => {
                    frame.control_signal = ControlSignal::Return(v);
                    break;
                }
                NodeResult::Break => {
                    frame.control_signal = ControlSignal::Break;
                    break;
                }
                NodeResult::Continue => {
                    frame.control_signal = ControlSignal::Continue;
                    break;
                }
            }
        }

        self.finish_frame(frame, fid, queue, depth, &graph);
    }

    /// Frame termination handling shared by the dataflow runner and the E5 linear runner:
    /// suspend passthrough, Cancelling cleanup (defer + Failed), LIFO defer execution
    /// (Bug #77 defer-waiter accounting), and the final Completed transition.
    fn finish_frame(
        &self,
        frame: &mut Frame,
        fid: FrameId,
        queue: &QueueHandle<'_>,
        depth: u32,
        _graph: &DataFlowGraph,
    ) {
        // Frame suspended: do not execute defer, do not mark Completed.
        if frame.state == FrameState::Suspended {
            return;
        }

        // Frame cancelled: execute defer cleanup + mark Failed (spec 5.3).
        if frame.state == FrameState::Cancelling {
            let defer_entries: Vec<crate::ir::Ir::RuntimeDefer> =
                std::mem::take(&mut frame.defer_stack);
            for entry in defer_entries.iter().rev() {
                let defer_fid = self.init_defer_frame(entry.body_subgraph, frame);
                let mut defer_frame = self.frames.lock().remove(&defer_fid);
                if let Some(df) = defer_frame.as_deref_mut() {
                    self.run_frame_nodes(df, defer_fid, queue, depth);
                }
                if let Some(df) = defer_frame {
                    if df.state != FrameState::Completed {
                        self.frames.lock().insert(defer_fid, df);
                    }
                }
            }
            frame.state = FrameState::Failed;
            return;
        }

        // Execute defer (LIFO): any termination path drains the frame's runtime
        // defer_stack. Only defers whose registration node actually EXECUTED are
        // on the stack — unreached defers (error `?`-exit before their statement)
        // never registered and never run, so they can no longer read unbound
        // slots (the old static defer_table was drained unconditionally and
        // crashed natively on that path).
        let defer_entries: Vec<crate::ir::Ir::RuntimeDefer> =
            std::mem::take(&mut frame.defer_stack);
        let mut pending_defer_count: u32 = 0;
        for entry in defer_entries.iter().rev() {
            let defer_fid = self.init_defer_frame(entry.body_subgraph, frame);
            let mut defer_frame = self.frames.lock().remove(&defer_fid);
            if let Some(df) = defer_frame.as_deref_mut() {
                self.run_frame_nodes(df, defer_fid, queue, depth);
            }
            if let Some(df) = defer_frame {
                if df.state != FrameState::Completed {
                    pending_defer_count += 1;
                    self.frames.lock().insert(defer_fid, df);
                } else {
                    // Defer frame finished synchronously: unregister it so process_frame's
                    // Completed branch does not mis-route it as a defer-waiter wakeup.
                    self.defer_frames.lock().remove(&defer_fid);
                }
            }
        }

        if pending_defer_count > 0 {
            // Bug #77: some defer frames are still running (suspended on calls/awaits inside the
            // defer body). The frame must wait for all defer frames to finish before it can be
            // marked Completed. Suspend the frame and register a defer-waiter; each defer frame's
            // completion (in process_frame) decrements the count and, when it reaches zero, resumes
            // this frame for the final Completed transition.
            self.defer_waiters.lock().insert(fid, pending_defer_count);
            frame.state = FrameState::Suspended;
            frame.suspend_state = SuspendState::WaitingSubgraph(FrameId(0));
            frame.suspend_event = None;
            return;
        }

        // Mark the frame completed.
        frame.state = FrameState::Completed;
    }

    /// E5 dispatch: fresh frames with a linearized plan run linearly (no readiness machinery);
    /// everything else runs through the dataflow engine. `linear_fresh` is one-shot.
    pub(super) fn run_frame_dispatch(&self, frame: &mut Frame, fid: FrameId, queue: &QueueHandle<'_>, depth: u32) {
        if frame.linear_fresh {
            frame.linear_fresh = false;
            if let Some(plan) = self.graph.linear_plan(frame.subgraph_id.0 as usize) {
                if !plan.is_empty() {
                    self.run_linear(frame, fid, plan, queue, depth);
                    return;
                }
            }
        }
        self.run_frame_nodes(frame, fid, queue, depth);
    }

    /// E5 linear runner: executes the sg's own nodes in precomputed topological order —
    /// no pending_inputs countdown, no ready queue, no notify_downstream (values live until
    /// frame end, the documented frame-level fallback semantics). Launch nodes (Gate/Call/
    /// Await/EventSource — is_launch_kind) bail to the dataflow engine, which rebuilds the
    /// readiness state for the remaining nodes and continues seamlessly.
    fn run_linear(
        &self,
        frame: &mut Frame,
        fid: FrameId,
        plan: &[NodeId],
        queue: &QueueHandle<'_>,
        depth: u32,
    ) {
        let graph = frame.graph.clone();

        // The frame's ready queue holds prepare-time seeds (and E2 arg-injection
        // pushes) for OWN nodes — all of which the plan itself runs. They must
        // not survive into the post-gate drain: a stale seed re-executes its
        // node, whose notify can re-arm (and re-fire) an already-executed gate
        // — double same-frame launches and mid-read slot clears.
        frame.ready_queue.clear();

        match self.exec_plan(frame, fid, queue, depth, plan, &graph) {
            PlanFlow::Done => {
                self.finish_frame(frame, fid, queue, depth, &graph);
            }
            PlanFlow::Bailed => {
                rebuild_linear_bailout(frame, &graph);
                self.run_frame_nodes(frame, fid, queue, depth);
            }
        }
    }

    /// E9 recursive segmented-linear executor: runs plan nodes directly (no
    /// queue, no pending countdown, no notify). At a Gate, launches the taken
    /// branch same-frame and executes the BRANCH's own linear plan the same
    /// way (recursion); branches without a plan fall back to the queue-based
    /// drain. Relay check on every value produced: a branch's return node
    /// fires its (ret → gate) relay inline.
    fn exec_plan(
        &self,
        frame: &mut Frame,
        fid: FrameId,
        queue: &QueueHandle<'_>,
        depth: u32,
        plan: &[NodeId],
        graph: &DataFlowGraph,
    ) -> PlanFlow {
        let node_start = frame.node_offset;
        for &gid in plan {
            if !matches!(frame.control_signal, ControlSignal::None) {
                return PlanFlow::Done;
            }
            if frame.state == FrameState::Cancelling || frame.state == FrameState::Suspended {
                return PlanFlow::Done;
            }
            let local = NodeId(gid.0.wrapping_sub(node_start));
            if frame.value_table.is_ready(local.0 as usize) {
                continue;
            }
            let node = graph.node(gid.0 as usize);
            let ctx = EvalContext { node_start, graph };
            let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, gid, &ctx);
            match result {
                NodeResult::Value(v) => {
                    let cc = graph.downstream_count(gid.0 as usize);
                    frame.set_value(local, v, cc);
                    if !frame.branch_relays.is_empty() {
                        self.maybe_relay_branch(frame, local, graph);
                    }
                }
                NodeResult::Batch(results) => {
                    for &(lid, ref v) in &results {
                        let g2 = lid.0 + node_start;
                        let cc = graph.downstream_count(g2 as usize);
                        frame.set_value(lid, v.clone(), cc);
                    }
                    if !frame.branch_relays.is_empty() {
                        for &(lid, _) in &results {
                            self.maybe_relay_branch(frame, lid, graph);
                        }
                    }
                }
                NodeResult::Return(v) => {
                    frame.control_signal = ControlSignal::Return(v);
                    return PlanFlow::Done;
                }
                NodeResult::Break => {
                    frame.control_signal = ControlSignal::Break;
                    return PlanFlow::Done;
                }
                NodeResult::Continue => {
                    frame.control_signal = ControlSignal::Continue;
                    return PlanFlow::Done;
                }
                NodeResult::Call(pending) => {
                    // E9 segmented-linear: a Gate at its topo position. The plan
                    // verifier guaranteed same-frame eligibility statically; the
                    // runtime re-check is defense in depth.
                    let gate_gid = NodeId(pending.call_node_local.0 + node_start);
                    if !pending.is_async
                        && pending.closure_val.is_none()
                        && same_frame_branch_ok(graph, frame, gate_gid, pending.target_sg)
                    {
                        self.launch_same_frame_branch(
                            frame,
                            pending.call_node_local,
                            pending.target_sg,
                            &pending.args,
                        );
                        // Branch's own linear plan → direct recursive execution;
                        // otherwise the queue-based drain (arms with Calls etc.).
                        if let Some(bplan) = graph.linear_plan(pending.target_sg.0 as usize) {
                            if !bplan.is_empty() {
                                match self.exec_plan(frame, fid, queue, depth, bplan, graph) {
                                    PlanFlow::Done => {}
                                    PlanFlow::Bailed => return PlanFlow::Bailed,
                                }
                            }
                        } else {
                            let (ibs, ibe) =
                                graph.subgraphs[pending.target_sg.0 as usize].node_range;
                            if !self.drain_same_frame(frame, graph, (ibs.0, ibe.0)) {
                                return PlanFlow::Bailed;
                            }
                        }
                        continue;
                    }
                    return PlanFlow::Bailed;
                }
                _ => {
                    // Engine-needing result (Await/ChannelNotify/Cancel/SelectWait):
                    // bail — the dataflow engine re-drives it.
                    return PlanFlow::Bailed;
                }
            }
        }
        PlanFlow::Done
    }

    /// E9 drain: executes the same-frame-injected subtree (branch seeds, plus
    /// anything further launched from it — nested gates recurse through
    /// `launch_same_frame_branch`) until this frame's ready queue is empty.
    /// Returns false when the remainder needs the dataflow driver (a Call that
    /// is not same-frame eligible — cross-function calls, suspends, capture
    /// shapes): the caller bails via rebuild_linear_bailout.
    ///
    /// A scoped re-implementation of run_frame_nodes' steady-state loop minus
    /// suspension bookkeeping; control signals (arm return/break/continue)
    /// stop the drain and propagate to the plan loop / frame.
    fn drain_same_frame(&self, frame: &mut Frame, graph: &DataFlowGraph, initial_range: (u32, u32)) -> bool {
        let node_start = frame.node_offset;
        // Range filter: linear-frame pending accounting is NOT consistent (the
        // plan sets values silently; notifies from launched branches decrement
        // stale pendings), so a notify can spuriously arm a PLAN node and push
        // it. Executing it here (before its real inputs are ready) reads
        // garbage. Only nodes INSIDE same-frame-launched branch ranges belong
        // to the drain; everything else is deferred (re-queued untouched —
        // the plan runs plan nodes; stale entries die at frame end).
        let mut ranges: Vec<(u32, u32)> = vec![initial_range];
        let mut deferred: Vec<NodeId> = Vec::new();
        while let Some(local) = frame.pop_ready() {
            if !matches!(frame.control_signal, ControlSignal::None) {
                break;
            }
            if frame.state == FrameState::Cancelling || frame.state == FrameState::Suspended {
                break;
            }
            let gid = NodeId(local.0 + node_start);
            if !ranges.iter().any(|&(a, b)| gid.0 >= a && gid.0 < b) {
                deferred.push(local);
                continue;
            }
            let node = graph.node(gid.0 as usize);
            let ctx = EvalContext { node_start, graph };
            let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, gid, &ctx);
            match result {
                NodeResult::Value(v) => {
                    let cc = graph.downstream_count(gid.0 as usize);
                    frame.set_value(local, v, cc);
                    notify_downstream(frame, graph, local, gid, NodeId(node_start));
                    if !frame.branch_relays.is_empty() {
                        self.maybe_relay_branch(frame, local, graph);
                    }
                }
                NodeResult::Batch(results) => {
                    for &(lid, ref v) in &results {
                        let g2 = lid.0 + node_start;
                        let cc = graph.downstream_count(g2 as usize);
                        frame.set_value(lid, v.clone(), cc);
                    }
                    for &(lid, _) in &results {
                        frame.ready_queue.retain(|n| *n != lid);
                    }
                    for &(lid, _) in &results {
                        let g2 = NodeId(lid.0 + node_start);
                        notify_downstream(frame, graph, lid, g2, NodeId(node_start));
                    }
                    if !frame.branch_relays.is_empty() {
                        for &(lid, _) in &results {
                            self.maybe_relay_branch(frame, lid, graph);
                        }
                    }
                }
                NodeResult::Call(pending) => {
                    let gate_gid = NodeId(pending.call_node_local.0 + node_start);
                    if !pending.is_async
                        && pending.closure_val.is_none()
                        && same_frame_branch_ok(graph, frame, gate_gid, pending.target_sg)
                    {
                        let (nbs, nbe) =
                            graph.subgraphs[pending.target_sg.0 as usize].node_range;
                        ranges.push((nbs.0, nbe.0));
                        self.launch_same_frame_branch(
                            frame,
                            pending.call_node_local,
                            pending.target_sg,
                            &pending.args,
                        );
                        // seeds were pushed — keep draining
                    } else {
                        return false;
                    }
                }
                NodeResult::Return(v) => {
                    frame.control_signal = ControlSignal::Return(v);
                    for d in deferred {
                        frame.push_ready(d);
                    }
                    return true;
                }
                NodeResult::Break => {
                    frame.control_signal = ControlSignal::Break;
                    for d in deferred {
                        frame.push_ready(d);
                    }
                    return true;
                }
                NodeResult::Continue => {
                    frame.control_signal = ControlSignal::Continue;
                    for d in deferred {
                        frame.push_ready(d);
                    }
                    return true;
                }
                _ => {
                    for d in deferred {
                        frame.push_ready(d);
                    }
                    return false;
                }
            }
        }
        for d in deferred {
            frame.push_ready(d);
        }
        true
    }

    /// Processes one frame: timer check + run_frame_nodes + state transition.
    /// Returns (); the result is communicated via `self.result.lock()`.
    pub(super) fn process_frame(&self, fid: FrameId, queue: &QueueHandle<'_>) {
        // Check for timer events.
        self.check_timers(queue);

        // Take out the frame (keep it boxed: the heap address stays stable across the
        // remove/insert cycle, so parent_frame_ptr/root_frame_ptr held by other frames do not
        // dangle).
        let mut frame_box = match self.frames.lock().remove(&fid) {
            Some(b) => b,
            None => return,
        };
        let frame: &mut Frame = &mut *frame_box;

        // Set up frame-chain pointers: walk the caller chain in the HashMap to set
        // parent_frame_ptr/root_frame_ptr. All parent frames are still in the HashMap at this point
        // (Box addresses are stable).
        self.setup_frame_chain(frame);

        // Execute the frame's ready nodes (lock-free).
        self.run_frame_dispatch(frame, fid, queue, 0);

        // Handle the frame state.
        let state = frame.state;
        let has_caller = frame.caller.is_some();

        match state {
            FrameState::Suspended => {
                // Bug #77: if this frame is a defer-waiter (suspended waiting for its defer
                // frames to complete), do not process pending_completions/pending_events — it
                // must only be resumed when all its defer frames finish (handled in the
                // defer-frame Completed/Failed branches below).
                let is_defer_waiter = self.defer_waiters.lock().contains_key(&fid);
                if is_defer_waiter {
                    self.frames.lock().insert(fid, frame_box);
                    return;
                }
                let event = frame.suspend_event;
                // Insert the frame FIRST, then drain the stashed completions/events.
                // Both rendezvous paths (complete_and_wake_caller / on_event_arrived)
                // stash ONLY when the frame is absent from the map; a pre-insert check
                // races against a stash landing between the (empty) check and the
                // insert — the stash is then never consumed and the frame sleeps
                // forever (the second Multi await-loop hang window; the events side
                // was already fixed this way, the completions side was not).
                self.frames.lock().insert(fid, frame_box);
                // Both stash guards are scoped to their block expressions: released
                // before the take-back below re-acquires frames (nesting either stash
                // lock -> frames deadlocks against the stash writers' frames -> stash
                // paths). The take-back uses a plain `let` — an if-let scrutinee guard
                // would live through the whole body and self-deadlock the
                // non-reentrant parking_lot mutex when the body re-acquires frames.
                let completions: Vec<_> = {
                    let mut pc = self.pending_completions.lock();
                    pc.remove(&fid).unwrap_or_default()
                };
                let stashed_events: Vec<_> = {
                    let mut pe = self.pending_events.lock();
                    pe.remove(&fid).unwrap_or_default()
                };
                if !completions.is_empty() {
                    // Pending completion(s) present: take the frame back, consume the
                    // completion events directly, reinsert Ready, re-queue.
                    let fb_opt = self.frames.lock().remove(&fid);
                    if let Some(mut frame) = fb_opt {
                    // Pending completion(s) present: consume the completion events directly.
                    if let Some(e) = event {
                        if let Some(bucket) = self.event_waiters.lock().get_mut(&e) {
                            bucket.retain(|wf| *wf != fid);
                        }
                    } else {
                        let mut ew = self.event_waiters.lock();
                        for bucket in ew.values_mut() {
                            bucket.retain(|wf| *wf != fid);
                        }
                    }
                    // Use frame.node_offset rather than subgraph.node_range.0 (same-function
                    // branch frame correction).
                    let caller_offset = NodeId(frame.node_offset);
                    // Walk all completions, writing back each return value + propagating the
                    // signal + notifying downstream.
                    for (call_node, return_value, child_signal) in completions {
                        let call_graph_id = NodeId(call_node.0 + caller_offset.0);
                        let consumer_count =
                            self.graph.downstream_count(call_graph_id.0 as usize);
                        frame.set_value(call_node, return_value, consumer_count);
                        if !frame.branch_relays.is_empty() {
                            relay_branch_value(&mut frame, &self.graph, call_node);
                        }
                        // Gate branch subgraph control-signal propagation (consistent with the
                        // normal path in complete_and_wake_caller).
                        // Bug #78: propagate control_signal for all call nodes (not just Gate),
                        // because LoopBody completion may also arrive via pending_completions when
                        // the loop_frame is being processed by another worker. This is deliberately
                        // BROADER than Ir::should_propagate_control_signal: on this race path a
                        // dropped signal cannot be recovered later, so any non-None signal is
                        // forwarded and the receiver's loop/Gate protocol sorts it out.
                        // W4c exception: capture gates must never propagate — the Return is the
                        // inlined value (already written above), not a signal.
                        let capture_gate = self.graph.node(call_graph_id.0 as usize).kind
                            == crate::ir::Ir::NodeKind::Gate
                            && self
                                .graph
                                .gate_branches_at(call_graph_id.0 as usize)
                                .map(|gb| gb.capture)
                                .unwrap_or(false);
                        if !capture_gate && !matches!(child_signal, ControlSignal::None) {
                            frame.control_signal = child_signal;
                        }
                        notify_downstream(
                            &mut frame,
                            &self.graph,
                            call_node,
                            call_graph_id,
                            caller_offset,
                        );
                    }
                        frame.state = FrameState::Ready;
                        frame.suspend_state = SuspendState::NotSuspended;
                        frame.suspend_event = None;
                        // Put back the same Box (address unchanged).
                        self.frames.lock().insert(fid, frame);
                        queue.push(fid);
                    } else {
                        // Frame concurrently taken (direct wake / cached-body
                        // relaunch): its processing drives it — RE-STASH the
                        // completions so a later suspension still finds them.
                        // Consuming them here without delivery would lose the
                        // wake forever (the join entry / waiter are already gone).
                        let mut pc = self.pending_completions.lock();
                        pc.entry(fid).or_default().extend(completions);
                    }
                } else if !stashed_events.is_empty() {
                    // Events arrived while the frame was absent (multi-slot: the
                    // frame may hold several; a single-slot map used to overwrite
                    // the earlier one — when the survivor was stale the frame's
                    // real wait was lost forever). Apply the FIRST entry matching
                    // the frame's current wait — WaitingEvent on exactly that
                    // event, or select-style suspend_event=None (any readiness
                    // wins). The rest are stale by definition and dropped.
                    let fb_opt = self.frames.lock().remove(&fid);
                    if let Some(mut fb) = fb_opt {
                        let select_form = matches!(fb.suspend_state, SuspendState::WaitingEvent(_))
                            && fb.suspend_event.is_none();
                        let mut applied = false;
                        if matches!(fb.suspend_state, SuspendState::WaitingEvent(_)) {
                            for (evt, evt_val) in stashed_events {
                                if fb.suspend_event == Some(evt) || select_form {
                                    if self.apply_event_to_frame(&mut fb, evt_val) {
                                        applied = true;
                                    }
                                    break;
                                }
                            }
                        }
                        if applied {
                            self.frames.lock().insert(fid, fb);
                            queue.push(fid);
                        } else {
                            self.frames.lock().insert(fid, fb);
                        }
                    } else {
                        // Frame concurrently taken: RE-STASH the events for its
                        // next suspension (see the completions arm above).
                        let mut pe = self.pending_events.lock();
                        pe.entry(fid).or_default().extend(stashed_events);
                    }
                }
            }
            FrameState::Completed => {
                // Cycle-collection pressure valve: at frame completion (a
                // quiescent point) collect cyclic garbage when the registry
                // grows past the threshold. Soundness rests on (a) a
                // single-threaded engine (worker_count <= 1: no concurrent
                // alloc/drop while the mark phase walks raw pointers) and (b)
                // ROOT COMPLETENESS — the sweep replaces every unmarked
                // registered object IN PLACE with Range(0,0), so any live
                // Value location the roots miss is corrupted, not leaked.
                // Root set: every frame table (live + pooled + this completing
                // one) with its construct_cache, the graph's const_cache (the
                // only holder of materialized constants between node
                // executions — its omission swept ""-literals to Range(0,0)
                // mid-run and corrupted str accumulators), pending
                // completions/events, the result, async-join results, the
                // ENGINE arena's ref slots, and the thread_local GLOBAL_ARENA's
                // ref slots (Closure bound_args handles live there, not in
                // self.arena).
                // Leak-suspicion heuristic (乙①): fire only when the live
                // set is LARGE and GROWING. A cyclic leak grows monotonically
                // and is still caught; a stable large live set (e.g. a 1M
                // record array) no longer pays a full-root mark at every
                // frame completion — that transient (roots Vec + marked set)
                // dominated peak RSS on big-live-set programs.
                static VALVE_PREV: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let live_now = crate::value::Registry::registered_count();
                let live_prev = VALVE_PREV.swap(live_now, std::sync::atomic::Ordering::Relaxed);
                // The entry frame's completion IS program end — a full-root
                // mark there is pure transient peak (dead=0 on healthy
                // programs); teardown collection is the correct moment.
                let is_entry = frame.subgraph_id == frame.graph.entry_subgraph.unwrap_or(frame.subgraph_id);
                if !is_entry && live_now > (1 << 16) && live_now > live_prev + (1 << 15) {
                    let mut roots: Vec<crate::value::Value> = Vec::new();
                    // The completing frame itself was already TAKEN out of the
                    // frames map at the top of process_frame and is only held
                    // in the local frame_box — root its value table here too,
                    // or everything referenced solely by this frame (e.g. two
                    // 5万-entry maps in main) reads as garbage and gets its
                    // edges released → use-after-free at teardown.
                    roots.extend(frame.value_table.values.iter().cloned());
                    roots.extend(frame.construct_cache.iter().map(|(_, v)| v.clone()));
                    let frames = self.frames.lock();
                    for f in frames.values() {
                        roots.extend(f.value_table.values.iter().cloned());
                        roots.extend(f.construct_cache.iter().map(|(_, v)| v.clone()));
                    }
                    drop(frames);
                    // The graph's materialized constants (shared Arcs, often
                    // the ONLY holder between node executions).
                    roots.extend(self.graph.const_cache.iter().cloned());
                    // Global variables live in graph.global_var_storage (the
                    // place model's GlobalSlotRef home — NOT "the root frame's
                    // slots" as older comments claimed). frondc's loader
                    // caches are globals; without this root the first valve
                    // fire swept the entire module-path/AST-string universe.
                    for slot in self.graph.global_var_storage.iter() {
                        let g = slot.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(v) = g.as_ref() {
                            roots.push(v.clone());
                        }
                    }
                    // Engine-side Value holders outside the frame map:
                    // pooled frames, pending completions/events, the final
                    // result, resolved-but-unjoined async results, and the
                    // arena's handle-backed slots (opaque to the edge walk,
                    // so rooted conservatively here).
                    for f in self.frame_pool.lock().iter() {
                        roots.extend(f.value_table.values.iter().cloned());
                        roots.extend(f.construct_cache.iter().map(|(_, v)| v.clone()));
                    }
                    for v in self.pending_completions.lock().values() {
                        for (_, val, _) in v.iter() {
                            roots.push(val.clone());
                        }
                    }
                    for v in self.pending_events.lock().values() {
                        for (_, val) in v {
                            roots.push(val.clone());
                        }
                    }
                    if let Some(v) = self.result.lock().as_ref() {
                        roots.push(v.clone());
                    }
                    self.async_join_runtime.lock().collect_results(&mut roots);
                    {
                        let arena = self.arena.lock();
                        let mut arcs: Vec<std::sync::Arc<crate::value::HeapObj>> = Vec::new();
                        arena.collect_ref_arcs(&mut arcs);
                        drop(arena);
                        for a in arcs {
                            roots.push(crate::value::Value::from_ref(a));
                        }
                    }
                    // The thread_local GLOBAL_ARENA is a DIFFERENT arena from
                    // self.arena: Closure bound_args (and formerly Newtype
                    // inner) handles are allocated there and were never rooted
                    // by the self.arena pass above.
                    crate::value::ValueArena::with_global(|g| {
                        let mut arcs: Vec<std::sync::Arc<crate::value::HeapObj>> = Vec::new();
                        g.collect_ref_arcs(&mut arcs);
                        for a in arcs {
                            roots.push(crate::value::Value::from_ref(a));
                        }
                    });
                    crate::value::Registry::collect_cycles(&roots);
                }

                // Bug #77: check if this is a defer frame completing. Defer frames are
                // registered in `defer_frames` by init_defer_frame; their completion must
                // decrement the parent's defer-waiter count and, when all defer frames are
                // done, finalize the parent frame (without re-running run_frame_nodes, which
                // would re-execute defer).
                let is_defer_frame = self.defer_frames.lock().remove(&fid);
                if is_defer_frame {
                    let caller = frame.caller;
                    self.release_frame(frame_box);
                    if let Some((parent_fid, _)) = caller {
                        let resume_parent = {
                            let mut waiters = self.defer_waiters.lock();
                            if let Some(count) = waiters.get_mut(&parent_fid) {
                                *count -= 1;
                                if *count == 0 {
                                    waiters.remove(&parent_fid);
                                    true
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        };
                        if resume_parent {
                            // All defer frames finished: finalize the parent frame directly.
                            let mut parent_box = self.frames.lock().remove(&parent_fid);
                            if let Some(pf) = parent_box.as_deref_mut() {
                                pf.state = FrameState::Completed;
                                pf.suspend_state = SuspendState::NotSuspended;
                                pf.suspend_event = None;
                            }
                            if let Some(pf) = parent_box {
                                let parent_has_caller = pf.caller.is_some();
                                // The parent may itself have been spawned as an ASYNC call:
                                // its await-er waits on AsyncJoin(parent_async_id), which
                                // complete_and_wake_caller never fires (it only completes
                                // SubgraphComplete waiters). Route through the async_join
                                // completion exactly like the normal Completed branch, or
                                // the await after the call is lost forever (silent drop).
                                let async_id =
                                    self.async_join_runtime.lock().find_by_child(parent_fid);
                                if let Some(async_id) = async_id {
                                    let return_value =
                                        extract_child_return(&pf, &self.graph);
                                    self.async_join_runtime
                                        .lock()
                                        .set_result(async_id, return_value.clone());

                                    let woken = self.on_event_arrived(
                                        RuntimeEvent::AsyncJoin(async_id),
                                        return_value,
                                        queue,
                                    );
                                    if woken > 0 {
                                        self.async_join_runtime.lock().remove_entry(async_id);
                                    }
                                    self.release_frame(pf);
                                } else if parent_has_caller {
                                    self.complete_and_wake_caller(*pf, queue);
                                } else {
                                    // Top-level frame (e.g. main): set the result.
                                    let ret = extract_child_return(&pf, &self.graph);
                                    *self.result.lock() = Some(ret);
                                    self.release_frame(pf);
                                }
                            }
                        }
                    }
                    return;
                }
                if has_caller {
                    // Distinguish sync-call vs async-call child-frame completion.
                    let async_id = self.async_join_runtime.lock().find_by_child(fid);
                    if let Some(async_id) = async_id {
                        // async child frame completed: set the result + fire the AsyncJoin event.
                        let return_value =
                            extract_child_return(frame, &self.graph);
                        self.async_join_runtime
                            .lock()
                            .set_result(async_id, return_value.clone());
                        // frame_box is dropped (not put back).
                        let woken = self.on_event_arrived(
                            RuntimeEvent::AsyncJoin(async_id),
                            return_value,
                            queue,
                        );
                        // The waiter has been woken (value injected via the event), so the entry
                        // can be safely cleaned up. If woken == 0 (no waiter), the entry is kept for
                        // a consuming read by try_get_result.
                        if woken > 0 {
                            self.async_join_runtime.lock().remove_entry(async_id);
                        }
                        // Return the async child frame to the pool.
                        self.release_frame(frame_box);
                    } else {
                        // sync child frame completed: clean up the waiter + write back + wake the
                        // caller.
                        self.event_waiters
                            .lock()
                            .remove(&RuntimeEvent::SubgraphComplete(fid));
                        // Frame consumed: unbox and hand to complete_and_wake_caller.
                        self.complete_and_wake_caller(*frame_box, queue);
                    }
                } else {
                    // Top-level frame completed: return the result.
                    let ret = extract_child_return(frame, &self.graph);
                    *self.result.lock() = Some(ret);
                    self.release_frame(frame_box);
                }
            }
            FrameState::Failed => {
                // Bug #77: a failed defer frame must also decrement the parent's defer-waiter
                // count (treat failure as completion for accounting purposes).
                let is_defer_frame = self.defer_frames.lock().remove(&fid);
                if is_defer_frame {
                    let caller = frame.caller;
                    self.release_frame(frame_box);
                    if let Some((parent_fid, _)) = caller {
                        let resume_parent = {
                            let mut waiters = self.defer_waiters.lock();
                            if let Some(count) = waiters.get_mut(&parent_fid) {
                                *count -= 1;
                                if *count == 0 {
                                    waiters.remove(&parent_fid);
                                    true
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        };
                        if resume_parent {
                            let mut parent_box = self.frames.lock().remove(&parent_fid);
                            if let Some(pf) = parent_box.as_deref_mut() {
                                pf.state = FrameState::Failed;
                                pf.suspend_state = SuspendState::NotSuspended;
                                pf.suspend_event = None;
                            }
                            if let Some(pf) = parent_box {
                                let parent_has_caller = pf.caller.is_some();
                                // Same async-parent routing as the Completed branch:
                                // an await-er waits on AsyncJoin, not SubgraphComplete.
                                let async_id =
                                    self.async_join_runtime.lock().find_by_child(parent_fid);
                                if let Some(async_id) = async_id {
                                    let return_value = Value::NULL;
                                    self.async_join_runtime
                                        .lock()
                                        .set_result(async_id, return_value.clone());
                                    let woken = self.on_event_arrived(
                                        RuntimeEvent::AsyncJoin(async_id),
                                        return_value,
                                        queue,
                                    );
                                    if woken > 0 {
                                        self.async_join_runtime.lock().remove_entry(async_id);
                                    }
                                    self.release_frame(pf);
                                } else if parent_has_caller {
                                    self.complete_and_wake_caller(*pf, queue);
                                } else {
                                    *self.result.lock() = Some(Value::NULL);
                                    self.release_frame(pf);
                                }
                            }
                        }
                    }
                    return;
                }
                if has_caller {
                    // Failed child frame (after cancel): clean up the waiter + wake the caller.
                    self.event_waiters
                        .lock()
                        .remove(&RuntimeEvent::SubgraphComplete(fid));
                    self.complete_and_wake_caller(*frame_box, queue);
                } else {
                    // Top-level frame Failed: return NULL.
                    *self.result.lock() = Some(Value::NULL);
                    self.release_frame(frame_box);
                }
            }
            _ => {
                // Ready (control signal triggered but not suspended): put back + re-enqueue.
                self.frames.lock().insert(fid, frame_box);
                queue.push(fid);
            }
        }
    }
    /// E7 same-frame branch launch: executes a small same-function branch
    /// subgraph in the CALLER's frame — no child frame, no acquire/copy/prepare.
    ///
    /// Mechanics (mirrors what a fresh child frame would do, scoped to the
    /// branch's own nodes):
    /// 1. Clear the branch's static slot range (params re-injected below; own
    ///    nodes recompute from seeds; nested sub-ranges relaunch themselves).
    /// 2. Initialize `pending_inputs` for the branch's own (non-nested) nodes:
    ///    every in-table input not currently ready. Outer values that are not
    ///    ready yet simply hold the node in THIS frame's queue — same-frame
    ///    notification reaches them, unlike a child frame's cross-frame gap.
    /// 3. Inject the branch arguments into the sg's param slots (notifying
    ///    branch-internal consumers).
    /// 4. Seed 0-pending own non-param nodes (Consts and nodes whose inputs all
    ///    became ready above) into this frame's ready queue.
    /// 5. Register a relay: when the sg's return node produces a value, the
    ///    driver copies it to the gate's slot and notifies the gate's consumers
    ///    (see `maybe_relay_branch`). An already-ready return (param forward /
    ///    outer-value passthrough arm) relays immediately.
    ///
    /// Control signals from the arm (break/continue/return) set THIS frame's
    /// control_signal — the parent IS the frame that must react, matching the
    /// child-frame propagation semantics. Capture gates never reach here
    /// (their Return must become the gate's value instead).
    fn launch_same_frame_branch(
        &self,
        frame: &mut Frame,
        gate_local: NodeId,
        target_sg: SubGraphId,
        args: &[Value],
    ) {
        let graph = frame.graph.clone();
        let sg = &graph.subgraphs[target_sg.0 as usize];
        let (bs, be) = sg.node_range;
        let offset = frame.node_offset;
        let table_len = frame.value_table.len();
        let pending_len = frame.pending_inputs.len();
        let nested: &[(u32, u32)] = graph.sg_nested_ranges(target_sg.0 as usize);
        let is_nested = |gid: u32| nested.iter().any(|&(s, e)| gid >= s && gid < e);

        // 1. Clear the branch's slot range.
        let clear_start = bs.0.wrapping_sub(offset) as usize;
        let clear_end = ((be.0.wrapping_sub(offset) as usize)).min(table_len);
        for i in clear_start..clear_end {
            frame.value_table.reset_slot(i);
        }

        // 1.5. Outer-input snapshot (the child-frame scheme's launch-time copy):
        // branch nodes may read enclosing-frame values (e.g. the loop's
        // condition chain) whose slots in THIS frame are unready — the E6
        // delta runs at the iteration boundary BEFORE the loop frame
        // recomputes them, and nothing refreshes them mid-iteration. For
        // each own-node input OUTSIDE this frame's own sg range that is
        // unready here, pull the current value through the frame chain.
        // In-own-range inputs are left pending (this frame computes them).
        {
            let (own_start, own_end) =
                graph.subgraphs[frame.subgraph_id.0 as usize].node_range;
            for gid in bs.0..be.0 {
                if is_nested(gid) {
                    continue;
                }
                let node = graph.node(gid as usize);
                let inputs = graph.inputs(node.inputs_offset, node.input_count);
                for &inp in inputs {
                    if inp.0 >= own_start.0 && inp.0 < own_end.0 {
                        continue;
                    }
                    let il = inp.0.wrapping_sub(offset) as usize;
                    if il >= table_len || frame.value_table.is_ready(il) {
                        continue;
                    }
                    if let Some(v) = snapshot_outer_value(frame, inp) {
                        frame.value_table.values[il] = v;
                        frame.value_table.set_ready(il);
                        frame.value_table.refcounts[il] = 0;
                    }
                }
            }
        }

        // 2. Own-node pending initialization.
        for gid in bs.0..be.0 {
            if is_nested(gid) {
                continue;
            }
            let local = gid.wrapping_sub(offset) as usize;
            if local >= pending_len || local >= table_len {
                continue;
            }
            let node = graph.node(gid as usize);
            if node.kind == crate::ir::Ir::NodeKind::EventSource {
                // Excluded by eligibility; defensive.
                frame.pending_inputs[local] = PENDING_EXTERNAL;
                continue;
            }
            let inputs = graph.inputs(node.inputs_offset, node.input_count);
            let mut p: u16 = 0;
            for &inp in inputs {
                let il = inp.0.wrapping_sub(offset) as usize;
                if il < table_len && !frame.value_table.is_ready(il) {
                    p += 1;
                }
            }
            frame.pending_inputs[local] = p;
        }

        // 3. Seeds FIRST — strictly BEFORE parameter injection: a node whose
        // pending the injection's notify will drop to 0 gets pushed by that
        // notify exactly once; seeding after the injection would see it as
        // (pending==0, not-ready, still queued) and push it a SECOND time —
        // double execution, double downstream decrements, pending underflow.
        // With params still unready here, their consumers hold pending>0 and
        // are correctly left to the injection's notify.
        let param_count = sg.param_count as usize;
        for gid in bs.0..be.0 {
            if (gid - bs.0) < param_count as u32 {
                continue;
            }
            if is_nested(gid) {
                continue;
            }
            let local = gid.wrapping_sub(offset) as usize;
            if local >= pending_len || local >= table_len {
                continue;
            }
            if frame.pending_inputs[local] == 0 && !frame.value_table.is_ready(local) {
                frame.push_ready(NodeId(local as u32));
            }
        }

        // 4. Parameter injection (+ branch-internal consumer notification).
        for (i, arg) in args.iter().enumerate().take(param_count) {
            let pgid = bs.0 + i as u32;
            let pl = pgid.wrapping_sub(offset) as usize;
            if pl < table_len {
                let cc = graph.downstream_count(pgid as usize);
                frame.set_value(NodeId(pl as u32), arg.clone(), cc);
                notify_downstream(frame, &graph, NodeId(pl as u32), NodeId(pgid), NodeId(offset));
            }
        }

        // 5. Return relay.
        let ret_gid = sg.return_node;
        let ret_local = ret_gid.0.wrapping_sub(offset) as usize;
        if ret_local < table_len
            && (gate_local.0 as usize) < table_len
            && frame.value_table.is_ready(ret_local)
        {
            // Param forward or outer-value passthrough arm: relay now.
            let v = frame.get_value(NodeId(ret_local as u32));
            let gate_gid = NodeId(gate_local.0 + offset);
            let cc = graph.downstream_count(gate_gid.0 as usize);
            frame.set_value(gate_local, v, cc);
            notify_downstream(frame, &graph, gate_local, gate_gid, NodeId(offset));
            self.maybe_relay_branch(frame, gate_local, &graph);
        } else {
            frame.branch_relays.push((NodeId(ret_local as u32), gate_local));
        }
    }

    /// E7 relay: `local` just produced a value and is a registered branch
    /// return — copy the value to the gate's slot and notify the gate's
    /// consumers. One relay per firing; a re-registered relay for the same
    /// pair is idempotent.
    fn maybe_relay_branch(&self, frame: &mut Frame, local: NodeId, graph: &DataFlowGraph) {
        relay_branch_value(frame, graph, local);
    }
}

/// Offload executor outcome: plan finished, or hit an arm needing the engine.
pub(super) enum PlanFlowCore {
    Done,
    EngineNeeded,
}

pub(super) fn exec_plan_offload(
    frame: &mut Frame,
    plan: &[NodeId],
    graph: &DataFlowGraph,
) -> PlanFlowCore {
        let node_start = frame.node_offset;
        for &gid in plan {
            if !matches!(frame.control_signal, ControlSignal::None) {
                return PlanFlowCore::Done;
            }
            if frame.state == FrameState::Cancelling || frame.state == FrameState::Suspended {
                return PlanFlowCore::Done;
            }
            let local = NodeId(gid.0.wrapping_sub(node_start));
            if frame.value_table.is_ready(local.0 as usize) {
                // Params / injected slots / already-executed nodes.
                continue;
            }
            let node = graph.node(gid.0 as usize);
            let ctx = EvalContext { node_start, graph };
            let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, gid, &ctx);
            match result {
                NodeResult::Value(v) => {
                    let cc = graph.downstream_count(gid.0 as usize);
                    frame.set_value(local, v, cc);
                    if !frame.branch_relays.is_empty() {
                        relay_branch_value(frame, graph, local);
                    }
                }
                NodeResult::Batch(results) => {
                    for &(lid, ref v) in &results {
                        let g2 = lid.0 + node_start;
                        let cc = graph.downstream_count(g2 as usize);
                        frame.set_value(lid, v.clone(), cc);
                    }
                    if !frame.branch_relays.is_empty() {
                        for &(lid, _) in &results {
                            relay_branch_value(frame, graph, lid);
                        }
                    }
                }
                NodeResult::Return(v) => {
                    frame.control_signal = ControlSignal::Return(v);
                    return PlanFlowCore::Done;
                }
                NodeResult::Break => {
                    frame.control_signal = ControlSignal::Break;
                    return PlanFlowCore::Done;
                }
                NodeResult::Continue => {
                    frame.control_signal = ControlSignal::Continue;
                    return PlanFlowCore::Done;
                }
                NodeResult::Call(pending) => {
                    let _ = pending;
                    return PlanFlowCore::EngineNeeded;
                }
                _ => {
                    // Engine-needing result (Await/ChannelNotify/Cancel/SelectWait):
                    // bail — the dataflow engine re-drives it.
                    return PlanFlowCore::EngineNeeded;
                }
            }
        }
        PlanFlowCore::Done
    }
