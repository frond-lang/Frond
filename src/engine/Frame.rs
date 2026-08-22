//! Frame lifecycle management: allocation, initialization, per-iteration reset, and frame-chain
//! pointers.

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::{Value, HeapObj, ArrayValue};

/// Copies ready values from the `src` frame to the `dst` frame, skipping nodes in the
/// `[branch_start, branch_end)` range.
///
/// Used by same_function branch frames (defer body / loop body) to inherit values already
/// computed by the enclosing function: the branch frame's value_table is sized to the parent
/// function, the node offset equals the parent function's node_offset, and only ready values
/// outside the branch range are copied (nodes inside the branch range are recomputed by the
/// branch frame itself).
fn copy_outer_ready_values(dst: &mut Frame, src: &Frame, count: usize, branch_start: u32, branch_end: u32) {
    let offset = src.node_offset;
    for i in 0..count {
        let gid = (offset as usize + i) as u32;
        if gid >= branch_start && gid < branch_end {
            continue;
        }
        if src.value_table.is_ready(i) {
            dst.value_table.values[i] = src.value_table.values[i].clone();
            dst.value_table.set_ready(i);
            dst.value_table.refcounts[i] = 0;
        }
    }
}

/// Sets up pending_inputs + enqueues 0-input nodes for a same_function branch frame.
/// Extracted from `Engine::prepare_same_function_frame` so the sync execution path
/// (ir::Compute::run_defers_sync) can reuse the same logic without an Engine instance.
pub fn prepare_same_function_frame_sync(frame: &mut Frame, graph: &DataFlowGraph) {
    let parent_start = frame.node_offset;
    let parent_node_count = frame.value_table.len();
    let sg_id = frame.subgraph_id;
    let sg = &graph.subgraphs[sg_id.0 as usize];
    let branch_start = sg.node_range.0 .0;
    let branch_end = sg.node_range.1 .0;
    let branch_param_count = sg.param_count as usize;

    // E3 steady-state cache: the derivation is a pure function of (parent-ready bitmap, graph,
    // sg). If the copied-in ready bitmap matches the snapshot taken at derivation time, reuse
    // the cached pending_inputs + seed list (memcpy instead of a per-node nested-range scan).
    // Loop iterations with a stable outer-ready set hit this every round.
    if let Some(cache) = frame.same_fn_prep_cache.take() {
        let (snap_ready, snap_pending, snap_seed) = *cache;
        let hit = snap_ready.len() == frame.value_table.ready.len()
            && snap_pending.len() == parent_node_count
            && frame.pending_inputs.len() >= parent_node_count
            && snap_ready[..] == frame.value_table.ready[..];
        if hit {
            frame.pending_inputs[..parent_node_count].copy_from_slice(&snap_pending);
            for &local in &snap_seed {
                if !frame.value_table.is_ready(local.0 as usize) {
                    frame.push_ready(local);
                }
            }
            frame.same_fn_prep_cache = Some(Box::new((snap_ready, snap_pending, snap_seed)));
            return;
        }
        // Miss: fall through to re-derive; the tail stores a fresh snapshot.
    }

    let nested_ranges: &[(u32, u32)] = graph.sg_nested_ranges(sg_id.0 as usize);
    let is_nested = |gid: u32| nested_ranges.iter().any(|&(s, e)| gid >= s && gid < e);

    // 1. Set pending_inputs.
    for i in 0..parent_node_count {
        let gid = (parent_start as usize + i) as u32;
        let in_branch = gid >= branch_start && gid < branch_end;
        if !in_branch || is_nested(gid) {
            frame.pending_inputs[i] = PENDING_EXTERNAL;
            continue;
        }
        let node = graph.node(gid as usize);
        if node.kind == NodeKind::EventSource {
            frame.pending_inputs[i] = PENDING_EXTERNAL;
        } else {
            let inputs = graph.inputs(node.inputs_offset, node.input_count);
            let mut pending = 0u16;
            for &inp in inputs {
                let il = inp.0.wrapping_sub(parent_start) as usize;
                if il < parent_node_count {
                    let inp_gid = (parent_start as usize + il) as u32;
                    let inp_in_branch = inp_gid >= branch_start && inp_gid < branch_end;
                    if inp_in_branch && !frame.value_table.is_ready(il) {
                        pending += 1;
                    }
                }
            }
            frame.pending_inputs[i] = pending;
        }
    }

    // 2. Enqueue in-branch 0-input non-Param nodes (collect for the cache snapshot).
    let mut seed: Vec<NodeId> = Vec::new();
    for i in 0..parent_node_count {
        let gid = (parent_start as usize + i) as u32;
        let in_branch = gid >= branch_start && gid < branch_end;
        if !in_branch || is_nested(gid) {
            continue;
        }
        let local_in_branch = (gid - branch_start) as usize;
        if local_in_branch < branch_param_count {
            continue;
        }
        if frame.pending_inputs[i] == 0 && !frame.value_table.is_ready(i) {
            frame.push_ready(NodeId(i as u32));
            seed.push(NodeId(i as u32));
        }
    }

    frame.same_fn_prep_cache = Some(Box::new((
        frame.value_table.ready.clone(),
        frame.pending_inputs[..parent_node_count].to_vec(),
        seed,
    )));
}

/// Creates a defer body frame for synchronous execution.
/// Mirrors `Engine::init_defer_frame`: uses the parent frame's node_offset + value_table size,
/// copies the parent's ready values, sets up pending_inputs via `prepare_same_function_frame_sync`,
/// and wires frame-chain pointers so the defer body can read/write the parent function's locals.
pub fn prepare_defer_frame_sync(
    parent_frame: &Frame,
    body_subgraph: SubGraphId,
    graph: &DataFlowGraph,
) -> Frame {
    let parent_start = parent_frame.node_offset;
    let parent_node_count = parent_frame.value_table.len();
    let child_sg = &graph.subgraphs[body_subgraph.0 as usize];
    let (branch_start, branch_end) = child_sg.node_range;

    let mut frame = Frame::new(
        FrameId(u32::MAX),
        body_subgraph,
        parent_node_count,
        parent_frame.graph.clone(),
    );
    frame.node_offset = parent_start;

    // Copy the parent frame's ready values (skipping nodes inside the defer body range).
    copy_outer_ready_values(&mut frame, parent_frame, parent_node_count, branch_start.0, branch_end.0);

    // Set pending_inputs + prefill Const + enqueue 0-input nodes.
    prepare_same_function_frame_sync(&mut frame, graph);

    // Frame-chain pointers: the defer body accesses outer variables through the frame chain.
    let parent_ptr = parent_frame as *const Frame as *mut Frame;
    let parent_root = if !parent_frame.root_frame_ptr.is_null() {
        parent_frame.root_frame_ptr
    } else {
        parent_ptr
    };
    frame.parent_frame_ptr = parent_ptr;
    frame.root_frame_ptr = parent_root;

    frame
}

// =========================================================================
// impl<S: LockStrategy> Engine<S> — frame management methods
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// Allocates a frame id.
    pub(super) fn alloc_frame_id(&self) -> FrameId {
        let mut next = self.next_frame_id.lock();
        let id = *next;
        assert!(next.0 < u32::MAX, "FrameId overflow: too many frames allocated");
        next.0 += 1;
        id
    }

    /// Frame-pool capacity cap (prevents unbounded growth).
    const FRAME_POOL_MAX: usize = 32;

    /// Acquires a reusable frame from the pool, or creates a new one.
    /// On reuse the Vec capacity is retained (resize does not reallocate), eliminating frequent
    /// alloc/dealloc.
    pub(super) fn acquire_frame(
        &self,
        id: FrameId,
        subgraph_id: SubGraphId,
        node_count: usize,
    ) -> Box<Frame> {
        let mut pool = self.frame_pool.lock();
        if let Some(mut frame_box) = pool.pop() {
            let frame = &mut *frame_box;
            frame.id = id;
            frame.subgraph_id = subgraph_id;
            frame.graph = self.graph.clone();
            frame.value_table.resize(node_count);
            frame.value_table.reset_all();
            // E6: re-targeted pooled frame — dirty tracking belongs to the previous
            // loop-boundary pairing only.
            frame.value_table.disable_dirty_tracking();
            frame.pending_inputs.resize(node_count, 0);
            frame.ready_queue.clear();
            frame.state = FrameState::Ready;
            frame.caller = None;
            frame.node_offset = 0;
            frame.control_signal = ControlSignal::None;
            frame.suspend_state = SuspendState::NotSuspended;
            frame.defer_stack.clear();
            frame.suspend_event = None;
            frame.select_timers.clear();
            frame.root_frame_ptr = std::ptr::null_mut();
            frame.parent_frame_ptr = std::ptr::null_mut();
            frame.cached_child_frame = None;
            // E2: drop any stashed loop-body frame (a completed frame must not carry one into
            // the pool; plain drop — the pool mutex is held here, so no nested release_frame).
            frame.hot_body = None;
            // E3: per-subgraph derivation cache is invalid once the frame is re-targeted.
            frame.same_fn_prep_cache = None;
            frame.closure_val = None;
            frame.branch_relays.clear();
            frame.construct_cache.clear();
            frame_box
        } else {
            drop(pool);
            Box::new(Frame::new(id, subgraph_id, node_count, self.graph.clone()))
        }
    }

    /// Returns a frame to the pool for reuse (dropped if the pool is full).
    pub(super) fn release_frame(&self, frame_box: Box<Frame>) {
        let mut pool = self.frame_pool.lock();
        if pool.len() < Self::FRAME_POOL_MAX {
            pool.push(frame_box);
        }
        // else: pool full, frame_box drops
    }

    /// Initializes the entry frame (main function) with default argument injection.
    ///
    /// The base frame allocation leaves Param nodes unset (expecting the caller to inject
    /// them via `start_subgraph`). But main has no caller — its Param slots are never
    /// filled, so accessing them (e.g. `args.len()`) reads uninitialised memory and
    /// crashes silently.
    ///
    /// This method injects a default value (empty array) into each Param slot so that
    /// `main(args: str[])` receives a valid empty array when no CLI args are provided.
    pub(super) fn init_entry_frame(&self, subgraph_id: SubGraphId) -> FrameId {
        let (node_start, node_end) = self.graph.subgraphs[subgraph_id.0 as usize].node_range;
        let node_count = (node_end.0 - node_start.0) as usize;
        let fid = self.alloc_frame_id();
        let mut frame = self.acquire_frame(fid, subgraph_id, node_count);
        self.prepare_frame(&mut frame);

        // Inject default entry arguments — main has no caller to inject them.
        let param_count = self.graph.subgraphs[subgraph_id.0 as usize].param_count as usize;
        let offset = node_start.0 as usize;
        for i in 0..param_count {
            let local_id = NodeId(i as u32);
            let global_id = NodeId((offset + i) as u32);
            let consumer_count = self.graph.downstream_count(offset + i);
            // Default: empty array (main's `args: str[]` receives an empty array).
            let default_arg = Value::ref_val(HeapObj::Array(ArrayValue::new(Vec::new())));
            frame.set_value(local_id, default_arg, consumer_count);
            notify_downstream(&mut frame, &self.graph, local_id, global_id, node_start);
        }

        self.frames.lock().insert(fid, frame);
        fid
    }

    /// Initializes a defer body frame: same_function branch frame setup (Bug #52).
    ///
    /// The defer body subgraph is compiled as a same_function branch subgraph (function_id = parent
    /// function), but the base frame allocation creates the frame using the defer body's own
    /// node_range, so the node_offset and value_table size do not match the parent function,
    /// causing WriteBack to compute an out-of-bounds local index
    /// (`writeback target out of current frame range`).
    ///
    /// This method instead creates the frame using the parent frame's node_offset and value_table
    /// size, copies the parent frame's ready values, then uses `prepare_same_function_frame` to set
    /// up pending_inputs, so the defer body can correctly read and write the parent function's
    /// local variables.
    pub(super) fn init_defer_frame(
        &self,
        body_subgraph: SubGraphId,
        parent_frame: &Frame,
    ) -> FrameId {
        let parent_start = parent_frame.node_offset;
        let parent_node_count = parent_frame.value_table.len();
        let child_sg = &self.graph.subgraphs[body_subgraph.0 as usize];
        let (branch_start, branch_end) = child_sg.node_range;

        let fid = self.alloc_frame_id();
        let mut frame = self.acquire_frame(fid, body_subgraph, parent_node_count);
        frame.node_offset = parent_start;

        // Copy the parent frame's ready values (skipping nodes inside the defer body range).
        copy_outer_ready_values(&mut frame, parent_frame, parent_node_count, branch_start.0, branch_end.0);

        // Set pending_inputs + prefill Const + enqueue 0-input nodes.
        self.prepare_same_function_frame(&mut frame);

        // Frame-chain pointers: the defer body accesses outer variables through the frame chain (Bug #47).
        let parent_ptr = parent_frame as *const Frame as *mut Frame;
        let parent_root = if !parent_frame.root_frame_ptr.is_null() {
            parent_frame.root_frame_ptr
        } else {
            parent_ptr
        };
        frame.parent_frame_ptr = parent_ptr;
        frame.root_frame_ptr = parent_root;

        // Bug #77: set caller to the parent frame so that when this defer frame completes (in
        // process_frame's Completed/Failed branch) the parent frame can be woken. The call_node
        // value is unused for defer frames (defer frames are identified via `defer_frames`), so
        // NodeId(0) is a safe placeholder.
        frame.caller = Some((parent_frame.id, crate::ir::Ir::NodeId(0)));

        self.frames.lock().insert(fid, frame);
        // Bug #77: register this frame as a defer frame so process_frame can distinguish it from
        // ordinary child frames and route its completion to defer_waiter wakeup.
        self.defer_frames.lock().insert(fid);
        fid
    }

    /// Frame node initialization: reset + prefill.
    pub(super) fn prepare_frame(&self, frame: &mut Frame) {
        // Reset frame state (mandatory on reuse to avoid stale values).
        frame.value_table.reset_all();
        frame.ready_queue.clear();
        frame.control_signal = ControlSignal::None;
        // E5: fresh cross-function frame — eligible for one linear run.
        frame.linear_fresh = true;
        // Below: use prepare_frame_nodes to set node_offset + pending_inputs + Const prefill.
        prepare_frame_nodes(frame, &self.graph);
    }

    /// Per-iteration reset: after body_sg completes, reset the loop frame and reuse the body_sg
    /// frame. Ported from the Engine version, switched to `&self` + `&mut Frame` parameters.
    pub(super) fn reset_loop_iteration(
        &self,
        loop_frame: &mut Frame,
        loop_fid: FrameId,
        body_frame: &mut Frame,
    ) {
        let loop_sg_id = loop_frame.subgraph_id;
        // Borrow the plan from the graph instead of cloning: reset_plan stays immutable for the
        // whole run, and both this borrow and reset_condition_tree(&self) are shared borrows.
        let sg = &self.graph.subgraphs[loop_sg_id.0 as usize];
        let (loop_kind, cond_node, return_node, iter_next_node) =
            (sg.loop_kind, sg.cond_node, sg.return_node, sg.iter_next_node);
        let reset_plan = sg.reset_plan.as_ref();
        // Use loop_frame.node_offset rather than subgraph.node_range.0 (same-function branch frame
        // correction).
        let loop_offset = loop_frame.node_offset;

        // 0. Place-model phi carries: loop-carried cell values ride the sg's
        // param slots. The body's final values are READ now (before its frame
        // is reset_all'd) and STASHED; the slot writes + consumer pokes happen
        // at the END of this reset — the gate is only reset to pending=1 in
        // step 4 below, so poking any earlier lands on an already-executed
        // gate and the re-check never fires (loop stalls after one round).
        // The `carries_cell` variant dereferences through the Cell (for
        // conditionally stored vars with no statically known final node).
        let mut stashed_carries: Vec<(NodeId, Value)> = Vec::new();
        if let Some(plan) = reset_plan {
            for &(param_gid, src_gid) in &plan.carries_value {
                let v = body_frame.get_value_by_global(src_gid);
                stashed_carries.push((param_gid, v));
            }
            for &(param_gid, cell_gid) in &plan.carries_cell {
                let cv = body_frame.get_value_by_global(cell_gid);
                let v = match cv.heap_ref() {
                    Some(arc) => match arc.as_ref() {
                        crate::value::HeapObj::Cell(c) => c.get(),
                        _ => cv,
                    },
                    None => cv,
                };
                stashed_carries.push((param_gid, v));
            }
        }

        // 0b. Clear ready_queue (must precede steps 1-3 pushing cond/iter_next/gate).
        loop_frame.ready_queue.clear();

        // 1-3. Data-driven reset via ResetPlan when present; otherwise fall back to the LoopKind
        // branch.
        if let Some(plan) = reset_plan {
            for &nid in &plan.reset_to_zero {
                let local = NodeId(nid.0.wrapping_sub(loop_offset));
                Self::reset_node_ready(loop_frame, local);
                loop_frame.push_ready(local);
            }
            for &nid in &plan.reset_to_one {
                let local = NodeId(nid.0.wrapping_sub(loop_offset));
                Self::reset_node_pending(loop_frame, local, 1);
            }
            // W5: apply the precomputed condition-tree plan mechanically —
            // same (node, pending) pairs the DFS below would produce, flattened
            // once at build/load time. Falls back to the DFS when the plan was
            // not precomputed.
            if !plan.condition_tree_plan.is_empty() {
                for &(gid, pending) in &plan.condition_tree_plan {
                    let local = NodeId(gid.0.wrapping_sub(loop_offset));
                    Self::reset_node_pending(loop_frame, local, pending);
                }
                for &(gid, pending) in &plan.condition_tree_plan {
                    if pending == 0 {
                        loop_frame.push_ready(NodeId(gid.0.wrapping_sub(loop_offset)));
                    }
                }
            } else {
                for &nid in &plan.reset_condition_tree {
                    self.reset_condition_tree(loop_frame, loop_sg_id, nid, loop_offset);
                }
            }
        } else {
            // Fallback: LoopKind branch (subgraphs without a ResetPlan, e.g. TailRec).
            if loop_kind == crate::ir::Ir::LoopKind::For {
                if let Some(next_node) = iter_next_node {
                    let next_local = NodeId(next_node.0.wrapping_sub(loop_offset));
                    Self::reset_node_ready(loop_frame, next_local);
                    loop_frame.push_ready(next_local);
                }
            }
            if let Some(cond_node) = cond_node {
                let cond_local = NodeId(cond_node.0.wrapping_sub(loop_offset));
                if loop_kind == crate::ir::Ir::LoopKind::For {
                    Self::reset_node_pending(loop_frame, cond_local, 1);
                } else {
                    self.reset_condition_tree(loop_frame, loop_sg_id, cond_node, loop_offset);
                }
            }
        }

        // 4. Reset the Gate node (pending=1, waiting for cond notify).
        let gate_local = NodeId(return_node.0.wrapping_sub(loop_offset));
        Self::reset_node_pending(loop_frame, gate_local, 1);

        // 4. Reset the body_sg frame (reuse).
        // The body frame is a same_function branch frame: node_offset = parent function
        // node_start, and value_table is sized to the parent function. prepare_frame_nodes must
        // not be called (it would reset node_offset to the body subgraph's node_range.0, misaligning
        // value-table indices and causing WriteBack's target - node_offset to compute a wrong local,
        // skipping the body frame).
        //
        // E6 incremental path: the body frame's value table is sized to the whole parent
        // function, but the slots it actually owns are its subgraph's node range. Clearing
        // that static range is O(body) instead of reset_all's O(function); out-of-branch
        // slots are refreshed from the loop frame's dirty set below (only the loop frame
        // needs dirty tracking — every write the body makes outside its range is mirrored
        // into the loop frame by WriteBack's chain walk, so the delta source sees it).
        // Untouched out-of-branch slots still mirror the loop frame from the previous
        // boundary, so skipping their re-copy is value-identical.
        let body_sg = &self.graph.subgraphs[body_frame.subgraph_id.0 as usize];
        let (body_branch_start, body_branch_end) = body_sg.node_range;
        // First boundary runs the legacy full reset + full copy, which leaves the two frames
        // value-synchronized; tracking is enabled there, so from the second boundary on the
        // delta path has a complete dirty history.
        let delta_eligible = reset_plan.is_some()
            && !super::env_flag("FROND_NO_DELTA_RESET")
            && !super::env_flag("FROND_NO_REUSECHAIN");
        let delta_reset = delta_eligible && loop_frame.value_table.dirty_tracking_enabled();
        let body_offset = body_frame.node_offset;
        let body_len = body_frame.value_table.len();
        if delta_reset {
            let clear_start = body_branch_start.0.wrapping_sub(body_offset) as usize;
            let clear_end = body_branch_end.0.wrapping_sub(body_offset) as usize;
            let clear_end = clear_end.min(body_len);
            for i in clear_start..clear_end {
                body_frame.value_table.reset_slot(i);
            }
        } else {
            body_frame.value_table.reset_all();
        }
        body_frame.ready_queue.clear();
        body_frame.select_timers.clear();
        body_frame.branch_relays.clear();
        body_frame.cached_child_frame = None;
        body_frame.hot_body = None;
        body_frame.control_signal = ControlSignal::None;

        // Apply the stashed phi carries BEFORE copy_outer_ready_values: the
        // copy materializes the loop frame's ready slots (params included)
        // into the body frame — carrying after it would leave the body with
        // the previous iteration's param values. The gate was reset to
        // pending=1 in step 4 (before the body reset), so the pokes land on a
        // waiting gate and re-fire the condition chain.
        if !stashed_carries.is_empty() {
            // E2 fused path: when the precomputed plan is present, the
            // application is direct slot arithmetic — no chain walk, no
            // downstream-slice lookup, no consumer-count recompute. Falls
            // back to the generic path when the plan is missing (fresh
            // build before precompute, exotic frames).
            let fused = reset_plan.and_then(|p| {
                (!p.fused_carries.is_empty()).then(|| &p.fused_carries)
            });
            match fused {
                Some(fused) => {
                    // The stashed values were read BEFORE the body frame's
                    // reset_all (which wipes it) — match them to the fused
                    // plan by param id, never re-read the body frame here.
                    for fc in fused {
                        let param_gid = NodeId(fc.param_local.wrapping_add(loop_offset));
                        let v = stashed_carries
                            .iter()
                            .find(|(g, _)| *g == param_gid)
                            .map(|(_, v)| v.clone())
                            .unwrap_or(Value::NULL);
                        let pl = fc.param_local as usize;
                        if pl < loop_frame.value_table.len() {
                            loop_frame.value_table.values[pl] = v;
                            loop_frame.value_table.set_ready(pl);
                            loop_frame.value_table.record_dirty_slot(pl);
                            for &ds in &fc.consumers {
                                let d = ds as usize;
                                if d < loop_frame.pending_inputs.len() {
                                    let p = loop_frame.pending_inputs[d];
                                    if p > 0 && p != PENDING_EXTERNAL {
                                        let np = p - 1;
                                        loop_frame.pending_inputs[d] = np;
                                        if np == 0
                                            && !loop_frame.value_table.is_ready(d)
                                        {
                                            loop_frame.push_ready(NodeId(ds));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                None => {
                    let graph = &self.graph;
                    for &(param_gid, ref v) in &stashed_carries {
                        let param_local = NodeId(param_gid.0.wrapping_sub(loop_offset));
                        let consumers = graph.downstream_count(param_gid.0 as usize);
                        loop_frame.set_value(param_local, v.clone(), consumers);
                        notify_downstream(loop_frame, graph, param_local, param_gid, NodeId(loop_offset));
                    }
                }
            }
        }

        // Re-copy outer-variable values from loop_frame first (consistent with start_subgraph).
        // This copy must happen before prepare_same_function_frame: the 0-input enqueue logic in
        // prepare_same_function_frame relies on the value-table ready state to derive pending_inputs;
        // if outer variables are not ready, nodes depending on outer variables would be incorrectly
        // marked pending==0 and enqueued, then read empty values at execution (repro: a for-in loop
        // body executing only its first iteration).
        if delta_reset {
            // E6 delta copy: refresh only the loop-frame slots written since the last
            // boundary. In-branch slots are the body's own (recomputed each iteration —
            // skipped, matching the full copy's branch-range skip). An unready loop slot
            // (e.g. a condition-tree node reset by the plan) clears the body's mirror.
            let loop_dirty_len = loop_frame.value_table.dirty_len();
            for k in 0..loop_dirty_len {
                let idx = loop_frame.value_table.dirty_slot(k);
                let gid = loop_offset.wrapping_add(idx);
                if gid >= body_branch_start.0 && gid < body_branch_end.0 {
                    continue;
                }
                let i = idx as usize;
                if i >= body_len {
                    continue;
                }
                if loop_frame.value_table.is_ready(i) {
                    body_frame.value_table.copy_slot_from(i, &loop_frame.value_table);
                } else {
                    body_frame.value_table.reset_slot(i);
                }
            }
            loop_frame.value_table.end_dirty_generation();
        } else {
            let copy_count = loop_frame.value_table.len().min(body_frame.value_table.len());
            copy_outer_ready_values(body_frame, loop_frame, copy_count, body_branch_start.0, body_branch_end.0);
            if delta_eligible {
                // The full reset + full copy above synchronized the frames; start tracking
                // so the NEXT boundary can take the delta path. Only the loop frame needs
                // tracking — it is the delta source.
                loop_frame.value_table.enable_dirty_tracking();
            }
        }

        // After copying outer variables, set pending_inputs + enqueue 0-input nodes.
        // E5: when the body has a linearized plan, skip the readiness derivation entirely —
        // the linear runner needs neither pending_inputs nor seeds, and a mid-body launch node
        // rebuilds them on demand (rebuild_linear_bailout).
        if self.graph.linear_plan(body_frame.subgraph_id.0 as usize).is_some() {
            body_frame.linear_fresh = true;
        } else {
            self.prepare_same_function_frame(body_frame);
        }

        // Rebind the body_sg frame's caller.
        body_frame.caller =
            Some((loop_fid, NodeId(return_node.0.wrapping_sub(loop_offset))));
        // Bug #100: keep the frame chain connected across iterations. The loop frame is
        // in hand here and Box addresses are stable (process_frame reuses the Box), so
        // pointing the body at it is safe; nulling the pointers orphaned the body's
        // WriteBacks (loop-variable updates never reached the loop frame).
        let loop_ptr = loop_frame as *const Frame as *mut Frame;
        body_frame.parent_frame_ptr = loop_ptr;
        body_frame.root_frame_ptr = if !loop_frame.root_frame_ptr.is_null() {
            loop_frame.root_frame_ptr
        } else {
            loop_ptr
        };


        // 5. Reset the loop frame state.
        loop_frame.branch_relays.clear();
        loop_frame.control_signal = ControlSignal::None;
        loop_frame.state = FrameState::Ready;
        loop_frame.suspend_state = SuspendState::NotSuspended;
        loop_frame.suspend_event = None;
    }

    /// same_function branch frame reset: keeps node_offset (= parent function node_start) and sets
    /// up pending_inputs + prefills Const + enqueues 0-input nodes.
    ///
    /// Difference from prepare_frame_nodes: prepare_frame_nodes resets node_offset to the subgraph's
    /// own node_range.0, which suits cross-function call frames. A same_function branch frame's
    /// node_offset is the parent function's node_start (value table sized to the parent function)
    /// and must be left unchanged.
    fn prepare_same_function_frame(&self, frame: &mut Frame) {
        prepare_same_function_frame_sync(frame, &self.graph);
    }

    /// Recursively resets the condition dependency tree (While/Loop per-iteration reset).
    ///
    /// When `reset_loop_iteration` resets only the top-level cond_node, the cond_node's input nodes
    /// (e.g. the comparison operands `lt1`/`lt2` of `&&`/`||`) retain stale values from the previous
    /// round, so cond_node reads a stale comparison result (condition stuck true -> infinite loop).
    ///
    /// This method recursively collects every node in the cond_node dependency tree that lies inside
    /// the loop subgraph (excluding nested subgraphs body_sg/void_sg and Gate nodes), resets its
    /// value, and sets pending_inputs according to dependencies, ensuring each iteration is
    /// re-evaluated from scratch.
    fn reset_condition_tree(
        &self,
        loop_frame: &mut Frame,
        loop_sg_id: SubGraphId,
        cond_node: NodeId,
        loop_offset: u32,
    ) {
        let sg = &self.graph.subgraphs[loop_sg_id.0 as usize];
        let (sg_start, sg_end) = sg.node_range;

        // Collect nested subgraph ranges (body_sg, void_sg).
        let nested_ranges: Vec<(u32, u32)> = self
            .graph
            .subgraphs
            .iter()
            .filter(|s| {
                s.id != loop_sg_id
                    && s.node_range.0 .0 >= sg_start.0
                    && s.node_range.1 .0 <= sg_end.0
            })
            .map(|s| (s.node_range.0 .0, s.node_range.1 .0))
            .collect();
        let is_nested = |gid: u32| nested_ranges.iter().any(|&(s, e)| gid >= s && gid < e);
        // The sg's PARAM-prefix nodes are excluded: their slots are injected
        // (call args / phi carries), never re-computed — re-firing their
        // CF_NOOP would overwrite the injected value with VOID.
        let param_end = sg_start.0 + sg.param_count as u32;
        let is_in_sg =
            |gid: u32| gid >= param_end && gid < sg_end.0 && !is_nested(gid);

        // DFS-collect all nodes in the cond_node dependency tree that lie inside the loop subgraph.
        //
        // Gate nodes (from short-circuit && / || lowering) MUST be included so that they are
        // reset and re-evaluated on each loop iteration. Skipping them was Bug #38: the Gate's
        // condition_input (lhs) was never reset, so the while-loop's Gate read a stale value
        // and the loop exited after one iteration.
        let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut stack = vec![cond_node];
        let mut cond_nodes: Vec<NodeId> = Vec::new();
        while let Some(gid) = stack.pop() {
            if !visited.insert(gid.0) {
                continue;
            }
            if !is_in_sg(gid.0) {
                continue;
            }
            cond_nodes.push(gid);
            let node = self.graph.node(gid.0 as usize);
            let inputs = self
                .graph
                .inputs(node.inputs_offset, node.input_count);
            for &inp in inputs {
                stack.push(inp);
            }
        }

        // Phase 1: reset each node's value + set pending_inputs.
        for &gid in &cond_nodes {
            let local = NodeId(gid.0.wrapping_sub(loop_offset));
            let node = self.graph.node(gid.0 as usize);
            let inputs = self
                .graph
                .inputs(node.inputs_offset, node.input_count);

            // pending = number of inputs within the dependency tree (re-evaluated
            // this iteration) PLUS param-prefix inputs (their completion signal
            // is the phi carry's poke — counting them prevents an early fire
            // that would read freshly-cleared sibling slots as 0). Other
            // external inputs (variables outside the loop) are accessed via
            // the frame chain and already ready, so they are not counted.
            let in_param_prefix =
                |gid: u32| gid >= sg_start.0 && gid < sg_start.0 + sg.param_count as u32;
            let pending: u16 = inputs
                .iter()
                .filter(|&&inp| {
                    in_param_prefix(inp.0) || (visited.contains(&inp.0) && is_in_sg(inp.0))
                })
                .count() as u16;

            Self::reset_node_pending(loop_frame, local, pending);
        }

        // Phase 2: enqueue 0-pending nodes (Const nodes also take this path — compute_fn returns a
        // value).
        for &gid in &cond_nodes {
            let local = NodeId(gid.0.wrapping_sub(loop_offset));
            let i = local.0 as usize;
            let pending = if i < loop_frame.pending_inputs.len() {
                loop_frame.pending_inputs[i]
            } else {
                0
            };
            if pending == 0 {
                loop_frame.push_ready(local);
            }
        }
    }

    /// Resets a node to the ready state (pending=0, value cleared, not enqueued). Associated
    /// function.
    pub(super) fn reset_node_ready(frame: &mut Frame, node_local: NodeId) {
        let i = node_local.0 as usize;
        if i < frame.pending_inputs.len() {
            frame.pending_inputs[i] = 0;
        }
        if i < frame.value_table.len() {
            frame.value_table.reset_slot(i);
        }
    }

    /// Resets a node to the pending state (pending=N, value cleared). Associated function.
    pub(super) fn reset_node_pending(frame: &mut Frame, node_local: NodeId, pending: u16) {
        let i = node_local.0 as usize;
        if i < frame.pending_inputs.len() {
            frame.pending_inputs[i] = pending;
        }
        if i < frame.value_table.len() {
            frame.value_table.reset_slot(i);
        }
    }

    /// Sets the frame-chain pointers: walks the caller chain in the HashMap to set
    /// parent_frame_ptr/root_frame_ptr.
    ///
    /// Must be called after the frame is removed from the HashMap but before execution (at that
    /// point all parent frames are still in the HashMap, so the `Box<Frame>` heap addresses are
    /// stable and are not moved even if the HashMap rehashes).
    ///
    /// Same-function subgraphs (if-else/match arm/loop body): parent_frame_ptr points to the
    /// immediate caller frame; root_frame_ptr points to the function's root frame (the farthest
    /// same-function frame found by walking the caller chain upward).
    /// Cross-function calls: both pointers are null (cross-function access to outer variables is
    /// disallowed).
    ///
    /// If the caller frame is not in the HashMap (being executed by another worker), the existing
    /// pointers are left unchanged (start_subgraph already set the initial pointers at creation).
    pub(super) fn setup_frame_chain(&self, frame: &mut Frame) {
        let Some((caller_fid, _)) = frame.caller else {
            return; // Top-level frame, no parent.
        };

        let frames = self.frames.lock();
        let Some(caller_box) = frames.get(&caller_fid) else {
            return; // caller not in the HashMap (currently executing); keep existing pointers.
        };

        let frame_fn_id = self.graph.subgraphs[frame.subgraph_id.0 as usize].function_id;
        let caller_fn_id =
            self.graph.subgraphs[caller_box.subgraph_id.0 as usize].function_id;

        // Cross-function call: do not set frame-chain pointers.
        if caller_fn_id != frame_fn_id {
            return;
        }

        let caller_ptr = caller_box.as_ref() as *const Frame as *mut Frame;
        frame.parent_frame_ptr = caller_ptr;

        // root_frame_ptr: walk the caller chain upward to find the farthest same-function frame.
        let mut root_ptr = caller_ptr;
        let mut current_box = caller_box;
        loop {
            match current_box.caller {
                Some((grandparent_fid, _)) => {
                    match frames.get(&grandparent_fid) {
                        Some(gp_box) => {
                            let gp_fn_id =
                                self.graph.subgraphs[gp_box.subgraph_id.0 as usize].function_id;
                            if gp_fn_id != frame_fn_id {
                                break; // Cross-function boundary.
                            }
                            root_ptr = gp_box.as_ref() as *const Frame as *mut Frame;
                            current_box = gp_box;
                        }
                        None => break, // Grandparent frame not in the HashMap.
                    }
                }
                None => break, // Reached the top of the same-function chain.
            }
        }
        frame.root_frame_ptr = root_ptr;
    }
}
