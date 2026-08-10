//! Frame lifecycle management: allocation, initialization, per-iteration reset, and frame-chain
//! pointers.

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;

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
            frame.closure_val = None;
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

    /// Initializes a frame: allocation + prefill. Returns the FrameId (the frame is already
    /// inserted into `frames`).
    pub(super) fn init_frame(&self, subgraph_id: SubGraphId) -> FrameId {
        let (node_start, node_end) = self.graph.subgraphs[subgraph_id.0 as usize].node_range;
        let node_count = (node_end.0 - node_start.0) as usize;
        let fid = self.alloc_frame_id();
        let mut frame = self.acquire_frame(fid, subgraph_id, node_count);
        self.prepare_frame(&mut frame);
        self.frames.lock().insert(fid, frame);
        fid
    }

    /// Initializes a defer body frame: same_function branch frame setup (Bug #52).
    ///
    /// The defer body subgraph is compiled as a same_function branch subgraph (function_id = parent
    /// function), but `init_frame` creates the frame using the defer body's own node_range, so the
    /// node_offset and value_table size do not match the parent function, causing WriteBack to
    /// compute an out-of-bounds local index (`writeback target out of current frame range`).
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

        self.frames.lock().insert(fid, frame);
        fid
    }

    /// Frame node initialization: reset + prefill.
    pub(super) fn prepare_frame(&self, frame: &mut Frame) {
        // Reset frame state (mandatory on reuse to avoid stale values).
        frame.value_table.reset_all();
        frame.ready_queue.clear();
        frame.control_signal = ControlSignal::None;
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
        let (loop_kind, cond_node, return_node, iter_next_node, reset_plan) = {
            let sg = &self.graph.subgraphs[loop_sg_id.0 as usize];
            (sg.loop_kind, sg.cond_node, sg.return_node, sg.iter_next_node, sg.reset_plan.clone())
        };
        // Use loop_frame.node_offset rather than subgraph.node_range.0 (same-function branch frame
        // correction).
        let loop_offset = loop_frame.node_offset;

        // 0. Clear ready_queue (must precede steps 1-3 pushing cond/iter_next/gate).
        loop_frame.ready_queue.clear();

        // 1-3. Data-driven reset via ResetPlan when present; otherwise fall back to the LoopKind
        // branch.
        if let Some(plan) = &reset_plan {
            for &nid in &plan.reset_to_zero {
                let local = NodeId(nid.0.wrapping_sub(loop_offset));
                Self::reset_node_ready(loop_frame, local);
                loop_frame.push_ready(local);
            }
            for &nid in &plan.reset_to_one {
                let local = NodeId(nid.0.wrapping_sub(loop_offset));
                Self::reset_node_pending(loop_frame, local, 1);
            }
            for &nid in &plan.reset_condition_tree {
                self.reset_condition_tree(loop_frame, loop_sg_id, nid, loop_offset);
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
        body_frame.value_table.reset_all();
        body_frame.ready_queue.clear();
        body_frame.select_timers.clear();
        body_frame.cached_child_frame = None;
        body_frame.control_signal = ControlSignal::None;

        // Re-copy outer-variable values from loop_frame first (consistent with start_subgraph).
        // This copy must happen before prepare_same_function_frame: the 0-input enqueue logic in
        // prepare_same_function_frame relies on the value-table ready state to derive pending_inputs;
        // if outer variables are not ready, nodes depending on outer variables would be incorrectly
        // marked pending==0 and enqueued, then read empty values at execution (repro: a for-in loop
        // body executing only its first iteration).
        let body_sg = &self.graph.subgraphs[body_frame.subgraph_id.0 as usize];
        let (body_branch_start, body_branch_end) = body_sg.node_range;
        let copy_count = loop_frame.value_table.len().min(body_frame.value_table.len());
        copy_outer_ready_values(body_frame, loop_frame, copy_count, body_branch_start.0, body_branch_end.0);

        // After copying outer variables, set pending_inputs + enqueue 0-input nodes.
        self.prepare_same_function_frame(body_frame);

        // Rebind the body_sg frame's caller.
        body_frame.caller =
            Some((loop_fid, NodeId(return_node.0.wrapping_sub(loop_offset))));
        // Frame-chain pointers set to null (HashMap addresses are unstable).
        body_frame.root_frame_ptr = std::ptr::null_mut();
        body_frame.parent_frame_ptr = std::ptr::null_mut();

        // 5. Reset the loop frame state.
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
        let parent_start = frame.node_offset;
        let parent_node_count = frame.value_table.len();
        let sg_id = frame.subgraph_id;
        let sg = &self.graph.subgraphs[sg_id.0 as usize];
        let branch_start = sg.node_range.0 .0;
        let branch_end = sg.node_range.1 .0;
        let branch_param_count = sg.param_count as usize;

        // Use the precomputed nested_ranges (filled at build time) to avoid a runtime full-graph scan.
        let nested_ranges: &[(u32, u32)] = self.graph.sg_nested_ranges(sg_id.0 as usize);
        let is_nested = |gid: u32| nested_ranges.iter().any(|&(s, e)| gid >= s && gid < e);

        // 1. Set pending_inputs.
        for i in 0..parent_node_count {
            let gid = (parent_start as usize + i) as u32;
            let in_branch = gid >= branch_start && gid < branch_end;
            if !in_branch || is_nested(gid) {
                frame.pending_inputs[i] = PENDING_EXTERNAL;
                continue;
            }
            let node = self.graph.node(gid as usize);
            if node.kind == NodeKind::EventSource {
                frame.pending_inputs[i] = PENDING_EXTERNAL;
            } else if node.kind == NodeKind::Gate
                && self.graph.has_select_info(gid as usize)
            {
                frame.pending_inputs[i] = 0;
            } else {
                let inputs =
                    self.graph
                        .inputs(node.inputs_offset, node.input_count);
                let mut pending = 0u16;
                for &inp in inputs {
                    let il = inp.0.wrapping_sub(parent_start) as usize;
                    if il < parent_node_count {
                        let inp_gid = (parent_start as usize + il) as u32;
                        let inp_in_branch = inp_gid >= branch_start && inp_gid < branch_end;
                        // In-branch node: count toward pending when not ready.
                        // Outer variable (!in_branch): accessed via frame-chain penetration, not counted.
                        if inp_in_branch && !frame.value_table.is_ready(il) {
                            pending += 1;
                        }
                    }
                    // Outside the frame range -> frame-chain penetration, not counted.
                }
                frame.pending_inputs[i] = pending;
            }
        }

        // 2. Enqueue in-branch 0-input non-Param nodes (Const nodes also take this path — compute_fn
        // returns a value).
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
            }
        }
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
        let is_in_sg = |gid: u32| gid >= sg_start.0 && gid < sg_end.0 && !is_nested(gid);

        // DFS-collect all nodes in the cond_node dependency tree that lie inside the loop subgraph
        // (excluding Gate nodes).
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
            if self.graph.node(gid.0 as usize).kind == NodeKind::Gate {
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

            // pending = number of inputs within the dependency tree (these inputs will be
            // re-evaluated). External inputs (e.g. variables outside the loop) are accessed via the
            // frame chain and are already ready, so they are not counted.
            let pending: u16 = inputs
                .iter()
                .filter(|&&inp| {
                    visited.contains(&inp.0)
                        && is_in_sg(inp.0)
                        && self.graph.node(inp.0 as usize).kind != NodeKind::Gate
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
