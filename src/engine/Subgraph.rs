//! Subgraph invocation and return: switch_subgraph + start_subgraph + complete_and_wake_caller.

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::Value;

/// Tail-call graph jump: reuses the current frame to execute the target subgraph (zero pool
/// allocation).
pub fn switch_subgraph(frame: &mut Frame, graph: &DataFlowGraph, target_sg: SubGraphId, args: &[Value]) {
    let (node_start, node_end) = graph.subgraphs[target_sg.0 as usize].node_range;
    let node_count = (node_end.0 - node_start.0) as usize;

    // Update subgraph_id + resize the arrays.
    frame.subgraph_id = target_sg;
    if frame.value_table.len() != node_count {
        frame.value_table.resize(node_count);
    }
    if frame.pending_inputs.len() != node_count {
        frame.pending_inputs.resize(node_count, 0);
    }

    // Clear value_table (prepare_frame_nodes does not do this).
    frame.value_table.reset_all();
    frame.ready_queue.clear();
    frame.control_signal = ControlSignal::None;
    frame.cached_child_frame = None;
    frame.defer_stack.clear();
    frame.select_timers.clear();
    frame.root_frame_ptr = std::ptr::null_mut();
    frame.parent_frame_ptr = std::ptr::null_mut();
    frame.state = FrameState::Ready;
    frame.suspend_state = SuspendState::NotSuspended;
    frame.suspend_event = None;
    // caller is unchanged: the return value goes straight to the original caller's call node.

    // prepare_frame_nodes: set node_offset + pending_inputs + Const prefill.
    prepare_frame_nodes(frame, graph);

    // Argument injection.
    let offset = node_start.0 as usize;
    let param_count = graph.subgraphs[target_sg.0 as usize].param_count as usize;
    for (i, arg) in args.iter().enumerate().take(param_count) {
        let local_id = NodeId(i as u32);
        let global_id = NodeId((offset + i) as u32);
        let consumer_count = graph.downstream_slice(offset + i).len() as u16;
        frame.set_value(local_id, arg.clone(), consumer_count);
        // Do not push_ready: the parameter value is already set by set_value; notify_downstream
        // propagates it to downstream. If we did push_ready, compute_const would be invoked and
        // return VOID, overwriting the parameter value.
        notify_downstream(frame, graph, local_id, global_id, NodeId(node_start.0));
    }
}

// =========================================================================
// impl<S: LockStrategy> Engine<S> — subgraph methods
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// Starts a subgraph: creates a child frame + injects arguments + binds the caller.
    /// For same-function branch subgraphs (if-else/match arm): the value table is sized to the
    /// parent function and parent-frame values are copied, so branch nodes can directly access
    /// outer variables via get_value_by_global (no frame-chain pointers needed).
    pub(super) fn start_subgraph(
        &self,
        caller_fid: FrameId,
        call_node: NodeId,
        subgraph_id: SubGraphId,
        args: &[Value],
        parent_frame: &Frame,
        closure_val: Option<Value>,
    ) -> FrameId {
        let child_fid = self.alloc_frame_id();
        let parent_sg = &self.graph.subgraphs[parent_frame.subgraph_id.0 as usize];
        let child_sg = &self.graph.subgraphs[subgraph_id.0 as usize];
        // The same_function path is used for in-function branch subgraphs (if-else/match arm),
        // whose node_range is strictly contained within the parent function's node_range and which
        // need parent-frame values copied. Direct self-recursion (child_sg.id == parent
        // subgraph_id) must not take this path — it needs a fresh call frame, not a parent-value
        // copy. Direct recursion takes the cross-function path.
        let same_function = parent_sg.function_id == child_sg.function_id
            && subgraph_id != parent_frame.subgraph_id;

        if super::env_flag("KUZO_DEBUG_STALL") {
            let (cs, ce) = child_sg.node_range;
            let child_sz = ce.0 - cs.0;
            if child_sz <= 3 {
                let nested: Vec<(u32, u32, u32)> = self.graph.subgraphs.iter()
                    .filter(|sg| sg.id != subgraph_id
                        && sg.node_range.0 .0 >= cs.0
                        && sg.node_range.1 .0 <= ce.0)
                    .map(|sg| (sg.node_range.0 .0, sg.node_range.1 .0, sg.id.0))
                    .collect();
                eprintln!("[START-SG] target_sg={} same_func={} child_range=[{},{}) parent_sg={} parent_range=[{},{}) nested_count={} nested={:?}",
                    subgraph_id.0, same_function, cs.0, ce.0,
                    parent_frame.subgraph_id.0,
                    parent_sg.node_range.0 .0, parent_sg.node_range.1 .0,
                    nested.len(), nested);
            }
        }

        if same_function {
            // Same-function branch: value table is sized to the parent frame and parent-frame
            // values are copied.
            // Use the parent frame's node_offset/value_table.len() rather than parent_sg.node_range,
            // because a nested closure frame's layout is determined by the grandparent frame (e.g.
            // outer's node_offset is main's node_start, not outer's own node_range.0). Using
            // subgraph.node_range would misalign value-table indices and cause nodes to be
            // mistakenly marked ready, skipping compute_fn.
            let parent_start = parent_frame.node_offset;
            let parent_node_count = parent_frame.value_table.len();
            let (branch_start, _branch_end) = child_sg.node_range;
            let branch_param_count = child_sg.param_count as usize;

            let mut child = self.acquire_frame(child_fid, subgraph_id, parent_node_count);
            child.node_offset = parent_start;

            // Copy the parent frame's ready values (refcount set to 0 = never reclaimed; released
            // all at once when the frame ends).
            // Skip nodes inside the child_sg range: on recursive calls child_sg is the function
            // body subgraph, and stale results of in-branch nodes (e.g. n-1) must not be copied,
            // otherwise the child frame would not recompute them and the recursive argument would
            // not decrement (fact(n-1) repeatedly receives the same stale n-1 value).
            for i in 0..parent_node_count {
                let gid = (parent_start as usize + i) as u32;
                let in_child = gid >= branch_start.0 && gid < child_sg.node_range.1 .0;
                if in_child {
                    continue;
                }
                if parent_frame.value_table.is_ready(i) {
                    child.value_table.values[i] = parent_frame.value_table.values[i].clone();
                    child.value_table.set_ready(i);
                    child.value_table.refcounts[i] = 0;
                }

            }

            // Use the precomputed nested_ranges (filled at build time) to avoid a runtime
            // full-graph scan.
            let nested_ranges: &[(u32, u32)] = self.graph.sg_nested_ranges(subgraph_id.0 as usize);
            let is_nested = |gid: u32| nested_ranges.iter().any(|&(s, e)| gid >= s && gid < e);

            // Set pending_inputs: in-branch nodes count actually-unready inputs; non-branch nodes
            // are marked EXTERNAL.
            for i in 0..parent_node_count {
                let gid = (parent_start as usize + i) as u32;
                let in_branch = gid >= branch_start.0 && gid < child_sg.node_range.1 .0;
                if !in_branch || is_nested(gid) {
                    child.pending_inputs[i] = PENDING_EXTERNAL;
                    continue;
                }
                let node = self.graph.node(gid as usize);
                if node.kind == NodeKind::EventSource {
                    child.pending_inputs[i] = PENDING_EXTERNAL;
                } else if node.kind == NodeKind::Gate && self.graph.has_select_info(gid as usize) {
                    child.pending_inputs[i] = 0;
                } else {
                    // Gate (non-select) and ordinary nodes are unified: count actually-unready
                    // in-frame inputs. Inputs outside the frame range (effect chains, outer
                    // variables) are accessed via frame-chain penetration and are not counted as
                    // pending.
                    let inputs = self.graph.inputs(node.inputs_offset, node.input_count);
                    let mut pending = 0u16;
                    for &inp in inputs {
                        let il = inp.0.wrapping_sub(parent_start) as usize;
                        if il < parent_node_count {
                            let inp_gid = (parent_start as usize + il) as u32;
                            let inp_in_branch = inp_gid >= branch_start.0 && inp_gid < child_sg.node_range.1 .0;
                            // In-branch node: count toward pending when not ready.
                            // Outer variable/effect (!in_branch): accessed via frame-chain
                            // penetration, not counted.
                            if inp_in_branch && !child.value_table.is_ready(il) {
                                pending += 1;
                            }
                        }
                        // Outside the frame range (il >= parent_node_count or underflow) ->
                        // frame-chain penetration, not counted.
                    }
                    child.pending_inputs[i] = pending;
                }
            }

            // Enqueue in-branch 0-input non-Param nodes (must precede argument injection!).
            // Ordering rationale: if argument injection ran first, notify_downstream would zero
            // pending for downstream nodes and enqueue them; the subsequent 0-input enqueue would
            // then see pending==0 && !ready and enqueue them again, causing nodes to execute twice
            // (e.g. for-in's next_call runs twice, consuming two iterator elements and skipping the
            // first element).
            // Putting 0-input enqueue before argument injection means downstream nodes still have
            // pending > 0 and are not enqueued; only 0-input constant nodes are enqueued. Argument
            // injection's notify_downstream then enqueues downstream once.
            // This matches the cross-function path (prepare_frame_nodes before argument injection).
            for i in 0..parent_node_count {
                let gid = (parent_start as usize + i) as u32;
                let in_branch = gid >= branch_start.0 && gid < child_sg.node_range.1 .0;
                if !in_branch || is_nested(gid) { continue; }
                let local_in_branch = (gid - branch_start.0) as usize;
                if local_in_branch < branch_param_count { continue; }
                if child.pending_inputs[i] == 0 && !child.value_table.is_ready(i) {
                    child.push_ready(NodeId(i as u32));
                }
            }

            // Argument injection (local index = branch_start - parent_start + i).
            // Actual arguments inject the arg values supplied by the caller.
            // Upvalue arguments inject the current parent-frame value (capture-by-reference
            // semantics), so the same_function call sees the latest outer-variable values (rather
            // than the snapshot taken at closure construction).
            let param_local_offset = branch_start.0.wrapping_sub(parent_start) as usize;
            let actual_param_count = branch_param_count
                .saturating_sub(child_sg.upvalue_count as usize);
            // Actual arguments.
            for (i, arg) in args.iter().enumerate().take(actual_param_count) {
                let lid = NodeId((param_local_offset + i) as u32);
                let gid = branch_start.0 as usize + i;
                let global_id = NodeId(gid as u32);
                let cc = self.graph.downstream_slice(gid).len() as u16;
                child.set_value(lid, arg.clone(), cc);
                // Do not push_ready: the parameter value is already set; notify_downstream
                // propagates it downstream.
                notify_downstream(&mut child, &self.graph, lid, global_id, NodeId(parent_start));
            }
            // Upvalue argument injection: read the latest value from the parent frame
            // (capture-by-reference semantics), so the same_function call sees the latest
            // outer-variable values (rather than the snapshot taken at closure construction).
            // Recursive-closure exception: the slot at self_upvalue_idx is injected with the
            // closure's own self reference, not a parent-frame value (the self slot in the parent
            // frame is a void_const placeholder).
            let self_upvalue_idx = closure_val.as_ref()
                .and_then(|v| v.heap_obj())
                .and_then(|h| match h {
                    crate::value::HeapObj::Closure(c) => Some(c.self_upvalue_idx),
                    crate::value::HeapObj::Partial(p) => Some(p.self_upvalue_idx),
                    _ => None,
                })
                .unwrap_or(-1);
            for (i, &outer_node) in self.graph.sg_upvalue_outer_nodes(subgraph_id.0 as usize).iter().enumerate() {
                let arg_idx = actual_param_count + i;
                if arg_idx >= branch_param_count { break; }
                let lid = NodeId((param_local_offset + arg_idx) as u32);
                let gid = branch_start.0 as usize + arg_idx;
                let global_id = NodeId(gid as u32);
                let cc = self.graph.downstream_slice(gid).len() as u16;
                let val = if self_upvalue_idx >= 0 && i == self_upvalue_idx as usize {
                    closure_val.clone().unwrap_or_else(|| parent_frame.get_value_by_global(outer_node))
                } else {
                    parent_frame.get_value_by_global(outer_node)
                };
                child.set_value(lid, val, cc);
                // Do not push_ready: the parameter value is already set; notify_downstream
                // propagates it downstream.
                notify_downstream(&mut child, &self.graph, lid, global_id, NodeId(parent_start));
            }

            child.caller = Some((caller_fid, call_node));

            // Frame-chain pointers are set later by setup_frame_chain in process_frame.
            child.root_frame_ptr = std::ptr::null_mut();
            child.parent_frame_ptr = std::ptr::null_mut();
            child.closure_val = closure_val;

            if std::env::var("KUZO_DEBUG_FORIN").is_ok() {
                let rq_len = child.ready_queue.len();
                let mut pending_info: Vec<(u32, u16, bool)> = Vec::new();
                for i in 0..parent_node_count {
                    let gid = (parent_start as usize + i) as u32;
                    if gid >= branch_start.0 && gid < child_sg.node_range.1 .0 {
                        let p = child.pending_inputs[i];
                        let r = child.value_table.is_ready(i);
                        if p != PENDING_EXTERNAL {
                            pending_info.push((gid, p, r));
                        }
                    }
                }
                eprintln!("[FORIN-SG-CREATE] sg={} rq_len={} pending_info={:?}",
                    subgraph_id.0, rq_len, &pending_info[..pending_info.len().min(15)]);
            }

            self.frames.lock().insert(child_fid, child);
            child_fid
        } else {
            // Cross-function call: original logic.
            let (node_start, node_end) = child_sg.node_range;
            let node_count = (node_end.0 - node_start.0) as usize;
            let offset = node_start.0 as usize;

            let mut child = self.acquire_frame(child_fid, subgraph_id, node_count);
            self.prepare_frame(&mut child);

            let param_count = child_sg.param_count as usize;
            for (i, arg) in args.iter().enumerate().take(param_count) {
                let local_id = NodeId(i as u32);
                let global_id = NodeId((offset + i) as u32);
                let consumer_count = self.graph.downstream_slice(offset + i).len() as u16;
                child.set_value(local_id, arg.clone(), consumer_count);
                // Do not push_ready: the parameter value is already set; notify_downstream
                // propagates it downstream.
                notify_downstream(&mut *child, &self.graph, local_id, global_id, NodeId(node_start.0));
            }

            child.caller = Some((caller_fid, call_node));
            child.root_frame_ptr = std::ptr::null_mut();
            child.parent_frame_ptr = std::ptr::null_mut();
            child.closure_val = closure_val;

            self.frames.lock().insert(child_fid, child);
            child_fid
        }
    }

    /// After a subgraph completes: write the return value back to the caller and wake the caller.
    /// Handles LoopBody completion detection and the pending_completions race. Uses iterative
    /// propagation of LoopBody break/return to avoid stack overflow on deeply nested loops.
    pub(super) fn complete_and_wake_caller(&self, mut child_frame: Frame, queue: &QueueHandle<'_>) {
        // LoopBody break/return propagation loop (iterative, replacing recursion).
        loop {
            let child_sg_id = child_frame.subgraph_id;
            let child_loop_kind = self.graph.subgraphs[child_sg_id.0 as usize].loop_kind;
            if child_loop_kind != crate::ir::Ir::LoopKind::LoopBody {
                break; // Not a LoopBody; enter the normal completion path.
            }
            let child_signal = child_frame.control_signal.clone();
            let (loop_fid, _call_node) = child_frame
                .caller
                .expect("LoopBody frame missing caller");
            match child_signal {
                ControlSignal::Break | ControlSignal::Return(_) => {
                    // break/return -> loop exits.
                    let mut loop_frame = self.frames.lock().remove(&loop_fid);
                    if let Some(lf) = loop_frame.as_deref_mut() {
                        lf.cached_child_frame = None;
                        lf.control_signal = child_signal.clone();
                    }
                    // Iterate on loop_frame (loop_kind is usually While/Loop/For, not LoopBody,
                    // but if it is a nested LoopBody we keep propagating iteratively to avoid
                    // recursive stack overflow).
                    match loop_frame {
                        Some(lf) => {
                            child_frame = *lf; // Iterate instead of recursing.
                            continue;
                        }
                        None => {
                            // Bug #78: loop_frame is being processed by another worker (concurrent
                            // race). Store the completion into pending_completions so process_frame
                            // propagates the control_signal when the loop_frame is re-inserted.
                            let return_value = super::Schedule::extract_child_return(&child_frame, &self.graph);
                            let call_node = child_frame
                                .caller
                                .map(|(_, cn)| cn)
                                .expect("LoopBody frame missing caller");
                            self.release_frame(Box::new(child_frame));
                            self.pending_completions
                                .lock()
                                .entry(loop_fid)
                                .or_insert_with(Vec::new)
                                .push((call_node, return_value, child_signal));
                            return;
                        }
                    }
                }
                ControlSignal::Continue => {
                    // continue -> loop reset (frame reuse).
                    let mut loop_frame_opt = self.frames.lock().remove(&loop_fid);
                    if let Some(loop_frame) = loop_frame_opt.as_deref_mut() {
                        let mut child = child_frame; // Take ownership to modify.
                        self.reset_loop_iteration(&mut *loop_frame, loop_fid, &mut child);
                        let body_id = child.id;
                        let loop_box = loop_frame_opt.take().unwrap();
                        self.frames.lock().insert(loop_fid, loop_box);
                        queue.push(loop_fid);
                        self.frames.lock().insert(body_id, Box::new(child));
                        return;
                    }
                    // Bug #78: loop_frame is being processed by another worker (concurrent race).
                    // Continue is treated as None so the loop_frame resumes normal iteration when
                    // re-inserted (the loop condition gate re-evaluates and starts a new body).
                    let return_value = super::Schedule::extract_child_return(&child_frame, &self.graph);
                    let call_node = child_frame
                        .caller
                        .map(|(_, cn)| cn)
                        .expect("LoopBody frame missing caller");
                    self.release_frame(Box::new(child_frame));
                    self.pending_completions
                        .lock()
                        .entry(loop_fid)
                        .or_insert_with(Vec::new)
                        .push((call_node, return_value, ControlSignal::None));
                    return;
                }
                ControlSignal::None => {
                    // Normal completion: check the caller's loop kind.
                    if std::env::var("KUZO_DEBUG_FORIN").is_ok() {
                        let bsg = &self.graph.subgraphs[child_frame.subgraph_id.0 as usize];
                        let rq_len = child_frame.ready_queue.len();
                        let (bs, be) = bsg.node_range;
                        let mut unready: Vec<u32> = Vec::new();
                        for i in 0..child_frame.value_table.len() {
                            let gid = (child_frame.node_offset as usize + i) as u32;
                            if gid >= bs.0 && gid < be.0 && !child_frame.value_table.is_ready(i) {
                                unready.push(gid);
                            }
                        }
                        eprintln!("[FORIN-BODY-DONE] child_sg={} rq_len={} unready_count={} unready={:?}",
                            child_frame.subgraph_id.0, rq_len, unready.len(), &unready[..unready.len().min(10)]);
                    }
                    let loop_frame_opt = self.frames.lock().remove(&loop_fid);
                    match loop_frame_opt {
                        Some(mut loop_frame) => {
                            let loop_kind = self.graph.subgraphs[loop_frame.subgraph_id.0 as usize].loop_kind;
                            if loop_kind == crate::ir::Ir::LoopKind::TailRec {
                                // TailRec loop: body_sg completes with no signal = base case hit.
                                // Extract body_sg's return value and convert it to a Return signal to exit
                                // the loop.
                                let return_value = super::Schedule::extract_child_return(&child_frame, &self.graph);
                                loop_frame.cached_child_frame = None;
                                loop_frame.control_signal = ControlSignal::Return(return_value);
                                child_frame = *loop_frame;
                                continue;
                            } else {
                                // Ordinary loop (While/Loop/For): normal completion -> loop reset (frame
                                // reuse).
                                let mut child = child_frame;
                                self.reset_loop_iteration(&mut *loop_frame, loop_fid, &mut child);
                                self.frames.lock().insert(loop_fid, loop_frame);
                                queue.push(loop_fid);
                                let body_id = child.id;
                                self.frames.lock().insert(body_id, Box::new(child));
                                return;
                            }
                        }
                        None => {
                            // Bug #78: loop_frame is being processed by another worker (concurrent
                            // race). Store the completion into pending_completions; the loop_frame
                            // resumes normal iteration when re-inserted. For TailRec the base-case
                            // return value is written to the call_node; the next iteration's body_sg
                            // will re-evaluate the base case and emit a Return signal through the
                            // normal path.
                            let return_value = super::Schedule::extract_child_return(&child_frame, &self.graph);
                            let call_node = child_frame
                                .caller
                                .map(|(_, cn)| cn)
                                .expect("LoopBody frame missing caller");
                            self.release_frame(Box::new(child_frame));
                            self.pending_completions
                                .lock()
                                .entry(loop_fid)
                                .or_insert_with(Vec::new)
                                .push((call_node, return_value, ControlSignal::None));
                            return;
                        }
                    }
                }
            }
        }

        // Not a LoopBody: write back the return value + wake the caller (with pending_completions
        // race handling).
        let child_sg_id = child_frame.subgraph_id;
        let return_value = super::Schedule::extract_child_return(&child_frame, &self.graph);
        let child_signal = child_frame.control_signal.clone();
        let caller = child_frame.caller;
        // Return the child frame to the pool (Vec capacity is retained for reuse).
        self.release_frame(Box::new(child_frame));

        if let Some((caller_fid, call_node)) = caller {
            let mut caller_frame_opt = self.frames.lock().remove(&caller_fid);
            if caller_frame_opt.is_none() {
                // The parent frame has not yet been inserted back into the HashMap; store the
                // completion info for a later retry.
                // Use a Vec to avoid concurrent completions of multiple child frames for the same
                // caller overwriting each other.
                self.pending_completions
                    .lock()
                    .entry(caller_fid)
                    .or_insert_with(Vec::new)
                    .push((call_node, return_value, child_signal));
                return;
            }
            if let Some(caller_frame) = caller_frame_opt.as_deref_mut() {
                // Use caller_frame.node_offset rather than subgraph.node_range.0:
                // a same-function branch frame's node_offset is the parent function's node_start,
                // whereas subgraph.node_range.0 is the branch subgraph's node_start — the two
                // differ. Using the wrong one yields an incorrect call_graph_id offset ->
                // notify_downstream finds no downstream -> the downstream node's ready flag is
                // never set -> the frame hangs.
                let caller_offset = NodeId(caller_frame.node_offset);
                let call_graph_id = NodeId(call_node.0 + caller_offset.0);
                let consumer_count =
                    self.graph.downstream_slice(call_graph_id.0 as usize).len() as u16;

                if std::env::var("KUZO_DEBUG_IFELSE").is_ok() {
                    let child_sg = &self.graph.subgraphs[child_sg_id.0 as usize];
                    eprintln!("[COMPLETE] child_sg={} caller_fid={:?} call_node_local={} call_graph_id={} caller_offset={} return_value={:?} child_loop_kind={:?} caller_sg={}",
                        child_sg_id.0, caller_fid, call_node.0, call_graph_id.0, caller_offset.0,
                        return_value, child_sg.loop_kind, caller_frame.subgraph_id.0);
                }

                caller_frame.set_value(call_node, return_value, consumer_count);
                caller_frame.state = FrameState::Ready;
                caller_frame.suspend_state = SuspendState::NotSuspended;
                caller_frame.suspend_event = None;

                // Control-signal propagation: the child frame's throw/return/break/continue signal
                // propagates to the caller frame. Propagation is in-function only (Gate branch
                // subgraphs, loop subgraphs):
                // - if-else/match arm (Gate node) throw/return/break/continue -> propagate to the
                //   parent frame (break/continue must penetrate to the LoopBody frame, otherwise an
                //   if-break inside a loop body has no effect).
                // - while/loop/for (loop frame) throw/return -> propagate to the function frame.
                // Cases that do NOT propagate:
                // - Cross-function calls: a function frame's Return signal is a function-level
                //   return; the return value has already been extracted via extract_child_return,
                //   so propagating it would make the caller frame exit prematurely.
                // - Lambda/nested-function calls (Call node + loop_kind==None + same function_id):
                //   although it shares the caller's function_id (for frame-chain penetration), it is
                //   an independent function call whose return value is already extracted;
                //   propagating Return would make the caller frame exit incorrectly (silent-exit
                //   bug).
                // - Loop frame's Break/Continue: already consumed by the loop; propagating would
                //   cause the function to exit incorrectly.
                let child_loop_kind = self.graph.subgraphs[child_sg_id.0 as usize].loop_kind;
                let is_gate = self.graph.node(call_graph_id.0 as usize).kind
                    == crate::ir::Ir::NodeKind::Gate;
                let should_propagate = match child_signal {
                    ControlSignal::Return(_) => {
                        // Return: propagate from Gate branches + loop frames; not from
                        // Lambda/function calls.
                        is_gate || child_loop_kind != crate::ir::Ir::LoopKind::None
                    }
                    ControlSignal::Break | ControlSignal::Continue => {
                        // Break/Continue: propagate from Gate branches only (penetrate to
                        // LoopBody). A loop frame's Break/Continue has already been consumed by the
                        // loop.
                        is_gate
                    }
                    ControlSignal::None => false,
                };
                if should_propagate {
                    let child_fn_id = self.graph.subgraphs[child_sg_id.0 as usize].function_id;
                    let caller_fn_id =
                        self.graph.subgraphs[caller_frame.subgraph_id.0 as usize].function_id;
                    if child_fn_id == caller_fn_id {
                        caller_frame.control_signal = child_signal;
                    }
                }

                notify_downstream(
                    caller_frame,
                    &self.graph,
                    call_node,
                    call_graph_id,
                    caller_offset,
                );
            }
            if let Some(caller_frame) = caller_frame_opt {
                self.frames.lock().insert(caller_fid, caller_frame);
                queue.push(caller_fid);
            }
        }
        // The child frame has completed; the caller is responsible for dropping it (it is not put
        // back into frames).
    }
}
