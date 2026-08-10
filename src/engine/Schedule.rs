//! Dataflow scheduling core: readiness-scheduling free functions, run_frame_nodes, process_frame.
//!
//! SIMD batching has been pushed down into compute_fn (via EvalContext + do_simd_batch), so the
//! engine hot loop no longer has batching-specialization checks.

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::Value;
use crate::ir::Compute::char_from_u32_or_nul;

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
            use crate::value::{HeapObj, KuzoStr};
            let off = offset as usize;
            let end = off + len as usize;
            let s = std::str::from_utf8(&pool[off..end]).unwrap_or("");
            Value::ref_val(HeapObj::Str(KuzoStr::new(s)))
        }
    }
}

/// Frame node initialization: sets node_offset + pending_inputs + prefills Const + enqueues Gate
/// into the ready queue.
pub fn prepare_frame_nodes(frame: &mut Frame, graph: &DataFlowGraph) {
    let sg_id = frame.subgraph_id;
    let (node_start, node_end) = graph.subgraphs[sg_id.0 as usize].node_range;
    let node_count = (node_end.0 - node_start.0) as usize;
    let offset = node_start.0 as usize;
    let node_end_global = node_start.0 + node_count as u32;

    // Use the precomputed nested_ranges (filled at build time) to avoid a runtime full-graph scan.
    let nested_ranges: &[(u32, u32)] = graph.sg_nested_ranges(sg_id.0 as usize);

    if super::env_flag("KUZO_DEBUG_STALL") {
        eprintln!("[PREPARE] sg={} node_range=[{},{}) nested={:?}",
            sg_id.0, node_start.0, node_end_global, nested_ranges);
    }

    let is_nested = |global_idx: u32| -> bool {
        nested_ranges.iter().any(|&(s, e)| global_idx >= s && global_idx < e)
    };

    // Set node_offset.
    frame.node_offset = node_start.0;

    // 1. Initialize pending_inputs (select Gate -> 0; other nodes count actual in-frame inputs).
    for i in 0..node_count {
        if is_nested((offset + i) as u32) {
            frame.pending_inputs[i] = PENDING_EXTERNAL;
        } else {
            let graph_node = graph.node(offset + i);
            if graph_node.kind == NodeKind::EventSource {
                frame.pending_inputs[i] = PENDING_EXTERNAL;
            } else if graph_node.kind == NodeKind::Gate
                && graph.has_select_info(offset + i)
            {
                frame.pending_inputs[i] = 0;
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
        if is_nested((offset + i) as u32) {
            continue;
        }
        if frame.pending_inputs[i] == 0 && !frame.value_table.is_ready(i) {
            frame.push_ready(NodeId(i as u32));
        }
    }
}

/// Notifies downstream nodes: decrements pending_inputs, and enqueues them when it reaches zero
/// (with bounds checks + slot-level RC).
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

// =========================================================================
// impl<S: LockStrategy> Engine<S> — scheduling core methods
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// Executes all ready nodes in the frame until the ready queue is empty or the frame suspends.
    pub(super) fn run_frame_nodes(&self, frame: &mut Frame, fid: FrameId, queue: &QueueHandle<'_>) {
        let graph = frame.graph.clone();

        let mut iter_guard: u64 = 0;
        loop {
        iter_guard += 1;
        if iter_guard > 500000 {
            // Over the limit: mark Failed to prevent process_frame from re-enqueuing and causing a
            // livelock. process_frame's Failed branch wakes the caller or returns NULL.
            frame.state = FrameState::Failed;
            return;
        }
            // Check the control signal (return/break/continue already triggered).
            if !matches!(frame.control_signal, ControlSignal::None) {
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
                    if super::env_flag("KUZO_DEBUG_STALL") {
                        let sg_id = frame.subgraph_id;
                        let (ns, ne) = graph.subgraphs[sg_id.0 as usize].node_range;
                        let ncnt = (ne.0 - ns.0) as usize;
                        eprintln!("[STALL] frame={} sg={} node_range=[{},{}) control={:?}",
                            fid.0, sg_id.0, ns.0, ne.0, frame.control_signal);
                        for i in 0..ncnt {
                            let gid = NodeId(i as u32 + ns.0);
                            let n = graph.node(gid.0 as usize);
                            let ready = i < frame.value_table.len() && frame.value_table.is_ready(i);
                            let pi = frame.pending_inputs[i];
                            if pi != PENDING_EXTERNAL || !ready {
                                let inputs = graph.inputs(n.inputs_offset, n.input_count);
                                eprintln!("  node={} kind={:?} cf={} ready={} pending={} inputs={:?}",
                                    gid.0, n.kind, n.compute_fn.0, ready, pi, inputs);
                            }
                        }
                    }
                    break;
                }
            };

            let node_start = frame.node_offset;
            let graph_node_id = NodeId(local_id.0 + node_start);
            let node = graph.node(graph_node_id.0 as usize);
            let ctx = EvalContext { node_start };

            // COMPUTE: uniformly invoke compute_fn, with no specialization checks.
            let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, graph_node_id, &ctx);

            // MATCH NodeResult: unified side-effect handling.
            match result {
                NodeResult::Value(v) => {
                    let cc = graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
                    frame.set_value(local_id, v, cc);
                    notify_downstream(frame, &graph, local_id, graph_node_id, NodeId(node_start));
                }
                NodeResult::Batch(results) => {
                    for &(lid, ref v) in &results {
                        let gid = NodeId(lid.0 + node_start);
                        let cc = graph.downstream_slice(gid.0 as usize).len() as u16;
                        frame.set_value(lid, v.clone(), cc);
                    }
                    for &(lid, _) in &results {
                        frame.ready_queue.retain(|n| *n != lid);
                    }
                    for &(lid, _) in &results {
                        let gid = NodeId(lid.0 + node_start);
                        notify_downstream(frame, &graph, lid, gid, NodeId(node_start));
                    }
                }
                NodeResult::Call(pending) => {
                    // Tail-call graph jump.
                    let graph_call_id = NodeId(pending.call_node_local.0 + frame.node_offset);
                    if graph.tail_call_flag(graph_call_id.0 as usize) {
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
                            self.event_waiters.lock().retain(|(_, f)| *f != caller_fid);
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

                    // LoopBody frame reuse.
                    let target_loop_kind =
                        graph.subgraphs[pending.target_sg.0 as usize].loop_kind;
                    let child_fid = if target_loop_kind
                        == crate::ir::Ir::LoopKind::LoopBody
                    {
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
                                        graph.downstream_slice(gid).len() as u16;
                                    bf.set_value(local_id, arg.clone(), consumer_count);
                                    // Do not push_ready: the parameter value is already set;
                                    // notify_downstream propagates it downstream.
                                    notify_downstream(bf, &graph, local_id, global_id, NodeId(parent_start));
                                }
                                bf.caller = Some((fid, pending.call_node_local));
                                bf.parent_frame_ptr = std::ptr::null_mut();
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
                            if std::env::var("KUZO_DEBUG_FORIN").is_ok() {
                                let bsg = &graph.subgraphs[pending.target_sg.0 as usize];
                                eprintln!("[FORIN-CREATE] body_sg={} bfid={:?} args={:?} body_range=[{},{})",
                                    pending.target_sg.0, bfid, pending.args,
                                    bsg.node_range.0 .0, bsg.node_range.1 .0);
                            }
                            frame.cached_child_frame = Some(bfid);
                            bfid
                        }
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
                            graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
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
                        continue;
                    } else {
                        queue.push(child_fid);
                        self.event_waiters.lock().push((
                            RuntimeEvent::SubgraphComplete(child_fid),
                            fid,
                        ));
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
                            graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
                        frame.set_value(await_node_local, value, consumer_count);
                        notify_downstream(
                            frame,
                            &graph,
                            await_node_local,
                            graph_node_id,
                            NodeId(node_start),
                        );
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
                        graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
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
                        graph.downstream_slice(graph_node_id.0 as usize).len() as u16;
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
                        let mut ready_branch: Option<SubGraphId> = None;
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
                                ready_branch = Some(branch.subgraph_id);
                                break;
                            }
                        }

                        if let Some(sg_id) = ready_branch {
                            let child_fid =
                                self.start_subgraph(fid, gate_local, sg_id, &[], frame, None);
                            queue.push(child_fid);
                            self.event_waiters.lock().push((
                                RuntimeEvent::SubgraphComplete(child_fid),
                                fid,
                            ));
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
                                self.event_waiters.lock().push((event, fid));
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

        // Frame suspended: do not execute defer, do not mark Completed.
        if frame.state == FrameState::Suspended {
            return;
        }

        // Frame cancelled: execute defer cleanup + mark Failed (spec 5.3).
        if frame.state == FrameState::Cancelling {
            let defer_entries: Vec<DeferEntry> = {
                let sg_id = frame.subgraph_id;
                graph.subgraphs[sg_id.0 as usize].defer_table.clone()
            };
            for entry in defer_entries.iter().rev() {
                let defer_fid = self.init_defer_frame(entry.body_subgraph, frame);
                let mut defer_frame = self.frames.lock().remove(&defer_fid);
                if let Some(df) = defer_frame.as_deref_mut() {
                    self.run_frame_nodes(df, defer_fid, queue);
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

        // Execute defer (LIFO): any termination path runs defer.
        let defer_entries: Vec<DeferEntry> = {
            let sg_id = frame.subgraph_id;
            graph.subgraphs[sg_id.0 as usize].defer_table.clone()
        };
        for entry in defer_entries.iter().rev() {
            let defer_fid = self.init_defer_frame(entry.body_subgraph, frame);
            let mut defer_frame = self.frames.lock().remove(&defer_fid);
            if let Some(df) = defer_frame.as_deref_mut() {
                self.run_frame_nodes(df, defer_fid, queue);
            }
            if let Some(df) = defer_frame {
                if df.state != FrameState::Completed {
                    self.frames.lock().insert(defer_fid, df);
                }
            }
        }

        // Mark the frame completed.
        frame.state = FrameState::Completed;
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
        self.run_frame_nodes(frame, fid, queue);

        // Handle the frame state.
        let state = frame.state;
        let has_caller = frame.caller.is_some();

        match state {
            FrameState::Suspended => {
                let event = frame.suspend_event;
                // Check pending_completions (the race where a child frame completes before the
                // parent frame is re-inserted). Use a Vec to support concurrent completion of
                // multiple child frames for the same caller (avoiding overwrites).
                let completions: Vec<_> =
                    self.pending_completions.lock().remove(&fid).unwrap_or_default();
                if !completions.is_empty() {
                    // Pending completion(s) present: consume the completion events directly.
                    if let Some(e) = event {
                        self.event_waiters
                            .lock()
                            .retain(|(we, wf)| !(*we == e && *wf == fid));
                    } else {
                        self.event_waiters
                            .lock()
                            .retain(|(_, wf)| *wf != fid);
                    }
                    // Use frame.node_offset rather than subgraph.node_range.0 (same-function
                    // branch frame correction).
                    let caller_offset = NodeId(frame.node_offset);
                    // Walk all completions, writing back each return value + propagating the
                    // signal + notifying downstream.
                    for (call_node, return_value, child_signal) in completions {
                        let call_graph_id = NodeId(call_node.0 + caller_offset.0);
                        let consumer_count =
                            self.graph.downstream_slice(call_graph_id.0 as usize).len() as u16;
                        frame.set_value(call_node, return_value, consumer_count);
                        // Gate branch subgraph control-signal propagation (consistent with the
                        // normal path in complete_and_wake_caller).
                        let is_gate = self.graph.node(call_graph_id.0 as usize).kind
                            == crate::ir::Ir::NodeKind::Gate;
                        if is_gate && !matches!(child_signal, ControlSignal::None) {
                            frame.control_signal = child_signal;
                        }
                        notify_downstream(
                            frame,
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
                    self.frames.lock().insert(fid, frame_box);
                    queue.push(fid);
                } else {
                    // Check pending_events (race fallback for when an event arrives while the frame
                    // is absent from the HashMap).
                    let pending_evt = self.pending_events.lock().remove(&fid);
                    if let Some((_evt, evt_val)) = pending_evt {
                        // Pending event present: inject the event value + wake.
                        // The waiter has already been removed in on_event_arrived, so no duplicate
                        // cleanup is needed.
                        if self.apply_event_to_frame(frame, evt_val) {
                            self.frames.lock().insert(fid, frame_box);
                            queue.push(fid);
                        } else {
                            // Frame is not WaitingEvent (state inconsistency): put it back, do not
                            // enqueue.
                            self.frames.lock().insert(fid, frame_box);
                        }
                    } else {
                        self.frames.lock().insert(fid, frame_box);
                    }
                }
            }
            FrameState::Completed => {
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
                        self.event_waiters.lock().retain(|(e, _)| {
                            !matches!(e, RuntimeEvent::SubgraphComplete(c) if *c == fid)
                        });
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
                if has_caller {
                    // Failed child frame (after cancel): clean up the waiter + wake the caller.
                    self.event_waiters.lock().retain(|(e, _)| {
                        !matches!(e, RuntimeEvent::SubgraphComplete(c) if *c == fid)
                    });
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
}
