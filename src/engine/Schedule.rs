//! 数据流调度核心：就绪调度自由函数、run_frame_nodes、process_frame。
//!
//! SIMD 批量化已下沉到 compute_fn 内部（通过 EvalContext + do_simd_batch），
//! engine 热循环不再有批处理特化检查。

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::Value;
use crate::ir::Compute::char_from_u32_or_nul;

// =========================================================================
// 帧操作辅助函数（纯函数，不依赖 Engine 状态）
// =========================================================================

/// 将 ConstValue 转换为 Value（不使用 arena，直接构造）。
/// `pool` = DataFlowGraph.string_pool 的字节切片，Str 变体需要从中读取字符串。
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

/// 帧节点初始化：设置 node_offset + pending_inputs + 预填充 Const + Gate 入就绪队列。
pub fn prepare_frame_nodes(frame: &mut Frame, graph: &DataFlowGraph) {
    let sg_id = frame.subgraph_id;
    let (node_start, node_end) = graph.subgraphs[sg_id.0 as usize].node_range;
    let node_count = (node_end.0 - node_start.0) as usize;
    let offset = node_start.0 as usize;
    let node_end_global = node_start.0 + node_count as u32;

    // 使用预计算的 nested_ranges（构建期填充），避免运行时全图扫描
    let nested_ranges: &[(u32, u32)] = graph.sg_nested_ranges(sg_id.0 as usize);

    if super::env_flag("KUZO_DEBUG_STALL") {
        eprintln!("[PREPARE] sg={} node_range=[{},{}) nested={:?}",
            sg_id.0, node_start.0, node_end_global, nested_ranges);
    }

    let is_nested = |global_idx: u32| -> bool {
        nested_ranges.iter().any(|&(s, e)| global_idx >= s && global_idx < e)
    };

    // 设置 node_offset
    frame.node_offset = node_start.0;

    // 1. 初始化 pending_inputs（select Gate→0；其他节点按实际 in-frame 输入计数）
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

    // 2. 0-input 节点入就绪队列（Const 节点也走此路径——compute_fn 返回值）
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

/// 通知下游节点：减 pending_inputs，归零则入就绪队列（含边界检查 + 槽级 RC）。
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
        // 边界检查：跳过跨子图下游
        if ds_local_id.0 as usize >= pending_len {
            continue;
        }

        let pidx = producer_local.0 as usize;
        // 消费 producer 的引用计数，但不清除 ready 标记。
        // ready 标记的语义是"此节点已产出值，不需要重新执行"。
        // 清除 ready 会导致节点变成 pending_inputs=0 && ready=false 状态，
        // 当上游被重新触发时，节点会被重新推入 ready_queue 并重复执行，
        // 造成指数级爆炸（尤其影响 call/closure_call 节点）。
        // 值保留在 value_table 中，直到帧结束（帧 drop 时自动释放）
        // 或被 reset_node_ready/reset_node_pending 显式重置（循环体复用场景）。
        let _still_has_consumers = frame.value_table.consume(pidx);

        // 跳过 PENDING_EXTERNAL 哨兵（嵌套子图节点/EventSource 节点）：
        // 这些节点由子帧或事件驱动，不应被父帧的 notify_downstream 递减。
        // 若递减会腐蚀哨兵（65535→65534），累计 65535 次后归零，嵌套节点被错误推入父帧执行。
        //
        // 只有 pending 从 >0 递减到 0 时才 push_ready。
        // pending=0 的节点已被 prepare_frame_nodes 的 0-input 入队推入，
        // 重复 push 会导致节点被多次执行（如 Gate 节点重复触发子帧启动，
        // 子帧返回值覆盖、条件值未就绪时 Gate 提前执行读到 null）。
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

/// 提取子帧返回值：优先取 control_signal 的 Return 值，否则取 return_node 值。
pub(super) fn extract_child_return(child: &Frame, graph: &DataFlowGraph) -> Value {
    match &child.control_signal {
        ControlSignal::Return(v) => v.clone(),
        ControlSignal::Break | ControlSignal::Continue => Value::VOID,
        ControlSignal::None => {
            let sg = &graph.subgraphs[child.subgraph_id.0 as usize];
            // child.node_offset：跨函数调用 = 子图 node_range.0；
            // 同函数分支帧 = 父函数 node_start（见 Frame.rs prepare_same_function_frame）。
            // 使用 child.node_offset（实际值）而非 sg.node_range.0，确保两种情况都正确。
            let return_local = NodeId(sg.return_node.0.wrapping_sub(child.node_offset));
            child.get_value(return_local)
        }
    }
}

// =========================================================================
// impl<S: LockStrategy> Engine<S> — 调度核心方法
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// 执行帧内所有就绪节点，直到就绪队列空或帧挂起。
    pub(super) fn run_frame_nodes(&self, frame: &mut Frame, fid: FrameId, queue: &QueueHandle<'_>) {
        let graph = frame.graph.clone();

        let mut iter_guard: u64 = 0;
        loop {
        iter_guard += 1;
        if iter_guard > 500000 {
            // 超限：标记 Failed 防止 process_frame 重新入队导致活锁
            // process_frame 的 Failed 分支会唤醒调用方或返回 NULL
            frame.state = FrameState::Failed;
            return;
        }
            // 检查控制信号（return/break/continue 已触发）
            if !matches!(frame.control_signal, ControlSignal::None) {
                break;
            }
            // 检查帧是否被取消
            if frame.state == FrameState::Cancelling {
                break;
            }
            // 检查帧是否挂起
            if frame.state == FrameState::Suspended {
                return;
            }

            // POP: 弹出就绪节点（局部 id）
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

            // COMPUTE: 统一调用 compute_fn，无特化检查
            let result = (graph.compute_fns[node.compute_fn.0 as usize])(frame, graph_node_id, &ctx);

            // MATCH NodeResult: 统一副作用处理
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
                    // 尾调用图跳转
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

                    // LoopBody 帧复用
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
                                    // 不 push_ready：参数值已设置，notify_downstream 传播给下游
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
                    // send 成功后仍需设置节点值 + 通知下游，否则后续语句永远不就绪
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

        // 帧挂起：不执行 defer，不标记 Completed
        if frame.state == FrameState::Suspended {
            return;
        }

        // 帧被取消：执行 defer 清理 + 标记 Failed（spec 5.3）
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

        // 执行 defer（LIFO）：任何终止路径都执行 defer
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

        // 标记帧完成
        frame.state = FrameState::Completed;
    }

    /// 处理一帧：timer 检查 + run_frame_nodes + 状态转换。
    /// 返回 ()，结果通过 self.result.lock() 传递
    pub(super) fn process_frame(&self, fid: FrameId, queue: &QueueHandle<'_>) {
        // 检查 timer 事件
        self.check_timers(queue);

        // 取出帧（保持 Box 不 unbox：堆地址在 remove/insert 周期中保持稳定，
        // 使其他帧持有的 parent_frame_ptr/root_frame_ptr 不会悬挂）
        let mut frame_box = match self.frames.lock().remove(&fid) {
            Some(b) => b,
            None => return,
        };
        let frame: &mut Frame = &mut *frame_box;

        // 设置帧链指针：从 HashMap 中查找 caller 链，设置 parent_frame_ptr/root_frame_ptr。
        // 此时所有父帧仍在 HashMap 中（Box 地址稳定）。
        self.setup_frame_chain(frame);

        // 执行帧就绪节点（无锁）
        self.run_frame_nodes(frame, fid, queue);

        // 处理帧状态
        let state = frame.state;
        let has_caller = frame.caller.is_some();

        match state {
            FrameState::Suspended => {
                let event = frame.suspend_event;
                // 检查 pending_completions（子帧先完成但父帧尚未 insert 的竞态）
                // 使用 Vec 支持同一 caller 多个子帧并发完成（避免互相覆盖）
                let completions: Vec<_> =
                    self.pending_completions.lock().remove(&fid).unwrap_or_default();
                if !completions.is_empty() {
                    // 有 pending completion(s)：直接消费完成事件
                    if let Some(e) = event {
                        self.event_waiters
                            .lock()
                            .retain(|(we, wf)| !(*we == e && *wf == fid));
                    } else {
                        self.event_waiters
                            .lock()
                            .retain(|(_, wf)| *wf != fid);
                    }
                    // 使用 frame.node_offset 而非 subgraph.node_range.0（同函数分支帧修正）
                    let caller_offset = NodeId(frame.node_offset);
                    // 遍历所有 completions，逐个回写返回值 + 信号传播 + 通知下游
                    for (call_node, return_value, child_signal) in completions {
                        let call_graph_id = NodeId(call_node.0 + caller_offset.0);
                        let consumer_count =
                            self.graph.downstream_slice(call_graph_id.0 as usize).len() as u16;
                        frame.set_value(call_node, return_value, consumer_count);
                        // Gate 分支子图的控制信号传播（与 complete_and_wake_caller 正常路径一致）
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
                    // 放回同一个 Box（地址不变）
                    self.frames.lock().insert(fid, frame_box);
                    queue.push(fid);
                } else {
                    // 检查 pending_events（事件到达时帧不在 HashMap 的竞态兜底）
                    let pending_evt = self.pending_events.lock().remove(&fid);
                    if let Some((_evt, evt_val)) = pending_evt {
                        // 有 pending event：注入事件值 + 唤醒
                        // waiter 已在 on_event_arrived 中移除，无需重复清理
                        if self.apply_event_to_frame(frame, evt_val) {
                            self.frames.lock().insert(fid, frame_box);
                            queue.push(fid);
                        } else {
                            // 帧非 WaitingEvent（状态不一致）：放回，不入队
                            self.frames.lock().insert(fid, frame_box);
                        }
                    } else {
                        self.frames.lock().insert(fid, frame_box);
                    }
                }
            }
            FrameState::Completed => {
                if has_caller {
                    // 区分 sync call vs async call 子帧完成
                    let async_id = self.async_join_runtime.lock().find_by_child(fid);
                    if let Some(async_id) = async_id {
                        // async 子帧完成：设置 result + 触发 AsyncJoin 事件
                        let return_value =
                            extract_child_return(frame, &self.graph);
                        self.async_join_runtime
                            .lock()
                            .set_result(async_id, return_value.clone());
                        // frame_box drop（不放回）
                        let woken = self.on_event_arrived(
                            RuntimeEvent::AsyncJoin(async_id),
                            return_value,
                            queue,
                        );
                        // waiter 已被唤醒（值已通过事件注入），entry 可安全清理。
                        // 若 woken == 0（无 waiter），entry 保留供 try_get_result 消费式读取
                        if woken > 0 {
                            self.async_join_runtime.lock().remove_entry(async_id);
                        }
                        // 回收 async 子帧到池
                        self.release_frame(frame_box);
                    } else {
                        // sync 子帧完成：清理 waiter + 回写 + 唤醒调用方
                        self.event_waiters.lock().retain(|(e, _)| {
                            !matches!(e, RuntimeEvent::SubgraphComplete(c) if *c == fid)
                        });
                        // 帧被消费：unbox 传给 complete_and_wake_caller
                        self.complete_and_wake_caller(*frame_box, queue);
                    }
                } else {
                    // 顶层帧完成：返回结果
                    let ret = extract_child_return(frame, &self.graph);
                    *self.result.lock() = Some(ret);
                    self.release_frame(frame_box);
                }
            }
            FrameState::Failed => {
                if has_caller {
                    // Failed 子帧（cancel 后）：清理 waiter + 唤醒调用方
                    self.event_waiters.lock().retain(|(e, _)| {
                        !matches!(e, RuntimeEvent::SubgraphComplete(c) if *c == fid)
                    });
                    self.complete_and_wake_caller(*frame_box, queue);
                } else {
                    // 顶层帧 Failed：返回 NULL
                    *self.result.lock() = Some(Value::NULL);
                    self.release_frame(frame_box);
                }
            }
            _ => {
                // Ready（控制信号触发但未挂起）：放回 + 重新入队
                self.frames.lock().insert(fid, frame_box);
                queue.push(fid);
            }
        }
    }
}
