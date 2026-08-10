//! 子图调用与返回：switch_subgraph + start_subgraph + complete_and_wake_caller。

use super::*;
use crate::ir::Ir::*;
use crate::ir::Ir::Frame;
use crate::value::Value;

/// 尾调用图跳转：复用当前帧执行目标子图（帧池零分配）。
pub fn switch_subgraph(frame: &mut Frame, graph: &DataFlowGraph, target_sg: SubGraphId, args: &[Value]) {
    let (node_start, node_end) = graph.subgraphs[target_sg.0 as usize].node_range;
    let node_count = (node_end.0 - node_start.0) as usize;

    // 更新 subgraph_id + 调整数组尺寸
    frame.subgraph_id = target_sg;
    if frame.value_table.len() != node_count {
        frame.value_table.resize(node_count);
    }
    if frame.pending_inputs.len() != node_count {
        frame.pending_inputs.resize(node_count, 0);
    }

    // 清空 value_table（prepare_frame_nodes 不做此操作）
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
    // caller 保持不变：返回值直达原始调用方的 call 节点

    // prepare_frame_nodes：设置 node_offset + pending_inputs + Const 预填充
    prepare_frame_nodes(frame, graph);

    // 参数注入
    let offset = node_start.0 as usize;
    let param_count = graph.subgraphs[target_sg.0 as usize].param_count as usize;
    for (i, arg) in args.iter().enumerate().take(param_count) {
        let local_id = NodeId(i as u32);
        let global_id = NodeId((offset + i) as u32);
        let consumer_count = graph.downstream_slice(offset + i).len() as u16;
        frame.set_value(local_id, arg.clone(), consumer_count);
        // 不 push_ready：参数值已由 set_value 设置，notify_downstream 传播给下游即可。
        // 若 push_ready，compute_const 会被调用并返回 VOID 覆盖参数值。
        notify_downstream(frame, graph, local_id, global_id, NodeId(node_start.0));
    }
}

// =========================================================================
// impl<S: LockStrategy> Engine<S> — 子图方法
// =========================================================================

impl<S: LockStrategy> Engine<S> {
    /// 启动子图：创建子帧 + 参数注入 + 绑定 caller。
    /// 同函数分支子图（if-else/match arm）：值表扩展到父函数大小，复制父帧值，
    /// 使分支节点可直接通过 get_value_by_global 访问外层变量（无需帧链指针）。
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
        // same_function 路径用于同函数内分支子图（if-else/match arm），这些子图
        // 的 node_range 严格包含在父函数 node_range 内，需要复制父帧值。
        // 递归调用自身（child_sg.id == parent subgraph_id）不应走此路径——
        // 它需要全新的调用帧，而非父帧值复制。直接递归走跨函数路径。
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
            // 同函数分支：值表扩展到父帧大小，复制父帧值。
            // 使用父帧的 node_offset/value_table.len() 而非 parent_sg.node_range，
            // 因为嵌套闭包帧的布局由祖父帧决定（如 outer 帧的 node_offset 是 main 的
            // node_start，而非 outer 子图的 node_range.0），使用 subgraph.node_range
            // 会导致值表索引错位、节点被误标记为 ready 从而跳过 compute_fn。
            let parent_start = parent_frame.node_offset;
            let parent_node_count = parent_frame.value_table.len();
            let (branch_start, _branch_end) = child_sg.node_range;
            let branch_param_count = child_sg.param_count as usize;

            let mut child = self.acquire_frame(child_fid, subgraph_id, parent_node_count);
            child.node_offset = parent_start;

            // 复制父帧已就绪的值（refcount 设 0 = 永不回收，帧结束时统一释放）
            // 跳过 child_sg 范围内的节点：递归调用时 child_sg 是函数体子图，
            // 分支内节点（如 n-1）的旧计算结果不能复制，否则子帧不会重新计算，
            // 导致递归参数不递减（fact(n-1) 反复传入相同的旧 n-1 值）。
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

            // 使用预计算的 nested_ranges（构建期填充），避免运行时全图扫描
            let nested_ranges: &[(u32, u32)] = self.graph.sg_nested_ranges(subgraph_id.0 as usize);
            let is_nested = |gid: u32| nested_ranges.iter().any(|&(s, e)| gid >= s && gid < e);

            // 设置 pending_inputs：分支节点按实际未就绪输入计数，非分支节点标记 EXTERNAL
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
                    // Gate（非 select）和普通节点统一：按实际 in-frame 未就绪输入计数
                    // 帧范围外的输入（effect 链、外层变量）通过帧链穿透访问，不计为 pending
                    let inputs = self.graph.inputs(node.inputs_offset, node.input_count);
                    let mut pending = 0u16;
                    for &inp in inputs {
                        let il = inp.0.wrapping_sub(parent_start) as usize;
                        if il < parent_node_count {
                            let inp_gid = (parent_start as usize + il) as u32;
                            let inp_in_branch = inp_gid >= branch_start.0 && inp_gid < child_sg.node_range.1 .0;
                            // 分支内节点：未就绪则计入 pending
                            // 外层变量/effect（!in_branch）：通过帧链穿透访问，不计 pending
                            if inp_in_branch && !child.value_table.is_ready(il) {
                                pending += 1;
                            }
                        }
                        // 帧范围外（il >= parent_node_count 或下溢）→ 帧链穿透，不计 pending
                    }
                    child.pending_inputs[i] = pending;
                }
            }

            // 分支内 0-input 非 Param 节点入队（必须在参数注入之前！）
            // 顺序原因：若参数注入在前，notify_downstream 会使下游节点 pending 归零并入队；
            // 随后 0-input 入队又检查到 pending==0 && !ready 再次入队，导致节点被执行两次
            // （如 for-in 的 next_call 被执行两次，消耗两个迭代器元素，跳过首个元素）。
            // 将 0-input 入队放在参数注入之前：此时下游节点 pending 仍 >0 不会被入队，
            // 仅 0-input 常量节点入队；参数注入的 notify_downstream 随后将下游入队一次。
            // 此顺序与跨函数路径（prepare_frame_nodes 先于参数注入）一致。
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

            // 参数注入（local 索引 = branch_start - parent_start + i）
            // 实际参数注入调用方传入的 arg 值；
            // upvalue 参数注入当前父帧值（引用捕获语义），使 same_function 调用
            // 能看到外层变量的最新值（而非闭包构造时的快照）。
            let param_local_offset = branch_start.0.wrapping_sub(parent_start) as usize;
            let actual_param_count = branch_param_count
                .saturating_sub(child_sg.upvalue_count as usize);
            // 实际参数
            for (i, arg) in args.iter().enumerate().take(actual_param_count) {
                let lid = NodeId((param_local_offset + i) as u32);
                let gid = branch_start.0 as usize + i;
                let global_id = NodeId(gid as u32);
                let cc = self.graph.downstream_slice(gid).len() as u16;
                child.set_value(lid, arg.clone(), cc);
                // 不 push_ready：参数值已设置，notify_downstream 传播给下游
                notify_downstream(&mut child, &self.graph, lid, global_id, NodeId(parent_start));
            }
            // upvalue 参数注入：从父帧读取最新值（引用捕获语义），使 same_function
            // 调用能看到外层变量的最新值（而非闭包构造时的快照）。
            // 递归闭包例外：self_upvalue_idx 对应的 slot 注入闭包自身引用，
            // 而非父帧值（父帧中 self slot 是 void_const 占位）。
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
                // 不 push_ready：参数值已设置，notify_downstream 传播给下游
                notify_downstream(&mut child, &self.graph, lid, global_id, NodeId(parent_start));
            }

            child.caller = Some((caller_fid, call_node));

            // 帧链指针在 process_frame 的 setup_frame_chain 中设置
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
            // 跨函数调用：原有逻辑
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
                // 不 push_ready：参数值已设置，notify_downstream 传播给下游
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

    /// 子图完成后：回写返回值到调用方 + 唤醒调用方。
    /// 含 LoopBody 完成检测 + pending_completions 竞态处理。
    /// 使用迭代式处理 LoopBody break/return 传播，避免深度嵌套循环的栈溢出。
    pub(super) fn complete_and_wake_caller(&self, mut child_frame: Frame, queue: &QueueHandle<'_>) {
        // LoopBody break/return 传播循环（迭代式，替代递归）
        loop {
            let child_sg_id = child_frame.subgraph_id;
            let child_loop_kind = self.graph.subgraphs[child_sg_id.0 as usize].loop_kind;
            if child_loop_kind != crate::ir::Ir::LoopKind::LoopBody {
                break; // 非 LoopBody，进入正常完成路径
            }
            let child_signal = child_frame.control_signal.clone();
            let (loop_fid, _call_node) = child_frame
                .caller
                .expect("LoopBody frame missing caller");
            match child_signal {
                ControlSignal::Break | ControlSignal::Return(_) => {
                    // break/return → 循环退出
                    let mut loop_frame = self.frames.lock().remove(&loop_fid);
                    if let Some(lf) = loop_frame.as_deref_mut() {
                        lf.cached_child_frame = None;
                        lf.control_signal = child_signal;
                    }
                    // 迭代处理 loop_frame（loop_kind 通常是 While/Loop/For，非 LoopBody，
                    // 但若为嵌套 LoopBody 则继续迭代传播，避免递归栈溢出）
                    match loop_frame {
                        Some(lf) => {
                            child_frame = *lf; // 迭代而非递归
                            continue;
                        }
                        None => panic!(
                            "complete_and_wake_caller: LoopBody break/return 但 loop_frame {:?} 不在 frames（不变量违反：body 帧的 caller 引用的 loop 帧必须存在）",
                            loop_fid
                        ),
                    }
                }
                ControlSignal::Continue => {
                    // continue → 循环重置（帧复用）
                    let mut loop_frame = self.frames.lock().remove(&loop_fid).unwrap_or_else(|| {
                        panic!(
                            "complete_and_wake_caller: LoopBody continue 但 loop_frame {:?} 不在 frames（不变量违反：body 帧的 caller 引用的 loop 帧必须存在）",
                            loop_fid
                        )
                    });
                    let mut child = child_frame; // 取得所有权以便修改
                    self.reset_loop_iteration(&mut *loop_frame, loop_fid, &mut child);
                    self.frames.lock().insert(loop_fid, loop_frame);
                    queue.push(loop_fid);
                    let body_id = child.id;
                    self.frames.lock().insert(body_id, Box::new(child));
                    return;
                }
                ControlSignal::None => {
                    // 正常完成：检查 caller 循环类型
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
                    let mut loop_frame = self.frames.lock().remove(&loop_fid).unwrap_or_else(|| {
                        panic!(
                            "complete_and_wake_caller: LoopBody none 但 loop_frame {:?} 不在 frames（不变量违反：body 帧的 caller 引用的 loop 帧必须存在）",
                            loop_fid
                        )
                    });
                    let loop_kind = self.graph.subgraphs[loop_frame.subgraph_id.0 as usize].loop_kind;
                    if loop_kind == crate::ir::Ir::LoopKind::TailRec {
                        // TailRec 循环：body_sg 无信号完成 = base case 命中。
                        // 提取 body_sg 返回值，转换为 Return 信号让循环退出。
                        let return_value = super::Schedule::extract_child_return(&child_frame, &self.graph);
                        loop_frame.cached_child_frame = None;
                        loop_frame.control_signal = ControlSignal::Return(return_value);
                        child_frame = *loop_frame;
                        continue;
                    } else {
                        // 普通循环（While/Loop/For）：正常完成 → 循环重置（帧复用）
                        let mut child = child_frame;
                        self.reset_loop_iteration(&mut *loop_frame, loop_fid, &mut child);
                        self.frames.lock().insert(loop_fid, loop_frame);
                        queue.push(loop_fid);
                        let body_id = child.id;
                        self.frames.lock().insert(body_id, Box::new(child));
                        return;
                    }
                }
            }
        }

        // 非 LoopBody：回写返回值 + 唵醒 caller（含 pending_completions 竞态处理）
        let child_sg_id = child_frame.subgraph_id;
        let return_value = super::Schedule::extract_child_return(&child_frame, &self.graph);
        let child_signal = child_frame.control_signal.clone();
        let caller = child_frame.caller;
        // 回收子帧到池（Vec 容量保留供复用）
        self.release_frame(Box::new(child_frame));

        if let Some((caller_fid, call_node)) = caller {
            let mut caller_frame_opt = self.frames.lock().remove(&caller_fid);
            if caller_frame_opt.is_none() {
                // 父帧尚未 insert 回 HashMap，存储完成信息等待重试。
                // 使用 Vec 避免同一 caller 多个子帧并发完成时互相覆盖。
                self.pending_completions
                    .lock()
                    .entry(caller_fid)
                    .or_insert_with(Vec::new)
                    .push((call_node, return_value, child_signal));
                return;
            }
            if let Some(caller_frame) = caller_frame_opt.as_deref_mut() {
                // 使用 caller_frame.node_offset 而非 subgraph.node_range.0：
                // 同函数分支帧的 node_offset 是父函数的 node_start，
                // 而 subgraph.node_range.0 是分支子图的 node_start，两者不同。
                // 用错会导致 call_graph_id 偏移错误 → notify_downstream 找不到下游
                // → 下游节点 ready 标记永远不被设置 → 帧挂起。
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

                // 控制信号传播：子帧的 throw/return/break/continue 信号传播给调用方帧。
                // 仅在同函数内传播（Gate 分支子图、循环子图）：
                // - if-else/match arm（Gate 节点）内 throw/return/break/continue → 传播给父帧
                //   （break/continue 需穿透到 LoopBody 帧，否则循环体内的 if-break 无效）
                // - while/loop/for（循环帧）内 throw/return → 传播给函数帧
                // 不传播的情况：
                // - 跨函数调用：函数帧的 Return 信号是函数级返回，返回值已通过
                //   extract_child_return 提取，传播会导致调用方帧错误提前退出
                // - Lambda/嵌套函数调用（Call 节点 + loop_kind==None + 同 function_id）：
                //   虽然与调用方共享 function_id（为帧链穿透），但它是独立函数调用，
                //   返回值已提取，传播 Return 会导致调用方帧错误退出（静默退出 bug）
                // - 循环帧的 Break/Continue：已被循环消费，传播会导致函数错误退出
                let child_loop_kind = self.graph.subgraphs[child_sg_id.0 as usize].loop_kind;
                let is_gate = self.graph.node(call_graph_id.0 as usize).kind
                    == crate::ir::Ir::NodeKind::Gate;
                let should_propagate = match child_signal {
                    ControlSignal::Return(_) => {
                        // Return：Gate 分支 + 循环帧传播；Lambda/函数调用不传播
                        is_gate || child_loop_kind != crate::ir::Ir::LoopKind::None
                    }
                    ControlSignal::Break | ControlSignal::Continue => {
                        // Break/Continue：仅 Gate 分支传播（穿透到 LoopBody）
                        // 循环帧的 Break/Continue 已被循环消费
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
        // 子帧已完成，由调用方负责 drop（不放回 frames）
    }
}
