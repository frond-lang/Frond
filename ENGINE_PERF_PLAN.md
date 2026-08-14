# Kuzo 引擎执行效率优化方案（E0-E5）

> 2026-08-15 · 基于引擎全链路勘探（engine/ ir/Compute/ value/ solidify/）+ release 基线实测
> 前置：IR_OPTIMIZATION_PLAN.md P0-P4 已收官（正确性优化）。本轮纯执行层，不动 pass / 序列化 schema / FFI。
> 决策：范围含 E5 线性化；不用 git，每阶段前 zip 备份到 `.backups/`。

## 0. 现状诊断

### 0.1 Release 基线（2026-08-15，本机，5 次取中位）

| 基准 | 耗时 | 折算 |
|---|---|---|
| loop_sum (i32 1M iter) | ~1457 ms | ~1.5 µs/迭代 |
| loop_sum (i64 1M iter) | ~1636 ms | ~1.6 µs/迭代 |
| fib 迭代版 (100k iter) | ~131 ms | ~1.3 µs/迭代 |
| match_dispatch (1M iter) | ~10304 ms | ~10.3 µs/迭代 |
| recursion_tco countDown/sumTo (1M iter) | ~3946 / ~3906 ms | ~3.9 µs/迭代 |
| string_build (10k append) | ~21 / ~11 ms | ~2 µs/append |

热路径开销是逐节点调度与帧往返，不是计算本身。

### 0.2 执行模型

数据流就绪调度（pending_inputs 倒计时 + ready_queue FIFO）+ 帧队列。**同步调用和每轮循环迭代都要在全局队列往返**：caller 挂起 → child 入队 → child process_frame → complete_and_wake_caller → caller 重新入队 → process_frame。

### 0.3 逐项热点清单（文件:行号，2026-08-15 时点）

1. **每循环迭代 2 次 `std::env::var` 全环境扫描**：Schedule.rs:363（KUZO_NO_REUSECHAIN）、Subgraph.rs:423（KUZO_DEBUG_FORIN）；每次调用完成再 +1（Subgraph.rs:522 KUZO_DEBUG_IFELSE）。Windows 单次数百 ns。
2. **每节点执行 Arc 原子增减×2**：所有 compute_fn 开头 `frame.graph.clone()`（如 Compute.rs:2229/2359/492）；EvalContext 只带 node_start（Ir.rs:1122）。
3. **每循环迭代克隆整个 ResetPlan**（4 个 Vec 堆分配）：Frame.rs:306。
4. **每次 Gate 执行深克隆 GateBranches**（≥2 Vec）：Accessors.rs:467-494（mmap 路径逐次现场解析）+ Compute.rs:2382 inputs.clone()。
5. **每帧完成克隆 defer_table**：Schedule.rs:656/677。
6. **字符串常量每帧重新物化**（2 次堆分配/次）：compute_const（Compute.rs:3252）→ alloc_const_value（Schedule.rs:40-46）。
7. **每迭代帧机制税**：~10 次锁 + frames HashMap remove/insert ×4 + setup_frame_chain 链走（Frame.rs:565）+ Box::new 重新装箱（Subgraph.rs:402/458）+ copy_outer_ready_values 全量 Value 克隆（Frame.rs:17）+ prepare_same_function_frame O(节点×nested_ranges)（Frame.rs:35）。
8. **每次同步调用**：args Vec 分配（Compute.rs:2254）+ event_waiters push/retain + 两轮 process_frame。

## 1. 设计原则

1. **语义零变更**：控制信号传播矩阵、defer 语义、帧链指针规则（Bug #78/#100）、ResetPlan 领地（F-6①）原样保留——只改"由谁驱动、经不经队列"。
2. **同步即内联**：`!has_suspend` 的子图调用在当前线程当前栈内执行至完成；队列只服务真正的并发（async/await/channel/timer）。
3. **派生数据一处派生**：新增图派生数据（线性计划、pending 模板、嵌套标志）一律 build 末尾 + load 后重算、optimizer rebuild 后清空重算（F-7 铁律），不进 .kzo schema。
4. **每阶段有门禁**：functional 门禁（48 过 / 5 存量豁免 / 0 新失败）+ negative + KUZO_VERIFY 验证器 + perf 红线（任一基准回退 >10% 即停）。

## 2. 阶段详设

### Phase 0 — 度量与安全网（S）✅ 2026-08-15
- 备份惯例：`tar -a -cf .backups/pre_phaseN.zip Kuzo/src Kuzo/Cargo.toml Kuzo/build.rs`。
- `tests/scripts/run_functional.sh`（53 项目，RESULT: ALL PASSED 断言，存量失败豁免清单）；`tests/scripts/run_perf.sh`（5 项目 ×N 取中位）。
- 基线见 0.1。functional 门禁基线：48 PASS / 5 known-fail（edge_ffi_inline=FFI重塑期编译错误；enum_u8_bug/edge_nested_types/str_writeback_bug/edge_tailrec=旧式无 RESULT 行）。

### Phase 1 — E0 快赢：消灭固定税（M，零语义变更）
1. engine 侧 env_flag 全量缓存（EngineCore.rs:23 改 OnceLock<HashMap>，收编 5 处裸 env::var）。
2. `EvalContext<'a> { node_start, graph: &'a DataFlowGraph }`——ComputeFn 签名不变，compute_fn 体改用 `ctx.graph`，消灭每节点 Arc 原子操作。
3. reset_loop_iteration 借用 reset_plan（删每迭代 4 Vec clone）。
4. defer_table clone→借用（Schedule.rs 两处）。
5. gate_branches_at mmap 路径 load 一次性物化；Gate 执行点借用式访问。
6. 字符串常量 graph 级一次性物化缓存。
- 预期：loop_sum -20~40%，match_dispatch -10~20%。

### Phase 2 — E1 同步调用内联（L）
- complete_and_wake_caller（Subgraph.rs:329）完成逻辑重构为可复用纯函数（LoopBody 信号处理/TailRec 基例/返回值提取/capture-gate 判定/传播矩阵）。
- run_frame_nodes 的 NodeResult::Call（非 async、非 tail-call、非 LoopBody）：child 帧本地迭代执行至 Completed/Failed——不进 frames map、不经队列、caller 不挂起；帧链指针从 in-hand parent 直连。
- 回退：child 挂起 → 按现队列路径补登记；内联深度 >256 → 回退队列模式（队列模式无原生递归）。
- Multi 同样内联（同步本就串行）；async/spawn 不动；内联帧无并发完成 → 竞态路径不触发。
- 预期：match_dispatch -50%+。

### Phase 3 — E2 循环热路径（M-L）
- LoopBody 完成的 Continue/None 分支：reset_loop_iteration 后不 requeue，直接在当前 run_frame_nodes 内继续驱动 loop 帧（cond/Gate 重估 → 下一轮 body 走 Phase 2 内联）。
- 准入：body sg 无 has_suspend；loop 帧挂起在非 SubgraphComplete 事件即回退队列。Bug G defer 排空、TailRec 语义经 Phase 2 纯函数等价保留。
- 预期：循环类基准在 Phase 1 基础上再 -50%+。

### Phase 4 — E3 帧初始化瘦身（M）
- nested 判定：nested_ranges 线性扫描 → per-node 嵌套标志字节（派生）。
- 跨函数帧初始 pending_inputs 模板 per-sg 预计算（load 后重算），prepare_frame_nodes 退化为 memcpy。
- copy_outer_ready_values → 仅复制 body 实际引用的 outer 节点集（per-sg outer_read_set 派生）。

### Phase 5 — E4 微优化收尾（S-M）
- downstream_counts: Vec<u16> 平铺（替代每 set_value 的 downstream_slice().len()）。
- PendingCall.args 小数组内联；complete_and_wake_caller 改传 Box<Frame> 免每迭代 re-box。

### Phase 6 — E5 直线区线性化（L+）
- E5a：`SubGraph.linear_plan: Option<Vec<NodeId>>`（sg 内排除嵌套 sg 节点/EventSource/挂起点则 None）；build 末尾 + load 后 + rebuild 后计算（F-7）；Verifier V7（计划节点恰出现一次、生产者先于消费者）。
- E5b：`run_linear`：按 plan 顺序 compute_fn + set_value，免 pending/ready_queue/notify_downstream/refcount；每节点后查 control_signal；遇 Call 交 Phase 2 内联（async→bail）；遇 Await/Select → 用 ready 位图重建剩余 pending 回退队列模式。
- E5c：循环 sg 每轮重置退化为"ResetPlan 驱动清 ready 位 + 重跑 plan"。
- SIMD batch 保留给队列模式；线性区天然消解其调度开销。
- 预期：loop_sum ~0.1-0.3 µs/迭代，match_dispatch ~1-2 µs/迭代。

## 3. 路线图

| 阶段 | 内容 | 规模 | 门禁重点 |
|---|---|---|---|
| P0 | 度量+备份+脚本 | S | 基线表可复现 |
| P1 | E0 固定税 | M | 全套件；loop -20%↑ |
| P2 | E1 调用内联 | L | edge_closures/recursion/inline_capture；match -50%↑ |
| P3 | E2 循环热路径 | M-L | edge_loop/loop_nesting/control_flow/defer 套件 |
| P4 | E3 帧初始化 | M | records/traits 宽函数场景 |
| P5 | E4 微优化 | S-M | perf 无回退 |
| P6 | E5 线性化 | L+ | 全套件 + verifier V7 + async 混合循环 |

## 4. 风险与对策

- **F-7**：新派生数据 rebuild 后失效 → rebuild 清空 + Pipeline/load 双保险重算。
- **F-6①（循环重置领地）**：不移动节点，ResetPlan 语义与成员原样，只改驱动方式。
- **帧链指针（#78/#100）**：内联帧 parent_frame_ptr 指向 in-hand 帧（Box 地址稳定性论证照旧）；内联帧不进 frames map，无并发完成。
- **深递归栈溢出**：内联深度守卫 256 + 队列模式兜底。
- **Multi 语义**：同步内联不改变可观测并发行为；edge_async/edge_channels 全程作门禁。

## 5. 明确不做

不换 CFG+SSA；不动 IR 序列化 schema（派生数据加载后重算）；不动 pass 层与 FFI 线；Multi 语义不变；不引入新外部依赖。

## 6. 实施记录

- **2026-08-15 P0 ✅**：备份 pre_phase1.zip；runner 脚本就位；functional 门禁基线 48/0/5；perf 基线见 0.1。
- **2026-08-15 P1 ✅**（E0 固定税全部落地）：
  - env_flag 全量缓存（EngineCore.rs OnceLock<HashMap>，收编 5 处裸 env::var）
  - EvalContext<'a> 携带 &DataFlowGraph；wrap_fn/121 个内层 compute_fn/直接注册 fns 全部改 ctx.graph，消灭每节点 Arc 原子增减
  - reset_loop_iteration 借用 ResetPlan（删每迭代 4 Vec clone）
  - defer_table clone→借用（×2）
  - gate_branches_at 改借用返回 + 两条 loader 末尾 materialize_gate_branches() 一次性物化；compute_gate_launch 免 GateBranches 深克隆与 branch_inputs.clone()
  - const_cache：EngineRef::new 咨询点一次性物化全部常量（字符串常量全程共享一个 Arc）；compute_const 优先读缓存
  - 门禁：functional 48/0/5 全绿；perf（5 次中位）：fib 133→78ms(-41%)、loop_sum 1447→1013/1619→1124(-30%)、match_dispatch 10336→7712(-25%)、recursion_tco 3949→3357/3937→3333(-15%)、string_build 20→15/11→8(-27%)。零回退。
- **2026-08-15 P2 ✅**（E1 同步调用内联）：
  - start_subgraph 拆出 start_subgraph_frame 核心（不进 frames map；same_function 帧链直连 in-hand parent，Bug #100 布线），queue 包装器保持置空+setup_frame_chain 旧约
  - run_frame_nodes 增 depth 参数（INLINE_MAX_DEPTH=256，超出回退队列模式=无原生递归上限）；NodeResult::Call 非 async 非 LoopBody 非 tail-call → 子帧当前栈执行至 Completed/Failed，finish_call_in_caller（complete_and_wake_caller 尾段纯函数化：返回值/capture-gate/传播矩阵/notify）写回后继续执行 caller；子帧挂起→精确复刻旧挂起协议（入 map+入队+event_waiters+caller Suspended，Bug #78 pending_completions 竞态解法原样适用）
  - Multi 模式同样内联（同步本就串行）；async/spawn/LoopBody 路径不变
  - 门禁：functional 48/0/5（3 轮稳定）+ negative 22/0；perf：match_dispatch 7712→6647(-14%)、recursion_tco ~-6%、其余持平零回退。调用路径开销已消除，剩余大头是 LoopBody 循环帧机制（Phase 3 目标）
- **2026-08-15 P3 ✅**（E2 循环热路径）：
  - Frame 增 hot_body: Option<(FrameId, Box<Frame>)> 携带槽；LoopBody 调用（body sg 无挂起点且深度余量）在当前栈逐轮驱动：参数注入+链布线 → run_frame_nodes(body) → Break/Return 退出（含 Bug G defer 排空）/TailRec 基例/Continue/None→reset_loop_iteration → hot_body 携带 → continue 外层循环（cond/Gate 重估）。子体挂起→精确退回队列协议（入 map+cached_child_frame+event_waiters）；nested 循环递归生效
  - **教训（新）**：run_frame_nodes 的 iter_guard>500000 活锁保护与热路径冲突——整个循环跑在一次调用内，1M 迭代×每轮多次弹出超限被误标 Failed（edge_recursion/mixed_types/edge_async 三测挂）；修复=热路径每轮 continue 时 iter_guard=0（一轮 body 完成=确凿进展）
  - 门禁：functional 48/0/5 全绿；perf：fib 82→59ms（累计-56%）、loop_sum 1006/1125→800/933（累计-45%/-42%）、match_dispatch 6647→6511（累计-37%）、recursion_tco →2926/2899（累计-26%）、string_build →13/7（累计-35%/-36%）
- **2026-08-15 P4 ✅**（E3 帧初始化瘦身）：
  - 图级 sg_initial_pending/sg_initial_seed 模板（EngineRef::new 预计算，跨函数帧 prepare_frame_nodes 退化为 memcpy+seed 重放；LSP/同步解释器路径回退旧派生）
  - Frame.same_fn_prep_cache 稳态缓存：same_function 帧的 pending/seed 派生是 (父ready位图, graph, sg) 的纯函数——位图指纹匹配即 memcpy 复用；循环稳态每轮命中（清缺点：acquire_frame 复用/switch_subgraph 换 sg/长度失配自然失效）
  - outer_read_set 未做（copy 的 scalar clone 24B 占比小，先测模板收益）
  - 门禁：functional 48/0/5 全绿；perf：fib 59→52（累计-61%）、loop_sum 800/933→642/756（累计-56%/-53%）、match_dispatch 6511→5467（累计-47%）、recursion_tco →2882/2851（累计-27%）、string_build →11/6（累计-45%）
- **2026-08-15 P5 ✅**（E4 微优化，做精华项）：
  - downstream_counts: Vec<u16> 平铺（EngineRef::new 物化），accessor downstream_count() 带 CSR 回退；20 处 `downstream_slice(x).len() as u16` 全部替换
  - PendingCall.args 小数组内联与 Box 复用未做：E2 后 complete_and_wake_caller 已不在热循环路径；args 分配留待 E5 后按需评估
  - 门禁：functional 48/0/5 全绿；perf 与 P4 持平（match 5366/-2%、tco 2802/-3%，微收益）
- **2026-08-15 P6 ✅**（E5 直线区线性化）：
  - 图级 linear_plans: Vec<Option<Vec<NodeId>>>（EngineRef::new 物化，Kahn 拓扑序）+ DataFlowGraph::linear_plan() 访问器；**仅完全无 launch 节点（Gate/Call/Await/EventSource）的 sg 生成计划**，EventSource/环 → None
  - Frame.linear_fresh 一次性标志（prepare_frame/switch_subgraph/same_function 创建/reset_loop_iteration 置位；dispatch 消费后清零）；run_frame_dispatch 入口按 plan 分流
  - run_linear：按 plan 顺序执行 compute_fn + set_value，**免 pending 倒计时/ready 队列/notify_downstream/refcount**（值存活到帧结束=帧级回退语义）；ready 槽跳过（参数/注入槽，防 compute 覆写）；Return/Break/Continue 信号短路；意外引擎型结果 → rebuild_linear_bailout（从 ready 位图重建 pending+seed，语义对齐 prepare 族）+ run_frame_nodes 续跑（安全网）
  - 循环体 reset：body 有 plan 时跳过 prepare_same_function_frame（线性不需要 readiness，bailout 按需重建），reset_loop_iteration 直达线性态
  - **教训（新）**：首版"计划含 launch 节点+前缀线性+bail"使 match_dispatch 回退 +13%（每迭代 bail+重建比纯数据流贵；rebuild 的 nested 查找误放循环内 O(n×nested)）——改为完全线性 sg 专用 + nested 外提后消除
  - **教训（新）**：debug 构建（未优化栈帧每层数 KB）下 INLINE_MAX_DEPTH=256 内联嵌套溢出 1MB Windows 主线程栈（edge_recursion 栈溢出，release 通过）——cfg!(debug_assertions) 时上限降为 48（超出回退队列模式，无原生递归）；后续若引入更深内联，考虑大栈线程承载引擎
  - 门禁：functional 48/0/5 全绿 + negative 22/0 + debug 构建 KUZO_VERIFY 全套件；perf 终表见下

### 终态性能表（release，5 次中位，2026-08-15）

| 基准 | 基线 | E0 | +E1 | +E2 | +E3 | +E4 | +E5 | 总提升 |
|---|---|---|---|---|---|---|---|---|
| fib 迭代 (100k) | 133ms | 78 | 82 | 59 | 52 | 54 | **39** | **-71%** |
| loop_sum i32 (1M) | 1447ms | 1013 | 1006 | 800 | 642 | 651 | **562** | **-61%** |
| loop_sum i64 (1M) | 1619ms | 1124 | 1125 | 933 | 756 | 761 | **661** | **-59%** |
| match_dispatch (1M) | 10336ms | 7712 | 6647 | 6511 | 5467 | 5366 | **5437** | **-47%** |
| recursion_tco (1M) | 3949ms | 3357 | 3145 | 2926 | 2882 | 2802 | **2809** | **-29%** |
| string_build (10k) | 20/11ms | 15/8 | 15/8 | 13/7 | 11/6 | 11/6 | **10/5** | **-50%/-55%** |

折算：loop_sum ~1.45µs→~0.56µs/迭代；match_dispatch ~10.3µs→~5.4µs/迭代；fib ~1.3µs→~0.39µs/迭代。
