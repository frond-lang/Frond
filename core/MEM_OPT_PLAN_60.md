# 对象内存 -60% 方案讨论 (2026-09-03)

目标:两字段记录实测驻留从 ~279B/对象 砍到 **≤112B(-60%)**,口径=`tmp_probe/memcmp` live1M 基准(峰值增量/对象数)。

## 实测构成(诊断已做)

fill-only 底盘 30.7MB → 每记录真实驻留 ≈247B(全测试口径 279B):

| 成分 | 字节 | 说明 |
|---|---|---|
| 数组槽 Value | 24 | |
| Arc<HeapObj> 壳 | 128 | 120 实际 + mimalloc 取整;**其中 72B 是枚举最大变体(TraitValue 104B)的 padding 纯浪费** |
| fields Vec | 48 | 2×24B Value,独立堆块 |
| Registry 表 | ~10 | FxHashSet 8B 键/对象+装填 |
| 瞬峰/留存杂项 | ~30-40 | 数组倍增瞬峰、分配器留存 |

## 方案阶梯(叠加式,预测基于上表)

| 方案 | 每对象 | 降幅 | 工程量 | 核心风险 |
|---|---|---|---|---|
| 甲 单块 DST 分配 | ~135B | -52% | 中高 | 手写 Arc 单块布局(unsafe 集中在 alloc/drop 一处) |
| **乙 甲+字段按 shape 打包** | **~97B** | **-65% ✓** | 中高+ | 读写装卸路径 |
| 乙 + Value 16B(P2-6) | ~85B | -70% | +大面积机械改 | 触面广但机械 |
| 丙 全内联小记录 | ~28B | -90% | 高 | `&rec.x` 活性 → 取址时装箱(escape) |

### 甲:单块 DST 分配(Header-Vec 模式)

一次分配 `[Arc 计数 16][HeapObj tag][shape ptr][arity][内联 fields × arity]`,fields 不再是独立 Vec。`RecordValue.fields` 从 `Vec<Value>` 改为 len+非拥有尾指针,访问器 `field(i)`。Arc clone/Drop 天然复用;`Arc::make_mut` 写时复制=整块复制,与现有语义等价;registry/深克隆/相等/哈希全走访问器;solidify 不涉及运行时值。**消灭 128B 壳里的 72B padding 与 48B Vec 块**。

### 乙:shape 驱动字段打包

`RecordShape` 增加 per-field 存储 kind(i8/i32/f64/.../通用 Value);纯标量字段按原生宽度进尾区(shape 启动时算好偏移表),`field(i)` 读=构造 Value,写=窄位直写;含引用字段退回 24B 槽。业务记录(Point/Size/Config 类)命中率预计高。块体 72→~40B(mimalloc class 48)。

### 丙:内联小记录(二期)

≤2 标量字段直接进 Value 的 16B 载荷(shape_id+12B 数据),ValueTag 加 InlineRecord(不进 arena 无 handle 冲突)。值语义与内联天然契合;`&rec.x` 活性由取址时装箱为乙形态解决。记录数组槽自动获得 SoA 效果。

## 叠加项(可选)

- **Registry 触发式重建**:出生不登记,压力阀触发时从根集全量重建注册表——表成本 O(分配)→O(存活),顺带解决阀门覆盖≈0(阀门改造单独立案)。
- P2-6 Value 16B:所有 `Vec<Value>` -33%,含记录数组槽。

## 推荐路线

1. **先甲后乙**(同 PR 链):甲验证单块机制(-52%),乙上打包达标(-65%);乙预留丙的升级路径。
2. P2-6 独立机械批随后(-70% 合计)。
3. 丙二期冲 -90%。

## 验证口径

- `tmp_probe/memcmp` live1M:<112B/对象(硬指标);
- 功能 95+负向 64+perf 门禁全套;
- 六语料 diff 不受影响(运行时表示不进序列化);std/ 不动。

---

# 附录:现有依赖的组合利用(2026-09-03 补)

零新增依赖,三个现成构件直接进方案:

## 1. bumpalo(已是依赖,collections 特性已开)→ 帧局部对象层

代码里已留好挂点:`compute_record_construct_stack` / `compute_array_construct_stack`(Compute.rs,分析器标记"非逃逸"分配点专用),其文档注释明言"为将来真正的栈/帧局部分配保留分离点"。组合:

- Frame 挂 `bumpalo::Bump`(帧池复用,帧终整场 reset,O(1) 回收);
- 非逃逸构造(静态标记)→ 竞技场单块,**无 Arc、无 registry、分配=指针步进**;
- 表示:Value 新增非拥有引用形态(帧指针),单帧所有权纪律与现有 Box<Frame> 跨线程契约一致;共享/就地改走 COW(复制一块 ~40B,热循环内本就少见);
- 任何共享/逃逸升级 → 走乙的单块 Arc 形态(升级=深拷一次)。

该层对象成本 ≈ 纯载荷(2×i32 打包后 ~8B,通用 2×24B)。

## 2. zerocopy(已是依赖,derive 已开)→ 乙方案打包字段的安全视图

字段打包区的读写不再手写 transmute:`FromBytes` 视图按 shape 偏移取 `&i32/&f64`,把乙方案里最危险的位级装卸变成编译期校验的安全转换。

## 3. rustc-hash(已在用)→ registry 微贡献;parking_lot → 锁成本微降(可选)

## 合并后的三层对象模型与预测

| 层 | 对象 | 每对象 | 覆盖 |
|---|---|---|---|
| 0 帧局部(bumpalo) | 分析器证明非逃逸的构造 | **~8-48B**(-83~97%) | 热循环内临时记录 |
| 1 逃逸(乙:单块+shape打包) | 其余记录/ADT | **~97B**(-65%) | 容器/长命/跨帧 |
| 2 内联(丙,二期) | ≤2 标量字段 | ~28B | 终极形态 |

仅第 1 层已独立达标 -65%;第 0 层按工作负载逃逸比再压。mimalloc 继续服务第 1 层与通用路径,第 0 层完全绕过分配器。

---

# 甲落地结果(2026-09-03)

**甲(单块 DST 分配)已合并并全量验证。**

- 新架构:`Value::Record(RecordRef)`——手写引用计数瘦指针,单块 `[strong][shape Arc][field_count][内联字段尾]`,一次分配;`HeapObj::Record/Adt/Newtype` 三变体与 `RecordValue/AdtValue/NewtypeValue` 三结构体删除;`ShapeKind` 保住 Record/Adt/Newtype 可观察 kind;环收集器改造为混合堆(记录块指针低位打标,垫层/释放路径分列);就地改走 `set_field` 单写者纪律。
- **实测:live1M 每对象 279B → 165B(-41%),峰值 277.4 → 176.0 MB**;功能 95+负向 64 全绿(2 环境挂与既档同);churn 校验和/耗时持平。
- 分配次数:每记录 2 次(壳+Vec)→ **1 次**。

## ★关键诊断(乙的靶子重排)

`FROND_NO_CYCLES=1` A/B:live1M 峰值 176 → **107.4 MB**——**环注册表自身成本 ≈69B/对象(占 42%)**,来源为 FxHashSet 键存储+rehash 翻倍瞬峰。NO_CYCLES 下每对象 ≈96B,**已等于 -66%**。

乙的执行序因此调整为:
1. **乙① 注册表结构瘦身**(收益 ~69B/对象):候选=链式哈希集(无 rehash 瞬峰)/分片固定容量/触发式重建(出生不登记,阀门触发时从根集重建——顺带修阀门覆盖≈0)。注意 NO_CYCLES 牺牲环回收,不可作为默认。
2. **乙② 字段打包**(收益 ~32B/对象):shape 增 per-field 打包 tag+偏移表;RecordLitInfo 需从 Builder/sema 拿构造器字段静态类型(ModuleEnv 的 `field_type_reprs` 已有数据源);标量字段按原生宽进打包区,其余走 Value 槽。
3. 全落齐后预期 ~65-90B/对象(-68~75%)。

---

# 乙① 落地结果(2026-09-03)

**侵入式链表登记 + 阀门增长判据已合并。**

- 记录块头自带 `reg_next/reg_prev`(块 32→48B 头,整块 80→96B),登记/注销=锁内两次指针写;哈希表对记录彻底退役;`registered_count`=哈希(HeapObj 种类)+链表计数(原子 O(1));收集器记录枚举改走 `record_list_walk`(打标指针复用混合堆约定)。
- 压力阀加**增长性判据**(>64K 且较上次净增>32K 才开火):真泄漏单调增长仍被逮;稳定大存活集不再全量标记。live1M(fill 共享→逐个独立化,天然单调增长)仍会开火一次属预期。
- **实测:live1M 峰值 174.2MB ≈163B/对象(环安全模式,-42%)**;`FROND_NO_CYCLES` **111.5B/对象(-60% 达标,但牺牲环回收)**;churn 5M 全平(链表出入平衡零泄漏);功能 95+负向 64 全绿;perf 门禁 match_dispatch 1634→1598ms(乙①补偿构造路径)。
- **立案待查**:环安全与无环模式仍有 ~51MB 差,其中阀门单发标记 ≈20MB(roots 扩展+marked 集+work 栈),余 ~30MB 疑似 **mimalloc 分配交错碎片**(thread/unthread 改变了 96B/48B 块的交替分配序列→span 碎片);后续可用 std allocator 对照或 mimalloc 调参(MI_OPTION)验证。
- 下一步(乙②字段打包)不变:块头 48+字段尾 48 → 打包后 ~48-56,预期再砍 40-48B/对象。

---

# 乙② 落地结果(2026-09-03)——总目标达成

**shape 驱动字段打包已合并。全战役终态:live1M 每对象 279B → 81B(-71%),峰值 277.4 → 92.2MB。**

## 最终数字

| 阶段 | 每对象 | 降幅 | live1M 峰值 |
|---|---|---|---|
| 基线 | 279 B | — | 277.4 MB |
| 甲(单块 DST) | 165 B | -41% | 176.0 MB |
| 乙①(侵入式链表+阀门判据) | 163 B | -42% | 174.2 MB |
| **乙②(字段打包+入口帧免阀)** | **81 B** | **-71%** | **92.2 MB** |

churn 5M 全平(零泄漏);功能 95+负向 64 全绿(2 环境挂与基线一致);perf 门禁持平(match_dispatch 1581ms vs 基线 1547,+2%);records 套件 .fndo 双模式 ALL PASSED。

## 乙② 实施要点

- `RecordShape` 增 `field_packs/pack_offsets/value_region_bytes`(编译期算好);尾区=[标量原生宽打包区][Value 槽区]。
- tags 源:**sema 构造器 `field_type_reprs`** → `TypeFieldInfo.field_tags` → `RecordLitInfo.field_tags`(序列化/.fndo 双编解码点+Accessors 惰性解码);其他 8+ 构造站点默认空。
- **物化守卫**:tags 与字段数不匹配(Newtype 空名表/字面量路径)→ 整体退回通用槽;Newtype 按 arity=1 取 pack。
- `pack_write` **按目标家族数值转换**(int→i128 lane/float→f64/f16/f128 lane):`Instant(nanos:i128)` 吃 i64 extern 返回值这类隐式加宽,sema 不插 cast 节点,旧 24B 槽无感,打包必须自转换——这是最深的一个坑。
- `field(i)` 访问器取代 `fields()` 切片(标量即时构造内联 Value);收集器只 Drop Value 槽;就地写按 pack 分派。
- **入口帧免阀**:主帧完成=程序结束,全根标记在那儿是纯瞬峰(dead=0);交给 teardown 收集。此项单独贡献 ~20B/对象。
- 修复过程中另拔掉自家乌龙:`RecordRef::new` 的 `tail.max(n*24)` 尺寸兜底曾把打包收益全部吃掉。

## 遗留(低优先)

- mimalloc 交错碎片疑案(~10-20B/对象):std allocator A/B 可定论。
- 无 tags 的构造站点(记录字面量路径)仍走通用槽——补齐 reprs 管道可再省。
- P2(Value 16B/帧槽 SoA/丙内联)未动,下一战役量级。

---

# 遗留三件处理结果(2026-09-03 收口)

## ① 分配器碎片案 —— 销案

std allocator A/B 实测(同一构建仅换全局分配器,各 3 跑取最大峰值):

| 分配器 | live1M 峰值 |
|---|---|
| mimalloc(现行) | 92.2 MB |
| std::alloc::System | 91.3 MB |
| mimalloc + PURGE_DELAY=0 | 92.1 MB |

差 0.9MB(<1B/对象,噪声级)。乙② 消掉阀门瞬峰与哈希表后,残差就是底账本身(块 48B + 槽 + 引擎底座),**与分配器无关**——乙① 时代记录的"~30MB 疑碎片"实为当时仍在的阀门标记+注册表成本。**mimalloc 保留**(其余负载的线程缓存性能收益不受影响),A/B 对照二进制留档 `tmp_probe/memcmp/frond_sysalloc.exe`。

## ② 记录字面量路径 tags —— 已接

`compile_record_like`(Err/IOError 类位置构造)与 `compile_record_lit`(`{x: v}` 字面量)经 `expr_type_name` 取字段静态类型进 tags。全量验证:功能 95+负向 64 绿,perf 门禁持平(match_dispatch 1586ms)。

**战役终态(全部 .fndo 重建后实测)**:hello 底座 **7.9MB**;live1M **92.2MB ≈ 84B/对象(-70%)**;churn 5M 全平零泄漏。尺寸探针 memsizes 已随新布局更新(RecordShape 112B、块体运行期随 shape 变化)。

## ③ P2 —— 立项不动刀(用户裁决)

**Value 16B 曾做过、很麻烦**(用户裁决,2026-09-03):i128/f128 联合占满 16B,tag 无处安放,除非 128 位标量改堆句柄——语义与性能风险都大,不轻启。帧槽 SoA / 丙内联同为大战役,立项待批。**本战役到此收口。**

## 战役总账

基线 279B/对象(277.4MB)→ **84B/对象(92.2MB),-70%**;每记录分配 2 次→1 次;记录构造、字段读写、match、深克隆、相等、环回收全链保持语义;功能 95+负向 64+perf 门禁+六语料口径全绿;工具链(探针/基准/A-B)留档 tmp_probe 可复跑。

---

# 吞吐批次(2026-09-03)

## 分解实测(churn 5M 变体,A/B/C 三段)

| 段 | 耗时 | 拆解 |
|---|---|---|
| A 纯循环骨架(compare/add/store/incr) | 1337 ms | 267ns/迭代 ≈ 65ns/节点(已知的 E 系列后派发水平) |
| B +构造即弃 | 2454 ms | **构造+释放 = 223ns/次** |
| C +字段读(=churn) | 3176 ms | 字段读+加法 = 144ns(2 节点) |

## 已落地(微优化,验证绿)

1. **REC_LIST_LOCK: std::Mutex → RAII 自旋锁**(临界区=两次指针写;含让步防 offload 竞态烧核)。首版 bool 守卫忘解锁造成挂死,已改 RAII——教训入库。
2. **compute_record_construct → RecordRef::new_from_iter**:字段直写块尾,消中间 `Vec<Value>` 往返。
3. 实测合计 ~11ns/构造(churn 3236→3212ms,-1%);**功能 95+负向 64+perf 门禁全绿**。

## 构造路径成本账本(223ns 实测构成)

mimalloc alloc+free ≈60(分配器底板,thread-cache 快路)· strong/shape 原子 ≈30 · 输入读取+帧槽机械 ≈90 · 注册自旋锁+开关查询 ≈20 · pack 读写 ≈忽略。**单点优化空间已尽**,中间 Vec/锁等显性浪费均非大头。

## 下一杠杆(立项待批):标量链覆盖记录循环

C 段 635ns/迭代中约 380ns 是派发机械(构造节点+字段读+骨架)。若 Scalarizer 把「全标量 packed 记录构造+字段读」作为标量链可编译模式,整循环免派发,预期 churn 类负载 ~1.8×。**核心设计冲突**:记录构造带分配+注册副作用,与标量链纯度模型相斥——需受控副作用或"构造保派发、读入链"的折中,须专项设计。

---

# 系统吞吐战役·第一批:优化器可达性预筛(2026-09-03)

## 动机与发现(FROND_BUILD_TIME 分段计时,新增诊断工具)

热构建分段(fib):load 7.5ms / **sema 43ms / ir_build 31ms / optimize 172ms(67%)**。诊断揭示:**图携带 6321 子图/927 函数,入口可达仅 31**——std 依赖闭包被整编进图,fib 只用一次 Instant。

## 落地:mark-only 预筛(正确性优先)

1. phase1 **之前**用 `pass_func_dce` 同源的 `compute_liveness`(fresh ctx ⇒ is_live≡true ⇒ 严格保守)把不可达函数标记 dead(节点+子图),**不 rebuild**——phase1 依赖的 analysis NodeId 保持有效;
2. LICM/Unroll 的收集与遍历跳过 dead_sgs(**收集前过滤**——克隆 std 全部循环的不变量快照曾是 phase1 热点);
3. phase1 尾部既有 rebuild 物理清除(单次重编号);phase1 不跑时(O1/无 analysis)独立 rebuild 兜底;
4. phase2 fixpoint 首轮的 pass_func_dce 作为权威复核(标记即轮1注定)。

## 实测

| | 战前 optimize | 战后 | 热构建墙钟(战前→战后) |
|---|---|---|---|
| fib | 172 ms | **27 ms(-84%)** | 0.28 → **0.14 s(-50%)** |
| mem | 341 ms | **115 ms(-66%)** | 0.48 → **0.25 s(-48%)** |
| tls13 | — | 39+12 ms | 0.34 → **0.18 s(-47%)** |

phase2 从多轮 fixpoint 变 1 轮收敛(死代码不再参与轮间振荡)。验证:功能 95+负向 64 全绿;perf 门禁持平(fib 38/loop_sum 265-276/match 1590/recursion 1083/string 4-2);六语料 diff 口径为前端阶段,优化器内部变化不受镜像纪律约束。

## 遗留(下一批,按收益排)

1. **phase1 的 run_guarded 快照 ~25ms**:Never-corrupt 策略对未裁剪全图做深拷贝(licm/unroll 本体仅 4µs)。方案:prepass 先物理 rebuild + **重映射 analysis 的 loop_analysis 两表**(需 rebuild 暴露 sg 新旧映射)→ phase1 快照缩小 30×。
2. **sema 43-60ms(std 全量重查)**:std 束缓存(内容哈希+引擎版本双键)。
3. ir_build 31-40ms:同受预筛思路(可达集先裁)但需 sema 层可达性,复杂。

---

# 系统吞吐·第二批:phase1 快照重构(2026-09-03)

## 落地

预筛从 mark-only 改为**物理 rebuild 前置**;phase1 的循环分析改为**在裁剪图上重跑 `analyze_loops`**(该函数本就是 loop_analysis 的产地,裁剪图上亚毫秒),替代"保 NodeId 不重编号"的妥协;`pass_licm`/`pass_loop_unroll` 签名随之从 `Option<&AnalysisReport>` 简化为 `&LoopAnalysisReport`。架构更干净:phase1 全程工作在小图上。

## 计量修正(诚实账)

原假设"phase1 剩余 25ms=run_guardved 整图快照"**证伪**:内部分解显示=liveness 1.9ms + **rebuild 紧缩重排 18.4ms**(O(全图) 的固有紧缩成本:扫全部节点+重映射 ~30 张元数据表)+ analyze_loops + phase1 运行。快照本身已随重构消失。rebuild 成本要消只能让图从一开始就小(见下)。

## 终态(fib 全链热构建 140ms)

| 段 | 耗时 | 下一杠杆 |
|---|---|---|
| load_modules | 8 ms | std 束 |
| **sema** | **43 ms** | **std 束(大头)** |
| **ir_build** | **32 ms** | **std 束/按需编译** |
| optimize(含预筛 rebuild 18ms) | 27 ms | std 束(裁剪图直接小) |
| 合计 | ~110 ms | **std 束可吃掉 ~100ms** |

perf 门禁持平(fib 39/loop_sum 264/match 1622/recursion 1092/string 4-2);95+64 全绿。

## 下一批:std 预检束(设计要点)

缓存粒度候选(按可行性):**「std 裁剪后图片段+用户编译所需最小 sema 面」整束缓存**,键=(引擎版本+std 内容哈希+入口 import 集)。命中时跳过 load/sema/ir_build/预筛,直接拼接片段+编译用户代码。难点=SemaResult/env 的可序列化(TypeArena 句柄网);失效安全=六语料+功能套件逐字节。备选更轻切口:若 std 模块可达闭包(从入口 import 后序展开)远小于全库,**按需装载**(不编不可达 std)可先吃掉 ir_build+rebuild 的大半——但触碰装载/sema 语义(全 std 预载是名称解析的历史前提),须先证不破坏裸名/跨模块兼容,风险高于优化器内部改动。

---

# 系统吞吐·第三批:按需装载探路(2026-09-03)——结论:此路不通,已干净回退

## 实验与发现

测得 std 预载 90 模块 vs 入口闭包(fib **1**/mem 8/tls13 14/json 23),且镜像 frondc 本就按闭包装载(loaddeps)——遂试「闭包化」。两级均失败,根因都是**全 std 预载是语义承重墙**:

1. **IR 层**:std 模块间存在**无 import 的相互引用**(Instant.frond 裸调 Duration(),靠全 std 环境绑定)——闭包化 ir_build 直接 "call to unknown function 'Duration'"。
2. **Sema 层**:std 模块的 **check 有全局副作用**——方法签名表/vtable/解析表在逐模块 check 时填充,用户代码查 `Duration.as_micros`、闭包内 std 模块查 `Detail.now` 都靠它。闭包化 sema 一片红(no method/missing ExprInfo)。有趣:sema 闭包通过 predeclare 轮能完成**类型检查**(43→3ms),但下游 IR/方法分派立即断粮。

已干净回退,95+64+perf 全绿复核。

## 对 std 束设计的修正

「跳过 std 检查」不可行——检查产物(方法表/解析表)正是用户编译的依赖。真正的 std 束=**缓存 SemaResult 全量状态**(TypeArena+env+方法表+解析表的可序列化快照),命中时整表还原,仅查入口模块。这是大工程(句柄网序列化+失效键),单独立项。

## 当前系统吞吐终态(编译腿)

| | 战前 | 批1+2 后 |
|---|---|---|
| fib 热构建 | 0.28 s | **0.14 s** |
| mem | 0.48 s | **0.25 s** |
| tls13 | 0.34 s | **0.18 s** |

剩余构成:load 8 + sema 43 + ir_build 32 + optimize 27ms。sema/ir_build 的下一步是 SemaResult 快照束(立项);执行腿 L3/L2/L1 按序待启。

---

# L3 调用路径侦察(2026-09-03)——地图完成,主刀重定向

## 取证(callbench 微基准+引擎计数器)

| 实验 | 结果 |
|---|---|
| A 内联算术 vs B 小函数调用 | B **更快**——可内联调用已被构建期内联,无剩余空间 |
| C TCO 自递归 2M | 1083ns/次;`switch_subgraph=0`,`tail_flag_hit=0` |
| 内联阈值 15→32 实验 | match_dispatch 1590→1730ms(**更慢**),已回退——阈值已最优 |

## 三条颠覆性发现

1. **自递归尾调用根本不产生 Call 节点**:Recursion.rs 把它编译成 while 循环+**参数 Cell 写回**(param_cells + cell_barrier + WriteBack)。recursion_tco 的 1083ns = 循环机械 ~265 + **Cell 参数机械 ~800**(每迭代参数写入 Arc<Cell> 堆对象、屏障、条件从 Cell 读回,均为 Value 克隆)。
2. **尾调用旗标路径(switch_subgraph 帧内跳跃)从未被走过**(hit=0)——它服务于的是"跨函数尾调用"这类形态,与基准负载无关。
3. **match_dispatch 的 compute() 不内联是正确权衡**:4 臂 match 体超 15 AST 阈值;强行内联(阈值 32)因节点派发成本反而慢 9%。

## L3 重定向(真正的刀口)

- **L3'槽化尾递归参数**:tail-rec 变换的 param_cells → 直接写循环帧值槽(参数不被嵌套捕获时),消 ~800ns/迭代的 Cell 机械。与 E6 增量重置/home 槽纪律交互深,须专项设计。
- **L3''同帧被调执行**(原方案,适用于不可内联的跨函数调用):E7 同帧分支先例扩展到函数体,挂起/效果链判定是前置。
- 通用调用快路的剩余价值集中在"中等体量非尾函数"负载(match_dispatch 类),与 L3'' 同刀。

验证:全部侦察插桩已移除,95+64+perf 全绿复核(match_dispatch 1647ms 回到基线噪声带)。

---

# match 专门优化·M1 落地 + M2 评估(2026-09-03)

## M1 构造器判别符 —— 已合并(机制地基,零回归)

- 全局驻留器 `(type_name, constructor) → u32 disc`(0 保留);`RecordShape.disc`;全部 9 处 shape 创建点接线;`pattern_disc_sets` 派生表(engine start 物化,含**继承反闭包**:臂的接受集 = {disc(T',ctor) : T'==模式类型或继承之},等价于旧字符串比较+type_inherits 图遍历);`compute_pattern_ctor_match` Adt 臂改 u32 扫描,无裁定/非 Adt 保字符串慢路。
- 验证:机制命中(match_dispatch 4/4 臂有集,records 套件全绿);95+64+perf 全绿。

## 计量诚实账:M1 对 match_dispatch ≈ 0%

臂测试的字符串比较本就不热(短路+短串 memcmp,~tens of ns/call),1630ns 的大头仍是:compute() 调用机械 ~800ns + 内层整数 match + ADT 构造 + 循环机械。讨论时预估 5-15% 偏乐观,实测归零。**M1 的价值=判别符地基**(M4 构造器 def-use 传播、M3 内联臂选择都消费它)+ 宽 match 场景的普适小赢。

## M2 construct_cache —— 不做(有据)

帧内缓存 ≤4 项时线性扫描(~10ns)快于任何 map(~20ns);本负载无收益。仅当单帧零参构造 >16 时再议。

## 下一步(M 序列重排)

真正的大头回到 **M3 侦查**(内联后 match 臂是否失去 E7 同帧资格→修好=内联+同帧组合吃掉 ~800ns 调用)与 **L3'**(槽化尾递归参数)。两者都以"消调用机械"为主刀,match 臂测试已被 M1 垫平。

---

# M3 侦查闭环(2026-09-03)——内联之谜定量破解,L3'' 方案定型

## 取证(同帧判定计数器,阈值 15)

match_dispatch:**sameframe hit=8 / miss=1,000,016**。解读:
- miss=1M = **compute() 调用本身**(函数调用过该判定点,`target_sg==function_id` 必失格——正确行为,非回归);
- 臂不在该路径上(经 relay 机制执行),hit=8 为零星分支。

## 内联变慢之谜的定量闭环

阈值 32 实验(+140ms/+140ns/迭代)现在完全解释:内联 compute() 到调用者循环 = 每迭代多执行 ~8-10 个节点(4 构造器测试+门+算术)× 65ns 派发 ≈ +650ns,换来省 ~800ns 子帧启动——**净 +140ns,与实测吻合**。内联不是错在资格,是错在"以节点派发换帧启动"在当前 65ns/节点的派发价下不划算。

## L3'' 方案定型(下一刀主刀)

**同帧被调执行**:对叶子直线函数(compute 类:无挂起/无循环/分支臂全同函数),Call 处理器不再启动子帧,而是:
1. 保存调用者上下文(subgraph_id/node_offset/value_table 整表 move 出走);
2. `switch_subgraph`(既有尾调用机器)原地切到被调体执行;
3. Return 触发时还原调用者表,值写回调用节点。

消掉的是:池取帧+整表复制+派生+还池(~800ns),付出一次表 move(~O(表长) memcpy,直线下被调表通常小)。预期 match_dispatch 1584→~800ms(**~2×**)。风险:效果链/挂起/闭包上值捕获的路径判定须完备——与 E7 资格判定同构,复用其排除清单。插桩已净除,95+64+perf 全绿。

---

# L3'' 首次实现(2026-09-03)——已全量回退,调查结论入库

## 做了什么

完整机制落地:SavedCallCtx 上下文栈(value_table 整表 move + 帧状态保存)、switch_subgraph 原地切换、**双完成路径拦截**(直线函数队列跑空→读 return_node 槽还原;早期 return→ControlSignal 拦截)、嵌套调用栈式支持。最小复现(big 算术函数+println)**结果正确**。

## 失败与回退

真实套件:放宽资格(嵌套调用+同函数包裹链)→ **50 挂**;收紧(跨函数+叶子+嵌套子图扫描)→ **95 挂**(且更严反而更差,疑与构建缓存混杂,未及定论)。**已全量回退至干净基线**(95+64+perf 全绿复核,数字回基线带)。回退途中误删 snapshot_outer_value 已按原文重建。

## 三个硬发现(下次开工的起点)

1. **直线函数的完成不走 Return 信号**:返回值在 return_node 槽,队列耗尽即完成——子帧协议由 extract_child_return 读槽;同帧版必须拦截队列空路径(已实现且最小复现正确)。
2. **表达式体 match 函数的调用链是 壳sg→包裹sg(While)→体sg(LoopBody)**:W4c 包裹复用循环子图,每次调用 3 跳子帧;同帧化它需要"包裹感知资格"(识别一次性 While 包裹),直接放开 loop_kind 守卫是 50 挂的嫌疑主因。
3. **破坏面待定位**:50→95 的失败模式表明存在一个比资格判定更根本的缺陷(候选:被调值表与调用者帧链的 get_value_by_global 交互/construct_cache gid 语义/pending_completions 路径不知道 l3 栈)。下次应先用**最小失败用例**(如 arrays 套件单跑+二分)定位,而非套件级试错。

L3' 未动。L3'' 机制代码留档 tmp_probe/patch_l3pp.py 可复生。

---

# L3' 槽化尾递归参数(2026-09-04)——落地

## 结果

**recursion_tco 1081/1057ms → 444-451/419-424ms(≈2.5×,-59%)**;功能 95(2 个 pre-existing 挂不变)+负向 64+perf 门禁全绿。改动 = `ir/Builder/` 三文件 352 行**纯新增**(零引擎改动)+ solidify/Format.rs 加载顺序修正 9 行。

## 机制:slot 通道取代参数 Cell

资格(单递归点形状):恰一个带条件 base case + 恰一个 rec 分支;cond/args/exit 纯且自由标识符 ⊆ 参数(白名单走查:字面量/Ident/Binary/Unary/As);参数无入口 Cell(③被赋值)/无 lambda 捕获/无取址。不合格 → 原 Cell 路径不变(回退)。

- 参数驻 while_sg PARAM 槽(param Const 首 P 节点;入场 Call 注入);循环迭代传输 = `ResetPlan.carries_value`(引擎既有 phi-carry 机制,此前休眠——builder 从未填充)。
- **递归参数投机提升**:rec args 在 while 帧编译,经 CF_SEQ 链**全部**链入条件树根——只有条件树成员每迭代重置重算(首版只链最后一个 arg,n 的 carry 恒为陈旧值 1,n≥2 全挂——最小用例二分速破)。
- 尾调用点降为裸 void 节点:无 Cell 写、**无 Continue barrier**;loop_kind 改 While——body 正常完成即"继续",base case 由 cond 否决走 exit_sg(bare-Ident exit 用 sg 内 CF_SEQ 包装防空 sg 外部 return_node)。
- body 通用编译不动(前导语句/死 base 臂/嵌套结构语义原样);converter 盖章临时 lifted → rec 臂(单 void Const)获 E7 同帧资格。

## 连带修复:.fndo 装载顺序 bug(休眠机制激活暴露)

装载路径 `precompute_reset_plans` 跑在 downstream 派生**之前**——其内部 E2 fused carry 派生读 downstream_slice,装载图上该表为空 → 首个非空 carry 的装载图直接 panic(owned+zerocopy 两路同病)。修正=两处 loader 把 `materialize_gate_branches`/`compute_downstream(s|_csr)` 提前到 precompute 之前。此前 carries 全库恒空,此路径从未走过。

## 遗留(下一刀)

- loop_sum seg0 269→285 疑机器噪声(本机已知混合 CPU 调度方差;L3' 不触其 IR,同构建重复跑 283-290 与账本带重叠)。待 CI 或钉核复核。
- v2 细化空间:reset_to_zero 承载 args(省 SEQ 链 2 节点/迭代)、body 内 if-cond 与 while-cond 同源去重(省 cmp_if+gate_if ≈130ns/迭代)、多递归点 phi 合并(当前回退 Cell)。
- L3''(同帧被调执行,match_dispatch ~2×)地图已画完,重启须最小用例二分(教训见上)。
