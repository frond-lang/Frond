# 核心内存优化 — 当前基线 (2026-09-03)

优化战役前的完整基线快照。所有数字来自本机实测(probe 见文末)。

## ★ 战果(2026-09-03 三批落地后,见文末"战役结果"节)

三批改动已合并:①Registry 快修(FxHash+NO_CYCLES 全免+叶子豁免)②Str 单层化(Value::Str(Arc<str>),删 HeapObj::Str/Arena 句柄桶改 Value)③Shape 共享(RecordShape 图上物化+动态站点缓存)。验证:功能 95+负向 64 全绿(2 环境挂与既档一致)+perf 门禁全线持平。

## 0. 环境

- 二进制:`Frond/core/target/release/frond.exe`(2026-09-03 构建,含 env 门控探针,关闭时近零开销)
- 分配器:mimalloc(全局)
- perf 口径:`tests/scripts/run_perf.sh`,5 跑取中位
- 注:同日首轮 perf 曾普遍慢 20-39%(loop_sum 430ms 等),判定为机器噪声已弃用;下表为复跑两轮稳定值。

## 1. 结构精确尺寸(cargo example `memsizes` 实测)

| 结构 | 尺寸 | 说明 |
|---|---|---|
| `Value` | **24 B** (align 8) | Null/Void/Scalar(16B union + 1B tag)/Ref(Arc) |
| `ScalarValue` | 16 B | 容纳 i128/f128 |
| `HeapObj` | **104 B** (align 8) | 24 变体内联,最大 TraitValue=104/ForeignFnValue=96 |
| `Arc<HeapObj>` 分配 | **≥120 B** | 16B Arc 头 + 104B 枚举,**任何种类一律 120B 起** |
| `Str` | 16 B | 内部 `Arc<str>` → Str 值=两层分配两跳 |
| `ArrayValue` / `RecordValue` / `AdtValue` | 80 B | |
| `AdtField` | 48 B | Option\<String\> 24 + Value 24 |
| `TraitValue` | 104 B | |
| `Closure` / `PartialApplication` | 64 B | |
| `Cell` / `Range` / `ThrowValue` | 24-32 B | 但分配仍付 120B 壳 |
| `ValueTable` | 128 B | 每帧一个 |
| `Frame` | 440 B | 池化复用 |
| 帧槽综合成本 | ~30.1 B/槽 | 24 值 + 2 rc + 1/8 ready + 4 dirty |

推论:3 字段 Record 实例 ≈ 120(壳)+ 72(fields Vec)+ 72(field_names Vec)+ type_name/字段名堆块 ≈ **300+ B,其中有效载荷 72 B(~4× 放大)**;1 字符 Str ≈ 120 + 17 ≈ **137 B、2 次分配(>10× 放大)**。

## 2. perf 门禁基线(probe 关,5 跑中位 × 2 轮稳定)

| 负载 | seg0 | seg1 |
|---|---|---|
| fib | 0 ms | 36 ms |
| loop_sum | 261 ms | 262 ms |
| match_dispatch | 1553/1571 ms | — |
| recursion_tco | 1027 ms | 1008 ms |
| string_build | 5 ms | 2 ms |

## 3. 进程内存峰值(OS 口径,单跑)

| 负载 | 峰值工作集 | 峰值 commit |
|---|---|---|
| fib | 127.4 MB | 189.8 MB |
| loop_sum | 128.0 MB | 187.6 MB |
| match_dispatch | 137.4 MB | 207.0 MB |
| recursion_tco | 136.6 MB | 207.1 MB |
| string_build | 128.3 MB | 192.3 MB |

基线水位 ~127-137 MB,由引擎启动+std 装载主导,负载差异 <10 MB。

## 4. 堆分配分布(FROND_MEM_PROBE=1,单跑,计数与计时无关)

| 负载 | 总分配 | 分布 | peak_live |
|---|---|---|---|
| fib | 146 | Str 65(44.5%)· **Record 55(37.7%)** · ThrowVal 18 · Adt 5 · Cell 3 | 94 |
| loop_sum | 92 | Str 65(70.7%)· ThrowVal 18 · Adt 5 · Cell 4 | 77 |
| match_dispatch | 77 | Str 56(72.7%)· ThrowVal 12 · Adt 7 · Cell 2 | 64 |
| recursion_tco | 92 | Str 65(70.7%)· ThrowVal 18 · Adt 5 · Cell 4 | 75 |
| string_build | **15096** | **Str 15069(99.8%)** · 其余为启动常量 | 83 |

要点:
- **Str 是绝对主导分配源**。string_build 每次拼接 = 120B 壳 + Arc\<str\> + 2 次全局注册锁。
- 纯标量循环(loop_sum/recursion_tco)稳态几乎零堆分配(92 次全是启动常量),证明帧池+槽化已到位。
- **peak_live 全程 <100**:注册表税在"分配/释放"每次发生,与存活集大小无关;15k 次分配全部即生即死,仍付 30k 次全局 Mutex+SipHash 操作。

## 5. Registry/环回收行为基线

五个负载一致:
- `count_checks=1` —— `FrameState::Completed` 压力阀臂**每个程序只执行 1 次**(热路径帧完成绕过该状态机臂);
- `fires=1, roots_total=0, reclaimed=50-57` —— 唯一一次回收是引擎退出时 `EngineCore.rs:1016` 的**空根清扫**,把退出时仍活着的启动常量(65 Str 等)一并收回;
- 即:**正常运行中环垃圾零回收**,循环造环负载将持续膨胀到进程结束(压力阀实际覆盖 ≈ 0)。
- `deregister > register`:清扫的 in-place 替换导致 drop 侧对已移除指针重复注销(计数含 miss,仅诊断用)。

## 6. 探针工具(战役期间保留)

- 门控:`FROND_MEM_PROBE=1`;输出在 stderr 退出时打印(报告:进程峰值/注册表计数/按 RefKind 分配分布)。
- 插桩点:`value/Registry.rs`(probe 模块+register/deregister/count/collect 钩子,标 `TEMP-PROBE`)、`value/Value.rs:1166-1180`(ref_val/register_arc 漏斗)、`cli/mod.rs`(退出报告)。
- 尺寸探针:`cargo rustc --release --example memsizes -- -C link-arg=/WHOLEARCHIVE:frond_extern.lib`。
- 回归口径:功能/负向套件 + 六语料 diff + 本表 perf 门禁;对照时先跑本文件 §2/§4 同款命令。

---

# 战役结果(2026-09-03 三批落地)

## 改动清单

1. **批次① Registry 快修**(`value/Registry.rs`):`HashSet<usize>`(SipHash)→`FxHashSet`;`FROND_NO_CYCLES` 现在**分配侧登记也全免**(原先只免回收不免登记);叶子类型豁免(Range/OpaquePtr/LibVal/ForeignFnVal/GlobalSlotRef 无 Value 边,永不入环,跳过登记/注销,Drop 侧同门控)。
2. **批次② Str 单层化**:`Value` 新增 `Str(Arc<str>)` 变体,**删除 `HeapObj::Str` 与 `Str` 包装结构体**——每个字符串从「104B 壳分配+数据分配+2 次注册锁」变为**单次数据分配、零注册**。Arena 句柄桶 `Bucket<Arc<HeapObj>>`→`Bucket<Value>`(str 句柄照常工作);Compute/Arena/Marshal/Reflect/Schedule/Ir 全部 Str 站点改走 `Value::as_str()/str_val/str_from_string`。连带修复两处浪费:`pattern_str_eq` 原为比较先各 to_string(现零分配直接比);`cast_to_str` 对 str 输入先格式化复制再重建(现 Arc 直通零分配)。
3. **批次③ Shape 共享**:新增 `RecordShape{type_name, constructor, field_names}`(56B),**编译期按 construct 节点物化进 `graph.record_shapes`(与 const_cache 同构的派生表,不序列化)**;RecordValue/AdtValue/NewtypeValue 从每实例携带 `type_name:String + field_names:Vec<Option<String>>(+每字段名 String)` 变为持一个 `Arc<RecordShape>`;`AdtField` 结构体删除(名字入 shape,值平行数组)。动态站点:record `{...spread}` 扩展按 (基 shape 指针, 节点) 缓存派生 shape;memo_check 用 OnceLock 静态 shape;heap_equals 增 `Arc::ptr_eq` 快路。

## 数字对比(基线 §4 → 现在)

| 负载 | registry 注册数 | 峰值存活 |
|---|---|---|
| string_build | 15096 → **27**(-99.8%) | 83 → **17** |
| fib | 146 → **81** | 94 → **41** |
| loop_sum | 92 → **27** | 77 → **17** |
| match_dispatch | 77 → **21** | 64 → **11** |
| recursion_tco | 92 → **27** | 75 → **15** |

(string_build 的 15069 次字符串分配全部脱离壳分配+注册税;各负载剩余注册数=启动期非字符串堆对象。)

## 结构尺寸对比(实测)

| 结构 | 前 | 后 |
|---|---|---|
| Value | 24 B | 24 B(Str 变体吸收,无增长) |
| RecordValue | 80 B | **32 B** |
| AdtValue | 80 B | **32 B** |
| NewtypeValue | 48 B | **32 B** |
| HeapObj | 104 B | 104 B(TraitValue 仍为最大变体——Box 化是下一批候选) |

每 3 字段 Record 实例分配数:**6-7 次 → 2 次**(壳+fields Vec;shape 全程共享);短字符串:**2 次分配 → 1 次**。

## 验证

- 功能 95 通过+2 环境挂(crypto_primitives/llvm_probe,与战役前完全一致);负向 64/64。
- perf 门禁(N=5):fib 39/36ms、loop_sum 269/268ms、match_dispatch 1547ms、recursion_tco 1072/1041ms、string_build 6/3ms——与基线 36/261/1553/1027/5 持平(±5% 噪声内),无回归。
- 进程峰值 125-135MB 持平(启动+std 主导,存活集本就 <100)。

## 遗留与下一批候选

- **HeapObj 104B 壳**:TraitValue(104)/ForeignFnValue(96) 仍把小对象(Cell 24B)撑到 120B 分配——大变体 Box 化(P1-4)待做,注意 Box 化对大变体+1 次分配,与"减少分配次数"目标权衡。
- **环回收阀门覆盖≈0** 仍未修(基线 §5):运行中造环负载仍会膨胀到进程结束,需单独立案(阀门挂点改到真正的帧完成热路径,或 trial-deletion)。
- Value 16B / 帧槽静态 SoA / 句柄表 vs NaN-boxing:P2 战略项,见对话讨论稿。
- ⚠️ 本机存在**外部进程周期性改写/回滚源文件**(战役期间 Registry.rs 被打回原始版两次、cli/Value/Function/Optimizer 多处被注入或剥离,均已修复);`tmp_probe/restore_registry.py`+`registry_full.rs` 保留作快速恢复。

---

# 跨语言定位(2026-09-03 实测,本机 Windows x64)

方法:同等微基准,`tmp_probe/memcmp/`(可复跑 `python bench.py`);峰值 RSS=ctypes 轮询进程 PeakWorkingSet;耗时=程序自报中位(3 跑)。Frond 用 .fndo 纯运行模式(已修 record_shapes 在 mmap 加载图上的对齐 bug,见下)。

## T1 启动足迹(峰值 RSS)

| 运行时 | MB |
|---|---|
| C (MSVC) | 3.8 |
| Rust | 4.1(Rust+mimalloc 5.3) |
| CPython 3.14 | 10.6 |
| **Frond(.fndo 运行时)** | **11.2** |
| Frond(编译+运行) | 131.7(工具链成本,类比 javac) |
| 文献锚点(本机未装) | Go ~2-25 · Node/V8 ~30-50 · JVM ~35-100+ |

## T2 存活 100 万个两 i32 字段对象(峰值增量 → B/对象)

| 运行时 | 构造耗时 | B/对象 |
|---|---|---|
| C(逐个 malloc) | 19.4 ms | ~24 |
| Rust(Box) | 21.3 ms | ~24 |
| CPython __slots__ | 225 ms | ~123 |
| CPython dict | 259 ms | ~170 |
| **Frond record** | 558 ms | **~279** |
| 文献锚点 | JVM ~32(JOL) · V8 ~56-88 · Go 16+GC余量 |

构成:24B Value 槽 + 120B HeapObj 壳 + 48B 字段 Vec + mimalloc 开销/碎片。

## T3 构造即弃 500 万次(ns/次 = 派发+分配+释放全包;RSS 是否平坦=泄漏检验)

| 运行时 | ns/次 | churn 峰值 |
|---|---|---|
| C | 22 | 平(=基线) |
| Rust / Rust+mi | 23 | 平 |
| CPython slots | 143 | 平 |
| **Frond** | **601** | **平(7.9MB,零泄漏)** |

Frond 的 601ns 含每迭代 5-6 节点 × ~100ns 派发机械(28× 战役已定标);纯分配份额 ~几十 ns,与 Python 同量级。

## T4 字符串构建(1 万次单字符 / 5 千次双字符拼接)

C 0.01/0.01 ms · Rust 0.02/0.01 · Python 1.62/0.84 · **Frond 4/2**。

## 定位结论

1. **运行时足迹:解释器第一梯队**(≈CPython,远低于 Node/JVM 文献值);编译工具链 131MB 独立看待。
2. **对象密度:偏胖档**——279B/两字段记录 vs CPython 123-170B、原生 24B;P1-4(Box 化壳 104→~40B)+P2(Value 16B)做完预计进 CPython 区间(~140-150B)。
3. **分配吞吐瓶颈在派发不在分配**;回收语义与 C/Rust 同级(RC 即时、churn 全平零泄漏、无 GC 停顿/堆倍增)。

## 连带修复:.fndo 加载图 record_shapes 对齐 bug

`materialize_record_shapes` 原读 `record_lit_infos` Vec——mmap 加载图该 Vec 为空(打包段须走 `record_lit_info_at()` 访问器),导致 .fndo 模式所有构造节点落到空 shape(构造名丢失→match 无臂 panic/静默零值)。改走访问器后 records 套件 .fndo 模式 ALL PASSED;项目模式本就正确。功能 95+负向 64 复跑全绿。
