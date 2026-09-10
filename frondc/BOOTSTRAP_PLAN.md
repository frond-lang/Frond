# 自举计划(BOOTSTRAP_PLAN)

用 Frond 写的 Frond 编译器(frndc),最终自己编译自己,原生产出。
2026-08-27 立项,2026-08-28 计划文档化。本文是自举主线的唯一计划源;
名称解析根治是支线,见 `Frond/core/NAME_RESOLUTION_PLAN.md`。

## 一、目标与终局

**自举闭环的定义**:在 Rust 引擎上跑 `frondc`(Frond 写的编译器),
让它编译 `frondc` 自身,经 LLVM 产出原生 `fronc.exe`;此后原生 fronc
接管编译,Rust 引擎降级为 bootstrap 工具与开发期运行器(可退役)。

**明确不移植的**(自举路径不需要):
- `ir/`(15k):IR 生成器/Compute/Optimizer/Verifier——解释器路径专属;
- `engine/`(6k):帧执行机——同上;
- `pass/Analyzer`(4.7k):post-sema 咨询件(死代码/记忆化),非正确性必需;
- `solidify/`(3.6k):.fndo 序列化,同上。

自举走 **AST→LLVM 直下**:`Analyzer` 之后已是单态化、类型完备的代码,
直接 lower 成 LLVM IR 的 SSA,让 LLVM 自己优化(`LLVMRunPasses` O2)。

## 二、梯子(Stage 0-4)

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Stage 0** | Rust 编译器+引擎(现状) | ✓ 存在 |
| **Stage 1** | frondc 跑在 Rust 引擎上:词法+语法+全套 sema(含 monomorph)。**不碰 LLVM**——验证"Frond 语言表达力足以承载自己的语义系统" | ✓ **完成(2026-08-31:全节差分绿 + 终局验收三级)** |
| **Stage 2** | frondc 的后端模块用 std.llvm(Frond 代码调 LLVM-C)lower AST→.obj,内嵌 lld 解出后 spawn 链接(零宿主工具链,见 四) | **切片 0/1/2 落地(2026-09-04~07):Back lowering 字面量/算术/main + 控制流 + 聚合(record/数组/str,bump arena);`native` 子命令端到端,退出码验收 16/16**;探针绿(2026-08-31 macOS 实证 + CI) |
| **Stage 3** | 引擎里跑 frondc.frond 编译它自己 → 原生 fronc.exe。**此刻闭环达成** | 未开始 |
| **Stage 4** | (可选)Rust 引擎退役,fronc 为唯一编译器 | 未开始 |

梯子的核心洞察:**Stage 1 不需要 LLVM**——后端从第一天就用 Frond 写
(经 std.llvm),不用先写 Rust 后端再移植(双写税 ×2);std.llvm 已趟平
FFI 的雷。

## 三、Stage 1 里程碑与验收

移植分层(每层独立差分验收,不留死区):

| 里程碑 | 内容 | Rust 对应物 | 状态 |
|---|---|---|---|
| 前置① | canonical sema dump(oracle 发生器) | cli/Dump.rs | ✓ 完成 |
| 前置② | 表达力探针 → tests/functional/expressiveness | — | ✓ 完成 |
| **1A 词法器** | 字节级状态机,全部怪癖 | Parser.rs(Lexer 部分) | ✓ **408 文件差分全对齐** |
| **1B 语法器** | AST 定义 + S 表达式 printer + 递归下降(Pratt/虚拟>/三路回溯/插值子解析) | Ast.rs + Parser.rs(主体) | ✓ **401+10 差分全对齐** |
| **1C 模块加载** | mini-TOML(Toml.frond)+ 模块加载器六步解析 + import 后序图(Loader.frond)+ stdlib 有序清单(StdPaths.frond,生成) | module/Loader.rs(717)+ cli/Manifest.rs | ✓ **load-dump v1 全语料对齐** |
| 1D 类型系统 | 类型 ADT/TypeArena/unify/occurs/kind/display | types/(核心 1810:Tag+Ty+Arena+Display) | ✓ **ty-ops 差分 75 操作逐字节对齐** |
| 1E sema 全家 | 推理/trait 见证/继承/monomorph | sema/(~17.4k) | ✓ **完成(2026-08-31:全节差分绿 + 终局验收三级,见 三f)** |

**1E 终局验收三级**:87+62 套件双跑等价 → **std/ 全库自检**(frondc-check
检查它未来要编译的代码)→ apps 语料(editor + llvmfetch)。

当前规模:frondc 已有 ~8450 行 Frond,按目录模块组织(2026-08-28
模块化,拓扑镜像 Rust core):

```text
src/
  Main.frond            薄 CLI(分发 + cmd_load + dump_load_entry 编排)
  syntax/               1A/1B 前端:Lex / Ast / Parser / Dump
  module/               1C 模块加载(镜像 Rust core/src/module/):
                        StdPaths(生成)/ Toml / Loader / Manifest
  types/                1D 类型系统(镜像 Rust core/src/types/):
                        Ty(ADT+名字表)/ Arena(竞技场+unify)/ Tyops(差分 battery)
```

包内互引用走限定文件路径(`import syntax.Lex`);Main 用逐文件限定
导入。将来 1E 新增 sema/ 目录。(模块原名 Parse 曾暂避
std.json.Parse 撞名,引擎根治推进后已恢复——限定导入形态下短名
调用按导入解析绑定。)

## 三e、1E 进展(2026-08-28 开始,M1 声明层落地)

**sema 包已建**(src/sema/,~2500 行 Frond,拓扑镜像 Rust sema/):
`Data`(ADT)/`Envs`(env 竞技场)/`Semares`(SemaResult 注册表)/`Ictx`
(推断上下文+绑定栈)/`Populate`(populate_module+ast→sema 转换+concretize+
继承合并+方法表扩展)/`Typeast`(type_from_ast 族)/`Modenv`(register_
builtins/模块 env 层级/predeclare/imports 再输出/witness 填充)/`Check`
(check_module_with_env 编排+alias 环/ctor 重名检查+builtin 合成类型注册)/
`Sdump`(sema-dump v1 render)。
Main 接线 `check|checkmany` 命令(镜像 run_sema_pipeline_or_exit 的
prepipeline+check 循环)。
**(2026-08-28 二更)双 Arena 撞名别名桥(Tya/Asta)已物理删除**:
`syntax.Ast` 的 Arena 真改名 `AstArena`、`types.Arena` 的 Arena 真改名
`TypeArena`(对齐引擎侧命名,根除撞名);`TArena`/`AstArena`(桥)/
`new_tarena`/`zero_span` 全部消失,sema 文件裸名直用;唯一值位工厂
`Main.check_one` 的竞技场构造走全限定表达式 `types.Arena.new_arena()`
(限定路径特性首次自举吃狗粮)。差分 447/437/77(tyops) 复验绿。

**表示裁决**(超越镜像原则):Sym interner 不移植(dump 全节显式排序,
Map<str> 语义等价);type_defs 族 = List(位置即 u16 分配序);module_
imports/std_binding_origins 多值 = ',' 连接串;ctor_def_index 多值 =
packed u32 的 csv。

**自举逼出的引擎根修**:① **is_turbofish_call 的 GtGt 虚拟拆分缺失**
(ast/Parser.rs)——表达式位嵌套泛型 `f<A<B<C>>>()` 的闭合 `>>` 在
lookahead 中不计数 → 探测失败 → 整段被 parse 成比较表达式(垃圾 Lt/Shr
节点,ExprInfo 缺失 → IR 编译失败)。修 = lookahead 认 GtGt/GtGtEq
(depth 有符号化,减 2)。② **`T?` 赋值不放宽**是 Frond 写法约束(不是
bug):nullable 赋值用 match 表达式(null 臂 + 值臂 join)。
**(2026-08-28 更新)类型注解位/trait 名位/match 模式位已支持限定路径**
(`A.Point` 注解、`A.Box<i32>` 泛型头、`std.collections.List<i64>` std 全
限定、`<T: A.Eq>` trait 界、`A.TEf`/`A.Point(x,y)` 模式;sema 走
resolve_module_qualifier 四级映射:imports 逐序 → std 前缀裸键 → 全局唯
一;镜像 Parse.frond 同步,差分 449/439 绿)——冻结条款的显式例外,
写法约束清单里"签名/字段/match 模式全部裸名"条目作废(双 Arena 撞名的
Tya/Asta 别名桥随后一并物理删除,见上文 1E 段二更)。

**遗留立案**(下一步):
0. ~~loadmany 运行期崩~~ **(2026-08-28 同日根修销案)**。真实根因与文件
   IO 无关(fn 名归属是 sg 重编号后的错位):**撞名零参变体作构造实参静默
   编译为 void**——`sema.Data.TDK.TAdt`(零参)与 `types.Ty.TAdt`(一元)
   同裸名,Populate 里 `TDef(name, TAdt, ...)` 的裸名实参在 IR
   compile_ident 走首胜字符串构造器表拿到一元条目,守卫失败落入
   `compile_const()` → void 进 kind 槽 → populate_field_ids 的 TDK match
   四臂全灭 → non-exhaustive panic(最小复现:双模块同名变体,TMP 复现
   一次成形)。修(S2 铁律补零参值引用缺口):sema
   `infer_nullary_ctor_with_expected` 期望类型裁决后 record_ctor_resolution;
   IR compile_ident 先消费 ctor_resolutions(`ctor_tf_info_from_resolution`)
   再落字符串回退;静默 void 改为报错 "constructor 'X' requires
   arguments"(模块接收者 `Path.from` 形态豁免——Path 0 不消费 recv)。
   连带修复 M2 期 loadmany 输出漂移:run_sema 试跑块摘除(sema 归
   check/checkmany),deps/modules 清单恢复走 `Loader.render_graph`。
   **验收:diff_load 12/12 复绿;frondc check/loadmany 双通路复活
   (name_resolution 全套 sema-dump 0 错 0 警);functional 93(新增
   ctor_name_clash 回归锚)+ negative 64 + lex 450 + ast 440+10skip +
   tyops 77 全绿。**
1. ~~record 的 Map 字段类型推断串台~~ **(2026-08-29 同日根修销案,四件套)**。
   因果链(定案):泛型链返回值(`wits.get(i)`)推断时刻是 pending TypeVar
   → BinaryOp::Elvis 的 Nullable 分支**只返回内型不与 RHS 统一** →
   coalesce 结果变量永悬 → 字段读取产生新未绑变量 → 其作接收者的
   `.get` 落入 **Path-0 自由函数回退**(CallInfer.rs:`recv.m(args)` 糖)
   → 首个同名自由函数 `types.Arena.get(a, h)` 元数恰合 →
   `unify(params[0], 裸var)` 即时绑定 → 字段 ExprInfo 被毒化。**修**:
   ① BinaryOp::Elvis Nullable 分支对 pending 内型 try_widen/unify_or_constrain
   绑定到默认值类型(真源头;注意 Expr::Elvis 是死臂,活臂在 infer_binary);
   ② 类型驱动方法路径补 **this 参与接收者统一**(签名类型参从 recv 绑定);
   ③ Path-0 对裸 TypeVar 接收者**跳过急切候选绑定**并记录挂起;
   ④ check_module_with_env 新增 **9.2 挂起方法调用重试轮**(求解后/void
   缺省前,≤8 轮;仍悬者回落 fresh-var 语义 = 原尾行为)——承重的
   pending-recv 自由函数解析由重试轮治愈。**验收**:用户态完整复现
   (双模块+push 局部 List+keys 循环)从静默错值→正确;镜像 Sdump
   witness slots 明细行**直接形态恢复**(不再规避),check 双套件 0 错;
   functional 94(新增 map_field_crosstalk 锚)+ negative 64 + lex 453 +
   ast 443+10skip + tyops 77 + load 12 全绿。附带:diff_lex/diff_ast 镜像
   侧分批(每批 100)——453+ 语料超 Windows 命令行长度上限(exec 126)。
2. **check 运行期 compute_match_fallback panic**:populate 到
   builtin/error/Error.frond(第一个 type decl)时踩无输入的死 fallback
   节点(sg33/n1613;Option match 的 Gate 编译正确)。dump_ir 缺
   fn_id→函数名表是定位障碍——补上后从 5 个 cf311 节点反查函数。
3. **嵌套泛型在类型位**(`List<Map<str,u32>>`)词法 GtGt 由类型 parser
   的 expect_close_angle 虚拟拆分处理;**表达式位**靠 is_turbofish_call
   修复(两侧已同步,ast 差分 434/434 实证);**两处路径不等价**是技术债。
4. ~~check 命令端到端尚差~~(**2026-08-29 片4a 落地 + sema 差分门建成**):
   ① `tests/scripts/diff_sema.sh`——声明前缀契约(modules/types/ctors/
   methods/traits/witness/field-ids/errors/warnings,到 !monomorph 前),
   6 语料字节级绿;片5 就位后并入全节比对。② 片4a:`sema/Bodyinfer.frond`
   表达式全量访问集(FunDecl/Type+Trait 方法体/全局 ExprDecl[引擎口径:
   stmt 在则走 stmt——void dummy 不计]/局部声明递归/defer/lambda/match
   臂 guard/select 两臂/interp 部件/数组填充对[仅 has_fill]/LBlock 为
   expr id)。stats 的 **expr_types 从 0 → 27295(引擎 27288,恒差 7**=
   引擎 infer_ident 流窄化命中提前返回不落表,随片4b 真实推断消除)。
   键差分方法论:双侧 [ek]/[mk] 键清单 dump + sort -u diff(探针已摘)。
   **(同日算法优化轮,超越镜像原则)**:walk 计数器+seen 位图去重
   (每模块 bool[],等价引擎键集语义——AST 存在 73 处结构重复可达,位图
   恰好复现 Map 去重;6.2s→1.5s);render List 化(拼接非大头,分发是,
   待片5);**origin 一次预建索引**(subtree/direct 两表,替代逐 import
   全量扫描+split;imports 段);**ctor_def_index csv→List<i64>**(消
   读写解析)。镜像 check 总时长 46.2s→38.7s(-16%)。基线:引擎 sema
   97ms,镜像 check-loop 21.6s(五段分解在案);Map<str> ~230µs/op 的
   机制 = 键哈希走 "{k}" 格式化 + 桶行 ++ 整行拷贝 —— IntMap 化热表
   是下一刀。教训:改 record 字段必须核对 ctor 字面量实参数(元数错位
   会伪装成名称劫持)。
   **(第三轮;StrMap 已按用户裁决完整回退——std 是用户领地,未经
   批准不得改动)**:① StrMap 曾往 std/collections/Map.frond 加 str
   快路子类并切换镜像热表;实测零收益,且用户明令不动 std,已回退
   干净(std 零残留、镜像 27+25 位切回 Map、计数复原 27295=引擎+7、
   七门复绿)。② **alias/dup 检查增量化**
   (SemaRes.checks_scan_base 水位,每模块只扫新增 type_defs;环在末
   成员落位时被发现,语义等价)。实测:计时无显著变化(46.2→38.7 主要
   来自前两轮)——**Map 代价主因是键拼接(已除)而非插值哈希**;
   StrMap 已回退(见上)。终局分解(计时摘除前):builtin 0.8s /
   **std 检查环 7.0s / render 3.8s**——两者均为泛型分发本体,真正
   解锁 = 片4b/片5(monomorph)。计数保真:镜像 27310 = 引擎
   27303+7(Map.frond 自增 StrMap 代码两侧同步 +15)。七门全绿。
   剩余:~~片4b(真实 ExprInfo/CallInfer/Solver → resolved/call_inst/
   dispatches 计数)~~(**2026-08-29 片4b 落地**,见下)→ 片5
   (monomorph/inherited:collect_monomorph_instances + 实例化模式 +
   resolved_types/call_instantiations/mono_local_expr_types)→ stats 全
   对齐 → 差分全节。

   **(2026-08-29 片4b 落地:真实体推断)**:`sema/Infer.frond`(~6.4k 行,
   镜像拆分前的 Inference.rs 单文件形态)+ `sema/Relate.frond`
   (types_equal/is_subtype/peer/type_name/check_type_node)。覆盖:
   Subst(instantiate/freshen)/Solver(等式定点+候选+null-join+歧义)/
   Flow(流窄化事实栈)/Unify(unify_return_type/unify_call_arg/
   try_widen_unify/propagate)/Helpers(数值/迭代器/reflect/lib/隐私门/
   缺返回值/Throw 尾包)/Stmt/Expr/Call(Path 0a/0b/0/1 全族 + super 三层
   + 隐式 this)/Match(GADT 细化 + Maranget 穷尽性)/check_decl;
   Check.frond 接真实推断循环 + kind 检查 + 9.1 歧义 + 9.2 挂起重试 +
   9.4 默认 void。**表示裁决**:推断期状态类型居 Data.frond 低层
   (Ictx 引用避免环);大单文件避 Frond 模块循环导入;IR 专用写表
   (ctor_resolutions/dispatch_targets/captures/module_*_recv)跳过——
   不进 dump/stats/HM 无回读;method_dispatches 仅 intrinsic 标记。
   **验收**:diff_sema 声明前缀 **6/6 绿(真实推断)**;stats **expr_types
   六语料与引擎逐位相等**(28015/27585/27619/27518/27523/27409;4a 的
   ±7 残差随真实推断消失);method_dispatches=51(引擎 62-70 的 HM 部分,
   差值 = monomorph 条目,片5);lex 455 + ast 445+10skip + tyops 77 +
   load 12 全绿。
   **连带引擎侧支线(同日,用户裁决)**:S5 歧义硬化 + 零静默原则
   (NAME_RESOLUTION_PLAN 第七节)——表达式位裸名多主首胜、模式位
   find_ctor_def 首胜回退均改歧义错;接收者位与 Error 接口开放域两豁免;
   镜像 `resolve_type_key_in` 带点拼写补 map_qualified_key_in 规范化
   (std.collections.List → 裸键 List;声明层潜伏缺口,推断统一句柄时
   显形)。镜像侧新增引擎修复:无(纯镜像)。
   **镜像写法约束新增**:跨模块同名词汇表(TDK × Ty)表达式位一律
   `Data.TRecord` 限定或前缀(match 模式位有类型消歧,表达式位无);
   Frond 侧 `?? 0` 产生 i32( peer 数值化)——u32/i64 槽须 `?? 0u32`/
   后缀或 as 转换;T→T? 赋值必须 `= if true { e } else { null }` 或
   注解 val;语句位 match 尾逗号非法(臂分隔逗号合法);字符串内
   大括号必须转义(含错误文案)。

**回归终态(2026-08-28)**:functional 91+negative 62+ast 差分 434/434
+ty-ops 77 逐字节+load 12/12+perf 基线同量级,全绿。



- **差分口径 `ty-ops`**:Rust `frond debug --stage ty-ops <file>`(cli/
  Dump.rs dump_tyops)vs frondc `tyops` — 固定 75 操作 battery
  (标量名/display 往返、类型变量绑定与 resolve 压缩、occurs 拒绝、
  刚性变量、never/unknown 原槽吸收、标量/fn/record/nullable/ref/
  adt/generic/trait/array/throw/单参泛型族/trait_object 的 unify
  正反例、kind 变量与合一、display 全形态)+ stats 块。
  **handle 序号是差分锚**(操作全序贯,两侧分配一致)。
  工具:tests/scripts/diff_tyops.sh(77 行逐字节)。
- **范围裁决**:types/ 的 Ops.rs(915,动态操作注册表)/Ctype.rs(C 映射)
  /Kind.rs(reflect 常量)属 IR/reflect 域,归 1E 后期与 Stage 2;
  TypeFamily 完整分类/bit_width 族同理按需。核心(Tag+Ty+Arena+
  Display ≈1810 行 Rust)全量移植。
- **表示映射**:TypeHandle/DetailId/EnvId=u32;Type 变体 T 前缀;
  **Frond 无元组** — parts 族返回 pair 记录(U2/U2O/U2B/FU/SU/SM/SB);
  Option<u64> 尺寸=i64?;ModuleRef 的 EnvId 用 u32 占位(1E 接 env)。
- **镜像纪律**:resolve_mut 路径压缩;unify 的 never/unknown **原槽
  覆写**(display 原句柄可见吸收后类型);make_nullable 的 T?? 塌缩;
  Trait~TraitObject 同名互通;单参泛型 display 只打名不带参数
  (Display.rs 行为);kind_debug 镜像 Rust Debug 形态
  (`Arrow { param: Star, result: Star }`)。
- **语言坑(1D 新踩,均为 Frond 侧写法约束)**:标识符不能以关键字
  开头(throw_parts/val 字段名;词法按前缀切词);字符串字面大括号
  必须 `\{`/`\}` 转义(dyn Trait 输出、kind_debug 双双踩中);
  **match 会窄化 var**(resolve_kind 的循环形态被 KVar 臂窄化卡死,
  改递归);顶层 fun **不支持 `&` 前缀**(变异靠容器共享);
  `return throw X` 形态有运行期风险(统一改语句位 `throw 构造`,
  std File.read 先例);**void 载荷的 Throw 有运行期风险**
  (unify_kind 从 Result<(),()> 改 bool);跨模块 ADT 构造器撞名
  (SemKind.KArrow vs Ast.Kind.KArrow → SK 前缀)。

## 三b、1C 细节(2026-08-28 完成)

- **差分口径 `load-dump v1`**:Rust `frond debug --stage load <entry>` 与
  frondc `loaddeps/loadmany [--std <dir>] <entry>...` 逐字节一致。四节:
  manifest(root=no/bad/yes+四字段;从入口父链向上找,锚定入口非 CWD)
  / deps(load_transitive_imports 后序,含"后声明先展开"入栈序)
  / modules(全部已载键的逻辑路径集合,字节序排序)
  / errors(发生序:not_found/parsefail/circular)。入口解析失败 →
  `! fatal`(文案不对齐,同 1B 后置项)。
- **std 内容来源**:Rust 编译期 include_str! 内嵌 ↔ frondc 读 std 根目录
  磁盘(--std > FRONDC_STD_ROOT > CWD 向上探测)。键映射:
  "builtin/x" → <root>/builtin/x;"std/x" → <root>/x。
  **有序清单 StdPaths.frond 由 tests/scripts/gen_std_paths.py 从
  StdlibEmbed.rs 生成**(兄弟回退 5b 的迭代序承重;增删 std 文件后必跑)。
- **夹具**:tests/functional/loaddeps(文件模块+目录模块 pack+兄弟符号
  回退,正路径)+ tests/fixtures/loaddeps_neg/{missing,circular,badmanifest}
  (错误路径,仅供 diff_load.sh)。工具:tests/scripts/diff_load.sh。
- **自举逼出的引擎修复**:Monomorph.rs infer_type_args 显式类型实参
  (turbofish)曾用被调方 arena(fd_ast)解析调用方节点 → 跨模块 arena
  越界 panic;id 恰好装得下时静默解析错节点(正确性洞)。修复 = 改用
  调用方 arena(ast)。
- **模块化逼出的引擎修复(2026-08-28)**:限定导入(`import sub.L`)下,
  `L.Eof` 这类「模块.枚举变体」值访问,IR 的 Module.Ctor 分支用
  ctor.def_module **全逻辑路径 == recv 短名**匹配——限定导入把
  def_module 变深路径("sub.L")后永不命中,回落成对 ModuleRef 的垃圾
  FieldAccess;变体比较恒假,is_at_end 式循环守卫失效,toks[len] 越界
  panic(或更糟的静默错行为)。修 = 尾段匹配(ir/Builder/Access.rs)。
  最小复现 ~40 行(sub/L 带 Eof 哨兵枚举 + 判 Eof 循环)。另发现:
  import-as 语法不存在;限定短名与 std 模块撞名(Parse.parse vs
  std.json.Parse.parse)IR 层正确报二义,建议三段限定。
- **模块短名撞名的完整根治(2026-08-28,用户裁决「Rust 侧根修」)**,
  三件套落地,裸导入/限定导入两形态、正序/反序声明全部验证:
  ① `module_func_call_targets`(sema Path 0a 记录导入解析的模块逻辑
  路径):IR 的 MethodCall Path 0 在此裁决存在时**直接按全路径 mangled
  键绑定**(`path0_sema_target`),完全绕过字符串键族 — 短键 tripwire
  记录的是**历史**撞名,曾把 sema 已裁决的调用推翻成二义硬错;
  ② `func_short_index` 独立容器:短键(tail.fn)与 own-mangled 曾共用
  func_subgraphs,**src/ 根用户模块的 own-mangled("Parse.parse",无
  目录前缀)与 std 模块短键字符串相同**,预注册的「同键复用」把两个
  不同函数混为一个子图,后编译的用户体**静默覆盖** std 子图内容
  (std 全路径调用执行用户函数)。分容器修复;
  ③ resolve_func recv 分支先全路径(func_subgraphs)后短键
  (func_short_index),与 sema 导入解析严格一致。
  语义终态:**导入即真理** — `import Parse` 后 `Parse.parse(...)` 绑
  用户模块(声明序无关);`import std.json` 后同名调用绑 json;三段
  全路径恒正确。判例:name_resolution 案 8(二段短名用户赢 + 三段
  std 可达 + 枚举值访问),复现样板 /tmp/clash。
  附带发现:import-as 语法不存在(限定导入即自由命名,无需别名)。
- **async 递归平方律根修(2026-08-28,用户裁决「修复」)**:
  `setup_frame_chain`(engine/Frame.rs)的 root_frame_ptr 沿 caller 链
  上溯,**判定只看 caller_fn_id == frame_fn_id** — 自递归的 fn_id 恒相等,
  每层新帧 setup 时把整条递归链走到顶:O(深度)/帧 × n 帧 = O(n²)。
  且自递归是跨函数语义(Bug #102 家族),本就不该设 same_function 链
  指针 — 修复 = 函数体 sg 帧(sg == function_id)不设链,分支子图帧
  照旧。**效果:10k/30k/100k 深度 2s/20.8s/>300s → 0.2/0.56/1.2s(线性,
  100k 处 246×+);recursion_tco/match_dispatch 同步递归亦受益**。
  排查路径记录:quiescence sweep/TracedMutex/多 worker 惊群/await 路径
  逐一排除(event_waiters 的 Vec→HashMap 化是顺手的正确性收窄,保留);
  定位靠**分段步进曲线**(每万步耗时随挂载数对称涨落 = 成本∝当前
  挂载集 → 必有每步 O(挂载数) 的遍历 → setup_frame_chain 链走)。
  验收:91+62+差分 12+ty-ops 77+perf 套件全绿。
  方法论教训:根治途中曾得出「loader 5b 吞噬 dep 键」的第七层结论并
  立案 — 复核发现是**实验卫生事故**(对照实验中 cp 在错误 cwd 失败,
  复现现场的模块文件实际缺失,「模块 env 为空」是文件缺失的直接后果)。
  撤销立案、重建干净现场后真正残案只有一层(IR tripwire 推翻 sema
  裁决),一次修复收官。教训:多层挖掘中**每一层的现场完整性必须
  复核**(ls 文件、验 env),否则会在自造的迷宫里立案假 bug。
- **语言坑(踩过)**:await 只有后缀 `.await()` 形态;async 值函数签名
  必须写 `Async<T>`(void 除外),早 return 值可用但软整型要先 `as i64`;
  `val` 是关键字(参数/字段/方法名都禁);字符串里 `{` 是插值起始,
  字面大括号要转义;`str?` 赋值不自动放宽(用空串哨兵,Env.get 约定);
  `Error(e)` 构造会撞 builtin Error 类型名(用 `throw e` 重抛)。

## 三c、超越镜像原则(2026-08-28 用户裁决)

**差分锁的是可观察行为,不锁内部算法**:frondc 对 Rust 侧的镜像义务
止于 oracle 输出(load-dump/sema-dump 逐字节);内部实现鼓励超越,
每项优化的门禁 = 差分全绿 + 回归全绿。已落地(1C,差分 12/12 不变,
全语料差分耗时 15-20 分钟 → 40 秒):

- **std-first 加载序**:Rust 在依赖解析期跑兄弟回退 5b — 逐个试探
  **解析**未载兄弟找导出,未命中者的解析被丢弃,force_load_std 又
  全部重解析一遍(双重浪费)。frondc 先全量加载 std(终局反正要载),
  5b 退化为纯缓存查询,每个 std 文件恰解析一次。
- **跨入口共享解析缓存**(std_cache):parse 是纯函数、ModuleAst
  载后只读,loadmany 的 N 个入口共享一份 — std 解析量 127×N → 127。
  仅 std 路径使用;用户模块保持逐入口独立语义。
- **线性扫描代替哈希集**(failed/visited/visiting):百级规模下线性
  str 扫描远快于 Map<str> ~200µs/op 的哈希查找 — 这是一处"看似
  朴素实则更优"的选型;规模上千再换(1E 时按实测裁决)。

候补(未做,量级太小):str_lt 免拷贝比较(现每次 bytes() 复制)、
5a 前缀索引、渲染聚合。标准:有实测收益才做。

## 四、Stage 2 设计裁决(已定,待实施)

**零宿主工具链 + 平台一致性(2026-08-31 用户裁决)**:frondc 发行包
自包含全部工具链资产,宿主机不需要装 LLVM/SDK/链接器;驱动不变量 =
跨平台一致性。

### 4a 一致性规则(设计律)

**平台差异只允许住在两个地方**,两层之上(语言语义/std 包装层/构建
行为/发行结构/验收口径)必须平台无感;新增平台相关物必须归入其一,
两头都不属于 = 设计缺陷:

| 层 | 允许的平台差异 | 对外契约 |
|---|---|---|
| C 原语体(`#{ }#` 内 `#if`) | Win32/POSIX API 选择、编码桥 | 统一:out-buffer -1/-2、句柄 0=失败、退出码 128+signal、UTF-8 进出 |
| 链接视图资产(按 triple 分发) | crt/import 库/CRT 名 | 统一逻辑名 + 内容寻址缓存,`Assets.extract` 同一 API |

### 4b 内嵌三件套(同一资产管道,三种消费方式)

**单产物 + 全内嵌 + 无工具链子命令(2026-08-31 用户裁决,再次确认)**:
最终 frondc 每平台一个同源构建,工具链资产(LLVM-C/lld/linkview/后续
C 层)全部内嵌于二进制。运行期按系统自适应的**只有加载与链接策略**
(lld `-flavor`、链接参数表、dynamic-loader 路径、arch)——统一文件名
裁决已把"打开哪个文件"也抹平(恒为 llvm.dll/.so/.dylib 对应形态)。
工具链管理不进 CLI(无 fetch/use 类子命令);llvmfetch 仅为开发/CI 侧
资产获取器。**资产解析序**:内嵌解出(发布态默认,`Assets.extract`)
→ `FRONDC_TOOLCHAIN` 环境变量(开发态换版本不重编)→ `<exe 同目录>/
assets/toolchain`(套件/便携形态)。探针已按此实现(Env 哨兵 = 空串)。

| 资产 | 形态 | 消费方式 |
|---|---|---|
| LLVM-C | 动态库(llvm-static 21.1.8,五 triple) | `Lib.embed` → 解出 → dlopen/LoadLibraryW |
| lld | 可执行(lld 无 C API,进程内免谈;**与 LLVM 同仓同 tag 构建**——vendored lld 21.1.8 两段式独立构建,对着刚建好的 LLVM 树静态互链,自包含) | 提取 → `os.spawn`(单二进制按调用名分派 lld-link/ld.lld/ld64.lld) |
| C 层(std C 原语 + frond_rt + 启动对象) | `.obj` × 5 triple | 提取 → 链接输入 |

- **C 层发布期预编译**(CI 五 triple 全工具链),构建期只链接不编译 C
  ——主线不需要 C 编译器,**clang 不内嵌**;用户手写 `@extern` C 片段的
  编译能力 = 可选后续(宿主 clang 或再议)。
- **资产管道统一**(2026-08-31 已落地到 llvm 仓工作流):llvm-static release
  = frond 工具链资产集——同一 tag、五 triple 同构 tarball:`lib/`(LLVM 库)
  + `bin/lld` + `linkview/`(钉下限的链接视图)+ MANIFEST + sha256 sidecar;
  llvmfetch 一并拉取校验。资产分管道 = 版本漂移 = 一致性事故。
  **动态库统一文件名 `lib/llvm.{dll,so,dylib}`**(Windows import 库配对
  `llvm.lib`;版本标识移入 tarball 的 `VERSION`):Frond 永远按显式路径
  dlopen,文件名不参与符号搜索,统一安全;按名链接(SONAME/install_name)
  的路径在新设计里不存在,配对改名顺带消掉这层含糊。
  **边界**:C 层 `.obj` 不在此管道——它是 std 的版本锁定代码,归 Frond 仓库
  release CI 预编译(需要 std 源码,天然住在那边)。
- **`Assets.extract(name): Path` 泛化**(提取与 Lib 解耦);修 TOCTOU
  (临时名+rename 原子落盘)与提取目录(专用缓存目录 %LOCALAPPDATA%/
  ~/.cache/frond——temp 即写即载的 DLL 易被 AV 拦,只读环境另议)。
- **原生 fronc.exe 的资产内嵌落点**:字节数组 `.c` 进同一 cc 管线
  (零新工具;objcopy/.res 三平台各异);降级 = 发行包 assets/ 同目录
  + `Lib.embed` 失败回退 `Lib.open`。
- **frond_rt 清单具体化**(2026-08-31 盘点):argv 三件套
  (`frond_runtime_argc/arg_ptr/arg_len`——std C 原语唯一的引擎宿主
  符号耦合,os/Raw)+ UTF-16 桥两 helper(从 Gen.rs 逐文件注入收编为
  frond_rt 单一定义)+ Arc 加减/分配/字符串/panic/abort + dlopen/
  dlsym + ForeignFn 动态调用 trampoline(AbiTable 117 臂的 C 移植;
  或 frondc 后端对字面量 lookup 常量折叠成直调,frondc 自身零
  trampoline 依赖)。

### 4c 下限矩阵(第二档:单一钉死下限,产物与构建机无关)

| 平台 | 链接视图 | 产物下限 |
|---|---|---|
| linux-gnu | manylinux_2_28 的 crt(Scrt1/crti/crtn)+ 真身 `libc/libm/libdl`(仅链接期,tarball 内置) | glibc ≥ 2.28(RHEL8+/Ubuntu 20.04+/Debian 10+) |
| windows-msvc | mingw-w64 crt + import 档案(crt2/dllcrt2/libmingwex + kernel32/ws2_32/msvcrt 等,tarball 内置;来源 = niXman 16.1.0 **msvcrt** 定版下载,GCC 运行库在其版本化目录) | 一切 x64 Windows(msvcrt.dll 冻结随系统) |
| apple-darwin | 零资产,链接旗标 `-platform_version macos 11.0 …` 钉住 | macOS ≥ 11 |

- 符号版本绑定发生在**链接期** → C 层 `.obj` 版本中立,一套 `.obj`
  配不同链接视图即可(多版本全换不用重编)。
- Windows flavor 裁决:**msvcrt 优先于 ucrt**(下限 = 一切 x64 机器 vs
  Win10+;std Windows 分支全是老稳 API,msvcrt.dll 全覆盖)。**C 层
  Windows 资产换靶 `clang --target x86_64-w64-mingw32`**(头/启动/
  import 库同属 mingw 视图;引擎侧 Rust 构建不动)。
- **资产自身下限同批钉死**(2026-08-31 落地 llvm 仓工作流):
  ① Linux 基线定为 **glibc 2.28(manylinux_2_28 容器)**——这是机制允许的
  绝对下限:actions 的 node20 运行时自身要求 glibc ≥ 2.28,manylinux2014
  (2.17)容器无法执行任何 step;且 LLVM 21 需 gcc ≥ 11,centos7 生态 EOL
  得不偿失。CI 加了下限断言(objdump 查 libLLVM.so/lld 引用的最高
  GLIBC 符号版本 ≤ 2.28),换 runner/依赖变化时红灯在 CI 而非用户机器。
  ② macOS 资产加 `CMAKE_OSX_DEPLOYMENT_TARGET=11.0` + minos 断言(否则
  macos-latest SDK 默认值 = macOS 版 glibc 事故);③ Windows `libLLVM.dll`
  已 /MT 零 CRT 依赖。
  ④ Linux 链接细节:glibc 2.34 之前 dl/m 未并入 libc → linkview 带真身
  `libc.so.6/libm.so.6/libdl.so.2`(libdl 的 soname 历来是 .so.2;
  frondc 自身 dlopen libLLVM 要 libdl),
  链接期按绝对路径直接吃 `libc.so.6` 真身,绕开用户机无 glibc-dev 时缺失
  的 libc.so 链接脚本。
- 多版本 glibc 全家桶(Zig 式逐版 stub)= 交叉编译特性,Stage 4 后;
  musl(MIT)全静态 = 不碰 Lib 的用户程序可选(frndc 自身不可用——要
  dlopen libLLVM)。许可:glibc LGPL 未修改再分发(工具链常规,指源);
  mingw-w64 CRT/import 档案 permissive。
- **macOS 记账(2026-08-31 探针实测)**:macOS 26 起系统库全部进 dyld
  共享缓存,**盘上已无 /usr/lib/libSystem 真身**——「零资产」假设对
  链接期失效(链接视图只在 SDK 的 libSystem.tbd,装 CLT 才有)。两个
  事实:① 零外部符号的程序可**零库链接**(产物无 LC_LOAD_DYLIB,dyld
  直跑 LC_MAIN——探针即此形态,零资产成立);② 真 C 层(frond_rt 调
  libc)时代需要 macOS 链接视图,裁决待做:xcrun 探测 SDK tbd(要求
  CLT)vs 资产自带 tbd(零依赖)。

### 4d 验收口径

- **最小宿主环境 = 三平台零工具链**(发行包 = fronc + 三件套资产)。
- 第一刀探针(五 triple):main ret 42 → `Assets.extract(lld)` → 链接
  → 跑 exe 断言退出码;外加下限断言:产物在 glibc 2.17 容器跑通、
  Windows 产物 `llvm-readobj` 查 import ∈ {msvcrt, kernel32, ws2_32}。
- **支持矩阵(明写,替代隐性约束)**:linux-gnu(glibc≥2.28)/
  windows-msvc x64 / apple-darwin(x64+arm64)。`posix_spawn_file_
  actions_addchdir_np` 的 glibc/macOS 专属性由此背书;macOS 26 SDK 已
  标其 deprecated(换 `posix_spawn_file_actions_addchdir`),记账。
- **CI 平台矩阵 = 一致性的执法者**(五 runner × functional + negative +
  差分 + llvm_bind):设计文档不保证同源同行为,矩阵才保证。

### 4e 其余裁决(沿旧)

- **降低起点**:AST(Analyzer 后)→LLVM 全新路径,不复用 .fndo IR
  (帧模型是解释器机械,E 系列优化对原生无意义)。
- **值表示**:v0 统一 Value 盒(正确性优先,约解释器 2-5×);
  v1 标量 unbox(i64/f64 进寄存器,聚合保持盒式)。
- **运行时库**:frond_rt C 库(~50-100 函数,清单见 4b);
  **cycle collector 明确后置**(v0 可泄漏)。
- **async/defer**:~~v1 后端 sync-only(frndc 自身写成纯 sync)~~
  (**2026-09-07 修正**:实测 frondc 有 22 个 async fun + 62 处
  `.await()`,并非纯 sync——但 std/frondc 的 async fun 全是
  "签名 async、函数体同步原语直调"(std/io/File.frond:11 注释自证);
  原生 = **直通降级**:普通函数直调 + await 恒等,零状态机,见附九);
  defer 2 处低频(切片 5/6 顺手);async→状态机仍是自举之后的大件。
- ~~**Throw**:先 setjmp/longjmp,后迁零成本 unwinding~~(**2026-09-07
  改判,见附九**:两态值直落,零运行时。引擎实证——`throw` = 构造
  `ThrowVal(Err)` + **仅当前函数早退**(Compute.rs compute_throw_wrap_
  err);调用返回的 Throw 是**数据,不隐式穿透**(Bug #65 明确修掉);
  `?` = 调用点显式传播(compute_propagate)。setjmp/longjmp 是隐式
  跨层 unwind = 引擎已移除的语义,不是简化是错译,废弃)。

## 五、差分基础设施(验收方法论)

| 工具 | 口径 | 现状 |
|---|---|---|
| `debug --stage tokens` vs `frondc lex` | 词法逐字节 | 411/411 ✓ |
| `debug --stage ast` vs `frondc parse` | AST 逐字节 | 401+10 ✓ |
| `debug --stage load` vs `frondc loaddeps` | load-dump v1 逐字节 | 12 入口 ✓ |
| `debug --stage sema` vs `fronc check`(未来) | sema-dump v1 逐字节 | Rust 侧就绪 |
| 退出码 + stderr | reject 用例对齐 | 后续项 |

工程要点:frondc 走 `lexmany`/`parsemany` 批量(408 次冷启动→1 次);
切片必须纯 bash while read(**Windows gawk 文本模式剥 \r** 会无声损坏
CRLF 词素);批解析 411 文件 ≈5 分钟,timeout ≥900。

## 六、已完成的支线与修复

自举逼出的引擎修复(详见 NAME_RESOLUTION_PLAN.md 终态):
- 名称解析 S0-S4(构造 ID 贯通回退 0 / 分派 98% / LSP 单源 / ambient 可见性墙);
- predeclare-populate 顺序、单 ctor ADT 注册序、Module.Ctor 值、
  同名类型劫持、元数消歧等六个 P0;
- 表达力套件 7 探针 + name_resolution 套件 7 案,全绿。

## 七、风险与对策

| 风险 | 对策 |
|---|---|
| 1E 体量(17.4k Rust → ~20k Frond) | 按 Inference 文件族切片,每片接差分;速度按实测 80k 行/两周 |
| 引擎 bug 阻塞 | 最小复现立案再修;每轮全量回归 |
| 双写期语言演进税 | 语言语义冻结于 v1,自举完成前只修 bug 不加特性 |
| Map<str> ~200µs/op | frondc 符号表选型优先句柄/整数键(IntMap)或届时上 C 内核 |
| Stage 2 值表示返工 | v0 盒式先闭环,v1 unbox 增量演进 |

## 八、下一步

**里程碑(2026-09-01):CI 矩阵 + 差分门 + llvm_probe 全绿**——四
suite 作业通过(mac/linux×2/windows;第五平台 macOS-x64 因 Intel
runner 退役并入资产层覆盖),differential 首次完整跑完,llvm_probe
全链(Lib → LLVM-C → .obj → 资产 lld+linkview → 原生 exe → 42)在
Linux(libc_nonshared.a 补齐)与 Windows(mingw 运行库按文件名 find)
收官。教训入册:① 绕开 libc.so 链接脚本直吃真身,必须自带
libc_nonshared.a(__libc_csu_init/fini 的家);② 目录名排序选目录必
踩坑(16.1.0/ 与 lib/ 并存,sort -V 挑错),文件一律按名 find;
③ 链接失败必须透传诊断(探针 `FAIL: lld|` 前缀穿透 runner 的
失败报告 grep——没有它这轮要多三趟盲改)。遗留:mem 计时断言按
CI 并行负载放宽(意图不变);tls13 留 NOCI 待引擎修。
**下一步 = Stage 2 后端本体**:~~frondc/src/backend 已立(Toolchain
资产解析序 + Llvm 绑定副本),下一片 = Back lowering(切片 0:
字面量/算术/main)→ `native` 子命令端到端 → AST 全量~~(切片 0/1/2/3
已落地,见 附二/附四/附八/附九;**下一步 = 切片 4+(见 附九)**)。

## 附二:Stage 2 切片 0 记录(2026-09-04)

**范围**:字面量(0x/0o/0b/负号/后缀)/一元(- ! ~)/算术(含
f16-f128)/比较(O 族 IEEE754)/位运算/移位/AsCast 全转型族/局部
val/var 栈槽/重赋值/return;`frondc native [--run] [--std] <entry>`
端到端(sema 管线复用 → Back lower → verify → default\<O2\> →
TargetMachine .obj → 资产 lld 链接 → 跑产物回显退出码)。

**验收**:`tests/fixtures/native_slice0`(4 用例:基本算术/整型族/
浮点+转型/混合;fixtures 而非 functional——run_functional.sh 按目录
跑 `frond run`,本验收是脚本驱动)退出码全中 31/509/25/147,driver =
`tests/scripts/native_slice0.sh`(EXPECT-EXIT 头注释 + FRONDC_TOOLCHAIN
资产序);差分 diff_sema 6/6 + diff_load 12/12 + diff_tyops 77/77
(sema 管线重构 run_sema_core 返回 SemaOk 记录,check/native 共用,
字节契约不变)+ functional/negative 全绿。

**三雷(全修)**:
1. **LLVM 21 LLVMBuildICmp/FCmp 谓词在第 2 位**((builder, pred, lhs,
   rhs, name);旧文献/旧版是第 4 位)——传错位 = 谓词落 LHS 槽被当
   Value* unwrap,段错误(03 过 04 崩,隔离用例二分定位);
2. **LLVMRunPasses 的 Options 不可 NULL**——21.1.8 实现 `unwrap(Options)`
   直接解引用;必须 LLVMCreatePassBuilderOptions 真建对象,用毕 Dispose;
3. **引擎洞:大模块内 match 臂返回裸零参 ADT 变体 → 静默 null**
   ([ctor-match-null] 警告 + 调用方拿到 null 返回值;ty_kind 返回
   BT ADT 时 llty_key 全灭)。handleprobe 小规模复刻不触发(if 臂/
   match 臂/or-pattern/List 读回全绿),frondc 模块规模触发——
   规避 = Back.frond 分类全用 i64 标签 + 整数字面量臂,不再定义
   返回位 ADT。根因未深挖(疑优化器吃臂家族),复现资产在
   tmp_probe/handleprobe。
连带:backend/Llvm.frond `val` 参数名(const_int)从未被引擎解析过
(backend 从未被 import)——首跑显形,改 value;libs/llvm/Llvm.frond
同病潜伏未修。

**工程事实(handleprobe 实证,Back 架构依据)**:LLVM u64 句柄经
局部值/函数参数/函数返回/数组元素/List 元素/Map 值/nullable/模块 var
读回后 FFI marshal **全部安全**。~~record 字段读回 = 软类型值,FFI
call 收 0~~(**2026-09-04 复测已被引擎治好**,见 附三)。字面量走 LLVMConstIntOfString/LLVMConstRealOfString(文本
直达 APInt/APFloat,f128 全精度;radix/前缀/下划线自剥,负号按二补)。

**值表示记账**:切片 0 标量全程 SSA/栈槽零堆值;v0 Value 盒(计划 4e)
随聚合/str/引用切片(需 frond_rt C 层)进入,骨架表示无关。

**下一片 = 切片 1**:~~控制流(IfE/嵌套 BlockE/逻辑短路/循环)+ 多
函数与调用(命名/mangled)→ 再聚合/str + frond_rt + v0 盒~~
(**2026-09-04 切片 1 落地**,见 附四;短路先行落地见 附三)。

## 附四:Stage 2 切片 1 记录(2026-09-04)

**范围**:IfE 值表达式(结果槽 + cond_br 两臂 store + merge load;
无 else 非 void = 报错,对齐引擎口径)/块表达式作用域(**副本式**:
块内定义只进 slots/tyslots 副本,退出即弃,免保存-恢复栈)/嵌套
遮蔽/WhileS(条件块回边)/LoopS/BreakS/ContinueS/免花括号循环体的
AssignE/入口模块多函数(**两相装配**:原型先行解互递归;签名取
`Envs.lookup_local(root_env)` 的 TFn → `Arena.fn_parts`;main 强制
i32() ABI,其余 mangle `frond_<name>` 防 CRT 碰撞)/CallE(实参逐一
conv 到参数槽类型)/ReturnS 按函数返回类型泛化(void → ret_void)。

**终止路径纪律(flow_stopped 模块 var)**:return/break/continue 发
终结指令后置位;一切表达式组合器(binary/unary/AsCast/call/if-cond/
short-circuit/赋值)在消费子表达式值前检查——stopped 路径的 0u64
占位句柄**不得**再进任何 build(FFI null = 段错误)。if 双臂终止 →
merge 无前驱,补 `unreachable` 保 verifier。while/loop 体后回边仅在
未终止时发出;循环上下文(break/continue 目标块)List<u64> 栈。

**验收**:`tests/fixtures/native_slice1`(5 用例)退出码全中
11/210/37/244/31——if 值嵌套/while 累加/loop+continue+break/fib(12)
递归+双函数/块遮蔽+if 值;driver = `tests/scripts/native_slice1.sh`,
CI 五平台矩阵已挂(负套件后)。语料退出码全锁 0..255(POSIX wait
status 8 位截断,保 CI 可移植)。checkmany 引擎侧 sema 先行验证 0 错。

**雷**:无 else 的 if **不得**预创建 else 块——空块无终结指令 =
"Basic Block does not have terminator" Broken module(LLVM 硬规则:
每块必恰一条终结指令,孤儿块同罪);块按形态懒创建。

**下一片 = 切片 2**:聚合(record 构造/字段)+ 数组 + str 字面量 →
bump arena 纯 IR(**2026-09-07 落地**,见 附八落地记录);
跨模块函数与 monomorph 实例 lower 随自举临近再排。

## 附八:Stage 2 切片 2 计划与落地记录(2026-09-06 计划,2026-09-07 落地)

**范围**:单构造器 record(构造/字段读/FieldAssignS 写/作参数返回)、
数组(ArrayLitE 含 fill 循环/IndexE 读写/复合写/越界 trap)、str 字面量
(全局常量块/len 内在/== != 内联字节循环)、聚合嵌套组合(record 数组/
record 内数组/str 字段)。多 ctor ADT/match、nullable 族、str 拼接索引
切片、`++` 拼接与 push 族、引用、RecordLitE/RecordExtendE、深相等、
跨模块与 monomorph——全部响亮报错,后续切片。

**值表示(镜像引擎 oracle,Value.rs 已核实)**:record = Arc 单块共享
(**引用语义**——赋值拷指针,`b.x=10` 经 `a` 可见;引擎 Record(RecordRef)
同款);str = 单分配 UTF-8 不可变;数组 = len + 可变元素堆块。原生 v0:
record/str/数组全 `ptr`(标量维持切片 0/1 SSA/栈槽不动)——
record=`{f0,...}` 字面 struct(字段序=ctor 声明序)、str=`{i64 len,
[N x i8]}` 弹性尾、数组=`{i64 len, elem...}` 弹性尾;无 rc 头无 tag
(静态特化);`llty_key`:str=`pstr`/数组=`parr:<elem>`(递归)/
record=`prec:<模块>::<类型名>`。FieldAssignS/arr[i]=x 经指针原地
store,共享语义自动与引擎一致。

**§2 分配策略(裁决变更:原计划"frond_rt C 层起步"→ bump arena)**:
甲(extern malloc)破 macOS 零库链接;乙(frond_rt.c 现在起步)把五
triple C 交叉编译管道拖进本切片且真消费者(rc/panic/argv/UTF-16 桥/
dlopen)全在后续切片;**丙 = Back 每模块 emit `@frond_heap`(1GiB
.bss 零页)+ `@frond_heap_top`(i64 前沿)+ 辅助函数 `frond_alloc(size)
-> ptr`(load top → 16 对齐 → icmp 越界 → store → ret 旧顶;越界
`llvm.trap`)**。零外部符号(macOS 零库链接不破)、零新资产(五 triple
CI 零改动)、泄漏 = 4e 已接受(v0 可泄漏)。frond_rt 真进场时
`frond_alloc` 换 C 实现即可,调用点不动。多 .obj 各带 arena,块跨模块
传递安全(分配归属不区分);arena 收敛单点分配器随 frond_rt 一起做。
1GiB 虚拟段 Windows PE 表现 = lld 实测,异常降 256MB。

**绑定新增**:struct_type/build_struct_gep/build_gep/add_global/
const 家族(const_struct/const_array)——LLVM 21 API 位次先探针后
进主线(谓词位次教训,附二雷 1)。trap intrinsic 复用 declare 通道。

**实施序**(每步全量门禁绿再进):绑定+布局地基 → str → record →
数组 → fixtures/driver/CI。

**验收**:`tests/fixtures/native_slice2/cases/`(11-16:record 构造+深
访问/**共享语义锚**(var 别名字段写可见)+translate 链/宽 record 10
字段+混合标量/数组字面量+fill+原地写+len 迭代/record 数组+Matrix
嵌套/str 相等+len+作字段读回),退出码锁 0..255;driver =
`tests/scripts/native_slice2.sh`(拷 slice1 模板),CI 五平台矩阵挂
负套件后;每 fixture 先过引擎 checkmany 0 错;门禁 = native 0/1/2 +
functional + negative + diff_sema/load/tyops 全绿。

**风险**:新 LLVM API 位次雷(独立探针先行);record 共享语义 fixture
写完先引擎跑确认 oracle;fill/元素求值序以引擎 dump_ir 为准;1GiB
.bss PE 实测;多 ctor 误入构造分支(严格 constructors.len()==1 &&
ctor 名==类型名,否则报 "multi-ctor ADT arrives with match (slice 3)")。

**落地记录(2026-09-07)**:

**验收**:`tests/fixtures/native_slice2/cases/`(11-16)退出码全中
57/112/110/176/96/87——record 深链构造+共享语义(translate 链)+
宽 record 10 字段混合标量/数组字面量+fill+原地写+复合写+越界 trap/
record 数组 Matrix 嵌套深链写/str 相等+!= +len UTF-8 字符数+作字段
读回;driver = `tests/scripts/native_slice2.sh`,CI 五平台矩阵已挂
(ci.yml:100,负套件后)。门禁全绿:native 0/1/2(5+5+6)+ functional 98
(llvm_probe 补跑)+ negative 69 + diff_lex 489 + diff_ast 479+10skip +
diff_load 12 + diff_tyops IDENTICAL + diff_sema 6 = **1167 用例零回归**。

**关键修复(实施期定位)**:
1. **GEP i8 基型单索引**:字节步进 `[头, off]` 双索引报 "Invalid
   indices for GEP pointer type";修 = 先 `build_add` 加总成单索引
   `boff = 8 + i` 再 `gep i8, ptr, [boff]`(i8 无嵌套层可索引)。
2. **str_len_chars 谓词反转**:续字节判定 `(b & 0xC0) == 0x80` 用
   `icmp ne`(33u32)→ iscont 真 = 续字节却跳过计数,数的是续字节
   非 0 的("héllo"=5 字符数成 3);改 `icmp eq`(32u32)→ iscont
   真 = 续字节,跳过;首字节才 inc。diag16 退出码 71040016→87 实证。
3. **ty_same 数组忽略 size**:`[T; N]` 宽化到 `T[]`(引擎允许,值
   形态同为 ptr);不忽略 size 时 conv_value 报 mismatched aggregate。
4. **record 构造器 TAdt 名义引用**:CallE 推断为 TAdt(detail id
   每次新分配不可靠),布局按类型名解析;conv_value 的 (7,7) 分支
   按 `record_name_of` 同名判等。

**顺带修复**:run_functional.sh / negative/run.sh 的 FROND 路径
解析 bug——`cd "$(dirname "$0")"` 后再拼同相对串会解析到错误位置
(从仓库根调用时 FROND 误判 not found → exit 2 零输出);CI 显式
设 FROND env 一直掩盖,本地裸跑必中招;改纯相对路径 `../../Frond/
core/target/release`(与 native_slice*/diff_* 同款)。

**切片 3(和类型一刀切)已落地(2026-09-07)**,规划与落地记录见 附九。

## 附九:Stage 2 切片 3+ 规划(2026-09-07,自举终点倒排)

**现状盘点**(frondc 源码实测:33 文件 / 28044 行 / 902 fun /
208 import;由 Stage 3 终局倒排优先级):

| 特性 | frondc 内使用 | 承重度 |
|---|---|---|
| match(多 ctor ADT) | 1052 | 极高——最大单缺口 |
| `??` / nullable | 448 | 高 |
| `.push()` | 395(多数 List 接收者) | 高(→ 跨模块) |
| `++` 拼接(数组为主) | 180 | 高 |
| throw / Ok-Error 消费 | 92 / 385 | 高 |
| async fun / `.await()` | 22 / 62 | 高(直通降级,见 4e 修正) |
| or-pattern | ~200 | 中 |
| 引用 `&` | **0** | 移出关键路径 |
| defer / RecordLitE / RecordExtendE | 2 / 7 / 8 | 顺手件 |

std 依赖闭包(切片 6 面向):collections.List/Map(全员)+
io.Path/Fs/File/Dir + os.Proc/Os/Env/Info + math + time ≈ 20 文件,
底层 C 原语(`#{ }#`/@extern)→ 预编译 .obj 资产链接。

**三裁决(2026-09-07 用户拍板)**:
1. **切片 3 和类型一刀切**:多 ctor ADT + match 全模式族 + nullable
   合并一片,统一 `{i64 tag, payload}` tagged-union 表示。T? 本质 =
   2-ctor 特例(tag 0 = null,tag 1 = 值;nullable 无 Some 包装,
   镜像引擎 T?? 塌缩语义);分开做会重复设计表示层。
2. **throw 两态值直落,废弃 setjmp/longjmp**(4e 已改判):引擎实证
   ——`throw` = 构造 ThrowVal(Err) + 仅当前函数早退;调用返回 Throw
   是数据不隐式穿透(Bug #65);`?` = 调用点显式传播(Throw 的
   Ok 解包/Err 早退与 nullable 的 null 早退/非空直通同一通道,
   compute_propagate)。原生:throw 语句 = 构造 `{1, payload}` +
   `ret`;`?` = 内联 tag 检查双臂;match Ok/Error = 普通 2 臂;
   零运行时、零 C 层、零新资产、零 ABI 差异,IR 膨胀仅在显式 `?`
   点(frndc 几乎不用 `?`,主流是 match 消费 + `??`)。
3. **async 直通降级**(4e 已修正):async fun lower = 普通函数,
   `.await(e)` = e 恒等;无需状态机。

**切片序列**:
- **切片 3(和类型)**:ADT 布局 + match 编译 + nullable(细则下);
- **切片 4(Throw + async 直通)**:throw/`?`/Ok-Err 构造 lowering +
  async 签名直通 + await 恒等——两态值直落后全部零资产小件;
- **切片 5(动态聚合)**:数组 `++` 拼接、str 拼接、str/数组索引、
  `[a..b]` 切片、record/数组深相等(`==`/`!=`);纯 IR 走
  frond_alloc(切片 2 既有),零新资产;defer 顺手;
- **切片 6(跨模块 + monomorph + std)**:mangled 多模块 lower、
  call_instantiations 实例重放、方法分派/witness、std 依赖闭包 +
  C 原语 @extern → declare + C 层 .obj 链接输入 + frond_rt 最小
  内核(dlopen/LoadLibrary/argv/UTF-16 桥/spawn;进场时
  `frond_alloc` 换 C 实现,调用点不动——§2 策略丙预留);RecordLitE/
  RecordExtendE 顺手;
- **Stage 3(自举闭环)**:引擎跑 `frondc native` 编 frondc 自身 →
  原生 fronc.exe;fronc 再编 frondc 平方验证 + 产物 diff + 三平台。

**切片 3 细则**:
- ADT 块 = `{i64 tag, payload 区}`:tag = ctor 声明序;payload 布局
  两案(最大公共 payload vs per-ctor 偏移表)以 IR 体积/GEP 次数
  实测裁决,独立探针先行(附二雷 1 教训:LLVM 21 API 位次先探后进);
  字段访问 = 8 字节头 + 字节偏移 GEP(**切片 2 的 i8 基型单索引
  铁律复用**:偏移先 build_add 加总成单索引);零参 ctor payload 空
  (引擎 ctors 表判定,严格 `constructors.len()`,勿依赖 detail id)。
- 布局键 `padt:<模块>::<类型名>`;值形态 = opaque ptr(与其他聚合
  一致,ty_same 按 record_name_of 同名判等的既有口径推广)。
- match 编译:scrutinee 单次求值入槽 → load tag → switch(小 ctor
  数用 icmp 链);模式族 = 字面量/绑定/ctor 嵌套(递归子模式)/
  or-pattern(同块多比较)/guard(谓词假 fall-through 下一臂)/
  通配;臂体 = 子块 + 结果槽(镜像 IfE 切片 1 槽位法);穷尽性
  sema 已保证(Maranget),不可达尾兜底 `unreachable`;绑定回填在
  臂内先于谓词/臂体。
- nullable `T?`:标量 payload 直接内联(块 = `{tag, 标量}`),
  聚合 payload = ptr;`??` Elvis = tag 检查 + 双臂选择;赋值不放宽
  语义照镜像(附二/三既有写法约束)。
- **语义锚**:每 fixture 先引擎跑 oracle(checkmany 0 错;求值序以
  引擎 dump_ir 为准)。
- **验收**:`tests/fixtures/native_slice3/cases/`(枚举判别+嵌套
  ctor 模式/or-pattern/guard/nullable `??` 链/深嵌套 match 值);
  退出码锁 0..255;driver = `tests/scripts/native_slice3.sh`(拷
  slice2 模板);CI 五平台矩阵挂负套件后;门禁 = native 0/1/2/3 +
  functional + negative + 差分五套全绿(串行)。

**风险与记账**:
- payload 布局抉择影响后续所有 ADT 代码的 IR 质量,探针先行;
- match 深嵌套模式的状态机式回退(绑定污染/臂间泄漏)——副本式
  slots(切片 1 块作用域法)天然隔离,验收 fixture 覆盖;
- 引擎 ctor-match-null 洞(附二雷 3)原生侧不存在(静态特化),
  但 Back 自身规避形态(整数字面量臂)保持不变;
- 跨模块同名 ADT 布局键带模块名,不撞;
- Stage 3 端到端时长:引擎解释执行 frondc 编 frondc(28k 行 + std
  闭包),Map 热表 ~200µs/op(§7)可能放大——IntMap 化或提前到
  切片 6 前,实测裁决;
- 引用 0 使用,正式移出自举关键路径(Stage 3 后按需再议)。

**落地记录(2026-09-07)**:

**验收**:`tests/fixtures/native_slice3/cases/`(21-26)退出码全中
31/44/60/102/155/81——多 ctor 枚举判别/带字段 ctor 嵌套模式/
nullable 构造装箱+Elvis 链+Ident 窄化/match guard+or-pattern+字面量
模式/nullable 聚合(数组元素 nullable+record 装 nullable)/递归
ADT(Cons 链 sum/len)+record 装多 ctor ADT 字段访问+PRecord 按名
按位+match 返回 str;driver = `tests/scripts/native_slice3.sh`
(slice2 模板,CI 五平台矩阵待挂)。门禁全绿(2026-09-07 实测):
native 0/1/2/3(5+5+6+6,slice3 同源先批)+ functional 98 +
negative 69 + diff_lex 495 + diff_ast 485+10skip + diff_load 12 +
diff_tyops IDENTICAL + diff_sema 6 = **1187 用例零回归**。

**布局裁决(细则两案 → per-ctor struct 胜出)**:探针实测弃"统一
{i64 tag, 最大公共 payload}+8 字节头字节偏移 GEP"案——**每 ctor
独立 LLVM struct `{i64, fields...}`**(键 `adtc:<类型名>:<ctor索引>`,
字段序 = ctor 声明序,复用切片 2 record 全链);tag 判别 = 首字段
i64 `icmp eq`(32u32),字段访问 = `struct_gep`(免 i8 单索引加总),
内存 = per-ctor 尺寸;T? = 同款 2-ctor 特例 `{i64, T}`(键
`nulv:<inner>`,tag0=null 零参/tag1=Some;标量 payload 内联,聚合
payload = ptr);null 字面量 = 每 nullable 类型一个 zeroinit 私有
全局(`nulcst:`)。值形态全 opaque ptr,ty_same 按类型名判等
(切片 2 record_name_of 口径推广到 8/9 两类)。

**关键修复(实施期定位)**:
1. **ValDeclS 绑定类型**:IR 定位槽位误用初值类型 + 初值未按绑定
   类型 conv(`val x: i64? = 整数` 槽位成标量非 2 字块,后继
   `??`/match 全炸);修 = 绑定类型三处落地(Semares 新
   local_decl_types 表 + Infer.check_local_decl 存 +
   Back.ValDeclS 取),槽位按绑定类型分配,初值 conv 装箱后入槽。
2. **PRecord 模式推断**:Infer.infer_pattern 的 PRecord 分支原用
   fresh_type_var(镜像引擎同款——引擎动态执行无碍,native 需具体
   类型,26_match_deep 炸 "unresolved type for expr");修 = 按期望
   record(TAdt 名义)的 record_shape 实际字段类型绑定子模式,fid
   口径镜像 Back.pat_bind(按位 Prf.name = 十进制键 int_to_key 形态,
   按名 record_field_idx;sema 层 dec_idx_local 本地副本免跨层
   import)。期望非名义/字段未命中 → fresh_type_var 旧径保守回退。
3. **字面量模式 scrutinee 谓词**:标量 scrutinee 的字面量模式被
   `k >= 0` 误拒(标量 kind 全 >= 0);改 `k >= 5`(仅 str/聚合
   不可作字面量匹配面)。
4. **nullable arm 窄化**:null arm 之后的绑定 arm 已知非 null(镜像
   引擎 Match.rs 窄化),arm_h 窄化到 inner + conv unbox;
   Ident 引用 nullable 局部同理(conv_value 9→inner 直落)。
5. **Elvis/比较/断言族**:`??` 短路双臂(tag 判别+取值/右值);
   T? == T? 三臂分发(双 null 真/单 null 假/双值 inner 递归);
   `!!` 非空断言 null 臂 `llvm.trap`;Eq/Neq 与 sema Data 枚举
   撞名 → `Ast.Eq`/`Ast.Neq` 显式消歧。
6. **conv_value 标量→T? 装箱**:inner 类型与标量源不同宽时先转
   inner 再装箱(i32 字面量 → i64? 槽)。
7. **CI linux-x64 PIE 链接(切片 3 推送后显形,病根在切片 2)**:
   str 字面量私有全局常量 + 默认 RelocDefault(x86-64 ELF → Static
   绝对 32 位寻址 R_X86_64_32S),glibc 默认 PIE 的 lld 直接拒
   ("recompile with -fPIC",仅 linux-x64 挂,arm64/musl/macos/
   windows 非 PIE 或本就 PC 相对);修 = TargetMachine reloc 参数
   0→2(LLVMRelocPIC,RIP 相对)——本地 .o 实证:修复前 ADDR32,
   修复后 .text 全 REL32(ELF 对应 R_X86_64_PC32),exit 87 不变。

**Frond 语言顺带发现**:赋值不放宽 `str` → `str?`(nullable 变量
赋非空值须 `if true { x } else { null }` idiom);本切片统一改用
非空哨兵(空串)变量 + nullable 数组承载,规避之。

## 附十:Stage 2 切片 4 落地(2026-09-08,Throw 两态值直落 + async 直通)

**范围(附九三裁决 2/3 的实现)**:throw 语句 / `?` 传播 / Ok 构造 /
match Ok-Error 消费 / async fun 直通 / `.await()` 恒等——全部零运行
时、零 C 层、零新资产、零 ABI 差异。

**表示**:Throw 值 = per-臂块 `{i64 tag, payload}`(tag 0=Ok / 1=Err;
V、E 各自 payload 型,聚合载荷 = ptr——同 nullable 纪律),值 = 块
地址(opaque ptr);`agg_kind` 新码 10,tag@0 读取复用 nullable_tag。
Err-Throw 构造仅经 throw 语句(引擎无 Error 构造器 builtin,只有 Ok
——镜像 Modenv 同口径)。

**接线七处**(Back.frond):①`throw_arm_layout/box/unbox` 三 helper
(键 `thr:{o|e}:{payload-key}`);②`Ok(v)` 特判(fn_index miss 后、
record 构造器前;按调用点 sema 型 TThrow 的 V 部装箱);③`ThrowS`
语句(载荷 conv 到 E 部 → box tag=1 → ret 早退;cur_ret_h 非 TThrow
响亮报错);④`PropagateE`(tag 检查双臂:Err → conv 后 ret 回传
Throw 本值;Ok → unbox V 续行);⑤ctor_test k==10 臂(Ok=tag 0 /
Error|Err=tag 1;单子模式按 payload 型递归 pat_test——镜像 Infer
refine_constructor_pattern 同口径);⑥MethodCallE `.await()` 恒等
(async fun 已按内型注册,调用产物即 payload);⑦收集门放宽
(is_throwing/is_async 入列;签名返 TAsync → register_fn 按内型 T
注册)。ty_same/conv_value 补 Throw 双部递归同型。

**验收**:`tests/fixtures/native_slice4/cases/`(27-31)退出码全中
46/77/25/37/19——显式 Ok + throw 早退 + match Error/Ok 消费 / `?`
传播链(Ok 直通 + Err 早退回传)/ str+record 聚合载荷 / async 直通
+ 串 await 链 / Async<Throw<V,E>> 组合(frondc 自身形态);driver =
`tests/scripts/native_slice4.sh`(slice3 模板);CI 补挂 slice3+
slice4 步(五平台矩阵,负套件后)。语料先行纪律:五件先过引擎
oracle(打印形态数值验证,顺带修正两处 EXPECT 算术 46/37),再进
Back——首跑五件全中,零雷。

**门禁(2026-09-08 实测)**:functional 98(llvm_probe 本地环境性)
+ negative 72 + native slice0/1/2/3/4(5+5+6+6+5)+ battery x3
+ diff_lex 503 + diff_ast 493+10skip + diff_load 12 + diff_tyops
IDENTICAL + diff_sema 6/6 全量逐字节——零回归。

**下一片 = 切片 5(动态聚合)**:数组 `++` 拼接、str 拼接、str/数组
索引、`[a..b]` 切片、record/数组深相等;纯 IR 走 frond_alloc,零新
资产;defer 顺手。(**2026-09-08 切片 5 落地**,见 附十一)

## 附十一:Stage 2 切片 5 落地(2026-09-08,动态聚合)

**范围(附九序列第 5 片)**:str `+` 拼接 / 数组 `++` 拼接 / str 码点
索引 / `[a..b]`+`[a..=b]` 切片(str 码点·数组元素)/ record+数组深相等
(`==`/`!=`,引擎内容相等语义;此前非 str 聚合 == 是**静默 ptr 比较 =
引用相等,语义错**)/ defer(函数级 LIFO 链)。全部纯 IR 走 frond_alloc
(动态尺寸新 `call_alloc_dyn`),零新资产。

**实现**:六件 helper(str_concat / str_index_char 含 1-4 字节 UTF-8
解码四路链 / str_slice 单趟码点偏移 / arr_concat / arr_slice /
deep_eq 双类型递归——**双侧各按己型装载 + 叶级 conv 统一**,peer 型
数组 i32[] vs i64[] 引擎语义相等)+ 接线五处(lower_binary 的 Add-on-str
/ ConcatList / 深相等分派;IndexE str 臂;SliceE 新臂;DeferS 臂 +
ReturnS/ThrowS/`?`-err/函数尾四处出口统一发射)。

**排雷六颗(全为潜伏雷首踩,自举价值高)**:
1. **谓词位次**:UTF-8 解码 is1 用 33(NEQ)应 36(ULT)——多字节
   引码全进 1 字节路返回首字节原值(195/228 探针实证);
2. **循环块结构**:str_concat B 循环 br 自身清零块(k 每轮归零死循环,
   300s 超时 143 假象)+ 双 concat 的 B 循环条件反转(more→done);
3. **深相等 peer 型**:单类型装载把 i32 存储当 i64 读——垃圾比较;
4. **elem TypeVar**:++ 表达式 sema 型 elem 是 unify 绑定的 var,
   `array_parts().a` 裸取不 resolve → llty_for(TTypeVar) 炸——
   `arr_elem_of`(resolve 到底)统一四点;
5. **void 函数双终结符(切片 1 起潜伏)**:tail-null 路径
   emit_implicit_ret 后 flow_stopped 未置位 → 外层兜底再补一发
   `ret void` ×2(Broken module;verify AbortProcess 模式下 IR 不可见,
   verify 前临时 dump 实证);
6. **void 调用三重坑**:带名 call 的 void 值 verifier 拒 / 空串 cstr
   引擎编组 NULL = Twine(nullptr) UB / 名位传 u64-0 = setName(NULL)
   段错误——正解 = **建后清名**(LLVMSetValueName2(v, 非空指针, 0),
   StringRef(ptr,0) 安全);另裸 `0u64` 尾值强转 null Throw,调用方
   match 即崩——必须 `Ok(0u64)` 包裹。

**验收**:`tests/fixtures/native_slice5/cases/`(32-37)退出码全中
41/79/52/54/70/58——str+/arr++(链式)/ASCII+多字节码点(é=233/
中=20013)/str+数组切片含闭区间/深相等(record+数组+嵌套数组 in
record+peer 型)/str 数组拼接后码点访问/defer 函数级链;driver =
`tests/scripts/native_slice5.sh`;CI 挂 slice5 步(slice3/4 上一批已
挂)。语料先行:六件引擎打印 oracle 验值(修正 EXPECT 算术 54/70/58,
37 号模块级 var 改数组盒形态——Back 未支持模块 var)。

**defer 语义近似**:函数级链(块级作用域/循环体内 defer 为函数出口
统一触发,非每迭代);frondc 自身两处均为函数级清理,自举面无伤。

**门禁(2026-09-08 实测)**:functional 98(llvm_probe 本地环境性)
+ negative 72 + native slice0-5(5+5+6+6+5+6)+ battery x3 + diff_lex
509 + diff_ast 499+10skip + diff_load 12 + diff_tyops IDENTICAL
+ diff_sema 6/6 全量逐字节——零回归。

**下一片 = 切片 6(跨模块 + monomorph + std)**:mangled 多模块
lower、call_instantiations 实例重放、方法分派/witness、std 依赖闭包
+C 原语 @extern → declare + C 层 .obj 链接输入 + frond_rt 最小内核
(dlopen/LoadLibrary/argv/UTF-16 桥/spawn);RecordLitE/RecordExtendE
顺手——自举前最后一片大件。(**6a 跨模块 2026-09-10 落地**,见 附十二)

## 附十二:Stage 2 切片 6a 落地(2026-09-10,跨模块)

**范围(附九切片 6 的第一子片)**:非泛型多模块 lowering——依赖用户
模块的函数收集、跨模块裸名解析、跨模块 record/ADT/ctor/match、
ADT 深相等补齐。monomorph(6b)/方法分派(6c)/std+frond_rt(6d)
续后。

**实现(七处)**:①`lower_entry` 加 `dep_mods: List<ModuleAst>`
(Main 侧按 `user_module_paths` 序组装,loader 键=文件路径经
key_to_logical 对齐,入口去重;std/builtin 天然排除);②收集循环扩
多模块(fn_mods/fn_asts 两列:体用己 arena、sema 表键同源);
③`register_fn` 加 `llvm_name` 参数(入口模块保旧 `frond_<名>` ABI,
依赖模块 `frond_u_<路径清洗>_<名>`,'/'/'.'→'_');④`fn_index` 键改
`mod\0name`(模块隔离);⑤per-module 签名 env(`env_of_module`:
module_envs 按逻辑路径,兜底 root);⑥`lower_call` 本地键先行,miss
→ `resolve_cross_module`(func_sig_owners 的唯一 USER 拥有者,
std./builtin. 前缀排除;多主放弃→调用点响亮报错);发射体抽
`lower_call_row` 两路共用;⑦deep_eq 补 k==8(tag 相等 + per-ctor
icmp 链逐字段递归;`Semares.adt_ctor_count` 新增)+ 分派条件放行
kind 8——**修掉 ADT == 静默 ptr 比较(引用相等)的语义错**。

**验收**:`tests/fixtures/native_slice6/cases/`(38-40,目录项目
形态)退出码全中 81/63/144——选择性导入裸调用 + 跨模块 record
构造/字段读 + 跨模块 ADT ctor/match / 三模块链 Main→Mid→Base
(依赖模块内再跨模块调用)/ 跨模块 ADT 带字段 ctor 嵌套模式 +
深相等 + 跨模块 Throw。driver = `tests/scripts/native_slice6.sh`
(slice5 模板 + 目录用例支持:`cases/*/src/Main.frond` 与单文件
并存);CI 挂 slice6 步。语料先行:引擎 oracle 验值(修正 EXPECT
算术 63/144;嵌套 `_` 子模式引擎不支持→绑定变量;多行括号臂体
需花括号)。

**门禁(2026-09-10 实测)**:functional 98(llvm_probe 本地环境性)
+ negative 72 + native slice0-6(5+5+6+6+5+6+3)+ battery x3 + diff_lex
516 + diff_ast 506+10skip + diff_load 12 + diff_tyops IDENTICAL
+ diff_sema 6/6 全量逐字节——零回归。

**余片**:6b monomorph(call_instantiations 实例重放 + 类型代换
expr_ty 查询)、6c 方法分派/witness、6d std 闭包 + C 原语 declare +
frond_rt 最小内核(dlopen/argv/UTF-16 桥/spawn)+ RecordLitE/
RecordExtendE 顺手。

**6b 部分落地(2026-09-10 同日,单层泛型)**:
- **机制**:Mono.return_type 对直接类型参数是**未解占位 Adt("T")**
  (镜像 dump 自留口径,diff 实证)——参数/返回专用解析器
  `inst_param_ty`(声明 type_params 按名位置映射 type_args;T[]/T?
  递归;非参数名回落 resolve_tn_flat 空实参;TGeneric=6d)。实例注册
  `register_instance_fn`(LLVM 名 `frond_g<id>_<fn>`);发射期逐实例
  **重放→立即 lower**(`Mono.replay_instance_types` 新 pub 包装——
  多实例共享全局 expr_types 体键,镜像 check 期重放末者胜,按
  instance_id 序回放与 check 终态一致);重放内嵌套实例化经
  `register_pending_instances` 动态拾取;调用点路由
  call_instantiations(键=调用表达式 id)优先于常规解析,空实参
  实例不进行表(与常规收集重复,未命中自然回落)。
- **排雷三颗**:①register_instance_fn 与收集循环**双份 push 列**
  = 行错位 → 路由到错误实例参数(int→str conv 假象);②泛型调用
  点的数组字面量型 elem 是 unify 绑定 var,`array_parts().a` 裸取
  不 resolve 即炸(lower_array_lit 补 arr_elem_of);③赋值不放宽
  ModuleAst→ModuleAst?(if-else idiom)。
- **验收**:41_mono_basic(同泛型函数双实例 idt<i32>+idt<str>+
  pick+ksize)退出码 73;n1/n3/n4 系列单实例探针全中。
- **6b 嵌套泛型收官(2026-09-10 同日,42 号回归)**:镜像 inst 模式
  重放**跳过 unify**,内层泛型调用点/其派生表达式的记录型是**悬空
  fresh var**(非绑定,deep_resolve 无效)。四处物化/直通:
  ①`expr_ty` 查询位——记录型 open(悬空 var 或壳内 open elem)且
  call_instantiations 路由到泛型实例 → `instance_call_ret` 用声明返回
  AST + type_args 经 inst_param_ty 重建(cur_ftab 模块 var 供查询);
  ②lower_binary——操作数基准型 open 的非调用表达式(如 ++)以 LHS
  物化型为基准;③数组字面量 elem open → 从首元素物化型派生;
  ④conv_value——单侧 open 的同形聚合转换直通(open 侧是悬空记录型
  无布局信息,值本体已按具体派生构造)。
- **验收**:42_mono_nested(泛型调泛型 both→wrap / sum_len→both,
  T[] 数组形参双实例)退出码 7;n1/n2/n3/m4 探针链全程实证。
- **门禁(2026-09-10 实测)**:functional 98(llvm_probe 本地环境性)
  + negative 72 + native slice0-6(5+5+6+6+5+6+5)+ battery x3 + diff_lex
  520 + diff_ast 510+10skip + diff_load 12 + diff_tyops IDENTICAL
  + diff_sema 6/6 全量逐字节——零回归。6b monomorph 全部收官。

**6c 方法分派落地(2026-09-10 同日)**:静态已知接收者型的直接分派。

- **机制**:方法以独立函数发射(`frond_m_<Type>_<method>`,trait
  默认 `_d` 后缀;canonical 名注册 `method_rows`);**parser 已注入
  `this:TThis` 首参**(Md.params[0])——TThis 解析为接收者型
  (make_adt canonical),调用 = [recv] + args。trait 默认经 TDefI
  实例注册,默认体从声明 trait 的模块 TraitDeclD 取(TTDef/TMS 无
  body 持久化);自有覆写优先(method_rows 键已占即跳过)。隐式
  this 两路:expr_types 的 `this` 记录型是占位 Adt("This") →
  expr_ty 查询位以行 this 型(cur_this_h)替换;方法体内裸字段
  (x → this.x,槽装载后字段定位)与裸方法调用(sum() → this.sum())
  回落。InhI 继承方法 = 6d 面(布局前缀未建模)。
- **排雷三颗**:①默认行 fn_mods/fn_asts 误推 entry——默认体在
  声明 trait 模块的 arena,错 arena = 乱解释(主症状:main 的
  'identifier not a local' 假象 + 默认值乱);②this 槽是 alloca,
  字段定位/隐式调用前须 load;③TMS/TTDef 的 get 是数组索引非
  List.get;④Md 兜底 ctor 的 body 字面量须 0i64(i64? 参数)。
- **验收**:43_methods_basic(自有方法:record/ADT、带参、this
  match、方法调方法、链式、跨模块)= 52;44_trait_default(TDD:
  默认体经 this 虚派到覆写 name,默认调默认)= 132。
- **门禁(2026-09-10 实测)**:functional 98(llvm_probe 本地环境性)
  + negative 72 + native slice0-6(5+5+6+6+5+6+7)+ battery x3 + diff_lex
  524 + diff_ast 514+10skip + diff_load 12 + diff_tyops IDENTICAL
  + diff_sema 6/6 全量逐字节——零回归。

**6d 第一段落地(2026-09-10 同日,rt-as-IR + @extern 直调)**:
- **rt-as-IR 裁决(替代 C 文件 + clang 编译)**:工具链资产 bin/ 仅
  lld 无 clang——rt 以 IR 由 Back 直接发射,CRT/kernel32 走 declare
  (链接视图已含),**零新资产、零新子进程、五平台天然**。
- **frond_alloc 换 malloc(策略丙)**:调用点不动;bump 1GiB 段退役;
  OOM → trap。现有全部语料(重分配 concat/methods)免改通过。
- **rt 写原语**:`__stdout/__stderr_write_raw(ptr, i64) -> i32`
  (C ABI = str 的 data+len 双参形态);Windows = kernel32
  GetStdHandle(-11/-12) + WriteFile(已链接),Unix = write(fd)。
  控制台 ANSI 渲染为 v0 已知近似,管道/重定向精确。
- **@extern 调用链**:extern_llvm/fts/params 三表;调用点先于跨模块
  解析(builtin 拥有者不在收集面);str 参数 ABI 编组 = 块地址拆
  (data@8 gep + len@0 load)。镜像 sema 放行 builtin extern 直调
  (@internal 拦截是引擎 IR 层规则;native 面合法)。
- **验收**:45_rt_print——两行 stdout 原生打印 + exit 40(rt rc
  语义 0+0+40);**首个带 I/O 的原生程序**。
- **门禁(2026-09-10 实测)**:functional 98(llvm_probe 本地环境性)
  + negative 72 + native slice0-6(5+5+6+6+5+6+8)+ battery x3 + diff_lex
  525 + diff_ast 515+10skip + diff_load 12 + diff_tyops IDENTICAL
  + diff_sema 6/6 全量逐字节——零回归。
- **6d 余段**:可达性驱动发射(worklist——builtin 全量收集的前置;
  ~230 个 __ 包装含不支持构造,须只编可达)+ Console 模块函数收集
  → println 全链(泛型 6b + Throw 4 + extern 6d)+ repr() intrinsic
  (str 恒等 / 整型 itoa rt 函数)+ InhI 继承(布局前缀建模)+
  RecordLitE/RecordExtendE。

**6d 第二段落地(2026-09-10 同日,println 全链)**:
- **worklist 可达性发射**:mark_reachable(调用/方法分派位标记)+
  游标队列;builtin 全量原型注册(~230 个 __ 包装含不支持构造——
  容错注册跳过,不触即不编)。
- **排雷三颗(全为链路型)**:①用户 fn 收集循环**双份 push**(容错
  match 内 + 旧无条件 push 残留)= 列错位 8 行 → 参数/体张冠李戴
  (症状:alloca 名 %s 而非 %x + load ptr from i32 槽);②Throw 的
  TGeneric("Throw",[V,E]) 在 inst_param_ty 须双参递归(resolve_tn_flat
  丢实参 → 裸 TGeneric 不可 lower);③itoa 谓词位次:slt=40(误 48)、
  循环 ugt=34(误 uge=35 = 无符号恒真 → 死循环)。
- **println 全链六件合璧**:泛型实例(6b)→ x.repr() intrinsic
  (6d:名字即分派;str 恒等 / i64 itoa rt 函数)→ __stdout_write
  (跨模块 builtin 6a+6d)→ Throw 直落(4)→ @extern rt(6d 一)→
  WriteFile(kernel32)。VoidLit/Ok(void)→ i64 哑 box;Throw→Throw
  conv 恒等(V 位 void/open 哑差不改 ptr)。
- **验收**:46_println = "hello native println"/"42"/"-12345" 三行
  原生输出 + exit 41。
- **6d 排雷续(29 号回归修复)**:void/open 容错的 open 判定误用
  `ty_kind == -1` — **TAdt(record)也是 -1** → 所有 record 载荷的
  Ok(P(...)) 走了 i64 哑box 路径(症状:字段全 0 + 无 throw 包装;
  named+packed 组合死循环 = 乱 IR 级联)。修 = `TTypeVar(_) |
  TUnknown` 显式 match(非 -1);三处同步(Ok 特判 / throw_arm_
  layout / PropagateE)。★Frond 无 matches! 宏——Rust 语法直接
  编译错,须展开为 match 表达式。二分链 p2→p9→pa 定位:p9
  (packed 单独)= 16(应为 30)即破案点。
- **6d 余段(收窄)**:InhI 继承(布局前缀建模)+ RecordLitE/
  RecordExtendE + eprint/stdin + std 泛型(List/Map 族)。

## 附五:S2c 导入优先级格 + 裸名多主零静默(2026-09-04,用户裁决"地基要稳")

**动机**:import .{} 现代化试点(Relate 转换)三连炸,暴露语言层三处
结构性缺口——选择性导入此前在 Frond 是半残特性。

1. **(a) 显式条目权威**:sema 在裸调用点记裁决(`bare_call_targets`,
   键 = module_expr_key 哈希,与 call_instantiations 同族);IR 裸调用
   分支(`compile_call`)按完整 mangled 键直绑,绕过被 std 名(`get`×3)
   争夺的裸键绊线——此前 0b 段把别名注册记成冲突键,绊线推翻 sema 已
   裁决的显式绑定。
2. **(b) 裸名多主零静默**:裸名自由函数经 **root(std 预declare)层**
   解析且全局 ≥2 主人 → 调用点响亮错("ambiguous call '{n}': [owners]
   — qualify the call or bind it with a selective import")。守卫 =
   `lookup_at_root`(仅 root 层命中才算歧义候选:局部/本模块定义/显式
   条目都在更深层,天然胜出不报)。此前 env last-writer-wins 静默,
   饿死下游推断,在模式位爆出误导性歧义错(离病因一个间接层)。
3. **env 层优先级**:选择性条目 `define` 失败(与同层既有绑定撞名,
   define 首胜制)→ `redefine` 覆盖。治 frondc 镜像管线 dep 模块共享
   root env 导致条目丢失、裸名回落 std 绑定的病灶(引擎管线每模块
   独立 env 故沙箱不复现——镜像管线结构性差异首次显形)。

**落地**:引擎 Sema.rs(lookup_at_root/bare_call_targets/purge)+
ModuleEnv.rs(redefine)+ CallInfer.rs(裁决记录+零静默)+ Builder/Call.rs
(bare_sema_target 直绑);镜像 Envs/Modenv/Semares(import_alias_owners)/
Infer 四文件同步(消息逐字节一致)。Rust/Java/Haskell 同款"显式>glob"
格 + 零静默教义从构造器位扩展到自由函数位。

**验收**:三场景沙箱(显式胜出跑通/无显式响亮错带主人清单/单主无扰);
**Relate 全量转换(Arena 23 名 + Semares 2 名,130 处)树内加载绿**;
顺带咬出并修 Tyops 裸 `resolve`(与 std.net.Dns 撞名,潜伏 last-wins
赌对——零静默的第一笔红利)。全量门禁见当日电池。

**import{} 现代化装备**:`tmp_probe/selimport.py` 转换器(std 黑名单
防线保留为冗余保险;根修后 get 类不再需要);推广(Infer 848 处/
Parse 的 Lex.* 零参值)待后续批次。

## 附三:地基整固三裁决(2026-09-04,用户拍板"地基要稳")

1. **libs/llvm 模板删除**:零 import 消费者(README 自述拷贝式分发,
   实际三个消费者全是物理副本)、三副本发散、val 参数名 bug 潜伏数日
   无人发现——backend/Llvm.frond 自此为唯一来源(头注释已改);
   llvm_bind/llvm_probe 测试副本随引擎验收各自演进。
2. **"record 字段软类型"引擎洞销案**:handleprobe 五形态实证(单字段/
   混合家族/嵌套/List 装载/跨函数的 record 字段读回值过 LLVMTypeOf 真
   FFI oracle 全绿;迭代拼接 str 过 lookup 亦绿)——洞已被引擎侧治好
   (marshal as_i64 仍只认硬 tag,治在字段读取侧:读回重建硬 tag)。
   三副本 Llvm.frond + Back.frond 头注释过时段落全部更正;Back 的
   Map/参数状态架构保留(已验证稳定,非性能约束),record 形态解禁。
3. **&&/|| 短路化**(切片 1 前置件):引擎语义 = 短路(Bug #38,
   `lhs && rhs => if lhs { rhs } else { false }`),切片 0 的非短路
   bitwise 形态只在纯操作数下等价——Back 改为槽位 + cond_br 真短路
   (短路值预存槽、RHS 仅在求值侧覆写、merge load 收口);fref(函数
   句柄)穿进 lowering 链(新块创建需要,控制流切片的地基)。验收:
   05_shortcircuit(x=0 除零守卫,非短路实现原生侧 = sdiv 0 崩溃)
   exit 1 ✓,native_slice0 5/5。

## 附:Stage 2 首刀记录(2026-08-31,macOS 首绿)

`tests/functional/llvm_probe` 首跑绿:Lib.open 资产 llvm.dylib →
LLVM-C 装配 main ret 42 → verify → TargetMachine emit .obj → spawn
资产 lld(macOS 零库链接,-platform_version 11.0)→ 跑产物 →
**退出码 42**。绑定层新增 const_int / function_type0(空数组不可
cbuf 编组,零参走 NULL)。**引擎边角立案(未深挖)**:顶层全局 `val`
以模块函数调用作初始化器(读 Env)→ 运行期全局初始化 panic(index
out of bounds len 0)——探针规避为函数体内解析,根因待查。

## 三f、片5 进展与阻断(2026-08-29 夜)

**已落地**:镜像 `sema/Mono.frond`(monomorph 收集器 + 实例化模式重放 +
trait 三验证器/收集器);Check 步骤 8/10a 接线;stats 七计数 Sdump 真实
化。**ctor_name_clash 语料七计数已逐位相等**(27409/431/20/0/62/436/0);
PQ 最小泛型语料实例序与引擎逐位一致(连 resolve 展平器 ret=T 怪癖都
一致)。`resolve_type_key_in` 带点拼写规范化(std.collections.List → 裸
键)为片5 连带镜像修复。

**片5 连带引擎根修(已落地+回归)**:HM turbofish 消费——`Expr::Call/
MethodCall` 的显式类型实参此前被 HM 推断忽略(`List.empty<str>()` 的 T
悬空 → 9.4 默认成 void → 下游臂绑定静默退化 void 接收者)。修 =
`instantiate_fn_type_with_hints`(按遍历序把前 N 个未绑变量绑到提示类
型;rigid 拒 unify,故在实例化替换层而非 env 签名上绑);接入 Call 通用
路径 + 方法糖 0a/0b/Path-0。验收:探针全绿 + functional 94 + negative
64 + 语料 sema 0 错 + perf 同量级。

**片5 阻断(已销案,2026-08-31 复核)**:~~check 运行期 compute_match_
fallback panic / 三层 match 臂嵌套 × 循环内副作用语句静默丢失~~——
引擎侧修复随 2026-08-31 提交(75a6496,EngineCore/Schedule/Subgraph/
Frame 一组)落地。**复核证据**:① `stmt_drop_repro.frond` 输出正确
(post×3 + 3 + len=3,缺失的第二条 println 回归);② arithmetic 语料
**全节差分逐字节一致**(含 `! monomorph 823`/`! inherited 53`——片5
最后一块 inherited 对齐达成;引擎侧基准同数);③ 默认 6 语料 5/6 绿,
warnings 节随全节比对通过(原「镜像多 6 条 unreachable」残差同灭)。
**新立案(未复现)**:qualified_types 的一次 checkmany 运行 >900s 被
超时杀(stdout 空致差分假红),此后同命令 3/3 复跑全绿(39s/2551 行,
warm cache)。疑引擎 async 调度偶发挂起,观察项:再遇即取 stack/计时
分段定位,不阻断主线。
**引擎网络栈观察(2026-08-31 CI 首跑)**:llvmfetch(自研 TLS 1.3 客户端)
在 runner 上不稳——macos 报 `tls: alert level 2 code 20`、linux-x64
跑 fetch **段错误**、linux-arm64 则全成(132MB 校验过)——同代码三种
命运,环境相关性大。CI 预取已改 curl 直拉(零依赖);std.tls 作为承重

**引擎 UAF 双案(2026-09-01,ASAN -Zbuild-std 全插桩定位)**:
**案一(已修)**:marshal 的 AoS 序列化臂序错误——u8-blob 兼容臂先于
同 tag 臂,`arr[0..n]` 切片(i64 元素向量)被逐元素压成 N 字节,而配对
Int 槽仍是元素数 → C 侧 `count*8` 读短缓冲(ASAN:8 字节区读 24;
无 ASAN:静默读相邻堆=错值,或 SIGSEGV)。**CI 的 await 家族抖动 =
此 UB 的显形**。修 = 臂序对调(Value.rs,5bedfdc);List.from+push
最小复现、collections/edge_async ASAN 全绿、全套件回归绿。
**案二(立案待修,帧链所有权)**:reuse chain(parent_frame_ptr/
root_frame_ptr 裸指针)与帧池无生命周期契约——祖先帧完成即池化/
溢出释放,裸指针悬空(get_value_by_global 走链 UAF,双栈:
释放 T13 process_frame→release vs 访问 T17 run_frame_nodes)。
引用计数+墓园方案已原型并验证到「确定性复现」(20/20),但残余
**帧所有权双释放**(complete_and_wake_caller 与 worker 执行竞争,
账本正确走完仍崩)超出链会计范围——已回退,黄金资产留下:
ASAN 全插桩构建命令、审计日志法(chain_refs 流水+差分平衡分析)、
确定性触发(await_loop 单跑 ASAN)。根治方向:release 前清链 +
set/drop 全走 set_chain_ptrs 单点 + acquire 重置账本(三者缺一
不可,均已原型过),最后一步需吃透 complete_and_wake_caller 的
所有权语义(引擎作者领域)。
件(apps/llvmfetch 立身之本)的跨环境稳定性立案待查。**tls13_handshake
在 CI 四平台全红**(localhost:47631 自建监听,`tcp connect failed`;本地
同代码全绿)——引擎 async/网络在 runner 环境起不来,同族问题,暂以
NOCI 标记跳过(tls13_handshake/NOCI,修好引擎后删)。

**Stage 2 绑定层铁律(2026-08-31 CI 首跑教训)**:LLVM 的错误消息出参
必须给真缓冲——`LLVMTargetMachineEmitToFile(..., 0u64)` 在失败路径
(如输出目录不存在)直接**段错误**而非返回错误。emit_to_file 四份副本
已全部加固(err_buf + read_cstr + DisposeMessage);套件侧的 out/
目录由 runner 预建兜底(out/ 在 .gitignore,CI 全新 checkout 没有)。**1E 剩余 = 终局验收三级(2026-08-31 全数达成,Stage 1 收官)**:
① 双跑等价:functional 93 过/2 平台跳过(ffi_lib、crypto_primitives
= Windows 特化夹具,PLATFORMS 声明)+ llvm_bind 待平台资产(CI 预取
覆盖)+ **negative 64/64**;② **std 全库自检**:checkmany 全部 128 个
std 文件作入口,0 错误,唯一 warning 为 frondc 自身 module/Toml.frond
的 unreachable(历史良性);③ **apps 语料**:editor + llvmfetch 引擎
vs 镜像 sema-dump 逐字节一致。计时备注:std 自检单进程 66 分钟——
parse 已跨入口共享(std_cache/AST 盘缓存),sema 检查环仍逐入口重跑;
跨入口 sema 共享 = 优化积压项(与 §7 IntMap 同族,不阻断)。

**镜像侧已知残差(已销案,2026-08-31)**:~~`! warnings` 镜像多 6 条
"unreachable match arm"~~——随引擎修复(75a6496)消散:默认语料全节
比对(含 warnings 节)逐字节通过。

## 附六:立案未决(2026-09-05,随 import{} 推广收尾)

1. **type_name 方法糖隔离**:Infer 的 `Relate.type_name` 永久限定——
   std Err trait 方法 `type_name()` 与自由函数名共享方法糖回落空间
   (`recv.m(args)` → `m(recv,args)`),条目 redefine 后 std 接口方法
   解析被带偏,症状为隐式 this 字段连锁失联。根修方向 = 接口方法名
   与自由函数名的空间隔离(或 witness 装配不消费 root 层函数绑定)。
2. **async 裸调用 null**:裸化 Main 的 native_entry(async fun)内,
   经选择性条目绑定的裸调用(`read_text`/`parse_module_text` 族)
   运行时返回 null(Error|Ok scrutinee 崩);限定形态正常。async +
   bare_call_targets 裁决 + IR 绑定的交互失效,复现资产:selimp 沙箱
   + 怪胎 Main(b3d6717 版,已以 b638666 限定版回退规避)。
3. **环境白名单收缩(方案 B,进行中)**:root 预declare 的 330 名
   隐式 glob 收缩为原语白名单,见附七。

**清账结论(2026-09-06,未决 bug 清账会话,案 1/案 2 双销案)**:

- **案 1(async 裸调用 null)——销案,真凶更深**:复现剖析 = 沙箱
  混态树(旧 b3d6717 引擎 + 新树 CallInfer)的 **arity mismatch**:
  裸调用 `read_text(path, loader, null, false)` 4 实参喂 5 形参
  函数,尾部形参静默绑 null 流入 match → [ctor-match-null]。
  自洽树验证原 bug 引擎已治好;但暴露真缺陷 = **跨模块函数调用
  arity 校验缺失**:本地调用有校验,ModuleRef 调用路径对实/形参
  数量不匹配零静默放行。根修(CallInfer):arity_error 覆盖
  ModuleRef 路径 + 构造器超参报错(保留构造器少参默认构造语义;
  本地非构造器少参仍为合法部分应用)。负例三件落 negative:
  arity_module_call / arity_module_over / arity_local_over。
- **案 2(type_name 方法糖)——复测已愈,销案**:typename_probe
  沙箱双形态(零导入 / 选择性导入 `Str.{starts_with}`)验证
  i32/str/f64 的 `type_name()` 全部正常——方法解析走 trait
  witness 表,与 import 形态无关;上表 1 的"方法糖回落带偏"症状
  当前引擎不复现。
- **门禁硬化(同会话落地):ctor-match-null 警告升三态门**
  (引擎 Builder/ControlFlow + Compute,自举侧 Back 不涉):
  - scrutinee 静态**非空**(expr_type_handle → handle_nullability,
    resolve 后非 TypeVar/Unknown):挂 nonnull_assert 为 ctor_match
    的 inputs[1] 调度依赖(and_bool inputs[2] 同模式)——运行时
    null 即 "non-null assertion failed" panic 硬失败,不再静默
    判 false;
  - 静态**可空**(T?):挂 is_null 探针为 inputs[1]——Compute 侧
    见 input_count>1 即安静 fall-through(合法 null 的警告噪音
    清零);**零新元数据、零序列化改动**:input_count 天然随 .fndo
    Inputs section 持久化,规避 bool_flag 家族的 SectionKind 端
    成本;非空位则依赖调度序(assert 先于 ctor_match 执行)在
    序列化前后同构;
  - **unknown**(typevar/嵌套 record field):单输入,保留警告兜底。
  嵌套 ctor 子模式 nullability 由父 ctor_def.field_types 逐位推导;
  守护套件 pattern_nullable_gate(Tree? 上 null/Node/Leaf 判别 +
  嵌套 ctor 子模式 + 纯 null/变量混合,15 check)零警告全绿;panic
  路径以临时 paranoid 探针(强制非空判定)端到端复现后移除。
  **回归门**:functional 97(llvm_probe 缺 llvm.dll 资产,环境性,
  native slice0/1 门同因留 CI)+ negative 69 + lex 483 + ast 473
  + load 12 + sema 6 + tyops 77 全绿。

**清账复审(2026-09-07,案 1 rev.2 = 09-06 销案补完)**:

- **复审发现 09-06 修复有洞**:N4 最小复现(选择性导入 5 参函数喂
  3 实参、进 match scrutinee)在含 4862f74 修复的引擎上**仍崩**
  (non-exhaustive match,与 frondc native_entry 崩溃同签名);根因
  = bare import-entry 调用经 `infer_expr(callee)` 落入通用段的
  **"默认柯里化"**(Bug #160 裁决)——少参被静默定型为部分应用
  `Fn(余参)->ret`,但**任何层都未实现部分应用**:IR 对本地急切调用
  缺参垃圾补槽(N1 侥幸值)、对跨模块裸调产 Partial 值死于 match
  scrutinee(N4/b3d6717)、真用"部分应用"则 panic "input is not
  callable"(N2)。原立案的 async 是红鲱鱼(b3d6717 的 native_entry
  恰好 async;真凶 = lower_entry 在 b638666 4→5 参演进)。
- **裁决 rev.2:默认柯里化废止**——under-arity 双向硬错误,与
  ModuleRef/超参口径统一(全库零依赖隐式柯里化,`fun &` 仅接口
  方法标记);构造器少参零填充默认构造语义保留(`!callee_is_ctor`
  豁免,正常+实例化两段)。
- **N5 同族洞**:调用非函数局部值(如 `arr(1)`)正常段静默落回退
  → 运行期 panic。补"cannot call non-function value of type '{}'"
  硬错(镜像实例化段已有,正常段新增);**豁免名单** = TypeVar/
  Unknown/Never(泛型高阶/死代码)、ModuleRef(裸尾段匹配构造器
  callee,Sdump 的 `Mono(...)` 形态由 IR 构造器表真派发)、
  **Adt 型 Ident callee**(Bug#69 家族:零参 ctor 注册为值、函数
  局部类型不进全局构造器表——`Marker()`/`Leaf()` 须与裸值等价,
  edge_adt 回归实证后定豁免)。
- **镜像同步**:Infer.frond 五处(实例化 ModuleRef/通用 + 正常
  ModuleRef/通用 + 回退前非 Fn),诊断文案与引擎逐字节一致
  (三新负例双引擎同串实证);旧镜像零元数诊断(合法语料差分
  不暴露的错误路径 parity 缺口)就此补齐。
- **负例三件**:arity_bare_import_under / arity_local_under /
  call_nonfn_value(negative 69→72)。
- **门禁(2026-09-07 实测)**:functional 98(llvm_probe 环境性)
  + negative 72 + native slice0/1(5+5)+ battery x3 + diff_lex 498
  + diff_ast 488+10skip + diff_load 12 + diff_tyops IDENTICAL
  + diff_sema 6/6 全量逐字节 + b3d6717 怪胎 Main 复验 =
  `lower_entry expects 5 argument(s) but got 4` 编译期诊断。

## 附七:方案 B 环境白名单收缩落地(2026-09-05,@export 制)

**裁决演进**:命名模式白名单(println 族+`__` 前缀)→ 用户改判
**@export 声明位 opt-in**(显式意图 > 调用侧启发式;环境面 = grep
@end…@export)。分层律:builtin = 打标特权层(@export/@extern/
@internal/内嵌 C),std = 干净公共库,用户 = 显式导入。

**引擎改动**(ModuleEnv + 镜像 Modenv 同步):
1. **@export 门**:predeclare 的 root define(全局裸可见)只收
   `@export` 且 **builtin-only** 生效(用户/std 打标无效——环境面是
   引擎特权层声明)。历史"全函数 root 双注册"(330 名隐式 glob,
   `get`×3/`parse`×20 碰撞面、方法糖污染源)终结。
2. **用户模块导入成员再输出**("导入即真理"裸名臂):`import Helper`
   把被导入模块 env 的全部绑定 define 进导入方 env——1C 语义
   (整模块导入成员裸调)从 root glob 意外承载改为导入语句显式承载;
   selective 条目随后的 redefine 保持 S2c 优先级。

**环境面清单**(@export 打标):Console 六件套 print/println/eprint/
eprintln/scan/scanln + for-in 基础设施 iter/str_iter + 方法糖回落
家族 sort/sort_by/is_sorted/lower_bound/upper_bound + hash_iter +
builtin 内部互调 `__` 族(~230 名,crypto/encoding/mem/sort/io/net/
os/raw 全家)。

**欠账清偿**:std 层 Power/TcpListener/UdpSocket 选择性导入
(ldexp 族/tcp_close);frondc 层 import_sweep 一轮清零(8 文件);
Back 的函数签名查询改 module_envs(root 面退役连锁)。

**坑**:①引擎 std 是 StdlibEmbed 编译期内嵌——改 std 源必须重编
引擎(沙箱跑旧烤入副本,@export"不生效"假象);②方法糖回落
(`recv.m()`)吃自由函数环境面——iter/sort 家族断供即"no method";
③`_impl` 后缀/math 家族跨模块裸调是 std 禁裸名欠账。

**门禁**:slice0/1 各 5/5 + functional 98 + negative 66 + tyops 77
+ load 12/12 + sema 6/6(镜像同步字节精确)+ ast 469/0。

**Phase 2 落地(2026-09-05 同日收官,三件)**:
1. **@internal 规则化**:语义考证 = "用户禁调"(IR internal_funcs 注册
   表 + internal_access_blocked),非名字空间机制;std 本体的 @internal
   全打在模块内**私有**函数上 —— 改由引擎规则统一覆盖:IR 注册条件
   = @internal 标记 **或** std 层非 pub 函数(选择性导入绕过可见性时
   的守卫);builtin 的显式 @internal 保留(声明位形态)。
2. **std 本体 @internal 清标**(18 文件,hash/fmt/math/io/net/os/time
   全族,~300 处):私有函数天然受规则保护,逐个打标是历史欠账。
3. **tag 权限收紧 builtin-only**:@internal 在 std 打 → 报错("reserved
   for builtin modules",引擎+镜像同步;负向 EXPECT 两处同步)。

**诊断细化三件(同日,镜像字节一致)**:
1. 调用位:callee 裸名全 miss(环境/函数表/ctor/模块尾段/方法体豁免)
   → `undefined function 'X'`(值引用位仍 undefined variable);
2. 模块尾段:`Fmt.dec(42)` → `module 'Fmt' exists — import it
   explicitly (e.g. import std.core.fmt.Fmt) to use its members`;
3. 类型位(check 期局部注解):裸类型未注册/非 pending/非内建 →
   `unknown type 'Fmt'`(此前**静默造 Adt 零静默缺口**;仅局部注解位
   ——predeclare/早期 populate 的签名解析存在合法跨模块时序前向引用,
   不能上解析干道,Raw 签名的 IOError 即此)。
★坑:①调用位前置曾误杀隐式 this 方法调用(`base(...)` 方法糖回落
形态)——方法体内放行;②cargo exe 锁(后台 frond 进程)致 build
静默失败,旧二进制跑出幽灵红。

**Phase 2 留档**:std 本体 @internal 族(hash/fmt/math 可搬 builtin,
io/Path 族类型反向依赖不可搬)迁 builtin + tag 权限收紧 builtin-only
(现 std+builtin 都可打);~~未导入裸调错误文案升级~~(**已落地,当日
收尾**):Ident 解析失败不再一律"undefined variable"——函数注册表 →
构造器表逐级回查,存在即给归属+导入导引(引擎 ExprInfer + 镜像 Infer
同步,文案逐字节一致):`function 'fnv1a' exists in module(s)
[std.core.hash.Hash] — import it explicitly (e.g. import
std.core.hash.Hash.{fnv1a}) or use the qualified form`;真未定义名维持
原文案;方法糖位(模块短名 recv)提示待补。负向 EXPECT 同步更新,
66/66 + sema 差分绿。


## 附十四:Stage 2 切片 6e 落地(2026-09-11,std 泛型 List/Map)

**范围(附九 6d 余段的最大一片)**:std 泛型聚合(List<T>/Map<K,V> 族)
进 native 收集面,frondc 自身 395 处 `.push()` 的静态特化基础全链打通。

**七层接线**:
1. **std 进收集面**(Main.frond native_entry):builtin.* 之外加 std.*
   (全量原型注册,可达性发射只编被调者);
2. **命名空间接收者路由**(Back MethodCallE 前置):sema 的
   module_func_recv_exprs 标记 + call_instantiations 实例分派,非泛型
   经 ftabs 路径后缀定归属(recv 不进实参);
3. **inst_param_ty TGeneric 泛型族**:Throw/Channel/Async/Lazy/Atomic/
   Sender/Receiver/ForeignFn 结构变体 + 用户泛型聚合 →
   make_adt(canonical, 代换实参)(镜像 Typeast 口径);
4. **泛型聚合代换**:agg_kind 7/8 收编带实参 TAdt;ty_key 实参感知
   (rg/dg 前缀);record_layout_h(声明字段 subst_ty 名字键代换——
   探针实证声明位 T = TAdt("T") 名字占位);rec_field_place 字段型
   代换;
5. **泛型构造器实参反推**(lower_call):声明字段 vs 实参类型结构对位
   (infer_ctor_args/match_param_arg)→ 代换布局 → 产物型入
   ty_overlay(expr_ty 先读 overlay);
6. **泛型类型方法实例**:meth_inst_rows(键 = canonical+method+ty_key)+
   调用点动态注册 + Mono.replay_method_body(this 绑定重放,TThis/裸
   字段经 this 解析;check 期只为自由函数建实例,类型方法共享软值
   subgraph);隐式 this 字段读写(AssignS/Ident 双侧);
7. **rt 原语 + intrinsic**:emit_rt_mem_copy(18 元素型,memmove 方向
   语义)/__hash_key_str(FNV-1a+"-0"归一)/__fnv1a_bytes;type_name()
   静态折叠 display;bytes() str→u8[] 块拷;StrInterp(str/整型);
   动态数组 fill([v, ..n] 运行期计数);void→void 恒等 + void 值槽
   i8 哑位 + 数组↔str/数组→数组 as 直通(type_name 分派死分支)。

**毒型家族(本片最大敌情)**:check 期方法糖调用的共享刚性槽末者胜
——泛型类型声明的 rigid T 被覆写(TVoid 实证:List 构造调用点全局型
= List<void>),引擎软值不可见,native 静态布局受害。对策三件:
实例分派行返回型 overlay(lower_call_row/lower_method_call)+
ValDeclS 先 lower 后取型 + 毒型回落绑定槽型(expr_ty_sl;ty_generic_
poisoned 深度上限 8)。**实参归一**(sanitize_arg,出口在 expr_ty/
conv_value 双侧):sema 局部声明/实参位的嵌套泛型聚合是 arena
TGeneric 句柄,消费面统一 TAdt。幻影实例门(args_resolvable):泛型
体内嵌套调用的 check 期实例(实参 = 未解类型参)不注册。

**验收**:native_slice6e(47-52)6/6——List 全方法面(i32/str/嵌套/
增长翻倍)+ Map<str,i64>(set/get/has/remove/len,hash str 快路)。

**留洞(下一片 6f 面)**:Map keys()/values()——泛型体内
`List.empty()` 的 T 需 HM 回流绑定(check 期实例实参 = 未解 K,重放
一次性推断不跑 solver);i64 键走非 str 哈希路径(itoa+bytes 链已
lower,待语料);entries() = HashIterator builtin。InhI/RecordLitE/
stdin 维持 6d 余段原位。

## 附十五:Stage 2 切片 6f 落地(2026-09-11,keys/values 使用驱动绑定)

**范围(附十四留洞的收口)**:Map keys()/values() + i64 键(非 str
哈希路径)native 全通。零引擎/零 check 改动(纯 Back 层)。

**机制:幽灵实例的使用驱动派生**(derive_ghost_instance 双路):
泛型体内零参工厂(`List.empty()`)的 check 期实例实参 = 未解占位
(TAdt("T")——**工厂自身参名**,非外层类型的 K;真值由 check 期
solver 回流绑定,Back 的一次性重放不跑 solver)。两路:
- A 名字代换:占位名 ∈ 当前发射行类型参(K/V 形)→ subst_ty;
- B 使用驱动:扫当前函数体(scan_stmt_uses/scan_expr_uses 递归
  BlockE/If/While/Match/表达式树)找绑定名(init == 调用 eid 的
  Val)上的首个带参方法调用(out.push(x))→ 方法声明参型占位
  (inst_param_ty 空 tps/args)与实参类型 match_param_arg 对位闭包
  接收者开放槽 → 工厂 rt 模式对位(match_rt_pattern,声明 rt AST
  vs 闭包后句柄)→ 工厂类型参绑定 → 虚拟实例(追加进
  monomorph_instances,id = 追加索引,register_instance_fn + 行簿记
  + worklist 重放机制全复用)。

**配套**:发射行类型参绑定上下文(cur_gtps/cur_gargs,自由函数
实例行 + 方法实例行 -2 置位)+ 体上下文(cur_body_eid/cur_farena)。
标量→str as 死分支哑值(conv_value:**原值直通会令下游 str 拆参
gep 到标量,LLVM verifier 拒——必须 null ptr 类型正确哑值**)。

**验收**:native_slice6e 扩至 8 案(53_map_iteration/54_map_i64_keys)
全绿;探针 t0-t6 13/13;i64 键全链 = "{"{k}"}" itoa 插值 +
normalize_zero + bytes + fnv1a_bytes rt。

**余洞**:entries() = HashIterator builtin(迭代器协议未入 native);
InhI/RecordLitE/stdin 维持原位。
