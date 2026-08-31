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
| **Stage 2** | frondc 的后端模块用 std.llvm(Frond 代码调 LLVM-C)lower AST→.obj,内嵌 lld 解出后 spawn 链接(零宿主工具链,见 四) | **探针首跑绿(2026-08-31,macOS 实证;CI 五平台待首跑)** |
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
- **async/defer**:v1 后端 sync-only(frndc 自身写成纯 sync);
  async→状态机是自举之后的大件。
- **Throw**:先 setjmp/longjmp,后迁零成本 unwinding。

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

**Stage 2 第一刀探针:首跑绿(2026-08-31,macOS/aarch64 实证)**——
`tests/functional/llvm_probe`:Lib.open 资产 llvm.dylib → LLVM-C 装配
main ret 42 → verify → TargetMachine emit .obj → spawn 资产 lld
(macOS 零库链接,-platform_version 11.0)→ 跑产物 → **退出码 42**,
runner PASS。绑定层新增 const_int / function_type0(空数组不可 cbuf 编
组,零参走 NULL)。**引擎边角立案(未深挖)**:顶层全局 `val` 以模块
函数调用作初始化器(读 Env)→ 运行期全局初始化 panic(index out of
bounds len 0)——探针规避为函数体内解析,根因待查。资产 =
`assets/toolchain`(git 不跟踪,CI 预取步骤按 triple 拉 tarball 填
充;Linux/Windows 分支的 lld 参数表已写好,**待 CI 五平台首跑实证**——
linkview 完整性缺口将在此暴露)。macOS
链接视图的新事实见 4c 记账。下一步:CI 首跑收五平台 → frondc 后端
模块(lower AST→LLVM)→ frondc 自举(Stage 3 闭环)。

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
分段定位,不阻断主线。**1E 剩余 = 终局验收三级(2026-08-31 全数达成,Stage 1 收官)**:
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
