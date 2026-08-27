# frondc — 自举 Stage 1(自托管编译器)

用 Frond 写的 Frond 编译器。Stage 1 目标:`fronc check` —— 词法 + 语法 +
全套 sema(含 monomorph),跑在 Rust 引擎上,与 Rust 编译器(`Frond/core`)
差分对齐。不移植 ir/、engine/、pass/、solidify/(解释器路径专属,自举走
AST→LLVM 直下,见 Stage 2)。

## 里程碑状态

- **1A 词法器:完成(2026-08-27)。** `src/Lex.frond`(忠实转写 Rust
  Lexer,字节级状态机含全部怪癖:列按字节计/非 ASCII 特判/#{...}# 原始块/
  插值嵌套串)。`frond run -- lex <file>` 输出与 `frond debug --stage
  tokens` 逐字节一致。差分验收:`tests/scripts/diff_lex.sh`,语料
  std+libs+frondc+tests+apps 共 406 文件全对齐。
- 下一步 1B:语法器(AST 定义 + 递归下降),对 `debug --stage ast` 差分。

## 布局(为什么在 Frond/ 而不是 apps/)

frondc 不是"用 Frond 写的应用",而是语言本体的一部分 —— 与 core(Rust
实现)、std、libs 平级。Stage 3 自举闭环后,这里就是唯一的编译器,core
降级为 bootstrap 工具。

```
Frond/frondc/
  Root.toml            工程清单(入口 src/Main.frond)
  src/                 frondc 本体(逐里程碑生长)
    Main.frond         CLI 入口(1A 起:lex 子命令)
```

表达力探针已转正为功能套件:`tests/functional/expressiveness/`
(七探针含结果与耗时记录,验收口径 RESULT: ALL PASSED)。

## 差分 oracle(Stage 1 的验收发生器)

```sh
# Rust 侧 canonical dump(cli/Dump.rs,格式版本 sema-dump v1):
Frond/core/target/release/frond.exe debug --stage sema <entry.frond>
# frondc 侧须逐字节复现同一输出;reject 用退出码 + stderr 差分。
# tokens/ast 级别已有:debug --stage tokens / ast
# 1A 词法差分:tests/scripts/diff_lex.sh <file...>(Rust tokens vs frondc lex)
```

## 跑套件

```sh
cd tests/scripts && ./run_functional.sh expressiveness
```

## 探针结果(2026-08-27,首跑)

| 探针 | 结果 | 数据 |
|---|---|---|
| arena | PASS | 20 万节点构建 840ms / 索引跳跃遍历 194ms |
| mutate | PASS | 100 万次原地改 1009ms;别名浅共享验证通过 |
| deepmatch | PASS | 递归 ADT + 嵌套 ctor 模式全可用;1500 深 build 14ms / sum 734ms;20 万次分派 1845ms |
| maps | PASS | str 键 5 万 set 10.1s(约 200µs/op,偏慢,见下) |
| strbuild | PASS | 1 万次追加循环 16ms(线性!无 O(n²));Str.repeat 646ms |
| refparam | PASS | 见下:&self 写穿可用,plain &T 参数调不了 |
| recordkey | PASS | record 作 Map 键完全可用(取回 100,len=2) |

### 首跑发现(已修 / 已知)

- **引擎 bug(已修)**:pipeline 级 predeclare 在 populate 之前跑,
  `current_module_name` 为空 → 兄弟用户模块的构造器预绑定铸成裸名
  Adt('Ctx'),与规范化注册键('Mut.Ctx')永不相交,first-wins define
  让陈旧绑定存活 → 兄弟模块内 record 字段访问、递归函数返回用户类型
  全部报错。入口模块(check 内才首 predeclare)与 std(裸名机制)幸免。
  修法:predeclare_declarations 开头镜像 populate 的模块上下文设定
  (ModuleEnv.rs)。87+62 套件回归全绿,A/B 实测无 perf 回归。
- **语言缺口(记录在案)**:普通函数 `fun f(c: &Ctx)` 的 &T 参数声明
  可解析,但调用点 `f(&ctx)` / 裸传 `f(ctx)` 都过不了检查 —— solver
  移植的上下文传递改用"record + &self 方法"或显式返回。
- **性能注意**:Map<str,_> 约 200µs/op,sema 移植的符号表热路径若照此
  会在十万级符号时拖到分钟级 —— frondc 选型时优先 IntMap/句柄键,或
  届时给 Map 上 C 内核。
- Map 复合键(record key)可用,str 追加是线性的 —— 两个原疑点解除。

