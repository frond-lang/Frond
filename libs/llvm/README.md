# libs/llvm — LLVM C API 的 Frond 绑定

经 `Lib`(dlopen/LoadLibrary)驱动 LLVM C API 的纯 Frond 绑定层。
不内嵌进编译器、不进 std —— 消费者把 `Llvm.frond` 拷入项目源码目录,
以本地模块引入:`import Llvm`。

## 依赖资产

需要一个 llvm-static CI 产出的 frond 工具链资产 tarball
(release `llvm-static-21.1.8`,https://github.com/frond-lang/llvm/releases)。
动态库以**统一文件名**分发——Frond 永远按显式路径加载,文件名不参与
符号搜索(dlopen/LoadLibraryW 按路径吃内容,SONAME/内部名无关):
`lib/llvm.dll` / `lib/llvm.so` / `lib/llvm.dylib`(Windows 的 import 库
配对为 `lib/llvm.lib`;版本标识在 tarball 的 `VERSION`/`MANIFEST.txt`,
不编码在文件名里)。tarball 另含 `bin/lld` 与 `linkview/`(链接视图,
见 BOOTSTRAP_PLAN 4b/4c),本绑定层只消费动态库。

| 平台 | tar.gz 资产 | 库文件 |
|---|---|---|
| Windows x64 | llvm-static-x86_64-pc-windows-msvc.tar.gz | lib/llvm.dll |
| Linux x64/arm64 | llvm-static-*-unknown-linux-gnu.tar.gz | lib/llvm.so |
| macOS arm64/x64 | llvm-static-*-apple-darwin.tar.gz | lib/llvm.dylib |

加载方式二选一:
- `Lib.open("path/to/llvm.so")` — 开发态;
- `Lib.embed("assets/llvm-native.bin")` — 内嵌分发;文件名各平台统一,
  内容放对应库文件(LoadLibrary/dlopen 按内容识别,后缀无关)。

## 用法(句柄为裸 u64,lib 作首参)

```frond
import Llvm

val lib = Lib.embed("assets/llvm-native.bin")?
Llvm.init_host_target(lib)?
val ctx = Llvm.create_context(lib)?
// … i32_type / function_type / add_function / append_block / build_add …
val ir = Llvm.print_module_ir(lib, module)?   // read-cstr 组合读回 IR 文本
Llvm.emit_to_file(lib, tm, module, "out.o", true)?
```

验收套件:`tests/functional/llvm_bind`(五平台同源码,资产按平台预置)。
注意 `append_block(lib, ctx, func, name)` 的第三参是**函数值引用**
(`add_function` 的返回),不是 module。

read_cstr 组合依赖平台 CRT(msvcrt.dll / libSystem.B.dylib / libc.so.6),
`crt_lib_name()` 自适应。
