# OS API 待实现清单

> 2026-08-16 定稿,同日补记。范围:std.os 补充(io=文件系统域 / os=进程环境域 的既有边界不变;
> termios/poll/fd 原始读写按既定设计走 Lib FFI,不进 std)。
> 命名遵循既有约定:Go/Python 朴素风格,完整后缀词,snake_case。
> OsErrorKind ADT 保持六变体不动(EnvFailed/NotFound/PermissionDenied/BufferTooSmall/Unsupported/Other),本清单全部阶段都不需要新增 kind。
>
> **2026-08-16 已落地的前置项**(见文末"已完成前置"):
> `__str_array_join` C 体复活、Path 盘符/UNC 前缀模型、`std.io.Path.current_dir_path()` 联动。

## Path 联动约定(与 io 层 `_path` 双轨惯例对齐)

os 侧路径返回 API 各加一个 `_path` 变体,主返回保持 `str`:

```frond
// Info(P0)
pub fun tmp_dir(): str        →  pub fun tmp_dir_path(): Path
pub fun home_dir(): Throw<str, OsError>  →  pub fun home_dir_path(): Throw<Path, OsError>
// Proc(P1)— 原计划的 program_path 拆成一对,避免 program_path_path
pub fun executable(): str     →  pub fun executable_path(): Path
```

已落地:`std.io.Path.current_dir_path(): Throw<Path, OsError>`(cwd 是 os 域,
但函数放在 std.io.Path——类型位置无法表达跨包类型,无点号限定类型名;
见 Path.frond 注释)。后续 `tmp_dir_path` 等同理落在 std.io.Path。

## P0 第一批 — ✅ 已落地(2026-08-17)

P0 全部实现并通过(套件 `edge_os_proc`,54 功能/29 负向/19 单测全绿)。实现要点:
- argv 链路:cli `--` 透传(项目/工件两模式,clap 中位 `--` 残留在 cmd_run 剥离)→
  engine `ProgramArgs` 槽 → C 侧直接符号引用 `frond_runtime_argc/arg_ptr/arg_len`
  (同二进制 #[no_mangle],无需 dlsym)。
- **设计修正:`args()` 返回恰好 `--` 后透传的值,不含 argv[0]**(原计划对标 Go 含
  argv[0]——但运行模型里没有真实进程 argv,程序路径归 P1 `executable()`)。
- spawn 家族:`__os_spawn(program, NUL 分隔 args_blob)` 双平台(Windows CreateProcessA
  + MSVCRT 引号规则宏;POSIX posix_spawn);handle 约定 Win=HANDLE/POSIX=pid;
  wait 退出码(POSIX 信号死亡=128+sig);close。
- Env.vars 走 Win GetEnvironmentStringsA / POSIX environ(跳 '=' 开头伪项)。
- 新 POSIX 专属头 `spawn.h` 已入 Gen.rs 表(否则 MSVC 无条件 include 炸)。

## P0 原始清单(留档)

### Proc

```frond
// 命令行参数。含 argv[0](对标 Go os.Args / Python sys.argv),业务参数取 args()[1..]。
// 依赖:cli `--` 透传 + engine argv 注册(见"配套工程")。
pub fun args(): str[]

// 启动子进程跑到结束,不经 shell,继承 stdio,返回退出码。
// 对标 Proc.system 的非 shell 替代(system 保留作为 shell 逃生口)。
pub fun run(program: str, args: str[]): Throw<i32, OsError>
```

### Env

```frond
// 全部环境变量,"KEY=VALUE" 形式(对标 Go os.Environ;无 Map 类型故返回 str[])。
pub fun vars(): str[]
```

### Info

```frond
// 临时目录。TMP/TEMP/TMPDIR 逐级回退,最终 "/tmp"(Windows GetTempPathW)。永不失败。
pub fun tmp_dir(): str

// 用户家目录。USERPROFILE(Windows) / HOME(POSIX)。取不到时 Throw NotFound。
pub fun home_dir(): Throw<str, OsError>

// 当前用户名。USERNAME(Windows) / getpwuid(POSIX),getenv 回退。取不到时 Throw NotFound。
pub fun user_name(): Throw<str, OsError>
```

## P1 第二批 — 后台进程与输出捕获(纯 Raw+std 层,无引擎改动)

### Proc(新类型 + 方法)

```frond
// 不透明进程句柄的 std 层包装。handle 为 OS 句柄(Windows HANDLE / POSIX 即 pid 编码)。
type Process = Process(handle: u64, pid: i64)

// 后台启动子进程,不等待。立即返回句柄。
pub fun spawn(program: str, args: str[]): Throw<Process, OsError>

// 自身可执行文件绝对路径(Windows GetModuleFileNameW / POSIX /proc/self/exe)。
// 定位可执行文件旁的资源用。找不到时回退 argv[0],永不 Throw。
// Path 变体:executable_path(): Path(联动约定见文首)。
pub fun executable(): str

// 阻塞等子进程结束,返回退出码。重复调用返回同一结果(内部缓存)。
fun wait(): Throw<i32, OsError>          // Process 方法

// 强制终止(TerminateProcess / kill SIGKILL 语义,无信号参数)。
fun kill(): Throw<void, OsError>         // Process 方法

// 进程 id(仅展示/日志用;等待/终止一律走 Process 方法)。
fun id(): i64                            // Process 方法

// 跑到结束并捕获 stdout+stderr(v1 合并捕获,stderr 字段为空串;分流见 P2)。
pub fun capture(program: str, args: str[]): Throw<ProcOutput, OsError>

type ProcOutput = ProcOutput(code: i32, stdout: str, stderr: str)
```

## P2 第三批 — 细粒度控制(按需再做)

```frond
// 带超时等待:null = 超时仍在运行。Duration 复用 time 模块。
fun wait_timeout(timeout: Duration): Throw<i32?, OsError>   // Process 方法

// 按_pid 终止任意进程(无 Process 句柄时用;Windows OpenProcess+Terminate)。
pub fun kill_pid(pid: i64): Throw<void, OsError>

// 子进程环境定制。cwd=null 继承当前;env 为空数组=继承,非空=完整替换("K=V" 列表)。
type SpawnOptions = SpawnOptions(cwd: str?, env: str[], stdin: str?)
pub fun run_with(opts: SpawnOptions, program: str, args: str[]): Throw<i32, OsError>
pub fun spawn_with(opts: SpawnOptions, program: str, args: str[]): Throw<Process, OsError>
pub fun capture_with(opts: SpawnOptions, program: str, args: str[]): Throw<ProcOutput, OsError>

// capture 分流:stdout/stderr 各自独立管道(需要后)。
// (接口不变,ProcOutput.stderr 生效;仅 Raw 层从单管道改双管道。)

// 机器信息(可选,sysconf / GetSystemInfo)。
pub fun cpu_count(): i32
pub fun total_memory(): i64
```

## 配套工程(cli / engine / Raw)

### P0 必做(引擎链路,工作量主要在这)

1. **cli 透传**:`Args.rs` 支持 `--` 分隔;`frond run -- arg1 arg2`(项目模式)与
   `frond run out/x.fndo -- arg1 arg2`(工件模式)把余下参数原样传给被运行程序。
2. **engine argv 注册**:启动时把 argv 写入全局槽(参考 Channel/Lib builtin 的注册路径),
   `__os_arg_*` 原语读该槽。零语法改动,main 签名不变。
3. **Raw 原语**(builtin/os/Raw.frond 内嵌 C,双平台分支):
   - `__os_arg_count(): i32` / `__os_arg_get_into(i: i32, buf: u8[]): i64`(读引擎槽)
   - `__os_env_vars_count(): i64` / `__os_env_var_get_into(i: i32, buf: u8[]): i64`
     (Win GetEnvironmentStringsW / POSIX extern environ)
   - `__os_tmp_dir_into(buf): i64` / `__os_home_dir_into(buf): i64` / `__os_user_name_into(buf): i64`
   - `__os_spawn(program: str, args_blob: u8[]): u64` — 返回句柄,0=失败
     (Win CreateProcessW,C 侧拼命令行;POSIX posix_spawn 或 fork+execvp,句柄=pid 编码)
   - `__os_process_pid(handle: u64): i64`(Win GetProcessId / POSIX 直接返回)
   - `__os_process_wait(handle: u64): i32`(Win WaitForSingleObject+GetExitCodeProcess / POSIX waitpid;负值=失败)
   - `__os_process_close(handle: u64): i32`(Win CloseHandle / POSIX no-op)
   - `Proc.run` 在 std 层组合 spawn+wait+close,不单设 raw。

### P1 追加(仅 Raw+std)

- `__os_spawn_pipe(program, args_blob): u64` — stdout(及 stderr)接管道
- `__os_pipe_read_into(pipe: u64, buf: u8[]): i64` — 循环读到 EOF,std 层拼接
- `__os_pipe_close(pipe: u64): i32`
- `__os_program_path_into(buf): i64`
- kill:`__os_process_terminate(handle: u64): i32`

### P2 追加

- 超时等待:WaitForSingleObject 带时长 / waitpid WNOHANG 轮询
- 双管道分流、stdin 写入、kill_pid、cpu_count/total_memory

## 测试口径

- functional 新套件 `edge_os_proc`:args 回显、run 真子进程(拿自带 frond.exe 当被启程序最稳)、
  vars/tmp_dir/home_dir 冒烟。
- negative case:`run` 不存在的程序 → `Throw` NotFound。
- 门禁照旧:negative + functional 全绿,stdlib 改动后 touch build.rs。

## 明确不做(边界重申)

- termios/cfmakeraw/poll/read/write 原始 fd 操作 → Lib FFI(编辑器迁移蓝图既定)
- 文件系统操作 → std.io(Fs/File/Dir/Path 已覆盖)
- 信号体系(signal/自定义信号参数) → Windows 无对应,只保留无参 kill 语义

## 2026-08-17 追加调整(用户拍板)

1. **目录类 API 直接返回 Path**(不再走"io `_path` 变体 + Path 侧新 API"双轨):
   `Info.tmp_dir(): Path`、`Info.home_dir(): Throw<Path, OsError>`、`Os.current_dir(): Throw<Path, OsError>`;
   `Path.current_dir_path/tmp_dir_path/home_dir_path` 已删除。
   依据:std 内 `import std.io.Path` + 单名类型完全可用(此前误判为跨包类型不可表达——
   失败的只是点号限定名)。str 需求用 `.to_str()`。
2. **stdlib 数值后缀全面迁移到标注/推断写法**(172→0 处 `42i32` 风格;f16/f128 字面量保留后缀,
   无推断路径):标注绑定/参数/二元对侧/索引写/返回位均可推断;例外形态用助手锚定
   (Math.nan_f32 等,标注局部量——标注**不**传导进二元字面量对)。

## 已完成前置(2026-08-16)

1. **`__str_array_join` C 体复活**:原空体外加 `str[]` 参数不可映射(TYPE_MAP 无 str[]),
   符号从未进入 frond_extern,`Path.to_str` / `DateTime.format` / `Fs.read_file_path` 全链坏死。
   现为 `__str_join_packed_into(blob: u8[], sep: str, out: u8[]): i64` 真 C 体
   (打包格式 `[len32 LE][bytes]`*,两阶段 probe+fill)+ 同名 Frond 包装
   (调用点零改动;打包用 O(1) 下标写,`u8[] ++` 是 O(n²);str.len() 是字符数,
   尺寸计算一律用 bytes().len())。
2. **Path 盘符/UNC 前缀模型**:`Path(segments, prefix)`,prefix ∈ {"", "/", "X:/", "X:", "//"};
   绝对性按宿主平台判定("/x" 仅非 Windows 绝对;同 Go/Rust);UNC 双斜杠保留 roundtrip。
3. **`std.io.Path.current_dir_path(): Throw<Path, OsError>`** 落地(os↔io 联动第一点)。
4. 顺带修复两个幽灵方法(声明无实现,调用即 panic):`T?.is_null()`(Builder 补 nullable
   特判 → CF_IS_NULL)、`str/array.is_empty()`(新算子 343 CF_IS_EMPTY,
   COMPUTE_FN_TABLE_LEN→344,旧 .fndo 失效属预期)。
   新增回归套件 `tests/functional/edge_path`(平台条件断言,Windows/POSIX 可跑)。

**遗留已清(2026-08-16 同日通修)**:"第二次 async 失败杀进程 exit 127"根因是函数级
defer 静态表无条件 drain(未执行到的 defer 读未绑定槽位),已改为执行点动态注册,
详见记忆 defer-dynamic-registration-done;回归用例在 defer_async 套件。
