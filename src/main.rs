//! kuzo CLI — project-based 子命令集
//!
//! 子命令：
//!   kuzo init [name]               脚手架化新项目（创建 kuzo.toml + src/Main.kz）
//!   kuzo build [-O N]              只编译（项目内）→ out/<项目名>.resin
//!   kuzo run [-O N]                编译 + 立即执行（项目内，同 cargo run）
//!   kuzo run <file.resin>          执行指定产物（.resin 加载）
//!   kuzo debug --stage S           诊断模式
//!   kuzo inspect <file.resin>      查看 .resin 元信息
//!
//! build/run(无参)/debug 必须在项目内（向上查找 kuzo.toml），入口来自 manifest。
//! run <file.resin> 不依赖项目（产物分发语义）。
//! worker 数由 engine 自动判断（含 async → 多 worker，纯同步 → 单线程）。

use std::fs;
use std::io::{self, Read};
use std::process;

use clap::{Parser, Subcommand};

use kuzo::ast::Ast::{Module, Printer};
use kuzo::ast::Parser::{ErrorCollector, Lexer, Parser as KuzoParser, Token, TokenCollector};
use kuzo::engine::EngineRef;
use kuzo::pass::Analyzer;
use kuzo::ir::Builder::IrBuilder;
use kuzo::module::ModuleLoader;
use kuzo::module::Error::LoadError;
use kuzo::sema::Sema::{populate_module, SemaResult, TypeArena};
use kuzo::sema::Inference::InferContext;

/// Kuzo 语言 Rust 实现 CLI
#[derive(Parser)]
#[command(name = "kuzo", version, about = "Kuzo language Rust implementation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 脚手架化新项目
    Init {
        /// 项目名称（在 ./name 目录创建，省略则在当前目录）
        name: Option<String>,
    },
    /// 只编译（项目内）→ out/<项目名>.resin
    Build {
        /// 输出路径（覆盖 manifest [build] output_dir + 项目名）
        #[arg(short = 'o', long = "output", value_name = "PATH")]
        output: Option<String>,
        /// 优化等级 0-3（默认 2）
        #[arg(short = 'O', long = "opt-level", value_name = "LEVEL")]
        opt_level: Option<u8>,
    },
    /// 编译 + 立即执行（项目内，无参）；或执行指定产物（有参）
    Run {
        /// .resin 产物路径（有参 = 执行指定产物，无需项目；无参 = 项目内编译+执行）
        file: Option<String>,
        /// 优化等级 0-3（默认 2，仅无参模式有效）
        #[arg(short = 'O', long = "opt-level", value_name = "LEVEL")]
        opt_level: Option<u8>,
    },
    /// 诊断模式（默认完整 pipeline，--stage 指定到某阶段停止并输出）
    Debug {
        /// 入口文件（默认从 manifest 读，`-` 表示 stdin）
        file: Option<String>,
        /// 诊断阶段：tokens（仅词法）、ast（解析后打印 AST）、
        /// check（仅类型检查）、emit-c（提取 C 代码）、emit-ffi（生成 FFI 绑定）、
        /// full（默认，完整 pipeline + 执行统计）
        #[arg(long)]
        stage: Option<DebugStage>,
    },
    /// 查看 .resin 元信息
    Inspect {
        /// .resin 文件路径
        file: String,
        /// 显示每个 section 的详情（kind/offset/len）
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },
}

/// debug 子命令的阶段选项
#[derive(Clone, Debug, clap::ValueEnum)]
enum DebugStage {
    /// 仅词法分析，打印 Token 列表
    Tokens,
    /// 解析后打印 AST（S-表达式）
    Ast,
    /// 仅类型检查
    Check,
    /// 提取 @extern("C") 函数生成 .c 到 stdout
    EmitC,
    /// 生成 Rust FFI 绑定 + wrapper 到 stdout
    EmitFfi,
    /// 完整 pipeline + 执行统计（默认）
    Full,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { name } => cmd_init(name),
        Commands::Build { output, opt_level } => cmd_build(output, opt_level),
        Commands::Run { file, opt_level } => cmd_run(file, opt_level),
        Commands::Debug { file, stage } => cmd_debug(file, stage),
        Commands::Inspect { file, verbose } => cmd_inspect(&file, verbose),
    }
}

// ==================== 项目清单 ====================

/// 从 CLI u8 参数构造 OptLevel，越界值钳制到合法范围。
fn opt_level_from(v: Option<u8>) -> kuzo::pass::Optimizer::OptLevel {
    use kuzo::pass::Optimizer::OptLevel;
    match v {
        None => OptLevel::default(),
        Some(0) => OptLevel::O0,
        Some(1) => OptLevel::O1,
        Some(2) => OptLevel::O2,
        Some(3) => OptLevel::O3,
        Some(n) => {
            eprintln!("warning: opt-level {} out of range [0,3], clamped to 3", n);
            OptLevel::O3
        }
    }
}

/// 项目清单文件名
const MANIFEST_NAME: &str = "kuzo.toml";
/// 默认入口文件
const DEFAULT_ENTRY: &str = "src/Main.kz";
/// 默认输出目录
const DEFAULT_OUTPUT_DIR: &str = "out";

/// 项目清单（serde 反序列化，TOML 格式）
///
/// 格式：
/// ```toml
/// [package]
/// name = "myapp"           # 必需
/// entry = "src/Main.kz"  # 可选，默认 src/Main.kz
///
/// [build]
/// output_dir = "out"       # 可选，默认 "out"
/// opt_level = 2            # 可选，默认 2
/// ```
#[derive(serde::Deserialize)]
struct Manifest {
    package: Package,
    #[serde(default)]
    build: Build,
}

#[derive(serde::Deserialize)]
struct Package {
    name: String,
    #[serde(default = "default_entry")]
    entry: String,
}

#[derive(serde::Deserialize, Default)]
struct Build {
    #[serde(default = "default_output_dir")]
    output_dir: String,
    #[serde(default = "default_opt_level")]
    opt_level: u8,
}

fn default_entry() -> String { DEFAULT_ENTRY.to_string() }
fn default_output_dir() -> String { DEFAULT_OUTPUT_DIR.to_string() }
fn default_opt_level() -> u8 { 2 }

/// 从当前目录向上逐级查找包含清单文件的目录，返回项目根目录路径
/// （最多向上 64 级）
fn find_project_root() -> Option<String> {
    let mut current = std::env::current_dir().ok()?;
    for _ in 0..64 {
        let manifest_path = current.join(MANIFEST_NAME);
        if manifest_path.exists() {
            return Some(current.to_string_lossy().into_owned());
        }
        if !current.pop() {
            break;
        }
    }
    None
}

/// 加载项目清单：向上查找项目根，读取并解析 kuzo.toml
/// 无 manifest 时报错退出（project-based）
fn load_manifest() -> (String, Manifest) {
    let root = find_project_root().unwrap_or_else(|| {
        eprintln!("error: not a Kuzo project (no {} found in current or parent directories)", MANIFEST_NAME);
        eprintln!("  hint: run `kuzo init` to scaffold a new project");
        process::exit(1);
    });
    let manifest_path = std::path::Path::new(&root).join(MANIFEST_NAME);
    let content = fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        eprintln!("error: could not read {}: {}", manifest_path.display(), e);
        process::exit(1);
    });
    let manifest: Manifest = toml::from_str(&content).unwrap_or_else(|e| {
        eprintln!("error: invalid {}: {}", manifest_path.display(), e);
        eprintln!("  hint: ensure [package] section exists with `name = \"...\"` under it");
        process::exit(1);
    });
    (root, manifest)
}

/// 解析入口文件路径：显式 file 优先，否则从 manifest 读 entry（相对于项目根）
fn resolve_entry_path(file: Option<String>) -> String {
    match file {
        Some(f) => f,
        None => {
            let (root, manifest) = load_manifest();
            if std::path::Path::new(&manifest.package.entry).is_absolute() {
                manifest.package.entry
            } else {
                format!("{}/{}", root, manifest.package.entry)
            }
        }
    }
}

// ==================== init 子命令 ====================

fn cmd_init(name: Option<String>) {
    // target_dir 是完整路径，proj_name 取 basename 作为项目名
    let target_dir = name.as_deref().unwrap_or("");
    let proj_name = if target_dir.is_empty() {
        // 未指定路径：用当前目录名作为项目名
        std::env::current_dir().ok()
            .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "app".to_string())
    } else {
        // 指定路径：取 basename 作为项目名
        std::path::Path::new(target_dir)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "app".to_string())
    };

    // 检查目标目录状态
    if !target_dir.is_empty() {
        match fs::metadata(target_dir) {
            Ok(_) => {
                let manifest_path = format!("{}/{}", target_dir, MANIFEST_NAME);
                if fs::metadata(&manifest_path).is_ok() {
                    eprintln!("error: already a Kuzo project ({} contains {})", target_dir, MANIFEST_NAME);
                    process::exit(1);
                }
                if let Ok(entries) = fs::read_dir(target_dir) {
                    if entries.count() > 0 {
                        eprintln!("error: {} is not an empty directory", target_dir);
                        process::exit(1);
                    }
                }
            }
            Err(_) => {
                if let Err(e) = fs::create_dir_all(target_dir) {
                    eprintln!("error: could not create directory '{}': {}", target_dir, e);
                    process::exit(1);
                }
            }
        }
    }

    let manifest_path = if target_dir.is_empty() {
        MANIFEST_NAME.to_string()
    } else {
        format!("{}/{}", target_dir, MANIFEST_NAME)
    };
    let src_dir = if target_dir.is_empty() {
        "src".to_string()
    } else {
        format!("{}/src", target_dir)
    };

    if let Err(e) = fs::create_dir_all(&src_dir) {
        eprintln!("error: could not create directory '{}': {}", src_dir, e);
        process::exit(1);
    }

    let manifest_content = format!(
        "[package]\nname = \"{}\"\nentry = \"{}\"\n\n[build]\noutput_dir = \"{}\"\nopt_level = 2\n",
        proj_name, DEFAULT_ENTRY, DEFAULT_OUTPUT_DIR
    );
    if let Err(e) = fs::write(&manifest_path, manifest_content) {
        eprintln!("error: could not write '{}': {}", manifest_path, e);
        process::exit(1);
    }

    let main_path = if target_dir.is_empty() {
        DEFAULT_ENTRY.to_string()
    } else {
        format!("{}/{}", target_dir, DEFAULT_ENTRY)
    };
    // Console 在 builtin/io 下，默认可见无需 import
    let main_content = "fun main(): void {\n    println(\"Hello, Kuzo!\")\n}\n";
    if let Err(e) = fs::write(&main_path, main_content) {
        eprintln!("error: could not write '{}': {}", main_path, e);
        process::exit(1);
    }

    println!("Created Kuzo project '{}'", proj_name);
    println!("  {}", manifest_path);
    println!("  {}", main_path);
}

// ==================== debug 子命令 ====================

fn cmd_debug(file: Option<String>, stage: Option<DebugStage>) {
    let stage = stage.unwrap_or(DebugStage::Full);
    let entry_path = resolve_entry_path(file);
    let source = read_source(&entry_path);

    match stage {
        DebugStage::Tokens => debug_tokens(&source),
        DebugStage::Ast => debug_ast(&source),
        DebugStage::EmitC => debug_emit_c(&source),
        DebugStage::EmitFfi => debug_emit_ffi(&source),
        DebugStage::Check => debug_check(&source, &entry_path),
        DebugStage::Full => run_from_project(kuzo::pass::Optimizer::OptLevel::default(), true),
    }
}

/// 仅词法分析，打印 Token 列表
fn debug_tokens(source: &str) {
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens = sink.into_tokens();
    for tok in &tokens {
        println!(
            "{:>4}:{:<3} {:<20} {}",
            tok.line,
            tok.column,
            format!("{:?}", tok.kind),
            tok.lexeme
        );
    }
}

/// 解析并打印 AST（S-表达式）
fn debug_ast(source: &str) {
    let arena = bumpalo::Bump::new();
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = KuzoParser::new(tokens_ref, &arena, ErrorCollector::new());

    match parser.parse_module("stdin") {
        Ok(module) => {
            let mut printer = Printer::new(&module.arena);
            let output = printer.print_module(&module);
            print!("{}", output);
        }
        Err(err) => {
            eprintln!("Parse error at {}:{}: {}", err.line, err.column, err.message);
            process::exit(1);
        }
    }
    for err in parser.errors() {
        eprintln!("Warning: parse error at {}:{}: {}", err.line, err.column, err.message);
    }
}

/// 提取 @extern("C") 函数生成 .c 到 stdout
fn debug_emit_c(source: &str) {
    let arena = bumpalo::Bump::new();
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = KuzoParser::new(tokens_ref, &arena, ErrorCollector::new());

    match parser.parse_module("stdin") {
        Ok(module) => {
            if !parser.errors().is_empty() {
                for err in parser.errors() {
                    eprintln!("Error: parse error at {}:{}: {}", err.line, err.column, err.message);
                }
                process::exit(1);
            }
            match kuzo::ffi::ExternC::extract_c_from_module(&module) {
                Ok(c_code) => print!("{}", c_code),
                Err(e) => {
                    eprintln!("Error extracting C: {}", e);
                    process::exit(1);
                }
            }
        }
        Err(err) => {
            eprintln!("Parse error at {}:{}: {}", err.line, err.column, err.message);
            process::exit(1);
        }
    }
}

/// 生成 Rust FFI 绑定 + wrapper 到 stdout
fn debug_emit_ffi(source: &str) {
    let arena = bumpalo::Bump::new();
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = KuzoParser::new(tokens_ref, &arena, ErrorCollector::new());

    match parser.parse_module("stdin") {
        Ok(module) => {
            if !parser.errors().is_empty() {
                for err in parser.errors() {
                    eprintln!("Error: parse error at {}:{}: {}", err.line, err.column, err.message);
                }
                process::exit(1);
            }
            match kuzo::ffi::ExternC::extract_rust_ffi_from_module(&module) {
                Ok(ffi_code) => print!("{}", ffi_code),
                Err(e) => {
                    eprintln!("Error generating FFI: {}", e);
                    process::exit(1);
                }
            }
        }
        Err(err) => {
            eprintln!("Parse error at {}:{}: {}", err.line, err.column, err.message);
            process::exit(1);
        }
    }
}

// ==================== 公共管线（debug_check / cmd_run 共享） ====================

/// 解析入口模块。arena 必须在返回的 Module 存活期间保持有效。
fn parse_entry_module<'a>(arena: &'a bumpalo::Bump, source: &'a str, filename: &'a str) -> Module<'a> {
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = KuzoParser::new(tokens_ref, arena, ErrorCollector::new());

    let module = match parser.parse_module(filename) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("{}:{}:{}: parse error: {}", filename, err.line, err.column, err.message);
            process::exit(1);
        }
    };
    for err in parser.errors() {
        eprintln!("Warning: {}:{}:{}: {}", filename, err.line, err.column, err.message);
    }
    module
}

/// 加载全部模块（builtin + std + 用户依赖），返回 (loader, std_keys, dep_keys)。
/// 入口文件所在目录被添加为搜索路径，以解析用户模块（如 Math/Geometry.kz）。
fn load_all_modules(
    entry_module: &Module,
    entry_path: &str,
) -> (ModuleLoader, Vec<String>, Vec<String>) {
    let mut loader = ModuleLoader::new();
    if let Some(src_dir) = std::path::Path::new(entry_path).parent() {
        loader.add_search_path(src_dir);
    }
    let dep_keys = loader.load_transitive_imports(entry_module);

    let std_keys: Vec<String> = kuzo::module::STD_FILES
        .iter()
        .map(|(p, _)| p.to_string())
        .collect();
    for key in &std_keys {
        let parts: Vec<&str> = key.strip_suffix(".kz").unwrap().split('/').collect();
        let _ = loader.resolve_and_load(&parts);
    }

    if loader.has_load_errors() {
        for err in loader.load_errors() {
            match err {
                LoadError::ModuleNotFound { path } => {
                    eprintln!("error: module not found: {}", path);
                }
                LoadError::ParseFailed { path, line, column, message } => {
                    eprintln!("error: parse failed in {} at {}:{}: {}", path, line, column, message);
                }
                LoadError::CircularImport { path } => {
                    eprintln!("error: circular import detected: {}", path);
                }
            }
        }
        process::exit(1);
    }
    (loader, std_keys, dep_keys)
}

/// 运行完整 Sema 管线：注册内建类型 → predeclare 全部模块 → 逐模块检查。
/// 任何模块的类型错误都会打印并 exit(1)。成功时返回 (type_arena, sema_result)。
fn run_sema_pipeline(
    loader: &ModuleLoader,
    std_keys: &[String],
    dep_keys: &[String],
    entry_module: &Module,
    entry_filename: &str,
) -> (TypeArena, SemaResult) {
    let mut type_arena = TypeArena::new();
    let mut sema_result = SemaResult::new();
    let mut ctx = InferContext::new(&mut type_arena, &mut sema_result);

    ctx.reset_state();
    let root_env = ctx.env.root();
    ctx.register_builtins(root_env);

    let module_logical_paths: Vec<String> = loader
        .loaded_keys()
        .iter()
        .filter_map(|k| k.strip_suffix(".kz").map(|s| s.replace('/', ".")))
        .collect();
    ctx.register_module_aliases(root_env, &module_logical_paths);

    // predeclare：先注册所有模块的函数和类型构造器到 root_env，
    // 解决模块间前向引用问题。check_module_with_env 内部会再次 predeclare（幂等）。
    for (_, m) in loader.builtin_modules() {
        ctx.predeclare_declarations(m, root_env);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            ctx.predeclare_declarations(m, root_env);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            ctx.predeclare_declarations(m, root_env);
        }
    }

    let mut prev_err_len = 0usize;

    // populate：在 check 前填充所有模块的定义表（类型方法签名等），
    // 解决模块检查顺序导致的跨模块方法查找失败问题。
    // check_module_with_env 内部会再次调用（幂等，put_type_def 拒绝重复）。
    for (_, m) in loader.builtin_modules() {
        populate_module(ctx.arena, ctx.sema_result, m);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            populate_module(ctx.arena, ctx.sema_result, m);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            populate_module(ctx.arena, ctx.sema_result, m);
        }
    }
    populate_module(ctx.arena, ctx.sema_result, entry_module);

    // 构造 all_modules 列表：供跨模块单态化使用（泛型调用需访问被调函数所在模块的 arena）
    let mut all_modules: Vec<&Module> = Vec::new();
    for (_, m) in loader.builtin_modules() {
        all_modules.push(m);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            all_modules.push(m);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            all_modules.push(m);
        }
    }
    all_modules.push(entry_module);

    // check: builtin → std → dep → entry
    for (path, m) in loader.builtin_modules() {
        ctx.check_module_with_env(m, root_env, &all_modules);
        for err in &ctx.sema_result.errors[prev_err_len..] {
            eprintln!("{}:{}:{}: {}", path, err.line, err.column, err.message);
        }
        prev_err_len = ctx.sema_result.errors.len();
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            ctx.check_module_with_env(m, root_env, &all_modules);
            for err in &ctx.sema_result.errors[prev_err_len..] {
                eprintln!("{}:{}:{}: {}", key, err.line, err.column, err.message);
            }
            prev_err_len = ctx.sema_result.errors.len();
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            ctx.check_module_with_env(m, root_env, &all_modules);
            for err in &ctx.sema_result.errors[prev_err_len..] {
                eprintln!("{}:{}:{}: {}", k, err.line, err.column, err.message);
            }
            prev_err_len = ctx.sema_result.errors.len();
        }
    }
    ctx.check_module_with_env(entry_module, root_env, &all_modules);
    for err in &ctx.sema_result.errors[prev_err_len..] {
        eprintln!("{}:{}:{}: {}", entry_filename, err.line, err.column, err.message);
    }

    if !ctx.sema_result.errors.is_empty() {
        process::exit(1);
    }
    // ctx 借用 type_arena 和 sema_result，在此丢弃后两者所有权归还调用方。
    drop(ctx);
    (type_arena, sema_result)
}

/// 仅类型检查
fn debug_check(source: &str, filename: &str) {
    let arena = bumpalo::Bump::new();
    let entry_module = parse_entry_module(&arena, source, filename);
    let (loader, std_keys, dep_keys) = load_all_modules(&entry_module, filename);
    let (_type_arena, _sema_result) =
        run_sema_pipeline(&loader, &std_keys, &dep_keys, &entry_module, filename);
    println!("ok: {} (no type errors)", filename);
}

// ==================== 编译管线（build/run/debug 复用） ====================

/// 完整编译管线：Parse → Module Load → Sema → Analyzer → Build → Optimizer
///
/// 返回编译后的 `DataFlowGraph`（已优化）。`debug` 为 true 时打印各阶段摘要。
/// 任何阶段失败（类型错误、IR 错误、无入口）均打印并 exit(1)。
fn compile_graph(entry_path: &str, opt_level: kuzo::pass::Optimizer::OptLevel, debug: bool) -> kuzo::ir::Ir::DataFlowGraph {
    let source = read_source(entry_path);

    if debug {
        eprintln!("=== Kuzo Debug Mode ===");
        eprintln!("[1/5] Parsing {} ...", entry_path);
    }

    // 1. Parse
    let arena = bumpalo::Bump::new();
    let entry_module = parse_entry_module(&arena, &source, entry_path);

    if debug {
        eprintln!("  AST: {} declarations", entry_module.declarations.len());
        eprintln!("[2/5] Loading modules ...");
    }

    // 2. 模块加载
    let (loader, std_keys, dep_keys) = load_all_modules(&entry_module, entry_path);

    if debug {
        let builtin_count = loader.builtin_modules().count();
        eprintln!("  Loaded: {} builtin + {} std + {} deps",
            builtin_count, std_keys.len(), dep_keys.len());
        eprintln!("[3/5] Type checking ...");
    }

    // 3. Sema check（共享管线，任何模块类型错误均打印并 exit）
    let (type_arena, sema_result) =
        run_sema_pipeline(&loader, &std_keys, &dep_keys, &entry_module, entry_path);

    if debug {
        eprintln!("  Sema: OK (no type errors)");
        eprintln!("[4/5] Compiling IR ...");
    }

    // 4. 静态分析（Sema 后、IR 前）：死代码/死变量/死函数 + 记忆化策略
    //    对 entry 模块运行分析；debug 模式下打印报告摘要。
    let mut analysis_report = Analyzer::analyze(&entry_module, &entry_module.arena, &sema_result);
    if debug {
        eprintln!("  Analyzer: dead_code={} dead_var={} dead_func={} memo_candidates={} dead_param={} inline={} stack_alloc={} non_exhaustive={} unreachable_arms={}",
            analysis_report.dead_code.dead_stmts.len(),
            analysis_report.dead_var.dead_vars.len(),
            analysis_report.dead_func.dead.len(),
            analysis_report.memo.candidates.len(),
            analysis_report.dead_param.dead_params.len(),
            analysis_report.inline.candidates.len(),
            analysis_report.stack_alloc.candidates.len(),
            analysis_report.match_report.non_exhaustive.len(),
            analysis_report.match_report.unreachable_arms.len());
    }

    // 5. IR 编译
    // 收集所有非 entry 模块（builtin + std + dep），传给 IR builder 编译为子图
    let mut non_entry_modules: Vec<&_> = loader.builtin_modules().map(|(_, m)| m).collect();
    for key in &std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            non_entry_modules.push(m);
        }
    }
    for k in &dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            non_entry_modules.push(m);
        }
    }
    // 为每个非 entry 模块生成静态分析报告（memoize/dead_code/inline 等通用覆盖）
    // 持有 owned Box 避免泄漏：引用仅在 build() 期间有效，build 完成后随 owner 释放
    let mut graph = {
        let builtin_analyses_owned: Vec<Box<Analyzer::AnalysisReport>> = non_entry_modules
            .iter()
            .map(|m| Box::new(Analyzer::analyze(m, &m.arena, &sema_result)))
            .collect();
        let builtin_analyses: Vec<Option<&Analyzer::AnalysisReport>> = builtin_analyses_owned
            .iter()
            .map(|b| Some(b.as_ref()))
            .collect();
        IrBuilder::new(&sema_result, &type_arena, &entry_module)
            .with_builtins(non_entry_modules)
            .with_analysis(&analysis_report)
            .with_builtin_analyses(builtin_analyses)
            .build()
    };

    // 检查 IR 编译错误（未实现的特性降级、找不到函数等）
    if !graph.ir_errors.is_empty() {
        for err in &graph.ir_errors {
            eprintln!("{}: IR error: {}", entry_path, err);
        }
        process::exit(1);
    }

    // 检查入口子图：无 main 函数时优雅报错，避免 Engine panic
    if graph.entry_subgraph.is_none() {
        eprintln!("error: no entry point found in {} (expected a `main` function)", entry_path);
        process::exit(1);
    }

    if debug {
        eprintln!("  IR (before opt): {} nodes, {} subgraphs, {} compute_fns",
            graph.nodes.len(), graph.subgraphs.len(), graph.compute_fns.len());
    }

    // 循环分析（IR 后）：识别不变量 + 可展开循环，填充 analysis_report.loop_analysis
    analysis_report.loop_analysis = kuzo::pass::Analyzer::analyze_loops(&graph);
    if debug {
        eprintln!("  LoopAnalysis: invariants={} unrollable={}",
            analysis_report.loop_analysis.invariants.len(),
            analysis_report.loop_analysis.unrollable.len());
    }

    // IR 后优化：LICM/Unroll/Inline + ConstFold/CSE/CopyProp/DCE 固定点迭代
    // opt_level 驱动：O0 跳过，O1 仅固定点，O2 全量，O3 全量+提高迭代上限
    kuzo::pass::Optimizer::optimize_with_analysis(&mut graph, Some(&analysis_report), opt_level);

    if debug {
        eprintln!("  IR (after opt):  {} nodes, {} subgraphs, {} compute_fns",
            graph.nodes.len(), graph.subgraphs.len(), graph.compute_fns.len());
        if let Some(entry) = graph.entry_subgraph {
            eprintln!("  Entry subgraph: {:?}", entry);
        }
    }

    graph
}

// ==================== build 子命令 ====================

fn cmd_build(output: Option<String>, opt_level_cli: Option<u8>) {
    let (root, manifest) = load_manifest();
    // opt_level 优先级：CLI flag > manifest [build] opt_level > 默认 O2
    let opt_level = opt_level_from(opt_level_cli.or(Some(manifest.build.opt_level)));

    let entry = if std::path::Path::new(&manifest.package.entry).is_absolute() {
        manifest.package.entry.clone()
    } else {
        format!("{}/{}", root, manifest.package.entry)
    };

    let graph = compile_graph(&entry, opt_level, false);

    // 输出路径：-o 优先，否则 output_dir/<项目名>.resin
    let out_path = match output {
        Some(o) => o,
        None => {
            let dir = if std::path::Path::new(&manifest.build.output_dir).is_absolute() {
                manifest.build.output_dir.clone()
            } else {
                format!("{}/{}", root, manifest.build.output_dir)
            };
            // 确保输出目录存在
            if let Err(e) = fs::create_dir_all(&dir) {
                eprintln!("error: could not create output directory '{}': {}", dir, e);
                process::exit(1);
            }
            format!("{}/{}.resin", dir, manifest.package.name)
        }
    };

    // 序列化为 .resin
    let resin_data = kuzo::resin::Format::serialize_resin(&graph);
    if let Err(e) = fs::write(&out_path, &resin_data) {
        eprintln!("error: could not write '{}': {}", out_path, e);
        process::exit(1);
    }

    let size_kb = resin_data.len() as f64 / 1024.0;
    eprintln!("Compiled {} → {} ({:.1} KB, {} nodes, {} subgraphs, opt-level {})",
        manifest.package.entry, out_path, size_kb,
        graph.nodes.len(), graph.subgraphs.len(), opt_level as u8);
}

// ==================== run 子命令（重载） ====================

/// `kuzo run` 重载入口：
/// - 无参：项目内编译 + 立即执行（同 cargo run）
/// - 有参 <file.resin>：执行指定产物（.resin 加载）
fn cmd_run(file: Option<String>, opt_level_cli: Option<u8>) {
    match file {
        None => {
            // opt_level 优先级：CLI flag > manifest [build] opt_level > 默认 O2
            let (_, manifest) = load_manifest();
            let opt_level = opt_level_from(opt_level_cli.or(Some(manifest.build.opt_level)));
            run_from_project(opt_level, false)
        }
        Some(f) => run_from_resin(&f),
    }
}

/// 项目内编译 + 执行（debug full 也复用）
fn run_from_project(opt_level: kuzo::pass::Optimizer::OptLevel, debug: bool) {
    let entry_path = resolve_entry_path(None);
    if debug {
        eprintln!("[5/5] Executing ...");
    }
    let graph = compile_graph(&entry_path, opt_level, debug);
    // serialize → zerocopy load → run（验证 .resin zerocopy 路径）
    let resin_data = kuzo::resin::Format::serialize_resin(&graph);
    let graph = match kuzo::resin::Format::load_zerocopy_from_bytes(resin_data) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: failed to load serialized graph: {}", e);
            process::exit(1);
        }
    };
    // Engine 执行（worker 数自动判断）
    let result = EngineRef::new(graph).run();
    if debug {
        eprintln!("  Result: {:?}", result);
        eprintln!("=== Done ===");
    }
}

/// 执行指定 .resin 产物：mmap 加载 → 重建运行时字段 → Engine 执行
fn run_from_resin(path: &str) {
    // 校验扩展名
    if !path.ends_with(".resin") {
        eprintln!("error: expected .resin file, got: {}", path);
        eprintln!("  hint: run `kuzo build` first to compile, then `kuzo run out/<name>.resin`");
        process::exit(1);
    }
    if !std::path::Path::new(path).exists() {
        eprintln!("error: file not found: {}", path);
        process::exit(1);
    }
    let graph = match kuzo::resin::Format::load_resin_from_file(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: invalid .resin file {}: {}", path, e);
            process::exit(1);
        }
    };
    // 检查入口子图
    if graph.entry_subgraph.is_none() {
        eprintln!("error: no entry point in {}", path);
        process::exit(1);
    }
    // Engine 执行（worker 数自动判断）
    let _result = EngineRef::new(graph).run();
}

// ==================== inspect 子命令 ====================

fn cmd_inspect(file: &str, verbose: bool) {
    if !file.ends_with(".resin") {
        eprintln!("error: expected .resin file, got: {}", file);
        process::exit(1);
    }
    match kuzo::resin::Format::inspect_resin_from_file(file) {
        Ok(info) => {
            println!("RESIN File: {}", file);
            println!("  Schema:       v{}", info.schema_version);
            println!("  ABI:          v{}", info.abi_version);
            println!("  Nodes:        {}", info.node_count);
            println!("  Subgraphs:    {} (entry: {})",
                info.subgraph_count,
                info.entry_subgraph.map(|s| format!("#{}", s)).unwrap_or("none".to_string()));
            println!("  Inputs:       {}", info.input_count);
            println!("  Strings:      {} bytes", info.string_pool_len);
            println!("  Global vars:  {}", info.global_var_count);
            println!("  Memo tables:  {}", info.memo_table_count);
            println!("  Compute fns:  {}", info.compute_fn_count);
            println!("  Sections:     {}", info.section_count);
            println!("  Checksum:     0x{:08X}", info.crc32);
            println!("  Total size:   {:.1} KB", info.file_size as f64 / 1024.0);
            if verbose {
                println!("");
                println!("Section Details:");
                println!("  {:<22} {:>6} {:>10} {:>10}", "Kind", "u8", "Offset", "Len");
                println!("  {:<22} {:>6} {:>10} {:>10}", "----", "--", "------", "---");
                let mut total: u64 = 0;
                for &(kind_u8, offset, len) in &info.sections {
                    let name = kuzo::resin::Spec::SectionKind::from_u8(kind_u8)
                        .map(|k| k.name())
                        .unwrap_or("Unknown");
                    println!("  {:<22} {:>6} {:>10} {:>10}", name, kind_u8, offset, len);
                    total += len as u64;
                }
                let total_kb = total as f64 / 1024.0;
                let overhead = info.file_size as f64 - total as f64;
                println!("  {:<22} {:>6} {:>10} {:>10.1}", "TOTAL", "", "", total_kb);
                println!("  {:<22} {:>6} {:>10} {:>10.1}", "Overhead", "", "", overhead / 1024.0);
            }
        }
        Err(e) => {
            eprintln!("error: invalid .resin file: {}", e);
            process::exit(1);
        }
    }
}

// ==================== 公共工具 ====================

fn read_source(path: &str) -> String {
    if path == "-" {
        let mut buf = String::new();
        if io::stdin().read_to_string(&mut buf).is_err() {
            eprintln!("Error reading from stdin");
            process::exit(1);
        }
        buf
    } else {
        match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {}", path, e);
                process::exit(1);
            }
        }
    }
}
