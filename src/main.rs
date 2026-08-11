//! kuzo CLI — project-based subcommand set
//! (implicit this rebuild)
//!
//! Subcommands:
//!   kuzo init [name]               Scaffold a new project (creates kuzo.toml + src/Main.kz)
//!   kuzo build [-O N]              Compile only (within a project) → out/<project_name>.kzo
//!   kuzo run [-O N]                Compile + execute immediately (within a project, like cargo run)
//!   kuzo run <file.kzo>          Execute a specified artifact (.kzo load)
//!   kuzo debug --stage S           Diagnostic mode
//!   kuzo inspect <file.kzo>      View .kzo metadata
//!
//! build/run (no args)/debug must be run inside a project (searches upward for kuzo.toml); the entry point comes from the manifest.
//! run <file.kzo> does not require a project (artifact distribution semantics).
//! The worker count is determined automatically by the engine (async → multiple workers, pure sync → single-threaded).

use std::fs;
use std::io::{self, Read};
use std::process;

use clap::{Parser, Subcommand};

use kuzo::ast::Ast::Printer;
use kuzo::ast::Parser::{ErrorCollector, Lexer, Parser as KuzoParser, Token, TokenCollector};
use kuzo::engine::EngineRef;
use kuzo::pass::Analyzer;
use kuzo::ir::Builder::IrBuilder;
use kuzo::tooling::common::Pipeline;
use kuzo::tooling::fmt::Engine::{format as fmt_source, FmtConfig};
use kuzo::tooling::lint::{lint_file, LintConfig};
use kuzo::tooling::lint::Report;
use kuzo::tooling::lsp::Server::LspServer;

/// Kuzo language Rust implementation CLI.
#[derive(Parser)]
#[command(name = "kuzo", version, about = "")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Scaffold a new project.
    Init {
        /// Project name (created in ./name directory; defaults to the current directory when omitted).
        name: Option<String>,
    },
    /// Compile only (within a project) → out/<project_name>.kzo.
    Build {
        /// Output path (overrides manifest [build] output_dir + project name).
        #[arg(short = 'o', long = "output", value_name = "PATH")]
        output: Option<String>,
        /// Optimization level 0-3 (default 2).
        #[arg(short = 'O', long = "opt-level", value_name = "LEVEL")]
        opt_level: Option<u8>,
    },
    /// Compile + execute immediately (within a project, no args); or execute a specified artifact (with args).
    Run {
        /// .kzo artifact path (with args = execute the specified artifact, no project needed; without args = compile + execute within a project).
        file: Option<String>,
        /// Optimization level 0-3 (default 2, only effective in no-arg mode).
        #[arg(short = 'O', long = "opt-level", value_name = "LEVEL")]
        opt_level: Option<u8>,
    },
    /// Diagnostic mode (default: full pipeline; --stage stops at a specified stage and outputs).
    Debug {
        /// Entry file (defaults to the manifest; `-` means stdin).
        file: Option<String>,
        /// Diagnostic stage: tokens (lex only), ast (print AST after parsing),
        /// check (type check only), emit-c (extract C code), emit-ffi (generate FFI bindings),
        /// full (default: full pipeline + execution statistics).
        #[arg(long)]
        stage: Option<DebugStage>,
    },
    /// View .kzo metadata.
    Inspect {
        /// .kzo file path.
        file: String,
        /// Show details for each section (kind/offset/len).
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },
    /// Format source code.
    Fmt {
        /// File or directory to format (default: src/ directory).
        path: Option<String>,
        /// Check formatting without modifying files (exit 1 if unformatted).
        #[arg(long = "check")]
        check: bool,
        /// Read from stdin, write to stdout.
        #[arg(long = "stdin")]
        stdin: bool,
    },
    /// Lint source code.
    Lint {
        /// File to lint (default: src/ directory).
        path: Option<String>,
        /// Output format: human (default) or json.
        #[arg(long = "format", value_name = "FORMAT")]
        format: Option<String>,
        /// Treat warnings as errors (exit 1).
        #[arg(long = "deny")]
        deny: Option<String>,
        /// Read from stdin.
        #[arg(long = "stdin")]
        stdin: bool,
    },
    /// Start LSP server (reads from stdin, writes to stdout).
    Lsp,
}

/// Stage options for the debug subcommand.
#[derive(Clone, Debug, clap::ValueEnum)]
enum DebugStage {
    /// Lexical analysis only; print the token list.
    Tokens,
    /// Print the AST (S-expressions) after parsing.
    Ast,
    /// Type check only.
    Check,
    /// Extract @extern("C") functions and emit .c to stdout.
    EmitC,
    /// Generate Rust FFI bindings + wrapper to stdout.
    EmitFfi,
    /// Full pipeline + execution statistics (default).
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
        Commands::Fmt { path, check, stdin } => cmd_fmt(path, check, stdin),
        Commands::Lint { path, format, deny, stdin } => cmd_lint(path, format, deny, stdin),
        Commands::Lsp => cmd_lsp(),
    }
}

// ==================== Project manifest ====================

/// Constructs an `OptLevel` from a CLI `u8` argument; out-of-range values are clamped to the valid range.
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

/// Project manifest file name.
const MANIFEST_NAME: &str = "kuzo.toml";
/// Default entry file.
const DEFAULT_ENTRY: &str = "src/Main.kz";
/// Default output directory.
const DEFAULT_OUTPUT_DIR: &str = "out";

/// Project manifest (deserialized via serde, TOML format).
///
/// Format:
/// ```toml
/// [package]
/// name = "myapp"           # required
/// entry = "src/Main.kz"  # optional, defaults to src/Main.kz
///
/// [build]
/// output_dir = "out"       # optional, defaults to "out"
/// opt_level = 2            # optional, defaults to 2
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

/// Searches upward from the current directory for a directory containing the manifest file,
/// returning the project root path (up to 64 levels up).
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

/// Loads the project manifest: searches upward for the project root, then reads and parses kuzo.toml.
/// Exits with an error if no manifest is found (project-based).
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

/// Resolves the entry file path: an explicit `file` takes priority; otherwise reads `entry` from the manifest (relative to the project root).
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

// ==================== init subcommand ====================

fn cmd_init(name: Option<String>) {
    // target_dir is the full path; proj_name takes the basename as the project name
    let target_dir = name.as_deref().unwrap_or("");
    let proj_name = if target_dir.is_empty() {
        // No path specified: use the current directory name as the project name
        std::env::current_dir().ok()
            .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "app".to_string())
    } else {
        // Path specified: take the basename as the project name
        std::path::Path::new(target_dir)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "app".to_string())
    };

    // Check target directory state
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
    // Console is under builtin/io and visible by default, so no import is needed.
    let main_content = "fun main(): void {\n    println(\"Hello, Kuzo!\")\n}\n";
    if let Err(e) = fs::write(&main_path, main_content) {
        eprintln!("error: could not write '{}': {}", main_path, e);
        process::exit(1);
    }

    println!("Created Kuzo project '{}'", proj_name);
    println!("  {}", manifest_path);
    println!("  {}", main_path);
}

// ==================== debug subcommand ====================

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

/// Lexical analysis only; print the token list.
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

/// Parse and print the AST (S-expressions).
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

/// Extract @extern("C") functions and emit .c to stdout.
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

/// Generate Rust FFI bindings + wrapper to stdout.
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

/// Type check only.
fn debug_check(source: &str, filename: &str) {
    let arena = bumpalo::Bump::new();
    let entry_module = Pipeline::parse_entry_module_or_exit(&arena, source, filename);
    let (loader, std_keys, dep_keys) = Pipeline::load_all_modules_or_exit(&entry_module, filename);
    let (_type_arena, _sema_result) =
        Pipeline::run_sema_pipeline_or_exit(&loader, &std_keys, &dep_keys, &entry_module, filename);
    println!("ok: {} (no type errors)", filename);
}

// ==================== Compile pipeline (shared by build/run/debug) ====================

/// Full compile pipeline: Parse → Module Load → Sema → Analyzer → Build → Optimizer.
///
/// Returns the compiled `DataFlowGraph` (optimized). When `debug` is true, prints per-stage summaries.
/// Any stage failure (type errors, IR errors, no entry point) is printed and exits with exit(1).
fn compile_graph(entry_path: &str, opt_level: kuzo::pass::Optimizer::OptLevel, debug: bool) -> kuzo::ir::Ir::DataFlowGraph {
    let source = read_source(entry_path);

    if debug {
        eprintln!("=== Kuzo Debug Mode ===");
        eprintln!("[1/5] Parsing {} ...", entry_path);
    }

    // 1. Parse
    let arena = bumpalo::Bump::new();
    let entry_module = Pipeline::parse_entry_module_or_exit(&arena, &source, entry_path);

    if debug {
        eprintln!("  AST: {} declarations", entry_module.declarations.len());
        eprintln!("[2/5] Loading modules ...");
    }

    // 2. Module loading
    let (loader, std_keys, dep_keys) = Pipeline::load_all_modules_or_exit(&entry_module, entry_path);

    if debug {
        let builtin_count = loader.builtin_modules().count();
        eprintln!("  Loaded: {} builtin + {} std + {} deps",
            builtin_count, std_keys.len(), dep_keys.len());
        eprintln!("[3/5] Type checking ...");
    }

    // 3. Sema check (shared pipeline; any module type error is printed and exits)
    let (type_arena, sema_result) =
        Pipeline::run_sema_pipeline_or_exit(&loader, &std_keys, &dep_keys, &entry_module, entry_path);

    if debug {
        eprintln!("  Sema: OK (no type errors)");
        eprintln!("[4/5] Compiling IR ...");
    }

    // 4. Static analysis (after Sema, before IR): dead code/dead vars/dead functions + memoization strategy.
    //    Runs analysis on the entry module; prints a report summary in debug mode.
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

    // 5. IR compilation
    // Collect all non-entry modules (builtin + std + dep) and pass them to the IR builder to compile as subgraphs.
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
    // Generate a static analysis report for each non-entry module (general coverage of memoize/dead_code/inline, etc.).
    // Hold owned Boxes to avoid leaks: references are only valid during build(); released with the owner after build completes.
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

    // Check for IR compilation errors (unimplemented feature fallbacks, missing functions, etc.).
    if !graph.ir_errors.is_empty() {
        for err in &graph.ir_errors {
            eprintln!("{}: IR error: {}", entry_path, err);
        }
        process::exit(1);
    }

    // Check the entry subgraph: report gracefully when there is no main function, to avoid an Engine panic.
    if graph.entry_subgraph.is_none() {
        eprintln!("error: no entry point found in {} (expected a `main` function)", entry_path);
        process::exit(1);
    }

    if debug {
        eprintln!("  IR (before opt): {} nodes, {} subgraphs, {} compute_fns",
            graph.nodes.len(), graph.subgraphs.len(), graph.compute_fns.len());
    }

    // Loop analysis (after IR): identify invariants + unrollable loops, populating analysis_report.loop_analysis.
    analysis_report.loop_analysis = kuzo::pass::Analyzer::analyze_loops(&graph);
    if debug {
        eprintln!("  LoopAnalysis: invariants={} unrollable={}",
            analysis_report.loop_analysis.invariants.len(),
            analysis_report.loop_analysis.unrollable.len());
    }

    // Post-IR optimization: LICM/Unroll/Inline + ConstFold/CSE/CopyProp/DCE fixed-point iteration.
    // Driven by opt_level: O0 skips, O1 fixed-point only, O2 full, O3 full + raised iteration limit.
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

// ==================== build subcommand ====================

fn cmd_build(output: Option<String>, opt_level_cli: Option<u8>) {
    let (root, manifest) = load_manifest();
    // opt_level priority: CLI flag > manifest [build] opt_level > default O2
    let opt_level = opt_level_from(opt_level_cli.or(Some(manifest.build.opt_level)));

    let entry = if std::path::Path::new(&manifest.package.entry).is_absolute() {
        manifest.package.entry.clone()
    } else {
        format!("{}/{}", root, manifest.package.entry)
    };

    let graph = compile_graph(&entry, opt_level, false);

    // Output path: -o takes priority; otherwise output_dir/<project_name>.kzo
    let out_path = match output {
        Some(o) => o,
        None => {
            let dir = if std::path::Path::new(&manifest.build.output_dir).is_absolute() {
                manifest.build.output_dir.clone()
            } else {
                format!("{}/{}", root, manifest.build.output_dir)
            };
            // Ensure the output directory exists
            if let Err(e) = fs::create_dir_all(&dir) {
                eprintln!("error: could not create output directory '{}': {}", dir, e);
                process::exit(1);
            }
            format!("{}/{}.kzo", dir, manifest.package.name)
        }
    };

    // Serialize to .kzo
    let kzo_data = kuzo::solidify::Format::serialize_solidify(&graph);
    if let Err(e) = fs::write(&out_path, &kzo_data) {
        eprintln!("error: could not write '{}': {}", out_path, e);
        process::exit(1);
    }

    let size_kb = kzo_data.len() as f64 / 1024.0;
    eprintln!("Compiled {} → {} ({:.1} KB, {} nodes, {} subgraphs, opt-level {})",
        manifest.package.entry, out_path, size_kb,
        graph.nodes.len(), graph.subgraphs.len(), opt_level as u8);
}

// ==================== run subcommand (overloaded) ====================

/// `kuzo run` overloaded entry:
/// - No args: compile + execute immediately within a project (like cargo run).
/// - With args <file.kzo>: execute the specified artifact (.kzo load).
fn cmd_run(file: Option<String>, opt_level_cli: Option<u8>) {
    match file {
        None => {
            // opt_level priority: CLI flag > manifest [build] opt_level > default O2
            let (_, manifest) = load_manifest();
            let opt_level = opt_level_from(opt_level_cli.or(Some(manifest.build.opt_level)));
            run_from_project(opt_level, false)
        }
        Some(f) => run_from_kzo(&f),
    }
}

/// Compile + execute within a project (also reused by debug full).
fn run_from_project(opt_level: kuzo::pass::Optimizer::OptLevel, debug: bool) {
    let entry_path = resolve_entry_path(None);
    if debug {
        eprintln!("[5/5] Executing ...");
    }
    let graph = compile_graph(&entry_path, opt_level, debug);
    // NOTE: zerocopy serialize/deserialize path temporarily disabled for debugging.
    // The zerocopy round-trip may lose call_target or node_range data.
    // let kzo_data = kuzo::solidify::Format::serialize_solidify(&graph);
    // let graph = match kuzo::solidify::Format::load_zerocopy_from_bytes(kzo_data) {
    //     Ok(g) => g,
    //     Err(e) => {
    //         eprintln!("error: failed to load serialized graph: {}", e);
    //         process::exit(1);
    //     }
    // };
    // Engine execution (worker count determined automatically)
    let result = EngineRef::new(graph).run();
    if debug {
        eprintln!("  Result: {:?}", result);
        eprintln!("=== Done ===");
    }
}

/// Execute a specified .kzo artifact: mmap load → rebuild runtime fields → Engine execution.
fn run_from_kzo(path: &str) {
    // Validate file extension
    if !path.ends_with(".kzo") {
        eprintln!("error: expected .kzo file, got: {}", path);
        eprintln!("  hint: run `kuzo build` first to compile, then `kuzo run out/<name>.kzo`");
        process::exit(1);
    }
    if !std::path::Path::new(path).exists() {
        eprintln!("error: file not found: {}", path);
        process::exit(1);
    }
    let graph = match kuzo::solidify::Format::load_solidify_from_file(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: invalid .kzo file {}: {}", path, e);
            process::exit(1);
        }
    };
    // Check the entry subgraph
    if graph.entry_subgraph.is_none() {
        eprintln!("error: no entry point in {}", path);
        process::exit(1);
    }
    // Engine execution (worker count determined automatically)
    let _result = EngineRef::new(graph).run();
}

// ==================== inspect subcommand ====================

fn cmd_inspect(file: &str, verbose: bool) {
    if !file.ends_with(".kzo") {
        eprintln!("error: expected .kzo file, got: {}", file);
        process::exit(1);
    }
    match kuzo::solidify::Format::inspect_solidify_from_file(file) {
        Ok(info) => {
            println!("KZO File: {}", file);
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
                    let name = kuzo::solidify::Spec::SectionKind::from_u8(kind_u8)
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
            eprintln!("error: invalid .kzo file: {}", e);
            process::exit(1);
        }
    }
}

// ==================== fmt subcommand ====================

fn cmd_fmt(path: Option<String>, check: bool, stdin: bool) {
    let config = FmtConfig::default();

    if stdin {
        let mut source = String::new();
        if io::stdin().read_to_string(&mut source).is_err() {
            eprintln!("error: failed to read from stdin");
            process::exit(1);
        }
        let formatted = fmt_source(&source, &config);
        print!("{}", formatted);
        return;
    }

    // find_project_root() returns the project root *directory* (containing kuzo.toml),
    // so join "src" directly to get the default format target.
    let target = path.unwrap_or_else(|| {
        find_project_root()
            .map(|p| {
                std::path::Path::new(&p)
                    .join("src")
                    .to_string_lossy()
                    .to_string()
            })
            .unwrap_or_else(|| "src".to_string())
    });

    let target_path = std::path::Path::new(&target);
    let ok = if target_path.is_file() {
        format_file(target_path, &config, check)
    } else if target_path.is_dir() {
        format_dir(target_path, &config, check)
    } else {
        eprintln!("error: path not found: {}", target);
        process::exit(1);
    };

    if check && !ok {
        process::exit(1);
    }
}

fn format_file(path: &std::path::Path, config: &FmtConfig, check: bool) -> bool {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return false;
        }
    };
    let formatted = fmt_source(&source, config);
    if check {
        if source != formatted {
            eprintln!("would reformat: {}", path.display());
            return false;
        }
        true
    } else {
        if source != formatted {
            if fs::write(path, &formatted).is_err() {
                eprintln!("error: cannot write {}", path.display());
                return false;
            }
        }
        true
    }
}

fn format_dir(dir: &std::path::Path, config: &FmtConfig, check: bool) -> bool {
    let mut all_ok = true;
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: cannot read directory {}: {}", dir.display(), e);
            return false;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            all_ok &= format_dir(&path, config, check);
        } else if path.extension().map(|e| e == "kz").unwrap_or(false) {
            all_ok &= format_file(&path, config, check);
        }
    }
    all_ok
}

// ==================== lint subcommand ====================

fn cmd_lint(path: Option<String>, format: Option<String>, deny: Option<String>, stdin: bool) {
    let config = LintConfig::default();
    let format_str = format.as_deref().unwrap_or("human");

    let (diagnostics, _source_file) = if stdin {
        let mut source = String::new();
        if io::stdin().read_to_string(&mut source).is_err() {
            eprintln!("error: failed to read from stdin");
            process::exit(1);
        }
        let diags = lint_file_string("<stdin>", &source, &config);
        (diags, "<stdin>".to_string())
    } else {
        // find_project_root() returns the project root directory (containing kuzo.toml),
        // so join "src" directly to get the default lint target (same as fmt).
        let target = path.unwrap_or_else(|| {
            find_project_root()
                .map(|p| {
                    std::path::Path::new(&p)
                        .join("src")
                        .to_string_lossy()
                        .to_string()
                })
                .unwrap_or_else(|| "src".to_string())
        });
        let target_path = std::path::Path::new(&target);
        let all_diags = if target_path.is_file() {
            lint_file(&target, &config)
        } else if target_path.is_dir() {
            lint_dir(&target, &config)
        } else {
            eprintln!("error: path not found: {}", target);
            process::exit(1);
        };
        (all_diags, target)
    };

    // Output
    match format_str {
        "json" => println!("{}", Report::format_json(&diagnostics)),
        _ => print!("{}", Report::format_human(&diagnostics)),
    }

    // Exit code: 1 if any Error (or Warning when --deny warning)
    let has_error = diagnostics.iter().any(|d| match d.severity {
        kuzo::tooling::common::Diagnostic::Severity::Error => true,
        kuzo::tooling::common::Diagnostic::Severity::Warning => deny.as_deref() == Some("warning"),
        kuzo::tooling::common::Diagnostic::Severity::Advice => false,
    });
    if has_error {
        process::exit(1);
    }
}

fn lint_dir(dir: &str, config: &LintConfig) -> Vec<kuzo::tooling::common::Diagnostic::Diagnostic> {
    let mut all = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return all,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            all.extend(lint_dir(&path.to_string_lossy(), config));
        } else if path.extension().map(|e| e == "kz").unwrap_or(false) {
            all.extend(lint_file(&path.to_string_lossy(), config));
        }
    }
    all
}

/// Lint a source string directly (for --stdin).
fn lint_file_string(filename: &str, source: &str, config: &LintConfig) -> Vec<kuzo::tooling::common::Diagnostic::Diagnostic> {
    // Write to a temp file, then lint it
    let temp_path = format!("/tmp/kuzo_lint_{}.kz", std::process::id());
    let _ = fs::write(&temp_path, source);
    let result = lint_file(&temp_path, config);
    let _ = fs::remove_file(&temp_path);
    // Fix source_file in diagnostics
    result.into_iter().map(|mut d| {
        d.source_file = filename.to_string();
        d
    }).collect()
}

// ==================== lsp subcommand ====================

fn cmd_lsp() {
    let server = LspServer::new();
    server.run(); // never returns
}

// ==================== Common utilities ====================

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
