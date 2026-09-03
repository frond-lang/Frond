//! Shared compile pipeline (parse → module load → sema → analyze → build → optimize).
//!
//! Used by build/run/debug subcommands.

use std::fs;
use std::io::{self, Read};
use std::process;

use crate::engine::EngineRef;
use crate::ir::Builder::IrBuilder;
use crate::pass::Analyzer;
use crate::tooling::Common::Pipeline;

/// Reads source from a file path or stdin (`-`).
pub fn read_source(path: &str) -> String {
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

/// Full compile pipeline: Parse → Module Load → Sema → Analyzer → Build → Optimizer.
///
/// Returns the compiled `DataFlowGraph` (optimized). When `debug` is true, prints per-stage summaries.
/// Any stage failure (type errors, IR errors, no entry point) is printed and exits with exit(1).
pub fn compile_graph(entry_path: &str, opt_level: crate::pass::Optimizer::OptLevel, debug: bool) -> crate::ir::Ir::DataFlowGraph {
    // Stability policy: the optimizer already contains its own per-round
    // snapshot rollback (see Optimizer::optimize_with_analysis); this net
    // catches panics from ANY other compile stage (parser/sema/analyzer/builder
    // internals) and converts them into a clean internal-error exit instead of
    // crashing the host process (CLI, LSP, embedded compiler).
    let args = (entry_path, opt_level, debug);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let (entry_path, opt_level, debug) = args;
        compile_graph_inner(entry_path, opt_level, debug)
    })) {
        Ok(graph) => graph,
        Err(payload) => {
            eprintln!(
                "error: internal compiler error: {}
(note: the optimizer degrades gracefully; this panic came from outside it — please report)",
                crate::pass::Optimizer::panic_payload_message(&payload)
            );
            process::exit(1);
        }
    }
}

fn compile_graph_inner(entry_path: &str, opt_level: crate::pass::Optimizer::OptLevel, debug: bool) -> crate::ir::Ir::DataFlowGraph {
    // Diagnostic stage timing (FROND_BUILD_TIME=1): per-stage wall time to
    // stderr; same env-gated diagnostic pattern as FROND_TRACE_CYCLES.
    let bt = std::env::var("FROND_BUILD_TIME").is_ok();
    let mut bt_prev = std::time::Instant::now();
    fn bt_mark(name: &str, bt: bool, prev: &mut std::time::Instant) {
        if bt {
            let now = std::time::Instant::now();
            eprintln!("[build-time] {name}: {} us", now.duration_since(*prev).as_micros());
            *prev = now;
        }
    }
    let source = read_source(entry_path);

    if debug {
        eprintln!("=== Frond Debug Mode ===");
        eprintln!("[1/5] Parsing {} ...", entry_path);
    }

    // 1. Parse
    let arena = bumpalo::Bump::new();
    let entry_module = Pipeline::parse_entry_module_or_exit(&arena, &source, entry_path);
    bt_mark("parse_entry", bt, &mut bt_prev);

    if debug {
        eprintln!("  AST: {} declarations", entry_module.declarations.len());
        eprintln!("[2/5] Loading modules ...");
    }

    // 2. Module loading
    let (loader, std_keys, dep_keys) = Pipeline::load_all_modules_or_exit(&entry_module, entry_path);
    // Closure scoping (frondc loaddeps parity): the full std preload keeps
    // the resolution env complete, but SEMA CHECK + IR BUILD only see the
    // entry's transitive import closure — the mirror compiler loads deps
    // only, and the optimizer's reachability pre-pass had been killing these
    // functions post-hoc at 3× the cost (compile + prune-rebuild).
    // (Closure scoping was tried and reverted — see MEM_OPT_PLAN_60: std
    // module CHECKS populate method/resolution tables that user code and
    // sibling std modules consume, and std modules reference each other
    // without imports, so neither sema nor ir-build can be import-scoped.)
    bt_mark("load_modules", bt, &mut bt_prev);

    if debug {
        let builtin_count = loader.builtin_modules().count();
        eprintln!("  Loaded: {} builtin + {} std + {} deps",
            builtin_count, std_keys.len(), dep_keys.len());
        eprintln!("[3/5] Type checking ...");
    }

    // 3. Sema check (shared pipeline; any module type error is printed and exits)
    let (type_arena, sema_result) =
        Pipeline::run_sema_pipeline_or_exit(&loader, &std_keys, &dep_keys, &entry_module, entry_path);
    bt_mark("sema", bt, &mut bt_prev);

    if debug {
        eprintln!("  Sema: OK (no type errors)");
        eprintln!("[4/5] Compiling IR ...");
    }

    // 4. Static analysis (after Sema, before IR): dead code/dead vars/dead functions + memoization strategy.
    //    Runs analysis on the entry module; prints a report summary in debug mode.
    let mut analysis_report = Analyzer::analyze(&entry_module, &entry_module.arena, &sema_result);
    bt_mark("analyze_entry", bt, &mut bt_prev);
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
    // Dedup by module identity: a std module reached BOTH via the user's
    // `import std.x.Y` (dep_keys) and the full-std preload (std_keys) is the
    // SAME loaded module — pushing it twice made the IR pre-registration mint
    // two placeholder subgraphs for one function, and the bare-key overwrite
    // hid it while every or_insert key (short-qualified/package) kept pointing
    // at the never-compiled first placeholder.
    let mut seen_module_ptrs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut non_entry_modules: Vec<&_> = loader
        .builtin_modules()
        .filter_map(|(_, m)| {
            (seen_module_ptrs.insert(m as *const _ as usize)).then_some(m)
        })
        .collect();
    // NOTE: IR build keeps the FULL std preload — std modules reference
    // each other WITHOUT imports (legacy cross-module bare-call compat,
    // e.g. Instant.frond bare-calls Duration()), so import-closure scoping
    // here breaks call binding. Sema stays closure-scoped: the predeclare
    // rounds register every loaded module's declarations, so type checking
    // resolves through them, while std bodies are trusted as pre-checked.
    for key in &std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            if seen_module_ptrs.insert(m as *const _ as usize) {
                non_entry_modules.push(m);
            }
        }
    }
    for k in &dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            if seen_module_ptrs.insert(m as *const _ as usize) {
                non_entry_modules.push(m);
            }
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
    bt_mark("ir_build", bt, &mut bt_prev);

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
    analysis_report.loop_analysis = crate::pass::Analyzer::analyze_loops(&graph);
    bt_mark("loop_analysis", bt, &mut bt_prev);
    if debug {
        eprintln!("  LoopAnalysis: invariants={} unrollable={}",
            analysis_report.loop_analysis.invariants.len(),
            analysis_report.loop_analysis.unrollable.len());
    }

    // TEMP (stmt-drop hunt): --dump-pre dumps the pre-optimizer graph so
    // build-time vs optimizer-introduced range scatter can be distinguished.

    // Post-IR optimization: LICM/Unroll/Inline + ConstFold/CSE/CopyProp/DCE fixed-point iteration.
    // Driven by opt_level: O0 skips, O1 fixed-point only, O2 full, O3 full + wider stall window.
    crate::pass::Optimizer::optimize_with_analysis(&mut graph, Some(&analysis_report), opt_level);

    // W5: optimization compaction (rebuild) invalidates the build-time
    // condition-tree reset plans — recompute them on the final graph so the
    // engine applies the mechanical fast path.
    graph.precompute_reset_plans();
    bt_mark("optimize", bt, &mut bt_prev);


    if debug {
        eprintln!("  IR (after opt):  {} nodes, {} subgraphs, {} compute_fns",
            graph.nodes.len(), graph.subgraphs.len(), graph.compute_fns.len());
        if let Some(entry) = graph.entry_subgraph {
            eprintln!("  Entry subgraph: {:?}", entry);
        }
    }

    graph
}


/// Compile + execute within a project (also reused by debug full).
pub fn run_from_project(opt_level: crate::pass::Optimizer::OptLevel, debug: bool) {
    let entry_path = super::Manifest::resolve_entry_path(None);
    if debug {
        eprintln!("[5/5] Executing ...");
    }
    let graph = compile_graph(&entry_path, opt_level, debug);
    // NOTE: zerocopy serialize/deserialize path temporarily disabled for debugging.
    // The zerocopy round-trip may lose call_target or node_range data.
    // let fndo_data = crate::solidify::Format::serialize_solidify(&graph);
    // let graph = match crate::solidify::Format::load_zerocopy_from_bytes(fndo_data) {
    //     Ok(g) => g,
    //     Err(e) => {
    //         eprintln!("error: failed to load serialized graph: {}", e);
    //         process::exit(1);
    //     }
    // };
    // Engine execution (worker count determined automatically).
    // Stability net: a runtime engine panic becomes a clean error exit
    // instead of crashing the process (the compile itself already succeeded).
    // The graph Arc is kept for the panic path so execution-coverage
    // instrumentation still reports what ran before the crash.
    let engine = EngineRef::new(graph);
    let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.run()
    })) {
        Ok(result) => result,
        Err(payload) => {
            eprintln!(
                "error: internal runtime error: {}",
                crate::pass::Optimizer::panic_payload_message(&payload)
            );
            process::exit(1);
        }
    };
    if debug {
        eprintln!("  Result: {:?}", result);
        eprintln!("=== Done ===");
    }
}
