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
    analysis_report.loop_analysis = crate::pass::Analyzer::analyze_loops(&graph);
    if debug {
        eprintln!("  LoopAnalysis: invariants={} unrollable={}",
            analysis_report.loop_analysis.invariants.len(),
            analysis_report.loop_analysis.unrollable.len());
    }

    // Post-IR optimization: LICM/Unroll/Inline + ConstFold/CSE/CopyProp/DCE fixed-point iteration.
    // Driven by opt_level: O0 skips, O1 fixed-point only, O2 full, O3 full + raised iteration limit.
    crate::pass::Optimizer::optimize_with_analysis(&mut graph, Some(&analysis_report), opt_level);

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
    // let kzo_data = crate::solidify::Format::serialize_solidify(&graph);
    // let graph = match crate::solidify::Format::load_zerocopy_from_bytes(kzo_data) {
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
