//! run subcommand — compile + execute (project) or execute .fndo artifact.

use std::process;

use crate::engine::EngineRef;

use super::Manifest::{load_manifest, opt_level_from};
use super::Pipeline::run_from_project;

/// `frond run` overloaded entry:
/// - No args: compile + execute immediately within a project (like cargo run).
/// - With args <file.fndo>: execute the specified artifact (.fndo load).
pub fn cmd_run(file: Option<String>, opt_level_cli: Option<u8>) {
    match file {
        None => {
            // opt_level priority: CLI flag > manifest [build] opt_level > default O2
            let (_, manifest) = load_manifest();
            let opt_level = opt_level_from(opt_level_cli.or(Some(manifest.build.opt_level)));
            run_from_project(opt_level, false)
        }
        Some(f) => run_from_fndo(&f),
    }
}

/// Execute a specified .fndo artifact: mmap load → rebuild runtime fields → Engine execution.
fn run_from_fndo(path: &str) {
    // Validate file extension
    if !path.ends_with(".fndo") {
        eprintln!("error: expected .fndo file, got: {}", path);
        eprintln!("  hint: run `frond build` first to compile, then `frond run out/<name>.fndo`");
        process::exit(1);
    }
    if !std::path::Path::new(path).exists() {
        eprintln!("error: file not found: {}", path);
        process::exit(1);
    }
    let graph = match crate::solidify::Format::load_solidify_from_file(path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: invalid .fndo file {}: {}", path, e);
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
