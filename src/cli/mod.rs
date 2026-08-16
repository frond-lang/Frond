//! cli — Frond CLI subcommand dispatcher.
//!
//! Split into data-flow-responsibility modules:
//! - `Args`: CLI argument definitions (Cli/Commands/DebugStage)
//! - `Manifest`: project manifest (Root.toml) loading + path resolution
//! - `Pipeline`: shared compile pipeline (compile_graph + read_source)
//! - `Init`/`Debug`/`Build`/`Run`/`Inspect`/`Fmt`/`Lint`/`Lsp`: per-subcommand implementations

pub mod Args;
pub mod Manifest;
pub mod Pipeline;
pub mod Init;
pub mod Debug;
pub mod Build;
pub mod Run;
pub mod Inspect;
pub mod Fmt;
pub mod Lint;
pub mod Lsp;

use clap::Parser;

use Args::{Cli, Commands};

/// CLI entry point: parse args and dispatch to the matching subcommand.
pub fn run() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { name } => Init::cmd_init(name),
        Commands::Build { output, opt_level } => Build::cmd_build(output, opt_level),
        Commands::Run { file, opt_level } => Run::cmd_run(file, opt_level),
        Commands::Debug { file, stage } => Debug::cmd_debug(file, stage),
        Commands::Inspect { file, verbose } => Inspect::cmd_inspect(&file, verbose),
        Commands::Fmt { path, check, stdin } => Fmt::cmd_fmt(path, check, stdin),
        Commands::Lint { path, format, deny, stdin } => Lint::cmd_lint(path, format, deny, stdin),
        Commands::Lsp => Lsp::cmd_lsp(),
    }
}
