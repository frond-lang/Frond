//! CLI argument definitions (Cli/Commands/DebugStage).

use clap::{Parser, Subcommand};

/// Frond language Rust implementation CLI.
#[derive(Parser)]
#[command(name = "frond", version, about = "")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
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
        /// check (type check only), emit-c (extract C code),
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
pub enum DebugStage {
    /// Lexical analysis only; print the token list.
    Tokens,
    /// Print the AST (S-expressions) after parsing.
    Ast,
    /// Type check only.
    Check,
    /// Extract @extern("C") functions and emit .c to stdout.
    EmitC,
    /// Full pipeline + execution statistics (default).
    Full,
}
