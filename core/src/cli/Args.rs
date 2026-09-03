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
    /// Compile only (within a project) → out/<project_name>.fndo.
    Build {
        /// Output path (overrides manifest [build] output_dir + project name).
        #[arg(short = 'o', long = "output", value_name = "PATH")]
        output: Option<String>,
        /// Optimization level 0-3 (default 2).
        #[arg(short = 'O', long = "opt-level", value_name = "LEVEL")]
        opt_level: Option<u8>,
    },
    /// Compile + execute immediately (within a project); or execute a specified
    /// .fndo artifact. Trailing arguments after `--` are forwarded to the program
    /// (visible via std.os.Proc.args()).
    Run {
        /// Trailing values: `[FILE.fndo] [--] [PROGRAM_ARGS...]`. The first value
        /// ending in `.fndo` selects artifact mode; everything after `--` (or
        /// beyond the artifact path) becomes the program's arguments.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
        /// Optimization level 0-3 (default 2, only effective in project mode).
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
    /// View .fndo metadata.
    Inspect {
        /// .fndo file path.
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
    /// Canonical sema dump (differential oracle; see cli/Dump.rs).
    Sema,
    /// Canonical module-load dump (1C differential oracle; see cli/Dump.rs).
    Load,
    /// Canonical type-arena operations dump (1D differential oracle; see cli/Dump.rs).
    TyOps,
    /// Extract @extern("C") functions and emit .c to stdout.
    EmitC,
    /// Full pipeline + execution statistics (default).
    Full,
}
