#![allow(non_snake_case)]
//! tooling — Developer tooling infrastructure (LSP, formatter, linter).
//!
//! Aggregates four tooling submodules:
//! - [`Common`]: shared types and pipeline (Diagnostic, Pipeline) used by all tooling subcommands
//! - [`Fmt`]: code formatter (token-stream reformatting with trivia preservation)
//! - [`Lint`]: rule-based static analysis with configurable severity
//! - [`Lsp`]: JSON-RPC language server protocol implementation

pub mod Common;
pub mod Fmt;
pub mod Lint;
pub mod Lsp;
