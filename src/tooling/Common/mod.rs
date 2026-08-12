#![allow(non_snake_case)]
//! Common — shared types and pipeline for tooling subcommands.
//!
//! Aggregates two submodules:
//! - [`Diagnostic`]: diagnostic types (Severity / Category / Range / Diagnostic / Suggestion)
//! - [`Pipeline`]: shared compile pipeline (parse / sema / incremental) used by LSP, lint, and CLI

pub mod Diagnostic;
pub mod Pipeline;

pub use Diagnostic::{Severity, Category, Range, Pos, Suggestion, expr_range, stmt_range};
pub use Pipeline::{ParseResult, SemaOutcome, SemaIncrementalOutcome};
