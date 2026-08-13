#![allow(non_snake_case)]
//! Lint — rule-based static analysis with configurable severity.
//!
//! Aggregates three submodules:
//! - [`Registry`]: rule registry + lint config (RuleRegistry / LintConfig)
//! - [`Rules`]: lint rule implementations by category (Correctness / Style / Perf / Idioms)
//! - [`Report`]: report formatting (text / JSON)

pub mod Registry;
pub mod Rules;
pub mod Report;

pub use Registry::{RuleRegistry, LintConfig};

use std::fs;
use bumpalo::Bump;
use crate::tooling::Common::Pipeline;
use crate::tooling::Common::Diagnostic::{Diagnostic, Severity, Category, Range};
use crate::pass::Analyzer;

/// Lint a single file: parse → sema → analyze → run rules.
/// Never exits; returns all diagnostics collected.
pub fn lint_file(path: &str, config: &LintConfig) -> Vec<Diagnostic> {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return vec![Diagnostic {
                severity: Severity::Error,
                code: "IO",
                category: Category::Correctness,
                message: format!("cannot read file: {}", e),
                range: Range::default(),
                source_file: path.to_string(),
                suggestion: None,
            }];
        }
    };

    let arena = Bump::new();
    let parse_result = Pipeline::parse_entry_module_lsp(&arena, &source, path);
    let mut all_diag = parse_result.diagnostics;

    let (loader, std_keys, dep_keys) = Pipeline::load_all_modules_or_exit(
        &parse_result.module, path
    );

    match Pipeline::run_sema_pipeline_lsp(
        &loader, &std_keys, &dep_keys, &parse_result.module, path
    ) {
        Pipeline::SemaOutcome::Ok { sema_result, diagnostics: sema_diag, .. } => {
            all_diag.extend(sema_diag);

            // Run Analyzer + lint rules
            let analysis = Analyzer::analyze(
                &parse_result.module,
                &parse_result.module.arena,
                &sema_result
            );

            let registry = RuleRegistry::new();
            let lint_diag = registry.run_all(
                &parse_result.module,
                &parse_result.module.arena,
                &sema_result,
                Some(&analysis),
                config
            );
            all_diag.extend(lint_diag);
        }
        Pipeline::SemaOutcome::Err(sema_diag) => {
            all_diag.extend(sema_diag);
        }
    }

    all_diag
}
