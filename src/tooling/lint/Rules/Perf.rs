//! Performance rules: PERF001-003 (wrap Analyzer output).

use crate::ast::Ast::Module;
use crate::pass::Analyzer::AnalysisReport;
use crate::sema::Sema::SemaResult;
use crate::tooling::Common::Diagnostic::{Diagnostic, Severity, Category, expr_range};

/// PERF001: memoizable function.
pub fn memoizable(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.memo.candidates.iter()
        .filter_map(|cand| {
            let meta = report.call_graph.get_func_meta(cand.func, module)?;
            Some(Diagnostic {
                severity: Severity::Advice,
                code: "PERF001",
                category: Category::Perf,
                message: format!("function `{}` is pure; consider @memoize", meta.name),
                range: expr_range(meta.body, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            })
        }).collect()
}

/// PERF002: inlineable function.
pub fn inlineable(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.inline.candidates.iter()
        .filter_map(|(func_id, _size)| {
            let meta = report.call_graph.get_func_meta(*func_id, module)?;
            Some(Diagnostic {
                severity: Severity::Advice,
                code: "PERF002",
                category: Category::Perf,
                message: format!("function `{}` is small and pure; consider inlining", meta.name),
                range: expr_range(meta.body, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            })
        }).collect()
}

/// PERF003: stack-allocable.
pub fn stack_allocable(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.stack_alloc.candidates.iter()
        .map(|expr_id| {
            Diagnostic {
                severity: Severity::Advice,
                code: "PERF003",
                category: Category::Perf,
                message: "allocation does not escape; consider stack allocation".to_string(),
                range: expr_range(*expr_id, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            }
        }).collect()
}
