//! Correctness rules: K001-K006 (wrap Analyzer output).

use crate::ast::Ast::Module;
use crate::pass::Analyzer::AnalysisReport;
use crate::sema::Sema::SemaResult;
use crate::tooling::common::Diagnostic::{Diagnostic, Severity, Category, expr_range, stmt_range};

/// K001: non-exhaustive match.
pub fn non_exhaustive_match(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.match_report.non_exhaustive.iter()
        .map(|(expr_id, type_name, missing)| {
            Diagnostic {
                severity: Severity::Error,
                code: "K001",
                category: Category::Correctness,
                message: format!("non-exhaustive match on `{}`: missing {}", type_name, missing.join(", ")),
                range: expr_range(*expr_id, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            }
        }).collect()
}

/// K002: unreachable match arm.
pub fn unreachable_match_arm(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.match_report.unreachable_arms.iter()
        .map(|(expr_id, arm_idx)| {
            Diagnostic {
                severity: Severity::Warning,
                code: "K002",
                category: Category::Correctness,
                message: format!("unreachable match arm #{}", arm_idx + 1),
                range: expr_range(*expr_id, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            }
        }).collect()
}

/// K003: unreachable code after return/break/continue/throw.
pub fn unreachable_code(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.dead_code.dead_stmts.iter()
        .map(|stmt_id| {
            Diagnostic {
                severity: Severity::Warning,
                code: "K003",
                category: Category::Correctness,
                message: "unreachable code".to_string(),
                range: stmt_range(*stmt_id, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            }
        }).collect()
}

/// K004: dead variable (declared but never used).
pub fn dead_variable(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.dead_var.dead_vars.iter()
        .filter_map(|var_id| {
            let def = report.def_use.defs.get(var_id.0 as usize)?;
            let stmt_id = def.stmt;
            if stmt_id.0 == u32::MAX {
                return None; // Skip parameter/pattern-bind definitions (no real stmt)
            }
            Some(Diagnostic {
                severity: Severity::Warning,
                code: "K004",
                category: Category::Correctness,
                message: format!("unused variable: `{}`", def.name),
                range: stmt_range(stmt_id, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            })
        }).collect()
}

/// K005: dead function (defined but never called).
pub fn dead_function(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.dead_func.dead.iter()
        .filter_map(|func_id| {
            let meta = report.call_graph.get_func_meta(*func_id, module)?;
            let name = meta.name;
            Some(Diagnostic {
                severity: Severity::Warning,
                code: "K005",
                category: Category::Correctness,
                message: format!("unused function: `{}`", name),
                range: expr_range(meta.body, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            })
        }).collect()
}

/// K006: dead parameter (never read by function body).
pub fn dead_parameter(report: &AnalysisReport, _sema: &SemaResult, module: &Module) -> Vec<Diagnostic> {
    report.dead_param.dead_params.iter()
        .filter_map(|(func_id, param_name)| {
            let meta = report.call_graph.get_func_meta(*func_id, module)?;
            Some(Diagnostic {
                severity: Severity::Advice,
                code: "K006",
                category: Category::Correctness,
                message: format!("unused parameter: `{}` in function `{}`", param_name, meta.name),
                range: expr_range(meta.body, &module.arena),
                source_file: module.name.to_string(),
                suggestion: None,
            })
        }).collect()
}
