//! Lint rule registry and configuration.

use crate::ast::Ast::{Module, AstArena};
use crate::pass::Analyzer::AnalysisReport;
use crate::sema::Sema::SemaResult;
use crate::tooling::common::Diagnostic::{Diagnostic, Severity, Category};
use rustc_hash::FxHashMap;

/// Lint configuration: per-rule severity overrides.
#[derive(Debug, Clone)]
pub struct LintConfig {
    pub default_severity: Severity,
    pub rules: FxHashMap<&'static str, SeverityOverride>,
}

#[derive(Debug, Clone, Copy)]
pub enum SeverityOverride {
    Severity(Severity),
    Off,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            default_severity: Severity::Warning,
            rules: FxHashMap::default(),
        }
    }
}

impl LintConfig {
    pub fn is_enabled(&self, code: &str) -> bool {
        match self.rules.get(code) {
            Some(SeverityOverride::Off) => false,
            _ => true,
        }
    }

    pub fn severity(&self, code: &str) -> Option<Severity> {
        match self.rules.get(code) {
            Some(SeverityOverride::Severity(s)) => Some(*s),
            Some(SeverityOverride::Off) => None,
            None => Some(self.default_severity),
        }
    }
}

/// Rule runner: either wraps Analyzer output or does independent AST walk.
pub enum RuleRunner {
    FromAnalysis(fn(&AnalysisReport, &SemaResult, &Module) -> Vec<Diagnostic>),
    AstWalk(fn(&Module, &AstArena, &SemaResult) -> Vec<Diagnostic>),
}

struct RuleEntry {
    code: &'static str,
    name: &'static str,
    category: Category,
    default_severity: Severity,
    runner: RuleRunner,
}

/// Registry of all lint rules.
pub struct RuleRegistry {
    rules: Vec<RuleEntry>,
}

impl RuleRegistry {
    pub fn new() -> Self {
        let mut registry = Self { rules: Vec::new() };
        registry.register_all();
        registry
    }

    fn register_all(&mut self) {
        // Correctness rules (wrap Analyzer)
        self.register("K001", "non-exhaustive-match", Category::Correctness, Severity::Error,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Correctness::non_exhaustive_match));
        self.register("K002", "unreachable-match-arm", Category::Correctness, Severity::Warning,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Correctness::unreachable_match_arm));
        self.register("K003", "unreachable-code", Category::Correctness, Severity::Warning,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Correctness::unreachable_code));
        self.register("K004", "dead-variable", Category::Correctness, Severity::Warning,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Correctness::dead_variable));
        self.register("K005", "dead-function", Category::Correctness, Severity::Warning,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Correctness::dead_function));
        self.register("K006", "dead-parameter", Category::Correctness, Severity::Advice,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Correctness::dead_parameter));

        // Performance rules (wrap Analyzer)
        self.register("PERF001", "memoizable", Category::Perf, Severity::Advice,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Perf::memoizable));
        self.register("PERF002", "inlineable", Category::Perf, Severity::Advice,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Perf::inlineable));
        self.register("PERF003", "stack-allocable", Category::Perf, Severity::Advice,
            RuleRunner::FromAnalysis(crate::tooling::lint::Rules::Perf::stack_allocable));

        // Style rules (AST walk)
        self.register("STYLE001", "naming", Category::Style, Severity::Advice,
            RuleRunner::AstWalk(crate::tooling::lint::Rules::Style::naming));
        self.register("STYLE002", "unused-import", Category::Style, Severity::Warning,
            RuleRunner::AstWalk(crate::tooling::lint::Rules::Style::unused_import));
        self.register("STYLE003", "redundant-paren", Category::Style, Severity::Advice,
            RuleRunner::AstWalk(crate::tooling::lint::Rules::Style::redundant_paren));

        // Idiom rules (AST walk)
        self.register("IDIOM001", "prefer-val", Category::Idiom, Severity::Advice,
            RuleRunner::AstWalk(crate::tooling::lint::Rules::Idioms::prefer_val));
        self.register("IDIOM002", "string-interp", Category::Idiom, Severity::Advice,
            RuleRunner::AstWalk(crate::tooling::lint::Rules::Idioms::string_interpolation));
    }

    fn register(
        &mut self,
        code: &'static str,
        name: &'static str,
        category: Category,
        default_severity: Severity,
        runner: RuleRunner,
    ) {
        self.rules.push(RuleEntry {
            code,
            name,
            category,
            default_severity,
            runner,
        });
    }

    /// Run all enabled rules, returning diagnostics.
    pub fn run_all(
        &self,
        module: &Module,
        arena: &AstArena,
        sema: &SemaResult,
        analysis: Option<&AnalysisReport>,
        config: &LintConfig,
    ) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for entry in &self.rules {
            if !config.is_enabled(entry.code) {
                continue;
            }

            let rule_diags = match &entry.runner {
                RuleRunner::FromAnalysis(f) => {
                    if let Some(report) = analysis {
                        f(report, sema, module)
                    } else {
                        Vec::new()
                    }
                }
                RuleRunner::AstWalk(f) => {
                    f(module, arena, sema)
                }
            };

            // Apply config severity override
            let severity = config.severity(entry.code).unwrap_or(entry.default_severity);
            for mut d in rule_diags {
                d.severity = severity;
                diagnostics.push(d);
            }
        }

        diagnostics
    }
}
