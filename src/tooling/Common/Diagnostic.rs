//! Shared diagnostic model: Severity, Range, Diagnostic — used by LSP, lint, and fmt (degraded).

use crate::ast::Ast::{Span, ExprId, StmtId, AstArena};

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Advice,
}

/// Diagnostic category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    Correctness,
    Style,
    Perf,
    Idiom,
}

/// Source position (1-based line/col + byte offset).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pos {
    pub line: u32,
    pub col: u32,
    pub offset: usize,
}

/// Source range [start, end].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Range {
    pub start: Pos,
    pub end: Pos,
}

impl Range {
    /// Construct a single-point range from a Span (start == end).
    /// Phase 1: AstArena Span has no end; Phase 2 will extend Span.
    pub fn from_span(span: Span) -> Self {
        Self {
            start: Pos { line: span.line, col: span.column, offset: 0 },
            end: Pos { line: span.line, col: span.column, offset: 0 },
        }
    }

    /// Construct from two Spans.
    pub fn new(start: Span, end: Span) -> Self {
        Self {
            start: Pos { line: start.line, col: start.column, offset: 0 },
            end: Pos { line: end.line, col: end.column, offset: 0 },
        }
    }
}

/// Auto-fix suggestion (Phase 2 --fix; Phase 1 always None).
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub range: Range,
    pub replacement: String,
    pub description: String,
}

/// Unified diagnostic type shared by LSP, lint, and fmt.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: &'static str,
    pub category: Category,
    pub message: String,
    pub range: Range,
    pub source_file: String,
    pub suggestion: Option<Suggestion>,
}

/// Get the range of an ExprId from the AST arena.
pub fn expr_range(expr_id: ExprId, arena: &AstArena) -> Range {
    let span = arena.expr(expr_id).span;
    Range::from_span(span)
}

/// Get the range of a StmtId from the AST arena.
pub fn stmt_range(stmt_id: StmtId, arena: &AstArena) -> Range {
    let span = arena.stmt(stmt_id).span;
    Range::from_span(span)
}
