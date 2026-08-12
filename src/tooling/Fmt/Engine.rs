//! Formatter engine: trivia-preserving lexer + token-stream reformatting.

use crate::ast::Parser::{Lexer, Token, TokenCollector, TokenKind};
use super::Rules::{Action, decide_spacing};

/// Trivia: comments, blank lines, whitespace between significant tokens.
#[derive(Debug, Clone)]
pub enum Trivia<'a> {
    /// Line comment `// ...`
    LineComment(&'a str),
    /// Block comment `/* ... */` (supports nesting)
    BlockComment(&'a str),
    /// Single newline
    Newline,
    /// Multiple consecutive blank lines (count >= 1)
    BlankLines(u32),
    /// Inline whitespace (spaces/tabs within a line)
    Whitespace(&'a str),
}

/// Formatter token: significant token + its leading trivia.
#[derive(Debug, Clone)]
pub struct FmtToken<'a> {
    pub kind: TokenKind,
    pub lexeme: &'a str,
    pub line: u32,
    pub col: u32,
    pub leading: Vec<Trivia<'a>>,
}

/// Tokenize source into FmtTokens with trivia.
/// Uses the existing Lexer for token classification, then walks the source
/// to extract trivia between tokens.
pub fn tokenize_with_trivia(source: &str) -> Vec<FmtToken<'_>> {
    // Step 1: Run existing lexer to get tokens
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let raw_tokens: Vec<Token<'_>> = sink.into_tokens();

    // Step 2: Walk the source between token positions to extract trivia
    let mut fmt_tokens = Vec::with_capacity(raw_tokens.len());
    let mut byte_cursor = 0usize;

    for tok in &raw_tokens {
        // Find the byte offset of this token by matching line/col
        let tok_byte_start = find_byte_offset(source, byte_cursor, tok.line, tok.column);

        // Extract trivia between byte_cursor and tok_byte_start
        let leading = extract_trivia(source, byte_cursor, tok_byte_start);

        fmt_tokens.push(FmtToken {
            kind: tok.kind,
            lexeme: tok.lexeme,
            line: tok.line,
            col: tok.column,
            leading,
        });

        byte_cursor = tok_byte_start + tok.lexeme.len();
    }

    fmt_tokens
}

/// Find the byte offset of a (line, col) position starting from `from`.
fn find_byte_offset(source: &str, from: usize, target_line: u32, target_col: u32) -> usize {
    let bytes = source.as_bytes();
    let mut line = 1u32;
    let mut col = 1u32;
    let mut i = 0usize;

    // Scan from 0 to `from` to correctly track (line, col) at `from`.
    while i < from && i < bytes.len() {
        if bytes[i] == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
        i += 1;
    }

    // Continue from `from` to the target (line, col).
    while i < bytes.len() && (line < target_line || col < target_col) {
        if bytes[i] == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
        i += 1;
    }
    i
}

/// Extract trivia from source[range_start..range_end].
fn extract_trivia<'a>(source: &'a str, range_start: usize, range_end: usize) -> Vec<Trivia<'a>> {
    if range_start >= range_end {
        return Vec::new();
    }

    let slice = &source[range_start..range_end];
    let bytes = slice.as_bytes();
    let mut trivia = Vec::new();
    let mut i = 0usize;
    let mut blank_count = 0u32;

    while i < bytes.len() {
        // Line comment
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            if blank_count > 0 {
                trivia.push(Trivia::BlankLines(blank_count));
                blank_count = 0;
            }
            let start = i;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            trivia.push(Trivia::LineComment(&slice[start..i]));
            continue;
        }

        // Block comment
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            if blank_count > 0 {
                trivia.push(Trivia::BlankLines(blank_count));
                blank_count = 0;
            }
            let start = i;
            let mut depth = 1;
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            trivia.push(Trivia::BlockComment(&slice[start..i]));
            continue;
        }

        // Newline
        if bytes[i] == b'\n' {
            if blank_count > 0 {
                blank_count += 1;
            } else {
                trivia.push(Trivia::Newline);
            }
            i += 1;
            continue;
        }

        // Carriage return
        if bytes[i] == b'\r' {
            i += 1;
            continue;
        }

        // Whitespace (space/tab)
        if bytes[i] == b' ' || bytes[i] == b'\t' {
            // Check if this is a blank line (whitespace followed by newline)
            let mut j = i;
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'\n' {
                // Blank line
                blank_count += 1;
                i = j; // Move to the newline; it will be processed next iteration
            } else {
                // Inline whitespace — skip (formatter recomputes)
                i = j;
            }
            continue;
        }

        // Unexpected character — skip
        i += 1;
    }

    if blank_count > 0 {
        trivia.push(Trivia::BlankLines(blank_count));
    }

    trivia
}

/// Formatter configuration.
#[derive(Debug, Clone)]
pub struct FmtConfig {
    pub indent_width: u32,
    pub max_blank_lines: u32,
}

impl Default for FmtConfig {
    fn default() -> Self {
        Self {
            indent_width: 4,
            max_blank_lines: 2,
        }
    }
}

/// The format engine: walks FmtTokens and produces formatted output.
pub struct FormatEngine<'a> {
    tokens: &'a [FmtToken<'a>],
    config: &'a FmtConfig,
    out: String,
    indent: u32,
    line_start: bool,
}

impl<'a> FormatEngine<'a> {
    pub fn new(tokens: &'a [FmtToken<'a>], config: &'a FmtConfig) -> Self {
        Self {
            tokens,
            config,
            out: String::with_capacity(tokens.len() * 8),
            indent: 0,
            line_start: true,
        }
    }

    /// Format the token stream into a string.
    pub fn format(mut self) -> String {
        for (i, tok) in self.tokens.iter().enumerate() {
            self.emit_leading_trivia(i);
            self.emit_token(tok, i);
        }
        // Ensure trailing newline
        if !self.out.ends_with('\n') && !self.out.is_empty() {
            self.out.push('\n');
        }
        self.out
    }

    fn emit_leading_trivia(&mut self, token_idx: usize) {
        let trivia = &self.tokens[token_idx].leading;
        let mut blank_count = 0u32;

        for t in trivia {
            match t {
                Trivia::LineComment(s) => {
                    if blank_count > 0 {
                        let preserved = blank_count.min(self.config.max_blank_lines);
                        for _ in 0..preserved {
                            self.emit_newline();
                        }
                        blank_count = 0;
                    }
                    self.emit_indent();
                    self.out.push_str(s);
                    // Don't emit newline here — the subsequent Newline/BlankLines
                    // trivia handles line breaks. Emitting here would cause the
                    // next Newline trivia to be counted as a blank line, breaking
                    // idempotency (each format pass adds an extra blank line).
                }
                Trivia::BlockComment(s) => {
                    if blank_count > 0 {
                        let preserved = blank_count.min(self.config.max_blank_lines);
                        for _ in 0..preserved {
                            self.emit_newline();
                        }
                        blank_count = 0;
                    }
                    self.emit_indent();
                    self.out.push_str(s);
                    // Block comment may or may not be followed by newline
                }
                Trivia::Newline => {
                    blank_count += 1;
                }
                Trivia::BlankLines(n) => {
                    blank_count += n;
                }
                Trivia::Whitespace(_) => {
                    // Discard — formatter recomputes spacing
                }
            }
        }

        // Emit preserved blank lines
        if blank_count > 0 {
            // Don't emit blank lines at the very start of output
            if !self.out.is_empty() {
                let preserved = blank_count.min(self.config.max_blank_lines);
                for _ in 0..preserved {
                    self.emit_newline();
                }
            }
        }
    }

    fn emit_token(&mut self, tok: &FmtToken, idx: usize) {
        // Adjust indent before emitting
        if tok.kind == TokenKind::RBrace {
            if self.indent > 0 {
                self.indent -= 1;
            }
        }

        // Emit indent if at line start
        self.emit_indent();

        // Emit the token
        self.out.push_str(tok.lexeme);
        self.line_start = false;

        // Adjust indent after emitting
        if tok.kind == TokenKind::LBrace {
            self.indent += 1;
        }

        // Emit spacing after this token (look ahead to next token)
        if idx + 1 < self.tokens.len() {
            let next = &self.tokens[idx + 1];
            // Don't emit spacing before the sentinel Eof token
            if next.kind == TokenKind::Eof {
                return;
            }
            let action = decide_spacing(tok.kind, next.kind);

            // Don't emit trailing space before newline
            match action {
                Action::Space => {
                    self.out.push(' ');
                }
                Action::NoSpace => {}
                Action::Newline => {
                    self.emit_newline();
                }
                Action::Newlines(n) => {
                    self.emit_newline();
                    for _ in 0..n {
                        self.emit_newline();
                    }
                }
                Action::KeepOriginal => {
                    self.out.push(' ');
                }
            }
        }
    }

    fn emit_indent(&mut self) {
        if self.line_start {
            for _ in 0..self.indent * self.config.indent_width {
                self.out.push(' ');
            }
            self.line_start = false;
        }
    }

    fn emit_newline(&mut self) {
        self.out.push('\n');
        self.line_start = true;
    }
}

/// Top-level format function: source → formatted source.
/// On parse error, degrades to token-only formatting (no AST structure).
pub fn format(source: &str, config: &FmtConfig) -> String {
    let tokens = tokenize_with_trivia(source);
    let engine = FormatEngine::new(&tokens, config);
    engine.format()
}
