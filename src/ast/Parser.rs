//! Parser.rs — Kuzo lexical and syntax analysis
//!
//! Split from Ast.rs. Contains: BinaryOp precedence table, Lexer (TokenKind/Token/TokenSink/Lexer),
//! ParseError/ParseErrorHandler, recursive-descent + Pratt Parser, and related helper functions.
//! Depends on crate::Ast (AST data model + AstArena + node enums).

use crate::ast::Ast::{
    AstArena, AssociatedType, Attribute, BinaryOp, CompoundAssignOp, ConstructorDef,
    ConstructorField, Decl, DelegateInfo, Expr, ExprRef, ImportItem,
    InterpolationPart, Kind, LambdaBody, MatchArm, MethodDecl, Module, Param,
    Pattern, PatternLiteral, PatternRecordField, PatternRef,
    RecordFieldExpr, RecordFieldType, SelectArm, Span, Spanned, Stmt,
    StmtRef, TraitBound, TypeConstraint, TypeDef, TypeNode, TypeParam,
    TypeRef, UnaryOp, Visibility,
};
// BinaryOp precedence table
//
// A single flat registry with numeric precedences, driving a single Pratt parser.
// Replaces 13 layers of parseXxx template functions (parseElvis/Or/And/BitOr/BitXor/BitAnd/Shift/
// Equality/Comparison/Range/Addition/Multiplication).
//
// Adding a new binary operator only requires appending an entry to BINARY_OPS; the parser needs no changes.


/// A single operator mapping: token kind -> BinaryOp + precedence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpMapping {
    pub token: TokenKind,
    pub op: BinaryOp,
    /// Higher value means tighter binding
    pub precedence: u8,
    /// Only `*` (multiplication vs deref ambiguity) requires cross-line checking
    pub check_multiline_deref: bool,
    /// Right-associative (e.g. the `??` elvis operator)
    pub right_assoc: bool,
}

// Precedence constants (from low to high)
pub const ELVIS_PREC: u8 = 1;
pub const OR_PREC: u8 = 2;
pub const AND_PREC: u8 = 3;
pub const BIT_OR_PREC: u8 = 4;
pub const BIT_XOR_PREC: u8 = 5;
pub const BIT_AND_PREC: u8 = 6;
pub const SHIFT_PREC: u8 = 7;
pub const EQUALITY_PREC: u8 = 8;
pub const COMPARISON_PREC: u8 = 9;
pub const RANGE_PREC: u8 = 10;
pub const ADDITION_PREC: u8 = 11;
pub const MULTIPLICATION_PREC: u8 = 12;

/// Lowest precedence (Pratt parser entry point)
pub const MIN_PREC: u8 = ELVIS_PREC;

/// Flat binary operator registry (single source of truth)
///
/// To add a new operator, simply append an entry here.
pub const BINARY_OPS: &[OpMapping] = &[
    // Elvis ?? (lowest, right-associative)
    OpMapping {
        token: TokenKind::QuestionQuestion,
        op: BinaryOp::Elvis,
        precedence: ELVIS_PREC,
        check_multiline_deref: false,
        right_assoc: true,
    },
    // Logical or ||
    OpMapping {
        token: TokenKind::PipePipe,
        op: BinaryOp::Or,
        precedence: OR_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Logical and &&
    OpMapping {
        token: TokenKind::AmpAmp,
        op: BinaryOp::And,
        precedence: AND_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Bitwise or |
    OpMapping {
        token: TokenKind::Pipe,
        op: BinaryOp::BitOr,
        precedence: BIT_OR_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Bitwise xor ^
    OpMapping {
        token: TokenKind::Caret,
        op: BinaryOp::BitXor,
        precedence: BIT_XOR_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Bitwise and &
    OpMapping {
        token: TokenKind::Ampersand,
        op: BinaryOp::BitAnd,
        precedence: BIT_AND_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Shift << >>
    OpMapping {
        token: TokenKind::LtLt,
        op: BinaryOp::Shl,
        precedence: SHIFT_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::GtGt,
        op: BinaryOp::Shr,
        precedence: SHIFT_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Equality == != === !==
    OpMapping {
        token: TokenKind::EqEq,
        op: BinaryOp::Eq,
        precedence: EQUALITY_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::BangEq,
        op: BinaryOp::NotEq,
        precedence: EQUALITY_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::RefEq,
        op: BinaryOp::RefEq,
        precedence: EQUALITY_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::RefNeq,
        op: BinaryOp::RefNeq,
        precedence: EQUALITY_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Comparison < > <= >=
    OpMapping {
        token: TokenKind::Lt,
        op: BinaryOp::Lt,
        precedence: COMPARISON_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::Gt,
        op: BinaryOp::Gt,
        precedence: COMPARISON_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::LtEq,
        op: BinaryOp::LtEq,
        precedence: COMPARISON_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::GtEq,
        op: BinaryOp::GtEq,
        precedence: COMPARISON_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Range .. ..=
    OpMapping {
        token: TokenKind::DotDot,
        op: BinaryOp::Range,
        precedence: RANGE_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::DotDotEq,
        op: BinaryOp::RangeInclusive,
        precedence: RANGE_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Add/sub + ++ -
    OpMapping {
        token: TokenKind::Plus,
        op: BinaryOp::Add,
        precedence: ADDITION_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::PlusPlus,
        op: BinaryOp::ConcatList,
        precedence: ADDITION_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::Minus,
        op: BinaryOp::Sub,
        precedence: ADDITION_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    // Mul/div/mod * / % (`*` requires cross-line deref checking)
    OpMapping {
        token: TokenKind::Star,
        op: BinaryOp::Mul,
        precedence: MULTIPLICATION_PREC,
        check_multiline_deref: true,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::Slash,
        op: BinaryOp::Div,
        precedence: MULTIPLICATION_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
    OpMapping {
        token: TokenKind::Percent,
        op: BinaryOp::Mod,
        precedence: MULTIPLICATION_PREC,
        check_multiline_deref: false,
        right_assoc: false,
    },
];

/// Look up a binary operator mapping by token kind; returns `None` if not found
pub fn lookup_binary_op(tok: TokenKind) -> Option<&'static OpMapping> {
    BINARY_OPS.iter().find(|m| m.token == tok)
}

// Lexer
//
// Scans a Kuzo source string character-by-character into a Token sequence. Supports keywords,
// identifiers, integers (binary/octal/hexadecimal), floating-point numbers, character and string
// literals (with interpolation), and various operators and delimiters. Tokens carry line/column
// information for error reporting.
//
// Semantically corresponds to the Zig original `src/parse/lexer.zig`, but rewritten using Rust idioms.
// Semicolons `;` are treated as whitespace and skipped; on lexical errors an `Err` token is emitted
// and scanning continues, so that the parser can collect more errors.

// =========================================================================
// TokenKind: covers all literals, keywords, operators, and delimiters
// =========================================================================

/// Lexical token kind: covers all literals, keywords, operators, and delimiters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // --- Literals (7) ---
    IntLiteral,
    FloatLiteral,
    CharLiteral,
    StringLiteral,
    TrueLiteral,
    FalseLiteral,
    NullLiteral,

    // --- Keywords (29) ---
    KwFun,
    KwType,
    KwTrait,
    KwOverride,
    KwPack,
    KwPub,
    KwImport,
    KwWith,
    KwAs,
    KwVal,
    KwVar,
    KwMatch,
    KwIf,
    KwElse,
    KwAsync,
    KwChannel,
    KwSelect,
    KwAtomic,
    KwLoop,
    KwFor,
    KwIn,
    KwWhile,
    KwBreak,
    KwContinue,
    KwReturn,
    KwThrow,
    KwLazy,
    KwDefer,
    KwThis,

    // Identifier
    Identifier,

    // --- Operators (42) ---
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqEq,
    BangEq,
    RefEq,
    RefNeq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    LtMinus,
    AmpAmp,
    PipePipe,
    Bang,
    Ampersand,
    Caret,
    QuestionDot,
    QuestionQuestion,
    Question,
    DotDot,
    DotDotEq,
    Ellipsis,
    Eq,
    PlusEq,
    PlusPlus,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    AmpEq,
    PipeEq,
    CaretEq,
    LtLt,
    GtGt,
    LtLtEq,
    GtGtEq,
    Tilde,
    EqGt,
    MinusGt,

    // --- Delimiters (10) ---
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Dot,
    Pipe,

    // --- Attributes and raw blocks ---
    At,        // @
    RawBlock,  // #{ ... }# raw block (lexeme is the inner content, excluding #{ and }#)

    // --- Special (2) ---
    Eof,
    Err,
}

// =========================================================================
// Token
// =========================================================================

/// Lexical token: kind, literal text (zero-copy reference into source), line, and column
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token<'a> {
    pub kind: TokenKind,
    pub lexeme: &'a str,
    pub line: u32,
    pub column: u32,
}

// =========================================================================
// LexerError
// =========================================================================

/// Error types that may occur during lexical analysis
#[derive(Debug, Clone)]
pub enum LexerError {
    UnterminatedString,
    UnterminatedChar,
    UnterminatedComment,
    InvalidEscape,
    InvalidUnicodeEscape,
    InvalidNumber,
    InvalidHexDigit,
    InvalidOctalDigit,
    InvalidBinaryDigit,
}

// =========================================================================
// TokenSink — Token receiver trait
// =========================================================================

/// Token receiver trait. The Lexer calls `emit_token` for each Token it produces.
/// The default implementation `TokenCollector` collects into a `Vec<Token>`.
pub trait TokenSink<'a> {
    fn emit_token(&mut self, token: Token<'a>);
}

/// Default receiver: collects into a Vec
pub struct TokenCollector<'a> {
    pub tokens: Vec<Token<'a>>,
}

impl<'a> TokenCollector<'a> {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    /// Consumes self and returns the collected Token list
    pub fn into_tokens(self) -> Vec<Token<'a>> {
        self.tokens
    }
}

impl<'a> Default for TokenCollector<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> TokenSink<'a> for TokenCollector<'a> {
    fn emit_token(&mut self, token: Token<'a>) {
        self.tokens.push(token);
    }
}

// =========================================================================
// Lexer
// =========================================================================

/// Lexer: holds the source code, scan position, line, and column
pub struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: u32,
    column: u32,
}

impl<'a> Lexer<'a> {
    /// Creates a lexer
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }

    /// Scans the entire source and streams Tokens to the sink, appending `Eof` at the end.
    ///
    /// On lexical errors, emits an `Err` token and continues scanning (does not abort),
    /// so that the parser can collect more errors.
    pub fn tokenize_into<S: TokenSink<'a>>(&mut self, sink: &mut S) {
        while self.pos < self.bytes.len() {
            let start = self.pos;
            let start_line = self.line;
            let start_col = self.column;
            match self.scan_token() {
                Ok(Some(tok)) => sink.emit_token(tok),
                Ok(None) => {}
                Err(_) => {
                    // Emit an error Token covering the consumed range, then continue scanning
                    sink.emit_token(Token {
                        kind: TokenKind::Err,
                        lexeme: &self.source[start..self.pos],
                        line: start_line,
                        column: start_col,
                    });
                }
            }
        }
        sink.emit_token(Token {
            kind: TokenKind::Eof,
            lexeme: "",
            line: self.line,
            column: self.column,
        });
    }

    // --- Basic character operations ---

    /// Consume the current character and advance; updates line/column on newline
    fn advance(&mut self) -> Option<u8> {
        let ch = *self.bytes.get(self.pos)?;
        self.pos += 1;
        if ch == b'\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }

    /// Consume and advance if the current character equals the expected one; returns whether it matched
    fn match_char(&mut self, expected: u8) -> bool {
        if self.pos >= self.bytes.len() {
            return false;
        }
        if self.bytes[self.pos] != expected {
            return false;
        }
        self.pos += 1;
        if expected == b'\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        true
    }

    /// Build a Token from start/end positions and line/column (lexeme is a zero-copy reference into source)
    fn make_token(&self, kind: TokenKind, start: usize, start_line: u32, start_col: u32) -> Token<'a> {
        Token {
            kind,
            lexeme: &self.source[start..self.pos],
            line: start_line,
            column: start_col,
        }
    }

    // --- Single-token scanning ---

    /// Scan a single token: dispatch to the matching branch based on the first character.
    /// Returns `Ok(None)` for whitespace/comments/semicolons (no Token produced).
    fn scan_token(&mut self) -> Result<Option<Token<'a>>, LexerError> {
        let start = self.pos;
        let start_line = self.line;
        let start_col = self.column;
        let ch = match self.advance() {
            Some(c) => c,
            None => return Ok(None),
        };
        match ch {
            // Whitespace is skipped directly
            b' ' | b'\t' | b'\r' | b'\n' => Ok(None),
            b'/' => {
                if self.match_char(b'/') {
                    self.skip_line_comment();
                    Ok(None)
                } else if self.match_char(b'*') {
                    self.skip_block_comment()?;
                    Ok(None)
                } else if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::SlashEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Slash, start, start_line, start_col)))
                }
            }
            b'(' => Ok(Some(self.make_token(TokenKind::LParen, start, start_line, start_col))),
            b')' => Ok(Some(self.make_token(TokenKind::RParen, start, start_line, start_col))),
            b'[' => Ok(Some(self.make_token(TokenKind::LBracket, start, start_line, start_col))),
            b']' => Ok(Some(self.make_token(TokenKind::RBracket, start, start_line, start_col))),
            b'{' => Ok(Some(self.make_token(TokenKind::LBrace, start, start_line, start_col))),
            b'}' => Ok(Some(self.make_token(TokenKind::RBrace, start, start_line, start_col))),
            b',' => Ok(Some(self.make_token(TokenKind::Comma, start, start_line, start_col))),
            // Semicolons are treated as whitespace and skipped
            b';' => Ok(None),
            b':' => Ok(Some(self.make_token(TokenKind::Colon, start, start_line, start_col))),
            b'%' => {
                if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::PercentEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Percent, start, start_line, start_col)))
                }
            }
            b'+' => {
                if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::PlusEq, start, start_line, start_col)))
                } else if self.match_char(b'+') {
                    Ok(Some(self.make_token(TokenKind::PlusPlus, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Plus, start, start_line, start_col)))
                }
            }
            b'*' => {
                if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::StarEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Star, start, start_line, start_col)))
                }
            }
            b'|' => {
                if self.match_char(b'|') {
                    Ok(Some(self.make_token(TokenKind::PipePipe, start, start_line, start_col)))
                } else if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::PipeEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Pipe, start, start_line, start_col)))
                }
            }
            b'=' => {
                if self.match_char(b'=') {
                    if self.match_char(b'=') {
                        Ok(Some(self.make_token(TokenKind::RefEq, start, start_line, start_col)))
                    } else {
                        Ok(Some(self.make_token(TokenKind::EqEq, start, start_line, start_col)))
                    }
                } else if self.match_char(b'>') {
                    Ok(Some(self.make_token(TokenKind::EqGt, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Eq, start, start_line, start_col)))
                }
            }
            b'!' => {
                if self.match_char(b'=') {
                    if self.match_char(b'=') {
                        Ok(Some(self.make_token(TokenKind::RefNeq, start, start_line, start_col)))
                    } else {
                        Ok(Some(self.make_token(TokenKind::BangEq, start, start_line, start_col)))
                    }
                } else {
                    Ok(Some(self.make_token(TokenKind::Bang, start, start_line, start_col)))
                }
            }
            b'<' => {
                if self.match_char(b'<') {
                    if self.match_char(b'=') {
                        Ok(Some(self.make_token(TokenKind::LtLtEq, start, start_line, start_col)))
                    } else {
                        Ok(Some(self.make_token(TokenKind::LtLt, start, start_line, start_col)))
                    }
                } else if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::LtEq, start, start_line, start_col)))
                } else if self.match_char(b'-') {
                    Ok(Some(self.make_token(TokenKind::LtMinus, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Lt, start, start_line, start_col)))
                }
            }
            b'>' => {
                if self.match_char(b'>') {
                    if self.match_char(b'=') {
                        Ok(Some(self.make_token(TokenKind::GtGtEq, start, start_line, start_col)))
                    } else {
                        Ok(Some(self.make_token(TokenKind::GtGt, start, start_line, start_col)))
                    }
                } else if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::GtEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Gt, start, start_line, start_col)))
                }
            }
            b'-' => {
                if self.match_char(b'>') {
                    Ok(Some(self.make_token(TokenKind::MinusGt, start, start_line, start_col)))
                } else if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::MinusEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Minus, start, start_line, start_col)))
                }
            }
            b'.' => {
                if self.match_char(b'.') {
                    if self.match_char(b'=') {
                        Ok(Some(self.make_token(TokenKind::DotDotEq, start, start_line, start_col)))
                    } else if self.match_char(b'.') {
                        Ok(Some(self.make_token(TokenKind::Ellipsis, start, start_line, start_col)))
                    } else {
                        Ok(Some(self.make_token(TokenKind::DotDot, start, start_line, start_col)))
                    }
                } else {
                    // A lone dot followed by a digit is treated as a .float (e.g. .5)
                    if self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                        self.scan_dot_float(start, start_line, start_col)
                    } else {
                        Ok(Some(self.make_token(TokenKind::Dot, start, start_line, start_col)))
                    }
                }
            }
            b'?' => {
                if self.match_char(b'.') {
                    Ok(Some(self.make_token(TokenKind::QuestionDot, start, start_line, start_col)))
                } else if self.match_char(b'?') {
                    Ok(Some(self.make_token(TokenKind::QuestionQuestion, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Question, start, start_line, start_col)))
                }
            }
            b'&' => {
                if self.match_char(b'&') {
                    Ok(Some(self.make_token(TokenKind::AmpAmp, start, start_line, start_col)))
                } else if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::AmpEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Ampersand, start, start_line, start_col)))
                }
            }
            b'^' => {
                if self.match_char(b'=') {
                    Ok(Some(self.make_token(TokenKind::CaretEq, start, start_line, start_col)))
                } else {
                    Ok(Some(self.make_token(TokenKind::Caret, start, start_line, start_col)))
                }
            }
            b'~' => Ok(Some(self.make_token(TokenKind::Tilde, start, start_line, start_col))),
            b'@' => Ok(Some(self.make_token(TokenKind::At, start, start_line, start_col))),
            b'#' => {
                if self.match_char(b'{') {
                    self.scan_raw_block(start, start_line, start_col)
                } else {
                    Ok(Some(self.make_token(TokenKind::Err, start, start_line, start_col)))
                }
            }
            b'\'' => self.scan_char(start, start_line, start_col),
            b'"' => self.scan_string(start, start_line, start_col),
            b'0'..=b'9' => self.scan_number(start, start_line, start_col),
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.scan_identifier(start, start_line, start_col),
            // Unknown character: emit an error Token (does not abort scanning)
            _ => Ok(Some(self.make_token(TokenKind::Err, start, start_line, start_col))),
        }
    }

    // --- Comments ---

    /// Skip a line comment (// to end of line, does not consume the newline)
    fn skip_line_comment(&mut self) {
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == b'\n' {
                break;
            }
            self.pos += 1;
            self.column += 1;
        }
    }

    /// Skip a block comment (/* */, supports nesting)
    fn skip_block_comment(&mut self) -> Result<(), LexerError> {
        let mut depth: u32 = 1;
        while self.pos < self.bytes.len() && depth > 0 {
            let ch = self.bytes[self.pos];
            if ch == b'/' && self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1] == b'*' {
                depth += 1;
                self.pos += 2;
                self.column += 2;
            } else if ch == b'*' && self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1] == b'/' {
                depth -= 1;
                self.pos += 2;
                self.column += 2;
            } else if ch == b'\n' {
                self.pos += 1;
                self.line += 1;
                self.column = 1;
            } else {
                self.pos += 1;
                self.column += 1;
            }
        }
        if depth > 0 {
            return Err(LexerError::UnterminatedComment);
        }
        Ok(())
    }

    // --- Numbers ---

    /// Scan a numeric literal, auto-detecting binary/octal/hexadecimal prefixes, decimal points, exponents, and type suffixes
    fn scan_number(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        if self.bytes[start] == b'0' && self.pos < self.bytes.len() {
            let prefix = self.bytes[self.pos];
            if prefix == b'x' || prefix == b'X' {
                self.pos += 1;
                self.column += 1;
                return self.scan_hex_number(start, start_line, start_col);
            } else if prefix == b'o' || prefix == b'O' {
                self.pos += 1;
                self.column += 1;
                return self.scan_octal_number(start, start_line, start_col);
            } else if prefix == b'b' || prefix == b'B' {
                self.pos += 1;
                self.column += 1;
                return self.scan_binary_number(start, start_line, start_col);
            }
        }
        while self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
            self.pos += 1;
            self.column += 1;
        }
        self.skip_underscore_digits(false);
        let mut is_float = false;
        // Fractional part
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.'
            && self.pos + 1 < self.bytes.len() && is_digit(self.bytes[self.pos + 1])
        {
            is_float = true;
            self.pos += 1;
            self.column += 1;
            while self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                self.pos += 1;
                self.column += 1;
            }
            self.skip_underscore_digits(false);
        }
        // Exponent part
        if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E') {
            is_float = true;
            self.pos += 1;
            self.column += 1;
            if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-') {
                self.pos += 1;
                self.column += 1;
            }
            if self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                while self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                    self.pos += 1;
                    self.column += 1;
                }
            } else {
                return Err(LexerError::InvalidNumber);
            }
        }
        // Type suffix (e.g. i32, f64): consume all identifier characters; validity is checked by Sema
        if self.pos < self.bytes.len() && is_identifier_start(self.bytes[self.pos]) {
            while self.pos < self.bytes.len() && is_identifier_continue(self.bytes[self.pos]) {
                self.pos += 1;
                self.column += 1;
            }
        }
        let kind = if is_float { TokenKind::FloatLiteral } else { TokenKind::IntLiteral };
        Ok(Some(self.make_token(kind, start, start_line, start_col)))
    }

    /// Skip underscore separators within digits (e.g. 1_000); `hex` controls whether to treat digits as hexadecimal
    fn skip_underscore_digits(&mut self, hex: bool) {
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == b'_' && self.pos + 1 < self.bytes.len() {
                let next = self.bytes[self.pos + 1];
                let valid = if hex { is_hex_digit(next) } else { is_digit(next) };
                if valid {
                    self.pos += 1;
                    self.column += 1;
                    self.pos += 1;
                    self.column += 1;
                    while self.pos < self.bytes.len() {
                        let ch = self.bytes[self.pos];
                        let ok = if hex { is_hex_digit(ch) } else { is_digit(ch) };
                        if !ok {
                            break;
                        }
                        self.pos += 1;
                        self.column += 1;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    /// Scan a float literal that starts with a dot (e.g. .5)
    fn scan_dot_float(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        while self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
            self.pos += 1;
            self.column += 1;
        }
        self.skip_underscore_digits(false);
        if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'e' || self.bytes[self.pos] == b'E') {
            self.pos += 1;
            self.column += 1;
            if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-') {
                self.pos += 1;
                self.column += 1;
            }
            if self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                while self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                    self.pos += 1;
                    self.column += 1;
                }
            } else {
                return Err(LexerError::InvalidNumber);
            }
        }
        // Type suffix: consume all identifier characters; validity is checked by Sema
        if self.pos < self.bytes.len() && is_identifier_start(self.bytes[self.pos]) {
            while self.pos < self.bytes.len() && is_identifier_continue(self.bytes[self.pos]) {
                self.pos += 1;
                self.column += 1;
            }
        }
        Ok(Some(self.make_token(TokenKind::FloatLiteral, start, start_line, start_col)))
    }

    /// Scan a hexadecimal numeric literal (0x prefix), supporting hexadecimal fractions and p exponents
    fn scan_hex_number(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        let mut has_digits = false;
        while self.pos < self.bytes.len() && is_hex_digit(self.bytes[self.pos]) {
            has_digits = true;
            self.pos += 1;
            self.column += 1;
        }
        self.skip_underscore_digits(true);
        let mut is_float = false;
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'.'
            && self.pos + 1 < self.bytes.len()
            && (is_hex_digit(self.bytes[self.pos + 1])
                || self.bytes[self.pos + 1] == b'p'
                || self.bytes[self.pos + 1] == b'P')
        {
            is_float = true;
            self.pos += 1;
            self.column += 1;
            while self.pos < self.bytes.len() && is_hex_digit(self.bytes[self.pos]) {
                self.pos += 1;
                self.column += 1;
            }
            self.skip_underscore_digits(true);
        }
        if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'p' || self.bytes[self.pos] == b'P') {
            is_float = true;
            self.pos += 1;
            self.column += 1;
            if self.pos < self.bytes.len() && (self.bytes[self.pos] == b'+' || self.bytes[self.pos] == b'-') {
                self.pos += 1;
                self.column += 1;
            }
            if self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                while self.pos < self.bytes.len() && is_digit(self.bytes[self.pos]) {
                    self.pos += 1;
                    self.column += 1;
                }
            } else {
                return Err(LexerError::InvalidNumber);
            }
        }
        if !has_digits {
            return Err(LexerError::InvalidHexDigit);
        }
        // Type suffix: consume all identifier characters; validity is checked by Sema
        if self.pos < self.bytes.len() && is_identifier_start(self.bytes[self.pos]) {
            while self.pos < self.bytes.len() && is_identifier_continue(self.bytes[self.pos]) {
                self.pos += 1;
                self.column += 1;
            }
        }
        let kind = if is_float { TokenKind::FloatLiteral } else { TokenKind::IntLiteral };
        Ok(Some(self.make_token(kind, start, start_line, start_col)))
    }

    /// Scan an octal numeric literal (0o prefix)
    fn scan_octal_number(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        let mut has_digits = false;
        while self.pos < self.bytes.len() && is_octal_digit(self.bytes[self.pos]) {
            has_digits = true;
            self.pos += 1;
            self.column += 1;
        }
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == b'_'
                && self.pos + 1 < self.bytes.len()
                && is_octal_digit(self.bytes[self.pos + 1])
            {
                has_digits = true;
                self.pos += 1;
                self.column += 1;
                self.pos += 1;
                self.column += 1;
                while self.pos < self.bytes.len() && is_octal_digit(self.bytes[self.pos]) {
                    self.pos += 1;
                    self.column += 1;
                }
            } else {
                break;
            }
        }
        if !has_digits {
            return Err(LexerError::InvalidOctalDigit);
        }
        // Type suffix: consume all identifier characters; validity is checked by Sema
        if self.pos < self.bytes.len() && is_identifier_start(self.bytes[self.pos]) {
            while self.pos < self.bytes.len() && is_identifier_continue(self.bytes[self.pos]) {
                self.pos += 1;
                self.column += 1;
            }
        }
        Ok(Some(self.make_token(TokenKind::IntLiteral, start, start_line, start_col)))
    }

    /// Scan a binary numeric literal (0b prefix)
    fn scan_binary_number(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        let mut has_digits = false;
        while self.pos < self.bytes.len() && is_binary_digit(self.bytes[self.pos]) {
            has_digits = true;
            self.pos += 1;
            self.column += 1;
        }
        while self.pos < self.bytes.len() {
            if self.bytes[self.pos] == b'_'
                && self.pos + 1 < self.bytes.len()
                && is_binary_digit(self.bytes[self.pos + 1])
            {
                has_digits = true;
                self.pos += 1;
                self.column += 1;
                self.pos += 1;
                self.column += 1;
                while self.pos < self.bytes.len() && is_binary_digit(self.bytes[self.pos]) {
                    self.pos += 1;
                    self.column += 1;
                }
            } else {
                break;
            }
        }
        if !has_digits {
            return Err(LexerError::InvalidBinaryDigit);
        }
        if self.pos < self.bytes.len() && is_identifier_start(self.bytes[self.pos]) {
            while self.pos < self.bytes.len() && is_identifier_continue(self.bytes[self.pos]) {
                self.pos += 1;
                self.column += 1;
            }
        }
        Ok(Some(self.make_token(TokenKind::IntLiteral, start, start_line, start_col)))
    }

    // --- Characters ---

    /// Scan a character literal ('x'), supporting escapes and Unicode escapes \u{...}
    fn scan_char(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        if self.pos >= self.bytes.len() {
            return Err(LexerError::UnterminatedChar);
        }
        if self.bytes[self.pos] == b'\\' {
            self.pos += 1;
            self.column += 1;
            if self.pos >= self.bytes.len() {
                return Err(LexerError::UnterminatedChar);
            }
            let escaped = self.bytes[self.pos];
            match escaped {
                b'n' | b't' | b'r' | b'\\' | b'\'' | b'0' => {
                    self.pos += 1;
                    self.column += 1;
                }
                b'u' => {
                    // Bug #36: support \uXXXX (4-digit hex) and \u{XXXX} (brace form)
                    self.pos += 1;
                    self.column += 1;
                    if self.pos >= self.bytes.len() {
                        return Err(LexerError::InvalidUnicodeEscape);
                    }
                    if self.bytes[self.pos] == b'{' {
                        // \u{XXXX} brace form: 1-6 hex digits
                        self.pos += 1;
                        self.column += 1;
                        let mut digit_count: usize = 0;
                        while self.pos < self.bytes.len() && self.bytes[self.pos] != b'}' {
                            if !is_hex_digit(self.bytes[self.pos]) {
                                return Err(LexerError::InvalidUnicodeEscape);
                            }
                            self.pos += 1;
                            self.column += 1;
                            digit_count += 1;
                        }
                        if digit_count == 0 || self.pos >= self.bytes.len() {
                            return Err(LexerError::InvalidUnicodeEscape);
                        }
                        self.pos += 1;
                        self.column += 1;
                    } else {
                        // \uXXXX without braces: exactly 4 hex digits
                        for _ in 0..4 {
                            if self.pos >= self.bytes.len() || !is_hex_digit(self.bytes[self.pos]) {
                                return Err(LexerError::InvalidUnicodeEscape);
                            }
                            self.pos += 1;
                            self.column += 1;
                        }
                    }
                }
                b'x' => {
                    // \xHH: exactly 2 hex digits, byte value 0x00-0xFF
                    self.pos += 1;
                    self.column += 1;
                    for _ in 0..2 {
                        if self.pos >= self.bytes.len() || !is_hex_digit(self.bytes[self.pos]) {
                            return Err(LexerError::InvalidEscape);
                        }
                        self.pos += 1;
                        self.column += 1;
                    }
                }
                _ => {
                    return Err(LexerError::InvalidEscape);
                }
            }
        } else {
            // Non-ASCII character: a multi-byte UTF-8 sequence, advance by character boundary
            // Avoid advancing by single byte which would leave pos mid-character and cause a slice panic
            let ch_start = self.pos;
            let first = self.bytes[self.pos];
            let utf8_len = if first < 0x80 {
                1
            } else if first < 0xC0 {
                1 // Invalid UTF-8 leading byte; advance by 1 byte for fault tolerance
            } else if first < 0xE0 {
                2
            } else if first < 0xF0 {
                3
            } else {
                4
            };
            let end = std::cmp::min(ch_start + utf8_len, self.bytes.len());
            self.pos = end;
            self.column += 1;
        }
        if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'\'' {
            return Err(LexerError::UnterminatedChar);
        }
        self.pos += 1;
        self.column += 1;
        Ok(Some(self.make_token(TokenKind::CharLiteral, start, start_line, start_col)))
    }

    // --- Strings ---

    /// Scan a string literal, supporting escapes, `{{`/`}}` literal braces, and `{expression}` interpolation.
    ///
    /// The entire string literal (including interpolation parts) is emitted as a single `StringLiteral` Token;
    /// the lexeme contains the raw text. Bare newlines are not allowed inside strings.
    fn scan_string(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        while self.pos < self.bytes.len() {
            let ch = self.bytes[self.pos];
            if ch == b'"' {
                self.pos += 1;
                self.column += 1;
                return Ok(Some(self.make_token(TokenKind::StringLiteral, start, start_line, start_col)));
            }
            if ch == b'\\' {
                self.pos += 1;
                self.column += 1;
                if self.pos >= self.bytes.len() {
                    return Err(LexerError::UnterminatedString);
                }
                let escaped = self.bytes[self.pos];
                match escaped {
                    b'"' | b'\\' | b'n' | b't' | b'r' | b'{' | b'}' | b'0' => {
                        self.pos += 1;
                        self.column += 1;
                    }
                    b'u' => {
                        // Bug #36: support \uXXXX (4-digit hex) and \u{XXXX} (brace form)
                        self.pos += 1;
                        self.column += 1;
                        if self.pos >= self.bytes.len() {
                            return Err(LexerError::InvalidUnicodeEscape);
                        }
                        if self.bytes[self.pos] == b'{' {
                            // \u{XXXX} brace form: 1-6 hex digits
                            self.pos += 1;
                            self.column += 1;
                            let mut digit_count: usize = 0;
                            while self.pos < self.bytes.len() && self.bytes[self.pos] != b'}' {
                                if !is_hex_digit(self.bytes[self.pos]) {
                                    return Err(LexerError::InvalidUnicodeEscape);
                                }
                                self.pos += 1;
                                self.column += 1;
                                digit_count += 1;
                            }
                            if digit_count == 0 || self.pos >= self.bytes.len() {
                                return Err(LexerError::InvalidUnicodeEscape);
                            }
                            self.pos += 1;
                            self.column += 1;
                        } else {
                            // \uXXXX without braces: exactly 4 hex digits
                            for _ in 0..4 {
                                if self.pos >= self.bytes.len() || !is_hex_digit(self.bytes[self.pos]) {
                                    return Err(LexerError::InvalidUnicodeEscape);
                                }
                                self.pos += 1;
                                self.column += 1;
                            }
                        }
                    }
                    b'x' => {
                        // \xHH: exactly 2 hex digits, byte value 0x00-0xFF
                        self.pos += 1;
                        self.column += 1;
                        for _ in 0..2 {
                            if self.pos >= self.bytes.len() || !is_hex_digit(self.bytes[self.pos]) {
                                return Err(LexerError::InvalidEscape);
                            }
                            self.pos += 1;
                            self.column += 1;
                        }
                    }
                    _ => {
                        return Err(LexerError::InvalidEscape);
                    }
                }
            } else if ch == b'{' {
                // {{ denotes a literal {; otherwise enter interpolation expression scanning
                if self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1] == b'{' {
                    self.pos += 2;
                    self.column += 2;
                } else {
                    self.pos += 1;
                    self.column += 1;
                    let mut brace_depth: u32 = 1;
                    while self.pos < self.bytes.len() && brace_depth > 0 {
                        let inner = self.bytes[self.pos];
                        if inner == b'\\' {
                            self.pos += 1;
                            self.column += 1;
                            if self.pos < self.bytes.len() {
                                self.pos += 1;
                                self.column += 1;
                            }
                            continue;
                        } else if inner == b'{' {
                            brace_depth += 1;
                        } else if inner == b'}' {
                            brace_depth -= 1;
                        } else if inner == b'"' {
                            // Bug #44/#46/#54: nested string literal — scan the complete nested string
                            // (including \" escapes), to avoid mistaking the outer string's closing quote for the start of a nested string
                            self.pos += 1;
                            self.column += 1;
                            while self.pos < self.bytes.len() {
                                let nc = self.bytes[self.pos];
                                if nc == b'\\' {
                                    self.pos += 1;
                                    self.column += 1;
                                    if self.pos < self.bytes.len() {
                                        self.pos += 1;
                                        self.column += 1;
                                    }
                                    continue;
                                }
                                if nc == b'"' {
                                    break;
                                }
                                if nc == b'\n' {
                                    self.line += 1;
                                    self.column = 1;
                                } else {
                                    self.column += 1;
                                }
                                self.pos += 1;
                            }
                            if self.pos < self.bytes.len() && self.bytes[self.pos] == b'"' {
                                self.pos += 1;
                                self.column += 1;
                            }
                            continue;
                        }
                        if inner == b'\n' {
                            self.line += 1;
                            self.column = 1;
                        } else {
                            self.column += 1;
                        }
                        self.pos += 1;
                    }
                }
            } else if ch == b'}' {
                // }} denotes a literal }
                if self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1] == b'}' {
                    self.pos += 2;
                    self.column += 2;
                } else {
                    self.pos += 1;
                    self.column += 1;
                }
            } else if ch == b'\n' {
                return Err(LexerError::UnterminatedString);
            } else {
                self.pos += 1;
                self.column += 1;
            }
        }
        Err(LexerError::UnterminatedString)
    }

    /// Scan a raw block #{ ... }#: scan character-by-character until a matching }# is found; the lexeme is the inner content (excluding #{ and }#).
    fn scan_raw_block(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        let content_start = self.pos;
        while self.pos < self.bytes.len() {
            let ch = self.bytes[self.pos];
            if ch == b'}' && self.pos + 1 < self.bytes.len() && self.bytes[self.pos + 1] == b'#' {
                let content = &self.source[content_start..self.pos];
                self.pos += 2;
                self.column += 2;
                // Emit a single Token whose lexeme is the inner content
                // Note: make_token uses start..self.pos which would include #{ and }#, so we construct it manually
                let _ = start;
                return Ok(Some(Token {
                    kind: TokenKind::RawBlock,
                    lexeme: content,
                    line: start_line,
                    column: start_col,
                }));
            }
            if ch == b'\n' {
                self.pos += 1;
                self.line += 1;
                self.column = 1;
            } else {
                self.pos += 1;
                self.column += 1;
            }
        }
        Err(LexerError::UnterminatedString)
    }

    // --- Identifiers ---

    /// Scan an identifier or keyword; the keyword table determines the final Token kind
    fn scan_identifier(&mut self, start: usize, start_line: u32, start_col: u32) -> Result<Option<Token<'a>>, LexerError> {
        while self.pos < self.bytes.len() && is_identifier_continue(self.bytes[self.pos]) {
            self.pos += 1;
            self.column += 1;
        }
        let text = &self.source[start..self.pos];
        let kind = keyword_type(text);
        Ok(Some(self.make_token(kind, start, start_line, start_col)))
    }
}

// =========================================================================
// Helper functions
// =========================================================================

/// Whether the character is a decimal digit
fn is_digit(ch: u8) -> bool {
    ch.is_ascii_digit()
}

/// Whether the character is a hexadecimal digit
fn is_hex_digit(ch: u8) -> bool {
    ch.is_ascii_hexdigit()
}

/// Whether the character is an octal digit
fn is_octal_digit(ch: u8) -> bool {
    (b'0'..=b'7').contains(&ch)
}

/// Whether the character is a binary digit
fn is_binary_digit(ch: u8) -> bool {
    ch == b'0' || ch == b'1'
}

/// Whether the character may start an identifier (ASCII only)
fn is_identifier_start(ch: u8) -> bool {
    ch.is_ascii_alphabetic() || ch == b'_'
}

/// Whether the character may continue an identifier (ASCII only)
fn is_identifier_continue(ch: u8) -> bool {
    is_identifier_start(ch) || is_digit(ch)
}

/// Look up whether the text is a keyword; otherwise returns `Identifier`
fn keyword_type(text: &str) -> TokenKind {
    match text {
        "fun" => TokenKind::KwFun,
        "type" => TokenKind::KwType,
        "trait" => TokenKind::KwTrait,
        "override" => TokenKind::KwOverride,
        "pack" => TokenKind::KwPack,
        "pub" => TokenKind::KwPub,
        "import" => TokenKind::KwImport,
        "with" => TokenKind::KwWith,
        "as" => TokenKind::KwAs,
        "val" => TokenKind::KwVal,
        "var" => TokenKind::KwVar,
        "match" => TokenKind::KwMatch,
        "if" => TokenKind::KwIf,
        "else" => TokenKind::KwElse,
        "async" => TokenKind::KwAsync,
        "channel" => TokenKind::KwChannel,
        "select" => TokenKind::KwSelect,
        "atomic" => TokenKind::KwAtomic,
        "loop" => TokenKind::KwLoop,
        "for" => TokenKind::KwFor,
        "in" => TokenKind::KwIn,
        "while" => TokenKind::KwWhile,
        "break" => TokenKind::KwBreak,
        "continue" => TokenKind::KwContinue,
        "return" => TokenKind::KwReturn,
        "throw" => TokenKind::KwThrow,
        "lazy" => TokenKind::KwLazy,
        "defer" => TokenKind::KwDefer,
        "this" => TokenKind::KwThis,
        "true" => TokenKind::TrueLiteral,
        "false" => TokenKind::FalseLiteral,
        "null" => TokenKind::NullLiteral,
        _ => TokenKind::Identifier,
    }
}


// Recursive-descent parser
//
// Parses a Token sequence into an AST. Core features:
// - Recursive descent + Pratt precedence climbing (driven by binary_op_table)
// - Virtual token splitting (`>>` -> two `>`, `>=` -> `>` + `=`, `>>=` -> `>` + `>=`)
// - Negative literal folding (`-42` -> IntLit, not Unary)
// - Two lambda syntaxes (`fun(params) body` and `(params) => expr`)
// - Three-way backtracking: record literal vs record extend vs grouping
// - String interpolation (reuses Parser state to parse sub-expressions)
// - Error recovery (synchronize skips to a declaration boundary)


use bumpalo::Bump;

// =========================================================================
// ParseError
// =========================================================================

/// Syntax error: carries source location and message
#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: u32,
    pub column: u32,
    pub message: String,
}

pub type ParseResult<T> = Result<T, ParseError>;

// =========================================================================
// ParseErrorHandler — parse error handler trait
// =========================================================================

/// Parse error handler trait. The Parser calls the hook on errors;
/// the default implementation `ErrorCollector` collects into a `Vec<ParseError>`.
pub trait ParseErrorHandler {
    /// Record a syntax error; returns a ParseError for propagation
    fn on_error(&mut self, line: u32, column: u32, message: &str) -> ParseError;
    /// Error recovery notification: called by the Parser after `synchronize` completes
    fn on_recover(&mut self) {}
    /// Returns the collected error list
    fn errors(&self) -> &[ParseError];
    /// Truncate the error list to the given length (used for speculative parse backtracking)
    fn truncate_errors(&mut self, len: usize);
}

/// Default error handler: collects errors into a Vec; on recovery, skips to a declaration boundary.
pub struct ErrorCollector {
    pub errors: Vec<ParseError>,
}

impl ErrorCollector {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }
}

impl Default for ErrorCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ParseErrorHandler for ErrorCollector {
    fn on_error(&mut self, line: u32, column: u32, message: &str) -> ParseError {
        let err = ParseError {
            line,
            column,
            message: message.to_string(),
        };
        self.errors.push(err.clone());
        err
    }

    fn errors(&self) -> &[ParseError] {
        &self.errors
    }

    fn truncate_errors(&mut self, len: usize) {
        self.errors.truncate(len);
    }
}

// =========================================================================
// Parser
// =========================================================================

/// Recursive-descent parser
pub struct Parser<'a, H: ParseErrorHandler> {
    tokens: &'a [Token<'a>],
    current: usize,
    /// bumpalo arena: used only for allocating dynamically built strings (negative literals,
    /// unescaped text, int_to_key) and the token array for interpolation sub-expressions.
    /// AST nodes themselves are stored in `ast`.
    arena: &'a Bump,
    /// Unified AST node storage (replaces per-node bumpalo allocation)
    ast: AstArena<'a>,
    handler: H,
    /// Virtual token splitting: `>=` split into `>` + `=`; after consuming `>`, set pending_eq
    pending_eq: bool,
    /// Virtual token splitting: `>>` split into `>` + `>`; after consuming the inner `>`, set pending_gt
    pending_gt: bool,
    /// Virtual token splitting: `>>=` split into `>` + `>=`; after consuming the inner `>`, set pending_gt_eq
    pending_gt_eq: bool,
}

// --- Parse helper macro ---

/// Generates a comma-separated list parsing method: parse one item, then repeatedly consume commas
/// until a terminator is encountered.
/// `check($tk)` uses the given TokenKind as terminator; `check_close_angle` uses a closing angle bracket.
macro_rules! impl_parse_comma_list {
    ($method:ident, $item:ty, $parse_fn:ident, check($tk:expr)) => {
        fn $method(&mut self, items: &mut Vec<$item>) -> ParseResult<()> {
            items.push(self.$parse_fn()?);
            while self.match_token(TokenKind::Comma) {
                if self.check($tk) {
                    break;
                }
                items.push(self.$parse_fn()?);
            }
            Ok(())
        }
    };
    ($method:ident, $item:ty, $parse_fn:ident, check_close_angle) => {
        fn $method(&mut self, items: &mut Vec<$item>) -> ParseResult<()> {
            items.push(self.$parse_fn()?);
            while self.match_token(TokenKind::Comma) {
                if self.check_close_angle() {
                    break;
                }
                items.push(self.$parse_fn()?);
            }
            Ok(())
        }
    };
}

impl<'a, H: ParseErrorHandler> Parser<'a, H> {
    /// Creates a parser; an error handler must be passed explicitly
    pub fn new(tokens: &'a [Token<'a>], arena: &'a Bump, handler: H) -> Self {
        Self {
            tokens,
            current: 0,
            arena,
            ast: AstArena::new(),
            handler,
            pending_eq: false,
            pending_gt: false,
            pending_gt_eq: false,
        }
    }

    /// Returns the collected error list
    pub fn errors(&self) -> &[ParseError] {
        self.handler.errors()
    }

    // =====================================================================
    // Token navigation
    // =====================================================================

    /// Peek at the current Token (handles virtual token injection)
    fn peek(&self) -> Token<'a> {
        if self.pending_eq {
            let base = self.base_token();
            return Token { kind: TokenKind::Eq, lexeme: "=", line: base.line, column: base.column + 1 };
        }
        if self.pending_gt {
            let base = self.base_token();
            return Token { kind: TokenKind::Gt, lexeme: ">", line: base.line, column: base.column + 1 };
        }
        if self.pending_gt_eq {
            let base = self.base_token();
            return Token { kind: TokenKind::GtEq, lexeme: ">=", line: base.line, column: base.column + 1 };
        }
        if self.current >= self.tokens.len() {
            return self.tokens[self.tokens.len() - 1];
        }
        self.tokens[self.current]
    }

    /// Returns the base Token used for virtual token position computation
    fn base_token(&self) -> Token<'a> {
        if self.current > 0 {
            self.tokens[self.current - 1]
        } else {
            self.tokens[0]
        }
    }

    /// Returns the most recently consumed Token
    fn previous(&self) -> Token<'a> {
        self.tokens[self.current - 1]
    }

    /// Whether the end of the Token sequence has been reached
    fn is_at_end(&self) -> bool {
        self.peek().kind == TokenKind::Eof
    }

    /// Consume the current Token and advance (handles virtual token consumption)
    fn advance(&mut self) -> Token<'a> {
        if self.pending_eq {
            self.pending_eq = false;
            let base = self.base_token();
            return Token { kind: TokenKind::Eq, lexeme: "=", line: base.line, column: base.column + 1 };
        }
        if self.pending_gt {
            self.pending_gt = false;
            let base = self.base_token();
            return Token { kind: TokenKind::Gt, lexeme: ">", line: base.line, column: base.column + 1 };
        }
        if self.pending_gt_eq {
            self.pending_gt_eq = false;
            let base = self.base_token();
            return Token { kind: TokenKind::GtEq, lexeme: ">=", line: base.line, column: base.column + 1 };
        }
        if !self.is_at_end() {
            self.current += 1;
        }
        self.tokens[self.current - 1]
    }

    /// Whether the current Token is of the given kind
    fn check(&self, kind: TokenKind) -> bool {
        if self.is_at_end() {
            return false;
        }
        self.peek().kind == kind
    }

    /// Consume and return true if the current Token matches
    fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            return true;
        }
        false
    }

    /// Expect and consume a Token of the given kind; on mismatch, records an error and returns Err
    fn expect(&mut self, kind: TokenKind, message: &str) -> ParseResult<Token<'a>> {
        if self.check(kind) {
            return Ok(self.advance());
        }
        let tok = self.peek();
        Err(self.report_error_at(tok.line, tok.column, message))
    }

    /// Whether the current Token is an identifier with the given name
    fn check_identifier(&self, name: &str) -> bool {
        self.peek().kind == TokenKind::Identifier && self.peek().lexeme == name
    }

    /// Detect whether the current position is a `>` closing a generic parameter list (including virtual-split forms)
    fn check_close_angle(&self) -> bool {
        if self.pending_gt || self.pending_gt_eq {
            return true;
        }
        if self.current >= self.tokens.len() {
            return false;
        }
        matches!(
            self.tokens[self.current].kind,
            TokenKind::Gt | TokenKind::GtEq | TokenKind::GtGt | TokenKind::GtGtEq
        )
    }

    /// Expect and consume the `>` that closes a generic parameter list, supporting virtual splits `>=` / `>>` / `>>=`
    fn expect_close_angle(&mut self, message: &str) -> ParseResult<()> {
        if self.check(TokenKind::Gt) {
            self.advance();
            return Ok(());
        }
        if self.pending_gt_eq {
            self.pending_gt_eq = false;
            self.pending_eq = true;
            return Ok(());
        }
        if self.current < self.tokens.len() {
            match self.tokens[self.current].kind {
                TokenKind::GtEq => {
                    self.current += 1;
                    self.pending_eq = true;
                    return Ok(());
                }
                TokenKind::GtGt => {
                    self.current += 1;
                    self.pending_gt = true;
                    return Ok(());
                }
                TokenKind::GtGtEq => {
                    self.current += 1;
                    self.pending_gt_eq = true;
                    return Ok(());
                }
                _ => {}
            }
        }
        let tok = self.peek();
        Err(self.report_error_at(tok.line, tok.column, message))
    }

    // =====================================================================
    // Error handling
    // =====================================================================

    /// Record a syntax error at the given location; returns a ParseError for propagation
    fn report_error_at(&mut self, line: u32, column: u32, message: &str) -> ParseError {
        self.handler.on_error(line, column, message)
    }

    /// Record a syntax error at the current Token
    fn report_error(&mut self, message: &str) -> ParseResult<()> {
        let tok = self.peek();
        Err(self.report_error_at(tok.line, tok.column, message))
    }

    /// Reject parenthesized conditions in conditional statements
    ///
    /// Only reject when `(...)` wraps the entire condition (C-style `if (cond)`).
    /// If `(...)` is just part of a larger expression (e.g. `while (v & 1) == 0`, where `)`
    /// is followed by a binary operator), do not reject — the parentheses are a legitimate
    /// sub-expression grouping, not a redundant condition wrapper.
    fn reject_paren_condition(&mut self, kw_name: &str) -> ParseResult<()> {
        if self.check(TokenKind::LParen) && self.paren_wraps_full_condition() {
            let msg = format!("parentheses are not allowed around the {} condition", kw_name);
            self.report_error(&msg)?;
            unreachable!()
        }
        Ok(())
    }

    /// Whether the current `(...)` group wraps the entire condition: scan to the matching `)`,
    /// then check whether it is followed by a binary operator.
    /// Followed by a binary operator → parentheses are a sub-expression (e.g. `(v & 1) == 0`) → return false (do not reject).
    /// Otherwise → parentheses wrap the entire condition → return true (reject).
    fn paren_wraps_full_condition(&self) -> bool {
        let mut i = self.current;
        if i >= self.tokens.len() || self.tokens[i].kind != TokenKind::LParen {
            return false;
        }
        let mut depth: usize = 0;
        while i < self.tokens.len() {
            match self.tokens[i].kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        let next = i + 1;
                        // `)` followed by a binary operator → parentheses are a sub-expression, not the entire condition
                        return next >= self.tokens.len()
                            || lookup_binary_op(self.tokens[next].kind).is_none();
                    }
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// Error recovery: skip Tokens until a declaration start or a closing brace is encountered
    fn synchronize(&mut self) {
        while !self.is_at_end() {
            match self.peek().kind {
                TokenKind::KwFun
                | TokenKind::KwType
                | TokenKind::KwTrait
                | TokenKind::KwImport
                | TokenKind::KwPack
                | TokenKind::KwPub
                | TokenKind::KwVal
                | TokenKind::KwVar => break,
                TokenKind::RBrace => {
                    self.advance();
                    break;
                }
                _ => {
                    self.advance();
                }
            }
        }
        self.handler.on_recover();
    }

    // =====================================================================
    // AST node allocation
    // =====================================================================

    fn alloc_expr(&mut self, span: Span, expr: Expr<'a>) -> ExprRef {
        self.ast.alloc_expr(span, expr)
    }

    fn alloc_stmt(&mut self, span: Span, stmt: Stmt<'a>) -> StmtRef {
        self.ast.alloc_stmt(span, stmt)
    }

    fn alloc_type(&mut self, span: Span, ty: TypeNode<'a>) -> TypeRef {
        self.ast.alloc_type(span, ty)
    }

    fn alloc_pattern(&mut self, span: Span, pat: Pattern<'a>) -> PatternRef {
        self.ast.alloc_pattern(span, pat)
    }

    fn spanned_decl(&self, span: Span, decl: Decl<'a>) -> Spanned<Decl<'a>> {
        Spanned { span, node: decl }
    }

    // =====================================================================
    // Module parsing
    // =====================================================================

    /// Parse an entire module
    pub fn parse_module(&mut self, module_name: &'a str) -> ParseResult<Module<'a>> {
        let mut declarations = Vec::new();
        while !self.is_at_end() {
            let at_decl_kw = matches!(
                self.peek().kind,
                TokenKind::KwFun
                    | TokenKind::KwType
                    | TokenKind::KwTrait
                    | TokenKind::KwImport
                    | TokenKind::KwPack
                    | TokenKind::KwPub
                    | TokenKind::At
            );
            if let Some(decl) = self.try_parse_decl() {
                declarations.push(decl);
                continue;
            }
            if at_decl_kw {
                self.synchronize();
                continue;
            }
            let before_expr = self.current;
            let expr = match self.parse_expr() {
                Ok(e) => e,
                Err(_) => {
                    self.synchronize();
                    continue;
                }
            };
            if self.current == before_expr {
                self.advance();
                continue;
            }
            let span = self.ast.expr(expr).span;
            if self.match_token(TokenKind::Eq) {
                let value = match self.parse_expr() {
                    Ok(v) => v,
                    Err(_) => {
                        self.synchronize();
                        continue;
                    }
                };
                let stmt = self.alloc_stmt(
                    span,
                    Stmt::Assignment {
                        target: expr,
                        value,
                    },
                );
                let void_expr = self.alloc_expr(span, Expr::VoidLit);
                declarations.push(self.spanned_decl(
                    span,
                    Decl::ExprDecl {
                        expr: void_expr,
                        stmt: Some(stmt),
                    },
                ));
            } else {
                declarations.push(self.spanned_decl(
                    span,
                    Decl::ExprDecl {
                        expr,
                        stmt: None,
                    },
                ));
            }
        }
        if !self.handler.errors().is_empty() {
            return Err(self.handler.errors()[0].clone());
        }
        Ok(Module {
            name: module_name,
            source_path: None,
            arena: std::mem::take(&mut self.ast),
            declarations,
        })
    }

    /// Parse 0..N attribute prefixes: @name or @name("arg") or @name "arg"
    fn parse_attributes(&mut self) -> Vec<Attribute<'a>> {
        let mut attrs = Vec::new();
        while self.check(TokenKind::At) {
            self.advance(); // @
            let name_tok = match self.expect(TokenKind::Identifier, "expected attribute name") {
                Ok(t) => t,
                Err(_) => break,
            };
            let mut args = Vec::new();
            if self.check(TokenKind::LParen) {
                self.advance(); // (
                while !self.check(TokenKind::RParen) && !self.is_at_end() {
                    if self.check(TokenKind::StringLiteral) {
                        let lex = self.advance().lexeme;
                        // Strip the surrounding quotes
                        if lex.len() >= 2 {
                            args.push(&lex[1..lex.len() - 1]);
                        } else {
                            args.push(lex);
                        }
                    } else if self.check(TokenKind::Identifier) {
                        args.push(self.advance().lexeme);
                    } else {
                        self.advance();
                    }
                    if self.check(TokenKind::Comma) {
                        self.advance();
                    }
                }
                let _ = self.expect(TokenKind::RParen, "expected ')' after attribute args");
            } else if self.check(TokenKind::StringLiteral) {
                let lex = self.advance().lexeme;
                if lex.len() >= 2 {
                    args.push(&lex[1..lex.len() - 1]);
                } else {
                    args.push(lex);
                }
            }
            attrs.push(Attribute { name: name_tok.lexeme, args });
        }
        attrs
    }

    /// Attempt to parse a top-level declaration (fault-tolerant; returns None on failure)
    fn try_parse_decl(&mut self) -> Option<Spanned<Decl<'a>>> {
        let saved = self.current;
        let attributes = self.parse_attributes();
        if !attributes.is_empty() && !self.check(TokenKind::KwPub) && !self.check(TokenKind::KwAsync) && !self.check(TokenKind::KwFun) {
            // Attributes must be followed by pub/async/fun
            self.current = saved;
            return None;
        }
        let mut visibility = Visibility::Private;
        if self.match_token(TokenKind::KwPub) {
            visibility = Visibility::Public;
        }
        let mut is_async = false;
        if self.match_token(TokenKind::KwAsync) {
            if !self.check(TokenKind::KwFun) {
                return None;
            }
            is_async = true;
        }
        if self.check(TokenKind::KwFun) {
            return self.parse_fun_decl(visibility, is_async, attributes).ok();
        }
        if self.check(TokenKind::KwType) {
            return self.parse_type_decl(visibility).ok();
        }
        if self.check(TokenKind::KwTrait) {
            return self.parse_trait_decl(visibility).ok();
        }
        if self.check(TokenKind::KwImport) {
            return self.parse_use_decl(visibility).ok();
        }
        if self.check(TokenKind::KwPack) {
            return self.parse_pack_decl(visibility).ok();
        }
        // pub val / pub var
        if visibility == Visibility::Public && (self.check(TokenKind::KwVal) || self.check(TokenKind::KwVar)) {
            if let Ok(stmt) = self.parse_stmt() {
                let stmt_spanned = self.ast.stmt(stmt);
                let mut s = stmt_spanned.node.clone();
                match &mut s {
                    Stmt::ValDecl { visibility: v, .. } => *v = visibility,
                    Stmt::VarDecl { visibility: v, .. } => *v = visibility,
                    _ => {}
                }
                let span = stmt_spanned.span;
                let stmt_ref = self.alloc_stmt(span, s);
                let dummy = self.alloc_expr(span, Expr::VoidLit);
                return Some(self.spanned_decl(
                    span,
                    Decl::ExprDecl {
                        expr: dummy,
                        stmt: Some(stmt_ref),
                    },
                ));
            }
            return None;
        }
        // Backtrack pub
        if visibility == Visibility::Public {
            self.current = saved;
        }
        // Top-level statement
        if matches!(
            self.peek().kind,
            TokenKind::KwVal
                | TokenKind::KwVar
                | TokenKind::KwFor
                | TokenKind::KwWhile
                | TokenKind::KwLoop
                | TokenKind::KwDefer
                | TokenKind::KwThrow
                | TokenKind::KwReturn
        ) {
            if let Ok(stmt) = self.parse_stmt() {
                let span = self.ast.stmt(stmt).span;
                let dummy = self.alloc_expr(span, Expr::VoidLit);
                return Some(self.spanned_decl(
                    span,
                    Decl::ExprDecl {
                        expr: dummy,
                        stmt: Some(stmt),
                    },
                ));
            }
        }
        None
    }

    // =====================================================================
    // Declaration parsing
    // =====================================================================

    /// Parse a function declaration: `fun name<TParams>(params): ReturnType with bounds { body }`
    fn parse_fun_decl(&mut self, visibility: Visibility, is_async: bool, attributes: Vec<Attribute<'a>>) -> ParseResult<Spanned<Decl<'a>>> {
        let fun_tok = self.advance(); // 'fun'
        let name_tok = self.expect(TokenKind::Identifier, "expected function name")?;
        let mut type_params = Vec::new();
        if self.match_token(TokenKind::Lt) {
            self.parse_type_param_list(&mut type_params)?;
            let _ = self.expect_close_angle("expected '>' to close type parameter list");
        }
        let mut params = Vec::new();
        let _ = self.expect(TokenKind::LParen, "expected '(' to start parameter list");
        if !self.check(TokenKind::RParen) {
            self.parse_param_list(&mut params)?;
        }
        let _ = self.expect(TokenKind::RParen, "expected ')' to close parameter list");
        let return_type = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            return Err(self.report_error_at(
                name_tok.line,
                name_tok.column,
                "function declaration must explicitly annotate the return type (use ': void' for no return value)",
            ));
        };
        let mut bounds = Vec::new();
        if self.match_token(TokenKind::KwWith) {
            self.parse_trait_bound_list(&mut bounds)?;
        }
        // @extern("C") function: body is a #{ }# raw block rather than a Kuzo expression
        let extern_c_body = if self.check(TokenKind::RawBlock) {
            let tok = self.advance();
            Some(tok.lexeme)
        } else {
            None
        };
        // When extern_c_body is present, use a placeholder expression as the body (Sema skips checking)
        let body = if extern_c_body.is_some() {
            self.alloc_expr(token_span(&fun_tok), Expr::VoidLit)
        } else {
            self.parse_expr()?
        };
        Ok(self.spanned_decl(
            token_span(&fun_tok),
            Decl::FunDecl {
                visibility,
                name: name_tok.lexeme,
                type_params,
                params,
                return_type,
                bounds,
                body,
                is_async,
                is_entry: name_tok.lexeme == "main",
                attributes,
                extern_c_body,
            },
        ))
    }

    /// Parse a type declaration: `type Name<TParams> : traits = def with constraints { methods }`
    fn parse_type_decl(&mut self, visibility: Visibility) -> ParseResult<Spanned<Decl<'a>>> {
        let type_tok = self.advance(); // 'type'
        let name_tok = self.expect(TokenKind::Identifier, "expected type name")?;
        let mut type_params = Vec::new();
        if self.match_token(TokenKind::Lt) {
            self.parse_type_param_list(&mut type_params)?;
            let _ = self.expect_close_angle("expected '>' to close type parameter list");
        }
        let mut implemented_traits = Vec::new();
        if self.match_token(TokenKind::Colon) {
            let has_paren = self.check(TokenKind::LParen);
            if has_paren {
                self.advance();
            }
            self.parse_trait_bound_list(&mut implemented_traits)?;
            if has_paren {
                let _ = self.expect(TokenKind::RParen, "expected ')' after trait list");
            }
        }
        let _ = self.expect(TokenKind::Eq, "expected '=' to define type body");
        let def = self.parse_type_def()?;
        let mut type_constraints = Vec::new();
        if self.match_token(TokenKind::KwWith) {
            self.parse_type_constraints(&mut type_constraints)?;
        }
        let mut methods = Vec::new();
        if self.match_token(TokenKind::LBrace) {
            self.parse_method_block(&mut methods)?;
            let _ = self.expect(TokenKind::RBrace, "expected '}' to close method block");
        }
        Ok(self.spanned_decl(
            token_span(&type_tok),
            Decl::TypeDecl {
                visibility,
                name: name_tok.lexeme,
                type_params,
                implemented_traits,
                type_constraints,
                def,
                methods,
            },
        ))
    }

    /// Parse a type definition body
    fn parse_type_def(&mut self) -> ParseResult<TypeDef<'a>> {
        if self.match_token(TokenKind::Pipe) {
            return self.parse_adt_body();
        }
        if self.check(TokenKind::LParen) {
            let saved = self.current;
            if let Some(def) = self.try_parse_record_type_def() {
                return Ok(def);
            }
            self.current = saved;
        }
        if self.check(TokenKind::Identifier) {
            let saved = self.current;
            let _name_tok = self.advance();
            if self.check(TokenKind::LParen) {
                self.advance();
                if !self.check(TokenKind::RParen) {
                    let saved2 = self.current;
                    if self.check(TokenKind::Identifier) {
                        self.advance();
                        if self.check(TokenKind::Colon) {
                            // name: Type -> record-style parameter
                            self.current = saved2;
                            let mut _params = Vec::new();
                            self.parse_param_list(&mut _params)?;
                            if self.expect(TokenKind::RParen, "expected ')'").is_err() {
                                self.current = saved;
                                let target = self.parse_type()?;
                                return Ok(TypeDef::Alias { target });
                            }
                            self.current = saved;
                            if let Some(def) = self.try_parse_single_ctor_adt() {
                                return Ok(def);
                            }
                            self.current = saved;
                        }
                        self.current = saved2;
                    }
                    self.current = saved;
                    if let Some(def) = self.try_parse_single_ctor_adt() {
                        return Ok(def);
                    }
                    self.current = saved;
                } else {
                    // Bug #69: Empty parens `Name()` — try single-constructor ADT.
                    // Without this, `type Unit = Unit()` falls through to the alias path
                    // (parsing `Unit()` as a type expression), and the constructor is never
                    // registered.
                    self.current = saved;
                    if let Some(def) = self.try_parse_single_ctor_adt() {
                        return Ok(def);
                    }
                    self.current = saved;
                }
            }
            self.current = saved;
        }
        let target = self.parse_type()?;
        if self.check(TokenKind::Pipe) {
            self.report_error(
                "each variant of a sum type must be prefixed with '|', including the first; for example `type Color = | Red | Green`",
            )?;
            unreachable!()
        }
        Ok(TypeDef::Alias { target })
    }

    /// Attempt to parse a single-constructor ADT
    fn try_parse_single_ctor_adt(&mut self) -> Option<TypeDef<'a>> {
        let name_tok = self.advance();
        if !self.check(TokenKind::LParen) {
            return None;
        }
        self.advance();
        if self.check(TokenKind::RParen) {
            self.advance();
            return Some(TypeDef::Adt {
                constructors: vec![ConstructorDef {
                    name: name_tok.lexeme,
                    fields: Vec::new(),
                    return_type: None,
                }],
            });
        }
        // Named fields
        if self.check(TokenKind::Identifier)
            && self.current + 1 < self.tokens.len()
            && self.tokens[self.current + 1].kind == TokenKind::Colon
        {
            let mut fields = Vec::new();
            if self.parse_constructor_field_list(&mut fields).is_err() {
                return None;
            }
            if self.expect(TokenKind::RParen, "expected ')'").is_err() {
                return None;
            }
            return Some(TypeDef::Adt {
                constructors: vec![ConstructorDef {
                    name: name_tok.lexeme,
                    fields,
                    return_type: None,
                }],
            });
        }
        // Positional fields
        let first_type = match self.parse_type() {
            Ok(t) => t,
            Err(_) => return None,
        };
        if self.check(TokenKind::Comma) {
            let mut fields = vec![ConstructorField {
                name: None,
                ty: first_type,
            }];
            while self.match_token(TokenKind::Comma) {
                if self.check(TokenKind::RParen) {
                    break;
                }
                let ty = match self.parse_type() {
                    Ok(t) => t,
                    Err(_) => return None,
                };
                fields.push(ConstructorField { name: None, ty });
            }
            if self.expect(TokenKind::RParen, "expected ')'").is_err() {
                return None;
            }
            return Some(TypeDef::Adt {
                constructors: vec![ConstructorDef {
                    name: name_tok.lexeme,
                    fields,
                    return_type: None,
                }],
            });
        }
        if self.expect(TokenKind::RParen, "expected ')'").is_err() {
            return None;
        }
        Some(TypeDef::Newtype {
            name: name_tok.lexeme,
            inner: first_type,
        })
    }

    /// Attempt to parse a record type definition
    fn try_parse_record_type_def(&mut self) -> Option<TypeDef<'a>> {
        self.advance(); // '('
        if self.peek().kind == TokenKind::Identifier {
            let name = self.advance();
            if self.check(TokenKind::Colon) {
                self.advance();
                let ty = self.parse_type().ok()?;
                let mut fields = vec![RecordFieldType {
                    name: name.lexeme,
                    ty,
                }];
                while self.match_token(TokenKind::Comma) {
                    if self.check(TokenKind::RParen) {
                        break;
                    }
                    let field_name = self.expect(TokenKind::Identifier, "expected field name").ok()?;
                    let _ = self.expect(TokenKind::Colon, "expected ':'");
                    let field_ty = self.parse_type().ok()?;
                    fields.push(RecordFieldType {
                        name: field_name.lexeme,
                        ty: field_ty,
                    });
                }
                let _ = self.expect(TokenKind::RParen, "expected ')'");
                return Some(TypeDef::Record { fields });
            }
            return None;
        }
        None
    }

    /// Parse an ADT constructor list
    fn parse_adt_body(&mut self) -> ParseResult<TypeDef<'a>> {
        let mut constructors = vec![self.parse_constructor_def()?];
        while self.match_token(TokenKind::Pipe) {
            constructors.push(self.parse_constructor_def()?);
        }
        Ok(TypeDef::Adt { constructors })
    }

    /// Parse a single constructor definition
    fn parse_constructor_def(&mut self) -> ParseResult<ConstructorDef<'a>> {
        let name_tok = self.expect(TokenKind::Identifier, "expected constructor name")?;
        let mut fields = Vec::new();
        if self.match_token(TokenKind::LParen) {
            if !self.check(TokenKind::RParen) {
                self.parse_constructor_field_list(&mut fields)?;
            }
            let _ = self.expect(TokenKind::RParen, "expected ')' to close constructor fields");
        }
        let return_type = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        Ok(ConstructorDef {
            name: name_tok.lexeme,
            fields,
            return_type,
        })
    }

    // Parse a constructor field list
    impl_parse_comma_list!(parse_constructor_field_list, ConstructorField<'a>, parse_constructor_field, check(TokenKind::RParen));

    /// Parse a single constructor field
    fn parse_constructor_field(&mut self) -> ParseResult<ConstructorField<'a>> {
        if self.peek().kind == TokenKind::Identifier {
            let saved = self.current;
            let name = self.advance();
            if self.match_token(TokenKind::Colon) {
                let ty = self.parse_type()?;
                return Ok(ConstructorField {
                    name: Some(name.lexeme),
                    ty,
                });
            }
            self.current = saved;
        }
        let ty = self.parse_type()?;
        Ok(ConstructorField { name: None, ty })
    }

    /// Parse a trait declaration
    fn parse_trait_decl(&mut self, visibility: Visibility) -> ParseResult<Spanned<Decl<'a>>> {
        let trait_tok = self.advance(); // 'trait'
        let name_tok = self.expect(TokenKind::Identifier, "expected trait name")?;
        let mut type_params = Vec::new();
        if self.match_token(TokenKind::Lt) {
            self.parse_type_param_list(&mut type_params)?;
            let _ = self.expect_close_angle("expected '>' to close type parameter list");
        }
        let mut parents = Vec::new();
        if self.match_token(TokenKind::LParen) {
            if !self.check(TokenKind::RParen) {
                self.parse_trait_bound_list_inner(&mut parents)?;
            }
            let _ = self.expect(TokenKind::RParen, "expected ')' to close parent trait list");
        }
        let mut associated_types = Vec::new();
        let mut methods = Vec::new();
        let _ = self.expect(TokenKind::LBrace, "expected '{' to start trait body");
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.check(TokenKind::KwType) {
                associated_types.push(self.parse_associated_type()?);
            } else {
                methods.push(self.parse_method_decl()?);
            }
        }
        let _ = self.expect(TokenKind::RBrace, "expected '}' to close trait body");
        Ok(self.spanned_decl(
            token_span(&trait_tok),
            Decl::TraitDecl {
                visibility,
                name: name_tok.lexeme,
                type_params,
                parents,
                associated_types,
                methods,
            },
        ))
    }

    /// Parse an associated type declaration
    fn parse_associated_type(&mut self) -> ParseResult<AssociatedType<'a>> {
        let _type_tok = self.advance(); // 'type'
        let name_tok = self.expect(TokenKind::Identifier, "expected associated type name")?;
        let kind = if self.match_token(TokenKind::Colon) {
            Some(Box::new(self.parse_kind()?))
        } else {
            None
        };
        Ok(AssociatedType {
            name: name_tok.lexeme,
            kind,
        })
    }

    /// Parse a method declaration.
    ///
    /// Syntax: `pub? override? async? fun &? name<tparams>(params): ret { body }`
    /// The optional `&` between `fun` and the method name marks the receiver as
    /// by-reference (`&this`); without `&` the receiver is by-value (`this`).
    /// The receiver is implicit: it is NOT written in the parameter list. The
    /// parser injects an internal `Param { name: "this", type_annotation: ThisType/RefType<ThisType> }`
    /// at params[0] so downstream sema/IR logic (is_this_param, arity, inputs[0]=recv) is unchanged.
    fn parse_method_decl(&mut self) -> ParseResult<MethodDecl<'a>> {
        let mut visibility = Visibility::Private;
        if self.match_token(TokenKind::KwPub) {
            visibility = Visibility::Public;
        }
        let is_override = self.match_token(TokenKind::KwOverride);
        let is_async = self.match_token(TokenKind::KwAsync);
        let _ = self.expect(TokenKind::KwFun, "expected 'fun'")?;
        // Detect by-reference receiver marker: `fun &name(...)` means `&this`.
        let is_ref_receiver = self.match_token(TokenKind::Ampersand);
        let name_tok = self.expect(TokenKind::Identifier, "expected method name")?;
        let mut type_params = Vec::new();
        if self.match_token(TokenKind::Lt) {
            self.parse_type_param_list(&mut type_params)?;
            let _ = self.expect_close_angle("expected '>'");
        }
        let mut params = Vec::new();
        let _ = self.expect(TokenKind::LParen, "expected '('");
        if !self.check(TokenKind::RParen) {
            self.parse_param_list(&mut params)?;
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        // Inject the implicit this parameter at params[0].
        // By-value: ThisType; by-reference (&): RefType<ThisType>.
        let this_type = if is_ref_receiver {
            let this_ty = self.alloc_type(token_span(&name_tok), TypeNode::ThisType);
            self.alloc_type(token_span(&name_tok), TypeNode::RefType { inner: this_ty })
        } else {
            self.alloc_type(token_span(&name_tok), TypeNode::ThisType)
        };
        params.insert(0, Param {
            name: "this",
            type_annotation: Some(this_type),
        });
        let return_type = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            self.report_error(
                "method declaration must explicitly annotate the return type (use ': void' for no return value)",
            )?;
            unreachable!()
        };
        let delegate = if self.match_token(TokenKind::Eq) {
            let trait_tok = self.expect(TokenKind::Identifier, "expected delegate trait name")?;
            let _ = self.expect(TokenKind::Dot, "expected '.'");
            let method_tok = self.expect(TokenKind::Identifier, "expected delegate method name")?;
            Some(DelegateInfo {
                trait_name: trait_tok.lexeme,
                method_name: method_tok.lexeme,
            })
        } else {
            None
        };
        let body = if delegate.is_none() && self.check(TokenKind::LBrace) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(MethodDecl {
            name: name_tok.lexeme,
            type_params,
            params,
            return_type,
            body,
            is_override,
            delegate,
            visibility,
            is_async,
        })
    }

    /// Parse an import declaration
    fn parse_use_decl(&mut self, visibility: Visibility) -> ParseResult<Spanned<Decl<'a>>> {
        let use_tok = self.advance(); // 'import'
        let first = self.expect(TokenKind::Identifier, "expected module name")?;
        let mut module_path = vec![first.lexeme];
        let mut dot_before_brace = false;
        while self.match_token(TokenKind::Dot) {
            if self.check(TokenKind::LBrace) {
                dot_before_brace = true;
                break;
            }
            let part = self.expect(TokenKind::Identifier, "expected module path segment")?;
            module_path.push(part.lexeme);
        }
        let items = if self.check(TokenKind::LBrace) {
            if !dot_before_brace {
                self.report_error(
                    "selective import must use '.{ }' syntax; expected '.' before '{'",
                )?;
                unreachable!()
            }
            self.advance(); // '{'
            let mut item_list = Vec::new();
            if !self.check(TokenKind::RBrace) {
                item_list.push(self.parse_import_item()?);
                while self.match_token(TokenKind::Comma) {
                    if self.check(TokenKind::RBrace) {
                        break;
                    }
                    item_list.push(self.parse_import_item()?);
                }
            }
            let _ = self.expect(TokenKind::RBrace, "expected '}'");
            Some(item_list)
        } else {
            None
        };
        Ok(self.spanned_decl(
            token_span(&use_tok),
            Decl::ImportDecl {
                module_path,
                items,
                visibility,
            },
        ))
    }

    /// Parse a single import item
    fn parse_import_item(&mut self) -> ParseResult<ImportItem<'a>> {
        let name = self.expect(TokenKind::Identifier, "expected import item name")?;
        let alias = if self.match_token(TokenKind::KwAs) {
            let alias_tok = self.expect(TokenKind::Identifier, "expected alias")?;
            Some(alias_tok.lexeme)
        } else {
            None
        };
        Ok(ImportItem {
            name: name.lexeme,
            alias,
        })
    }

    /// Parse a pack declaration
    fn parse_pack_decl(&mut self, visibility: Visibility) -> ParseResult<Spanned<Decl<'a>>> {
        let pack_tok = self.advance(); // 'pack'
        let name_tok = self.expect(TokenKind::Identifier, "expected pack name")?;
        Ok(self.spanned_decl(
            token_span(&pack_tok),
            Decl::PackDecl {
                visibility,
                name: name_tok.lexeme,
            },
        ))
    }

    // =====================================================================
    // Type parameters, kinds, parameters, constraints
    // =====================================================================

    impl_parse_comma_list!(parse_type_param_list, TypeParam<'a>, parse_type_param, check_close_angle);

    fn parse_type_param(&mut self) -> ParseResult<TypeParam<'a>> {
        let name_tok = self.expect(TokenKind::Identifier, "expected type parameter name")?;
        let mut kind = None;
        let mut bounds = Vec::new();
        if self.match_token(TokenKind::Colon) {
            if self.check(TokenKind::Identifier) {
                let has_paren = self.check(TokenKind::LParen);
                if has_paren {
                    self.advance();
                }
                let trait_name_tok = self.expect(TokenKind::Identifier, "expected trait name")?;
                bounds.push(TraitBound {
                    trait_name: trait_name_tok.lexeme,
                    type_args: Vec::new(),
                });
                if has_paren {
                    while self.match_token(TokenKind::Comma) {
                        if self.check(TokenKind::RParen) {
                            break;
                        }
                        let next_trait = self.expect(TokenKind::Identifier, "expected trait name")?;
                        bounds.push(TraitBound {
                            trait_name: next_trait.lexeme,
                            type_args: Vec::new(),
                        });
                    }
                    let _ = self.expect(TokenKind::RParen, "expected ')' after trait list");
                }
            } else {
                kind = Some(Box::new(self.parse_kind()?));
            }
        }
        if self.match_token(TokenKind::KwWith) {
            self.parse_trait_bound_list_inner(&mut bounds)?;
        }
        Ok(TypeParam {
            name: name_tok.lexeme,
            kind,
            bounds,
        })
    }

    fn parse_kind(&mut self) -> ParseResult<Kind> {
        self.parse_kind_arrow()
    }

    fn parse_kind_arrow(&mut self) -> ParseResult<Kind> {
        let left = self.parse_kind_primary()?;
        if self.match_token(TokenKind::MinusGt) {
            let right = self.parse_kind_arrow()?;
            return Ok(Kind::Arrow {
                param: Box::new(left),
                result: Box::new(right),
            });
        }
        Ok(left)
    }

    fn parse_kind_primary(&mut self) -> ParseResult<Kind> {
        if self.check(TokenKind::Star) {
            self.advance();
            return Ok(Kind::Star);
        }
        if self.match_token(TokenKind::LParen) {
            let kind = self.parse_kind_arrow()?;
            let _ = self.expect(TokenKind::RParen, "expected ')'");
            return Ok(kind);
        }
        self.report_error("expected kind (* or arrow kind)")?;
        unreachable!()
    }

    impl_parse_comma_list!(parse_param_list, Param<'a>, parse_param, check(TokenKind::RParen));

    fn parse_param(&mut self) -> ParseResult<Param<'a>> {
        let name_tok = self.expect(TokenKind::Identifier, "expected parameter name")?;
        let type_annotation = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        Ok(Param {
            name: name_tok.lexeme,
            type_annotation,
        })
    }

    fn parse_trait_bound_list(&mut self, bounds: &mut Vec<TraitBound<'a>>) -> ParseResult<()> {
        self.parse_trait_bound_list_inner(bounds)
    }

    impl_parse_comma_list!(parse_trait_bound_list_inner, TraitBound<'a>, parse_trait_bound, check(TokenKind::RParen));

    fn parse_trait_bound(&mut self) -> ParseResult<TraitBound<'a>> {
        let name_tok = self.expect(TokenKind::Identifier, "expected trait name")?;
        let mut type_args = Vec::new();
        if self.match_token(TokenKind::Lt) {
            self.parse_type_arg_list(&mut type_args)?;
            let _ = self.expect_close_angle("expected '>'");
        }
        Ok(TraitBound {
            trait_name: name_tok.lexeme,
            type_args,
        })
    }

    impl_parse_comma_list!(parse_type_constraints, TypeConstraint<'a>, parse_type_constraint, check(TokenKind::LBrace));

    fn parse_type_constraint(&mut self) -> ParseResult<TypeConstraint<'a>> {
        let type_param_tok = self.expect(TokenKind::Identifier, "expected type parameter name")?;
        self.expect(TokenKind::Colon, "expected ':' after type parameter")?;
        let concrete_type = self.parse_type()?;
        Ok(TypeConstraint {
            type_param: type_param_tok.lexeme,
            concrete_type,
        })
    }

    fn parse_method_block(&mut self, methods: &mut Vec<MethodDecl<'a>>) -> ParseResult<()> {
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            methods.push(self.parse_method_decl()?);
        }
        Ok(())
    }

    impl_parse_comma_list!(parse_type_arg_list, TypeRef, parse_type, check_close_angle);

    // =====================================================================
    // Type parsing
    // =====================================================================

    /// Type parsing entry point: handles prefix &T / *T
    fn parse_type(&mut self) -> ParseResult<TypeRef> {
        if self.match_token(TokenKind::Ampersand) {
            let span = token_span(&self.previous());
            let inner = self.parse_type()?;
            return Ok(self.alloc_type(span, TypeNode::RefType { inner }));
        }
        if self.match_token(TokenKind::AmpAmp) {
            let span = token_span(&self.previous());
            let inner = self.parse_type()?;
            let inner_ref = self.alloc_type(span, TypeNode::RefType { inner });
            return Ok(self.alloc_type(span, TypeNode::RefType { inner: inner_ref }));
        }
        if self.match_token(TokenKind::Star) {
            let span = token_span(&self.previous());
            let inner = self.parse_type()?;
            return Ok(self.alloc_type(span, TypeNode::RawPtr { inner }));
        }
        self.parse_function_type()
    }

    /// Parse a function type: `(P1, P2) -> R` or `A -> C`
    fn parse_function_type(&mut self) -> ParseResult<TypeRef> {
        if self.check(TokenKind::LParen) && self.paren_group_followed_by_arrow() {
            let span = token_span(&self.peek());
            self.advance(); // '('
            let mut params = Vec::new();
            if !self.check(TokenKind::RParen) {
                params.push(self.parse_type()?);
                while self.match_token(TokenKind::Comma) {
                    if self.check(TokenKind::RParen) {
                        break;
                    }
                    params.push(self.parse_type()?);
                }
            }
            let _ = self.expect(TokenKind::RParen, "expected ')'");
            let _ = self.expect(TokenKind::MinusGt, "expected '->'");
            let return_type = self.parse_type()?;
            return Ok(self.alloc_type(
                span,
                TypeNode::Function {
                    params,
                    return_type,
                },
            ));
        }
        let left = self.parse_nullable_type()?;
        if self.match_token(TokenKind::MinusGt) {
            let params = vec![left];
            let return_type = self.parse_type()?;
            let left_span = self.ast.ty(left).span;
            return Ok(self.alloc_type(
                left_span,
                TypeNode::Function {
                    params,
                    return_type,
                },
            ));
        }
        Ok(left)
    }

    /// Lookahead: whether a parenthesized group is immediately followed by an arrow
    fn paren_group_followed_by_arrow(&self) -> bool {
        let mut i = self.current;
        if i >= self.tokens.len() || self.tokens[i].kind != TokenKind::LParen {
            return false;
        }
        let mut depth: usize = 0;
        while i < self.tokens.len() {
            match self.tokens[i].kind {
                TokenKind::LParen => depth += 1,
                TokenKind::RParen => {
                    depth -= 1;
                    if depth == 0 {
                        let next = i + 1;
                        return next < self.tokens.len() && self.tokens[next].kind == TokenKind::MinusGt;
                    }
                }
                TokenKind::Eof => return false,
                _ => {}
            }
            i += 1;
        }
        false
    }

    /// Parse a nullable type: `T?` (chained)
    fn parse_nullable_type(&mut self) -> ParseResult<TypeRef> {
        let mut ty = self.parse_primary_type()?;
        while self.match_token(TokenKind::Question) {
            let span = self.ast.ty(ty).span;
            ty = self.alloc_type(span, TypeNode::Nullable { inner: ty });
        }
        Ok(ty)
    }

    /// Parse a primary type: named/generic, with suffix array `[N]`
    fn parse_primary_type(&mut self) -> ParseResult<TypeRef> {
        if self.check(TokenKind::LParen) {
            // Bug #68: support `(type)` parenthesized type expressions (e.g. `((i32) -> i32)[]`).
            // When the parentheses do not contain a record field pattern (`identifier :`), parse as a parenthesized type.
            if self.is_paren_record_type() {
                return self.parse_record_type();
            }
            // Parenthesized type expression: (T)
            let span = token_span(&self.peek());
            self.advance(); // '('
            let inner = self.parse_type()?;
            let _ = self.expect(TokenKind::RParen, "expected ')'");
            let mut ty = inner;
            // Suffix array type T[N]
            while self.match_token(TokenKind::LBracket) {
                let mut size: Option<u64> = None;
                if !self.check(TokenKind::RBracket) {
                    let size_tok = self.expect(TokenKind::IntLiteral, "expected array size")?;
                    size = Some(parse_u64(size_tok.lexeme).ok_or_else(|| {
                        self.report_error_at(size_tok.line, size_tok.column, "array size must be a positive integer")
                    })?);
                }
                let _ = self.expect(TokenKind::RBracket, "expected ']'");
                ty = self.alloc_type(span, TypeNode::Array {
                    element_type: ty,
                    size,
                });
            }
            return Ok(ty);
        }
        // `This` keyword in type position: resolves to the current type (TypeNode::ThisType).
        // Sema resolves it via current_this_type() during inference.
        // Accept both `this` (KwThis, lowercase instance keyword) and `This` (Identifier with
        // capitalized text, type keyword) — the language convention is:
        //   - `this` (lowercase): instance keyword (expression position)
        //   - `This` (capitalized): type keyword (type position, e.g. `fun clone(): This`)
        // The lexer is case-sensitive and only registers `this` as KwThis; `This` is lexed as an
        // Identifier, so we match it here by lexeme text.
        let is_this_type_kw = self.check(TokenKind::KwThis)
            || (self.check(TokenKind::Identifier) && self.peek().lexeme == "This");
        if is_this_type_kw {
            let tok = self.advance();
            let span = token_span(&tok);
            // Suffix array type This[N]
            let mut ty = self.alloc_type(span, TypeNode::ThisType);
            while self.match_token(TokenKind::LBracket) {
                let mut size: Option<u64> = None;
                if !self.check(TokenKind::RBracket) {
                    let size_tok = self.expect(TokenKind::IntLiteral, "expected array size")?;
                    size = Some(parse_u64(size_tok.lexeme).ok_or_else(|| {
                        self.report_error_at(size_tok.line, size_tok.column, "array size must be a positive integer")
                    })?);
                }
                let _ = self.expect(TokenKind::RBracket, "expected ']'");
                ty = self.alloc_type(span, TypeNode::Array {
                    element_type: ty,
                    size,
                });
            }
            return Ok(ty);
        }
        let name_tok = self.expect(TokenKind::Identifier, "expected type name")?;
        let span = token_span(&name_tok);
        let mut ty = if self.match_token(TokenKind::Lt) {
            let mut args = vec![self.parse_type()?];
            while self.match_token(TokenKind::Comma) {
                if self.check_close_angle() {
                    break;
                }
                args.push(self.parse_type()?);
            }
            let _ = self.expect_close_angle("expected '>' to close type parameters");
            self.alloc_type(
                span,
                TypeNode::Generic {
                    name: name_tok.lexeme,
                    args,
                },
            )
        } else {
            self.alloc_type(span, TypeNode::Named { name: name_tok.lexeme })
        };
        // Suffix array type T[N]
        while self.match_token(TokenKind::LBracket) {
            let mut size: Option<u64> = None;
            if !self.check(TokenKind::RBracket) {
                let size_tok = self.expect(TokenKind::IntLiteral, "expected array size")?;
                size = Some(parse_u64(size_tok.lexeme).ok_or_else(|| {
                    self.report_error_at(size_tok.line, size_tok.column, "array size must be a positive integer")
                })?);
            }
            let _ = self.expect(TokenKind::RBracket, "expected ']'");
            ty = self.alloc_type(span, TypeNode::Array {
                element_type: ty,
                size,
            });
        }
        Ok(ty)
    }

    /// Lookahead: determine whether the parentheses contain a record field pattern (`identifier :`).
    /// Used to distinguish between `(field: Type, ...)` record types and `(T)` parenthesized type expressions.
    fn is_paren_record_type(&self) -> bool {
        let mut i = self.current;
        if i >= self.tokens.len() || self.tokens[i].kind != TokenKind::LParen {
            return false;
        }
        i += 1; // skip '('
        // Empty parentheses `()` are not a record type (they are an empty record or empty type)
        if i >= self.tokens.len() || self.tokens[i].kind == TokenKind::RParen {
            return false;
        }
        // Check for the `identifier :` pattern
        if self.tokens[i].kind == TokenKind::Identifier {
            let next = i + 1;
            if next < self.tokens.len() && self.tokens[next].kind == TokenKind::Colon {
                return true;
            }
        }
        false
    }

    /// Parse a record type: `(field: Type, ...)`
    fn parse_record_type(&mut self) -> ParseResult<TypeRef> {
        let lparen = self.advance(); // '('
        let span = token_span(&lparen);
        let mut fields = Vec::new();
        if !self.check(TokenKind::RParen) {
            let name_tok = self.expect(TokenKind::Identifier, "expected field name")?;
            let _ = self.expect(TokenKind::Colon, "expected ':'");
            let ty = self.parse_type()?;
            fields.push(RecordFieldType {
                name: name_tok.lexeme,
                ty,
            });
            while self.match_token(TokenKind::Comma) {
                if self.check(TokenKind::RParen) {
                    break;
                }
                let field_name = self.expect(TokenKind::Identifier, "expected field name")?;
                let _ = self.expect(TokenKind::Colon, "expected ':'");
                let field_ty = self.parse_type()?;
                fields.push(RecordFieldType {
                    name: field_name.lexeme,
                    ty: field_ty,
                });
            }
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        if fields.is_empty() {
            return Ok(self.alloc_type(span, TypeNode::Named { name: "void" }));
        }
        Ok(self.alloc_type(span, TypeNode::Record { fields }))
    }

    // =====================================================================
    // Expression parsing
    // =====================================================================

    /// Expression parsing entry point
    pub fn parse_expr(&mut self) -> ParseResult<ExprRef> {
        self.parse_binary(MIN_PREC)
    }

    /// Single Pratt parser
    fn parse_binary(&mut self, min_prec: u8) -> ParseResult<ExprRef> {
        let mut left = self.parse_unary()?;
        // After a block/if/match expression, block when the next token is `-` (Minus) or `&` (Ampersand).
        // Because `;` is treated as whitespace and skipped, `while c { ... }; -1` is equivalent to
        // `while c { ... } -1`. Without blocking, parse_binary would treat `-1` as `{ ... } - 1`
        // (subtraction), when the intent is `-1` as a standalone unary negation trailing expression.
        // Similarly, `{ ... } & x` would be parsed as bitwise-and, when the intent may be `{ ... }`
        // followed by `&x` (reference).
        // `-` and `&` are the only two operators that have both binary (subtraction/bitwise-and) and
        // unary (negation/reference) forms; other operators (+ * / % etc.) have no unary form (`*`
        // cross-line deref is already handled by check_multiline_deref), so they have no ambiguity
        // and need no blocking.
        // To perform subtraction/bitwise-and after a block/if/match, use parentheses: `(if c { ... }) - 1`.
        if matches!(
            &self.ast.expr(left).node,
            Expr::Block { .. } | Expr::If { .. } | Expr::Match { .. }
        ) && matches!(self.peek().kind, TokenKind::Minus | TokenKind::Ampersand) {
            return Ok(left);
        }
        while let Some(mapping) = lookup_binary_op(self.peek().kind) {
            if mapping.precedence < min_prec {
                break;
            }
            // `*` across lines is treated as dereference
            if mapping.check_multiline_deref && self.current > 0 {
                let prev_tok = self.tokens[self.current - 1];
                if self.peek().line != prev_tok.line {
                    break;
                }
            }
            let op_tok = self.advance();
            let next_min = if mapping.right_assoc {
                mapping.precedence
            } else {
                mapping.precedence + 1
            };
            let right = self.parse_binary(next_min)?;
            left = self.alloc_expr(
                token_span(&op_tok),
                Expr::Binary {
                    op: mapping.op,
                    lhs: left,
                    rhs: right,
                },
            );
        }
        Ok(left)
    }

    /// Parse a unary operation
    fn parse_unary(&mut self) -> ParseResult<ExprRef> {
        if self.match_token(TokenKind::Bang) {
            let op_tok = self.previous();
            let operand = self.parse_unary()?;
            return Ok(self.alloc_expr(token_span(&op_tok), Expr::Unary {
                op: UnaryOp::Not,
                operand,
            }));
        }
        if self.match_token(TokenKind::Tilde) {
            let op_tok = self.previous();
            let operand = self.parse_unary()?;
            return Ok(self.alloc_expr(token_span(&op_tok), Expr::Unary {
                op: UnaryOp::BitNot,
                operand,
            }));
        }
        if self.match_token(TokenKind::Ampersand) {
            let op_tok = self.previous();
            let operand = self.parse_unary()?;
            return Ok(self.alloc_expr(token_span(&op_tok), Expr::RefOf(operand)));
        }
        if self.match_token(TokenKind::AmpAmp) {
            let op_tok = self.previous();
            let operand = self.parse_unary()?;
            let inner = self.alloc_expr(token_span(&op_tok), Expr::RefOf(operand));
            return Ok(self.alloc_expr(token_span(&op_tok), Expr::RefOf(inner)));
        }
        if self.match_token(TokenKind::Star) {
            let op_tok = self.previous();
            let operand = self.parse_unary()?;
            return Ok(self.alloc_expr(token_span(&op_tok), Expr::Deref(operand)));
        }
        if self.match_token(TokenKind::Minus) {
            let op_tok = self.previous();
            // When a minus sign is immediately followed by a numeric literal, fold them directly
            if self.check(TokenKind::IntLiteral) {
                let lit_tok = self.advance();
                return self.parse_negative_int_literal(lit_tok);
            }
            if self.check(TokenKind::FloatLiteral) {
                let lit_tok = self.advance();
                return self.parse_negative_float_literal(lit_tok);
            }
            let operand = self.parse_unary()?;
            return Ok(self.alloc_expr(token_span(&op_tok), Expr::Unary {
                op: UnaryOp::Neg,
                operand,
            }));
        }
        self.parse_postfix()
    }

    /// Parse a postfix operation
    fn parse_postfix(&mut self) -> ParseResult<ExprRef> {
        let mut expr = self.parse_primary()?;
        loop {
            if self.match_token(TokenKind::Question) {
                let op_tok = self.previous();
                expr = self.alloc_expr(token_span(&op_tok), Expr::Propagate(expr));
            } else if self.match_token(TokenKind::Bang) {
                let op_tok = self.previous();
                expr = self.alloc_expr(token_span(&op_tok), Expr::NonNullAssert(expr));
            } else if self.match_token(TokenKind::QuestionDot) {
                let op_tok = self.previous();
                let field_tok = self.expect(TokenKind::Identifier, "expected field or method name")?;
                if self.check(TokenKind::LParen) {
                    let (args, type_args) = self.parse_call_args()?;
                    expr = self.alloc_expr(token_span(&op_tok), Expr::SafeMethodCall {
                        recv: expr,
                        method: field_tok.lexeme,
                        args,
                        type_args,
                    });
                } else {
                    expr = self.alloc_expr(token_span(&op_tok), Expr::SafeAccess {
                        recv: expr,
                        field: field_tok.lexeme,
                    });
                }
            } else if self.match_token(TokenKind::Dot) {
                let op_tok = self.previous();
                let field_tok = self.expect(TokenKind::Identifier, "expected field or method name")?;
                if self.check(TokenKind::LParen) {
                    let (args, type_args) = self.parse_call_args()?;
                    expr = self.alloc_expr(token_span(&op_tok), Expr::MethodCall {
                        recv: expr,
                        method: field_tok.lexeme,
                        args,
                        type_args,
                    });
                } else {
                    expr = self.alloc_expr(token_span(&op_tok), Expr::FieldAccess {
                        recv: expr,
                        field: field_tok.lexeme,
                    });
                }
            } else if self.check(TokenKind::LParen) {
                // Function call f(args)
                if matches!(self.ast.expr(expr).node, Expr::Call { .. }) {
                    self.report_error(
                        "chained call f(a)(b) is not allowed; use default currying: bind the partial result to a variable first",
                    )?;
                    unreachable!()
                }
                let call_tok = self.peek();
                let (args, type_args) = self.parse_call_args()?;
                expr = self.alloc_expr(token_span(&call_tok), Expr::Call {
                    callee: expr,
                    args,
                    type_args,
                });
            } else if self.check(TokenKind::Lt) && self.is_turbofish_call() {
                // turbofish call f<T>(args)
                self.advance(); // '<'
                let mut type_args = Vec::new();
                self.parse_type_arg_list(&mut type_args)?;
                let _ = self.expect_close_angle("expected '>'");
                if matches!(self.ast.expr(expr).node, Expr::Call { .. }) {
                    self.report_error(
                        "chained call f(a)(b) is not allowed; use default currying: bind the partial result to a variable first",
                    )?;
                    unreachable!()
                }
                let call_tok = self.peek();
                let _ = self.expect(TokenKind::LParen, "expected '('");
                let mut args = Vec::new();
                if !self.check(TokenKind::RParen) {
                    args.push(self.parse_expr()?);
                    while self.match_token(TokenKind::Comma) {
                        if self.check(TokenKind::RParen) {
                            break;
                        }
                        args.push(self.parse_expr()?);
                    }
                }
                let _ = self.expect(TokenKind::RParen, "expected ')'");
                expr = self.alloc_expr(token_span(&call_tok), Expr::Call {
                    callee: expr,
                    args,
                    type_args: Some(type_args),
                });
            } else if self.match_token(TokenKind::LBracket) {
                // Index or slice
                let bracket_tok = self.previous();
                let start = self.parse_binary(ADDITION_PREC)?;
                if self.match_token(TokenKind::DotDotEq) || self.match_token(TokenKind::DotDot) {
                    let inclusive = self.previous().kind == TokenKind::DotDotEq;
                    let end = self.parse_binary(ADDITION_PREC)?;
                    let _ = self.expect(TokenKind::RBracket, "expected ']' after slice end");
                    expr = self.alloc_expr(token_span(&bracket_tok), Expr::Slice {
                        recv: expr,
                        start,
                        end,
                        inclusive,
                    });
                } else {
                    let _ = self.expect(TokenKind::RBracket, "expected ']'");
                    expr = self.alloc_expr(token_span(&bracket_tok), Expr::Index {
                        recv: expr,
                        index: start,
                    });
                }
            } else {
                break;
            }
        }
        Ok(expr)
    }

    /// Parse call arguments (already at `(`)
    fn parse_call_args(&mut self) -> ParseResult<(Vec<ExprRef>, Option<Vec<TypeRef>>)> {
        // Optional turbofish <T> before (
        let type_args = if self.match_token(TokenKind::Lt) {
            let mut ta = Vec::new();
            self.parse_type_arg_list(&mut ta)?;
            if self.match_token(TokenKind::Gt) {
                Some(ta)
            } else {
                // Backtrack
                self.current -= ta.len() + 1;
                None
            }
        } else {
            None
        };
        let _ = self.expect(TokenKind::LParen, "expected '('");
        let mut args = Vec::new();
        if !self.check(TokenKind::RParen) {
            args.push(self.parse_expr()?);
            while self.match_token(TokenKind::Comma) {
                if self.check(TokenKind::RParen) {
                    break;
                }
                args.push(self.parse_expr()?);
            }
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        Ok((args, type_args))
    }

    /// Detect a turbofish call of the form `f<T>(args)`
    fn is_turbofish_call(&self) -> bool {
        if !self.check(TokenKind::Lt) {
            return false;
        }
        let mut i = self.current + 1;
        let mut depth: usize = 1;
        let mut steps: usize = 0;
        while i < self.tokens.len() && steps < 256 {
            match self.tokens[i].kind {
                TokenKind::Lt => depth += 1,
                TokenKind::Gt => {
                    depth -= 1;
                    if depth == 0 {
                        return i + 1 < self.tokens.len() && self.tokens[i + 1].kind == TokenKind::LParen;
                    }
                }
                TokenKind::LBrace | TokenKind::RBrace | TokenKind::Eq | TokenKind::EqGt | TokenKind::Eof => {
                    return false
                }
                _ => {}
            }
            i += 1;
            steps += 1;
        }
        false
    }

    /// Parse a primary expression
    fn parse_primary(&mut self) -> ParseResult<ExprRef> {
        if self.match_token(TokenKind::IntLiteral) {
            let tok = self.previous();
            return Ok(self.parse_int_literal(tok));
        }
        if self.match_token(TokenKind::FloatLiteral) {
            let tok = self.previous();
            return Ok(self.parse_float_literal(tok));
        }
        if self.match_token(TokenKind::TrueLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_expr(token_span(&tok), Expr::BoolLit(true)));
        }
        if self.match_token(TokenKind::FalseLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_expr(token_span(&tok), Expr::BoolLit(false)));
        }
        if self.match_token(TokenKind::CharLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_expr(token_span(&tok), Expr::CharLit(parse_char_value(tok.lexeme))));
        }
        if self.match_token(TokenKind::StringLiteral) {
            let tok = self.previous();
            return self.parse_string_literal(tok);
        }
        if self.match_token(TokenKind::NullLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_expr(token_span(&tok), Expr::NullLit));
        }
        // fun(params) body -> lambda
        if self.check(TokenKind::KwFun)
            && self.tokens.len() > self.current + 1
            && self.tokens[self.current + 1].kind == TokenKind::LParen
        {
            return self.parse_lambda_fun(false);
        }
        // async fun(params) body -> async lambda
        if self.check(TokenKind::KwAsync)
            && self.tokens.len() > self.current + 1
            && self.tokens[self.current + 1].kind == TokenKind::KwFun
        {
            self.advance();
            return self.parse_lambda_fun(true);
        }
        if self.match_token(TokenKind::KwIf) {
            return self.parse_if_expr();
        }
        if self.match_token(TokenKind::KwMatch) {
            return self.parse_match_expr();
        }
        if self.match_token(TokenKind::KwLazy) {
            return self.parse_lazy_expr();
        }
        if self.match_token(TokenKind::KwAtomic) {
            return self.parse_atomic_expr();
        }
        if self.match_token(TokenKind::KwSelect) {
            return self.parse_select_expr();
        }
        if self.check_identifier("cast") {
            self.advance();
            return self.parse_cast_builder();
        }
        if self.check(TokenKind::KwTrait)
            && self.tokens.len() > self.current + 1
            && self.tokens[self.current + 1].kind == TokenKind::LBrace
        {
            return self.parse_inline_trait_value();
        }
        if self.match_token(TokenKind::LBracket) {
            return self.parse_array_literal();
        }
        if self.check(TokenKind::LBrace) {
            return self.parse_block_expr();
        }
        if self.match_token(TokenKind::LParen) {
            return self.parse_paren_or_record_or_lambda();
        }
        // `type` keyword in expression position followed by `(` -> treated as an identifier
        if self.check(TokenKind::KwType)
            && self.tokens.len() > self.current + 1
            && self.tokens[self.current + 1].kind == TokenKind::LParen
        {
            let tok = self.advance();
            return Ok(self.alloc_expr(token_span(&tok), Expr::Ident(tok.lexeme)));
        }
        // `this` keyword in expression position: resolve as the implicit receiver identifier "this".
        // The parser injects a "this" parameter at params[0] of every method;
        // sema/IR resolve it like any other named parameter.
        if self.check(TokenKind::KwThis) {
            let tok = self.advance();
            return Ok(self.alloc_expr(token_span(&tok), Expr::Ident("this")));
        }
        if matches!(
            self.peek().kind,
            TokenKind::Identifier | TokenKind::KwVal | TokenKind::KwVar | TokenKind::KwChannel
        ) {
            // void in expression position denotes the unit value
            if self.check(TokenKind::Identifier) && self.peek().lexeme == "void" {
                let tok = self.advance();
                return Ok(self.alloc_expr(token_span(&tok), Expr::VoidLit));
            }
            let tok = self.advance();
            return Ok(self.alloc_expr(token_span(&tok), Expr::Ident(tok.lexeme)));
        }
        self.report_error("expected expression")?;
        unreachable!()
    }

    // =====================================================================
    // Literal parsing
    // =====================================================================

    /// Parse an integer literal, separating the numeric part from the type suffix
    fn parse_int_literal(&mut self, tok: Token<'a>) -> ExprRef {
        let raw = tok.lexeme;
        let mut i: usize = 0;
        if raw.len() > 2 && raw.as_bytes()[0] == b'0' {
            let p = raw.as_bytes()[1];
            if p == b'x' || p == b'X' || p == b'o' || p == b'O' || p == b'b' || p == b'B' {
                i = 2;
            }
        }
        while i < raw.len() && is_digit_or_underscore(raw.as_bytes()[i]) {
            i += 1;
        }
        if i < raw.len() && raw.len() > 2 && raw.as_bytes()[0] == b'0' {
            let p = raw.as_bytes()[1];
            if p == b'x' || p == b'X' {
                while i < raw.len() && is_hex_or_underscore(raw.as_bytes()[i]) {
                    i += 1;
                }
            }
        }
        let suffix = if i < raw.len() { Some(&raw[i..]) } else { None };
        self.alloc_expr(token_span(&tok), Expr::IntLit {
            raw: &raw[..i],
            suffix,
        })
    }

    /// Parse a negative integer literal, folding the minus sign into raw
    fn parse_negative_int_literal(&mut self, lit_tok: Token<'a>) -> ParseResult<ExprRef> {
        let raw = lit_tok.lexeme;
        let mut i: usize = 0;
        if raw.len() > 2 && raw.as_bytes()[0] == b'0' {
            let p = raw.as_bytes()[1];
            if p == b'x' || p == b'X' || p == b'o' || p == b'O' || p == b'b' || p == b'B' {
                i = 2;
            }
        }
        while i < raw.len() && is_digit_or_underscore(raw.as_bytes()[i]) {
            i += 1;
        }
        if i < raw.len() && raw.len() > 2 && raw.as_bytes()[0] == b'0' {
            let p = raw.as_bytes()[1];
            if p == b'x' || p == b'X' {
                while i < raw.len() && is_hex_or_underscore(raw.as_bytes()[i]) {
                    i += 1;
                }
            }
        }
        let suffix = if i < raw.len() { Some(&raw[i..]) } else { None };
        // Allocate "-" + raw[..i] in the arena
        let mut s = bumpalo::collections::String::new_in(self.arena);
        s.push('-');
        s.push_str(&raw[..i]);
        let neg_raw: &'a str = s.into_bump_str();
        Ok(self.alloc_expr(token_span(&lit_tok), Expr::IntLit {
            raw: neg_raw,
            suffix,
        }))
    }

    /// Parse a float literal, separating the numeric part from the type suffix
    fn parse_float_literal(&mut self, tok: Token<'a>) -> ExprRef {
        let (num_part, suffix) = split_float_suffix(tok.lexeme);
        self.alloc_expr(token_span(&tok), Expr::FloatLit {
            raw: num_part,
            suffix,
        })
    }

    /// Parse a negative float literal
    fn parse_negative_float_literal(&mut self, lit_tok: Token<'a>) -> ParseResult<ExprRef> {
        let (num_part, suffix) = split_float_suffix(lit_tok.lexeme);
        let mut s = bumpalo::collections::String::new_in(self.arena);
        s.push('-');
        s.push_str(num_part);
        let neg_raw: &'a str = s.into_bump_str();
        Ok(self.alloc_expr(token_span(&lit_tok), Expr::FloatLit {
            raw: neg_raw,
            suffix,
        }))
    }

    /// Parse a string literal (including interpolation handling)
    fn parse_string_literal(&mut self, tok: Token<'a>) -> ParseResult<ExprRef> {
        let raw = tok.lexeme;
        if !contains_interpolation(raw) {
            let content = &raw[1..raw.len() - 1];
            let value = self.unescape_string(content);
            return Ok(self.alloc_expr(token_span(&tok), Expr::StrLit(value)));
        }
        let content = &raw[1..raw.len() - 1];
        let mut parts = Vec::new();
        let bytes = content.as_bytes();
        let mut i: usize = 0;
        let mut literal_start: usize = 0;
        while i < content.len() {
            if bytes[i] == b'\\' {
                // Bug #36: \uXXXX and \u{XXXX} escape sequences must skip the entire sequence
                if i + 1 < content.len() && bytes[i + 1] == b'u' {
                    i += 2; // skip \u
                    if i < content.len() && bytes[i] == b'{' {
                        while i < content.len() && bytes[i] != b'}' {
                            i += 1;
                        }
                        if i < content.len() {
                            i += 1; // skip }
                        }
                    } else {
                        i += 4; // skip 4 hex digits
                    }
                    continue;
                }
                // \xHH: skip the entire 4-character sequence to avoid mistaking
                // hex-encoded braces (e.g. \x7b = '{') for interpolation start
                if i + 1 < content.len() && bytes[i + 1] == b'x' {
                    i += 4; // skip \xHH
                    continue;
                }
                i += 2;
                continue;
            }
            if bytes[i] == b'{' {
                if i + 1 < content.len() && bytes[i + 1] == b'{' {
                    i += 2;
                    continue;
                }
                if i > literal_start {
                    let text = self.unescape_string(&content[literal_start..i]);
                    parts.push(InterpolationPart::Literal(text));
                }
                i += 1;
                let expr_start = i;
                let mut brace_depth: usize = 1;
                while i < content.len() && brace_depth > 0 {
                    if bytes[i] == b'{' {
                        brace_depth += 1;
                    } else if bytes[i] == b'}' {
                        brace_depth -= 1;
                    } else if bytes[i] == b'\\' {
                        // Skip escape sequences (\" \\ \n \t \r \{ \})
                        i += 1;
                    } else if bytes[i] == b'"' {
                        // Bug #54: nested string literal inside an interpolation expression —
                        // scan the complete nested string (including \" escapes), ensuring expr_text contains the correct string literal
                        i += 1;
                        while i < content.len() {
                            if bytes[i] == b'\\' {
                                i += 1;
                            } else if bytes[i] == b'"' {
                                break;
                            }
                            i += 1;
                        }
                    }
                    i += 1;
                }
                let expr_text = &content[expr_start..i - 1];
                // Bug #70: Empty interpolation `{}` — report a clear error instead of
                // silently failing inside parse_interpolation_expr (which truncates errors).
                if expr_text.trim().is_empty() {
                    let span = token_span(&tok);
                    return Err(ParseError {
                        line: span.line,
                        column: span.column + i as u32,
                        message: "empty interpolation expression in string literal; use {{}} for literal braces".to_string(),
                    });
                }
                // Bug #54: the interpolation expression text may contain escape sequences from the
                // outer string (e.g. \"); unescape it before passing to parse_interpolation_expr.
                let unescaped_expr = self.unescape_string(expr_text);
                let expr = if unescaped_expr == expr_text {
                    self.parse_interpolation_expr(expr_text)?
                } else {
                    // unescape produced different content; need a way to pass it with the 'a lifetime.
                    // Store the unescaped text in the arena and parse it.
                    let leaked: &'a str = self.arena.alloc_str(&unescaped_expr);
                    self.parse_interpolation_expr(leaked)?
                };
                parts.push(InterpolationPart::Expression(expr));
                literal_start = i;
                continue;
            }
            i += 1;
        }
        if literal_start < content.len() {
            let text = self.unescape_string(&content[literal_start..]);
            parts.push(InterpolationPart::Literal(text));
        }
        Ok(self.alloc_expr(token_span(&tok), Expr::StrInterp(parts)))
    }

    /// Lexically and syntactically parse an interpolation expression text
    fn parse_interpolation_expr(&mut self, text: &'a str) -> ParseResult<ExprRef> {
        let mut lexer = Lexer::new(text);
        let mut sink = TokenCollector::new();
        lexer.tokenize_into(&mut sink);
        let tokens = sink.into_tokens();
        let tokens_ref: &'a [Token<'a>] = self.arena.alloc_slice_copy(&tokens);

        let saved_tokens = self.tokens;
        let saved_current = self.current;
        let saved_pending_eq = self.pending_eq;
        let saved_pending_gt = self.pending_gt;
        let saved_pending_gt_eq = self.pending_gt_eq;
        let saved_error_count = self.handler.errors().len();

        self.tokens = tokens_ref;
        self.current = 0;
        self.pending_eq = false;
        self.pending_gt = false;
        self.pending_gt_eq = false;

        let result = self.parse_expr();
        // Restore state
        self.tokens = saved_tokens;
        self.current = saved_current;
        self.pending_eq = saved_pending_eq;
        self.pending_gt = saved_pending_gt;
        self.pending_gt_eq = saved_pending_gt_eq;
        if result.is_err() {
            self.handler.truncate_errors(saved_error_count);
        }
        result
    }

    /// Unescape a string
    fn unescape_string(&self, text: &'a str) -> &'a str {
        // Fast path: return zero-copy if no escapes
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < text.len() {
            if bytes[i] == b'\\' {
                break;
            }
            if bytes[i] == b'{' && i + 1 < text.len() && bytes[i + 1] == b'{' {
                break;
            }
            if bytes[i] == b'}' && i + 1 < text.len() && bytes[i + 1] == b'}' {
                break;
            }
            i += 1;
        }
        if i >= text.len() {
            return text;
        }
        // Slow path
        let mut result = bumpalo::collections::String::new_in(self.arena);
        let mut j = 0;
        while j < text.len() {
            if bytes[j] == b'\\' && j + 1 < text.len() {
                match bytes[j + 1] {
                    b'n' => {
                        result.push('\n');
                        j += 2;
                    }
                    b't' => {
                        result.push('\t');
                        j += 2;
                    }
                    b'r' => {
                        result.push('\r');
                        j += 2;
                    }
                    b'\\' => {
                        result.push('\\');
                        j += 2;
                    }
                    b'"' => {
                        result.push('"');
                        j += 2;
                    }
                    b'{' => {
                        result.push('{');
                        j += 2;
                    }
                    b'}' => {
                        result.push('}');
                        j += 2;
                    }
                    b'0' => {
                        // Bug #36: \0 → NUL (U+0000)
                        result.push('\0');
                        j += 2;
                    }
                    b'u' => {
                        // Bug #36: \uXXXX (4-digit hex) or \u{XXXX} (brace form).
                        // Lexical scanning (scan_string) has already validated the hex digits;
                        // use expect here to assert the invariant.
                        j += 2; // skip \u
                        let code = if j < text.len() && bytes[j] == b'{' {
                            // \u{XXXX} brace form: 1-6 hex digits
                            j += 1; // skip {
                            let hex_start = j;
                            while j < text.len() && bytes[j] != b'}' {
                                j += 1;
                            }
                            let hex_str = std::str::from_utf8(&bytes[hex_start..j])
                                .expect("scan_string validated hex digits");
                            j += 1; // skip }
                            u32::from_str_radix(hex_str, 16)
                                .expect("scan_string validated hex digits")
                        } else {
                            // \uXXXX without braces: exactly 4 hex digits
                            let hex_end = std::cmp::min(j + 4, text.len());
                            let hex_str = std::str::from_utf8(&bytes[j..hex_end])
                                .expect("scan_string validated hex digits");
                            j = hex_end;
                            u32::from_str_radix(hex_str, 16)
                                .expect("scan_string validated hex digits")
                        };
                        let c = char::from_u32(code)
                            .expect("scan_string validated codepoint range");
                        result.push(c);
                    }
                    b'x' => {
                        // \xHH: exactly 2 hex digits, byte value 0x00-0xFF.
                        // Lexical scanning (scan_string) has already validated the hex digits.
                        j += 2; // skip \x
                        let hex_end = std::cmp::min(j + 2, text.len());
                        let hex_str = std::str::from_utf8(&bytes[j..hex_end])
                            .expect("scan_string validated hex digits");
                        j = hex_end;
                        let code = u32::from_str_radix(hex_str, 16)
                            .expect("scan_string validated hex digits");
                        let c = char::from_u32(code)
                            .expect("scan_string validated byte range 0x00-0xFF");
                        result.push(c);
                    }
                    _ => {
                        result.push(bytes[j] as char);
                        j += 1;
                    }
                }
            } else if bytes[j] == b'{' && j + 1 < text.len() && bytes[j + 1] == b'{' {
                result.push('{');
                j += 2;
            } else if bytes[j] == b'}' && j + 1 < text.len() && bytes[j + 1] == b'}' {
                result.push('}');
                j += 2;
            } else {
                result.push(bytes[j] as char);
                j += 1;
            }
        }
        result.into_bump_str()
    }

    // =====================================================================
    // Special expressions
    // =====================================================================

    /// Parse a lambda starting with the fun keyword: `fun(params) body`
    fn parse_lambda_fun(&mut self, is_async: bool) -> ParseResult<ExprRef> {
        let fun_tok = self.advance(); // 'fun'
        let span = token_span(&fun_tok);
        let mut params = Vec::new();
        let _ = self.expect(TokenKind::LParen, "expected '('");
        if !self.check(TokenKind::RParen) {
            self.parse_param_list(&mut params)?;
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        let body_expr = self.parse_expr()?;
        Ok(self.alloc_expr(span, Expr::Lambda {
            params,
            body: LambdaBody::Block(body_expr),
            is_async,
            return_type: None,
        }))
    }

    /// Attempt to parse a lambda: `(params) => expr` (backtracks on failure)
    fn try_parse_lambda(&mut self, saved: usize, span: Span) -> Option<ExprRef> {
        let saved_error_count = self.handler.errors().len();
        let mut params = Vec::new();
        if !self.check(TokenKind::RParen)
            && self.parse_lambda_param_list(&mut params).is_err()
        {
            self.handler.truncate_errors(saved_error_count);
            self.current = saved;
            return None;
        }
        if !self.check(TokenKind::RParen) {
            self.handler.truncate_errors(saved_error_count);
            self.current = saved;
            return None;
        }
        self.advance(); // ')'
        if !self.check(TokenKind::EqGt) {
            self.current = saved;
            self.handler.truncate_errors(saved_error_count);
            return None;
        }
        self.advance(); // '=>'
        let body_expr = match self.parse_expr() {
            Ok(e) => e,
            Err(_) => {
                self.current = saved;
                self.handler.truncate_errors(saved_error_count);
                return None;
            }
        };
        Some(self.alloc_expr(span, Expr::Lambda {
            params,
            body: LambdaBody::Expression(body_expr),
            is_async: false,
            return_type: None,
        }))
    }

    impl_parse_comma_list!(parse_lambda_param_list, Param<'a>, parse_lambda_param, check(TokenKind::RParen));

    fn parse_lambda_param(&mut self) -> ParseResult<Param<'a>> {
        let name_tok = self.expect(TokenKind::Identifier, "expected parameter name")?;
        let type_annotation = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        Ok(Param {
            name: name_tok.lexeme,
            type_annotation,
        })
    }

    /// Parse a parenthesized expression: unit value, lambda, record literal, record extend, or grouping
    fn parse_paren_or_record_or_lambda(&mut self) -> ParseResult<ExprRef> {
        let lparen_tok = self.previous();
        let span = token_span(&lparen_tok);
        if self.match_token(TokenKind::RParen) {
            return Ok(self.alloc_expr(span, Expr::VoidLit));
        }
        let saved = self.current;
        if let Some(lambda) = self.try_parse_lambda(saved, span) {
            return Ok(lambda);
        }
        self.current = saved;
        // Record extend: (...base, field: value)
        if self.peek().kind == TokenKind::Ellipsis {
            self.advance();
            let base_expr = self.parse_expr()?;
            let mut updates = Vec::new();
            while self.match_token(TokenKind::Comma) {
                if self.check(TokenKind::RParen) {
                    break;
                }
                let field_name = self.expect(TokenKind::Identifier, "expected field name")?;
                let _ = self.expect(TokenKind::Colon, "expected ':'");
                let field_value = self.parse_expr()?;
                updates.push(RecordFieldExpr {
                    name: field_name.lexeme,
                    value: field_value,
                });
            }
            let _ = self.expect(TokenKind::RParen, "expected ')'");
            return Ok(self.alloc_expr(span, Expr::RecordExtend {
                base: base_expr,
                updates,
            }));
        }
        // Record literal: (field: value, ...)
        if self.peek().kind == TokenKind::Identifier {
            let name_tok = self.advance();
            if self.check(TokenKind::Colon) {
                self.advance();
                let value = self.parse_expr()?;
                let mut fields = vec![RecordFieldExpr {
                    name: name_tok.lexeme,
                    value,
                }];
                while self.match_token(TokenKind::Comma) {
                    if self.check(TokenKind::RParen) {
                        break;
                    }
                    if self.check(TokenKind::Ellipsis) {
                        // (field: value, ...base, more: value)
                        self.advance();
                        let base_expr = self.parse_expr()?;
                        let mut updates = fields.clone();
                        while self.match_token(TokenKind::Comma) {
                            if self.check(TokenKind::RParen) {
                                break;
                            }
                            let field_name = self.expect(TokenKind::Identifier, "expected field name")?;
                            let _ = self.expect(TokenKind::Colon, "expected ':'");
                            let field_value = self.parse_expr()?;
                            updates.push(RecordFieldExpr {
                                name: field_name.lexeme,
                                value: field_value,
                            });
                        }
                        let _ = self.expect(TokenKind::RParen, "expected ')'");
                        return Ok(self.alloc_expr(span, Expr::RecordExtend {
                            base: base_expr,
                            updates,
                        }));
                    }
                    let field_name = self.expect(TokenKind::Identifier, "expected field name")?;
                    let _ = self.expect(TokenKind::Colon, "expected ':'");
                    let field_value = self.parse_expr()?;
                    fields.push(RecordFieldExpr {
                        name: field_name.lexeme,
                        value: field_value,
                    });
                }
                let _ = self.expect(TokenKind::RParen, "expected ')'");
                return Ok(self.alloc_expr(span, Expr::RecordLit(fields)));
            }
            self.current = saved;
        }
        // Plain grouping expression
        let first_expr = self.parse_expr()?;
        if self.match_token(TokenKind::Comma) {
            // Anonymous tuples are not allowed
            self.report_error_at(
                lparen_tok.line,
                lparen_tok.column,
                "anonymous tuples are not allowed; use named record fields like (name: value, ...)",
            );
            return Ok(self.alloc_expr(token_span(&lparen_tok), Expr::VoidLit));
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        Ok(first_expr)
    }

    /// Parse a cast builder: `cast(expr).to(T)` / `cast(expr).try_to(T)`
    ///
    /// The special syntax is desugared into a plain function call:
    ///   cast(x).to(T)      -> __cast_to<T>(x)
    ///   cast(x).try_to(T)  -> __cast_try_to<T>(x)
    ///
    /// After sema infers the source type S, this resolves to a __cast_S_to_T(x) function call.
    fn parse_cast_builder(&mut self) -> ParseResult<ExprRef> {
        let cast_tok = self.previous();
        let span = token_span(&cast_tok);
        let _ = self.expect(TokenKind::LParen, "expected '(' after 'cast'");
        let expr = self.parse_expr()?;
        let _ = self.expect(TokenKind::RParen, "expected ')' after cast expression");
        let _ = self.expect(TokenKind::Dot, "expected '.to(...)' or '.try_to(...)' after cast(...)");
        if !self.check(TokenKind::Identifier) {
            self.report_error("expected 'to' or 'try_to' after cast(...).")?;
            unreachable!()
        }
        let method_tok = self.advance();
        let callee_name = match method_tok.lexeme {
            "to" => "__cast_to",
            "try_to" => "__cast_try_to",
            _ => {
                self.report_error("expected 'to' or 'try_to' after cast(...).")?;
                unreachable!()
            }
        };
        let _ = self.expect(TokenKind::LParen, "expected '(' after cast method");
        if !self.check(TokenKind::Identifier) {
            self.report_error("expected type name as cast target")?;
            unreachable!()
        }
        let type_tok = self.advance();
        let target = self.alloc_type(token_span(&type_tok), TypeNode::Named { name: type_tok.lexeme });
        let _ = self.expect(TokenKind::RParen, "expected ')' after cast target type");
        // Desugar into a plain Call: __cast_to<T>(x) / __cast_try_to<T>(x)
        let callee = self.alloc_expr(span, Expr::Ident(callee_name));
        Ok(self.alloc_expr(span, Expr::Call {
            callee,
            args: vec![expr],
            type_args: Some(vec![target]),
        }))
    }

    /// Parse an if expression
    fn parse_if_expr(&mut self) -> ParseResult<ExprRef> {
        let if_tok = self.previous();
        let span = token_span(&if_tok);
        self.reject_paren_condition("if")?;
        let cond = self.parse_expr()?;
        // Use parse_unary instead of parse_expr for then_branch/else_branch, to avoid greedily
        // consuming subsequent binary operators. Because `;` is treated as whitespace and skipped,
        // `if c { ... }; -1` is equivalent to `if c { ... } -1`; parse_expr would treat `-1` as
        // `{ ... } - 1` (subtraction) rather than a standalone expression. parse_unary parses only
        // a single unary expression (e.g. `{ ... }` block) and does not consume the trailing `-N`.
        let then_branch = self.parse_unary()?;
        let else_branch = if self.match_token(TokenKind::KwElse) {
            Some(self.parse_unary()?)
        } else {
            None
        };
        Ok(self.alloc_expr(span, Expr::If {
            cond,
            then_branch,
            else_branch,
        }))
    }

    /// Parse a match expression
    fn parse_match_expr(&mut self) -> ParseResult<ExprRef> {
        let match_tok = self.previous();
        let span = token_span(&match_tok);
        self.reject_paren_condition("match")?;
        let scrutinee = self.parse_expr()?;
        let _ = self.expect(TokenKind::LBrace, "expected '{'");
        let mut arms = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            let arm = match self.parse_match_arm() {
                Ok(a) => a,
                Err(_) => {
                    while !self.check(TokenKind::Comma) && !self.check(TokenKind::RBrace) && !self.is_at_end() {
                        self.advance();
                    }
                    if self.match_token(TokenKind::Comma) {
                        continue;
                    }
                    break;
                }
            };
            arms.push(arm);
            self.match_token(TokenKind::Comma);
        }
        let _ = self.expect(TokenKind::RBrace, "expected '}'");
        Ok(self.alloc_expr(span, Expr::Match { scrutinee, arms }))
    }

    /// Parse a match arm
    fn parse_match_arm(&mut self) -> ParseResult<MatchArm> {
        let pattern = self.parse_pattern()?;
        let guard = if self.match_token(TokenKind::KwIf) {
            self.reject_paren_condition("if guard")?;
            Some(self.parse_expr()?)
        } else {
            None
        };
        let _ = self.expect(TokenKind::EqGt, "expected '=>'");
        // Wrap control-flow statements as block expressions when used as an arm body
        let body = if matches!(
            self.peek().kind,
            TokenKind::KwThrow | TokenKind::KwReturn | TokenKind::KwBreak | TokenKind::KwContinue
        ) {
            let stmt_tok = self.peek();
            let stmt = self.parse_stmt()?;
            let span = token_span(&stmt_tok);
            self.alloc_expr(span, Expr::Block {
                stmts: vec![stmt],
                trailing: None,
            })
        } else {
            self.parse_expr()?
        };
        Ok(MatchArm {
            pattern,
            guard,
            body,
        })
    }

    fn parse_lazy_expr(&mut self) -> ParseResult<ExprRef> {
        let lazy_tok = self.previous();
        let expr = self.parse_expr()?;
        Ok(self.alloc_expr(token_span(&lazy_tok), Expr::Lazy(expr)))
    }

    fn parse_atomic_expr(&mut self) -> ParseResult<ExprRef> {
        let atomic_tok = self.previous();
        let value = self.parse_primary()?;
        Ok(self.alloc_expr(token_span(&atomic_tok), Expr::Atomic(value)))
    }

    fn parse_select_expr(&mut self) -> ParseResult<ExprRef> {
        let select_tok = self.previous();
        let span = token_span(&select_tok);
        let _ = self.expect(TokenKind::LBrace, "expected '{'");
        let mut arms = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            arms.push(self.parse_select_arm()?);
            self.match_token(TokenKind::Comma);
        }
        let _ = self.expect(TokenKind::RBrace, "expected '}'");
        Ok(self.alloc_expr(span, Expr::Select(arms)))
    }

    fn parse_select_arm(&mut self) -> ParseResult<SelectArm<'a>> {
        if self.check_identifier("timeout") {
            let _timeout_tok = self.advance();
            let _ = self.expect(TokenKind::LParen, "expected '('");
            let duration = self.parse_expr()?;
            let _ = self.expect(TokenKind::RParen, "expected ')'");
            let _ = self.expect(TokenKind::EqGt, "expected '=>'");
            let body = self.parse_expr()?;
            return Ok(SelectArm::Timeout { duration, body });
        }
        let channel_expr = self.parse_expr()?;
        let _ = self.expect(TokenKind::EqGt, "expected '=>'");
        let binding = if self.check(TokenKind::Identifier)
            && self.current + 1 < self.tokens.len()
            && self.tokens[self.current + 1].kind == TokenKind::EqGt
        {
            let name_tok = self.advance();
            self.advance(); // '=>'
            Some(name_tok.lexeme)
        } else {
            None
        };
        let body = self.parse_expr()?;
        Ok(SelectArm::Receive {
            channel_expr,
            binding,
            body,
        })
    }

    fn parse_inline_trait_value(&mut self) -> ParseResult<ExprRef> {
        let trait_tok = self.advance(); // 'trait'
        let span = token_span(&trait_tok);
        let _ = self.expect(TokenKind::LBrace, "expected '{'");
        let mut methods = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            methods.push(self.parse_method_decl()?);
        }
        let _ = self.expect(TokenKind::RBrace, "expected '}'");
        Ok(self.alloc_expr(span, Expr::InlineTrait(methods)))
    }

    /// Parse an array literal
    fn parse_array_literal(&mut self) -> ParseResult<ExprRef> {
        let bracket_tok = self.previous();
        let span = token_span(&bracket_tok);
        let mut elements = Vec::new();
        if !self.check(TokenKind::RBracket) {
            elements.push(self.parse_expr()?);
            // Array fill syntax [value, ..count]
            if self.match_token(TokenKind::Comma) {
                if self.match_token(TokenKind::DotDot) {
                    let count = self.parse_expr()?;
                    let _ = self.expect(TokenKind::RBracket, "expected ']' after array fill count");
                    let value = elements[0];
                    return Ok(self.alloc_expr(span, Expr::ArrayLit {
                        elements,
                        fill: Some((value, count)),
                    }));
                }
                // Regular multi-element
                if !self.check(TokenKind::RBracket) {
                    elements.push(self.parse_expr()?);
                    while self.match_token(TokenKind::Comma) {
                        if self.check(TokenKind::RBracket) {
                            break;
                        }
                        elements.push(self.parse_expr()?);
                    }
                }
            }
        }
        let _ = self.expect(TokenKind::RBracket, "expected ']'");
        Ok(self.alloc_expr(span, Expr::ArrayLit {
            elements,
            fill: None,
        }))
    }

    /// Parse a block expression
    fn parse_block_expr(&mut self) -> ParseResult<ExprRef> {
        let brace_tok = self.advance(); // '{'
        let span = token_span(&brace_tok);
        let mut stmts = Vec::new();
        let mut trailing = None;
        while !self.check(TokenKind::RBrace) && !self.is_at_end() {
            if self.is_stmt_start() {
                let stmt = self.parse_stmt()?;
                if matches!(self.ast.stmt(stmt).node, Stmt::Expression { .. }) && self.check(TokenKind::RBrace) {
                    if let Stmt::Expression { expr } = &self.ast.stmt(stmt).node {
                        trailing = Some(*expr);
                        break;
                    }
                }
                stmts.push(stmt);
            } else {
                let stmt = self.parse_expr_or_assignment_stmt()?;
                if matches!(self.ast.stmt(stmt).node, Stmt::Expression { .. }) && self.check(TokenKind::RBrace) {
                    if let Stmt::Expression { expr } = &self.ast.stmt(stmt).node {
                        trailing = Some(*expr);
                        break;
                    }
                }
                stmts.push(stmt);
            }
        }
        let _ = self.expect(TokenKind::RBrace, "expected '}'");
        Ok(self.alloc_expr(span, Expr::Block { stmts, trailing }))
    }

    // =====================================================================
    // Statement parsing
    // =====================================================================

    fn is_stmt_start(&self) -> bool {
        matches!(
            self.peek().kind,
            TokenKind::KwVal
                | TokenKind::KwVar
                | TokenKind::KwFun
                | TokenKind::KwType
                | TokenKind::KwTrait
                | TokenKind::KwReturn
                | TokenKind::KwDefer
                | TokenKind::KwThrow
                | TokenKind::KwBreak
                | TokenKind::KwContinue
                | TokenKind::KwFor
                | TokenKind::KwWhile
                | TokenKind::KwLoop
        )
    }

    /// Statement parsing entry point
    fn parse_stmt(&mut self) -> ParseResult<StmtRef> {
        if self.match_token(TokenKind::KwVal) {
            return self.parse_val_decl();
        }
        if self.match_token(TokenKind::KwVar) {
            return self.parse_var_decl();
        }
        if self.match_token(TokenKind::KwFun) {
            return self.parse_fun_stmt();
        }
        if self.check(TokenKind::KwType) {
            let decl = self.parse_type_decl(Visibility::Private)?;
            return Ok(self.alloc_stmt(decl.span, Stmt::LocalDecl { decl: Box::new(decl.node) }));
        }
        if self.check(TokenKind::KwTrait) {
            let decl = self.parse_trait_decl(Visibility::Private)?;
            return Ok(self.alloc_stmt(decl.span, Stmt::LocalDecl { decl: Box::new(decl.node) }));
        }
        if self.match_token(TokenKind::KwReturn) {
            return self.parse_return_stmt();
        }
        if self.match_token(TokenKind::KwDefer) {
            return self.parse_defer_stmt();
        }
        if self.match_token(TokenKind::KwThrow) {
            return self.parse_throw_stmt();
        }
        if self.match_token(TokenKind::KwBreak) {
            let tok = self.previous();
            return Ok(self.alloc_stmt(token_span(&tok), Stmt::Break));
        }
        if self.match_token(TokenKind::KwContinue) {
            let tok = self.previous();
            return Ok(self.alloc_stmt(token_span(&tok), Stmt::Continue));
        }
        if self.match_token(TokenKind::KwFor) {
            return self.parse_for_stmt();
        }
        if self.match_token(TokenKind::KwWhile) {
            return self.parse_while_stmt();
        }
        if self.match_token(TokenKind::KwLoop) {
            return self.parse_loop_stmt();
        }
        self.parse_expr_or_assignment_stmt()
    }

    /// Parse a fun statement
    /// Named fun -> LocalDecl(Decl::FunDecl) (unified nested declaration entry point)
    /// Anonymous fun(params) body -> Expression(Lambda)
    fn parse_fun_stmt(&mut self) -> ParseResult<StmtRef> {
        let fun_tok = self.previous();
        let span = token_span(&fun_tok);
        if self.check(TokenKind::Identifier) && !self.check_identifier("in") {
            let name_tok = self.expect(TokenKind::Identifier, "expected function name")?;
            let mut type_params = Vec::new();
            if self.match_token(TokenKind::Lt) {
                self.parse_type_param_list(&mut type_params)?;
                let _ = self.expect_close_angle("expected '>' to close type parameter list");
            }
            let mut params = Vec::new();
            let _ = self.expect(TokenKind::LParen, "expected '('");
            if !self.check(TokenKind::RParen) {
                self.parse_param_list(&mut params)?;
            }
            let _ = self.expect(TokenKind::RParen, "expected ')'");
            let return_type = if self.match_token(TokenKind::Colon) {
                Some(self.parse_type()?)
            } else {
                None
            };
            let mut bounds = Vec::new();
            if self.match_token(TokenKind::KwWith) {
                self.parse_trait_bound_list(&mut bounds)?;
            }
            let body = self.parse_expr()?;
            let decl = Decl::FunDecl {
                visibility: Visibility::Private,
                name: name_tok.lexeme,
                type_params,
                params,
                return_type,
                bounds,
                body,
                is_async: false,
                is_entry: false,
                attributes: Vec::new(),
                extern_c_body: None,
            };
            return Ok(self.alloc_stmt(span, Stmt::LocalDecl { decl: Box::new(decl) }));
        }
        // Anonymous lambda
        let mut params = Vec::new();
        let _ = self.expect(TokenKind::LParen, "expected '('");
        if !self.check(TokenKind::RParen) {
            self.parse_param_list(&mut params)?;
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        let body_expr = self.parse_expr()?;
        let lambda = self.alloc_expr(span, Expr::Lambda {
            params,
            body: LambdaBody::Block(body_expr),
            is_async: false,
            return_type: None,
        });
        Ok(self.alloc_stmt(span, Stmt::Expression { expr: lambda }))
    }

    fn parse_val_decl(&mut self) -> ParseResult<StmtRef> {
        let val_tok = self.previous();
        let name_tok = self.expect(TokenKind::Identifier, "expected variable name")?;
        let type_annotation = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let _ = self.expect(TokenKind::Eq, "expected '='");
        let value = self.parse_expr()?;
        Ok(self.alloc_stmt(token_span(&val_tok), Stmt::ValDecl {
            name: name_tok.lexeme,
            type_annotation,
            value,
            visibility: Visibility::Private,
        }))
    }

    fn parse_var_decl(&mut self) -> ParseResult<StmtRef> {
        let var_tok = self.previous();
        let name_tok = self.expect(TokenKind::Identifier, "expected variable name")?;
        let type_annotation = if self.match_token(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let _ = self.expect(TokenKind::Eq, "expected '='");
        let value = self.parse_expr()?;
        Ok(self.alloc_stmt(token_span(&var_tok), Stmt::VarDecl {
            name: name_tok.lexeme,
            type_annotation,
            value,
            visibility: Visibility::Private,
        }))
    }

    fn parse_return_stmt(&mut self) -> ParseResult<StmtRef> {
        let return_tok = self.previous();
        let value = if !self.check(TokenKind::RBrace) && !self.is_stmt_start() && !self.is_at_end() {
            Some(self.parse_expr()?)
        } else {
            None
        };
        Ok(self.alloc_stmt(token_span(&return_tok), Stmt::Return { value }))
    }

    fn parse_defer_stmt(&mut self) -> ParseResult<StmtRef> {
        let defer_tok = self.previous();
        let span = token_span(&defer_tok);
        let expr = self.parse_expr()?;
        if self.match_token(TokenKind::Eq) {
            let value = self.parse_expr()?;
            let expr_span = self.ast.expr(expr).span;
            let assign_expr = self.alloc_expr(expr_span, Expr::Assign {
                target: expr,
                value,
            });
            return Ok(self.alloc_stmt(span, Stmt::Defer {
                expr: assign_expr,
            }));
        }
        Ok(self.alloc_stmt(span, Stmt::Defer { expr }))
    }

    fn parse_throw_stmt(&mut self) -> ParseResult<StmtRef> {
        let throw_tok = self.previous();
        let expr = self.parse_expr()?;
        Ok(self.alloc_stmt(token_span(&throw_tok), Stmt::Throw { expr }))
    }

    fn parse_for_stmt(&mut self) -> ParseResult<StmtRef> {
        let for_tok = self.previous();
        let span = token_span(&for_tok);
        let name_tok = self.expect(TokenKind::Identifier, "expected iterator variable name")?;
        let _ = self.expect(TokenKind::KwIn, "expected 'in'");
        self.reject_paren_condition("for")?;
        let iterable = self.parse_expr()?;
        // Use parse_unary for the body (same as parse_while_stmt, to avoid greedily consuming subsequent binary operators)
        let body = self.parse_unary()?;
        Ok(self.alloc_stmt(span, Stmt::For {
            name: name_tok.lexeme,
            iterable,
            body,
        }))
    }

    fn parse_while_stmt(&mut self) -> ParseResult<StmtRef> {
        let while_tok = self.previous();
        let span = token_span(&while_tok);
        self.reject_paren_condition("while")?;
        let condition = self.parse_expr()?;
        // Use parse_unary instead of parse_expr for the body, to avoid greedily consuming
        // subsequent binary operators. Because `;` is treated as whitespace and skipped,
        // `{ ... }; -1` is equivalent to `{ ... } -1`; parse_expr would treat `-1` as
        // `{ ... } - 1` (subtraction) rather than a standalone expression. parse_unary parses
        // only a single unary expression (e.g. `{ ... }` block) and does not consume the trailing `-N`.
        let body = self.parse_unary()?;
        Ok(self.alloc_stmt(span, Stmt::While { condition, body }))
    }

    fn parse_loop_stmt(&mut self) -> ParseResult<StmtRef> {
        let loop_tok = self.previous();
        // Use parse_unary for the body (same as parse_while_stmt, to avoid greedily consuming subsequent binary operators)
        let body = self.parse_unary()?;
        Ok(self.alloc_stmt(token_span(&loop_tok), Stmt::Loop { body }))
    }

    fn parse_expr_or_assignment_stmt(&mut self) -> ParseResult<StmtRef> {
        let expr = self.parse_expr()?;
        if self.match_token(TokenKind::Eq) {
            let eq_tok = self.previous();
            let value = self.parse_expr()?;
            // Copy node info out of the arena first to avoid borrow conflict with alloc_stmt's &mut self
            let expr_span = self.ast.expr(expr).span;
            let (is_ident, field_info): (bool, Option<(ExprRef, &'a str)>) =
                match &self.ast.expr(expr).node {
                    Expr::Ident(_) => (true, None),
                    Expr::FieldAccess { recv, field, .. } => (false, Some((*recv, *field))),
                    _ => (false, None),
                };
            return Ok(if is_ident {
                self.alloc_stmt(expr_span, Stmt::Assignment {
                    target: expr,
                    value,
                })
            } else if let Some((recv, field)) = field_info {
                self.alloc_stmt(token_span(&eq_tok), Stmt::FieldAssignment {
                    object: recv,
                    field,
                    value,
                })
            } else {
                self.alloc_stmt(token_span(&eq_tok), Stmt::Assignment {
                    target: expr,
                    value,
                })
            });
        }
        if let Some(op) = self.peek_compound_assign() {
            self.advance();
            let op_tok = self.previous();
            let value = self.parse_expr()?;
            return Ok(self.alloc_stmt(token_span(&op_tok), Stmt::CompoundAssignment {
                target: expr,
                op,
                value,
            }));
        }
        let expr_span = self.ast.expr(expr).span;
        Ok(self.alloc_stmt(expr_span, Stmt::Expression { expr }))
    }

    fn peek_compound_assign(&self) -> Option<CompoundAssignOp> {
        match self.peek().kind {
            TokenKind::PlusEq => Some(CompoundAssignOp::AddAssign),
            TokenKind::MinusEq => Some(CompoundAssignOp::SubAssign),
            TokenKind::StarEq => Some(CompoundAssignOp::MulAssign),
            TokenKind::SlashEq => Some(CompoundAssignOp::DivAssign),
            TokenKind::PercentEq => Some(CompoundAssignOp::ModAssign),
            TokenKind::AmpEq => Some(CompoundAssignOp::BitAndAssign),
            TokenKind::PipeEq => Some(CompoundAssignOp::BitOrAssign),
            TokenKind::CaretEq => Some(CompoundAssignOp::BitXorAssign),
            TokenKind::LtLtEq => Some(CompoundAssignOp::ShlAssign),
            TokenKind::GtGtEq => Some(CompoundAssignOp::ShrAssign),
            _ => None,
        }
    }

    // =====================================================================
    // Pattern parsing
    // =====================================================================

    fn parse_pattern(&mut self) -> ParseResult<PatternRef> {
        self.parse_or_pattern()
    }

    fn parse_or_pattern(&mut self) -> ParseResult<PatternRef> {
        let mut left = self.parse_primary_pattern()?;
        while self.match_token(TokenKind::Pipe) {
            let pipe_tok = self.previous();
            let right = self.parse_primary_pattern()?;
            left = self.alloc_pattern(token_span(&pipe_tok), Pattern::OrPattern {
                left,
                right,
            });
        }
        Ok(left)
    }

    fn parse_primary_pattern(&mut self) -> ParseResult<PatternRef> {
        if self.check(TokenKind::Identifier) && self.peek().lexeme == "_" {
            let tok = self.advance();
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Wildcard));
        }
        if self.match_token(TokenKind::NullLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Literal(PatternLiteral::Null)));
        }
        if self.match_token(TokenKind::TrueLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Literal(PatternLiteral::Bool(true))));
        }
        if self.match_token(TokenKind::FalseLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Literal(PatternLiteral::Bool(false))));
        }
        if self.match_token(TokenKind::IntLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Literal(PatternLiteral::Int(tok.lexeme))));
        }
        if self.match_token(TokenKind::FloatLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Literal(PatternLiteral::Float(tok.lexeme))));
        }
        if self.match_token(TokenKind::CharLiteral) {
            let tok = self.previous();
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Literal(PatternLiteral::Char(parse_char_value(tok.lexeme)))));
        }
        if self.match_token(TokenKind::StringLiteral) {
            let tok = self.previous();
            let value = &tok.lexeme[1..tok.lexeme.len() - 1];
            return Ok(self.alloc_pattern(token_span(&tok), Pattern::Literal(PatternLiteral::String(value))));
        }
        if self.match_token(TokenKind::LParen) {
            return self.parse_record_pattern();
        }
        if self.check(TokenKind::KwVal) || self.check(TokenKind::KwVar) {
            let name_tok = self.advance();
            if self.check(TokenKind::LParen) {
                return self.parse_constructor_pattern(name_tok);
            }
            return Ok(self.alloc_pattern(token_span(&name_tok), Pattern::Variable {
                name: name_tok.lexeme,
            }));
        }
        if self.check(TokenKind::Identifier) {
            let name_tok = self.advance();
            if self.check(TokenKind::LParen) {
                return self.parse_constructor_pattern(name_tok);
            }
            return Ok(self.alloc_pattern(token_span(&name_tok), Pattern::Variable {
                name: name_tok.lexeme,
            }));
        }
        self.report_error("expected pattern")?;
        unreachable!()
    }

    /// Parse a constructor pattern
    fn parse_constructor_pattern(&mut self, name_tok: Token<'a>) -> ParseResult<PatternRef> {
        self.advance(); // '('
        let mut patterns = Vec::new();
        if !self.check(TokenKind::RParen) {
            patterns.push(self.parse_pattern()?);
            while self.match_token(TokenKind::Comma) {
                if self.check(TokenKind::RParen) {
                    break;
                }
                patterns.push(self.parse_pattern()?);
            }
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        Ok(self.alloc_pattern(token_span(&name_tok), Pattern::Constructor {
            name: name_tok.lexeme,
            patterns,
        }))
    }

    /// Parse a record pattern
    fn parse_record_pattern(&mut self) -> ParseResult<PatternRef> {
        let lparen = self.previous();
        let span = token_span(&lparen);
        let mut fields = Vec::new();
        if !self.check(TokenKind::RParen) {
            if self.peek().kind == TokenKind::Identifier {
                let saved = self.current;
                let name_tok = self.advance();
                if self.check(TokenKind::Colon) {
                    // Named-field pattern
                    self.advance();
                    let pattern = self.parse_pattern()?;
                    fields.push(PatternRecordField {
                        name: name_tok.lexeme,
                        pattern,
                    });
                    while self.match_token(TokenKind::Comma) {
                        if self.check(TokenKind::RParen) {
                            break;
                        }
                        let field_name = self.expect(TokenKind::Identifier, "expected field name")?;
                        let _ = self.expect(TokenKind::Colon, "expected ':'");
                        let field_pattern = self.parse_pattern()?;
                        fields.push(PatternRecordField {
                            name: field_name.lexeme,
                            pattern: field_pattern,
                        });
                    }
                    let _ = self.expect(TokenKind::RParen, "expected ')'");
                    return Ok(self.alloc_pattern(span, Pattern::Record { fields }));
                }
                self.current = saved;
            }
            // Positional pattern
            let first = self.parse_pattern()?;
            fields.push(PatternRecordField {
                name: int_to_key(self.arena, 0),
                pattern: first,
            });
            let mut idx: usize = 1;
            while self.match_token(TokenKind::Comma) {
                if self.check(TokenKind::RParen) {
                    break;
                }
                let p = self.parse_pattern()?;
                fields.push(PatternRecordField {
                    name: int_to_key(self.arena, idx),
                    pattern: p,
                });
                idx += 1;
            }
        }
        let _ = self.expect(TokenKind::RParen, "expected ')'");
        Ok(self.alloc_pattern(span, Pattern::Record { fields }))
    }
}

// =========================================================================
// Helper functions
// =========================================================================

fn token_span(tok: &Token<'_>) -> Span {
    Span::new(tok.line, tok.column)
}

fn parse_char_value(lexeme: &str) -> u32 {
    if lexeme.len() < 3 {
        return 0;
    }
    let content = &lexeme[1..lexeme.len() - 1];
    if content.is_empty() {
        return 0;
    }
    let bytes = content.as_bytes();
    if bytes[0] == b'\\' {
        if content.len() < 2 {
            return 0;
        }
        return match bytes[1] {
            b'n' => b'\n' as u32,
            b't' => b'\t' as u32,
            b'r' => b'\r' as u32,
            b'\\' => b'\\' as u32,
            b'\'' => b'\'' as u32,
            b'0' => 0,
            b'u' => {
                // Bug #36: \uXXXX (4-digit hex) or \u{XXXX} (brace form).
                // Lexical scanning (scan_char) has already validated the hex digits;
                // use expect here to assert the invariant.
                if content.len() > 2 && bytes[2] == b'{' {
                    // \u{XXXX} brace form
                    let close = content.find('}').expect("scan_char validated closing brace");
                    if close > 3 {
                        let hex_str = &content[3..close];
                        u32::from_str_radix(hex_str, 16).expect("scan_char validated hex digits")
                    } else {
                        unreachable!("scan_char validated non-empty hex in braces")
                    }
                } else if content.len() >= 6 {
                    // \uXXXX without braces (\u + 4 hex digits)
                    let hex_str = &content[2..6];
                    u32::from_str_radix(hex_str, 16).expect("scan_char validated hex digits")
                } else {
                    unreachable!("scan_char validated 4 hex digits without braces")
                }
            }
            b'x' => {
                // \xHH: exactly 2 hex digits, byte value 0x00-0xFF.
                // Lexical scanning (scan_char) has already validated the hex digits.
                if content.len() >= 4 {
                    let hex_str = &content[2..4];
                    u32::from_str_radix(hex_str, 16).expect("scan_char validated hex digits")
                } else {
                    unreachable!("scan_char validated 2 hex digits")
                }
            }
            _ => bytes[1] as u32,
        };
    }
    // Non-ASCII character: decode the full UTF-8 sequence to a Unicode code point
    // bytes[0] as u32 would only take the first byte (incorrect for multi-byte characters)
    content.chars().next().map(|c| c as u32).unwrap_or(0)
}

fn contains_interpolation(raw: &str) -> bool {
    if raw.len() < 2 {
        return false;
    }
    let bytes = raw.as_bytes();
    let mut i = 1;
    while i < raw.len() - 1 {
        if bytes[i] == b'\\' {
            // Bug #36: \uXXXX and \u{XXXX} escape sequences must skip the entire sequence
            if i + 1 < raw.len() && bytes[i + 1] == b'u' {
                i += 2; // skip \u
                if i < raw.len() && bytes[i] == b'{' {
                    // \u{XXXX} brace form: skip to }
                    while i < raw.len() && bytes[i] != b'}' {
                        i += 1;
                    }
                    if i < raw.len() {
                        i += 1; // skip }
                    }
                } else {
                    // \uXXXX without braces: skip 4 hex digits
                    i += 4;
                }
                continue;
            }
            // \xHH: skip the entire 4-character sequence to avoid mistaking
            // hex-encoded braces (e.g. \x7b = '{') for interpolation start
            if i + 1 < raw.len() && bytes[i + 1] == b'x' {
                i += 4; // skip \xHH
                continue;
            }
            i += 2;
            continue;
        }
        if bytes[i] == b'{' {
            if i + 1 < raw.len() - 1 && bytes[i + 1] == b'{' {
                i += 2;
                continue;
            }
            return true;
        }
        i += 1;
    }
    false
}

fn is_digit_or_underscore(ch: u8) -> bool {
    ch.is_ascii_digit() || ch == b'_'
}

fn is_hex_or_underscore(ch: u8) -> bool {
    is_digit_or_underscore(ch) || (b'a'..=b'f').contains(&ch) || (b'A'..=b'F').contains(&ch)
}

/// Forward-scans a float literal, separating the numeric part from the type suffix.
/// Correctly handles decimal (`.` and `e`/`E` exponents) and hexadecimal (`0x` prefix, `p`/`P` exponents).
/// The old backward scan would misidentify the `e300` in an unsuffixed scientific-notation `1e300` as a type suffix (Bug #20).
fn split_float_suffix(raw: &str) -> (&str, Option<&str>) {
    let bytes = raw.as_bytes();
    let mut i: usize = 0;
    if bytes.len() > 2 && bytes[0] == b'0' && (bytes[1] == b'x' || bytes[1] == b'X') {
        // Hexadecimal float: 0x<hex>.<hex>p<exp>
        i = 2;
        while i < bytes.len() && is_hex_or_underscore(bytes[i]) {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && is_hex_or_underscore(bytes[i]) {
                i += 1;
            }
        }
        if i < bytes.len() && (bytes[i] == b'p' || bytes[i] == b'P') {
            i += 1;
            if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                i += 1;
            }
            while i < bytes.len() && is_digit_or_underscore(bytes[i]) {
                i += 1;
            }
        }
    } else {
        // Decimal float: [int].[frac]e[exp] or .[frac]e[exp]
        while i < bytes.len() && is_digit_or_underscore(bytes[i]) {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && is_digit_or_underscore(bytes[i]) {
                i += 1;
            }
        }
        if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
            i += 1;
            if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                i += 1;
            }
            while i < bytes.len() && is_digit_or_underscore(bytes[i]) {
                i += 1;
            }
        }
    }
    if i < bytes.len() {
        (&raw[..i], Some(&raw[i..]))
    } else {
        (raw, None)
    }
}

fn int_to_key(arena: &Bump, idx: usize) -> &str {
    let s = idx.to_string();
    arena.alloc_str(&s)
}

fn parse_u64(s: &str) -> Option<u64> {
    s.parse::<u64>().ok()
}
