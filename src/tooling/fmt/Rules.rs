//! Format rules: spacing, indentation, line breaks.

use crate::ast::Parser::TokenKind;

/// Action: what the formatter should emit between two tokens.
#[derive(Debug, Clone, Copy)]
pub enum Action {
    Space,
    NoSpace,
    Newline,
    Newlines(u32),
    KeepOriginal,
}

/// Decide the action between two tokens.
pub fn decide_spacing(prev: TokenKind, next: TokenKind) -> Action {
    // Phase 1 basic rules
    use TokenKind::*;
    match (prev, next) {
        // No space before closing delimiters
        (_, RParen) | (_, RBrace) | (_, RBracket) => Action::NoSpace,
        // No space after opening delimiters
        (LParen, _) | (LBrace, _) | (LBracket, _) => Action::NoSpace,
        // Space after comma
        (Comma, _) => Action::Space,
        // No space before comma
        (_, Comma) => Action::NoSpace,
        // Default: space between tokens
        _ => Action::Space,
    }
}
