//! Comment handling: attachment and blank line preservation.

use super::Engine::Trivia;

/// Maximum blank lines to preserve (default 2).
pub const MAX_BLANK_LINES: u32 = 2;

/// Count blank lines in a trivia list.
pub fn count_blank_lines(trivia: &[Trivia]) -> u32 {
    trivia
        .iter()
        .map(|t| match t {
            Trivia::BlankLines(n) => *n,
            _ => 0,
        })
        .sum()
}
