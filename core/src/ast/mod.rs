#![allow(non_snake_case)]
//! ast — Abstract syntax tree and parsing modules.
//!
//! Aggregates the two AST-related submodules:
//! - [`Ast`]: AST data model (node enums, [`AstArena`], [`AstVisitor`], [`Printer`]).
//! - [`Parser`]: Lexical and syntactic analysis ([`Lexer`], [`Parser`],
//!   [`BINARY_OPS`] precedence table).
//!
//! [`AstArena`]: crate::ast::Ast::AstArena
//! [`AstVisitor`]: crate::ast::Ast::AstVisitor
//! [`Printer`]: crate::ast::Ast::Printer
//! [`Lexer`]: crate::ast::Parser::Lexer
//! [`Parser`]: crate::ast::Parser::Parser
//! [`BINARY_OPS`]: crate::ast::Parser::BINARY_OPS

pub mod Ast;
pub mod Parser;
