//! debug subcommand — diagnostic mode (tokens/ast/check/emit-c/full).

use std::process;

use crate::ast::Ast::Printer;
use crate::ast::Parser::{ErrorCollector, Lexer, Parser, Token, TokenCollector};
use crate::tooling::Common::Pipeline;

use super::Args::DebugStage;
use super::Manifest::resolve_entry_path;
use super::Pipeline::{read_source, run_from_project};

pub fn cmd_debug(file: Option<String>, stage: Option<DebugStage>) {
    let stage = stage.unwrap_or(DebugStage::Full);
    let entry_path = resolve_entry_path(file);
    let source = read_source(&entry_path);

    match stage {
        DebugStage::Tokens => debug_tokens(&source),
        DebugStage::Ast => debug_ast(&source),
        DebugStage::Sema => super::Dump::dump_sema(&entry_path),
        DebugStage::Load => super::Dump::dump_load(&entry_path),
        DebugStage::TyOps => super::Dump::dump_tyops(),
        DebugStage::EmitC => debug_emit_c(&source),
        DebugStage::Check => debug_check(&source, &entry_path),
        DebugStage::Full => run_from_project(crate::pass::Optimizer::OptLevel::default(), true),
    }
}

/// Lexical analysis only; print the token list.
fn debug_tokens(source: &str) {
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens = sink.into_tokens();
    for tok in &tokens {
        println!(
            "{:>4}:{:<3} {:<20} {}",
            tok.line,
            tok.column,
            format!("{:?}", tok.kind),
            tok.lexeme
        );
    }
}

/// Parse and print the AST (S-expressions).
fn debug_ast(source: &str) {
    let arena = bumpalo::Bump::new();
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = Parser::new(tokens_ref, &arena, ErrorCollector::new());

    match parser.parse_module("stdin") {
        Ok(module) => {
            let mut printer = Printer::new(&module.arena);
            let output = printer.print_module(&module);
            print!("{}", output);
        }
        Err(err) => {
            eprintln!("Parse error at {}:{}: {}", err.line, err.column, err.message);
            process::exit(1);
        }
    }
    for err in parser.errors() {
        eprintln!("Warning: parse error at {}:{}: {}", err.line, err.column, err.message);
    }
}

/// Extract @extern("C") functions and emit .c to stdout.
fn debug_emit_c(source: &str) {
    let arena = bumpalo::Bump::new();
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = Parser::new(tokens_ref, &arena, ErrorCollector::new());

    match parser.parse_module("stdin") {
        Ok(module) => {
            if !parser.errors().is_empty() {
                for err in parser.errors() {
                    eprintln!("Error: parse error at {}:{}: {}", err.line, err.column, err.message);
                }
                process::exit(1);
            }
            match crate::ffi::ExternC::extract_c_from_module(&module) {
                Ok(c_code) => print!("{}", c_code),
                Err(e) => {
                    eprintln!("Error extracting C: {}", e);
                    process::exit(1);
                }
            }
        }
        Err(err) => {
            eprintln!("Parse error at {}:{}: {}", err.line, err.column, err.message);
            process::exit(1);
        }
    }
}

/// Type check only.
fn debug_check(source: &str, filename: &str) {
    let arena = bumpalo::Bump::new();
    let entry_module = Pipeline::parse_entry_module_or_exit(&arena, source, filename);
    let (loader, std_keys, dep_keys) = Pipeline::load_all_modules_or_exit(&entry_module, filename);
    let (_type_arena, _sema_result) =
        Pipeline::run_sema_pipeline_or_exit(&loader, &std_keys, &dep_keys, &entry_module, filename);
    println!("ok: {} (no type errors)", filename);
}
