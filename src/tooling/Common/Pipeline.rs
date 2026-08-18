//! Pipeline: shared parse/load/sema entry points for CLI and tooling.
//!
//! These functions wrap the parse → module-load → sema stages of the compiler
//! pipeline. Each variant exits the process on failure (with diagnostics
//! printed to stderr), matching the behavior previously inlined in `main.rs`.

use std::process;

use bumpalo::Bump;
use rustc_hash::FxHashSet;

use crate::ast::Ast::Module;
use crate::ast::Parser::{ErrorCollector, Lexer, Parser, Token, TokenCollector};
use crate::module::{Error::LoadError, ModuleLoader};
use crate::sema::Inference::InferContext;
use crate::sema::Sema::{populate_module, SemaResult, TypeArena};

/// Parses the entry module. The arena must remain valid for the lifetime of the returned `Module`.
pub fn parse_entry_module_or_exit<'a>(
    arena: &'a Bump,
    source: &'a str,
    filename: &'a str,
) -> Module<'a> {
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = Parser::new(tokens_ref, arena, ErrorCollector::new());

    let module = match parser.parse_module(filename) {
        Ok(m) => m,
        Err(err) => {
            eprintln!("{}:{}:{}: parse error: {}", filename, err.line, err.column, err.message);
            process::exit(1);
        }
    };
    for err in parser.errors() {
        eprintln!("Warning: {}:{}:{}: {}", filename, err.line, err.column, err.message);
    }
    module
}

/// Loads all modules (builtin + std + user dependencies), returning (loader, std_keys, dep_keys).
/// The directory containing the entry file is added as a search path to resolve user modules (e.g. Math/Geometry.frond).
pub fn load_all_modules_or_exit(
    entry_module: &Module,
    entry_path: &str,
) -> (ModuleLoader, Vec<String>, Vec<String>) {
    let mut loader = ModuleLoader::new();
    if let Some(src_dir) = std::path::Path::new(entry_path).parent() {
        loader.add_search_path(src_dir);
    }
    let dep_keys = loader.load_transitive_imports(entry_module);

    let std_keys: Vec<String> = crate::module::STD_FILES
        .iter()
        .map(|(p, _)| p.to_string())
        .collect();
    for key in &std_keys {
        let parts: Vec<&str> = key.strip_suffix(".frond").unwrap().split('/').collect();
        let _ = loader.resolve_and_load(&parts);
    }

    if loader.has_load_errors() {
        for err in loader.load_errors() {
            match err {
                LoadError::ModuleNotFound { path } => {
                    eprintln!("error: module not found: {}", path);
                }
                LoadError::ParseFailed { path, line, column, message } => {
                    eprintln!("error: parse failed in {} at {}:{}: {}", path, line, column, message);
                }
                LoadError::CircularImport { path } => {
                    eprintln!("error: circular import detected: {}", path);
                }
            }
        }
        process::exit(1);
    }
    (loader, std_keys, dep_keys)
}

/// Runs the full Sema pipeline: register builtin types → predeclare all modules → check each module.
/// Any type error in any module is printed and exits with exit(1). Returns (type_arena, sema_result) on success.
pub fn run_sema_pipeline_or_exit(
    loader: &ModuleLoader,
    std_keys: &[String],
    dep_keys: &[String],
    entry_module: &Module,
    entry_filename: &str,
) -> (TypeArena, SemaResult) {
    let mut type_arena = TypeArena::new();
    let mut sema_result = SemaResult::new();
    let mut ctx = InferContext::new(&mut type_arena, &mut sema_result);

    ctx.reset_state();
    let root_env = ctx.env.root();
    ctx.register_builtins(root_env);

    let module_logical_paths: Vec<String> = loader
        .loaded_keys()
        .iter()
        .filter_map(|k| k.strip_suffix(".frond").map(|s| s.replace('/', ".")))
        .collect();
    ctx.register_module_aliases(root_env, &module_logical_paths);

    // predeclare: register all module functions and type constructors into root_env first,
    // to resolve cross-module forward references. check_module_with_env will predeclare again internally (idempotent).
    for (_, m) in loader.builtin_modules() {
        ctx.predeclare_declarations(m, root_env);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            ctx.predeclare_declarations(m, root_env);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            ctx.predeclare_declarations(m, root_env);
        }
    }

    let mut prev_err_len = 0usize;
    let mut prev_warn_len = 0usize;

    // populate: fill in the definition tables (type method signatures, etc.) for all modules before checking,
    // to resolve cross-module method lookup failures caused by module check ordering.
    // check_module_with_env will call it again internally (idempotent; put_type_def rejects duplicates).
    for (_, m) in loader.builtin_modules() {
        populate_module(ctx.arena, ctx.sema_result, m);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            populate_module(ctx.arena, ctx.sema_result, m);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            populate_module(ctx.arena, ctx.sema_result, m);
        }
    }
    populate_module(ctx.arena, ctx.sema_result, entry_module);

    // Build the all_modules list: used for cross-module monomorphization (generic calls need access to the callee module's arena).
    let mut all_modules: Vec<&Module> = Vec::new();
    for (_, m) in loader.builtin_modules() {
        all_modules.push(m);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            all_modules.push(m);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            all_modules.push(m);
        }
    }
    all_modules.push(entry_module);

    // check: builtin → std → dep → entry
    for (path, m) in loader.builtin_modules() {
        ctx.check_module_with_env(m, root_env, &all_modules);
        for err in &ctx.sema_result.errors[prev_err_len..] {
            eprintln!("{}:{}:{}: {}", path, err.line, err.column, err.message);
        }
        for warn in &ctx.sema_result.warnings[prev_warn_len..] {
            let wp = warn.file_path.as_deref().map(|s| s as &str).unwrap_or(path);
            eprintln!("{}:{}:{}: warning: {}", wp, warn.line, warn.column, warn.message);
        }
        prev_err_len = ctx.sema_result.errors.len();
        prev_warn_len = ctx.sema_result.warnings.len();
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            ctx.check_module_with_env(m, root_env, &all_modules);
            for err in &ctx.sema_result.errors[prev_err_len..] {
                eprintln!("{}:{}:{}: {}", key, err.line, err.column, err.message);
            }
            for warn in &ctx.sema_result.warnings[prev_warn_len..] {
                let wp = warn.file_path.as_deref().map(|s| s as &str).unwrap_or(key);
                eprintln!("{}:{}:{}: warning: {}", wp, warn.line, warn.column, warn.message);
            }
            prev_err_len = ctx.sema_result.errors.len();
            prev_warn_len = ctx.sema_result.warnings.len();
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            ctx.check_module_with_env(m, root_env, &all_modules);
            for err in &ctx.sema_result.errors[prev_err_len..] {
                eprintln!("{}:{}:{}: {}", k, err.line, err.column, err.message);
            }
            for warn in &ctx.sema_result.warnings[prev_warn_len..] {
                let wp = warn.file_path.as_deref().map(|s| s as &str).unwrap_or(k);
                eprintln!("{}:{}:{}: warning: {}", wp, warn.line, warn.column, warn.message);
            }
            prev_err_len = ctx.sema_result.errors.len();
            prev_warn_len = ctx.sema_result.warnings.len();
        }
    }
    ctx.check_module_with_env(entry_module, root_env, &all_modules);
    for err in &ctx.sema_result.errors[prev_err_len..] {
        eprintln!("{}:{}:{}: {}", entry_filename, err.line, err.column, err.message);
    }
    for warn in &ctx.sema_result.warnings[prev_warn_len..] {
        let wp = warn.file_path.as_deref().map(|s| s as &str).unwrap_or(entry_filename);
        eprintln!("{}:{}:{}: warning: {}", wp, warn.line, warn.column, warn.message);
    }

    if !ctx.sema_result.errors.is_empty() {
        process::exit(1);
    }
    // ctx borrows type_arena and sema_result; dropping it here returns ownership of both to the caller.
    drop(ctx);
    (type_arena, sema_result)
}

// ==================== LSP/lint variant (return error, never exit) ====================

use crate::ast::Ast::Span;
use crate::ast::Parser::ParseError;
use crate::tooling::Common::Diagnostic::{Diagnostic, Range, Severity, Category};

/// LSP parse result: always returns module (error-recovered) + diagnostics.
pub struct ParseResult<'a> {
    pub module: Module<'a>,
    pub diagnostics: Vec<Diagnostic>,
}

fn parse_error_to_diagnostic(err: &ParseError, filename: &str, severity: Severity) -> Diagnostic {
    Diagnostic {
        severity,
        code: "PARSE",
        category: Category::Correctness,
        message: err.message.clone(),
        range: Range::from_span(Span::new(err.line, err.column)),
        source_file: filename.to_string(),
        suggestion: None,
    }
}

/// LSP-safe parse: never exits, returns module + diagnostics.
/// On hard parse error, returns a minimal empty module + error diagnostic.
pub fn parse_entry_module_lsp<'a>(
    arena: &'a Bump,
    source: &'a str,
    filename: &'a str,
) -> ParseResult<'a> {
    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token<'_>> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);
    let mut parser = Parser::new(tokens_ref, arena, ErrorCollector::new());

    let mut diagnostics = Vec::new();

    let module = match parser.parse_module(filename) {
        Ok(m) => m,
        Err(err) => {
            diagnostics.push(parse_error_to_diagnostic(&err, filename, Severity::Error));
            // Return an empty module so callers can still proceed.
            Module {
                name: filename,
                source_path: Some(filename),
                arena: crate::ast::Ast::AstArena::new(),
                declarations: Vec::new(),
            }
        }
    };

    // Collect parser warnings as Warning diagnostics
    for err in parser.errors() {
        diagnostics.push(parse_error_to_diagnostic(err, filename, Severity::Warning));
    }

    ParseResult { module, diagnostics }
}

/// LSP sema result.
pub enum SemaOutcome {
    Ok {
        type_arena: TypeArena,
        sema_result: SemaResult,
        diagnostics: Vec<Diagnostic>,
    },
    Err(Vec<Diagnostic>),
}

fn sema_errors_to_diagnostics(
    errors: &[crate::sema::Sema::SemaError],
    filename: &str,
    severity: Severity,
) -> Vec<Diagnostic> {
    errors
        .iter()
        .map(|err| Diagnostic {
            severity,
            code: "SEMA",
            category: Category::Correctness,
            message: err.message.to_string(),
            range: Range::from_span(Span::new(err.line, err.column)),
            source_file: filename.to_string(),
            suggestion: None,
        })
        .collect()
}

/// LSP-safe sema: never exits, returns SemaOutcome with diagnostics.
/// Collects all errors (does not stop on first error).
/// The body is identical to run_sema_pipeline_or_exit, but:
/// 1. Does NOT call process::exit on error
/// 2. Collects errors as Diagnostic instead of printing to stderr
/// 3. Always returns SemaOutcome::Ok with partial results (even on error)
pub fn run_sema_pipeline_lsp(
    loader: &ModuleLoader,
    std_keys: &[String],
    dep_keys: &[String],
    entry_module: &Module,
    entry_filename: &str,
) -> SemaOutcome {
    let mut type_arena = TypeArena::new();
    let mut sema_result = SemaResult::new();
    let mut ctx = InferContext::new(&mut type_arena, &mut sema_result);

    ctx.reset_state();
    let root_env = ctx.env.root();
    ctx.register_builtins(root_env);

    let module_logical_paths: Vec<String> = loader
        .loaded_keys()
        .iter()
        .filter_map(|k| k.strip_suffix(".frond").map(|s| s.replace('/', ".")))
        .collect();
    ctx.register_module_aliases(root_env, &module_logical_paths);

    for (_, m) in loader.builtin_modules() {
        ctx.predeclare_declarations(m, root_env);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            ctx.predeclare_declarations(m, root_env);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            ctx.predeclare_declarations(m, root_env);
        }
    }

    for (_, m) in loader.builtin_modules() {
        populate_module(ctx.arena, ctx.sema_result, m);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            populate_module(ctx.arena, ctx.sema_result, m);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            populate_module(ctx.arena, ctx.sema_result, m);
        }
    }
    populate_module(ctx.arena, ctx.sema_result, entry_module);

    let mut all_modules: Vec<&Module> = Vec::new();
    for (_, m) in loader.builtin_modules() {
        all_modules.push(m);
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            all_modules.push(m);
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            all_modules.push(m);
        }
    }
    all_modules.push(entry_module);

    let mut diagnostics = Vec::new();

    for (path, m) in loader.builtin_modules() {
        let prev_err = ctx.sema_result.errors.len();
        let prev_warn = ctx.sema_result.warnings.len();
        ctx.check_module_with_env(m, root_env, &all_modules);
        diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.errors[prev_err..], path, Severity::Error));
        diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.warnings[prev_warn..], path, Severity::Warning));
    }
    for key in std_keys {
        if let Some(m) = loader.get_module_by_key(key) {
            let prev_err = ctx.sema_result.errors.len();
            let prev_warn = ctx.sema_result.warnings.len();
            ctx.check_module_with_env(m, root_env, &all_modules);
            diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.errors[prev_err..], key, Severity::Error));
            diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.warnings[prev_warn..], key, Severity::Warning));
        }
    }
    for k in dep_keys {
        if let Some(m) = loader.get_module_by_key(k) {
            let prev_err = ctx.sema_result.errors.len();
            let prev_warn = ctx.sema_result.warnings.len();
            ctx.check_module_with_env(m, root_env, &all_modules);
            diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.errors[prev_err..], k, Severity::Error));
            diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.warnings[prev_warn..], k, Severity::Warning));
        }
    }
    {
        let prev_err = ctx.sema_result.errors.len();
        let prev_warn = ctx.sema_result.warnings.len();
        ctx.check_module_with_env(entry_module, root_env, &all_modules);
        diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.errors[prev_err..], entry_filename, Severity::Error));
        diagnostics.extend(sema_errors_to_diagnostics(&ctx.sema_result.warnings[prev_warn..], entry_filename, Severity::Warning));
    }

    drop(ctx);
    SemaOutcome::Ok { type_arena, sema_result, diagnostics }
}

// ==================== Incremental sema variant ====================

/// Result of incremental sema recheck.
pub enum SemaIncrementalOutcome {
    /// Incremental recheck succeeded.
    Ok {
        type_arena: TypeArena,
        sema_result: SemaResult,
        diagnostics: Vec<Diagnostic>,
        /// Modules that were rechecked.
        rechecked: FxHashSet<String>,
    },
    /// Cannot increment — caller should fall back to full sema.
    NeedsFull,
    /// Sema errors (non-recoverable).
    Err(Vec<Diagnostic>),
}

/// Check if a module key refers to a builtin module.
fn is_builtin_module(key: &str) -> bool {
    use crate::module::StdlibEmbed::BUILTIN_FILES;
    BUILTIN_FILES.iter().any(|(p, _)| *p == key)
}

/// Check if incremental sema is possible for the given dirty closure.
fn can_increment(loader: &ModuleLoader, dirty: &FxHashSet<String>) -> bool {
    // 1. dirty includes builtin module — cannot increment
    if dirty.iter().any(|m| is_builtin_module(m)) {
        return false;
    }
    // 2. dirty closure exceeds 50% of workspace — not worth incremental
    let total = loader.loaded_keys().len();
    if total > 0 && dirty.len() > total / 2 {
        return false;
    }
    true
}

/// Topological sort of dirty modules by import dependency (deps first).
fn topological_sort(
    dirty: &FxHashSet<String>,
    loader: &ModuleLoader,
) -> Vec<String> {
    let mut result = Vec::new();
    let mut visited = FxHashSet::default();
    let mut temp = FxHashSet::default();

    fn visit(
        key: &str,
        dirty: &FxHashSet<String>,
        loader: &ModuleLoader,
        visited: &mut FxHashSet<String>,
        temp: &mut FxHashSet<String>,
        result: &mut Vec<String>,
    ) {
        if !dirty.contains(key) {
            return;
        }
        if visited.contains(key) {
            return;
        }
        if !temp.insert(key.to_string()) {
            return; // cycle — bail
        }

        if let Some(deps) = loader.get_forward_deps(key) {
            for dep in deps {
                visit(dep, dirty, loader, visited, temp, result);
            }
        }
        temp.remove(key);
        visited.insert(key.to_string());
        result.push(key.to_string());
    }

    for key in dirty {
        visit(key, dirty, loader, &mut visited, &mut temp, &mut result);
    }
    result
}

/// Incremental sema recheck: only recheck modules in the dirty closure.
///
/// Reuses sema products from clean modules (via `prev_sema_result`).
/// Falls back to full sema (`NeedsFull`) if increment is not possible.
///
/// Strategy (mirrors `run_sema_pipeline_lsp` but only checks dirty modules):
/// 1. Purge dirty modules' old sema products.
/// 2. Construct `InferContext` from existing arena/sema_result (preserves clean modules' products).
/// 3. Restore env: register builtins, module aliases, predeclare + populate ALL modules.
/// 4. Check only dirty modules in topological order (deps first).
pub fn run_sema_incremental(
    loader: &ModuleLoader,
    dirty_closure: &FxHashSet<String>,
    prev_type_arena: &mut TypeArena,
    prev_sema_result: &mut SemaResult,
) -> SemaIncrementalOutcome {
    // 1. Check incrementability
    if !can_increment(loader, dirty_closure) {
        return SemaIncrementalOutcome::NeedsFull;
    }

    // 2. Purge dirty modules' old sema products
    for m in dirty_closure {
        prev_sema_result.purge_module(m);
    }

    // 3. Construct InferContext from existing state (preserves clean modules' sema products)
    let mut ctx = InferContext::from_existing(prev_type_arena, prev_sema_result);
    ctx.reset_state();
    let root_env = ctx.env.root();
    ctx.register_builtins(root_env);

    // Register module aliases for all loaded modules
    let module_logical_paths: Vec<String> = loader
        .loaded_keys()
        .iter()
        .filter_map(|k| k.strip_suffix(".frond").map(|s| s.replace('/', ".")))
        .collect();
    ctx.register_module_aliases(root_env, &module_logical_paths);

    // 4. Predeclare ALL modules into root_env (builtins first for name priority).
    // from_existing creates a fresh env, so all symbol bindings must be restored.
    for (_, m) in loader.builtin_modules() {
        ctx.predeclare_declarations(m, root_env);
    }
    let all_keys = loader.loaded_keys();
    for key in &all_keys {
        if !is_builtin_module(key) {
            if let Some(m) = loader.get_module_by_key(key) {
                ctx.predeclare_declarations(m, root_env);
            }
        }
    }

    // 5. Populate ALL modules (idempotent for clean modules — type_defs already preserved).
    for (_, m) in loader.builtin_modules() {
        populate_module(ctx.arena, ctx.sema_result, m);
    }
    for key in &all_keys {
        if !is_builtin_module(key) {
            if let Some(m) = loader.get_module_by_key(key) {
                populate_module(ctx.arena, ctx.sema_result, m);
            }
        }
    }

    // 6. Build all_modules list (ALL loaded modules, for cross-module monomorphization).
    let mut all_modules: Vec<&Module> = Vec::new();
    for (_, m) in loader.builtin_modules() {
        all_modules.push(m);
    }
    for key in &all_keys {
        if !is_builtin_module(key) {
            if let Some(m) = loader.get_module_by_key(key) {
                all_modules.push(m);
            }
        }
    }

    // 7. Check dirty modules in topological order (deps first).
    let ordered = topological_sort(dirty_closure, loader);
    let mut diagnostics = Vec::new();
    for key in &ordered {
        if let Some(m) = loader.get_module_by_key(key) {
            let prev_err = ctx.sema_result.errors.len();
            let prev_warn = ctx.sema_result.warnings.len();
            ctx.check_module_with_env(m, root_env, &all_modules);
            diagnostics.extend(sema_errors_to_diagnostics(
                &ctx.sema_result.errors[prev_err..],
                key,
                Severity::Error,
            ));
            diagnostics.extend(sema_errors_to_diagnostics(
                &ctx.sema_result.warnings[prev_warn..],
                key,
                Severity::Warning,
            ));
        }
    }

    // 8. Release borrows and return ownership to caller.
    drop(ctx);
    SemaIncrementalOutcome::Ok {
        type_arena: std::mem::take(prev_type_arena),
        sema_result: std::mem::take(prev_sema_result),
        diagnostics,
        rechecked: dirty_closure.clone(),
    }
}
