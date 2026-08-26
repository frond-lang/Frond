//! Per-document state: text, parse cache, sema snapshot.

use bumpalo::Bump;

use crate::pass::Analyzer;
use crate::tooling::Common::Diagnostic::Diagnostic;
use crate::tooling::Common::Pipeline;
use crate::tooling::Lint::RuleRegistry;

/// Per-document state.
pub struct DocState {
    pub version: i32,
    pub text: String,
    pub content_hash: u64,
    pub diagnostics: Vec<Diagnostic>,
    pub symbols: Vec<SymbolInfo>,
    /// Previous declaration signatures — for is_api_change detection on next didChange.
    pub prev_decls: Option<Vec<DeclSignature>>,
    /// Cached module key (e.g. "src/Foo.frond") for incremental sema loader lookup.
    pub module_key: Option<String>,
}

/// Symbol info for documentSymbol and completion.
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: SymbolKind,
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum SymbolKind {
    Function,
    Type,
    Constant,
    Variable,
}

/// Declaration signature (no function body) — for is_api_change comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum DeclSignature {
    Fun {
        name: String,
        params: Vec<String>,
        return_type: Option<String>,
        type_params: Vec<String>,
    },
    Type {
        name: String,
        type_params: Vec<String>,
        /// Constructor/field names (order-sensitive)
        members: Vec<String>,
    },
    Trait {
        name: String,
        type_params: Vec<String>,
        /// Method names
        methods: Vec<String>,
    },
    Import {
        module_path: Vec<String>,
        items: Option<Vec<String>>,
    },
    Other,
}

impl DocState {
    pub fn new(text: String, version: i32) -> Self {
        let content_hash = hash_string(&text);
        Self {
            version,
            text,
            content_hash,
            diagnostics: Vec::new(),
            symbols: Vec::new(),
            prev_decls: None,
            module_key: None,
        }
    }

    pub fn update(&mut self, text: String, version: i32) {
        let new_hash = hash_string(&text);
        self.text = text;
        self.version = version;
        self.content_hash = new_hash;
    }

    /// Recheck: parse → sema → analyze → lint → update diagnostics.
    /// Phase 1: full recheck on every call (no incremental).
    pub fn recheck(&mut self, state: &super::Server::ServerState) -> Vec<Diagnostic> {
        let arena = Bump::new();
        let filename = "lsp_buffer.frond";

        // Parse (LSP-safe, never exits)
        let parse_result = Pipeline::parse_entry_module_lsp(&arena, &self.text, filename);
        let mut all_diag = parse_result.diagnostics;

        // Load + sema (LSP-safe)
        let (loader, std_keys, dep_keys) = Pipeline::load_all_modules_or_exit(
            &parse_result.module, filename
        );

        match Pipeline::run_sema_pipeline_lsp(
            &loader, &std_keys, &dep_keys, &parse_result.module, filename
        ) {
            Pipeline::SemaOutcome::Ok { sema_result, diagnostics: sema_diag, .. } => {
                all_diag.extend(sema_diag);

                // Analyze + lint
                let analysis = Analyzer::analyze(
                    &parse_result.module,
                    &parse_result.module.arena,
                    &sema_result
                );

                let registry = RuleRegistry::new();
                let lint_diag = registry.run_all(
                    &parse_result.module,
                    &parse_result.module.arena,
                    &sema_result,
                    Some(&analysis),
                    &state.lint_config
                );
                all_diag.extend(lint_diag);

                // Extract symbols
                self.symbols = extract_symbols(&parse_result.module);
            }
            Pipeline::SemaOutcome::Err(sema_diag) => {
                all_diag.extend(sema_diag);
            }
        }

        self.diagnostics = all_diag.clone();
        all_diag
    }
}

/// Simple string hash (FNV-1a).
fn hash_string(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Extract top-level symbols from a module.
pub fn extract_symbols(module: &crate::ast::Ast::Module) -> Vec<SymbolInfo> {
    use crate::ast::Ast::Decl;

    module.declarations.iter().filter_map(|decl| {
        match &decl.node {
            Decl::FunDecl { name, .. } => Some(SymbolInfo {
                name: name.to_string(),
                kind: SymbolKind::Function,
                line: decl.span.line,
                col: decl.span.column,
            }),
            Decl::TypeDecl { name, .. } | Decl::TraitDecl { name, .. } => Some(SymbolInfo {
                name: name.to_string(),
                kind: SymbolKind::Type,
                line: decl.span.line,
                col: decl.span.column,
            }),
            _ => None,
        }
    }).collect()
}

/// Extract declaration signatures from a module (ignores function bodies).
/// Used for is_api_change comparison.
pub fn extract_decl_signatures(module: &crate::ast::Ast::Module) -> Vec<DeclSignature> {
    use crate::ast::Ast::{Decl, TypeDef};

    module.declarations.iter().map(|decl| {
        match &decl.node {
            Decl::FunDecl { name, params, return_type, type_params, .. } => {
                DeclSignature::Fun {
                    name: name.to_string(),
                    params: params.iter().map(|p| format!("{:?}", p)).collect(),
                    return_type: return_type.map(|t| format!("{:?}", t)),
                    type_params: type_params.iter().map(|t| t.name.to_string()).collect(),
                }
            }
            Decl::TypeDecl { name, type_params, def, .. } => {
                let members = match def {
                    TypeDef::Record { fields, .. } => {
                        fields.iter().map(|f| f.name.to_string()).collect()
                    }
                    TypeDef::Adt { constructors } => {
                        constructors.iter().map(|c| c.name.to_string()).collect()
                    }
                    TypeDef::Alias { .. } => vec![],
                    TypeDef::Newtype { .. } => vec![],
                };
                DeclSignature::Type {
                    name: name.to_string(),
                    type_params: type_params.iter().map(|t| t.name.to_string()).collect(),
                    members,
                }
            }
            Decl::TraitDecl { name, type_params, methods, .. } => {
                DeclSignature::Trait {
                    name: name.to_string(),
                    type_params: type_params.iter().map(|t| t.name.to_string()).collect(),
                    methods: methods.iter().map(|m| m.name.to_string()).collect(),
                }
            }
            Decl::ImportDecl { module_path, items, .. } => {
                DeclSignature::Import {
                    module_path: module_path.iter().map(|s| s.to_string()).collect(),
                    items: items.as_ref().map(|is| {
                        is.iter().map(|i| i.name.to_string()).collect()
                    }),
                }
            }
            _ => DeclSignature::Other,
        }
    }).collect()
}

/// Check if two sets of declarations differ in API (affects other modules).
pub fn is_api_change(old: &[DeclSignature], new: &[DeclSignature]) -> bool {
    old != new
}
