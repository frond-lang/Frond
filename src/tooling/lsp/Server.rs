//! LSP server: JSON-RPC transport + message dispatch.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Write, Stdin, Stdout};
use std::path::PathBuf;

use bumpalo::Bump;
use rustc_hash::FxHashSet;

use crate::module::ModuleLoader;
use crate::pass::Analyzer;
use crate::sema::Sema::{SemaResult, TypeArena};
use crate::tooling::Common::Diagnostic::Diagnostic;
use crate::tooling::Common::Pipeline;
use crate::tooling::Fmt::Engine::FmtConfig;
use crate::tooling::Lint::{LintConfig, RuleRegistry};

use super::DocState::{
    extract_decl_signatures, extract_symbols, is_api_change, DocState, SymbolKind,
};
use super::Index::WorkspaceIndex;

/// JSON-RPC transport over stdio: reads/writes Content-Length framed messages.
pub struct LspTransport<R: BufRead, W: Write> {
    reader: R,
    writer: W,
}

impl LspTransport<BufReader<Stdin>, Stdout> {
    /// Create a transport over stdin/stdout.
    pub fn stdio() -> Self {
        Self {
            reader: BufReader::new(io::stdin()),
            writer: io::stdout(),
        }
    }
}

impl<R: BufRead, W: Write> LspTransport<R, W> {
    /// Create a transport over arbitrary reader/writer (for testing).
    pub fn new(reader: R, writer: W) -> Self {
        Self { reader, writer }
    }

    /// Read one JSON-RPC message (blocking).
    pub fn read_message(&mut self) -> io::Result<Option<serde_json::Value>> {
        let mut content_length = None;

        // Read headers
        loop {
            let mut line = String::new();
            if self.reader.read_line(&mut line)? == 0 {
                return Ok(None); // EOF
            }
            let line = line.trim_end_matches(['\r', '\n']);
            if line.is_empty() {
                break; // End of headers
            }
            if let Some(v) = line.strip_prefix("Content-Length: ") {
                content_length = Some(v.parse::<usize>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "bad Content-Length")
                })?);
            }
        }

        let len = content_length.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "no Content-Length header")
        })?;

        // Read body
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        Ok(Some(serde_json::from_slice(&buf)?))
    }

    /// Write one JSON-RPC message.
    pub fn write_message(&mut self, msg: &serde_json::Value) -> io::Result<()> {
        let body = serde_json::to_vec(msg)?;
        write!(self.writer, "Content-Length: {}\r\n\r\n", body.len())?;
        self.writer.write_all(&body)?;
        self.writer.flush()
    }
}

/// LSP server state.
pub struct ServerState {
    pub root: Option<PathBuf>,
    pub docs: HashMap<String, DocState>,
    pub lint_config: LintConfig,
    pub fmt_config: FmtConfig,
    pub shutdown: bool,
    /// Cached module loader for incremental sema (built on first didChange).
    pub loader: Option<ModuleLoader>,
    /// Cached sema products (type_arena, sema_result) for incremental reuse.
    pub sema_cache: Option<(TypeArena, SemaResult)>,
    /// Workspace symbol index for cross-file navigation.
    pub index: Option<WorkspaceIndex>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            root: None,
            docs: HashMap::new(),
            lint_config: LintConfig::default(),
            fmt_config: FmtConfig::default(),
            shutdown: false,
            loader: None,
            sema_cache: None,
            index: None,
        }
    }
}

/// The LSP server: transport + state + dispatch loop.
pub struct LspServer {
    transport: LspTransport<std::io::BufReader<std::io::Stdin>, std::io::Stdout>,
    state: ServerState,
}

impl LspServer {
    pub fn new() -> Self {
        Self {
            transport: LspTransport::stdio(),
            state: ServerState::default(),
        }
    }

    /// Main loop: read messages, dispatch, write responses.
    pub fn run(mut self) -> ! {
        while let Some(msg) = self.transport.read_message().unwrap_or(None) {
            let method = msg["method"].as_str().unwrap_or("");
            let id = msg.get("id").cloned();
            let params = msg.get("params").cloned().unwrap_or(serde_json::Value::Null);

            match method {
                "initialize" => self.handle_initialize(id, &params),
                "initialized" => {} // no-op
                "shutdown" => {
                    self.state.shutdown = true;
                    self.respond(id, serde_json::json!(null));
                }
                "exit" => {
                    std::process::exit(if self.state.shutdown { 0 } else { 1 });
                }
                _ if self.state.shutdown => {} // ignore after shutdown
                "textDocument/didOpen" => self.handle_did_open(&params),
                "textDocument/didChange" => self.handle_did_change(&params),
                "textDocument/didClose" => self.handle_did_close(&params),
                "textDocument/hover" => self.handle_hover(id, &params),
                "textDocument/completion" => self.handle_completion(id, &params),
                "textDocument/documentSymbol" => self.handle_doc_symbol(id, &params),
                "textDocument/definition" => self.handle_definition(id, &params),
                "textDocument/formatting" => self.handle_formatting(id, &params),
                "workspace/symbol" => self.handle_workspace_symbol(id, &params),
                _ => self.method_not_found(id, method),
            }
        }
        std::process::exit(0);
    }

    fn respond(&mut self, id: Option<serde_json::Value>, result: serde_json::Value) {
        if let Some(id) = id {
            let msg = serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result
            });
            let _ = self.transport.write_message(&msg);
        }
    }

    fn method_not_found(&mut self, id: Option<serde_json::Value>, method: &str) {
        if let Some(id) = id {
            let msg = serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": format!("method not found: {}", method)
                }
            });
            let _ = self.transport.write_message(&msg);
        }
    }

    fn handle_initialize(&mut self, id: Option<serde_json::Value>, params: &serde_json::Value) {
        // Set root path
        if let Some(root_uri) = params["rootUri"].as_str() {
            self.state.root = uri_to_path(root_uri);
        } else if let Some(root_path) = params["rootPath"].as_str() {
            self.state.root = Some(PathBuf::from(root_path));
        }

        // Build the workspace symbol index from the project root.
        if let Some(root) = &self.state.root {
            let index = WorkspaceIndex::build(root);
            self.state.index = Some(index);
        }

        let capabilities = serde_json::json!({
            "capabilities": {
                "textDocumentSync": {
                    "openClose": true,
                    "change": 1  // Full sync
                },
                "hoverProvider": true,
                "completionProvider": {
                    "triggerCharacters": [".", ":", " "]
                },
                "documentSymbolProvider": true,
                "definitionProvider": true,
                "workspaceSymbolProvider": true,
                "documentFormattingProvider": true,
            },
            "serverInfo": {
                "name": "kuzo-lsp",
                "version": "0.1.0"
            }
        });
        self.respond(id, capabilities);
    }

    fn handle_did_open(&mut self, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_string();
        let text = params["textDocument"]["text"].as_str().unwrap_or("").to_string();
        let version = params["textDocument"]["version"].as_i64().unwrap_or(0) as i32;

        let mut doc = DocState::new(text, version);
        let diagnostics = doc.recheck(&self.state);

        // Refresh the workspace index for this module so cross-file
        // definition/completion stay in sync with the opened buffer.
        if let Some(idx) = self.state.index.as_mut() {
            if let Some(module_key) = uri_to_module_key(&uri) {
                let arena = Bump::new();
                let parse_result =
                    Pipeline::parse_entry_module_lsp(&arena, &doc.text, "lsp_buffer.kz");
                idx.update_module(&module_key, &parse_result.module);
            }
        }

        self.state.docs.insert(uri.clone(), doc);
        self.publish_diagnostics(&uri, version, &diagnostics);
    }

    fn handle_did_change(&mut self, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_string();
        let version = params["textDocument"]["version"].as_i64().unwrap_or(0) as i32;

        // Full sync: take the last change
        let changes = params["contentChanges"].as_array();
        let text = changes
            .and_then(|c| c.last())
            .and_then(|c| c["text"].as_str())
            .unwrap_or("")
            .to_string();

        // Remove the doc from the map to avoid borrow conflicts with self.state.
        let mut doc = match self.state.docs.remove(&uri) {
            Some(d) => d,
            None => return,
        };

        // content_hash short-circuit: skip if text unchanged and we have diagnostics.
        let prev_hash = doc.content_hash;
        doc.update(text, version);
        if doc.content_hash == prev_hash && !doc.diagnostics.is_empty() {
            self.state.docs.insert(uri, doc);
            return;
        }

        // Parse (LSP-safe, never exits)
        let arena = Bump::new();
        let parse_result = Pipeline::parse_entry_module_lsp(&arena, &doc.text, "lsp_buffer.kz");
        let mut all_diags = parse_result.diagnostics;

        // Extract decl signatures for API change detection
        let new_decls = extract_decl_signatures(&parse_result.module);
        let is_api = match &doc.prev_decls {
            Some(old) => is_api_change(old, &new_decls),
            None => true,
        };
        doc.prev_decls = Some(new_decls);

        // Update symbols
        doc.symbols = extract_symbols(&parse_result.module);

        // Resolve module key (cached on the doc for next didChange)
        let module_key = doc
            .module_key
            .clone()
            .or_else(|| uri_to_module_key(&uri));
        doc.module_key = module_key.clone();

        // Take loader and sema_cache out of state to avoid borrow conflicts
        // during the sema/analyze/lint phase (publish_diagnostics needs &mut self).
        let loader_opt = self.state.loader.take();
        let cache_opt = self.state.sema_cache.take();

        let (new_loader, type_arena, sema_result, sema_diags, dirty_closure) =
            if let (Some(mut loader), Some((mut ta, mut sr))) = (loader_opt, cache_opt) {
                // Replace the changed module in the loader
                if let Some(key) = &module_key {
                    loader.replace_module(key, &doc.text);
                }

                // Compute dirty closure: full reverse-dep closure on API change,
                // just the changed module otherwise.
                let dirty = if is_api {
                    module_key
                        .as_ref()
                        .map(|k| loader.dirty_closure(k))
                        .unwrap_or_default()
                } else {
                    module_key.iter().cloned().collect()
                };
                // Clone before run_sema_incremental borrows it, so the closure
                // can be re-used for cross-document diagnostic publication.
                let dirty_closure = dirty.clone();

                match Pipeline::run_sema_incremental(&loader, &dirty, &mut ta, &mut sr) {
                    Pipeline::SemaIncrementalOutcome::Ok {
                        type_arena,
                        sema_result,
                        diagnostics,
                        ..
                    } => (Some(loader), type_arena, sema_result, diagnostics, dirty_closure),
                    Pipeline::SemaIncrementalOutcome::NeedsFull => {
                        // Fall back to full sema; old ta/sr are discarded.
                        let (ta, sr, d) =
                            full_sema(&loader, &parse_result.module, module_key.as_deref());
                        (Some(loader), ta, sr, d, dirty_closure)
                    }
                    Pipeline::SemaIncrementalOutcome::Err(d) => {
                        // Currently never returned by run_sema_incremental.
                        // Handle gracefully: merge error diags into all_diags,
                        // reuse the (partially modified) arena/sema_result.
                        all_diags = merge_diagnostics(all_diags, d);
                        (Some(loader), ta, sr, Vec::new(), dirty_closure)
                    }
                }
            } else {
                // No cache yet — build a fresh loader and run full sema.
                let mut loader = ModuleLoader::new();
                if let Some(root) = &self.state.root {
                    loader.add_search_path(root);
                }
                let _ = loader.load_transitive_imports(&parse_result.module);
                // Load all std modules (mirrors load_all_modules_or_exit without the exit).
                for (key, _) in crate::module::STD_FILES {
                    let parts: Vec<&str> =
                        key.strip_suffix(".kz").unwrap().split('/').collect();
                    let _ = loader.resolve_and_load(&parts);
                }
                let (ta, sr, d) =
                    full_sema(&loader, &parse_result.module, module_key.as_deref());
                (Some(loader), ta, sr, d, FxHashSet::default())
            };

        // Analyze (current module only)
        let analysis = Analyzer::analyze(
            &parse_result.module,
            &parse_result.module.arena,
            &sema_result,
        );

        // Lint
        let registry = RuleRegistry::new();
        let lint_diags = registry.run_all(
            &parse_result.module,
            &parse_result.module.arena,
            &sema_result,
            Some(&analysis),
            &self.state.lint_config,
        );

        // Merge diagnostics: parse + sema + lint
        all_diags = merge_diagnostics(all_diags, sema_diags);
        all_diags = merge_diagnostics(all_diags, lint_diags);
        doc.diagnostics = all_diags;

        // Restore loader and sema cache
        self.state.loader = new_loader;
        self.state.sema_cache = Some((type_arena, sema_result));

        // Publish diagnostics for the dirty closure and re-insert doc
        self.publish_diagnostics_for_closure(&uri, version, &doc.diagnostics, &dirty_closure);
        self.state.docs.insert(uri, doc);
    }

    fn handle_did_close(&mut self, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("").to_string();
        self.state.docs.remove(&uri);
        // Clear diagnostics
        self.publish_diagnostics(&uri, 0, &[]);
    }

    fn publish_diagnostics(&mut self, uri: &str, version: i32, diags: &[Diagnostic]) {
        let lsp_diags: Vec<serde_json::Value> = diags.iter().map(|d| {
            serde_json::json!({
                "range": {
                    "start": {
                        "line": d.range.start.line.saturating_sub(1),
                        "character": d.range.start.col.saturating_sub(1)
                    },
                    "end": {
                        "line": d.range.end.line.saturating_sub(1),
                        "character": d.range.end.col.saturating_sub(1)
                    }
                },
                "severity": match d.severity {
                    crate::tooling::Common::Diagnostic::Severity::Error => 1,
                    crate::tooling::Common::Diagnostic::Severity::Warning => 2,
                    crate::tooling::Common::Diagnostic::Severity::Advice => 3,
                },
                "code": d.code,
                "source": "kuzo",
                "message": d.message,
            })
        }).collect();

        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "version": version,
                "diagnostics": lsp_diags
            }
        });
        let _ = self.transport.write_message(&msg);
    }

    /// Publish diagnostics for the changed document plus any other open
    /// documents that fall inside the incremental sema dirty closure.
    fn publish_diagnostics_for_closure(
        &mut self,
        changed_uri: &str,
        changed_version: i32,
        changed_diags: &[Diagnostic],
        dirty: &FxHashSet<String>,
    ) {
        // 1. Current document
        self.publish_diagnostics(changed_uri, changed_version, changed_diags);

        // 2. Other open documents in dirty closure.
        // Collect target URIs first to avoid borrow conflicts with self.state.docs
        // while publishing (publish needs &mut self).
        let other_uris: Vec<String> = dirty
            .iter()
            .filter_map(|mk| {
                let uri = module_key_to_uri(mk);
                if uri == changed_uri {
                    return None;
                }
                if self.state.docs.contains_key(&uri) {
                    Some(uri)
                } else {
                    None
                }
            })
            .collect();

        for uri in other_uris {
            if let Some(d) = self.state.docs.get(&uri) {
                let version = d.version;
                let diags = d.diagnostics.clone();
                self.publish_diagnostics(&uri, version, &diags);
            }
        }
    }

    fn handle_hover(&mut self, id: Option<serde_json::Value>, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
        let line = params["position"]["line"].as_i64().unwrap_or(0) as u32;
        let character = params["position"]["character"].as_i64().unwrap_or(0) as u32;

        let doc = match self.state.docs.get(uri) {
            Some(d) => d,
            None => {
                self.respond(id, serde_json::json!(null));
                return;
            }
        };

        let word = match extract_word_at(&doc.text, line, character) {
            Some(w) => w,
            None => {
                self.respond(id, serde_json::json!(null));
                return;
            }
        };
        let module_key = uri_to_module_key(uri).unwrap_or_default();

        let hover = if let Some(sym) = doc.symbols.iter().find(|s| s.name == word) {
            Some(format!("**{}**: {}", sym.name, symbol_kind_label(sym.kind)))
        } else if let Some(idx) = &self.state.index {
            idx.find_definition(&word, &module_key)
                .map(|(module, sym)| {
                    format!(
                        "**{}** (in {}): {}",
                        sym.name,
                        module,
                        symbol_kind_label(sym.kind)
                    )
                })
        } else {
            None
        };

        match hover {
            Some(content) => self.respond(id, serde_json::json!({
                "contents": { "kind": "markdown", "value": content }
            })),
            None => self.respond(id, serde_json::json!(null)),
        }
    }

    fn handle_completion(&mut self, id: Option<serde_json::Value>, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
        let _line = params["position"]["line"].as_i64().unwrap_or(0) as u32;
        let _character = params["position"]["character"].as_i64().unwrap_or(0) as u32;
        let trigger = params["context"]["triggerCharacter"].as_str();

        let items: Vec<serde_json::Value> = match self.state.docs.get(uri) {
            Some(doc) => {
                let index = &self.state.index;
                match trigger {
                    Some(".") => complete_member_access(doc),
                    Some(":") => complete_type_name(doc, index),
                    _ => complete_general(doc, index),
                }
            }
            None => Vec::new(),
        };

        self.respond(id, serde_json::json!({ "isIncomplete": false, "items": items }));
    }

    fn handle_doc_symbol(&mut self, id: Option<serde_json::Value>, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("");

        let symbols: Vec<serde_json::Value> = if let Some(doc) = self.state.docs.get(uri) {
            doc.symbols.iter().map(|sym| {
                serde_json::json!({
                    "name": sym.name,
                    "kind": match sym.kind {
                        super::DocState::SymbolKind::Function => 12,  // Function
                        super::DocState::SymbolKind::Type => 5,       // Class
                        super::DocState::SymbolKind::Constant => 14,  // Constant
                        super::DocState::SymbolKind::Variable => 13,  // Variable
                    },
                    "range": {
                        "start": { "line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1) },
                        "end": { "line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1) }
                    },
                    "selectionRange": {
                        "start": { "line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1) },
                        "end": { "line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1) }
                    }
                })
            }).collect()
        } else {
            Vec::new()
        };

        self.respond(id, serde_json::json!(symbols));
    }

    fn handle_definition(&mut self, id: Option<serde_json::Value>, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("");
        let line = params["position"]["line"].as_i64().unwrap_or(0) as u32;
        let character = params["position"]["character"].as_i64().unwrap_or(0) as u32;

        let doc = match self.state.docs.get(uri) {
            Some(d) => d,
            None => {
                self.respond(id, serde_json::json!([]));
                return;
            }
        };
        let name = match extract_word_at(&doc.text, line, character) {
            Some(n) => n,
            None => {
                self.respond(id, serde_json::json!([]));
                return;
            }
        };

        let module_key = uri_to_module_key(uri).unwrap_or_default();
        let result = if let Some(idx) = &self.state.index {
            idx.find_definition(&name, &module_key)
        } else {
            // Fallback: search in current document symbols
            doc.symbols
                .iter()
                .find(|s| s.name == name)
                .map(|s| (module_key.clone(), s.clone()))
        };

        match result {
            Some((def_module, sym)) => {
                let def_uri = module_key_to_uri(&def_module);
                self.respond(id, serde_json::json!([{
                    "uri": def_uri,
                    "range": {
                        "start": {"line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1)},
                        "end": {"line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1) + name.len() as u32}
                    }
                }]));
            }
            None => self.respond(id, serde_json::json!([])),
        }
    }

    fn handle_workspace_symbol(&mut self, id: Option<serde_json::Value>, params: &serde_json::Value) {
        let query = params["query"].as_str().unwrap_or("");
        let symbols: Vec<serde_json::Value> = match &self.state.index {
            Some(idx) if !query.is_empty() => idx
                .search(query)
                .iter()
                .map(|(module, sym)| {
                    serde_json::json!({
                        "name": sym.name,
                        "kind": symbol_kind_to_lsp(sym.kind),
                        "location": {
                            "uri": module_key_to_uri(module),
                            "range": {
                                "start": {"line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1)},
                                "end": {"line": sym.line.saturating_sub(1), "character": sym.col.saturating_sub(1) + sym.name.len() as u32}
                            }
                        }
                    })
                })
                .collect(),
            _ => vec![],
        };
        self.respond(id, serde_json::json!(symbols));
    }

    fn handle_formatting(&mut self, id: Option<serde_json::Value>, params: &serde_json::Value) {
        let uri = params["textDocument"]["uri"].as_str().unwrap_or("");

        if let Some(doc) = self.state.docs.get(uri) {
            let formatted = crate::tooling::Fmt::Engine::format(&doc.text, &self.state.fmt_config);

            // Return a single TextEdit replacing the entire document
            let line_count = doc.text.lines().count() as u32;
            self.respond(id, serde_json::json!([{
                "range": {
                    "start": { "line": 0, "character": 0 },
                    "end": { "line": line_count, "character": 0 }
                },
                "newText": formatted
            }]));
        } else {
            self.respond(id, serde_json::json!([]));
        }
    }
}

/// Convert a file:// URI to a PathBuf.
fn uri_to_path(uri: &str) -> Option<PathBuf> {
    if let Some(path) = uri.strip_prefix("file://") {
        Some(PathBuf::from(path))
    } else {
        None
    }
}

/// Derive a module key (e.g. "src/Foo.kz" or "stdlib/io/File.kz") from a file URI.
/// Falls back to the basename if no recognizable source root is found.
fn uri_to_module_key(uri: &str) -> Option<String> {
    let path = uri.strip_prefix("file://").unwrap_or(uri);
    if let Some(idx) = path.find("/src/") {
        return Some(path[idx + 1..].to_string());
    }
    if let Some(idx) = path.find("/stdlib/") {
        return Some(path[idx + 1..].to_string());
    }
    path.rsplit('/').next().map(|s| s.to_string())
}

/// Convert a module key back to a file:// URI.
fn module_key_to_uri(key: &str) -> String {
    format!("file:///{}", key)
}

/// Extract the identifier word covering the given (line, character) position.
fn extract_word_at(text: &str, line: u32, character: u32) -> Option<String> {
    let line_str = text.lines().nth(line as usize)?;
    let char_idx = character as usize;
    if char_idx > line_str.len() {
        return None;
    }

    let is_word_char = |c: char| c.is_alphanumeric() || c == '_';
    let bytes = line_str.as_bytes();
    let mut start = char_idx;
    while start > 0 && is_word_char(bytes[start - 1] as char) {
        start -= 1;
    }
    let mut end = char_idx;
    while end < bytes.len() && is_word_char(bytes[end] as char) {
        end += 1;
    }

    if start == end {
        return None;
    }
    Some(line_str[start..end].to_string())
}

/// Map a Kuzo SymbolKind to an LSP SymbolKind integer.
fn symbol_kind_to_lsp(kind: SymbolKind) -> i64 {
    match kind {
        SymbolKind::Function => 12,
        SymbolKind::Type => 5,
        SymbolKind::Constant => 14,
        SymbolKind::Variable => 13,
    }
}

/// Human-readable label for a SymbolKind (used in hover).
fn symbol_kind_label(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Function => "function",
        SymbolKind::Type => "type",
        SymbolKind::Constant => "constant",
        SymbolKind::Variable => "variable",
    }
}

/// Member-access completion (triggered by `.`): Phase 2 simplified — return
/// the current document's symbols.
fn complete_member_access(doc: &DocState) -> Vec<serde_json::Value> {
    doc.symbols
        .iter()
        .map(|sym| serde_json::json!({ "label": sym.name, "kind": 3 }))
        .collect()
}

/// Type-name completion (triggered by `:`): document types + builtin types.
fn complete_type_name(doc: &DocState, _index: &Option<WorkspaceIndex>) -> Vec<serde_json::Value> {
    let mut items: Vec<serde_json::Value> = Vec::new();
    for sym in &doc.symbols {
        if matches!(sym.kind, SymbolKind::Type) {
            items.push(serde_json::json!({ "label": sym.name, "kind": 5 }));
        }
    }
    // Builtin types (21 scalars + 8 generics + This type keyword)
    for t in &[
        "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "isize", "usize",
        "f16", "f32", "f64", "f128", "bool", "char", "str", "void", "null", "Throw", "Channel",
        "Async", "Lazy", "Atomic", "Sender", "Receiver", "Timer", "This",
    ] {
        items.push(serde_json::json!({ "label": t, "kind": 5 }));
    }
    items
}

/// General completion: keywords + document symbols + builtin constructors.
fn complete_general(doc: &DocState, _index: &Option<WorkspaceIndex>) -> Vec<serde_json::Value> {
    let keywords = [
        "fun", "type", "trait", "override", "pack", "pub", "import", "with", "as", "val", "var",
        "match", "if", "else", "async", "channel", "select", "atomic", "loop", "for", "in",
        "while", "break", "continue", "return", "throw", "lazy", "defer", "this",
    ];
    let mut items: Vec<serde_json::Value> = keywords
        .iter()
        .map(|kw| serde_json::json!({ "label": kw, "kind": 14 }))
        .collect();
    for sym in &doc.symbols {
        items.push(serde_json::json!({
            "label": sym.name,
            "kind": symbol_kind_to_lsp(sym.kind)
        }));
    }
    for ctor in &[
        "Panic", "Ok", "channel", "Value", "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32",
        "u64", "u128", "isize", "usize", "f16", "f32", "f64", "f128", "bool", "char",
    ] {
        items.push(serde_json::json!({ "label": ctor, "kind": 3 }));
    }
    items
}

/// Merge two diagnostic vectors into one.
fn merge_diagnostics(a: Vec<Diagnostic>, b: Vec<Diagnostic>) -> Vec<Diagnostic> {
    let mut result = a;
    result.extend(b);
    result
}

/// Run full sema pipeline using an existing loader.
///
/// Computes std_keys and dep_keys from the loader's loaded modules, excluding
/// the entry module (identified by `exclude_key`) to avoid double-checking.
fn full_sema(
    loader: &ModuleLoader,
    module: &crate::ast::Ast::Module,
    exclude_key: Option<&str>,
) -> (TypeArena, SemaResult, Vec<Diagnostic>) {
    let std_keys: Vec<String> = crate::module::STD_FILES
        .iter()
        .map(|(p, _)| p.to_string())
        .collect();

    let dep_keys: Vec<String> = loader
        .loaded_keys()
        .into_iter()
        .filter(|k| {
            !std_keys.contains(k)
                && !crate::module::BUILTIN_FILES.iter().any(|(p, _)| *p == k)
                && Some(k.as_str()) != exclude_key
        })
        .collect();

    match Pipeline::run_sema_pipeline_lsp(loader, &std_keys, &dep_keys, module, "lsp_buffer.kz") {
        Pipeline::SemaOutcome::Ok {
            type_arena,
            sema_result,
            diagnostics,
        } => (type_arena, sema_result, diagnostics),
        Pipeline::SemaOutcome::Err(d) => (TypeArena::default(), SemaResult::new(), d),
    }
}
