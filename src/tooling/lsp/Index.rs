//! Workspace index: symbol table for navigation and cross-file definition.

use std::path::Path;

use rustc_hash::FxHashMap;

use crate::ast::Ast::Module;
use crate::tooling::common::Pipeline;
use crate::tooling::lsp::DocState::{extract_symbols, SymbolInfo};

#[derive(Default)]
pub struct WorkspaceIndex {
    /// Symbol name -> [(module_key, SymbolInfo)]
    symbols: FxHashMap<String, Vec<(String, SymbolInfo)>>,
    /// module_key -> [SymbolInfo]
    module_symbols: FxHashMap<String, Vec<SymbolInfo>>,
    /// Indexed module keys
    indexed: std::collections::HashSet<String>,
}

impl WorkspaceIndex {
    /// Build index by scanning .kz files under root.
    pub fn build(root: &Path) -> Self {
        let mut idx = Self::default();
        idx.scan_directory(root);
        idx
    }

    fn scan_directory(&mut self, dir: &Path) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if name == "target" || name == ".git" || name == "node_modules" {
                        continue;
                    }
                    self.scan_directory(&path);
                } else if path.extension().map(|e| e == "kz").unwrap_or(false) {
                    self.index_file(&path);
                }
            }
        }
    }

    fn index_file(&mut self, path: &Path) {
        if let Ok(source) = std::fs::read_to_string(path) {
            let key = path_to_module_key(path);
            if self.indexed.insert(key.clone()) {
                let arena = bumpalo::Bump::new();
                let parse_result = Pipeline::parse_entry_module_lsp(&arena, &source, &key);
                let syms = extract_symbols(&parse_result.module);
                for s in &syms {
                    self.symbols
                        .entry(s.name.clone())
                        .or_default()
                        .push((key.clone(), s.clone()));
                }
                self.module_symbols.insert(key, syms);
            }
        }
    }

    /// Update symbols for a single module (didOpen/didChange)
    pub fn update_module(&mut self, module_key: &str, module: &Module<'_>) {
        // Remove old symbols for this module
        if let Some(old_syms) = self.module_symbols.remove(module_key) {
            for s in &old_syms {
                if let Some(vec) = self.symbols.get_mut(&s.name) {
                    vec.retain(|(k, _)| k != module_key);
                    if vec.is_empty() {
                        self.symbols.remove(&s.name);
                    }
                }
            }
        }
        // Insert new symbols
        let syms = extract_symbols(module);
        for s in &syms {
            self.symbols
                .entry(s.name.clone())
                .or_default()
                .push((module_key.to_string(), s.clone()));
        }
        self.module_symbols.insert(module_key.to_string(), syms);
        self.indexed.insert(module_key.to_string());
    }

    /// Find definition: same module first, then global fallback
    pub fn find_definition(&self, name: &str, from_module: &str) -> Option<(String, SymbolInfo)> {
        // 1. Same module
        if let Some(syms) = self.module_symbols.get(from_module) {
            if let Some(s) = syms.iter().find(|s| s.name == name) {
                return Some((from_module.to_string(), s.clone()));
            }
        }
        // 2. Global fallback
        if let Some(candidates) = self.symbols.get(name) {
            if !candidates.is_empty() {
                return Some(candidates[0].clone());
            }
        }
        None
    }

    /// Workspace symbol search (workspace/symbol)
    pub fn search(&self, query: &str) -> Vec<(String, SymbolInfo)> {
        if query.is_empty() {
            return vec![];
        }
        let query_lower = query.to_lowercase();
        self.symbols
            .iter()
            .filter(|(name, _)| name.to_lowercase().contains(&query_lower))
            .flat_map(|(_, v)| v.iter().cloned())
            .collect()
    }
}

fn path_to_module_key(path: &Path) -> String {
    let s = path.to_string_lossy();
    if let Some(idx) = s.find("/src/") {
        s[idx + 1..].to_string()
    } else if let Some(idx) = s.find("/stdlib/") {
        s[idx + 1..].to_string()
    } else {
        s.to_string()
    }
}
