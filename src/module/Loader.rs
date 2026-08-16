//! Module loader core: module cache, search paths, and transitive dependency loading.
//!
//! Merges the stdlib embed table and the filesystem as two backends, transparently to the caller.
//! builtin modules are fully preloaded in `new()`.

use bumpalo::Bump;
use rustc_hash::{FxHashMap, FxHashSet};
use std::path::PathBuf;

use crate::ast::Ast::{Decl, ImportItem, Module, Visibility};
use crate::ast::Parser::{ErrorCollector, Lexer, ParseError, Parser, Token, TokenCollector};

use super::Error::{LoadError, LoadedModule};
use super::StdlibEmbed::{BUILTIN_FILES, STD_FILES, find};

/// The unified module loader.
///
/// Merges the stdlib embed table and the filesystem as two backends, transparently to the caller.
/// builtin modules are fully preloaded in `new()`.
pub struct ModuleLoader {
    /// Module cache: relative path (e.g. `"std/io/File.frond"`) → `LoadedModule`.
    modules: FxHashMap<String, LoadedModule>,
    /// Filesystem search paths for user modules.
    search_paths: Vec<PathBuf>,
    /// Load failure records (module not found / parse failed), in occurrence order.
    load_errors: Vec<LoadError>,
    /// Set of paths already attempted but failed, to avoid recording duplicate errors for the same path.
    failed_paths: FxHashSet<String>,
    /// Forward import graph: module_key -> directly imported module_keys
    forward_deps: FxHashMap<String, FxHashSet<String>>,
    /// Reverse import graph: module_key -> modules that directly import it
    reverse_deps: FxHashMap<String, FxHashSet<String>>,
}

impl ModuleLoader {
    /// Creates a new loader and fully preloads the builtin modules.
    pub fn new() -> Self {
        let mut loader = Self {
            modules: FxHashMap::default(),
            search_paths: Vec::new(),
            load_errors: Vec::new(),
            failed_paths: FxHashSet::default(),
            forward_deps: FxHashMap::default(),
            reverse_deps: FxHashMap::default(),
        };
        loader.preload_builtins();
        loader
    }

    /// Adds a filesystem search path for user modules.
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// Preloads the builtin modules (visible by default, no import needed).
    ///
    /// Iterates over `BUILTIN_FILES`, parsing and caching each `.frond` file.
    /// builtin modules are ordered by dependency (error → io → iter), ensuring dependencies
    /// are ready when subsequent checks run. Parse failures are recorded in `load_errors`
    /// to avoid silently swallowing errors.
    fn preload_builtins(&mut self) {
        for (path, source) in BUILTIN_FILES {
            match parse_source(path, source) {
                Ok((arena, source_owned, module)) => {
                    let exports = collect_exports(&module);
                    self.modules.insert(
                        path.to_string(),
                        LoadedModule {
                            _arena: arena,
                            _source: source_owned,
                            module,
                            exports,
                        },
                    );
                }
                Err(err) => {
                    self.failed_paths.insert(path.to_string());
                    self.load_errors.push(LoadError::ParseFailed {
                        path: path.to_string(),
                        line: err.line,
                        column: err.column,
                        message: err.message,
                    });
                }
            }
        }
    }

    /// Resolves and loads a module by its path segments.
    ///
    /// `path = ["std", "io", "File"]` → looks up `"std/io/File.frond"`
    /// Lookup order: cache → stdlib embed table → filesystem search paths
    ///
    /// Returns a reference to the loaded `Module`. On load failure (module not found /
    /// parse failed) returns `None`; the failure reason is recorded structurally in
    /// `load_errors` for the caller to report via `load_errors()`.
    pub fn resolve_and_load(&mut self, path: &[&str]) -> Option<&Module<'static>> {
        let path_str = module_path_to_file(path);

        // 1. Check the cache (already successfully loaded)
        if self.modules.contains_key(&path_str) {
            return self.modules.get(&path_str).map(|m| &m.module);
        }

        // 2. Known-failed path: do not record a duplicate error, just return None
        if self.failed_paths.contains(&path_str) {
            return None;
        }

        // 3. Look up the stdlib embed table
        if let Some(source) = find(&path_str) {
            match parse_source(&path_str, source) {
                Ok((arena, source_owned, module)) => {
                    let exports = collect_exports(&module);
                    self.modules.insert(
                        path_str.clone(),
                        LoadedModule {
                            _arena: arena,
                            _source: source_owned,
                            module,
                            exports,
                        },
                    );
                    return self.modules.get(&path_str).map(|m| &m.module);
                }
                Err(err) => {
                    self.failed_paths.insert(path_str.clone());
                    self.load_errors.push(LoadError::ParseFailed {
                        path: path_str,
                        line: err.line,
                        column: err.column,
                        message: err.message,
                    });
                    return None;
                }
            }
        }

        // 4. Look up the filesystem (user modules)
        for base in &self.search_paths {
            let full = base.join(&path_str);
            if full.exists() {
                match std::fs::read_to_string(&full) {
                    Ok(source) => {
                        match parse_source(&path_str, &source) {
                            Ok((arena, source_owned, module)) => {
                                let exports = collect_exports(&module);
                                self.modules.insert(
                                    path_str.clone(),
                                    LoadedModule {
                                        _arena: arena,
                                        _source: source_owned,
                                        module,
                                        exports,
                                    },
                                );
                                return self.modules.get(&path_str).map(|m| &m.module);
                            }
                            Err(err) => {
                                self.failed_paths.insert(path_str.clone());
                                self.load_errors.push(LoadError::ParseFailed {
                                    path: path_str,
                                    line: err.line,
                                    column: err.column,
                                    message: err.message,
                                });
                                return None;
                            }
                        }
                    }
                    Err(_) => continue,
                }
            }
        }

        // 4b. Directory module detection: `path` refers not to a file but to a directory (containing pack.frond).
        // e.g. `import Store` → Store.frond does not exist, but Store/pack.frond does.
        // Load pack.frond to obtain submodule declarations, then load each submodule file.
        let dir_name = path_str.strip_suffix(".frond").unwrap_or(&path_str);
        for base in &self.search_paths {
            let pack_file = base.join(dir_name).join("pack.frond");
            if !pack_file.exists() {
                continue;
            }
            let pack_source = match std::fs::read_to_string(&pack_file) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let pack_path_key = format!("{}/pack.frond", dir_name);
            let (pack_arena, pack_source_owned, pack_module) =
                match parse_source(&pack_path_key, &pack_source) {
                    Ok(result) => result,
                    Err(err) => {
                        self.failed_paths.insert(path_str.clone());
                        self.load_errors.push(LoadError::ParseFailed {
                            path: path_str,
                            line: err.line,
                            column: err.column,
                            message: err.message,
                        });
                        return None;
                    }
                };
            // Load each submodule declared by the pack
            for sub_name in collect_pack_submodules(&pack_module) {
                let sub_path_str = format!("{}/{}.frond", dir_name, sub_name);
                // The submodule may already be in the cache (e.g. loaded first via another path)
                if self.modules.contains_key(&sub_path_str) {
                    continue;
                }
                let sub_full = base.join(&sub_path_str);
                if let Ok(sub_source) = std::fs::read_to_string(&sub_full) {
                    if let Ok((sub_arena, sub_source_owned, sub_module)) =
                        parse_source(&sub_path_str, &sub_source)
                    {
                        let sub_exports = collect_exports(&sub_module);
                        self.modules.insert(
                            sub_path_str,
                            LoadedModule {
                                _arena: sub_arena,
                                _source: sub_source_owned,
                                module: sub_module,
                                exports: sub_exports,
                            },
                        );
                    }
                }
            }
            // Register the pack module as the directory module representative (key is the original path_str, e.g. "Store.frond")
            let pack_exports = collect_exports(&pack_module);
            self.modules
                .insert(path_str.clone(), LoadedModule {
                    _arena: pack_arena,
                    _source: pack_source_owned,
                    module: pack_module,
                    exports: pack_exports,
                });
            return self.modules.get(&path_str).map(|m| &m.module);
        }

        // 5. Neither stdlib nor the filesystem matched: check whether this is a type/symbol
        // exported by a sibling module. For example `import std.time.TimeComponents` →
        // TimeComponents is a type exported by SystemTime.frond, not a standalone module file.
        // In this case no error is reported; the symbol is visible through the already-loaded sibling module.
        if let Some(symbol_name) = extract_last_segment(&path_str) {
            let parent_prefix = parent_directory(&path_str);

            // 5a. First check already-loaded sibling modules
            let already_exported = self
                .modules
                .iter()
                .any(|(mod_path, mod_data)| {
                    mod_path.starts_with(&parent_prefix) && mod_data.exports.contains(&symbol_name)
                });

            if already_exported {
                self.failed_paths.insert(path_str);
                return None;
            }

            // 5b. Check sibling modules in the stdlib embed table that have not yet been loaded.
            // Iterate over all files in BUILTIN_FILES and STD_FILES sharing the same parent directory,
            // find the file that exports this symbol, and load it.
            for (sibling_file, _) in BUILTIN_FILES.iter().chain(STD_FILES.iter()) {
                if !sibling_file.starts_with(&parent_prefix) || *sibling_file == path_str {
                    continue;
                }
                // Also check exports of already-loaded modules
                if let Some(mod_data) = self.modules.get(*sibling_file) {
                    if mod_data.exports.contains(&symbol_name) {
                        self.failed_paths.insert(path_str);
                        return None;
                    }
                    continue;
                }
                // Unloaded sibling module: load it and check its exports
                if let Some(source) = find(sibling_file) {
                    if let Ok((sib_arena, sib_source_owned, module)) =
                        parse_source(sibling_file, source)
                    {
                        let exports = collect_exports(&module);
                        if exports.contains(&symbol_name) {
                            self.modules.insert(
                                sibling_file.to_string(),
                                LoadedModule {
                                    _arena: sib_arena,
                                    _source: sib_source_owned,
                                    module,
                                    exports,
                                },
                            );
                            self.failed_paths.insert(path_str);
                            return None;
                        }
                    }
                }
            }
        }

        // 6. Genuinely not found: record module-not-found
        self.failed_paths.insert(path_str.clone());
        self.load_errors.push(LoadError::ModuleNotFound { path: path_str });
        None
    }

    /// Returns the loaded builtin modules (in `BUILTIN_FILES` order).
    pub fn builtin_modules(&self) -> impl Iterator<Item = (&str, &Module<'static>)> {
        BUILTIN_FILES.iter().filter_map(|(path, _)| {
            self.modules.get(*path).map(|m| (*path, &m.module))
        })
    }

    /// Returns all load failure records (module not found / parse failed), in occurrence order.
    ///
    /// The caller should inspect and report these errors before running sema checks, to avoid
    /// a flood of cascading type false positives caused by missing modules that would mask the
    /// real root cause.
    pub fn load_errors(&self) -> &[LoadError] {
        &self.load_errors
    }

    /// Returns whether any load failure has occurred.
    pub fn has_load_errors(&self) -> bool {
        !self.load_errors.is_empty()
    }

    /// Recursively loads all transitive dependencies (imported modules) of `module`.
    ///
    /// Post-order traversal: depended-on modules appear earlier in the return value, so that
    /// when the caller checks modules in the returned order, the definitions of depended-on
    /// modules have already been populated into the `SemaResult`.
    /// builtin modules are preloaded in `new()` and are not included in the return value.
    ///
    /// Returns the module cache keys (in file-path form, e.g. `"std/io/File.frond"`) ordered for checking.
    pub fn load_transitive_imports(&mut self, module: &Module<'_>) -> Vec<String> {
        let mut order: Vec<String> = Vec::new();
        // visited: finalized modules (already registered in `order`)
        let mut visited: FxHashSet<String> = FxHashSet::default();
        // visiting: modules currently on the stack being expanded but not yet finalized; used for cycle detection.
        // Under a circular dependency (A↔B), the second time (A, false) is encountered, visiting.contains(A) hits
        // and we skip it, avoiding infinite expansion. Post-order traversal remains correct for the acyclic parts.
        let mut visiting: FxHashSet<String> = FxHashSet::default();
        // Stack element: (module path segments, whether child dependencies have been collected)
        let mut stack: Vec<(Vec<String>, bool)> = collect_imports(module)
            .into_iter()
            .map(|(p, _)| (p.iter().map(|s| s.to_string()).collect::<Vec<String>>(), false))
            .collect();

        while let Some((path_segments, expanded)) = stack.pop() {
            let path_refs: Vec<&str> = path_segments.iter().map(|s| s.as_str()).collect();
            let key = module_path_to_file(&path_refs);
            if visited.contains(&key) {
                continue;
            }
            if !expanded {
                // Cycle detection: if `key` is already on the current expansion path, record an error and skip to avoid infinite loops
                if visiting.contains(&key) {
                    self.load_errors.push(LoadError::CircularImport {
                        path: key.clone(),
                    });
                    continue;
                }
                visiting.insert(key.clone());
                // First visit: collect child dependency paths (owned), then push self back onto the stack
                let mut child_segs_list: Vec<Vec<String>> = Vec::new();
                if let Some(dep) = self.resolve_and_load(&path_refs) {
                    for (child_path, _) in collect_imports(dep) {
                        child_segs_list.push(
                            child_path.iter().map(|s| s.to_string()).collect::<Vec<String>>(),
                        );
                    }
                    // Directory module: submodules declared in pack.frond must also be added to the check order
                    // e.g. `import Store` → pack.frond declares `pub pack Memory` → submodule path ["Store", "Memory"]
                    for sub_name in collect_pack_submodules(dep) {
                        let mut child_segs: Vec<String> = path_segments.clone();
                        child_segs.push(sub_name.to_string());
                        child_segs_list.push(child_segs);
                    }
                }
                // Push self back onto the stack (marked expanded); register it into `order` after its children are processed
                stack.push((path_segments, true));
                // Push children onto the stack (LIFO ensures post-order: children are registered into `order` before self)
                for child_segs in child_segs_list {
                    stack.push((child_segs, false));
                }
            } else {
                visiting.remove(&key);
                visited.insert(key.clone());
                order.push(key);
            }
        }
        // Phase 2: build forward/reverse dependency graphs
        let all_keys: Vec<String> = self.modules.keys().cloned().collect();
        for key in &all_keys {
            if let Some(loaded) = self.modules.get(key) {
                let imports = collect_imports(&loaded.module);
                let mut new_deps: FxHashSet<String> = FxHashSet::default();
                for (path, _) in imports {
                    new_deps.insert(module_path_to_file(&path));
                }
                // Remove old reverse_deps edges that are no longer present
                let old_deps = self.forward_deps.get(key).cloned().unwrap_or_default();
                for old in &old_deps {
                    if !new_deps.contains(old) {
                        if let Some(rev) = self.reverse_deps.get_mut(old) {
                            rev.remove(key);
                        }
                    }
                }
                // Add new reverse_deps edges
                for new in &new_deps {
                    if !old_deps.contains(new) {
                        self.reverse_deps.entry(new.clone()).or_default().insert(key.clone());
                    }
                }
                self.forward_deps.insert(key.clone(), new_deps);
            }
        }
        order
    }

    /// Returns the loaded module by cache key (key is the return value of `module_path_to_file`, e.g. `"std/io/File.frond"`).
    pub fn get_module_by_key(&self, key: &str) -> Option<&Module<'static>> {
        self.modules.get(key).map(|m| &m.module)
    }

    /// Returns the cache keys of all loaded modules (in file-path form, e.g. `"std/io/File.frond"`).
    pub fn loaded_keys(&self) -> Vec<String> {
        self.modules.keys().map(|s| s.to_string()).collect()
    }

    /// Get forward deps for a module (for LSP/testing)
    pub fn get_forward_deps(&self, module_key: &str) -> Option<&FxHashSet<String>> {
        self.forward_deps.get(module_key)
    }

    /// Get reverse deps for a module (for LSP/testing)
    pub fn get_reverse_deps(&self, module_key: &str) -> Option<&FxHashSet<String>> {
        self.reverse_deps.get(module_key)
    }

    /// Compute dirty closure after module M changes (transitive reverse deps).
    /// Returns the set of modules that need sema recheck: M plus all modules
    /// that transitively import M.
    pub fn dirty_closure(&self, changed: &str) -> FxHashSet<String> {
        let mut dirty = FxHashSet::default();
        dirty.insert(changed.to_string());
        let mut frontier = vec![changed.to_string()];
        while let Some(m) = frontier.pop() {
            if let Some(importers) = self.reverse_deps.get(&m) {
                for imp in importers {
                    if dirty.insert(imp.clone()) {
                        frontier.push(imp.clone());
                    }
                }
            }
        }
        dirty
    }

    /// Replace a module's source code and re-parse it.
    /// Used by LSP didChange to update a single module without reloading everything.
    /// Returns true if the module was successfully parsed and cached.
    pub fn replace_module(&mut self, key: &str, source: &str) -> bool {
        match parse_source(key, source) {
            Ok((arena, source_owned, module)) => {
                // Update dep graph
                let imports = collect_imports(&module);
                let mut new_deps: FxHashSet<String> = FxHashSet::default();
                for (path, _) in imports {
                    new_deps.insert(module_path_to_file(&path));
                }

                // Remove old reverse_deps edges that are no longer present
                let old_deps = self.forward_deps.get(key).cloned().unwrap_or_default();
                for old in &old_deps {
                    if !new_deps.contains(old) {
                        if let Some(rev) = self.reverse_deps.get_mut(old) {
                            rev.remove(key);
                        }
                    }
                }
                // Add new reverse_deps edges
                for new in &new_deps {
                    if !old_deps.contains(new) {
                        self.reverse_deps
                            .entry(new.clone())
                            .or_default()
                            .insert(key.to_string());
                    }
                }
                self.forward_deps.insert(key.to_string(), new_deps);

                // Compute exports and replace in cache.
                // The old LoadedModule (if any) is dropped here, reclaiming its arena and source.
                let exports = collect_exports(&module);
                self.modules.insert(
                    key.to_string(),
                    LoadedModule {
                        _arena: arena,
                        _source: source_owned,
                        module,
                        exports,
                    },
                );
                true
            }
            Err(_) => false,
        }
    }
}

impl Default for ModuleLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helper functions ──────────────────────────────────────────────

/// Converts module path segments to a file path.
/// `["std", "io", "File"]` → `"std/io/File.frond"`
fn module_path_to_file(path: &[&str]) -> String {
    let joined = path.join("/");
    if joined.ends_with(".frond") {
        joined
    } else {
        format!("{}.frond", joined)
    }
}

/// Extracts the last path segment as the module name (stripping the `.frond` suffix).
/// `"std/time/TimeComponents.frond"` → `"TimeComponents"`
fn extract_last_segment(path: &str) -> Option<String> {
    path.rsplit('/')
        .next()
        .and_then(|last| last.strip_suffix(".frond"))
        .map(|s| s.to_string())
}

/// Returns the parent directory prefix of a file path.
/// `"std/time/TimeComponents.frond"` → `"std/time/"`
fn parent_directory(path: &str) -> String {
    match path.rfind('/') {
        Some(idx) => path[..=idx].to_string(),
        None => String::new(),
    }
}

/// Parses source code into a `Module<'static>`, owning the backing arena and source string.
///
/// Returns `(arena, source, module)` so the caller can keep them alive together — typically
/// stored in a `LoadedModule` whose field drop order (reverse declaration) drops `module`
/// before `_arena` and `_source`.
///
/// On a fatal parse error, a `ParseError` is returned and all allocations are freed.
/// Non-fatal parse errors (already recovered by the parser) are emitted to stderr as warnings
/// and do not block loading.
fn parse_source(
    path: &str,
    source: &str,
) -> Result<(Box<Bump>, Box<str>, Module<'static>), ParseError> {
    let arena: Box<Bump> = Box::new(Bump::new());
    let source_owned: Box<str> = source.into();

    let module: Module<'static> = {
        // Borrow arena for the parsing scope; path is copied in so Module.name outlives the call.
        let arena_ref: &Bump = &*arena;
        let path_ref: &str = arena_ref.alloc_str(path);

        let mut lexer = Lexer::new(&source_owned);
        let mut sink = TokenCollector::new();
        lexer.tokenize_into(&mut sink);
        let tokens: Vec<Token> = sink.into_tokens();
        let tokens_ref = arena_ref.alloc_slice_copy(&tokens);

        let mut parser = Parser::new(tokens_ref, arena_ref, ErrorCollector::new());

        match parser.parse_module(path_ref) {
            Ok(module) => {
                // Non-fatal parse errors (already recovered by the parser): emit warnings; the module is still usable
                for err in parser.errors() {
                    eprintln!(
                        "Warning: parse error in {} at {}:{}: {}",
                        path, err.line, err.column, err.message
                    );
                }
                // Safety: `module` contains `&'a str` references that point into `arena`
                // (dynamically built strings, copied path) and `source_owned` (token lexemes).
                // Both are returned alongside `module` and stored in the same `LoadedModule`,
                // which drops fields in reverse declaration order — `module` is dropped before
                // `_arena` and `_source`. `Module` has no custom `Drop`; dropping it only
                // frees `Vec` buffers (on the regular heap, not the arena) and drops `&str`
                // references (no-ops), never dereferencing arena-allocated data.
                unsafe { std::mem::transmute::<_, Module<'static>>(module) }
            }
            Err(err) => return Err(err),
        }
    };

    Ok((arena, source_owned, module))
}

/// Collects the public export symbols of a module.
///
/// Iterates over `Module.declarations`, collecting the names of all `pub`-visibility
/// functions/types. Used later for import alias registration.
fn collect_exports(module: &Module<'_>) -> FxHashSet<String> {
    let mut exports = FxHashSet::default();
    for decl in &module.declarations {
        match &decl.node {
            Decl::FunDecl {
                name,
                visibility: Visibility::Public,
                ..
            } => {
                exports.insert((*name).to_string());
            }
            Decl::TypeDecl {
                name,
                visibility: Visibility::Public,
                ..
            } => {
                exports.insert((*name).to_string());
            }
            Decl::PackDecl {
                name,
                visibility: Visibility::Public,
            } => {
                exports.insert((*name).to_string());
            }
            _ => {}
        }
    }
    exports
}

/// Extracts the list of submodule names from a module's `pub pack <Name>` declarations.
///
/// A directory module's `pack.frond` declares its contained submodules via `PackDecl`.
/// For example, `pub pack Memory` in `Store/pack.frond` → returns `["Memory"]`.
/// `load_transitive_imports` uses this result to construct submodule paths (e.g. `["Store", "Memory"]`),
/// ensuring submodules are added to the check order.
fn collect_pack_submodules<'a>(module: &'a Module<'a>) -> Vec<&'a str> {
    let mut subs = Vec::new();
    for decl in &module.declarations {
        if let Decl::PackDecl {
            name,
            visibility: Visibility::Public,
        } = &decl.node
        {
            subs.push(*name);
        }
    }
    subs
}

// ─── ImportDecl traversal helpers ──────────────────────────────────

/// Traverses the `ImportDecl`s in a module, returning a list of `(module_path, items)`.
///
/// Used by the compilation entry point to batch-process imports before `check_module`.
pub fn collect_imports<'a>(
    module: &'a Module<'a>,
) -> Vec<(Vec<&'a str>, Option<&'a [ImportItem<'a>]>)> {
    let mut imports = Vec::new();
    for decl in &module.declarations {
        if let Decl::ImportDecl {
            module_path,
            items,
            ..
        } = &decl.node
        {
            let items_ref = items.as_deref();
            imports.push((module_path.to_vec(), items_ref));
        }
    }
    imports
}
