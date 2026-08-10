//! 模块加载器主体：模块缓存、搜索路径、传递依赖加载。
//!
//! 合并 stdlib 嵌入表和文件系统两种 backend，对调用方透明。
//! builtin 模块在 `new()` 时全量预加载。

use rustc_hash::{FxHashMap, FxHashSet};
use std::path::PathBuf;

use crate::ast::Ast::{Decl, ImportItem, Module, Visibility};
use crate::ast::Parser::{ErrorCollector, Lexer, ParseError, Parser, Token, TokenCollector};

use super::Error::{LoadError, LoadedModule};
use super::StdlibEmbed::{BUILTIN_FILES, STD_FILES, find};

/// 统一的模块加载器
///
/// 合并 stdlib 嵌入表和文件系统两种 backend，对调用方透明。
/// builtin 模块在 `new()` 时全量预加载。
pub struct ModuleLoader {
    /// 模块缓存：相对路径（如 "std/io/File.kz"）→ LoadedModule
    modules: FxHashMap<String, LoadedModule>,
    /// 用户模块的文件系统搜索路径
    search_paths: Vec<PathBuf>,
    /// 加载失败记录（模块未找到 / 解析失败），按发生顺序排列
    load_errors: Vec<LoadError>,
    /// 已尝试加载但失败的路径集合，避免对同一路径重复记录错误
    failed_paths: FxHashSet<String>,
}

impl ModuleLoader {
    /// 创建新的加载器，并全量预加载 builtin 模块
    pub fn new() -> Self {
        let mut loader = Self {
            modules: FxHashMap::default(),
            search_paths: Vec::new(),
            load_errors: Vec::new(),
            failed_paths: FxHashSet::default(),
        };
        loader.preload_builtins();
        loader
    }

    /// 添加用户模块的文件系统搜索路径
    pub fn add_search_path(&mut self, path: impl Into<PathBuf>) {
        self.search_paths.push(path.into());
    }

    /// 预加载 builtin 模块（默认可见，无需 import）
    ///
    /// 遍历 BUILTIN_FILES，parse 每个 .kz 文件并缓存。
    /// builtin 模块按依赖顺序排列（error → io → iter），保证后续 check 时依赖已就绪。
    /// 解析失败时记录到 `load_errors`，避免错误被静默吞掉。
    fn preload_builtins(&mut self) {
        for (path, source) in BUILTIN_FILES {
            match parse_source(path, source) {
                Ok(module) => {
                    let exports = collect_exports(&module);
                    self.modules
                        .insert(path.to_string(), LoadedModule { module, exports });
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

    /// 按模块路径段解析并加载模块
    ///
    /// `path = ["std", "io", "File"]` → 查找 "std/io/File.kz"
    /// 优先查缓存 → stdlib 嵌入表 → 文件系统搜索路径
    ///
    /// 返回已加载的 Module 引用。加载失败（模块未找到 / 解析失败）时返回 None，
    /// 失败原因结构化记录到 `load_errors`，由调用方通过 `load_errors()` 统一报告。
    pub fn resolve_and_load(&mut self, path: &[&str]) -> Option<&Module<'static>> {
        let path_str = module_path_to_file(path);

        // 1. 检查缓存（已成功加载）
        if self.modules.contains_key(&path_str) {
            return self.modules.get(&path_str).map(|m| &m.module);
        }

        // 2. 已知失败路径：不重复记录错误，直接返回 None
        if self.failed_paths.contains(&path_str) {
            return None;
        }

        // 3. 查找 stdlib 嵌入表
        if let Some(source) = find(&path_str) {
            let path_static: &'static str = Box::leak(path_str.clone().into_boxed_str());
            match parse_source(path_static, source) {
                Ok(module) => {
                    let exports = collect_exports(&module);
                    self.modules
                        .insert(path_str.clone(), LoadedModule { module, exports });
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

        // 4. 查找文件系统（用户模块）
        for base in &self.search_paths {
            let full = base.join(&path_str);
            if full.exists() {
                match std::fs::read_to_string(&full) {
                    Ok(source) => {
                        let source_static: &'static str =
                            Box::leak(source.into_boxed_str());
                        let path_static: &'static str =
                            Box::leak(path_str.clone().into_boxed_str());
                        match parse_source(path_static, source_static) {
                            Ok(module) => {
                                let exports = collect_exports(&module);
                                self.modules.insert(
                                    path_str.clone(),
                                    LoadedModule { module, exports },
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

        // 4b. 目录模块检测：path 对应的不是文件而是目录（含 pack.kz）
        // 例如 import Store → Store.kz 不存在，但 Store/pack.kz 存在。
        // 加载 pack.kz 获取子模块声明，再加载每个子模块文件。
        let dir_name = path_str.strip_suffix(".kz").unwrap_or(&path_str);
        for base in &self.search_paths {
            let pack_file = base.join(dir_name).join("pack.kz");
            if !pack_file.exists() {
                continue;
            }
            let pack_source = match std::fs::read_to_string(&pack_file) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let pack_path_key = format!("{}/pack.kz", dir_name);
            let pack_path_static: &'static str = Box::leak(pack_path_key.into_boxed_str());
            let pack_source_static: &'static str = Box::leak(pack_source.into_boxed_str());
            let pack_module = match parse_source(pack_path_static, pack_source_static) {
                Ok(m) => m,
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
            // 加载 pack 声明的每个子模块
            for sub_name in collect_pack_submodules(&pack_module) {
                let sub_path_str = format!("{}/{}.kz", dir_name, sub_name);
                // 子模块可能已在缓存中（如被其他路径先加载）
                if self.modules.contains_key(&sub_path_str) {
                    continue;
                }
                let sub_full = base.join(&sub_path_str);
                if let Ok(sub_source) = std::fs::read_to_string(&sub_full) {
                    let sub_source_static: &'static str =
                        Box::leak(sub_source.into_boxed_str());
                    let sub_path_static: &'static str =
                        Box::leak(sub_path_str.clone().into_boxed_str());
                    if let Ok(sub_module) = parse_source(sub_path_static, sub_source_static) {
                        let sub_exports = collect_exports(&sub_module);
                        self.modules.insert(
                            sub_path_str,
                            LoadedModule {
                                module: sub_module,
                                exports: sub_exports,
                            },
                        );
                    }
                }
            }
            // 将 pack 模块注册为目录模块代表（key 为原始 path_str，如 "Store.kz"）
            let pack_exports = collect_exports(&pack_module);
            self.modules
                .insert(path_str.clone(), LoadedModule {
                    module: pack_module,
                    exports: pack_exports,
                });
            return self.modules.get(&path_str).map(|m| &m.module);
        }

        // 5. stdlib 和文件系统均未命中：检查是否为同级模块导出的类型/符号
        // 例如 import std.time.TimeComponents → TimeComponents 是 SystemTime.kz 导出的 type，
        // 而非独立模块文件。此时不报错，符号通过已加载的同级模块可见。
        if let Some(symbol_name) = extract_last_segment(&path_str) {
            let parent_prefix = parent_directory(&path_str);

            // 5a. 先检查已加载的同级模块
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

            // 5b. 检查 stdlib 嵌入表中尚未加载的同级模块
            // 遍历 BUILTIN_FILES 和 STD_FILES 中父目录相同的所有文件，
            // 找到导出该符号的文件并加载它。
            for (sibling_file, _) in BUILTIN_FILES.iter().chain(STD_FILES.iter()) {
                if !sibling_file.starts_with(&parent_prefix) || *sibling_file == path_str {
                    continue;
                }
                // 已加载的模块也检查导出
                if let Some(mod_data) = self.modules.get(*sibling_file) {
                    if mod_data.exports.contains(&symbol_name) {
                        self.failed_paths.insert(path_str);
                        return None;
                    }
                    continue;
                }
                // 未加载的同级模块：加载并检查导出
                if let Some(source) = find(sibling_file) {
                    let sibling_static: &'static str =
                        Box::leak(sibling_file.to_string().into_boxed_str());
                    if let Ok(module) = parse_source(sibling_static, source) {
                        let exports = collect_exports(&module);
                        if exports.contains(&symbol_name) {
                            self.modules
                                .insert(sibling_file.to_string(), LoadedModule { module, exports });
                            self.failed_paths.insert(path_str);
                            return None;
                        }
                    }
                }
            }
        }

        // 6. 确实未找到：记录模块未找到
        self.failed_paths.insert(path_str.clone());
        self.load_errors.push(LoadError::ModuleNotFound { path: path_str });
        None
    }

    /// 获取已加载模块的导出符号列表
    pub fn get_exports(&self, path: &[&str]) -> Option<&FxHashSet<String>> {
        let path_str = module_path_to_file(path);
        self.modules.get(&path_str).map(|m| &m.exports)
    }

    /// 获取已加载的 builtin 模块（按 BUILTIN_FILES 顺序）
    pub fn builtin_modules(&self) -> impl Iterator<Item = (&str, &Module<'static>)> {
        BUILTIN_FILES.iter().filter_map(|(path, _)| {
            self.modules.get(*path).map(|m| (*path, &m.module))
        })
    }

    /// 获取所有已加载模块的数量
    pub fn loaded_count(&self) -> usize {
        self.modules.len()
    }

    /// 判断模块是否已加载
    pub fn is_loaded(&self, path: &[&str]) -> bool {
        let path_str = module_path_to_file(path);
        self.modules.contains_key(&path_str)
    }

    /// 返回所有加载失败记录（模块未找到 / 解析失败），按发生顺序排列。
    ///
    /// 调用方应在 sema check 之前检查并报告这些错误，避免因模块缺失引发
    /// 大量级联类型误报，掩盖真正的根因。
    pub fn load_errors(&self) -> &[LoadError] {
        &self.load_errors
    }

    /// 是否存在加载失败
    pub fn has_load_errors(&self) -> bool {
        !self.load_errors.is_empty()
    }

    /// 递归加载 `module` 的所有传递依赖（import 的模块）。
    ///
    /// 后序遍历：被依赖的模块先出现在返回值中，保证调用方按返回顺序
    /// check 时，被依赖模块的定义已先 populate 到 SemaResult。
    /// builtin 模块已在 `new()` 中预加载，不包含在返回值中。
    ///
    /// 返回按 check 顺序排列的模块缓存 key（文件路径形式，如 `"std/io/File.kz"`）。
    pub fn load_transitive_imports(&mut self, module: &Module<'_>) -> Vec<String> {
        let mut order: Vec<String> = Vec::new();
        // visited：已 finalize 的模块（已登记到 order）
        let mut visited: FxHashSet<String> = FxHashSet::default();
        // visiting：当前栈中正在展开但未 finalize 的模块，用于检测循环依赖
        // 循环依赖（A↔B）下，第二次遇到 (A,false) 时 visiting.contains(A) 命中，
        // 直接跳过，避免无限展开。后序遍历对无环部分仍正确。
        let mut visiting: FxHashSet<String> = FxHashSet::default();
        // 栈元素：(模块路径段, 是否已展开收集子依赖)
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
                // 循环依赖检测：若 key 已在当前展开路径中，记录错误并跳过避免无限循环
                if visiting.contains(&key) {
                    self.load_errors.push(LoadError::CircularImport {
                        path: key.clone(),
                    });
                    continue;
                }
                visiting.insert(key.clone());
                // 首次访问：先收集子依赖路径（owned），再重新入栈自己
                let mut child_segs_list: Vec<Vec<String>> = Vec::new();
                if let Some(dep) = self.resolve_and_load(&path_refs) {
                    for (child_path, _) in collect_imports(dep) {
                        child_segs_list.push(
                            child_path.iter().map(|s| s.to_string()).collect::<Vec<String>>(),
                        );
                    }
                    // 目录模块：pack.kz 中声明的子模块也需加入 check 顺序
                    // 例如 import Store → pack.kz 声明 pub pack Memory → 子模块路径 ["Store", "Memory"]
                    for sub_name in collect_pack_submodules(dep) {
                        let mut child_segs: Vec<String> = path_segments.clone();
                        child_segs.push(sub_name.to_string());
                        child_segs_list.push(child_segs);
                    }
                }
                // 自己重新入栈（标记 expanded），等子依赖处理完后再登记到 order
                stack.push((path_segments, true));
                // 子依赖入栈（LIFO 保证后序：子依赖先于自己登记到 order）
                for child_segs in child_segs_list {
                    stack.push((child_segs, false));
                }
            } else {
                visiting.remove(&key);
                visited.insert(key.clone());
                order.push(key);
            }
        }
        order
    }

    /// 按缓存 key 获取已加载模块（key 为 `module_path_to_file` 的返回值，如 `"std/io/File.kz"`）。
    pub fn get_module_by_key(&self, key: &str) -> Option<&Module<'static>> {
        self.modules.get(key).map(|m| &m.module)
    }

    /// 返回所有已加载模块的缓存 key（文件路径形式，如 `"std/io/File.kz"`）。
    pub fn loaded_keys(&self) -> Vec<String> {
        self.modules.keys().map(|s| s.to_string()).collect()
    }
}

impl Default for ModuleLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ─── 辅助函数 ──────────────────────────────────────────────────────

/// 模块路径段 → 文件路径
/// `["std", "io", "File"]` → `"std/io/File.kz"`
fn module_path_to_file(path: &[&str]) -> String {
    let joined = path.join("/");
    if joined.ends_with(".kz") {
        joined
    } else {
        format!("{}.kz", joined)
    }
}

/// 从文件路径中提取最后一段的模块名（去掉 .kz 后缀）
/// `"std/time/TimeComponents.kz"` → `"TimeComponents"`
fn extract_last_segment(path: &str) -> Option<String> {
    path.rsplit('/')
        .next()
        .and_then(|last| last.strip_suffix(".kz"))
        .map(|s| s.to_string())
}

/// 获取文件路径的父目录前缀
/// `"std/time/TimeComponents.kz"` → `"std/time/"`
fn parent_directory(path: &str) -> String {
    match path.rfind('/') {
        Some(idx) => path[..=idx].to_string(),
        None => String::new(),
    }
}

/// 解析源码为 Module<'static>
///
/// source 和 path 必须是 'static（stdlib 的 include_str! 或 Box::leak 的用户源码）。
/// arena 通过 Box::leak 变为 'static，确保 Module<'static> 可安全缓存。
///
/// 返回 `Result`：解析成功返回 Module，致命解析错误返回 `ParseError`。
/// 非致命解析错误（parser 已恢复）通过 stderr 输出为警告，不阻断加载。
fn parse_source(path: &'static str, source: &'static str) -> Result<Module<'static>, ParseError> {
    // Box::leak arena：编译器进程内长期存活，退出时由 OS 回收
    let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));

    let mut lexer = Lexer::new(source);
    let mut sink = TokenCollector::new();
    lexer.tokenize_into(&mut sink);
    let tokens: Vec<Token> = sink.into_tokens();
    let tokens_ref = arena.alloc_slice_copy(&tokens);

    let mut parser = Parser::new(tokens_ref, arena, ErrorCollector::new());

    match parser.parse_module(path) {
        Ok(module) => {
            // 非致命 parse 错误（parser 已恢复）：输出警告，模块仍可用
            for err in parser.errors() {
                eprintln!(
                    "Warning: parse error in {} at {}:{}: {}",
                    path, err.line, err.column, err.message
                );
            }
            Ok(module)
        }
        Err(err) => Err(err),
    }
}

/// 收集模块的公开导出符号
///
/// 遍历 Module.declarations，收集所有 pub 可见性的函数/类型名称。
/// 用于后续 import 别名注册。
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

/// 从模块的 `pub pack <Name>` 声明中提取子模块名列表。
///
/// 目录模块的 `pack.kz` 通过 PackDecl 声明其包含的子模块。
/// 例如 `Store/pack.kz` 中的 `pub pack Memory` → 返回 `["Memory"]`。
/// `load_transitive_imports` 用此结果构造子模块路径（如 `["Store", "Memory"]`），
/// 确保子模块被加入 check 顺序。
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

// ─── ImportDecl 遍历辅助 ───────────────────────────────────────────

/// 遍历模块中的 ImportDecl，返回 (module_path, items) 列表
///
/// 用于编译入口在 check_module 前批量处理 import。
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
