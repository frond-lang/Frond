#![allow(non_snake_case)]
//! ModuleLoader — 统一的模块加载器
//!
//! 合并 stdlib 和用户模块的加载逻辑：
//! - builtin 模块在初始化时全量预加载（默认可见，无需 import）
//! - std/user 模块按需通过 resolve_and_load 加载（遇到 ImportDecl 时触发）
//! - 模块缓存避免重复 parse/check
//!
//! ## 生命周期策略
//!
//! stdlib 源码是 `&'static str`（include_str!），用户模块源码通过 `Box::leak`
//! 转为 `&'static str`。Bump arena 同样通过 `Box::leak` 变为 `&'static`，
//! 因此所有 parse 产出的 `Module<'static>` 可安全缓存。
//! 编译器进程退出时由 OS 回收内存，无泄漏风险。
//!
//! ## 模块路径约定
//!
//! `import std.io.File` → module_path = ["std", "io", "File"]
//! → 解析为文件路径 "std/io/File.kz"
//! → 先查 stdlib 嵌入表，再查文件系统搜索路径
//!
//! ## 文件组织
//!
//! - [`StdlibEmbed`]：标准库源码嵌入表（BUILTIN_FILES / STD_FILES）与查询
//! - [`LoadError`]：模块缓存条目 LoadedModule 与加载错误类型
//! - [`Loader`]：ModuleLoader 主体（缓存、搜索路径、传递依赖加载）

pub mod Error;
pub mod Loader;
pub mod StdlibEmbed;

pub use Loader::{collect_imports, ModuleLoader};
pub use StdlibEmbed::{find, find_by_prefix, BUILTIN_FILES, STD_FILES, StdlibFile};
