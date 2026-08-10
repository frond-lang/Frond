//! 模块加载的缓存条目与错误类型。

use rustc_hash::FxHashSet;

use crate::ast::Ast::Module;

/// 已加载的模块条目
pub struct LoadedModule {
    /// parse 产出的 AST 模块（'static 生命周期，可安全缓存）
    pub module: Module<'static>,
    /// 模块导出的公开符号（pub fun / pub type / pub val 的名称）
    pub exports: FxHashSet<String>,
}

/// 模块加载失败的原因
///
/// 所有加载失败（模块未找到 / 解析失败）均结构化记录到 `ModuleLoader::load_errors`，
/// 由调用方统一报告，避免错误被静默吞掉后引发 sema 级联误报。
#[derive(Debug, Clone)]
pub enum LoadError {
    /// 模块路径未找到（stdlib 嵌入表和文件系统搜索路径均未命中）
    ModuleNotFound { path: String },
    /// 模块源码解析失败（致命 parse 错误，AST 不可用）
    ParseFailed {
        path: String,
        line: u32,
        column: u32,
        message: String,
    },
    /// 循环导入检测到（A 导入 B，B 导入 A）
    CircularImport { path: String },
}

impl LoadError {
    /// 返回失败模块的路径
    pub fn path(&self) -> &str {
        match self {
            LoadError::ModuleNotFound { path } => path,
            LoadError::ParseFailed { path, .. } => path,
            LoadError::CircularImport { path } => path,
        }
    }
}
