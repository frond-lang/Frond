//! 跨版本迁移骨架（预留接口，当前不实现具体迁移）
//!
//! 当 .resin 格式演进（schema_version 或 abi_version 升级）时，
//! 旧版本产物需要迁移到新版本才能加载。本模块预留迁移注册表接口。
//!
//! 设计：
//! - 每个迁移函数 `MigrationFn = fn(&[u8]) -> Result<Vec<u8>, String>`
//! - 注册表 `MIGRATIONS: [(from_schema, to_schema, MigrationFn)]`
//! - load 时若 schema_version 不匹配，查找从 file_version → current_version 的迁移链
//!
//! 当前状态：仅 schema v1，无历史版本，注册表为空。
//! 未来新增版本时，在此添加迁移函数并注册。

use super::Spec::{RESIN_SCHEMA_VERSION, RESIN_ABI_VERSION};

/// 迁移函数类型：输入旧版本 body 字节，输出新版本 body 字节
pub type MigrationFn = fn(&[u8]) -> Result<Vec<u8>, String>;

/// 迁移注册表项：(from_schema, to_schema, migration_fn)
/// 当前为空（仅 v1，无历史版本需要迁移）
pub const MIGRATIONS: &[(u16, u16, MigrationFn)] = &[];

/// 检查是否可以从给定 schema/abi 版本迁移到当前版本。
/// 返回 Ok(()) 表示可加载（版本匹配或有可用迁移链），Err 描述不可加载原因。
pub fn can_load(file_schema: u16, file_abi: u16) -> Result<(), String> {
    if file_schema == RESIN_SCHEMA_VERSION && file_abi == RESIN_ABI_VERSION {
        return Ok(());
    }
    // schema 不匹配：尝试查找迁移链
    if file_schema != RESIN_SCHEMA_VERSION {
        if find_migration_chain(file_schema, RESIN_SCHEMA_VERSION).is_some() {
            return Ok(());
        }
        return Err(format!(
            "schema version {} cannot be migrated to current {} (no migration chain registered)",
            file_schema, RESIN_SCHEMA_VERSION
        ));
    }
    // abi 不匹配：ABI 变更需要重新编译（无法迁移字节流）
    Err(format!(
        "ABI version {} is incompatible with current {} (recompile required)",
        file_abi, RESIN_ABI_VERSION
    ))
}

/// 查找从 from_schema 到 to_schema 的迁移链，返回迁移函数列表。
/// 当前注册表为空，总是返回 None。
fn find_migration_chain(from: u16, to: u16) -> Option<Vec<MigrationFn>> {
    if from == to {
        return Some(Vec::new());
    }
    // TODO: 实现 BFS 查找迁移链（当有多个历史版本时需要）
    // 当前注册表为空，无法迁移
    None
}

/// 应用迁移链（预留，当前注册表为空不会调用）
#[allow(dead_code)]
pub fn apply_migrations(data: &[u8], chain: &[MigrationFn]) -> Result<Vec<u8>, String> {
    let mut current = data.to_vec();
    for mig in chain {
        current = mig(&current)?;
    }
    Ok(current)
}
