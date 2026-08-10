//! Cross-version migration skeleton (reserved interface, no concrete migrations yet).
//!
//! When the `.kzo` format evolves (schema_version or abi_version bumps), older
//! artifacts must be migrated to the new version before they can be loaded. This
//! module reserves a migration registry interface for that purpose.
//!
//! Design:
//! - Each migration function has type `MigrationFn = fn(&[u8]) -> Result<Vec<u8>, String>`.
//! - The registry is `MIGRATIONS: [(from_schema, to_schema, MigrationFn)]`.
//! - On load, if schema_version does not match, a migration chain from
//!   file_version -> current_version is looked up.
//!
//! Current state: only schema v1 exists, there are no historical versions, and
//! the registry is empty. When new versions are added in the future, register
//! migration functions here.

use super::Spec::{SOLIDIFY_SCHEMA_VERSION, SOLIDIFY_ABI_VERSION};

/// Migration function type: takes the old version body bytes and returns the new version body bytes.
pub type MigrationFn = fn(&[u8]) -> Result<Vec<u8>, String>;

/// Migration registry entry: `(from_schema, to_schema, migration_fn)`.
/// Currently empty (only v1, no historical versions to migrate).
pub const MIGRATIONS: &[(u16, u16, MigrationFn)] = &[];

/// Checks whether the given schema/abi version can be loaded into the current version.
/// Returns `Ok(())` when the artifact is loadable (versions match or a migration chain is
/// available), or an `Err` describing why it cannot be loaded.
pub fn can_load(file_schema: u16, file_abi: u16) -> Result<(), String> {
    if file_schema == SOLIDIFY_SCHEMA_VERSION && file_abi == SOLIDIFY_ABI_VERSION {
        return Ok(());
    }
    // Schema mismatch: try to find a migration chain.
    if file_schema != SOLIDIFY_SCHEMA_VERSION {
        if find_migration_chain(file_schema, SOLIDIFY_SCHEMA_VERSION).is_some() {
            return Ok(());
        }
        return Err(format!(
            "schema version {} cannot be migrated to current {} (no migration chain registered)",
            file_schema, SOLIDIFY_SCHEMA_VERSION
        ));
    }
    // ABI mismatch: an ABI change requires recompilation (the byte stream cannot be migrated).
    Err(format!(
        "ABI version {} is incompatible with current {} (recompile required)",
        file_abi, SOLIDIFY_ABI_VERSION
    ))
}

/// Looks up a migration chain from `from_schema` to `to_schema` and returns the list of
/// migration functions. The registry is currently empty, so this always returns `None`.
fn find_migration_chain(from: u16, to: u16) -> Option<Vec<MigrationFn>> {
    if from == to {
        return Some(Vec::new());
    }
    // TODO: implement a BFS migration-chain lookup (needed once there are multiple historical versions).
    // The registry is currently empty, so migration is impossible.
    None
}

/// Applies a migration chain (reserved; never called while the registry is empty).
#[allow(dead_code)]
pub fn apply_migrations(data: &[u8], chain: &[MigrationFn]) -> Result<Vec<u8>, String> {
    let mut current = data.to_vec();
    for mig in chain {
        current = mig(&current)?;
    }
    Ok(current)
}
