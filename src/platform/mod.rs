//! Platform abstraction layer — the **only** place in the entire project where
//! `#[cfg(target_*)]` is permitted.
//!
//! All platform differences are confined to this module's submodules. When
//! business code calls into these functions, it is entirely agnostic to whether
//! the current platform is Unix or Windows.
//!
//! When adding a new platform, **only this module needs to change** — this is
//! the verifiable criterion that platform dispatch is centralized here.

pub mod AbiTable;
pub mod Dylib;
pub mod ResolveSelfSymbol;
