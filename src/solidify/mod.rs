//! solidify — Frond binary executable format (`.kzo`)
//!
//! Persists the `DataFlowGraph` produced by the IR stage into a cross-platform
//! binary file (`.kzo`), supporting mmap zerocopy loading and giving Frond a
//! "source compilation -> artifact distribution -> runtime interpretation" workflow.
//!
//! Module composition:
//! - [`Spec`]: Format spec layer (constants, Header, Section, StringPool, CRC32, enum mappings)
//! - [`Format`]: Serialization/deserialization implementation (serialize/load/inspect)
//! - [`Accessors`]: Zerocopy accessor layer (DataFlowGraph accessor methods, mmap slice reading)
//! - [`Migration`]: Cross-version migration skeleton (reserved interface, no concrete migrations yet)

#![allow(non_snake_case)]

pub mod Spec;
pub mod Format;
pub mod Accessors;
pub mod Migration;
