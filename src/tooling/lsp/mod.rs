#![allow(non_snake_case)]
//! Lsp — JSON-RPC language server protocol implementation.
//!
//! Aggregates four submodules:
//! - [`Server`]: LSP transport + server loop (LspTransport / LspServer)
//! - [`DocState`]: document state tracking + symbol extraction
//! - [`Index`]: workspace indexing and query
//! - [`Handlers`]: LSP request/notification handlers

pub mod Server;
pub mod DocState;
pub mod Index;
pub mod Handlers;

pub use Server::LspTransport;
