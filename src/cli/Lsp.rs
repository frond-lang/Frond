//! lsp subcommand — start LSP server.

use crate::tooling::Lsp::Server::LspServer;

pub fn cmd_lsp() {
    let server = LspServer::new();
    server.run(); // never returns
}
