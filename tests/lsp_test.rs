use std::io::{BufReader, Cursor};
use kuzo::tooling::lsp::Server::LspTransport;

#[test]
fn test_lsp_initialize() {
    // This test verifies the LSP transport can parse a Content-Length message
    let input = "Content-Length: 58\r\n\r\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{}}";
    let reader = BufReader::new(Cursor::new(input.as_bytes().to_vec()));
    let mut writer_buf = Vec::new();
    let mut transport = LspTransport::new(reader, &mut writer_buf);

    let msg = transport.read_message().unwrap();
    assert!(msg.is_some());
    let msg = msg.unwrap();
    assert_eq!(msg["method"], "initialize");
}

#[test]
fn test_lsp_transport_roundtrip() {
    let input = "Content-Length: 2\r\n\r\n{}";
    let reader = BufReader::new(Cursor::new(input.as_bytes().to_vec()));
    let mut writer_buf = Vec::new();
    let mut transport = LspTransport::new(reader, &mut writer_buf);

    let msg = transport.read_message().unwrap();
    assert!(msg.is_some());

    // Write a message back
    let response = serde_json::json!({"jsonrpc": "2.0", "id": 1, "result": null});
    transport.write_message(&response).unwrap();

    let written = String::from_utf8(writer_buf).unwrap();
    assert!(written.starts_with("Content-Length: "));
    // serde_json sorts object keys alphabetically by default, so "id" precedes "jsonrpc"
    assert!(written.contains("\"jsonrpc\":\"2.0\""));
    assert!(written.contains("\"result\":null"));
}
