use kuzo::tooling::fmt::Engine::{format, FmtConfig};

#[test]
fn test_basic_format_preserves_code() {
    let source = "val x = 42\n";
    let config = FmtConfig::default();
    let result = format(source, &config);
    assert!(!result.is_empty());
    assert!(result.contains("val"));
    assert!(result.contains("x"));
    assert!(result.contains("42"));
}

#[test]
fn test_format_preserves_comments() {
    let source = "// This is a comment\nval x = 42\n";
    let config = FmtConfig::default();
    let result = format(source, &config);
    assert!(result.contains("// This is a comment"));
    assert!(result.contains("val x = 42"));
}

#[test]
fn test_format_idempotent() {
    let source = "val x = 42\n";
    let config = FmtConfig::default();
    let once = format(source, &config);
    let twice = format(&once, &config);
    assert_eq!(once, twice, "formatting should be idempotent");
}
