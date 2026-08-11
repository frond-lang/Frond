//! Lint output formatting: human-readable and JSON.

use crate::tooling::common::Diagnostic::{Diagnostic, Severity};

/// Format diagnostics as human-readable text.
pub fn format_human(diagnostics: &[Diagnostic]) -> String {
    let mut out = String::new();
    let mut errors = 0;
    let mut warnings = 0;
    let mut advice = 0;

    for d in diagnostics {
        let label = match d.severity {
            Severity::Error => { errors += 1; "error" }
            Severity::Warning => { warnings += 1; "warning" }
            Severity::Advice => { advice += 1; "advice" }
        };
        out.push_str(&format!(
            "{}:{}:{}: {} {}: {}\n",
            d.source_file, d.range.start.line, d.range.start.col,
            d.code, label, d.message
        ));
    }

    if !diagnostics.is_empty() {
        out.push_str(&format!("\nfound {} error{}, {} warning{}, {} advice{}\n",
            errors, if errors == 1 { "" } else { "s" },
            warnings, if warnings == 1 { "" } else { "s" },
            advice, if advice == 1 { "" } else { "s" },
        ));
    }

    out
}

/// Format diagnostics as JSON.
pub fn format_json(diagnostics: &[Diagnostic]) -> String {
    let entries: Vec<String> = diagnostics.iter().map(|d| {
        format!(
            r#"{{"code":"{}","severity":"{}","file":"{}","line":{},"col":{},"message":"{}"}}"#,
            d.code,
            match d.severity {
                Severity::Error => "error",
                Severity::Warning => "warning",
                Severity::Advice => "advice",
            },
            d.source_file.replace('\\', "/"),
            d.range.start.line,
            d.range.start.col,
            d.message.replace('"', "\\\"")
        )
    }).collect();
    format!("[\n{}\n]", entries.join(",\n"))
}
