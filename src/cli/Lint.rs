//! lint subcommand — lint source code.

use std::fs;
use std::io::{self, Read};
use std::process;

use crate::tooling::Lint::{lint_file, LintConfig};
use crate::tooling::Lint::Report;

use super::Manifest::find_project_root;

pub fn cmd_lint(path: Option<String>, format: Option<String>, deny: Option<String>, stdin: bool) {
    let config = LintConfig::default();
    let format_str = format.as_deref().unwrap_or("human");

    let (diagnostics, _source_file) = if stdin {
        let mut source = String::new();
        if io::stdin().read_to_string(&mut source).is_err() {
            eprintln!("error: failed to read from stdin");
            process::exit(1);
        }
        let diags = lint_file_string("<stdin>", &source, &config);
        (diags, "<stdin>".to_string())
    } else {
        // find_project_root() returns the project root directory (containing frond.toml),
        // so join "src" directly to get the default lint target (same as fmt).
        let target = path.unwrap_or_else(|| {
            find_project_root()
                .map(|p| {
                    std::path::Path::new(&p)
                        .join("src")
                        .to_string_lossy()
                        .to_string()
                })
                .unwrap_or_else(|| "src".to_string())
        });
        let target_path = std::path::Path::new(&target);
        let all_diags = if target_path.is_file() {
            lint_file(&target, &config)
        } else if target_path.is_dir() {
            lint_dir(&target, &config)
        } else {
            eprintln!("error: path not found: {}", target);
            process::exit(1);
        };
        (all_diags, target)
    };

    // Output
    match format_str {
        "json" => println!("{}", Report::format_json(&diagnostics)),
        _ => print!("{}", Report::format_human(&diagnostics)),
    }

    // Exit code: 1 if any Error (or Warning when --deny warning)
    let has_error = diagnostics.iter().any(|d| match d.severity {
        crate::tooling::Common::Diagnostic::Severity::Error => true,
        crate::tooling::Common::Diagnostic::Severity::Warning => deny.as_deref() == Some("warning"),
        crate::tooling::Common::Diagnostic::Severity::Advice => false,
    });
    if has_error {
        process::exit(1);
    }
}

fn lint_dir(dir: &str, config: &LintConfig) -> Vec<crate::tooling::Common::Diagnostic::Diagnostic> {
    let mut all = Vec::new();
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return all,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            all.extend(lint_dir(&path.to_string_lossy(), config));
        } else if path.extension().map(|e| e == "kz").unwrap_or(false) {
            all.extend(lint_file(&path.to_string_lossy(), config));
        }
    }
    all
}

/// Lint a source string directly (for --stdin).
fn lint_file_string(filename: &str, source: &str, config: &LintConfig) -> Vec<crate::tooling::Common::Diagnostic::Diagnostic> {
    // Write to a temp file, then lint it
    let temp_path = format!("/tmp/lint_{}.kz", std::process::id());
    let _ = fs::write(&temp_path, source);
    let result = lint_file(&temp_path, config);
    let _ = fs::remove_file(&temp_path);
    // Fix source_file in diagnostics
    result.into_iter().map(|mut d| {
        d.source_file = filename.to_string();
        d
    }).collect()
}
