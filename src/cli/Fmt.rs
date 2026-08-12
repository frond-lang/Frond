//! fmt subcommand — format source code.

use std::fs;
use std::io::{self, Read};
use std::process;

use crate::tooling::Fmt::Engine::{format as fmt_source, FmtConfig};

use super::Manifest::find_project_root;

pub fn cmd_fmt(path: Option<String>, check: bool, stdin: bool) {
    let config = FmtConfig::default();

    if stdin {
        let mut source = String::new();
        if io::stdin().read_to_string(&mut source).is_err() {
            eprintln!("error: failed to read from stdin");
            process::exit(1);
        }
        let formatted = fmt_source(&source, &config);
        print!("{}", formatted);
        return;
    }

    // find_project_root() returns the project root *directory* (containing kuzo.toml),
    // so join "src" directly to get the default format target.
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
    let ok = if target_path.is_file() {
        format_file(target_path, &config, check)
    } else if target_path.is_dir() {
        format_dir(target_path, &config, check)
    } else {
        eprintln!("error: path not found: {}", target);
        process::exit(1);
    };

    if check && !ok {
        process::exit(1);
    }
}

fn format_file(path: &std::path::Path, config: &FmtConfig, check: bool) -> bool {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path.display(), e);
            return false;
        }
    };
    let formatted = fmt_source(&source, config);
    if check {
        if source != formatted {
            eprintln!("would reformat: {}", path.display());
            return false;
        }
        true
    } else {
        if source != formatted {
            if fs::write(path, &formatted).is_err() {
                eprintln!("error: cannot write {}", path.display());
                return false;
            }
        }
        true
    }
}

fn format_dir(dir: &std::path::Path, config: &FmtConfig, check: bool) -> bool {
    let mut all_ok = true;
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: cannot read directory {}: {}", dir.display(), e);
            return false;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            all_ok &= format_dir(&path, config, check);
        } else if path.extension().map(|e| e == "kz").unwrap_or(false) {
            all_ok &= format_file(&path, config, check);
        }
    }
    all_ok
}
