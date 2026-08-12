//! init subcommand — scaffold a new project.

use std::fs;
use std::process;

use super::Manifest::{MANIFEST_NAME, DEFAULT_ENTRY, DEFAULT_OUTPUT_DIR};

pub fn cmd_init(name: Option<String>) {
    // target_dir is the full path; proj_name takes the basename as the project name
    let target_dir = name.as_deref().unwrap_or("");
    let proj_name = if target_dir.is_empty() {
        // No path specified: use the current directory name as the project name
        std::env::current_dir().ok()
            .and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "app".to_string())
    } else {
        // Path specified: take the basename as the project name
        std::path::Path::new(target_dir)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "app".to_string())
    };

    // Check target directory state
    if !target_dir.is_empty() {
        match fs::metadata(target_dir) {
            Ok(_) => {
                let manifest_path = format!("{}/{}", target_dir, MANIFEST_NAME);
                if fs::metadata(&manifest_path).is_ok() {
                    eprintln!("error: already a Kuzo project ({} contains {})", target_dir, MANIFEST_NAME);
                    process::exit(1);
                }
                if let Ok(entries) = fs::read_dir(target_dir) {
                    if entries.count() > 0 {
                        eprintln!("error: {} is not an empty directory", target_dir);
                        process::exit(1);
                    }
                }
            }
            Err(_) => {
                if let Err(e) = fs::create_dir_all(target_dir) {
                    eprintln!("error: could not create directory '{}': {}", target_dir, e);
                    process::exit(1);
                }
            }
        }
    }

    let manifest_path = if target_dir.is_empty() {
        MANIFEST_NAME.to_string()
    } else {
        format!("{}/{}", target_dir, MANIFEST_NAME)
    };
    let src_dir = if target_dir.is_empty() {
        "src".to_string()
    } else {
        format!("{}/src", target_dir)
    };

    if let Err(e) = fs::create_dir_all(&src_dir) {
        eprintln!("error: could not create directory '{}': {}", src_dir, e);
        process::exit(1);
    }

    let manifest_content = format!(
        "[package]\nname = \"{}\"\nentry = \"{}\"\n\n[build]\noutput_dir = \"{}\"\nopt_level = 2\n",
        proj_name, DEFAULT_ENTRY, DEFAULT_OUTPUT_DIR
    );
    if let Err(e) = fs::write(&manifest_path, manifest_content) {
        eprintln!("error: could not write '{}': {}", manifest_path, e);
        process::exit(1);
    }

    let main_path = if target_dir.is_empty() {
        DEFAULT_ENTRY.to_string()
    } else {
        format!("{}/{}", target_dir, DEFAULT_ENTRY)
    };
    // Console is under builtin/io and visible by default, so no import is needed.
    let main_content = "fun main(): void {\n    println(\"Hello, Kuzo!\")\n}\n";
    if let Err(e) = fs::write(&main_path, main_content) {
        eprintln!("error: could not write '{}': {}", main_path, e);
        process::exit(1);
    }

    println!("Created Kuzo project '{}'", proj_name);
    println!("  {}", manifest_path);
    println!("  {}", main_path);
}
