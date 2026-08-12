//! Project manifest (kuzo.toml) loading + path resolution.

use std::fs;
use std::process;

/// Project manifest file name.
pub const MANIFEST_NAME: &str = "kuzo.toml";
/// Default entry file.
pub const DEFAULT_ENTRY: &str = "src/Main.kz";
/// Default output directory.
pub const DEFAULT_OUTPUT_DIR: &str = "out";

/// Project manifest (deserialized via serde, TOML format).
///
/// Format:
/// ```toml
/// [package]
/// name = "myapp"           # required
/// entry = "src/Main.kz"  # optional, defaults to src/Main.kz
///
/// [build]
/// output_dir = "out"       # optional, defaults to "out"
/// opt_level = 2            # optional, defaults to 2
/// ```
#[derive(serde::Deserialize)]
pub struct Manifest {
    pub package: Package,
    #[serde(default)]
    pub build: Build,
}

#[derive(serde::Deserialize)]
pub struct Package {
    pub name: String,
    #[serde(default = "default_entry")]
    pub entry: String,
}

#[derive(serde::Deserialize, Default)]
pub struct Build {
    #[serde(default = "default_output_dir")]
    pub output_dir: String,
    #[serde(default = "default_opt_level")]
    pub opt_level: u8,
}

fn default_entry() -> String { DEFAULT_ENTRY.to_string() }
fn default_output_dir() -> String { DEFAULT_OUTPUT_DIR.to_string() }
fn default_opt_level() -> u8 { 2 }

/// Constructs an `OptLevel` from a CLI `u8` argument; out-of-range values are clamped to the valid range.
pub fn opt_level_from(v: Option<u8>) -> crate::pass::Optimizer::OptLevel {
    use crate::pass::Optimizer::OptLevel;
    match v {
        None => OptLevel::default(),
        Some(0) => OptLevel::O0,
        Some(1) => OptLevel::O1,
        Some(2) => OptLevel::O2,
        Some(3) => OptLevel::O3,
        Some(n) => {
            eprintln!("warning: opt-level {} out of range [0,3], clamped to 3", n);
            OptLevel::O3
        }
    }
}

/// Searches upward from the current directory for a directory containing the manifest file,
/// returning the project root path (up to 64 levels up).
pub fn find_project_root() -> Option<String> {
    let mut current = std::env::current_dir().ok()?;
    for _ in 0..64 {
        let manifest_path = current.join(MANIFEST_NAME);
        if manifest_path.exists() {
            return Some(current.to_string_lossy().into_owned());
        }
        if !current.pop() {
            break;
        }
    }
    None
}

/// Loads the project manifest: searches upward for the project root, then reads and parses kuzo.toml.
/// Exits with an error if no manifest is found (project-based).
pub fn load_manifest() -> (String, Manifest) {
    let root = find_project_root().unwrap_or_else(|| {
        eprintln!("error: not a Kuzo project (no {} found in current or parent directories)", MANIFEST_NAME);
        eprintln!("  hint: run `kuzo init` to scaffold a new project");
        process::exit(1);
    });
    let manifest_path = std::path::Path::new(&root).join(MANIFEST_NAME);
    let content = fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        eprintln!("error: could not read {}: {}", manifest_path.display(), e);
        process::exit(1);
    });
    let manifest: Manifest = toml::from_str(&content).unwrap_or_else(|e| {
        eprintln!("error: invalid {}: {}", manifest_path.display(), e);
        eprintln!("  hint: ensure [package] section exists with `name = \"...\"` under it");
        process::exit(1);
    });
    (root, manifest)
}

/// Resolves the entry file path: an explicit `file` takes priority; otherwise reads `entry` from the manifest (relative to the project root).
pub fn resolve_entry_path(file: Option<String>) -> String {
    match file {
        Some(f) => f,
        None => {
            let (root, manifest) = load_manifest();
            if std::path::Path::new(&manifest.package.entry).is_absolute() {
                manifest.package.entry
            } else {
                format!("{}/{}", root, manifest.package.entry)
            }
        }
    }
}
