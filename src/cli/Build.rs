//! build subcommand — compile only → out/<project_name>.kzo.

use std::fs;
use std::process;

use super::Manifest::{load_manifest, opt_level_from};
use super::Pipeline::compile_graph;

pub fn cmd_build(output: Option<String>, opt_level_cli: Option<u8>) {
    let (root, manifest) = load_manifest();
    // opt_level priority: CLI flag > manifest [build] opt_level > default O2
    let opt_level = opt_level_from(opt_level_cli.or(Some(manifest.build.opt_level)));

    let entry = if std::path::Path::new(&manifest.package.entry).is_absolute() {
        manifest.package.entry.clone()
    } else {
        format!("{}/{}", root, manifest.package.entry)
    };

    let graph = compile_graph(&entry, opt_level, false);

    // Output path: -o takes priority; otherwise output_dir/<project_name>.kzo
    let out_path = match output {
        Some(o) => o,
        None => {
            let dir = if std::path::Path::new(&manifest.build.output_dir).is_absolute() {
                manifest.build.output_dir.clone()
            } else {
                format!("{}/{}", root, manifest.build.output_dir)
            };
            // Ensure the output directory exists
            if let Err(e) = fs::create_dir_all(&dir) {
                eprintln!("error: could not create output directory '{}': {}", dir, e);
                process::exit(1);
            }
            format!("{}/{}.kzo", dir, manifest.package.name)
        }
    };

    // Serialize to .kzo
    let kzo_data = crate::solidify::Format::serialize_solidify(&graph);
    if let Err(e) = fs::write(&out_path, &kzo_data) {
        eprintln!("error: could not write '{}': {}", out_path, e);
        process::exit(1);
    }

    let size_kb = kzo_data.len() as f64 / 1024.0;
    eprintln!("Compiled {} → {} ({:.1} KB, {} nodes, {} subgraphs, opt-level {})",
        manifest.package.entry, out_path, size_kb,
        graph.nodes.len(), graph.subgraphs.len(), opt_level as u8);
}
