//! Bakes a procedural cumulus density grid into `data/grids/cumulus_64.qvg`.
//!
//! The path tracer's PT-vdb pipeline loads this `.qvg` at startup and
//! uploads it as a 3-D texture. The real-world OpenVDB ingest milestone
//! will replace this generator with a pyopenvdb-driven converter; for
//! now the procedural shape is intentionally distinct from the simple
//! sphere+fbm in `pathtrace.wgsl` so the rendered cloud reads as a
//! recognisable cumulus rather than a fuzzy ball.

use std::fs;
use std::path::PathBuf;

use quasi::pathtrace::grid::bake_cumulus;

fn main() {
    let out_dir = PathBuf::from("data/grids");
    fs::create_dir_all(&out_dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", out_dir.display()));

    // 64³ resolution, half-radius bounds — the path tracer's
    // `Material::cloud_center` + `cloud_radius` rebase this into
    // world space at render time.
    let dims = [64_u32, 64, 64];
    let radius = 0.5_f32;
    let grid = bake_cumulus(dims, radius);

    let path = out_dir.join("cumulus_64.qvg");
    grid.save_to_path(&path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
    let nonzero = grid.voxels.iter().filter(|&&v| v > 0).count();
    let mean: f64 =
        grid.voxels.iter().map(|&v| v as f64).sum::<f64>() / grid.voxels.len() as f64 / 255.0;
    println!(
        "wrote {} ({} bytes, {dims:?}, mean density = {mean:.3}, non-zero voxels = {nonzero} / {})",
        path.display(),
        std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
        grid.voxels.len(),
    );
}
