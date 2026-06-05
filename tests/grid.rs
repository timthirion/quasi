//! PT-vdb: integration tests for the dense `.qvg` density grid
//! format and the CPU mirror of the trilinear sampler.

use quasi::pathtrace::grid::{bake_cumulus, Grid3D};
use std::io::Cursor;

#[test]
fn qvg_exact_bytes_match_documented_format() {
    // Pins the on-disk format so Rust and Python can't silently
    // drift. The same input pinned in `scripts/test_qvg_writer.py`
    // produces exactly these bytes — if either side regresses,
    // its own test fails first.
    let mut g = Grid3D::new([3, 4, 5], [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
    for (i, v) in g.voxels.iter_mut().enumerate() {
        *v = i as u8;
    }
    let mut buf = Vec::new();
    g.save(&mut buf).unwrap();
    assert_eq!(&buf[0..4], b"QVG1");
    let read_u32 = |off: usize| u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
    let read_f32 = |off: usize| f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
    assert_eq!(read_u32(4), 3);
    assert_eq!(read_u32(8), 4);
    assert_eq!(read_u32(12), 5);
    assert_eq!(read_f32(16), -1.0);
    assert_eq!(read_f32(20), -2.0);
    assert_eq!(read_f32(24), -3.0);
    assert_eq!(read_f32(28), 1.0);
    assert_eq!(read_f32(32), 2.0);
    assert_eq!(read_f32(36), 3.0);
    assert_eq!(read_u32(40), 60);
    let payload: Vec<u8> = (0..60).collect();
    assert_eq!(&buf[44..], &payload[..]);
}

#[test]
fn qvg_round_trips_through_save_and_load() {
    let mut g = Grid3D::new([4, 5, 6], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    for (i, v) in g.voxels.iter_mut().enumerate() {
        *v = (i as u8).wrapping_mul(7);
    }
    let mut buf = Vec::new();
    g.save(&mut buf).unwrap();
    let mut cursor = Cursor::new(&buf);
    let h = Grid3D::load(&mut cursor).unwrap();
    assert_eq!(h.dims, g.dims);
    assert_eq!(h.bounds_min, g.bounds_min);
    assert_eq!(h.bounds_max, g.bounds_max);
    assert_eq!(h.voxels, g.voxels);
}

#[test]
fn sample_uvw_outside_returns_zero() {
    let g = Grid3D::new([4, 4, 4], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    for &p in &[
        [-0.1_f32, 0.5, 0.5],
        [0.5, 1.1, 0.5],
        [0.5, 0.5, -0.01],
        [2.0, 2.0, 2.0],
    ] {
        let v = g.sample_uvw(p);
        assert_eq!(v, 0.0, "expected 0 at {p:?}; got {v}");
    }
}

#[test]
fn sample_uvw_exact_voxel_centre_reads_back() {
    let mut g = Grid3D::new([4, 4, 4], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    g.set(1, 2, 3, 200);
    // Voxel (1, 2, 3) sits at the texel centre `(1.5, 2.5, 3.5) / 4`.
    let u = 1.5 / 4.0;
    let v = 2.5 / 4.0;
    let w = 3.5 / 4.0;
    let sampled = g.sample_uvw([u, v, w]);
    let expected = 200.0 / 255.0;
    assert!(
        (sampled - expected).abs() < 1e-6,
        "got {sampled}, expected {expected}",
    );
}

#[test]
fn sample_uvw_midpoint_averages_neighbours() {
    let mut g = Grid3D::new([2, 2, 2], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    // Pack a deterministic pattern that fits in u8 (0..=224).
    let mut values = [0_u8; 8];
    for i in 0..8_u8 {
        values[i as usize] = i * 32;
    }
    for iz in 0..2 {
        for iy in 0..2 {
            for ix in 0..2 {
                let idx = ix + iy * 2 + iz * 4;
                g.set(ix, iy, iz, values[idx as usize]);
            }
        }
    }
    let sampled = g.sample_uvw([0.5, 0.5, 0.5]);
    // At uv = (0.5, 0.5, 0.5), texel coords = `(0.5*2-0.5, ...) = (0.5, 0.5, 0.5)`
    // which sits exactly at the midpoint of all 8 voxels → 8-way average.
    let mean: f32 = values.iter().map(|&v| v as f32).sum::<f32>() / 8.0 / 255.0;
    assert!(
        (sampled - mean).abs() < 1e-5,
        "got {sampled}, expected mean {mean}",
    );
}

#[test]
fn sample_world_round_trips_through_bounds() {
    // Grid in world-space `[-2, +2]` on each axis. Set the centre
    // voxel to 100; sample world (0, 0, 0).
    let mut g = Grid3D::new([3, 3, 3], [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
    g.set(1, 1, 1, 100);
    let sampled = g.sample_world([0.0, 0.0, 0.0]);
    let expected = 100.0 / 255.0;
    assert!(
        (sampled - expected).abs() < 1e-6,
        "got {sampled}, expected {expected}",
    );
    assert_eq!(g.sample_world([-2.5, 0.0, 0.0]), 0.0, "out-of-bounds → 0");
}

#[test]
fn bake_cumulus_is_flat_bottomed() {
    // The cumulus baker emphasises top hemispheres and clips
    // bottom. Verify by comparing mean density of upper-half vs
    // lower-half voxels.
    let g = bake_cumulus([64, 64, 64], 0.5);
    let h = g.dims[1] as usize;
    let w = g.dims[0] as usize;
    let d = g.dims[2] as usize;
    let mut top_sum = 0.0_f64;
    let mut bot_sum = 0.0_f64;
    let mut top_n = 0_u32;
    let mut bot_n = 0_u32;
    for iz in 0..d {
        for iy in 0..h {
            for ix in 0..w {
                let v = g.get(ix as u32, iy as u32, iz as u32) as f64;
                if iy < h / 2 {
                    bot_sum += v;
                    bot_n += 1;
                } else {
                    top_sum += v;
                    top_n += 1;
                }
            }
        }
    }
    let top_mean = top_sum / top_n as f64;
    let bot_mean = bot_sum / bot_n as f64;
    assert!(
        top_mean > bot_mean * 1.5,
        "cumulus should be flat-bottomed: top mean {top_mean}, bottom mean {bot_mean}",
    );
}

#[test]
fn bake_cumulus_has_non_zero_voxels() {
    let g = bake_cumulus([32, 32, 32], 0.5);
    let nonzero = g.voxels.iter().filter(|&&v| v > 0).count();
    assert!(
        nonzero > g.voxels.len() / 20,
        "expected at least 5% non-zero voxels; got {nonzero} of {}",
        g.voxels.len(),
    );
}
