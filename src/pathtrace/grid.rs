//! Dense 3-D density grid — the data backing PT-vdb.
//!
//! On disk: `.qvg` ("quasi volume grid") — a tiny self-describing
//! binary format we own. Header pins dimensions + world-space
//! bounds; payload is `R8Unorm` voxel data in row-major
//! `(x, y, z)` order with x varying fastest.
//!
//! The path tracer uploads these as 3-D textures and samples in
//! `cloud_density` via trilinear interpolation; this module also
//! exposes a CPU mirror of that sampler so tests can pin the
//! analytic identities. The PT-vdb-ingest follow-up will add a
//! pyopenvdb-driven converter that produces `.qvg` from real
//! OpenVDB files.
//!
//! On-disk layout (little-endian throughout):
//! ```text
//! offset 0..4    : magic = b"QVG1"
//! offset 4..16   : dims u32 × 3 (w, h, d)
//! offset 16..28  : bounds_min f32 × 3
//! offset 28..40  : bounds_max f32 × 3
//! offset 40..44  : voxel_count u32 (must equal w * h * d)
//! offset 44..    : voxel data, w*h*d bytes
//! ```

use std::io::{self, Read, Write};
use std::path::Path;

const MAGIC: [u8; 4] = *b"QVG1";

/// Decode a `.qvg` from an embedded byte slice or fall back to a
/// 1×1×1 zero grid if the bytes don't parse. Used by the path
/// tracer to load the embedded default cumulus at startup.
pub fn from_bytes_or_empty(bytes: &[u8]) -> Grid3D {
    let mut cursor = std::io::Cursor::new(bytes);
    Grid3D::load(&mut cursor).unwrap_or_else(|_| {
        Grid3D::new([1, 1, 1], [0.0; 3], [1.0; 3])
    })
}

/// Load a `.qvg` from disk; on any I/O or parse failure, log a
/// warning and return the embedded default. The path-tracer
/// `--cloud-grid <path>` flag uses this so a typo in the CLI
/// degrades to the default rather than crashing the render.
pub fn load_from_path_or_default(path: &Path, default_bytes: &[u8]) -> Grid3D {
    match Grid3D::load_from_path(path) {
        Ok(g) => g,
        Err(e) => {
            log::warn!(
                "PT-vdb: failed to load grid from {}: {e}; falling back to embedded default",
                path.display(),
            );
            from_bytes_or_empty(default_bytes)
        }
    }
}

/// Dense 3-D density grid in CPU memory.
#[derive(Clone, Debug)]
pub struct Grid3D {
    pub dims: [u32; 3],
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub voxels: Vec<u8>,
}

impl Grid3D {
    pub fn new(dims: [u32; 3], bounds_min: [f32; 3], bounds_max: [f32; 3]) -> Self {
        let count = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        Self {
            dims,
            bounds_min,
            bounds_max,
            voxels: vec![0; count],
        }
    }

    pub fn voxel_count(&self) -> usize {
        (self.dims[0] as usize) * (self.dims[1] as usize) * (self.dims[2] as usize)
    }

    pub fn index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        let (w, h) = (self.dims[0] as usize, self.dims[1] as usize);
        (iz as usize) * w * h + (iy as usize) * w + (ix as usize)
    }

    pub fn set(&mut self, ix: u32, iy: u32, iz: u32, value: u8) {
        let i = self.index(ix, iy, iz);
        self.voxels[i] = value;
    }

    pub fn get(&self, ix: u32, iy: u32, iz: u32) -> u8 {
        self.voxels[self.index(ix, iy, iz)]
    }

    /// Trilinear sample at a normalised `uvw ∈ [0, 1]³`. Returns 0
    /// outside the unit cube. Matches the WGSL `textureSampleLevel`
    /// path byte-for-byte modulo float precision.
    pub fn sample_uvw(&self, uvw: [f32; 3]) -> f32 {
        if uvw.iter().any(|&c| !(0.0..=1.0).contains(&c)) {
            return 0.0;
        }
        let dims = [
            self.dims[0] as f32,
            self.dims[1] as f32,
            self.dims[2] as f32,
        ];
        // Texel-centred coords: same convention as wgpu's default
        // sampler (`uv * dims - 0.5`).
        let x = (uvw[0] * dims[0] - 0.5).max(0.0);
        let y = (uvw[1] * dims[1] - 0.5).max(0.0);
        let z = (uvw[2] * dims[2] - 0.5).max(0.0);
        let ix = (x.floor() as u32).min(self.dims[0] - 1);
        let iy = (y.floor() as u32).min(self.dims[1] - 1);
        let iz = (z.floor() as u32).min(self.dims[2] - 1);
        let ix1 = (ix + 1).min(self.dims[0] - 1);
        let iy1 = (iy + 1).min(self.dims[1] - 1);
        let iz1 = (iz + 1).min(self.dims[2] - 1);
        let fx = (x - ix as f32).clamp(0.0, 1.0);
        let fy = (y - iy as f32).clamp(0.0, 1.0);
        let fz = (z - iz as f32).clamp(0.0, 1.0);

        let v000 = self.get(ix, iy, iz) as f32;
        let v100 = self.get(ix1, iy, iz) as f32;
        let v010 = self.get(ix, iy1, iz) as f32;
        let v110 = self.get(ix1, iy1, iz) as f32;
        let v001 = self.get(ix, iy, iz1) as f32;
        let v101 = self.get(ix1, iy, iz1) as f32;
        let v011 = self.get(ix, iy1, iz1) as f32;
        let v111 = self.get(ix1, iy1, iz1) as f32;

        let x00 = v000 + (v100 - v000) * fx;
        let x10 = v010 + (v110 - v010) * fx;
        let x01 = v001 + (v101 - v001) * fx;
        let x11 = v011 + (v111 - v011) * fx;
        let y0 = x00 + (x10 - x00) * fy;
        let y1 = x01 + (x11 - x01) * fy;
        let z = y0 + (y1 - y0) * fz;
        z / 255.0
    }

    /// Convenience: world-space → normalised uvw → trilinear sample.
    pub fn sample_world(&self, world_pos: [f32; 3]) -> f32 {
        let mut uvw = [0.0; 3];
        for i in 0..3 {
            let span = (self.bounds_max[i] - self.bounds_min[i]).max(1e-30);
            uvw[i] = (world_pos[i] - self.bounds_min[i]) / span;
        }
        self.sample_uvw(uvw)
    }

    pub fn save_to_path(&self, path: &Path) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        self.save(&mut f)
    }

    pub fn save<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&MAGIC)?;
        for d in &self.dims {
            w.write_all(&d.to_le_bytes())?;
        }
        for b in &self.bounds_min {
            w.write_all(&b.to_le_bytes())?;
        }
        for b in &self.bounds_max {
            w.write_all(&b.to_le_bytes())?;
        }
        let count = self.voxel_count() as u32;
        w.write_all(&count.to_le_bytes())?;
        w.write_all(&self.voxels)?;
        Ok(())
    }

    pub fn load_from_path(path: &Path) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        Self::load(&mut f)
    }

    pub fn load<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0_u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad QVG magic: {magic:?}"),
            ));
        }
        let mut dims = [0_u32; 3];
        for d in &mut dims {
            let mut buf = [0_u8; 4];
            r.read_exact(&mut buf)?;
            *d = u32::from_le_bytes(buf);
        }
        let mut bounds_min = [0.0_f32; 3];
        for b in &mut bounds_min {
            let mut buf = [0_u8; 4];
            r.read_exact(&mut buf)?;
            *b = f32::from_le_bytes(buf);
        }
        let mut bounds_max = [0.0_f32; 3];
        for b in &mut bounds_max {
            let mut buf = [0_u8; 4];
            r.read_exact(&mut buf)?;
            *b = f32::from_le_bytes(buf);
        }
        let mut count_buf = [0_u8; 4];
        r.read_exact(&mut count_buf)?;
        let count = u32::from_le_bytes(count_buf) as usize;
        let expected = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
        if count != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("voxel count mismatch: header {count}, dims imply {expected}"),
            ));
        }
        let mut voxels = vec![0_u8; count];
        r.read_exact(&mut voxels)?;
        Ok(Grid3D {
            dims,
            bounds_min,
            bounds_max,
            voxels,
        })
    }
}

/// Procedural cumulus baker. Generates a 64³ density grid in
/// `[-radius, +radius]³` world-space around the origin (the caller
/// rebases via the `Material::cloud_center`). Anisotropic envelope
/// (wider than tall, flat bottom) plus 4-octave fbm; exact
/// formulas match the PT-cloud WGSL fbm so the visual style is
/// consistent.
pub fn bake_cumulus(dims: [u32; 3], radius: f32) -> Grid3D {
    let bounds_min = [-radius, -radius, -radius];
    let bounds_max = [radius, radius, radius];
    let mut grid = Grid3D::new(dims, bounds_min, bounds_max);
    let (w, h, d) = (dims[0], dims[1], dims[2]);
    for iz in 0..d {
        for iy in 0..h {
            for ix in 0..w {
                // World-space position for this voxel centre.
                let u = (ix as f32 + 0.5) / w as f32;
                let v = (iy as f32 + 0.5) / h as f32;
                let t = (iz as f32 + 0.5) / d as f32;
                let p = [
                    bounds_min[0] + u * (bounds_max[0] - bounds_min[0]),
                    bounds_min[1] + v * (bounds_max[1] - bounds_min[1]),
                    bounds_min[2] + t * (bounds_max[2] - bounds_min[2]),
                ];
                let density = cumulus_density(p, radius);
                grid.set(ix, iy, iz, (density.clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
    }
    grid
}

fn cumulus_density(p: [f32; 3], radius: f32) -> f32 {
    // Anisotropic envelope: wider on x/z, tighter on y, and shifted
    // upward so the cloud reads as flat-bottomed.
    let nx = p[0] / radius;
    let ny = (p[1] - 0.05 * radius) / (0.8 * radius);
    let nz = p[2] / radius;
    let r = (nx * nx + ny * ny + nz * nz).sqrt();
    if r >= 1.0 {
        return 0.0;
    }
    let envelope = smoothstep(1.0, 0.6, r);

    // Bottom falloff: sharp cutoff below y ≈ -0.4 · radius.
    let y_norm = p[1] / radius;
    let bottom = smoothstep(-0.4, -0.05, y_norm);

    // Vertical accent: emphasise the top hemisphere a touch so the
    // typical "cumulus puff" reads.
    let top_accent = smoothstep(-0.2, 0.5, y_norm);

    // 4-octave fbm using the same value-noise scheme as the WGSL
    // cloud helpers. Cheap and deterministic per position.
    let noise_freq = 3.5;
    let scaled = [
        p[0] * noise_freq,
        p[1] * noise_freq,
        p[2] * noise_freq,
    ];
    let n = fbm(scaled, 4);

    // Threshold + gain shape the puffiness. These values keep
    // typical interior density around ~0.4–0.7 (versus the previous
    // ~0.05 which read as too thin against `Material::scattering`).
    let threshold = 0.25;
    let gain = 3.5;
    let body = ((n - threshold) * gain).max(0.0);

    let density = envelope * bottom * body * (0.65 + 0.5 * top_accent);
    density.clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash3(p: [i32; 3]) -> u32 {
    let ux = (p[0].wrapping_add(73_856_093)) as u32;
    let uy = (p[1].wrapping_add(19_349_663)) as u32;
    let uz = (p[2].wrapping_add(83_492_791)) as u32;
    let mut h = ux
        .wrapping_mul(0x9e37_79b1)
        ^ uy.wrapping_mul(0x85eb_ca6b)
        ^ uz.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

fn value_at(p: [i32; 3]) -> f32 {
    hash3(p) as f32 / 4_294_967_296.0
}

fn value_noise(pos: [f32; 3]) -> f32 {
    let pf = [pos[0].floor(), pos[1].floor(), pos[2].floor()];
    let pi = [pf[0] as i32, pf[1] as i32, pf[2] as i32];
    let frac = [pos[0] - pf[0], pos[1] - pf[1], pos[2] - pf[2]];
    let s = [
        frac[0] * frac[0] * (3.0 - 2.0 * frac[0]),
        frac[1] * frac[1] * (3.0 - 2.0 * frac[1]),
        frac[2] * frac[2] * (3.0 - 2.0 * frac[2]),
    ];
    let c000 = value_at([pi[0], pi[1], pi[2]]);
    let c100 = value_at([pi[0] + 1, pi[1], pi[2]]);
    let c010 = value_at([pi[0], pi[1] + 1, pi[2]]);
    let c110 = value_at([pi[0] + 1, pi[1] + 1, pi[2]]);
    let c001 = value_at([pi[0], pi[1], pi[2] + 1]);
    let c101 = value_at([pi[0] + 1, pi[1], pi[2] + 1]);
    let c011 = value_at([pi[0], pi[1] + 1, pi[2] + 1]);
    let c111 = value_at([pi[0] + 1, pi[1] + 1, pi[2] + 1]);
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = mix(c000, c100, s[0]);
    let x10 = mix(c010, c110, s[0]);
    let x01 = mix(c001, c101, s[0]);
    let x11 = mix(c011, c111, s[0]);
    let y0 = mix(x00, x10, s[1]);
    let y1 = mix(x01, x11, s[1]);
    mix(y0, y1, s[2])
}

fn fbm(pos: [f32; 3], octaves: i32) -> f32 {
    let mut sum = 0.0_f32;
    let mut freq = 1.0_f32;
    let mut amp = 0.5_f32;
    let mut norm = 0.0_f32;
    for _ in 0..octaves {
        let p = [pos[0] * freq, pos[1] * freq, pos[2] * freq];
        sum += amp * value_noise(p);
        norm += amp;
        freq *= 2.0;
        amp *= 0.5;
    }
    sum / norm
}
