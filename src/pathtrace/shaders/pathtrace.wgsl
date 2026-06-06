// Cornell Box path tracer (NEE + MIS) over triangle meshes.
//
// T1: replaces the M0-M3 analytic-quad scene with storage-buffer
// triangle data loaded from glTF (pathtrace::mesh). Linear scan over
// triangles; T2 swaps the scan for a SAH BVH.
//
// Outputs four AOV color attachments per fragment (M2 MRT):
//   @location(0) radiance
//   @location(1) albedo   (first hit's albedo, or emission chromaticity)
//   @location(2) normal   (first hit's world-space normal in [-1, 1])
//   @location(3) depth    (first hit's t; alpha = 1.0 on hit, 0.0 on miss)
//
// Three samplers (PCG / Halton / Sobol) and two integrators (MIS+NEE /
// pure BSDF) dispatch at runtime off `U.sampler_kind` / `U.integrator_kind`.

const MAX_BOUNCES: i32 = 5;
// Total loop iterations including medium-boundary crossings, which
// don't count as real bounces (no BSDF eval, no scatter). Bounded
// generously above MAX_BOUNCES so a path can cross a couple of
// medium volumes without losing bounce budget.
const MAX_PATH_ITERATIONS: i32 = 32;
const PI: f32 = 3.14159265359;

const SAMPLER_PCG: u32 = 0u;
const SAMPLER_HALTON: u32 = 1u;
const SAMPLER_SOBOL: u32 = 2u;

const INTEGRATOR_MIS_NEE: u32 = 0u;
const INTEGRATOR_BSDF: u32 = 1u;

// BVH node leaf bit + traversal stack depth. The CPU side
// (`pathtrace::bvh`) pins the same constants; an out-of-sync value
// would silently corrupt traversal.
const LEAF_FLAG: u32 = 0x80000000u;
const LEAF_MASK: u32 = 0x7fffffffu;
const STACK_DEPTH: u32 = 32u;

// PT-textures: sentinel meaning "this material has no baseColorTexture;
// use `Material::albedo` as a constant." Matches `pathtrace::mesh::NO_TEXTURE`.
const NO_TEXTURE: u32 = 0xffffffffu;

// PT-beer-lambert: sentinel meaning "the ray is currently in vacuum
// / air — no participating-media attenuation." `path_trace` updates
// `current_medium` whenever the BSDF transmits through a dielectric
// interface.
const NO_MEDIUM: u32 = 0xffffffffu;

const U32_NORM: f32 = 4294967296.0;

struct Camera {
    position: vec3<f32>,
    fov: f32,
    direction: vec3<f32>,
    aspect: f32,
    up: vec3<f32>,
    _pad: f32,
};

struct Vertex {
    position: vec3<f32>,
    _pad0: f32,
    normal: vec3<f32>,
    _pad1: f32,
    uv: vec2<f32>,
    _pad2: vec2<f32>,
};

struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    emission: vec3<f32>,
    metallic: f32,
    base_color_texture_idx: u32,
    // PT-dielectrics: 0 = "not a dielectric"; > 0 routes the BSDF
    // onto the smooth-glass branch (sees Snell + Fresnel + TIR).
    ior: f32,
    // PT-mr-map: glTF metallic-roughness texture. G channel is
    // roughness, B channel is metallic. NO_TEXTURE = use the
    // scalar `roughness` + `metallic` instead.
    metallic_roughness_texture_idx: u32,
    // PT-normal-map: glTF normal map (tangent-space, +Y up).
    // NO_TEXTURE = use the geometric normal directly.
    normal_texture_idx: u32,
    // PT-beer-lambert: per-channel Beer-Lambert absorption coefficient
    // applied to throughput per unit of distance travelled *inside*
    // this material. `(0, 0, 0)` = no participating-media tinting.
    absorption: vec3<f32>,
    // PT-normal-map: scales the tangent-space XY components before
    // reconstruction. Mirrors glTF `normalTexture.scale`. 1.0 =
    // unscaled (the procedural maps); <1 softens the perturbation.
    normal_scale: f32,
    // PT-fog: per-channel scattering coefficient. Together with
    // `absorption` defines the medium's extinction `σ_t = σ_a + σ_s`
    // and scattering albedo `σ_s / σ_t`.
    scattering: vec3<f32>,
    // PT-hg: Henyey-Greenstein asymmetry parameter. `0` = isotropic
    // (PT-fog default); positive = forward-scattering (clouds);
    // negative = backward.
    phase_g: f32,
    // PT-cloud: procedural-cloud sphere centre + radius. When
    // `cloud_radius > 0`, the path tracer treats `absorption` and
    // `scattering` as MAXIMUM values and modulates them by an fbm
    // density inside the sphere via delta tracking. When zero, the
    // medium is homogeneous (PT-fog behaviour).
    cloud_center: vec3<f32>,
    cloud_radius: f32,
};

struct Uniforms {
    camera: Camera,
    triangle_count: u32,
    emissive_count: u32,
    frame_count: u32,
    viewport_width: u32,
    viewport_height: u32,
    sampler_kind: u32,
    integrator_kind: u32,
    use_bvh: u32,
    // PT-env: 1 if `env_texture` carries a real HDR map; 0 means the
    // stub 1×1 black pixel.
    has_environment: u32,
    env_width: u32,
    env_height: u32,
    _pad_env: u32,
};

struct BvhNode {
    aabb_min: vec3<f32>,
    left_or_first: u32,
    aabb_max: vec3<f32>,
    right_or_count: u32,
};

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> tri_indices: array<u32>;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read> tri_materials: array<u32>;
// PT-many-lights: each entry is `{ tri, _pad, cdf, _pad2 }`.
// `cdf` is the cumulative-power fraction; the WGSL inverse-CDF
// pick binary-searches this array.
struct EmissiveLight {
    tri: u32,
    _pad: u32,
    cdf: f32,
    _pad2: f32,
};
@group(0) @binding(5) var<storage, read> emissive_lights: array<EmissiveLight>;
@group(0) @binding(6) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(7) var<storage, read> bvh_tri_indices: array<u32>;
// PT-textures: scene base-color textures + a single shared linear-repeat
// sampler. Layer indices come from `Material::base_color_texture_idx`.
@group(0) @binding(8) var albedo_textures: texture_2d_array<f32>;
@group(0) @binding(9) var albedo_sampler: sampler;
// PT-vdb: 3-D density grid + clamp-to-edge linear sampler. R8Unorm
// so the trilinear sample is already in [0, 1].
@group(0) @binding(10) var cloud_grid: texture_3d<f32>;
@group(0) @binding(11) var cloud_grid_sampler: sampler;
// PT-env: equirectangular HDR env map (Rgba16Float) + linear sampler,
// plus a flat storage buffer with the importance-sampling tables. The
// tables are read only by `sample_env_importance` / `env_pdf_at_dir`
// (added in the NEE-on-env follow-up).
@group(0) @binding(12) var env_texture: texture_2d<f32>;
@group(0) @binding(13) var env_sampler: sampler;
@group(0) @binding(14) var<storage, read> env_data: array<f32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    out.uv = uv;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// ----- Samplers (same as M2/M3; the dispatch table is geometry-agnostic) -----

struct SamplerState {
    pcg: u32,
    sobol_index: u32,
    sobol_dim: u32,
    halton_index: u32,
    halton_dim: u32,
    pixel_seed: u32,
    scramble_x: u32,
    scramble_y: u32,
};

// PT-padded-sobol: maximum Sobol dimension we carry. Paths consuming
// more wrap modulo this number — the per-dim scramble makes the wrap
// unbiased. Mirrored as `MAX_SOBOL_DIM` in `pathtrace::sampler`.
const MAX_SOBOL_DIM: u32 = 32u;

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_pcg(s: ptr<function, SamplerState>) -> f32 {
    (*s).pcg = pcg_hash((*s).pcg);
    return f32((*s).pcg) / 4294967295.0;
}

fn radical_inverse(base: u32, n_in: u32) -> f32 {
    var n = n_in;
    let inv_base = 1.0 / f32(base);
    var inv_base_n = inv_base;
    var result: f32 = 0.0;
    for (var i = 0u; i < 32u && n > 0u; i = i + 1u) {
        let digit = f32(n % base);
        result = result + digit * inv_base_n;
        inv_base_n = inv_base_n * inv_base;
        n = n / base;
    }
    return result;
}

fn halton_base(dim: u32) -> u32 {
    switch (dim % 16u) {
        case 0u:  { return 2u; }
        case 1u:  { return 3u; }
        case 2u:  { return 5u; }
        case 3u:  { return 7u; }
        case 4u:  { return 11u; }
        case 5u:  { return 13u; }
        case 6u:  { return 17u; }
        case 7u:  { return 19u; }
        case 8u:  { return 23u; }
        case 9u:  { return 29u; }
        case 10u: { return 31u; }
        case 11u: { return 37u; }
        case 12u: { return 41u; }
        case 13u: { return 43u; }
        case 14u: { return 47u; }
        default:  { return 53u; }
    }
}

// PT-padded-sobol: Joe-Kuo direction-vector table for 32 dimensions.
// Computed at compile time on the CPU side (`pathtrace::sampler`) and
// transcribed here byte-for-byte. The Rust test
// `sobol_directions_first_vectors` pins the same values so any drift
// between the CPU and WGSL tables shows up immediately.
const SOBOL_DIRECTIONS: array<array<u32, 32>, 32> = array<array<u32, 32>, 32>(
    /* dim  0 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u, 0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u, 0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u, 0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u, 0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u, 0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u, 0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u, 0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u),
    /* dim  1 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u, 0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u, 0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u, 0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u, 0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u, 0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u, 0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u, 0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu),
    /* dim  2 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u, 0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u, 0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u, 0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u, 0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u, 0x6868e800u, 0x9c9c5c00u, 0xeeee8e00u, 0x5555c500u, 0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u, 0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u),
    /* dim  3 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u, 0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u, 0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u, 0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u, 0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u, 0xa0858800u, 0x914e5400u, 0xdbe79e00u, 0x25db6d00u, 0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u, 0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u),
    /* dim  4 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x20000000u, 0xb0000000u, 0xf8000000u, 0xdc000000u, 0x7a000000u, 0x9d000000u, 0x5a800000u, 0x2fc00000u, 0xa1600000u, 0xf0b00000u, 0xda880000u, 0x6fc40000u, 0x81620000u, 0x40bb0000u, 0x22878000u, 0xb3c9c000u, 0xfb65a000u, 0xddb2d000u, 0x78022800u, 0x9c0b3c00u, 0x5a0fb600u, 0x2d0ddb00u, 0xa2878080u, 0xf3c9c040u, 0xdb65a020u, 0x6db2d0b0u, 0x800228f8u, 0x400b3cdcu, 0x200fb67au, 0xb00ddb9du),
    /* dim  5 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x60000000u, 0x30000000u, 0xc8000000u, 0x24000000u, 0x56000000u, 0xfb000000u, 0xe0800000u, 0x70400000u, 0xa8600000u, 0x14300000u, 0x9ec80000u, 0xdf240000u, 0xb6d60000u, 0x8bbb0000u, 0x48008000u, 0x64004000u, 0x36006000u, 0xcb003000u, 0x2880c800u, 0x54402400u, 0xfe605600u, 0xef30fb00u, 0x7e48e080u, 0xaf647040u, 0x1eb6a860u, 0x9f8b1430u, 0xd6c81ec8u, 0xbb249f24u, 0x80d6d6d6u, 0x40bbbbbbu),
    /* dim  6 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0xa0000000u, 0xd0000000u, 0x58000000u, 0x94000000u, 0x3e000000u, 0xe3000000u, 0xbe800000u, 0x23c00000u, 0x1e200000u, 0xf3100000u, 0x46780000u, 0x67840000u, 0x78460000u, 0x84670000u, 0xc6788000u, 0xa784c000u, 0xd846a000u, 0x5467d000u, 0x9e78d800u, 0x33845400u, 0xe6469e00u, 0xb7673300u, 0x20f86680u, 0x104477c0u, 0xf8668020u, 0x4477c010u, 0x668020f8u, 0x77c01044u, 0x8020f866u, 0xc0104477u),
    /* dim  7 */ array<u32, 32>(0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u, 0x88000000u, 0x24000000u, 0x12000000u, 0x2d000000u, 0x76800000u, 0x9e400000u, 0x08200000u, 0x64100000u, 0xb2280000u, 0x7d140000u, 0xfea20000u, 0xba490000u, 0x1a248000u, 0x491b4000u, 0xc4b5a000u, 0xe3739000u, 0xf6800800u, 0xde400400u, 0xa8200a00u, 0x34100500u, 0x3a280880u, 0x59140240u, 0xeca20120u, 0x974902d0u, 0x6ca48768u, 0xd75b49e4u, 0xcc95a082u, 0x87639641u),
    /* dim  8 */ array<u32, 32>(0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u, 0x28000000u, 0xd4000000u, 0x6a000000u, 0x71000000u, 0x38800000u, 0x58400000u, 0xea200000u, 0x31100000u, 0x98a80000u, 0x08540000u, 0xc22a0000u, 0xe5250000u, 0xf2b28000u, 0x79484000u, 0xfaa42000u, 0xbd731000u, 0x18a80800u, 0x48540400u, 0x622a0a00u, 0xb5250500u, 0xdab28280u, 0xad484d40u, 0x90a426a0u, 0xcc731710u, 0x20280b88u, 0x10140184u, 0x880a04a2u, 0x84350611u),
    /* dim  9 */ array<u32, 32>(0x80000000u, 0x40000000u, 0xe0000000u, 0xb0000000u, 0x98000000u, 0x94000000u, 0x8a000000u, 0x5b000000u, 0x33800000u, 0xd9c00000u, 0x72200000u, 0x3f100000u, 0xc1b80000u, 0xa6ec0000u, 0x53860000u, 0x29f50000u, 0x0a3a8000u, 0x1b2ac000u, 0xd392e000u, 0x69ff7000u, 0xea380800u, 0xab2c0400u, 0x4ba60e00u, 0xfde50b00u, 0x60028980u, 0xf006c940u, 0x7834e8a0u, 0x241a75b0u, 0x123a8b38u, 0xcf2ac99cu, 0xb992e922u, 0x82ff78f1u),
    /* dim 10 */ array<u32, 32>(0x80000000u, 0x40000000u, 0xa0000000u, 0x10000000u, 0x08000000u, 0x6c000000u, 0x9e000000u, 0x23000000u, 0x57800000u, 0xadc00000u, 0x7fa00000u, 0x91d00000u, 0x49880000u, 0xced40000u, 0x880a0000u, 0x2c0f0000u, 0x3e0d8000u, 0x3317c000u, 0x5fb06000u, 0xc1f8b000u, 0xe18d8800u, 0xb2d7c400u, 0x1e106a00u, 0x6328b100u, 0xf7858880u, 0xbdc3c2c0u, 0x77ba63e0u, 0xfdf7b330u, 0xd7800df8u, 0xedc0081cu, 0xdfa0041au, 0x81d00a2du),
    /* dim 11 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x20000000u, 0x30000000u, 0x58000000u, 0xac000000u, 0x96000000u, 0x2b000000u, 0xd4800000u, 0x09400000u, 0xe2a00000u, 0x52500000u, 0x4e280000u, 0xc71c0000u, 0x629e0000u, 0x12670000u, 0x6e138000u, 0xf731c000u, 0x3a98a000u, 0xbe449000u, 0xf83b8800u, 0xdc2dc400u, 0xee06a200u, 0xb7239300u, 0x1aa80d80u, 0x8e5c0ec0u, 0xa03e0b60u, 0x703701b0u, 0x783b88c8u, 0x9c2dca54u, 0xce06a74au, 0x87239795u),
    /* dim 12 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0xa0000000u, 0x50000000u, 0xf8000000u, 0x8c000000u, 0xe2000000u, 0x33000000u, 0x0f800000u, 0x21400000u, 0x95a00000u, 0x5e700000u, 0xd8080000u, 0x1c240000u, 0xba160000u, 0xef370000u, 0x15868000u, 0x9e6fc000u, 0x781b6000u, 0x4c349000u, 0x420e8800u, 0x630bcc00u, 0xf7ad6a00u, 0xad739500u, 0x77800780u, 0x6d4004c0u, 0xd7a00420u, 0x3d700630u, 0x2f880f78u, 0xb1640ad4u, 0xcdb6077au, 0x824706d7u),
    /* dim 13 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u, 0x38000000u, 0xc4000000u, 0x42000000u, 0xa3000000u, 0xf1800000u, 0xaa400000u, 0xfce00000u, 0x85100000u, 0xe0080000u, 0x500c0000u, 0x58060000u, 0x54090000u, 0x7a038000u, 0x670c4000u, 0xb3842000u, 0x094a3000u, 0x0d6f1800u, 0x2f5aa400u, 0x1ce7ce00u, 0xd5145100u, 0xb8000080u, 0x040000c0u, 0x22000060u, 0x33000090u, 0xc9800038u, 0x6e4000c4u, 0xbee00042u, 0x261000a3u),
    /* dim 14 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x20000000u, 0xf0000000u, 0xa8000000u, 0x54000000u, 0x9a000000u, 0x9d000000u, 0x1e800000u, 0x5cc00000u, 0x7d200000u, 0x8d100000u, 0x24880000u, 0x71c40000u, 0xeba20000u, 0x75df0000u, 0x6ba28000u, 0x35d14000u, 0x4ba3a000u, 0xc5d2d000u, 0xe3a16800u, 0x91db8c00u, 0x79aef200u, 0x0cdf4100u, 0x672a8080u, 0x50154040u, 0x1a01a020u, 0xdd0dd0f0u, 0x3e83e8a8u, 0xaccacc54u, 0xd52d529au, 0xd91d919du),
    /* dim 15 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x20000000u, 0xd0000000u, 0xd8000000u, 0xc4000000u, 0x46000000u, 0x85000000u, 0xa5800000u, 0x76c00000u, 0xada00000u, 0x6ab00000u, 0x2da80000u, 0xaabc0000u, 0x0daa0000u, 0x7ab10000u, 0xd5a78000u, 0xbebd4000u, 0x93a3e000u, 0x3bb51000u, 0x3629b800u, 0x4d727c00u, 0x9b836200u, 0x27c4d700u, 0xb629b880u, 0x8d727cc0u, 0xbb836220u, 0xf7c4d7d0u, 0x6e29b858u, 0x49727c04u, 0xfd836266u, 0x72c4d755u),
    /* dim 16 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x20000000u, 0xf0000000u, 0x38000000u, 0x14000000u, 0xf6000000u, 0x67000000u, 0x8f800000u, 0x50400000u, 0x8aa00000u, 0x0ff00000u, 0x12a80000u, 0xabf40000u, 0xfcaa0000u, 0x28fb0000u, 0xbd298000u, 0x0bba4000u, 0x4e06e000u, 0x330c3000u, 0x59861800u, 0xc74d3400u, 0x3d2cb200u, 0x4bb2cb00u, 0x6e061880u, 0xc30d3440u, 0x618cb220u, 0xd342cbf0u, 0xcb2e18b8u, 0x2cb93454u, 0xe186b2d6u, 0x9349cb97u),
    /* dim 17 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x20000000u, 0xf0000000u, 0x68000000u, 0x64000000u, 0x36000000u, 0x6d000000u, 0x41800000u, 0xe0400000u, 0xd2e00000u, 0x9bf00000u, 0x0ce80000u, 0x52fc0000u, 0x5b6a0000u, 0x2fb30000u, 0xa00c8000u, 0x30054000u, 0x4807e000u, 0x940f9000u, 0x5e01f800u, 0x090e9400u, 0x778a5600u, 0x8d416b00u, 0x9369f880u, 0x7bb294c0u, 0xde005620u, 0xc9026bf0u, 0x578d78e8u, 0x7d4bd4a4u, 0xfb6db616u, 0x1fbefb9du),
    /* dim 18 */ array<u32, 32>(0x80000000u, 0x40000000u, 0xa0000000u, 0x50000000u, 0x98000000u, 0xf4000000u, 0xae000000u, 0xbb000000u, 0xe7800000u, 0x95c00000u, 0x1c200000u, 0xd0300000u, 0xdba80000u, 0x55f40000u, 0xff820000u, 0x21c10000u, 0x12238000u, 0x3b3a4000u, 0xa42b6000u, 0x3430f000u, 0x4da69800u, 0x4af3ec00u, 0x2e043a00u, 0xfb0a1f00u, 0x47851880u, 0xc5c9ac40u, 0x842f5aa0u, 0x243aef50u, 0x75a38018u, 0xeefa40b4u, 0x180b600eu, 0xb400f0ebu),
    /* dim 19 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0xe0000000u, 0xb0000000u, 0xb8000000u, 0x3c000000u, 0xce000000u, 0x41000000u, 0x21800000u, 0x51c00000u, 0x09600000u, 0x85700000u, 0xf2780000u, 0x8e9c0000u, 0x60020000u, 0x70030000u, 0x58038000u, 0x8c02c000u, 0x7602e000u, 0x7d00f000u, 0xef833800u, 0x10c10400u, 0x28e08600u, 0xd4b14700u, 0xfb182580u, 0x0bee15c0u, 0x9279c9e0u, 0xfe9d3a70u, 0x38000008u, 0xfc00000cu, 0x2e00000eu, 0xf100000bu),
    /* dim 20 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0xe0000000u, 0xd0000000u, 0x68000000u, 0x3c000000u, 0x8a000000u, 0x51000000u, 0xa9800000u, 0xddc00000u, 0x5ba00000u, 0x39d00000u, 0x95f80000u, 0x56d40000u, 0x0a020000u, 0x91030000u, 0x49838000u, 0x0dc34000u, 0x33a1a000u, 0x05d0f000u, 0x1ffa2800u, 0x07d54400u, 0xa380a600u, 0x4cc07700u, 0x1222ee80u, 0x3413a740u, 0xa65bf7e0u, 0x5305ab50u, 0x15f80008u, 0x96d4000cu, 0xea02000eu, 0x4103000du),
    /* dim 21 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x60000000u, 0xd0000000u, 0x38000000u, 0x8c000000u, 0x7e000000u, 0x71000000u, 0xc8800000u, 0x04c00000u, 0x1ba00000u, 0xbb700000u, 0x4a980000u, 0xc3bc0000u, 0xa6020000u, 0x6d010000u, 0xee818000u, 0x29c34000u, 0x9520e000u, 0x42b23000u, 0xe7b9f800u, 0x0d0dc400u, 0x3fb92200u, 0x110d1300u, 0x19bbee80u, 0x3c0cadc0u, 0x973a4a60u, 0xc5cf7ef0u, 0x3a180008u, 0x0b7c0004u, 0xa3a20006u, 0x7771000du),
    /* dim 22 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0xa0000000u, 0x90000000u, 0x08000000u, 0x64000000u, 0x6a000000u, 0x89000000u, 0xa5800000u, 0xcb400000u, 0x18200000u, 0xad900000u, 0xaf880000u, 0x72f40000u, 0x25820000u, 0x0b430000u, 0xb8228000u, 0x3d924000u, 0xa7882000u, 0x16f59000u, 0x4f83a800u, 0x82412400u, 0x1da01600u, 0xf6d16d00u, 0xbfa84080u, 0xbb672640u, 0xe0091620u, 0xf0b4efd0u, 0x38228008u, 0xfd92400cu, 0x0788200au, 0x86f59009u),
    /* dim 23 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x20000000u, 0xd0000000u, 0x48000000u, 0x8c000000u, 0xd6000000u, 0x39000000u, 0xd5800000u, 0x32400000u, 0xb2a00000u, 0x72100000u, 0x53d80000u, 0x82cc0000u, 0xcb820000u, 0x47430000u, 0x91208000u, 0xa9534000u, 0x7cf92000u, 0x4e9e3000u, 0xfcf95800u, 0x8e9fe400u, 0xdcf9d600u, 0x5e9c8900u, 0x94f96a80u, 0xd29fb840u, 0x42f9b760u, 0xeb9c9f30u, 0x97788008u, 0xd9df400cu, 0x25db2002u, 0xabcd300du),
    /* dim 24 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u, 0xd8000000u, 0xf4000000u, 0x3e000000u, 0x95000000u, 0x8f800000u, 0x3d400000u, 0xf3200000u, 0x2ef00000u, 0xadc80000u, 0x0a0c0000u, 0x8b220000u, 0x4af30000u, 0x6bc88000u, 0x3b0d4000u, 0xe2a16000u, 0x16b0d000u, 0x29687800u, 0xbdbf1400u, 0x33cb5e00u, 0x0f0c2500u, 0xfca1b480u, 0xd3b0afc0u, 0x7eeb6920u, 0x74fe4d30u, 0xfee87808u, 0xb4ff140cu, 0xdeeb5e02u, 0xe4fc2505u),
    /* dim 25 */ array<u32, 32>(0x80000000u, 0x40000000u, 0xa0000000u, 0xb0000000u, 0x98000000u, 0xa4000000u, 0x7a000000u, 0xd5000000u, 0x02800000u, 0x60400000u, 0x51e00000u, 0x88700000u, 0x8c280000u, 0x47c40000u, 0x0be20000u, 0xad710000u, 0xb6aa8000u, 0x3386c000u, 0xb8006000u, 0x54039000u, 0x42036800u, 0xc1019400u, 0xe0826a00u, 0x11431100u, 0x2960af80u, 0x3d3175c0u, 0xdf4a3aa0u, 0xaff49e10u, 0xd62b6808u, 0x62c59404u, 0x31606a0au, 0xd932110bu),
    /* dim 26 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0xa0000000u, 0x30000000u, 0x18000000u, 0x34000000u, 0x8a000000u, 0x9d000000u, 0x67800000u, 0x82400000u, 0x40e00000u, 0x60f00000u, 0x91480000u, 0x29440000u, 0x2d620000u, 0xbfb30000u, 0x162a8000u, 0xfbf4c000u, 0xe4ca6000u, 0xc207d000u, 0x2002a800u, 0xf001b400u, 0xb8037e00u, 0x04021900u, 0x92034b80u, 0xa90327c0u, 0xed81f320u, 0x1f40d810u, 0x27602808u, 0xe2b1740cu, 0xd1ab1e0au, 0x49b6c903u),
    /* dim 27 */ array<u32, 32>(0x80000000u, 0x40000000u, 0xa0000000u, 0xd0000000u, 0xa8000000u, 0x3c000000u, 0x7a000000u, 0x25000000u, 0xde800000u, 0xba400000u, 0xc4200000u, 0xae900000u, 0xc3980000u, 0x51840000u, 0xe8a20000u, 0x9dd10000u, 0xab3a8000u, 0x8c574000u, 0xe398a000u, 0xc185f000u, 0xe0a16800u, 0x71d2d400u, 0x793b5a00u, 0x95555900u, 0x471a5880u, 0x5ec7de40u, 0xfa011c60u, 0x65017b10u, 0x7e83e808u, 0x6a419404u, 0x6c21fa0au, 0x9291a90du),
    /* dim 28 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x20000000u, 0xf0000000u, 0x28000000u, 0xc4000000u, 0xee000000u, 0x6f000000u, 0xae800000u, 0x8ec00000u, 0x7f200000u, 0x57700000u, 0x92f80000u, 0x7d240000u, 0x12fa0000u, 0xbd270000u, 0x32fa8000u, 0x4d24c000u, 0x1afa2000u, 0x8927d000u, 0xf4f99800u, 0xe6266c00u, 0x5a7b2200u, 0x68e45700u, 0x255ade80u, 0x3f950ac0u, 0xb7a09560u, 0x42b0fe50u, 0xa55ade88u, 0xff950accu, 0x97a09562u, 0xb2b0fe5fu),
    /* dim 29 */ array<u32, 32>(0x80000000u, 0x40000000u, 0x60000000u, 0xf0000000u, 0x88000000u, 0x4c000000u, 0x7a000000u, 0x4b000000u, 0xe3800000u, 0x3f400000u, 0x3ca00000u, 0xb2b00000u, 0x67a80000u, 0x691c0000u, 0xfdaa0000u, 0x921d0000u, 0xf62b8000u, 0x115ec000u, 0x3889a000u, 0xa4eff000u, 0xc6a04800u, 0xb9b2dc00u, 0xe429c600u, 0xa65f2100u, 0x490ab480u, 0x6caf2bc0u, 0xeb808a20u, 0x33407fb0u, 0x26a2b488u, 0x09b32bc4u, 0x0c2a8a26u, 0x1a5d7fbfu),
    /* dim 30 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x20000000u, 0x30000000u, 0x68000000u, 0xec000000u, 0x72000000u, 0x5b000000u, 0x56800000u, 0x34c00000u, 0x37a00000u, 0xe5f00000u, 0x4ee80000u, 0x50ac0000u, 0xb86a0000u, 0x946f0000u, 0xc7ca8000u, 0xad9fc000u, 0x93232000u, 0x4a307000u, 0x0fc8e800u, 0xb19e1c00u, 0xa9233200u, 0xcd310f00u, 0x4348cc80u, 0x325ee8c0u, 0xba031f20u, 0x4702b670u, 0x6c824c88u, 0xb3c128ccu, 0x7b203f22u, 0x6632c673u),
    /* dim 31 */ array<u32, 32>(0x80000000u, 0xc0000000u, 0x60000000u, 0x30000000u, 0xc8000000u, 0x7c000000u, 0xe2000000u, 0xcb000000u, 0x46800000u, 0x0c400000u, 0x8b200000u, 0xe6300000u, 0x5d880000u, 0x73ec0000u, 0x530a0000u, 0xc3af0000u, 0x5a2b8000u, 0xde9fc000u, 0x8920a000u, 0xdd323000u, 0xb3092800u, 0x33ae1c00u, 0xf22bb200u, 0x929ded00u, 0xa3233e80u, 0x6a3345c0u, 0x1788e0a0u, 0xf4ef5670u, 0x3f88be88u, 0x78ec85ccu, 0x758840a6u, 0xffed6673u),
);

fn sobol_1d_raw(dim: u32, index: u32) -> u32 {
    let d = dim % MAX_SOBOL_DIM;
    var acc: u32 = 0u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if (((index >> i) & 1u) == 1u) {
            acc = acc ^ SOBOL_DIRECTIONS[d][i];
        }
    }
    return acc;
}

fn next_2d(s: ptr<function, SamplerState>) -> vec2<f32> {
    let kind = U.sampler_kind;
    if (kind == SAMPLER_HALTON) {
        let bx = halton_base((*s).halton_dim);
        let by = halton_base((*s).halton_dim + 1u);
        (*s).halton_dim = (*s).halton_dim + 2u;
        let sx = f32((*s).scramble_x) / U32_NORM;
        let sy = f32((*s).scramble_y) / U32_NORM;
        let rx = radical_inverse(bx, (*s).halton_index);
        let ry = radical_inverse(by, (*s).halton_index);
        return vec2<f32>(fract(rx + sx), fract(ry + sy));
    } else if (kind == SAMPLER_SOBOL) {
        // PT-padded-sobol: read dim `d` and `d+1`, each with its own
        // PCG-hashed scramble. `sobol_dim` advances per call so multi-2D
        // draws within one path consume independent Sobol axes.
        let d = (*s).sobol_dim;
        let scramble_x = pcg_hash((*s).pixel_seed + d);
        let scramble_y = pcg_hash((*s).pixel_seed + d + 1u);
        let x_raw = sobol_1d_raw(d, (*s).sobol_index) ^ scramble_x;
        let y_raw = sobol_1d_raw(d + 1u, (*s).sobol_index) ^ scramble_y;
        (*s).sobol_dim = (*s).sobol_dim + 2u;
        return vec2<f32>(f32(x_raw) / U32_NORM, f32(y_raw) / U32_NORM);
    } else {
        let a = rand_pcg(s);
        let b = rand_pcg(s);
        return vec2<f32>(a, b);
    }
}

fn next_1d(s: ptr<function, SamplerState>) -> f32 {
    return next_2d(s).x;
}

fn init_sampler(pixel: vec2<u32>, frame: u32, width: u32) -> SamplerState {
    var s: SamplerState;
    let pixel_seed = pixel.x + pixel.y * width;
    s.pcg = pcg_hash(pixel_seed + frame * 0x9e3779b9u);
    s.pixel_seed = pixel_seed;
    s.scramble_x = pcg_hash(pixel_seed);
    s.scramble_y = pcg_hash(s.scramble_x);
    // PT-padded-sobol: `sobol_index` is the sample-point index (one per
    // frame). `sobol_dim` starts at 0 and walks dimensions per call.
    s.sobol_index = frame + 1u;
    s.sobol_dim = 0u;
    s.halton_index = frame + 1u;
    s.halton_dim = 0u;
    return s;
}

// ----- Geometry helpers -----

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
};

struct TriVerts {
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
};

fn triangle_vertices(tri: u32) -> TriVerts {
    let i0 = tri_indices[tri * 3u + 0u];
    let i1 = tri_indices[tri * 3u + 1u];
    let i2 = tri_indices[tri * 3u + 2u];
    var t: TriVerts;
    t.v0 = vertices[i0].position;
    t.v1 = vertices[i1].position;
    t.v2 = vertices[i2].position;
    return t;
}

fn triangle_area(t: TriVerts) -> f32 {
    return 0.5 * length(cross(t.v1 - t.v0, t.v2 - t.v0));
}

// Möller-Trumbore intersection. Returns `vec3<f32>(t, u, v)` on hit
// (the `(u, v)` are the barycentric weights for the second and third
// vertex), or `(-1.0, 0.0, 0.0)` on miss. Double-sided — the path
// tracer doesn't cull backfaces.
fn intersect_triangle(ray: Ray, t: TriVerts, t_min: f32, t_max: f32) -> vec3<f32> {
    let edge1 = t.v1 - t.v0;
    let edge2 = t.v2 - t.v0;
    let h = cross(ray.dir, edge2);
    let a = dot(edge1, h);
    if (abs(a) < 1e-8) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let f = 1.0 / a;
    let s = ray.origin - t.v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let q = cross(s, edge1);
    let v = f * dot(ray.dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    let t_hit = f * dot(edge2, q);
    if (t_hit < t_min || t_hit > t_max) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }
    return vec3<f32>(t_hit, u, v);
}

struct Hit {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    tri: u32,
    mat: u32,
    hit: bool,
    // PT-dielectrics: the BSDF needs to know whether the ray is
    // entering (hit the geometric front face) or exiting (hit from
    // inside the medium). `normal` already gets flipped to face the
    // ray for shading convenience — front_face restores that info.
    // 1 = hit front face / entering; 0 = hit back face / exiting.
    front_face: u32,
};

/// Barycentric-interpolated UV at a triangle hit. The `(u, v)` come
/// straight out of `intersect_triangle`.
fn triangle_uv(tri: u32, u: f32, v: f32) -> vec2<f32> {
    let i0 = tri_indices[tri * 3u + 0u];
    let i1 = tri_indices[tri * 3u + 1u];
    let i2 = tri_indices[tri * 3u + 2u];
    let uv0 = vertices[i0].uv;
    let uv1 = vertices[i1].uv;
    let uv2 = vertices[i2].uv;
    return uv0 * (1.0 - u - v) + uv1 * u + uv2 * v;
}

/// Returns the Lambertian albedo to use at the hit — material's
/// constant `albedo` multiplied by the sampled `baseColorTexture`
/// when one is bound. `textureSampleLevel` (explicit lod = 0) so the
/// shader is portable to compute-style path tracers that don't have
/// fragment derivatives.
fn material_albedo(mat: Material, uv: vec2<f32>) -> vec3<f32> {
    if (mat.base_color_texture_idx == NO_TEXTURE) {
        return mat.albedo;
    }
    let tex = textureSampleLevel(
        albedo_textures,
        albedo_sampler,
        uv,
        i32(mat.base_color_texture_idx),
        0.0,
    );
    return mat.albedo * tex.rgb;
}

// PT-normal-map: per-triangle TBN. Computes a world-space tangent
// from triangle position + UV deltas (textbook derivation matches
// `compute_tangents` in `pathtrace::mesh`). Discontinuous across
// triangle edges — fine for low-poly mapped surfaces (e.g. our
// stone-tile floor), problematic on smooth meshes that would need
// per-vertex tangents. `apply_normal_map` Gram-Schmidts the
// tangent against the (geometric) normal so the TBN stays
// orthonormal even when the triangle is non-orthogonal in UV
// space.
struct Tbn {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    normal: vec3<f32>,
};

fn triangle_tangent_frame(tri: u32, normal: vec3<f32>) -> Tbn {
    let i0 = tri_indices[tri * 3u + 0u];
    let i1 = tri_indices[tri * 3u + 1u];
    let i2 = tri_indices[tri * 3u + 2u];
    let p0 = vertices[i0].position;
    let p1 = vertices[i1].position;
    let p2 = vertices[i2].position;
    let uv0 = vertices[i0].uv;
    let uv1 = vertices[i1].uv;
    let uv2 = vertices[i2].uv;
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let duv1 = uv1 - uv0;
    let duv2 = uv2 - uv0;
    let det = duv1.x * duv2.y - duv2.x * duv1.y;
    var tan: vec3<f32>;
    if (abs(det) < 1e-8) {
        // Degenerate UV — pick a stable axis-aligned tangent.
        if (abs(normal.x) < 0.9) {
            tan = vec3<f32>(1.0, 0.0, 0.0);
        } else {
            tan = vec3<f32>(0.0, 1.0, 0.0);
        }
    } else {
        let inv = 1.0 / det;
        tan = inv * (duv2.y * e1 - duv1.y * e2);
    }
    // Gram-Schmidt against the (already unit) normal.
    let proj = tan - normal * dot(tan, normal);
    let p_len = length(proj);
    let t_orth = select(
        normalize(proj),
        // Fallback when projection collapses (tangent ‖ normal).
        normalize(select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.x) >= 0.9)
                  - normal * dot(select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(normal.x) >= 0.9), normal)),
        p_len > 1e-6,
    );
    var out: Tbn;
    out.tangent = t_orth;
    out.bitangent = cross(normal, t_orth);
    out.normal = normal;
    return out;
}

// Tangent-space normal sample → world-space shading normal.
// Texture stored in OpenGL +Y-up convention (glTF mandated).
fn apply_normal_map(mat: Material, tri: u32, normal: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    if (mat.normal_texture_idx == NO_TEXTURE) {
        return normal;
    }
    let tex = textureSampleLevel(
        albedo_textures,
        albedo_sampler,
        uv,
        i32(mat.normal_texture_idx),
        0.0,
    );
    // [0, 1] → [-1, 1]. Scale the XY components by `normal_scale`.
    let ts = vec3<f32>(
        (tex.r * 2.0 - 1.0) * mat.normal_scale,
        (tex.g * 2.0 - 1.0) * mat.normal_scale,
        tex.b * 2.0 - 1.0,
    );
    let tbn = triangle_tangent_frame(tri, normal);
    let world = tbn.tangent * ts.x + tbn.bitangent * ts.y + tbn.normal * ts.z;
    return normalize(world);
}

// PT-mr-map: per-texel roughness + metallic, glTF 2.0 convention.
// Returns the **effective** scalars after texture multiply; when
// no MR texture is bound, the material's scalar fields pass
// through. A roughness floor of 0.04 prevents the perturbed
// micro-normal under MR-map streaking from producing fireflies
// in the GGX δ-function limit (standard PBR practice).
fn material_metallic_roughness(mat: Material, uv: vec2<f32>) -> vec2<f32> {
    var rough = mat.roughness;
    var metal = mat.metallic;
    if (mat.metallic_roughness_texture_idx != NO_TEXTURE) {
        let tex = textureSampleLevel(
            albedo_textures,
            albedo_sampler,
            uv,
            i32(mat.metallic_roughness_texture_idx),
            0.0,
        );
        rough = rough * tex.g;
        metal = metal * tex.b;
    }
    return vec2<f32>(max(rough, 0.04), metal);
}

// Record a triangle hit into the running closest. Factored so both
// the linear-scan and the BVH traversal use the same flip-normal
// convention. `bary_uv` is the (u, v) pair returned by
// `intersect_triangle` — the barycentric weights for the second and
// third vertex respectively.
fn record_hit(
    closest: ptr<function, Hit>,
    ray: Ray,
    tri: u32,
    verts: TriVerts,
    t_hit: f32,
    bary_uv: vec2<f32>,
) {
    (*closest).hit = true;
    (*closest).t = t_hit;
    (*closest).point = ray.origin + ray.dir * t_hit;
    (*closest).tri = tri;
    (*closest).mat = tri_materials[tri];
    (*closest).uv = triangle_uv(tri, bary_uv.x, bary_uv.y);
    let geom_n = normalize(cross(verts.v1 - verts.v0, verts.v2 - verts.v0));
    let front = dot(geom_n, ray.dir) < 0.0;
    var n = geom_n;
    if (!front) {
        n = -n;
    }
    (*closest).normal = n;
    if (front) {
        (*closest).front_face = 1u;
    } else {
        (*closest).front_face = 0u;
    }
}

fn trace_scene_linear(ray: Ray) -> Hit {
    var closest: Hit;
    closest.hit = false;
    closest.t = 1e30;
    for (var tri = 0u; tri < U.triangle_count; tri = tri + 1u) {
        let verts = triangle_vertices(tri);
        let hit = intersect_triangle(ray, verts, 0.001, closest.t);
        if (hit.x > 0.0) {
            record_hit(&closest, ray, tri, verts, hit.x, vec2<f32>(hit.y, hit.z));
        }
    }
    return closest;
}

fn occluded_linear(origin: vec3<f32>, dir: vec3<f32>, dist: f32) -> bool {
    let t_max = dist - 1e-3;
    var r: Ray;
    r.origin = origin;
    r.dir = dir;
    for (var tri = 0u; tri < U.triangle_count; tri = tri + 1u) {
        let m_idx = tri_materials[tri];
        let em = materials[m_idx].emission;
        if (em.x + em.y + em.z > 0.1) {
            continue;
        }
        let verts = triangle_vertices(tri);
        if (intersect_triangle(r, verts, 1e-3, t_max).x > 0.0) {
            return true;
        }
    }
    return false;
}

// ----- BVH traversal -----
//
// Slab-method AABB test using inverse direction. Handles infinities
// (axis-aligned rays) naturally because IEEE 754 0 × ∞ = NaN propagates
// out of the comparison and the test returns false safely.
fn intersect_aabb(ray_origin: vec3<f32>, inv_dir: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>, t_max: f32) -> bool {
    let t0 = (aabb_min - ray_origin) * inv_dir;
    let t1 = (aabb_max - ray_origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let tenter = max(max(tmin.x, tmin.y), tmin.z);
    let texit = min(min(tmax.x, tmax.y), tmax.z);
    return tenter <= texit && texit >= 0.0 && tenter < t_max;
}

fn trace_scene_bvh(ray: Ray) -> Hit {
    var closest: Hit;
    closest.hit = false;
    closest.t = 1e30;
    let inv_dir = vec3<f32>(1.0) / ray.dir;

    var stack: array<u32, 32>;
    stack[0] = 0u;
    var sp: i32 = 1;

    while (sp > 0) {
        sp = sp - 1;
        let node = bvh_nodes[stack[sp]];
        if (!intersect_aabb(ray.origin, inv_dir, node.aabb_min, node.aabb_max, closest.t)) {
            continue;
        }
        if ((node.left_or_first & LEAF_FLAG) != 0u) {
            let first = node.left_or_first & LEAF_MASK;
            let count = node.right_or_count;
            for (var i = 0u; i < count; i = i + 1u) {
                let tri = bvh_tri_indices[first + i];
                let verts = triangle_vertices(tri);
                let hit = intersect_triangle(ray, verts, 0.001, closest.t);
                if (hit.x > 0.0) {
                    record_hit(&closest, ray, tri, verts, hit.x, vec2<f32>(hit.y, hit.z));
                }
            }
        } else if (sp <= i32(STACK_DEPTH) - 2) {
            // Push both children. Near-far ordering optimisation is
            // left for a future plan — current Cornell scenes are
            // small enough that the savings don't matter.
            stack[sp] = node.right_or_count;
            sp = sp + 1;
            stack[sp] = node.left_or_first;
            sp = sp + 1;
        }
    }
    return closest;
}

fn occluded_bvh(origin: vec3<f32>, dir: vec3<f32>, dist: f32) -> bool {
    let t_max = dist - 1e-3;
    let inv_dir = vec3<f32>(1.0) / dir;
    var r: Ray;
    r.origin = origin;
    r.dir = dir;

    var stack: array<u32, 32>;
    stack[0] = 0u;
    var sp: i32 = 1;

    while (sp > 0) {
        sp = sp - 1;
        let node = bvh_nodes[stack[sp]];
        if (!intersect_aabb(origin, inv_dir, node.aabb_min, node.aabb_max, t_max)) {
            continue;
        }
        if ((node.left_or_first & LEAF_FLAG) != 0u) {
            let first = node.left_or_first & LEAF_MASK;
            let count = node.right_or_count;
            for (var i = 0u; i < count; i = i + 1u) {
                let tri = bvh_tri_indices[first + i];
                let m_idx = tri_materials[tri];
                let em = materials[m_idx].emission;
                if (em.x + em.y + em.z > 0.1) {
                    continue;
                }
                let verts = triangle_vertices(tri);
                if (intersect_triangle(r, verts, 1e-3, t_max).x > 0.0) {
                    return true;
                }
            }
        } else if (sp <= i32(STACK_DEPTH) - 2) {
            stack[sp] = node.right_or_count;
            sp = sp + 1;
            stack[sp] = node.left_or_first;
            sp = sp + 1;
        }
    }
    return false;
}

fn trace_scene(ray: Ray) -> Hit {
    if (U.use_bvh == 1u) {
        return trace_scene_bvh(ray);
    } else {
        return trace_scene_linear(ray);
    }
}

fn occluded(origin: vec3<f32>, dir: vec3<f32>, dist: f32) -> bool {
    if (U.use_bvh == 1u) {
        return occluded_bvh(origin, dir, dist);
    } else {
        return occluded_linear(origin, dir, dist);
    }
}

// PT-fog: shadow-ray transmittance from a point through any
// participating media to the target distance. Returns the per-
// channel attenuation (`vec3<f32>(1)` if nothing in the way, lower
// in any channel where the path crosses an absorbing or scattering
// medium, `vec3<f32>(0)` if blocked by an opaque surface).
//
// The traversal pattern: trace the nearest hit; if it's the light
// triangle the shadow ray was aimed at, we're done; if it's a
// medium-volume boundary, accumulate transmittance for the segment
// just travelled (in the *previous* medium), then advance past the
// boundary and continue with `current_medium` flipped. An opaque
// surface in between → return 0.
//
// Capped to a few boundary crossings — a single closed fog volume
// only needs two (enter, exit) per shadow ray. Heterogeneous /
// nested-media setups (a fog box with a glass sphere inside) would
// need to lift this cap.
const SHADOW_BOUNCE_CAP: i32 = 6;

// PT-cloud: per-segment medium transmittance. Homogeneous media
// use the closed-form `exp(-σ_t · t)`; heterogeneous (PT-cloud)
// media use ratio tracking — an unbiased Monte-Carlo estimate via
// `T *= 1 - σ_t(x_i) / σ_t_maj` at each null-collision step.
fn medium_transmittance_ratio_tracking(
    m: Material,
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_length: f32,
    s: ptr<function, SamplerState>,
) -> vec3<f32> {
    let sigma_t_max = medium_extinction(m);
    let sigma_t_maj = extinction_majorant(sigma_t_max);
    if (sigma_t_maj <= 0.0) {
        return vec3<f32>(1.0);
    }
    // Clip the loop to the ray's actual sphere-intersection range.
    // Outside the sphere `density == 0`, so iterating there only
    // wastes RNG samples and (because the per-pixel sample count
    // diverges) produces a visible bounding-box silhouette at
    // moderate spp.
    let range = cloud_sphere_range(origin, dir, m.cloud_center, m.cloud_radius);
    let t_start = max(0.0, range.x);
    let t_end = min(t_length, range.y);
    if (range.y <= 0.0 || t_start >= t_end) {
        return vec3<f32>(1.0);
    }
    var T = vec3<f32>(1.0);
    var t: f32 = t_start;
    for (var iter = 0; iter < HETERO_MAX_ITER; iter = iter + 1) {
        t = t - log(max(1.0 - next_1d(s), 1e-30)) / sigma_t_maj;
        if (t >= t_end) {
            return T;
        }
        let pos = origin + dir * t;
        let density = cloud_density(pos, m.cloud_center, m.cloud_radius);
        // density ∈ [0, ~1] doubles as σ_t(x) / σ_t_maj.
        T = T * (vec3<f32>(1.0) - vec3<f32>(density));
        let t_max_ch = max(T.x, max(T.y, T.z));
        if (t_max_ch < 1e-3) {
            return vec3<f32>(0.0);
        }
    }
    return T;
}

fn medium_segment_transmittance(
    medium: u32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    t_length: f32,
    s: ptr<function, SamplerState>,
) -> vec3<f32> {
    let m = materials[medium];
    if (m.cloud_radius > 0.0) {
        return medium_transmittance_ratio_tracking(m, origin, dir, t_length, s);
    }
    let sigma_t = medium_extinction(m);
    return exp(-sigma_t * t_length);
}

fn shadow_transmittance(
    origin: vec3<f32>,
    dir: vec3<f32>,
    dist: f32,
    start_medium: u32,
    s: ptr<function, SamplerState>,
) -> vec3<f32> {
    var trans = vec3<f32>(1.0);
    var medium = start_medium;
    var origin_cur = origin;
    var remaining = dist;
    for (var iter = 0; iter < SHADOW_BOUNCE_CAP; iter = iter + 1) {
        if (remaining <= 1e-3) {
            return trans;
        }
        var r: Ray;
        r.origin = origin_cur;
        r.dir = dir;
        let h = trace_scene(r);
        if (!h.hit || h.t > remaining - 1e-3) {
            // No occluder before the light — apply the final segment
            // transmittance and we're done.
            if (medium != NO_MEDIUM) {
                trans = trans * medium_segment_transmittance(
                    medium, origin_cur, dir, remaining, s);
            }
            return trans;
        }
        let mat = materials[h.mat];
        let em = mat.emission.x + mat.emission.y + mat.emission.z;
        if (em > 0.1) {
            // Aimed-at light triangle (or a coincidentally-aligned
            // other emitter). NEE treats this as "reached the
            // light" — apply the segment transmittance and stop.
            if (medium != NO_MEDIUM) {
                trans = trans * medium_segment_transmittance(
                    medium, origin_cur, dir, h.t, s);
            }
            return trans;
        }
        if (!is_medium_volume_material(mat)) {
            return vec3<f32>(0.0); // opaque occluder
        }
        // Medium boundary — attenuate the segment we just traversed
        // (in the previous medium), then flip across the boundary
        // and keep walking.
        if (medium != NO_MEDIUM) {
            trans = trans * medium_segment_transmittance(
                medium, origin_cur, dir, h.t, s);
        }
        if (h.front_face == 1u) {
            medium = h.mat;
        } else {
            medium = NO_MEDIUM;
        }
        origin_cur = h.point + dir * 1e-3;
        remaining = remaining - h.t - 1e-3;
    }
    return trans;
}

// ----- Light sampling + MIS -----

fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    let b2 = b * b;
    return a2 / (a2 + b2 + 1e-8);
}

struct LightSample {
    wi: vec3<f32>,
    dist: f32,
    pdf_w: f32,
    le: vec3<f32>,
    valid: bool,
};

// PT-many-lights: inverse-CDF emitter pick. Returns the index into
// `emissive_lights` whose CDF bin contains `xi`. Binary-search;
// O(log N). Caller is responsible for the early-out when
// `emissive_count == 0`.
fn pick_emissive(xi: f32) -> u32 {
    if (U.emissive_count <= 1u) {
        return 0u;
    }
    var lo: u32 = 0u;
    var hi: u32 = U.emissive_count - 1u;
    loop {
        if (hi <= lo + 1u) { break; }
        let mid = (lo + hi) >> 1u;
        if (emissive_lights[mid].cdf <= xi) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (emissive_lights[lo].cdf <= xi) {
        return lo + 1u;
    }
    return lo;
}

// PT-many-lights: power-fraction for the picked bin. Equivalent to
// `cdf[i] - cdf[i-1]` (or `cdf[0]` when `i == 0`).
fn emissive_pick_prob(i: u32) -> f32 {
    if (i == 0u) {
        return emissive_lights[0].cdf;
    }
    return emissive_lights[i].cdf - emissive_lights[i - 1u].cdf;
}

fn sample_light(p: vec3<f32>, s: ptr<function, SamplerState>) -> LightSample {
    var ls: LightSample;
    ls.valid = false;
    if (U.emissive_count == 0u) {
        return ls;
    }

    let xi = next_1d(s);
    let picked = pick_emissive(xi);
    let tri = emissive_lights[picked].tri;
    let pick_prob = emissive_pick_prob(picked);
    if (pick_prob <= 0.0) {
        return ls;
    }

    let verts = triangle_vertices(tri);
    let bary = next_2d(s);
    var u = bary.x;
    var v = bary.y;
    if (u + v > 1.0) {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    let x = verts.v0 + u * (verts.v1 - verts.v0) + v * (verts.v2 - verts.v0);

    let edge1 = verts.v1 - verts.v0;
    let edge2 = verts.v2 - verts.v0;
    let n_un = cross(edge1, edge2);
    let area = 0.5 * length(n_un);
    if (area < 1e-8) {
        return ls;
    }
    var n_l = normalize(n_un);

    let dvec = x - p;
    let dist = length(dvec);
    if (dist < 1e-4) {
        return ls;
    }
    let wi = dvec / dist;
    // Light face: flip the area normal toward the shaded point.
    if (dot(n_l, -wi) < 0.0) {
        n_l = -n_l;
    }
    let cos_l = dot(n_l, -wi);
    if (cos_l <= 0.0) {
        return ls;
    }

    ls.wi = wi;
    ls.dist = dist;
    // PT-many-lights solid-angle pdf:
    //   pdf_w = dist^2 / (cos_l * area * pick_prob)
    // `pick_prob` is the power-weighted probability of picking
    // this triangle. Reduces to `1 / N_emitters` when all
    // triangles share equal power (the uniform-pick limit).
    ls.pdf_w = (dist * dist) / (cos_l * area * pick_prob);
    ls.le = materials[tri_materials[tri]].emission;
    ls.valid = true;
    return ls;
}

// PT-many-lights: linear search for the emissive_lights entry
// whose `tri` matches `tri`. O(N) but N is small (single-digit
// emitters in our scenes) and this only fires on the MIS path
// when a BSDF ray hits an emissive triangle.
fn find_emissive_index(tri: u32) -> u32 {
    for (var i = 0u; i < U.emissive_count; i = i + 1u) {
        if (emissive_lights[i].tri == tri) {
            return i;
        }
    }
    // Not found — sentinel. Caller treats this as pick_prob = 0
    // (i.e. the triangle isn't in the emissive list, so the MIS
    // weight collapses to the BSDF branch).
    return 0xFFFFFFFFu;
}

fn light_pdf_solid_angle(p0: vec3<f32>, hit_point: vec3<f32>, hit_normal: vec3<f32>, tri: u32) -> f32 {
    let verts = triangle_vertices(tri);
    let area = triangle_area(verts);
    if (area < 1e-8) {
        return 0.0;
    }
    let dvec = hit_point - p0;
    let dist2 = dot(dvec, dvec);
    let dist = sqrt(dist2);
    if (dist < 1e-4) {
        return 0.0;
    }
    let wi = dvec / dist;
    let cos_l = abs(dot(hit_normal, -wi));
    if (cos_l < 1e-6) {
        return 0.0;
    }
    let idx = find_emissive_index(tri);
    if (idx == 0xFFFFFFFFu) {
        return 0.0;
    }
    let pick_prob = emissive_pick_prob(idx);
    if (pick_prob <= 0.0) {
        return 0.0;
    }
    return dist2 / (cos_l * area * pick_prob);
}

fn cosine_sample_hemisphere(normal: vec3<f32>, s: ptr<function, SamplerState>) -> vec3<f32> {
    let r = next_2d(s);
    let phi = 2.0 * PI * r.x;
    let cos_theta = sqrt(1.0 - r.y);
    let sin_theta = sqrt(r.y);

    let nrm = normalize(normal);
    var a: vec3<f32>;
    if (abs(nrm.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    let vv = normalize(cross(nrm, a));
    let uu = cross(nrm, vv);
    return normalize(uu * cos(phi) * sin_theta + vv * sin(phi) * sin_theta + nrm * cos_theta);
}

// ----- GGX microfacet BRDF (PT-ggx) -----
//
// The full conductor walk: Trowbridge-Reitz GGX D, Smith separable
// masking-shadowing G, Schlick Fresnel. Roughness 0 collapses to a
// perfect mirror (the δ-spike); we clamp `alpha` from below so the
// sample/pdf path stays finite. The CPU side
// (`pathtrace::ggx`) mirrors these formulas — the tests pin
// agreement.

const GGX_MIN_ALPHA: f32 = 0.0064;

fn ggx_alpha(roughness: f32) -> f32 {
    let r = max(roughness, sqrt(GGX_MIN_ALPHA));
    return r * r;
}

fn ggx_d(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn smith_g1(n_dot_x: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let denom = n_dot_x + sqrt(a2 + (1.0 - a2) * n_dot_x * n_dot_x);
    return 2.0 * n_dot_x / max(denom, 1e-8);
}

fn smith_g(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    return smith_g1(n_dot_v, alpha) * smith_g1(n_dot_l, alpha);
}

fn schlick_fresnel(v_dot_h: f32, f0: vec3<f32>) -> vec3<f32> {
    let s = pow(1.0 - v_dot_h, 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * s;
}

// Importance-sample a GGX half-vector in world space. Returns the
// world-space half vector — the caller reflects `wo` about it to get
// the outgoing direction.
fn sample_ggx_half(normal: vec3<f32>, alpha: f32, s: ptr<function, SamplerState>) -> vec3<f32> {
    let r = next_2d(s);
    let a2 = alpha * alpha;
    let cos_theta_2 = (1.0 - r.x) / (r.x * (a2 - 1.0) + 1.0);
    let cos_theta = sqrt(max(cos_theta_2, 0.0));
    let sin_theta = sqrt(max(1.0 - cos_theta_2, 0.0));
    let phi = 2.0 * PI * r.y;

    let nrm = normalize(normal);
    var a: vec3<f32>;
    if (abs(nrm.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    let vv = normalize(cross(nrm, a));
    let uu = cross(nrm, vv);
    return normalize(uu * cos(phi) * sin_theta + vv * sin(phi) * sin_theta + nrm * cos_theta);
}

// Solid-angle pdf for a GGX importance-sampled reflection direction.
// pdf_h = D(h) * (n · h); pdf_l = pdf_h / (4 (v · h)) — the Jacobian
// of the half-vector → outgoing-direction reflection.
fn ggx_pdf(n_dot_h: f32, v_dot_h: f32, alpha: f32) -> f32 {
    return ggx_d(n_dot_h, alpha) * n_dot_h / (4.0 * max(v_dot_h, 1e-6));
}

// ----- Procedural cloud density (PT-cloud) -----
//
// 3-D value noise with smoothstep trilinear interpolation, three
// octaves of fbm, smoothly windowed by a sphere defined by
// `Material::cloud_center` + `Material::cloud_radius`. The density
// function is deterministic per world-space position (the hash is
// keyed off integer lattice coords), so different paths through
// the same cloud see the same density field — required for an
// unbiased renderer.

const CLOUD_NOISE_FREQ: f32 = 4.0;
const CLOUD_OCTAVES: i32 = 4;
// Threshold + gain shape the cloud: smaller threshold + larger gain
// produces puffy "cumulus" structure; lower gain gives uniform
// haze. These values land somewhere between.
const CLOUD_NOISE_THRESHOLD: f32 = 0.2;
const CLOUD_NOISE_GAIN: f32 = 1.8;

fn cloud_hash3(p: vec3<i32>) -> u32 {
    let ux = u32(p.x + 73856093);
    let uy = u32(p.y + 19349663);
    let uz = u32(p.z + 83492791);
    var h: u32 = ux * 0x9e3779b1u
              ^ uy * 0x85ebca6bu
              ^ uz * 0xc2b2ae35u;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return h;
}

fn cloud_value_at(p: vec3<i32>) -> f32 {
    return f32(cloud_hash3(p)) / 4294967296.0;
}

fn cloud_value_noise(pos: vec3<f32>) -> f32 {
    let pf = floor(pos);
    let pi = vec3<i32>(pf);
    let frac = pos - pf;
    let s = frac * frac * (vec3<f32>(3.0) - 2.0 * frac);

    let c000 = cloud_value_at(pi + vec3<i32>(0, 0, 0));
    let c100 = cloud_value_at(pi + vec3<i32>(1, 0, 0));
    let c010 = cloud_value_at(pi + vec3<i32>(0, 1, 0));
    let c110 = cloud_value_at(pi + vec3<i32>(1, 1, 0));
    let c001 = cloud_value_at(pi + vec3<i32>(0, 0, 1));
    let c101 = cloud_value_at(pi + vec3<i32>(1, 0, 1));
    let c011 = cloud_value_at(pi + vec3<i32>(0, 1, 1));
    let c111 = cloud_value_at(pi + vec3<i32>(1, 1, 1));

    let x00 = mix(c000, c100, s.x);
    let x10 = mix(c010, c110, s.x);
    let x01 = mix(c001, c101, s.x);
    let x11 = mix(c011, c111, s.x);
    let y0 = mix(x00, x10, s.y);
    let y1 = mix(x01, x11, s.y);
    return mix(y0, y1, s.z);
}

fn cloud_fbm(pos: vec3<f32>) -> f32 {
    var sum: f32 = 0.0;
    var freq: f32 = 1.0;
    var amp: f32 = 0.5;
    var norm: f32 = 0.0;
    for (var i = 0; i < CLOUD_OCTAVES; i = i + 1) {
        sum = sum + amp * cloud_value_noise(pos * freq);
        norm = norm + amp;
        freq = freq * 2.0;
        amp = amp * 0.5;
    }
    return sum / norm;
}

// Normalised density at `pos` for the cloud defined by `(center,
// radius)`. Returns 0 outside the sphere, ramping up through the
// edge falloff toward the noise-modulated interior. Bounded in
// `[0, ~1.3]` (the noise term is in `[0, ~1]` after threshold +
// gain, edge falloff in `[0, 1]`).
// PT-vdb: world-space density via the baked `.qvg` grid texture.
// The grid was baked in `[-radius, +radius]³` around the origin;
// remap world-space into normalised `[0, 1]³` using the material's
// `cloud_center` + `cloud_radius` AABB. Outside the cube → 0
// (also enforced upstream by the sphere-clip in delta + ratio
// tracking).
fn cloud_density(pos: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let half = vec3<f32>(radius);
    let lo = center - half;
    let hi = center + half;
    let span = hi - lo;
    let uvw = (pos - lo) / span;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return 0.0;
    }
    return textureSampleLevel(cloud_grid, cloud_grid_sampler, uvw, 0.0).r;
}

// Analytic ray-sphere intersection used to clip volume tracking to
// the support of the cloud density (the sphere bounded by
// `cloud_radius`). Outside the sphere `density == 0`, so iterating
// through that range only burns RNG samples without affecting the
// estimator — and the sample-count mismatch between pixels whose
// rays cross the cloud bounding box but miss the sphere vs. those
// that don't show up as a hard square boundary at moderate spp.
//
// Returns `(t_enter, t_exit)`. When the ray misses the sphere
// outright, both come back negative; the caller checks `t_exit > 0`
// before doing any work.
fn cloud_sphere_range(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    radius: f32,
) -> vec2<f32> {
    let oc = ray_origin - center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if (disc <= 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    let sd = sqrt(disc);
    return vec2<f32>(-b - sd, -b + sd);
}

// ----- Participating media (PT-beer-lambert + PT-fog) -----
//
// Two helpers feed the volumetric loop in `path_trace`:
//
//   - `sample_volume_distance` — exponential inverse-CDF on the
//     scalar extinction majorant. Returns a `VolumeSample` whose
//     `weight` is the per-channel correction (the true vector
//     transmittance over the chosen sampling pdf). The `scattered`
//     flag tells the caller whether this segment ended in a
//     volume-scattering event or carried all the way to the next
//     surface hit.
//
//   - `shadow_transmittance` — replaces `occluded()` for NEE shadow
//     rays cast from inside a medium. Walks the ray through (up to
//     a fixed cap on) medium-volume boundaries, accumulating
//     `exp(-σ_t · t)` per segment; bails out and returns 0 on the
//     first opaque-surface hit.
//
// Pure-absorption dielectrics (PT-beer-lambert's glass bunny) still
// take the closed-form deterministic Beer-Lambert step. Sampling
// would converge to the same expectation but with substantially
// higher variance — see comment in `path_trace`.

struct VolumeSample {
    /// 1 when the segment terminated in a volume-scattering event;
    /// 0 when it reached the next surface hit unscattered.
    scattered: u32,
    /// Distance along the ray. For surface-hit endings this is the
    /// caller-supplied `t_max`.
    t: f32,
    /// Per-channel weight to multiply into `throughput`. For a
    /// volume scatter event this folds in `σ_s` and the inverse pdf;
    /// for a surface-hit ending it's the transmittance over the
    /// no-scatter probability.
    weight: vec3<f32>,
};

fn medium_extinction(medium_mat: Material) -> vec3<f32> {
    return medium_mat.absorption + medium_mat.scattering;
}

fn extinction_majorant(sigma_t: vec3<f32>) -> f32 {
    return max(sigma_t.x, max(sigma_t.y, sigma_t.z));
}

// Heterogeneous (PT-cloud) — delta tracking. Steps through the
// medium with the scalar majorant, rejecting "fictitious"
// collisions with probability `1 - σ_t(x) / σ_t_maj`. On a real
// collision: scatter with probability σ_s/σ_t (the single-
// scattering albedo) or absorb (path terminates). Iteration cap
// protects against pathological loops in extreme-density volumes.
const HETERO_MAX_ITER: i32 = 256;

fn sample_volume_distance_heterogeneous(
    m: Material,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t_max: f32,
    s: ptr<function, SamplerState>,
) -> VolumeSample {
    var out: VolumeSample;
    out.scattered = 0u;
    out.t = t_max;
    out.weight = vec3<f32>(1.0);
    let sigma_t_max = medium_extinction(m);
    let sigma_t_maj = extinction_majorant(sigma_t_max);
    if (sigma_t_maj <= 0.0) {
        return out;
    }
    // Single-scattering albedo at maximum density. For PT-cloud's
    // grey scenes this is the same as the local albedo at every
    // point; coloured media would want spectral MIS (deferred).
    var albedo_max: f32 = 0.0;
    if (sigma_t_max.x > 1e-30) {
        albedo_max = clamp(m.scattering.x / sigma_t_max.x, 0.0, 1.0);
    }
    // Clip delta tracking to the ray's actual sphere-intersection
    // range (see comment on `medium_transmittance_ratio_tracking`
    // for why — bounding-box silhouette artifact at moderate spp).
    let range = cloud_sphere_range(ray_origin, ray_dir, m.cloud_center, m.cloud_radius);
    let t_start = max(0.0, range.x);
    let t_end = min(t_max, range.y);
    if (range.y <= 0.0 || t_start >= t_end) {
        return out;
    }
    var t: f32 = t_start;
    for (var iter = 0; iter < HETERO_MAX_ITER; iter = iter + 1) {
        t = t - log(max(1.0 - next_1d(s), 1e-30)) / sigma_t_maj;
        if (t >= t_end) {
            return out;
        }
        let pos = ray_origin + ray_dir * t;
        let density = cloud_density(pos, m.cloud_center, m.cloud_radius);
        let p_real = density;  // density ∈ [0, ~1] doubles as σ_t/σ_t_maj
        if (next_1d(s) < p_real) {
            // Real collision. Decide scatter vs absorb on the
            // local single-scattering albedo (same as albedo_max
            // for grey media since σ_s and σ_a scale together).
            if (next_1d(s) < albedo_max) {
                out.scattered = 1u;
                out.t = t;
                out.weight = vec3<f32>(1.0);
                return out;
            }
            // Absorbed — path terminates.
            out.scattered = 0u;
            out.t = t;
            out.weight = vec3<f32>(0.0);
            return out;
        }
        // Null collision — continue stepping.
    }
    return out;
}

fn sample_volume_distance(
    medium: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    t_max: f32,
    s: ptr<function, SamplerState>,
) -> VolumeSample {
    var out: VolumeSample;
    out.scattered = 0u;
    out.t = t_max;
    out.weight = vec3<f32>(1.0);
    if (medium == NO_MEDIUM) {
        return out;
    }
    let m = materials[medium];
    if (m.cloud_radius > 0.0) {
        return sample_volume_distance_heterogeneous(m, ray_origin, ray_dir, t_max, s);
    }
    let sigma_t = medium_extinction(m);
    let sigma_t_maj = extinction_majorant(sigma_t);
    if (sigma_t_maj <= 0.0) {
        return out;
    }
    let xi = next_1d(s);
    let t_sample = -log(max(1.0 - xi, 1e-30)) / sigma_t_maj;
    if (t_sample < t_max) {
        let trans = exp(-sigma_t * t_sample);
        let pdf = sigma_t_maj * exp(-sigma_t_maj * t_sample);
        out.scattered = 1u;
        out.t = t_sample;
        out.weight = m.scattering * trans / pdf;
        return out;
    }
    // No scatter — survived all the way to the surface. Per-channel
    // transmittance over the (scalar) no-scatter probability.
    let trans = exp(-sigma_t * t_max);
    let pdf = exp(-sigma_t_maj * t_max);
    out.weight = trans / pdf;
    return out;
}

// Henyey-Greenstein phase function (PT-hg).
//
// `p(cos θ; g) = (1 - g²) / (4π · (1 + g² - 2g cos θ)^{3/2})`
//
// `g ∈ [-1, 1]`. `g = 0` is the isotropic phase (`1 / (4π)`);
// positive `g` peaks forward (cos θ → 1), negative peaks backward.
// `phase_hg_eval` is also the sample pdf — the phase function IS
// its own importance-sampling pdf (zonal symmetry collapses the
// problem to 1-D cosine inversion).

const PHASE_ISOTROPIC: f32 = 0.07957747154594767; // 1 / (4 * π)

fn phase_hg_eval(cos_theta: f32, g: f32) -> f32 {
    if (abs(g) < 1e-4) {
        return PHASE_ISOTROPIC;
    }
    let denom = 1.0 + g * g - 2.0 * g * cos_theta;
    return (1.0 - g * g) / (4.0 * PI * denom * sqrt(max(denom, 1e-30)));
}

// Sample a Henyey-Greenstein direction relative to the incoming
// direction `incoming` (unit vector along the ray that's about to
// scatter). Returns a unit world-space direction.
fn sample_hg_direction(incoming: vec3<f32>, g: f32, s: ptr<function, SamplerState>) -> vec3<f32> {
    let r = next_2d(s);
    var cos_theta: f32;
    if (abs(g) < 1e-4) {
        cos_theta = 1.0 - 2.0 * r.x;
    } else {
        let sqr = (1.0 - g * g) / (1.0 - g + 2.0 * g * r.x);
        cos_theta = clamp((1.0 + g * g - sqr * sqr) / (2.0 * g), -1.0, 1.0);
    }
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = 2.0 * PI * r.y;

    // Local orthonormal basis aligned to `incoming` (+z). Same
    // axis-swap trick as `cosine_sample_hemisphere`.
    let nrm = normalize(incoming);
    var a: vec3<f32>;
    if (abs(nrm.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    let vv = normalize(cross(nrm, a));
    let uu = cross(nrm, vv);
    return normalize(uu * cos(phi) * sin_theta + vv * sin(phi) * sin_theta + nrm * cos_theta);
}

// A "medium volume" material has no dielectric BSDF (ior == 0) but
// carries non-zero extinction (absorption or scattering). The ray
// passes through without surface scattering — only the
// `current_medium` toggle and the segment transmittance apply.
fn is_medium_volume_material(m: Material) -> bool {
    let has_ext = m.absorption.x + m.absorption.y + m.absorption.z
                + m.scattering.x + m.scattering.y + m.scattering.z > 0.0;
    return m.ior == 0.0 && has_ext;
}

// ----- Smooth dielectric (PT-dielectrics) -----
//
// Snell + the full unpolarised Fresnel equations. The "smooth" in
// the name is load-bearing: we treat the BSDF as a δ-function, so
// NEE shadow rays can't evaluate it (they get 0 from `eval_bsdf` /
// `bsdf_pdf` below). Visibility on a glass hit rides on the
// importance-sampled bounce hitting an emitter directly.
//
// Refraction is non-symmetric across the interface: the radiance
// changes by `(eta_t/eta_i)²` going into a denser medium (and the
// reverse coming out). We track that through `throughput *=
// (eta_i/eta_t)²` on the transmit branch — over a closed
// enter-then-exit path the factors cancel.

fn fresnel_dielectric(cos_theta_i: f32, eta_i: f32, eta_t: f32) -> f32 {
    let cti = clamp(cos_theta_i, 0.0, 1.0);
    let eta_ratio = eta_i / eta_t;
    let sin_t2 = eta_ratio * eta_ratio * max(1.0 - cti * cti, 0.0);
    if (sin_t2 >= 1.0) {
        return 1.0;
    }
    let cos_theta_t = sqrt(1.0 - sin_t2);
    let r_par = (eta_t * cti - eta_i * cos_theta_t)
              / (eta_t * cti + eta_i * cos_theta_t);
    let r_perp = (eta_i * cti - eta_t * cos_theta_t)
               / (eta_i * cti + eta_t * cos_theta_t);
    return 0.5 * (r_par * r_par + r_perp * r_perp);
}

// Snell. `wo` points away from the surface (towards the camera) and
// `n` is oriented into the incident side. Returns the refracted
// direction (also away from the surface, into the transmitted side),
// or a length-zero vector on total internal reflection — the caller
// is responsible for the TIR fallback.
fn refract_through(wo: vec3<f32>, n: vec3<f32>, eta_i: f32, eta_t: f32) -> vec3<f32> {
    let cos_i = dot(n, wo);
    let eta_ratio = eta_i / eta_t;
    let sin_t2 = eta_ratio * eta_ratio * max(1.0 - cos_i * cos_i, 0.0);
    if (sin_t2 >= 1.0) {
        return vec3<f32>(0.0);
    }
    let cos_t = sqrt(1.0 - sin_t2);
    return -wo * eta_ratio + n * (eta_ratio * cos_i - cos_t);
}

// ----- Unified BSDF dispatch -----
//
// `metallic > 0.5` picks GGX; else Lambertian. Both branches return
// the same `weight = f * cos(θ_l) / pdf_l` so the integrator code
// downstream is uniform. For Lambertian this collapses to `albedo`
// because (albedo/π) * cos / (cos/π) = albedo — bit-identical with
// the M3 path, by construction.

struct BsdfSample {
    wi: vec3<f32>,
    weight: vec3<f32>,
    pdf: f32,
    valid: bool,
};

fn sample_bsdf(
    mat: Material,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
    front_face: u32,
    s: ptr<function, SamplerState>,
) -> BsdfSample {
    var out: BsdfSample;
    out.valid = false;
    out.wi = vec3<f32>(0.0);
    out.weight = vec3<f32>(0.0);
    out.pdf = 0.0;

    // Smooth dielectric branch — fires first because dielectric
    // materials are typically also Lambertian-default-coloured
    // (metallic=0, albedo=white).
    if (mat.ior > 0.0) {
        // `normal` is already flipped to face `wo` (shading
        // convention from record_hit). `front_face = 1` means the
        // ray hit the geometric front (entering the medium); 0 means
        // it hit from inside (exiting). The shading normal is
        // already the incident-side normal in both cases.
        let n = normal;
        var eta_i: f32 = 1.0;
        var eta_t: f32 = mat.ior;
        if (front_face == 0u) {
            eta_i = mat.ior;
            eta_t = 1.0;
        }
        let cos_i = dot(n, wo);
        let fr = fresnel_dielectric(cos_i, eta_i, eta_t);
        let u = next_1d(s);
        if (u < fr) {
            // Reflect — F/F = 1, no eta² scaling.
            out.wi = reflect(-wo, n);
            out.weight = vec3<f32>(1.0);
            out.pdf = fr;
            out.valid = dot(n, out.wi) > 0.0;
        } else {
            let wi_t = refract_through(wo, n, eta_i, eta_t);
            if (length(wi_t) < 0.5) {
                // TIR — fall back to mirror reflection. fresnel
                // already returned 1.0 here, so in practice we
                // shouldn't reach this branch, but the guard keeps
                // floating-point edge cases honest.
                out.wi = reflect(-wo, n);
                out.weight = vec3<f32>(1.0);
                out.pdf = 1.0;
                out.valid = dot(n, out.wi) > 0.0;
            } else {
                let eta_ratio = eta_i / eta_t;
                // (1-F)/(1-F) * (eta_i/eta_t)² · albedo, where the
                // eta² accounts for the radiance change across the
                // interface. Albedo lets us tint glass without
                // mediating absorption.
                out.wi = wi_t;
                out.weight = albedo * vec3<f32>(eta_ratio * eta_ratio);
                out.pdf = 1.0 - fr;
                out.valid = true;
            }
        }
        return out;
    }

    if (mat.metallic > 0.5) {
        let alpha = ggx_alpha(mat.roughness);
        let h = sample_ggx_half(normal, alpha, s);
        let wi = reflect(-wo, h);
        let n_dot_l = dot(normal, wi);
        let n_dot_v = dot(normal, wo);
        let n_dot_h = dot(normal, h);
        let v_dot_h = dot(wo, h);
        if (n_dot_l <= 0.0 || n_dot_v <= 0.0 || n_dot_h <= 0.0 || v_dot_h <= 0.0) {
            return out;
        }
        let f = schlick_fresnel(v_dot_h, albedo);
        let g = smith_g(n_dot_v, n_dot_l, alpha);
        out.wi = wi;
        // f * cos(l) / pdf_l, with f = D*G*F / (4 n·v n·l) and
        //                     pdf_l = D * n·h / (4 v·h)
        //   → F * G * v·h / (n·v * n·h)
        out.weight = f * g * v_dot_h / (n_dot_v * n_dot_h);
        out.pdf = ggx_pdf(n_dot_h, v_dot_h, alpha);
        out.valid = true;
    } else {
        let wi = cosine_sample_hemisphere(normal, s);
        let cos_wi = max(dot(normal, wi), 0.0);
        out.wi = wi;
        out.weight = albedo;
        out.pdf = cos_wi / PI;
        out.valid = cos_wi > 0.0;
    }
    return out;
}

fn eval_bsdf(
    mat: Material,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    wo: vec3<f32>,
    wi: vec3<f32>,
) -> vec3<f32> {
    if (mat.ior > 0.0) {
        // δ-function BSDF — NEE shadow rays can't see it. The
        // BSDF-sample-then-direct-emission path handles glass
        // visibility.
        return vec3<f32>(0.0);
    }
    if (mat.metallic > 0.5) {
        let n_dot_l = dot(normal, wi);
        let n_dot_v = dot(normal, wo);
        if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
            return vec3<f32>(0.0);
        }
        let h = normalize(wo + wi);
        let n_dot_h = dot(normal, h);
        let v_dot_h = dot(wo, h);
        if (n_dot_h <= 0.0 || v_dot_h <= 0.0) {
            return vec3<f32>(0.0);
        }
        let alpha = ggx_alpha(mat.roughness);
        let d = ggx_d(n_dot_h, alpha);
        let g = smith_g(n_dot_v, n_dot_l, alpha);
        let f = schlick_fresnel(v_dot_h, albedo);
        return d * g * f / (4.0 * n_dot_v * n_dot_l);
    }
    return albedo / PI;
}

fn bsdf_pdf(mat: Material, normal: vec3<f32>, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    if (mat.ior > 0.0) {
        return 0.0;
    }
    if (mat.metallic > 0.5) {
        let n_dot_l = dot(normal, wi);
        let n_dot_v = dot(normal, wo);
        if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
            return 0.0;
        }
        let h = normalize(wo + wi);
        let n_dot_h = max(dot(normal, h), 0.0);
        let v_dot_h = max(dot(wo, h), 0.0);
        if (n_dot_h <= 0.0 || v_dot_h <= 0.0) {
            return 0.0;
        }
        let alpha = ggx_alpha(mat.roughness);
        return ggx_pdf(n_dot_h, v_dot_h, alpha);
    }
    return max(dot(normal, wi), 0.0) / PI;
}

// ----- Camera -----

fn get_camera_ray(uv_in: vec2<f32>, s: ptr<function, SamplerState>) -> Ray {
    let cam = U.camera;
    let j = next_2d(s) - vec2<f32>(0.5);
    let uv = uv_in + j * 0.001;

    let theta = cam.fov * PI / 180.0;
    let h = tan(theta / 2.0);
    let vh = 2.0 * h;
    let vw = cam.aspect * vh;

    let w = normalize(-cam.direction);
    let right = normalize(cross(cam.up, w));
    let up = cross(w, right);

    let horizontal = vw * right;
    let vertical = vh * up;
    let lower_left = cam.position - horizontal * 0.5 - vertical * 0.5 - w;
    let pixel_pos = lower_left + uv.x * horizontal + uv.y * vertical;

    var ray: Ray;
    ray.origin = cam.position;
    ray.dir = normalize(pixel_pos - ray.origin);
    return ray;
}

// ----- Environment map (PT-env) -----
//
// Equirectangular convention matches `pathtrace::env`:
//   φ = atan2(dir.z, dir.x); θ = acos(dir.y).
//   u = φ / 2π; v = θ / π.
//
// Returns black when no env map is bound (`has_environment == 0`).
fn env_radiance_at_dir(dir: vec3<f32>) -> vec3<f32> {
    if (U.has_environment == 0u) {
        return vec3<f32>(0.0);
    }
    let d = normalize(dir);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    var phi = atan2(d.z, d.x);
    if (phi < 0.0) {
        phi = phi + 2.0 * PI;
    }
    let u = phi / (2.0 * PI);
    let v = theta / PI;
    let rgba = textureSampleLevel(env_texture, env_sampler, vec2<f32>(u, v), 0.0);
    return rgba.rgb;
}

// --- PT-env: importance-sampling tables packed into `env_data`. ---
//
// Layout, matching `build_environment_resources` on the Rust side:
//   [0 .. H + 1]                        marginal_cdf  (H+1 entries)
//   [H + 1 .. H + 1 + H]                marginal_pdf  (H   entries)
//   [H+1+H .. H+1+H + (W+1) * H]        conditional_cdf  ((W+1) * H entries)
//   [.. + W * H]                        conditional_pdf  (W * H   entries)

fn env_marginal_cdf_at(i: u32) -> f32 {
    return env_data[i];
}
fn env_marginal_pdf_at(i: u32) -> f32 {
    return env_data[U.env_height + 1u + i];
}
fn env_conditional_cdf_at(y: u32, x: u32) -> f32 {
    let off = U.env_height + 1u + U.env_height;
    return env_data[off + y * (U.env_width + 1u) + x];
}
fn env_conditional_pdf_at(y: u32, x: u32) -> f32 {
    let off = U.env_height + 1u + U.env_height
            + (U.env_width + 1u) * U.env_height;
    return env_data[off + y * U.env_width + x];
}

// Binary search: largest `i` in [0, H-1] with marginal_cdf[i] <= xi.
fn env_invert_marginal(xi: f32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = U.env_height;
    loop {
        if (hi <= lo + 1u) { break; }
        let mid = (lo + hi) >> 1u;
        if (env_marginal_cdf_at(mid) <= xi) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Binary search inside a single row: largest `i` in [0, W-1] with
// conditional_cdf[row][i] <= xi.
fn env_invert_conditional(row: u32, xi: f32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = U.env_width;
    loop {
        if (hi <= lo + 1u) { break; }
        let mid = (lo + hi) >> 1u;
        if (env_conditional_cdf_at(row, mid) <= xi) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

struct EnvSample {
    dir: vec3<f32>,
    pdf: f32,
    le: vec3<f32>,
    valid: bool,
};

fn sample_env_importance(xi: vec2<f32>) -> EnvSample {
    var es: EnvSample;
    es.valid = false;
    if (U.has_environment == 0u) {
        return es;
    }
    let row = env_invert_marginal(xi.y);
    let col = env_invert_conditional(row, xi.x);
    let phi = (f32(col) + 0.5) / f32(U.env_width)  * 2.0 * PI;
    let theta = (f32(row) + 0.5) / f32(U.env_height) *       PI;
    let st = sin(theta);
    if (st < 1e-4) {
        return es;
    }
    let dir = vec3<f32>(st * cos(phi), cos(theta), st * sin(phi));
    let p_marginal = env_marginal_pdf_at(row);
    let p_cond     = env_conditional_pdf_at(row, col);
    let pdf = p_marginal * p_cond / (2.0 * PI * PI * st);
    if (pdf <= 0.0) {
        return es;
    }
    let u = (f32(col) + 0.5) / f32(U.env_width);
    let v = (f32(row) + 0.5) / f32(U.env_height);
    let rgba = textureSampleLevel(env_texture, env_sampler, vec2<f32>(u, v), 0.0);
    es.dir = dir;
    es.pdf = pdf;
    es.le = rgba.rgb;
    es.valid = true;
    return es;
}

// PDF (solid-angle measure) of `dir` under env importance sampling.
// Snaps the direction to the nearest texel via floor — matches the
// `pdf_at_direction` quantisation in the CPU mirror.
fn env_pdf_at_dir(dir: vec3<f32>) -> f32 {
    if (U.has_environment == 0u) {
        return 0.0;
    }
    let d = normalize(dir);
    let theta = acos(clamp(d.y, -1.0, 1.0));
    var phi = atan2(d.z, d.x);
    if (phi < 0.0) {
        phi = phi + 2.0 * PI;
    }
    let st = sin(theta);
    if (st < 1e-4) {
        return 0.0;
    }
    let u = phi / (2.0 * PI);
    let v = theta / PI;
    var x = u32(u * f32(U.env_width));
    var y = u32(v * f32(U.env_height));
    if (x >= U.env_width)  { x = U.env_width  - 1u; }
    if (y >= U.env_height) { y = U.env_height - 1u; }
    let p_marginal = env_marginal_pdf_at(y);
    let p_cond     = env_conditional_pdf_at(y, x);
    return p_marginal * p_cond / (2.0 * PI * PI * st);
}

// ----- Integrator -----

struct Sample {
    radiance: vec3<f32>,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    hit: bool,
};

fn path_trace(ray_in: Ray, s: ptr<function, SamplerState>) -> Sample {
    var result: Sample;
    result.radiance = vec3<f32>(0.0);
    result.albedo = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0);
    result.depth = 0.0;
    result.hit = false;

    var ray = ray_in;
    var throughput = vec3<f32>(1.0);
    var specular_bounce = true;
    var prev_bsdf_pdf = 0.0;
    var prev_point = ray.origin;
    var current_medium: u32 = NO_MEDIUM;
    let mis_nee_mode = U.integrator_kind == INTEGRATOR_MIS_NEE;

    // `bounce` counts REAL light bounces (surface BSDF + volume
    // scattering). `iter` counts everything including medium-volume
    // boundary crossings (which don't sample a direction or
    // attenuate throughput on their own — they're just bookkeeping).
    // Keeping the two separate matters: without it, a camera ray
    // that enters and exits a cloud bounding box "spends" two of its
    // MAX_BOUNCES on doing nothing, so the in-silhouette pixels get
    // ~2 fewer indirect-light bounces than out-of-silhouette pixels
    // and the bounding-box silhouette reads as a visible darker
    // square in the rendered image.
    var bounce: i32 = 0;
    var iter: i32 = 0;
    loop {
        if (bounce >= MAX_BOUNCES || iter >= MAX_PATH_ITERATIONS) {
            break;
        }
        iter = iter + 1;
        var hit = trace_scene(ray);
        if (!hit.hit) {
            // PT-env: a missed ray escapes into the env dome. The
            // BSDF-sample path collects env emission with a MIS
            // weight against the NEE env-importance pdf, matching the
            // triangle-light MIS pattern (`emit > 0.1` branch below).
            // Camera rays (bounce == 0) and post-specular bounces
            // take the full unweighted contribution.
            if (U.has_environment == 1u) {
                let env_rad = env_radiance_at_dir(ray.dir);
                var wmis = 1.0;
                if (mis_nee_mode && bounce > 0 && !specular_bounce) {
                    let env_p = env_pdf_at_dir(ray.dir);
                    if (env_p > 0.0) {
                        wmis = power_heuristic(prev_bsdf_pdf, env_p);
                    }
                }
                result.radiance = result.radiance + throughput * env_rad * wmis;
                if (bounce == 0) {
                    result.hit = true;
                    result.depth = 1e6;
                    result.normal = -ray.dir;
                    result.albedo = env_rad;
                }
            }
            break;
        }
        // PT-beer-lambert + PT-fog: handle the segment we just
        // traversed inside `current_medium`. Two regimes:
        //
        //   1. Pure absorption (σ_s = 0, e.g. the green-glass
        //      bunny) — closed-form deterministic Beer-Lambert
        //      step. Sampling a distance here would converge to
        //      the same expectation but with massive variance
        //      (scatter events are δ-spikes with zero σ_s weight,
        //      so the path "dies" ~half the time).
        //   2. With scattering (σ_s > 0, fog) — sample a distance
        //      from the exponential CDF. If the sample falls
        //      *inside* the segment, scatter event: NEE + phase-
        //      function bounce + continue. If it overshoots, just
        //      apply the unbiased weight and proceed to the
        //      surface hit.
        if (current_medium != NO_MEDIUM) {
            let medium_mat = materials[current_medium];
            let sigma_s_sum = medium_mat.scattering.x
                            + medium_mat.scattering.y
                            + medium_mat.scattering.z;
            if (sigma_s_sum <= 0.0) {
                throughput = throughput * exp(-medium_mat.absorption * hit.t);
            } else {
                let vs = sample_volume_distance(
                    current_medium, ray.origin, ray.dir, hit.t, s);
                throughput = throughput * vs.weight;
                if (vs.scattered == 1u) {
                    // Volume scatter event inside the medium.
                    let scatter_pos = ray.origin + ray.dir * vs.t;

                    // NEE through media — phase function evaluated
                    // at the shadow-ray direction (Henyey-Greenstein
                    // for non-zero `phase_g`, isotropic otherwise).
                    // Phase pdf is the function itself; we use it as
                    // both `f` and the MIS BSDF pdf.
                    let g = medium_mat.phase_g;
                    if (mis_nee_mode) {
                        let ls = sample_light(scatter_pos, s);
                        if (ls.valid) {
                            let trans = shadow_transmittance(
                                scatter_pos, ls.wi, ls.dist, current_medium, s);
                            let trans_sum = trans.x + trans.y + trans.z;
                            if (trans_sum > 0.0) {
                                let cos_nee = dot(ray.dir, ls.wi);
                                let phase = phase_hg_eval(cos_nee, g);
                                let f = vec3<f32>(phase);
                                let wlight = power_heuristic(ls.pdf_w, phase);
                                result.radiance = result.radiance
                                    + throughput * f * trans * ls.le * wlight / ls.pdf_w;
                            }
                        }
                    }

                    // Phase-function bounce. `f / pdf = 1` because
                    // we importance-sample directly from the phase
                    // function — bs.weight stays 1.
                    let wi = sample_hg_direction(ray.dir, g, s);
                    let cos_bounce = dot(ray.dir, wi);
                    prev_bsdf_pdf = phase_hg_eval(cos_bounce, g);
                    prev_point = scatter_pos;
                    specular_bounce = false;
                    // `current_medium` is unchanged — the scatter
                    // event stays inside the medium.

                    if (bounce > 2) {
                        let pr = max(0.05,
                            max(throughput.x, max(throughput.y, throughput.z)));
                        if (next_1d(s) > pr) {
                            break;
                        }
                        throughput = throughput / pr;
                    }
                    ray.origin = scatter_pos;
                    ray.dir = wi;
                    // Volume scattering counts as a real bounce.
                    bounce = bounce + 1;
                    continue;
                }
            }
        }

        var m = materials[hit.mat];

        // Medium-volume boundary (e.g. the fog box) — the ray
        // passes through without surface BSDF evaluation. We toggle
        // `current_medium` and spawn the next segment from just
        // past the boundary; the next iteration's volume step
        // handles attenuation across the *new* medium.
        if (is_medium_volume_material(m)) {
            if (hit.front_face == 1u) {
                current_medium = hit.mat;
            } else {
                current_medium = NO_MEDIUM;
            }
            ray.origin = hit.point + ray.dir * 1e-3;
            // ray.dir unchanged; bounce counter does NOT advance
            // along the BSDF-bounce axis — but the loop counter
            // still ticks (so a runaway "stuck on a boundary" path
            // can't spin forever).
            continue;
        }

        // PT-textures: sample the material's baseColorTexture (if any)
        // at the hit's interpolated UV. Falls back to `m.albedo` when
        // the material doesn't carry a texture.
        let albedo = material_albedo(m, hit.uv);

        // PT-mr-map: fold the metallic-roughness texture (if any)
        // into the material's scalar fields. Every downstream BSDF
        // call (NEE eval, BSDF sample, specular-bounce detector)
        // then reads the effective per-texel values without code
        // changes. `var m = ...` above makes the local copy mutable.
        let mr = material_metallic_roughness(m, hit.uv);
        m.roughness = mr.x;
        m.metallic = mr.y;

        // PT-normal-map: perturb the geometric normal in tangent
        // space. `hit.normal` is overwritten so every downstream
        // shading dot product (NEE cos, BSDF eval, BSDF sample,
        // env NEE cos) uses the perturbed shading normal. Self-
        // intersection offsets later on still compute relative to
        // the perturbed normal — fine for the showcase scenes
        // (stone-tile floor, smooth meshes); per-vertex tangents
        // + a stored geometric normal would tighten this further.
        if (m.normal_texture_idx != NO_TEXTURE) {
            hit.normal = apply_normal_map(m, hit.tri, hit.normal, hit.uv);
        }

        if (bounce == 0) {
            result.hit = true;
            result.depth = hit.t;
            result.normal = hit.normal;
            let emit_lum = max(m.emission.x, max(m.emission.y, m.emission.z));
            if (emit_lum > 0.0) {
                result.albedo = m.emission / max(emit_lum, 1e-3);
            } else {
                result.albedo = albedo;
            }
        }

        let emit = max(m.emission.x, max(m.emission.y, m.emission.z));
        if (emit > 0.1) {
            // Pure BSDF: full emission. MIS+NEE on a "specular" first hit
            // (camera ray) also gets full emission. Otherwise MIS-weight.
            if (!mis_nee_mode || specular_bounce) {
                result.radiance = result.radiance + throughput * m.emission;
            } else {
                let lp = light_pdf_solid_angle(prev_point, hit.point, hit.normal, hit.tri);
                var wmis = 1.0;
                if (lp > 0.0) {
                    wmis = power_heuristic(prev_bsdf_pdf, lp);
                }
                result.radiance = result.radiance + throughput * m.emission * wmis;
            }
            break;
        }

        let wo_dir = -ray.dir;

        // NEE — only in MIS+NEE mode. PT-fog routes through
        // `shadow_transmittance`: the function returns full ones
        // when no media intervene (so existing scenes stay
        // identical) and accumulates `exp(-σ_t · t)` across any
        // fog boundary crossings on the way to the light.
        if (mis_nee_mode) {
            let ls = sample_light(hit.point, s);
            if (ls.valid) {
                let cos_surf = dot(hit.normal, ls.wi);
                if (cos_surf > 0.0) {
                    let shadow_o = hit.point + hit.normal * 0.001;
                    let trans = shadow_transmittance(
                        shadow_o, ls.wi, ls.dist, current_medium, s);
                    let trans_sum = trans.x + trans.y + trans.z;
                    if (trans_sum > 0.0) {
                        let f = eval_bsdf(m, albedo, hit.normal, wo_dir, ls.wi);
                        let bsdf_p = bsdf_pdf(m, hit.normal, wo_dir, ls.wi);
                        let wlight = power_heuristic(ls.pdf_w, bsdf_p);
                        result.radiance = result.radiance
                            + throughput * trans * f * cos_surf * ls.le * wlight / ls.pdf_w;
                    }
                }
            }

            // PT-env NEE: sample the environment dome via inverse-CDF
            // on the luminance × sin θ tables. Independent of triangle
            // NEE (additive multi-light), each MIS-weighted against
            // BSDF. Shadow ray uses `LARGE_FAR` for `dist` — any
            // opaque triangle on the way blocks it; a clean miss
            // accumulates segment transmittance only.
            if (U.has_environment == 1u) {
                let xi_env = next_2d(s);
                let es = sample_env_importance(xi_env);
                if (es.valid) {
                    let cos_env = dot(hit.normal, es.dir);
                    if (cos_env > 0.0) {
                        let shadow_o = hit.point + hit.normal * 0.001;
                        let trans = shadow_transmittance(
                            shadow_o, es.dir, 1e10, current_medium, s);
                        let trans_sum = trans.x + trans.y + trans.z;
                        if (trans_sum > 0.0) {
                            let f = eval_bsdf(m, albedo, hit.normal, wo_dir, es.dir);
                            let bsdf_p = bsdf_pdf(m, hit.normal, wo_dir, es.dir);
                            let wlight = power_heuristic(es.pdf, bsdf_p);
                            result.radiance = result.radiance
                                + throughput * trans * f * cos_env * es.le * wlight / es.pdf;
                        }
                    }
                }
            }
        }

        // BSDF sampling — dispatches GGX (metal) or Lambertian on
        // `material.metallic`. Mirror-like roughness collapses into a
        // δ-spike that NEE shadow rays cannot evaluate, so the
        // BSDF-then-emission path carries the visibility.
        let bs = sample_bsdf(m, albedo, hit.normal, wo_dir, hit.front_face, s);
        if (!bs.valid || bs.pdf <= 0.0) {
            break;
        }
        prev_bsdf_pdf = bs.pdf;
        prev_point = hit.point;
        // Smooth dielectrics and sharp GGX are δ-function BSDFs;
        // their NEE shadow rays carry no contribution, so the next-
        // bounce direct-emission hit must collect the full unweighted
        // radiance.
        specular_bounce = m.ior > 0.0
            || (m.metallic > 0.5 && m.roughness < 0.2);
        throughput = throughput * bs.weight;

        if (bounce > 2) {
            let pr = max(0.05, max(throughput.x, max(throughput.y, throughput.z)));
            if (next_1d(s) > pr) {
                break;
            }
            throughput = throughput / pr;
        }

        // Offset along the *outgoing* hemisphere — `hit.normal`
        // faces wo, so a reflected wi sits on the same side and the
        // offset prevents self-intersection. For dielectric
        // transmission, wi is on the *opposite* side, so flip the
        // offset to land inside the medium instead of grazing the
        // surface from outside.
        let transmitted = dot(hit.normal, bs.wi) < 0.0;
        let offset_n = select(hit.normal, -hit.normal, transmitted);
        ray.origin = hit.point + offset_n * 0.001;
        ray.dir = bs.wi;

        // PT-beer-lambert: a dielectric transmit swaps which medium
        // the ray is travelling through. Entering (front_face = 1)
        // puts the ray inside this material's medium; exiting drops
        // us back to vacuum. Reflect branches leave `current_medium`
        // alone — the ray stays on the side it came from.
        if (m.ior > 0.0 && transmitted) {
            if (hit.front_face == 1u) {
                current_medium = hit.mat;
            } else {
                current_medium = NO_MEDIUM;
            }
        }

        // Surface BSDF bounce — counts as a real light bounce.
        bounce = bounce + 1;
    }
    return result;
}

struct PathTraceOut {
    @location(0) radiance: vec4<f32>,
    @location(1) albedo: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) depth: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut) -> PathTraceOut {
    let pixel = vec2<u32>(in.position.xy);
    var s = init_sampler(pixel, U.frame_count, U.viewport_width);

    let ray = get_camera_ray(in.uv, &s);
    let sample = path_trace(ray, &s);

    var out: PathTraceOut;
    out.radiance = vec4<f32>(sample.radiance, 1.0);
    out.albedo = vec4<f32>(sample.albedo, 1.0);
    out.normal = vec4<f32>(sample.normal, 1.0);
    let mask = select(0.0, 1.0, sample.hit);
    out.depth = vec4<f32>(sample.depth, 0.0, 0.0, mask);
    return out;
}
