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
    _pad0: u32,
    _pad1: u32,
    // PT-beer-lambert: per-channel Beer-Lambert absorption coefficient
    // applied to throughput per unit of distance travelled *inside*
    // this material. `(0, 0, 0)` = no participating-media tinting.
    absorption: vec3<f32>,
    _pad2: f32,
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
@group(0) @binding(5) var<storage, read> emissive_triangles: array<u32>;
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
    halton_index: u32,
    halton_dim: u32,
    scramble_x: u32,
    scramble_y: u32,
};

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

fn sobol_dim0(index: u32) -> u32 {
    var acc: u32 = 0u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if (((index >> i) & 1u) == 1u) {
            acc = acc ^ (1u << (31u - i));
        }
    }
    return acc;
}

fn sobol_dim1(index: u32) -> u32 {
    var acc: u32 = 0u;
    var m: u32 = 1u;
    for (var i = 0u; i < 32u; i = i + 1u) {
        if (((index >> i) & 1u) == 1u) {
            acc = acc ^ (m << (31u - i));
        }
        if (i < 31u) {
            m = (m << 1u) ^ m;
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
        let x_raw = sobol_dim0((*s).sobol_index) ^ (*s).scramble_x;
        let y_raw = sobol_dim1((*s).sobol_index) ^ (*s).scramble_y;
        (*s).sobol_index = (*s).sobol_index + 1u;
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
    s.scramble_x = pcg_hash(pixel_seed);
    s.scramble_y = pcg_hash(s.scramble_x);
    s.sobol_index = frame * 16u + 1u;
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

fn sample_light(p: vec3<f32>, s: ptr<function, SamplerState>) -> LightSample {
    var ls: LightSample;
    ls.valid = false;
    if (U.emissive_count == 0u) {
        return ls;
    }

    let pick = next_1d(s);
    var picked = u32(pick * f32(U.emissive_count));
    if (picked >= U.emissive_count) {
        picked = U.emissive_count - 1u;
    }
    let tri = emissive_triangles[picked];

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
    // Solid-angle pdf for sampling THIS triangle THIS barycentric point:
    //   pdf_w = dist^2 / (cos_l * area * N_emitters)
    // where 1/N_emitters comes from uniformly picking among emitters.
    ls.pdf_w = (dist * dist) / (cos_l * area * f32(U.emissive_count));
    ls.le = materials[tri_materials[tri]].emission;
    ls.valid = true;
    return ls;
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
    return dist2 / (cos_l * area * f32(U.emissive_count));
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
        let hit = trace_scene(ray);
        if (!hit.hit) {
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

        let m = materials[hit.mat];

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
