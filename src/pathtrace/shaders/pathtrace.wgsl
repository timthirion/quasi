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
};

struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    emission: vec3<f32>,
    metallic: f32,
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

// Möller-Trumbore intersection. Returns t (>0) on hit, or a negative
// sentinel on miss. Double-sided (path tracer doesn't cull backfaces).
fn intersect_triangle(ray: Ray, t: TriVerts, t_min: f32, t_max: f32) -> f32 {
    let edge1 = t.v1 - t.v0;
    let edge2 = t.v2 - t.v0;
    let h = cross(ray.dir, edge2);
    let a = dot(edge1, h);
    if (abs(a) < 1e-8) {
        return -1.0;
    }
    let f = 1.0 / a;
    let s = ray.origin - t.v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return -1.0;
    }
    let q = cross(s, edge1);
    let v = f * dot(ray.dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return -1.0;
    }
    let t_hit = f * dot(edge2, q);
    if (t_hit < t_min || t_hit > t_max) {
        return -1.0;
    }
    return t_hit;
}

struct Hit {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    tri: u32,
    mat: u32,
    hit: bool,
};

// Record a triangle hit into the running closest. Factored so both
// the linear-scan and the BVH traversal use the same flip-normal
// convention.
fn record_hit(closest: ptr<function, Hit>, ray: Ray, tri: u32, verts: TriVerts, t_hit: f32) {
    (*closest).hit = true;
    (*closest).t = t_hit;
    (*closest).point = ray.origin + ray.dir * t_hit;
    (*closest).tri = tri;
    (*closest).mat = tri_materials[tri];
    var n = normalize(cross(verts.v1 - verts.v0, verts.v2 - verts.v0));
    if (dot(n, ray.dir) > 0.0) {
        n = -n;
    }
    (*closest).normal = n;
}

fn trace_scene_linear(ray: Ray) -> Hit {
    var closest: Hit;
    closest.hit = false;
    closest.t = 1e30;
    for (var tri = 0u; tri < U.triangle_count; tri = tri + 1u) {
        let verts = triangle_vertices(tri);
        let t_hit = intersect_triangle(ray, verts, 0.001, closest.t);
        if (t_hit > 0.0) {
            record_hit(&closest, ray, tri, verts, t_hit);
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
        if (intersect_triangle(r, verts, 1e-3, t_max) > 0.0) {
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
                let t_hit = intersect_triangle(ray, verts, 0.001, closest.t);
                if (t_hit > 0.0) {
                    record_hit(&closest, ray, tri, verts, t_hit);
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
                if (intersect_triangle(r, verts, 1e-3, t_max) > 0.0) {
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
    let mis_nee_mode = U.integrator_kind == INTEGRATOR_MIS_NEE;

    for (var bounce = 0; bounce < MAX_BOUNCES; bounce = bounce + 1) {
        let hit = trace_scene(ray);
        if (!hit.hit) {
            break;
        }
        let m = materials[hit.mat];

        if (bounce == 0) {
            result.hit = true;
            result.depth = hit.t;
            result.normal = hit.normal;
            let emit_lum = max(m.emission.x, max(m.emission.y, m.emission.z));
            if (emit_lum > 0.0) {
                result.albedo = m.emission / max(emit_lum, 1e-3);
            } else {
                result.albedo = m.albedo;
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

        // NEE — only in MIS+NEE mode.
        if (mis_nee_mode) {
            let ls = sample_light(hit.point, s);
            if (ls.valid) {
                let cos_surf = dot(hit.normal, ls.wi);
                if (cos_surf > 0.0) {
                    let shadow_o = hit.point + hit.normal * 0.001;
                    if (!occluded(shadow_o, ls.wi, ls.dist)) {
                        let f = m.albedo / PI;
                        let bsdf_pdf = cos_surf / PI;
                        let wlight = power_heuristic(ls.pdf_w, bsdf_pdf);
                        result.radiance = result.radiance
                            + throughput * f * cos_surf * ls.le * wlight / ls.pdf_w;
                    }
                }
            }
        }

        // BSDF sampling (cosine-weighted Lambertian).
        let wi = cosine_sample_hemisphere(hit.normal, s);
        let cos_wi = max(dot(hit.normal, wi), 0.0);
        prev_bsdf_pdf = cos_wi / PI;
        prev_point = hit.point;
        specular_bounce = false;
        throughput = throughput * m.albedo;

        if (bounce > 2) {
            let pr = max(0.05, max(throughput.x, max(throughput.y, throughput.z)));
            if (next_1d(s) > pr) {
                break;
            }
            throughput = throughput / pr;
        }

        ray.origin = hit.point + hit.normal * 0.001;
        ray.dir = wi;
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
