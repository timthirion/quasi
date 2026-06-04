// Cornell Box path tracer with next-event estimation + MIS.
//
// Renders **four** color attachments per fragment (MRT) so AOVs accumulate
// in lockstep with radiance:
//   @location(0) radiance (rgb, throughput = a)
//   @location(1) albedo   (rgb of the first hit)
//   @location(2) normal   (xyz of the first hit, encoded in [-1, 1])
//   @location(3) depth    (scalar distance to first hit; w channel = 1 on hit)
//
// Three samplers are dispatched at runtime from `U.sampler_kind`:
//   0: PCG (i.i.d.)
//   1: Halton (per-pixel Cranley-Patterson rotation)
//   2: Sobol  (per-pixel XOR scramble)
//
// CPU mirrors of all three live in `src/pathtrace/sampler.rs` so the
// sequences can be pinned to canonical values in `cargo test`.

const MAX_QUADS: u32 = 32u;
const MAX_BOUNCES: i32 = 5;
const PI: f32 = 3.14159265359;

const SAMPLER_PCG: u32 = 0u;
const SAMPLER_HALTON: u32 = 1u;
const SAMPLER_SOBOL: u32 = 2u;

// 2^32 as f32. Used for u32 -> [0,1) float conversion (matches CPU).
const U32_NORM: f32 = 4294967296.0;

struct Camera {
    position: vec3<f32>,
    fov: f32,
    direction: vec3<f32>,
    aspect: f32,
    up: vec3<f32>,
    _pad: f32,
};

struct Quad {
    origin: vec3<f32>,
    _p0: f32,
    u: vec3<f32>,
    _p1: f32,
    v: vec3<f32>,
    _p2: f32,
};

struct Material {
    albedo: vec3<f32>,
    roughness: f32,
    emission: vec3<f32>,
    metallic: f32,
};

struct Uniforms {
    camera: Camera,
    quad_count: u32,
    frame_count: u32,
    light_index: u32,
    viewport_width: u32,
    viewport_height: u32,
    sampler_kind: u32,
    _pad0: u32,
    _pad1: u32,
    quads: array<Quad, 32>,
    materials: array<Material, 32>,
};

@group(0) @binding(0) var<uniform> U: Uniforms;

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

// ----- Samplers -----
//
// One state struct serves all three families: PCG advances `pcg`; Sobol
// advances `sobol_index`; Halton advances `halton_dim` (the sequence
// index is fixed per (pixel, frame)). Per-pixel scrambles decorrelate
// pixels for the QMC samplers.

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
    // 32 is enough to exhaust any u32 in any base >= 2.
    for (var i = 0u; i < 32u && n > 0u; i = i + 1u) {
        let digit = f32(n % base);
        result = result + digit * inv_base_n;
        inv_base_n = inv_base_n * inv_base;
        n = n / base;
    }
    return result;
}

fn halton_base(dim: u32) -> u32 {
    // Matches the first 16 entries of HALTON_PRIMES on the CPU side.
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

// Sobol direction-vector mixing for dim 0 (van der Corput in base 2) and
// dim 1 (polynomial x + 1, m_1 = 1). The recurrence `m_{i+1} = (m_i << 1)
// ^ m_i` is unrolled per loop iter; matches `build_sobol_directions` on
// the CPU side.
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
        // PCG (default).
        let a = rand_pcg(s);
        let b = rand_pcg(s);
        return vec2<f32>(a, b);
    }
}

fn next_1d(s: ptr<function, SamplerState>) -> f32 {
    // 1-D draws (e.g. Russian roulette) get the x component of a fresh
    // 2-D point. Wastes one Sobol/Halton coordinate but keeps the
    // dispatch logic uniform — and RR happens once per bounce at most.
    return next_2d(s).x;
}

fn init_sampler(pixel: vec2<u32>, frame: u32, width: u32) -> SamplerState {
    var s: SamplerState;
    let pixel_seed = pixel.x + pixel.y * width;
    s.pcg = pcg_hash(pixel_seed + frame * 0x9e3779b9u);
    s.scramble_x = pcg_hash(pixel_seed);
    s.scramble_y = pcg_hash(s.scramble_x);
    // Sobol: each frame advances by 16 to skip the dimensions used in
    // the previous frame's path. Bounded by ~ MAX_BOUNCES * 2 + jitter.
    s.sobol_index = frame * 16u + 1u;
    s.halton_index = frame + 1u;
    s.halton_dim = 0u;
    return s;
}

// ----- Geometry -----

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
};

struct Hit {
    t: f32,
    point: vec3<f32>,
    normal: vec3<f32>,
    mat: u32,
    hit: bool,
};

fn intersect_quad(ray: Ray, q: Quad, mat_idx: u32, t_min: f32, t_max: f32) -> Hit {
    var rec: Hit;
    rec.hit = false;

    let n = cross(q.u, q.v);
    let area_sq = dot(n, n);
    if (area_sq < 1e-8) {
        return rec;
    }
    let normal = n * inverseSqrt(area_sq);
    let d = dot(normal, q.origin);
    let denom = dot(normal, ray.dir);
    if (abs(denom) < 1e-8) {
        return rec;
    }
    let t = (d - dot(normal, ray.origin)) / denom;
    if (t < t_min || t > t_max) {
        return rec;
    }
    let p = ray.origin + ray.dir * t;
    let planar = p - q.origin;
    let w = n / area_sq;
    let alpha = dot(w, cross(planar, q.v));
    let beta = dot(w, cross(q.u, planar));
    if (alpha < 0.0 || alpha > 1.0 || beta < 0.0 || beta > 1.0) {
        return rec;
    }

    rec.hit = true;
    rec.t = t;
    rec.point = p;
    rec.mat = mat_idx;
    if (denom < 0.0) {
        rec.normal = normal;
    } else {
        rec.normal = -normal;
    }
    return rec;
}

fn trace_scene(ray: Ray) -> Hit {
    var closest: Hit;
    closest.hit = false;
    closest.t = 1e30;
    for (var i = 0u; i < U.quad_count && i < MAX_QUADS; i = i + 1u) {
        let h = intersect_quad(ray, U.quads[i], i, 0.001, closest.t);
        if (h.hit) {
            closest = h;
        }
    }
    return closest;
}

fn occluded(origin: vec3<f32>, dir: vec3<f32>, dist: f32) -> bool {
    let t_max = dist - 1e-3;
    var r: Ray;
    r.origin = origin;
    r.dir = dir;
    for (var i = 0u; i < U.quad_count && i < MAX_QUADS; i = i + 1u) {
        if (i == U.light_index) {
            continue;
        }
        let h = intersect_quad(r, U.quads[i], i, 1e-3, t_max);
        if (h.hit) {
            return true;
        }
    }
    return false;
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

    let q = U.quads[U.light_index];
    let r = next_2d(s);
    let x = q.origin + r.x * q.u + r.y * q.v;

    let n_un = cross(q.u, q.v);
    let area = length(n_un);
    if (area < 1e-8) {
        return ls;
    }
    let n_l = n_un / area;

    let dvec = x - p;
    let dist = length(dvec);
    if (dist < 1e-4) {
        return ls;
    }
    let wi = dvec / dist;
    let cos_l = dot(n_l, -wi);
    if (cos_l <= 0.0) {
        return ls;
    }

    ls.wi = wi;
    ls.dist = dist;
    ls.pdf_w = (dist * dist) / (cos_l * area);
    ls.le = U.materials[U.light_index].emission;
    ls.valid = true;
    return ls;
}

fn light_pdf_solid_angle(p0: vec3<f32>, hit_point: vec3<f32>, hit_normal: vec3<f32>) -> f32 {
    let q = U.quads[U.light_index];
    let area = length(cross(q.u, q.v));
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
    return dist2 / (cos_l * area);
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

    for (var bounce = 0; bounce < MAX_BOUNCES; bounce = bounce + 1) {
        let hit = trace_scene(ray);
        if (!hit.hit) {
            break;
        }
        let m = U.materials[hit.mat];

        if (bounce == 0) {
            // First-hit AOVs.
            result.hit = true;
            result.depth = hit.t;
            result.normal = hit.normal;
            // For emissives we record the emission's intensity-normalized
            // colour as an AOV so the surface still has a meaningful tint.
            let emit_lum = max(m.emission.x, max(m.emission.y, m.emission.z));
            if (emit_lum > 0.0) {
                result.albedo = m.emission / max(emit_lum, 1e-3);
            } else {
                result.albedo = m.albedo;
            }
        }

        let emit = max(m.emission.x, max(m.emission.y, m.emission.z));
        if (emit > 0.1) {
            if (specular_bounce) {
                result.radiance = result.radiance + throughput * m.emission;
            } else {
                let lp = light_pdf_solid_angle(prev_point, hit.point, hit.normal);
                var wmis = 1.0;
                if (lp > 0.0) {
                    wmis = power_heuristic(prev_bsdf_pdf, lp);
                }
                result.radiance = result.radiance + throughput * m.emission * wmis;
            }
            break;
        }

        // Next-event estimation.
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
    // Encode normal in [-1, 1]; EXR keeps it as-is, PNG previews remap.
    out.normal = vec4<f32>(sample.normal, 1.0);
    // Hits write t; misses write 0 with alpha = 0 so the accumulator
    // can still average them as 0.
    let mask = select(0.0, 1.0, sample.hit);
    out.depth = vec4<f32>(sample.depth, 0.0, 0.0, mask);
    return out;
}
