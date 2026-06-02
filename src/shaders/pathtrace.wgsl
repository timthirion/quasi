// Cornell Box path tracer with next-event estimation + MIS.
// Ported from the verified reference integrator. PCG sampling (M1); other
// samplers come later. Renders one sample/frame to an HDR texture.

const MAX_QUADS: u32 = 32u;
const MAX_BOUNCES: i32 = 5;
const PI: f32 = 3.14159265359;

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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

// ----- PCG RNG -----

fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand(rng: ptr<function, u32>) -> f32 {
    *rng = pcg_hash(*rng);
    return f32(*rng) / 4294967295.0;
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

fn sample_light(p: vec3<f32>, rng: ptr<function, u32>) -> LightSample {
    var ls: LightSample;
    ls.valid = false;

    let q = U.quads[U.light_index];
    let r1 = rand(rng);
    let r2 = rand(rng);
    let x = q.origin + r1 * q.u + r2 * q.v;

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

fn cosine_sample_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let r1 = rand(rng);
    let r2 = rand(rng);
    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(1.0 - r2);
    let sin_theta = sqrt(r2);

    let w = normalize(normal);
    var a: vec3<f32>;
    if (abs(w.x) > 0.9) {
        a = vec3<f32>(0.0, 1.0, 0.0);
    } else {
        a = vec3<f32>(1.0, 0.0, 0.0);
    }
    let v = normalize(cross(w, a));
    let u = cross(w, v);
    return normalize(u * cos(phi) * sin_theta + v * sin(phi) * sin_theta + w * cos_theta);
}

// ----- Camera -----

fn get_camera_ray(uv_in: vec2<f32>, rng: ptr<function, u32>) -> Ray {
    let cam = U.camera;
    let jitter = vec2<f32>(rand(rng), rand(rng)) - 0.5;
    let uv = uv_in + jitter * 0.001;

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

fn path_trace(ray_in: Ray, rng: ptr<function, u32>) -> vec3<f32> {
    var ray = ray_in;
    var color = vec3<f32>(0.0);
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
        let emit = max(m.emission.x, max(m.emission.y, m.emission.z));

        if (emit > 0.1) {
            if (specular_bounce) {
                color = color + throughput * m.emission;
            } else {
                let lp = light_pdf_solid_angle(prev_point, hit.point, hit.normal);
                var wmis = 1.0;
                if (lp > 0.0) {
                    wmis = power_heuristic(prev_bsdf_pdf, lp);
                }
                color = color + throughput * m.emission * wmis;
            }
            break;
        }

        // Next-event estimation.
        let ls = sample_light(hit.point, rng);
        if (ls.valid) {
            let cos_surf = dot(hit.normal, ls.wi);
            if (cos_surf > 0.0) {
                let shadow_o = hit.point + hit.normal * 0.001;
                if (!occluded(shadow_o, ls.wi, ls.dist)) {
                    let f = m.albedo / PI;
                    let bsdf_pdf = cos_surf / PI;
                    let wlight = power_heuristic(ls.pdf_w, bsdf_pdf);
                    color = color + throughput * f * cos_surf * ls.le * wlight / ls.pdf_w;
                }
            }
        }

        // BSDF sampling (cosine-weighted Lambertian).
        let wi = cosine_sample_hemisphere(hit.normal, rng);
        let cos_wi = max(dot(hit.normal, wi), 0.0);
        prev_bsdf_pdf = cos_wi / PI;
        prev_point = hit.point;
        specular_bounce = false;
        throughput = throughput * m.albedo;

        if (bounce > 2) {
            let pr = max(0.05, max(throughput.x, max(throughput.y, throughput.z)));
            if (rand(rng) > pr) {
                break;
            }
            throughput = throughput / pr;
        }

        ray.origin = hit.point + hit.normal * 0.001;
        ray.dir = wi;
    }
    return color;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pixel = vec2<u32>(in.position.xy);
    let seed = pixel.x + pixel.y * U.viewport_width
        + U.frame_count * U.viewport_width * U.viewport_height;
    var rng = pcg_hash(seed);

    let ray = get_camera_ray(in.uv, &rng);
    let c = path_trace(ray, &rng);
    return vec4<f32>(c, 1.0);
}
