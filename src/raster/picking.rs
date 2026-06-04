//! Mouse → world picking math for the raster widget.
//!
//! Kept in its own non-wasm-gated module so the helpers are testable
//! natively under `cargo test`. The actual event handlers that *call*
//! these live in [`crate::raster::web`] which is wasm-only by nature
//! (they need `web_sys` + `wasm_bindgen`).

use crate::gpu::OrbitCamera;

use crate::raster::wire::WirePose;

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Builds a world-space ray through the cursor at canvas-local
/// `(mx, my)` (CSS pixels). Same convention as the renderer's
/// perspective: the camera looks down its `forward()` direction with
/// world-up `[0, 1, 0]`.
pub fn camera_ray(
    camera: &OrbitCamera,
    canvas_w: f32,
    canvas_h: f32,
    mx: f32,
    my: f32,
) -> ([f32; 3], [f32; 3]) {
    let eye = camera.position();
    let forward = camera.direction();
    let right = normalize3(cross3(forward, [0.0, 1.0, 0.0]));
    let view_up = cross3(right, forward);

    let aspect = canvas_w / canvas_h;
    let half_h = (camera.fov * std::f32::consts::PI / 180.0 / 2.0).tan();
    let half_w = aspect * half_h;
    let nx = 2.0 * mx / canvas_w - 1.0;
    let ny = 1.0 - 2.0 * my / canvas_h;
    let sx = nx * half_w;
    let sy = ny * half_h;
    let dir = normalize3([
        sx * right[0] + sy * view_up[0] + forward[0],
        sx * right[1] + sy * view_up[1] + forward[1],
        sx * right[2] + sy * view_up[2] + forward[2],
    ]);
    (eye, dir)
}

/// Returns the world-space hit point if `(origin, dir)` crosses the
/// `y = 0` plane in the `+t` direction.
pub fn ray_floor_hit(origin: [f32; 3], dir: [f32; 3]) -> Option<[f32; 3]> {
    if dir[1].abs() < 1e-6 {
        return None;
    }
    let t = -origin[1] / dir[1];
    if t <= 0.0 {
        return None;
    }
    Some([origin[0] + t * dir[0], 0.0, origin[2] + t * dir[2]])
}

/// True iff the (normalised-direction) ray intersects the sphere
/// centred at `center` with radius `radius`. Picking accepts any
/// intersection — the front-vs-back distinction doesn't matter for a
/// click that starts a drag.
pub fn ray_hits_sphere(origin: [f32; 3], dir: [f32; 3], center: [f32; 3], radius: f32) -> bool {
    let oc = [
        origin[0] - center[0],
        origin[1] - center[1],
        origin[2] - center[2],
    ];
    let b = 2.0 * (dir[0] * oc[0] + dir[1] * oc[1] + dir[2] * oc[2]);
    let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius * radius;
    b * b - 4.0 * c >= 0.0
}

/// Serialises a [`WirePose`] back to JSON in the same nalgebra-flavoured
/// shape the renderer accepts. Used by the goal-drag callback so the JS
/// host can parse it the same way it parsed `setGoal`.
pub fn serialize_pose(pose: &WirePose) -> String {
    let q = pose.rotation.quaternion.coords;
    let t = pose.translation.vector;
    format!(
        r#"{{"pose":{{"rotation":{{"quaternion":{{"coords":[{},{},{},{}]}}}},"translation":{{"vector":[{},{},{}]}}}}}}"#,
        q[0], q[1], q[2], q[3], t[0], t[1], t[2]
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::wire::{parse_goal, WireQuaternion, WireRotation, WireTranslation};

    #[test]
    fn ray_floor_hit_for_camera_looking_down() {
        let origin = [0.0, 5.0, 0.0];
        let dir = [0.0, -1.0, 0.0];
        let h = ray_floor_hit(origin, dir).expect("should hit");
        assert_eq!(h, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn ray_floor_hit_skips_upward_rays() {
        let origin = [0.0, 5.0, 0.0];
        let dir = [0.0, 1.0, 0.0];
        assert!(ray_floor_hit(origin, dir).is_none());
    }

    #[test]
    fn ray_floor_hit_translates_origin() {
        // 45° downward ray from (1, 2, 0): hits (1+2, 0, 0) = (3, 0, 0).
        let origin = [1.0, 2.0, 0.0];
        let dir = normalize3([1.0, -1.0, 0.0]);
        let h = ray_floor_hit(origin, dir).expect("should hit");
        assert!((h[0] - 3.0).abs() < 1e-5, "got {h:?}");
        assert!((h[2]).abs() < 1e-5);
    }

    #[test]
    fn sphere_hit_detection_basics() {
        let origin = [0.0, 0.0, 5.0];
        let dir = [0.0, 0.0, -1.0];
        assert!(ray_hits_sphere(origin, dir, [0.0, 0.0, 0.0], 0.5));
        assert!(!ray_hits_sphere(origin, dir, [10.0, 0.0, 0.0], 0.5));
        // Right at the silhouette: hits.
        assert!(ray_hits_sphere(origin, dir, [0.5, 0.0, 0.0], 0.5));
    }

    #[test]
    fn camera_ray_centre_goes_forward() {
        let camera = OrbitCamera::new();
        let (origin, dir) = camera_ray(&camera, 800.0, 600.0, 400.0, 300.0);
        // Default orbit camera looks down -Z from (0, 1, 3.5).
        assert!((dir[2] + 1.0).abs() < 0.05, "got {dir:?}");
        assert_eq!(origin, camera.position());
    }

    #[test]
    fn camera_ray_right_edge_tilts_right() {
        let camera = OrbitCamera::new();
        let (_, dir) = camera_ray(&camera, 800.0, 600.0, 800.0, 300.0);
        // A ray through the right edge of the screen should have a
        // positive X component in world space (right of the centre).
        assert!(dir[0] > 0.0, "dir.x should be positive, got {dir:?}");
    }

    #[test]
    fn serialize_pose_round_trips_through_parse_goal() {
        let pose = WirePose {
            rotation: WireRotation {
                quaternion: WireQuaternion {
                    coords: [0.0, 0.0, 0.0, 1.0],
                },
            },
            translation: WireTranslation {
                vector: [1.5, 0.75, -0.25],
            },
        };
        let json = serialize_pose(&pose);
        let g = parse_goal(&json).expect("parse round-trips");
        assert_eq!(g.pose.translation.vector, [1.5, 0.75, -0.25]);
        assert_eq!(g.pose.rotation.quaternion.coords, [0.0, 0.0, 0.0, 1.0]);
    }
}
