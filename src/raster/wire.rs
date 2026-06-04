//! Motum-shaped JSON wire format for the raster widget.
//!
//! R4 of plan 0002. The wasm-bindgen API on [`crate::raster::web::RasterInstance`]
//! accepts JSON strings; this module owns the deserializable Rust mirror
//! types they parse into.
//!
//! ## Compatibility with motum
//!
//! - [`WireWorldState`] is **byte-for-byte compatible** with motum's
//!   `motum::world::WorldState` — including the nalgebra `Isometry3<f64>`
//!   inner structure (`{ rotation: { quaternion: { coords } }, translation:
//!   { vector } }`). A motum process can `serde_json::to_string` its world
//!   state and the resulting string drops straight into `setWorldState`.
//!
//! - [`WireTrajectory`] is **renderer-friendly** — its waypoints carry
//!   world-space [`WireWorldState`]s rather than motum's joint-space
//!   `Configuration`s. The renderer has no robot model, no FK; expecting
//!   it to evaluate forward kinematics would be a layering violation.
//!   The motum-side glue applies FK once per waypoint before sending.
//!
//! - [`WireTreeOverlay`] is **renderer-friendly** too — flat 3-D edges
//!   and points, not motum's `PlannerTree { nodes: Vec<Configuration>,
//!   parents: Vec<Option<usize>> }`. Same reason: motum projects each
//!   tree node into world space (typically end-effector position) and
//!   sends edges as world-space line segments.
//!
//! - [`WireGoal`] is a single world-space [`WirePose`]; the renderer
//!   uses it to position the draggable goal handle.
//!
//! Errors are returned as [`WireError`] (the wasm setter wraps it in
//! `JsValue::from_str`).

use serde::Deserialize;

/// Nalgebra-shaped translation vector. Mirrors
/// `nalgebra::Translation3<f64>`'s serde output.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct WireTranslation {
    pub vector: [f64; 3],
}

/// Nalgebra-shaped unit quaternion. The four components are stored as
/// `[i, j, k, w]` in `coords` (this is how nalgebra serializes
/// `Quaternion`).
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct WireQuaternion {
    pub coords: [f64; 4],
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct WireRotation {
    pub quaternion: WireQuaternion,
}

/// Motum's `Pose` = nalgebra's `Isometry3<f64>`. Identity is
/// `{ rotation: { quaternion: { coords: [0,0,0,1] } }, translation: {
/// vector: [0,0,0] } }`.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub struct WirePose {
    pub rotation: WireRotation,
    pub translation: WireTranslation,
}

impl WirePose {
    /// Builds a column-major 4×4 model matrix in the same convention as
    /// [`crate::raster::scene::Instance::model`].
    pub fn to_model_matrix(&self) -> [[f32; 4]; 4] {
        let [qi, qj, qk, qw] = self.rotation.quaternion.coords;
        let [tx, ty, tz] = self.translation.vector;

        // Standard quaternion → rotation matrix expansion. Column-major
        // output to match the existing `translation` / `scale` helpers.
        let xx = qi * qi;
        let yy = qj * qj;
        let zz = qk * qk;
        let xy = qi * qj;
        let xz = qi * qk;
        let yz = qj * qk;
        let wx = qw * qi;
        let wy = qw * qj;
        let wz = qw * qk;

        [
            [
                (1.0 - 2.0 * (yy + zz)) as f32,
                (2.0 * (xy + wz)) as f32,
                (2.0 * (xz - wy)) as f32,
                0.0,
            ],
            [
                (2.0 * (xy - wz)) as f32,
                (1.0 - 2.0 * (xx + zz)) as f32,
                (2.0 * (yz + wx)) as f32,
                0.0,
            ],
            [
                (2.0 * (xz + wy)) as f32,
                (2.0 * (yz - wx)) as f32,
                (1.0 - 2.0 * (xx + yy)) as f32,
                0.0,
            ],
            [tx as f32, ty as f32, tz as f32, 1.0],
        ]
    }
}

/// World snapshot: parallel arrays of per-link poses and (optional)
/// geometry handles. Matches `motum::world::WorldState` exactly.
#[derive(Clone, Debug, Deserialize)]
pub struct WireWorldState {
    pub link_poses: Vec<WirePose>,
    pub link_geometry: Vec<Option<u32>>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WireTrajectoryWaypoint {
    pub time: f64,
    pub world_state: WireWorldState,
}

/// World-space piecewise-linear trajectory: a sequence of waypoints
/// each carrying a [`WireWorldState`]. The renderer samples by snapping
/// to the closest waypoint at the current playhead time — proper
/// lerp-between-waypoints is future work.
#[derive(Clone, Debug, Deserialize)]
pub struct WireTrajectory {
    pub waypoints: Vec<WireTrajectoryWaypoint>,
}

/// One edge in a planner-search-tree overlay: a line segment in world
/// space between two 3-D points.
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct WireTreeEdge {
    pub from: [f64; 3],
    pub to: [f64; 3],
}

/// Renderable planner-tree overlay. Edges become depth-tested overlay
/// lines; nodes (if present) become depth-tested overlay points.
#[derive(Clone, Debug, Deserialize, Default)]
pub struct WireTreeOverlay {
    #[serde(default)]
    pub edges: Vec<WireTreeEdge>,
    #[serde(default)]
    pub nodes: Vec<[f64; 3]>,
}

/// Pickable goal marker — a single world-space pose, rendered as an
/// emphasised instance the user can drag.
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct WireGoal {
    pub pose: WirePose,
}

#[derive(Debug)]
pub enum WireError {
    Json(serde_json::Error),
}

impl core::fmt::Display for WireError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Json(e) => write!(f, "wire JSON parse error: {e}"),
        }
    }
}

impl std::error::Error for WireError {}

impl From<serde_json::Error> for WireError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

pub fn parse_world_state(json: &str) -> Result<WireWorldState, WireError> {
    Ok(serde_json::from_str(json)?)
}

pub fn parse_trajectory(json: &str) -> Result<WireTrajectory, WireError> {
    Ok(serde_json::from_str(json)?)
}

pub fn parse_tree_overlay(json: &str) -> Result<WireTreeOverlay, WireError> {
    Ok(serde_json::from_str(json)?)
}

pub fn parse_goal(json: &str) -> Result<WireGoal, WireError> {
    Ok(serde_json::from_str(json)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This is the exact shape motum produces. If nalgebra ever changes
    /// its `Isometry3` serialization, this test catches the drift
    /// immediately.
    fn identity_pose_json() -> &'static str {
        r#"{
            "rotation": { "quaternion": { "coords": [0.0, 0.0, 0.0, 1.0] } },
            "translation": { "vector": [0.0, 0.0, 0.0] }
        }"#
    }

    #[test]
    fn parses_motum_identity_pose() {
        let p: WirePose = serde_json::from_str(identity_pose_json()).unwrap();
        assert_eq!(p.translation.vector, [0.0, 0.0, 0.0]);
        assert_eq!(p.rotation.quaternion.coords, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn identity_pose_yields_identity_matrix() {
        let p: WirePose = serde_json::from_str(identity_pose_json()).unwrap();
        let m = p.to_model_matrix();
        // Column-major identity.
        let expected: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for c in 0..4 {
            for r in 0..4 {
                assert!(
                    (m[c][r] - expected[c][r]).abs() < 1e-6,
                    "mismatch at [{c}][{r}]: {} vs {}",
                    m[c][r],
                    expected[c][r]
                );
            }
        }
    }

    #[test]
    fn pose_with_translation_only() {
        let json = r#"{
            "rotation": { "quaternion": { "coords": [0.0, 0.0, 0.0, 1.0] } },
            "translation": { "vector": [1.0, 2.0, 3.0] }
        }"#;
        let p: WirePose = serde_json::from_str(json).unwrap();
        let m = p.to_model_matrix();
        // Last column = translation.
        assert!((m[3][0] - 1.0).abs() < 1e-6);
        assert!((m[3][1] - 2.0).abs() < 1e-6);
        assert!((m[3][2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn ninety_degree_y_rotation() {
        // Quaternion for 90° rotation about +Y: (0, sin(45°), 0, cos(45°))
        let s = (std::f64::consts::FRAC_PI_4).sin();
        let c = (std::f64::consts::FRAC_PI_4).cos();
        let json = format!(
            r#"{{
                "rotation": {{ "quaternion": {{ "coords": [0.0, {s}, 0.0, {c}] }} }},
                "translation": {{ "vector": [0.0, 0.0, 0.0] }}
            }}"#
        );
        let p: WirePose = serde_json::from_str(&json).unwrap();
        let m = p.to_model_matrix();
        // R_y(90°) sends +X (column 0) to -Z.
        assert!(m[0][0].abs() < 1e-5);
        assert!(m[0][1].abs() < 1e-5);
        assert!((m[0][2] + 1.0).abs() < 1e-5, "got {}", m[0][2]);
    }

    #[test]
    fn parses_world_state_with_geometry_handles() {
        let json = r#"{
            "link_poses": [
                {"rotation": {"quaternion": {"coords": [0,0,0,1]}}, "translation": {"vector": [0,0,0]}},
                {"rotation": {"quaternion": {"coords": [0,0,0,1]}}, "translation": {"vector": [1,0,0]}}
            ],
            "link_geometry": [null, 0]
        }"#;
        let w = parse_world_state(json).unwrap();
        assert_eq!(w.link_poses.len(), 2);
        assert_eq!(w.link_geometry, vec![None, Some(0)]);
        assert_eq!(w.link_poses[1].translation.vector, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn parses_tree_overlay_with_defaults() {
        let edges_only = r#"{"edges": [{"from": [0,0,0], "to": [1,0,0]}]}"#;
        let t = parse_tree_overlay(edges_only).unwrap();
        assert_eq!(t.edges.len(), 1);
        assert!(t.nodes.is_empty(), "missing nodes should default to empty");

        let empty = r#"{}"#;
        let t = parse_tree_overlay(empty).unwrap();
        assert!(t.edges.is_empty());
        assert!(t.nodes.is_empty());
    }

    #[test]
    fn parses_goal() {
        let json = r#"{
            "pose": {
                "rotation": {"quaternion": {"coords": [0,0,0,1]}},
                "translation": {"vector": [1.5, 0.5, 0.0]}
            }
        }"#;
        let g = parse_goal(json).unwrap();
        assert_eq!(g.pose.translation.vector, [1.5, 0.5, 0.0]);
    }

    #[test]
    fn parses_trajectory() {
        let json = r#"{
            "waypoints": [
                {"time": 0.0, "world_state": {"link_poses": [], "link_geometry": []}},
                {"time": 1.5, "world_state": {"link_poses": [], "link_geometry": []}}
            ]
        }"#;
        let t = parse_trajectory(json).unwrap();
        assert_eq!(t.waypoints.len(), 2);
        assert_eq!(t.waypoints[0].time, 0.0);
        assert_eq!(t.waypoints[1].time, 1.5);
    }

    #[test]
    fn bad_json_surfaces_a_json_error() {
        match parse_world_state("not json") {
            Err(WireError::Json(_)) => {}
            other => panic!("expected JSON parse error, got {other:?}"),
        }
    }
}
