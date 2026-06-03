//! Orbit camera used by both rendering pipelines.

/// Orbit camera: spherical coordinates around a target.
#[derive(Debug)]
pub struct OrbitCamera {
    pub target: [f32; 3],
    pub distance: f32,
    pub azimuth: f32,
    pub elevation: f32,
    pub fov: f32,
    pub dragging: bool,
    pub last_cursor: (f64, f64),
    pub dirty: bool,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self::new()
    }
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self {
            target: [0.0, 1.0, 0.0],
            distance: 3.5,
            azimuth: 0.0,
            elevation: 0.0,
            fov: 40.0,
            dragging: false,
            last_cursor: (0.0, 0.0),
            dirty: true,
        }
    }

    pub fn position(&self) -> [f32; 3] {
        let ce = self.elevation.cos();
        [
            self.target[0] + self.distance * self.azimuth.sin() * ce,
            self.target[1] + self.distance * self.elevation.sin(),
            self.target[2] + self.distance * self.azimuth.cos() * ce,
        ]
    }

    pub fn direction(&self) -> [f32; 3] {
        let p = self.position();
        let d = [
            self.target[0] - p[0],
            self.target[1] - p[1],
            self.target[2] - p[2],
        ];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        [d[0] / len, d[1] / len, d[2] / len]
    }

    /// Records the cursor without rotating (use on press to set the drag origin).
    pub fn press(&mut self, x: f64, y: f64) {
        self.dragging = true;
        self.last_cursor = (x, y);
    }

    pub fn release(&mut self) {
        self.dragging = false;
    }

    pub fn on_cursor(&mut self, x: f64, y: f64) {
        if self.dragging {
            let dx = (x - self.last_cursor.0) as f32;
            let dy = (y - self.last_cursor.1) as f32;
            self.azimuth -= dx * 0.005;
            self.elevation = (self.elevation + dy * 0.005).clamp(-1.5, 1.5);
            self.dirty = true;
        }
        self.last_cursor = (x, y);
    }

    pub fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance - amount * 0.15).clamp(1.0, 10.0);
        self.dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-4, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn default_camera_looks_down_negative_z() {
        let c = OrbitCamera::new();
        close(c.position(), [0.0, 1.0, 3.5]);
        close(c.direction(), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn direction_is_normalized() {
        let mut c = OrbitCamera::new();
        c.azimuth = 0.7;
        c.elevation = 0.4;
        let d = c.direction();
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn zoom_clamps_distance() {
        let mut c = OrbitCamera::new();
        c.zoom(1000.0);
        assert!((c.distance - 1.0).abs() < 1e-6);
        c.zoom(-1000.0);
        assert!((c.distance - 10.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_only_while_dragging() {
        let mut c = OrbitCamera::new();
        c.on_cursor(10.0, 10.0); // not dragging: no rotation, just records cursor
        assert_eq!(c.azimuth, 0.0);
        c.press(10.0, 10.0);
        c.on_cursor(110.0, 10.0); // dx = 100 -> azimuth -= 100 * 0.005
        assert!((c.azimuth + 0.5).abs() < 1e-5);
    }

    #[test]
    fn elevation_is_clamped() {
        let mut c = OrbitCamera::new();
        c.press(0.0, 0.0);
        c.on_cursor(0.0, 1.0e6);
        assert!((c.elevation - 1.5).abs() < 1e-4);
    }
}
