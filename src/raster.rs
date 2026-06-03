//! Real-time rasterized renderer.
//!
//! Forward-shaded triangle pipeline targeting 60fps interactive scenes —
//! the second of Quasi's two pipelines (see plan `0002-realtime-
//! rasterization`). R1 lands a single shaded mesh; R2 grows this into an
//! instanced scene with a geometry library; R3 adds line / point overlays
//! for planner artifacts; R4 wires up the motum-shaped JSON ingestion and
//! a draggable goal handle.
//!
//! Shares nothing with [`pathtrace`](crate::pathtrace) below the
//! [`gpu`](crate::gpu) seam — different scene representation, different
//! shaders, different draw path.

pub mod mesh;

use bytemuck::{Pod, Zeroable};

use crate::gpu::OrbitCamera;
use mesh::{cube_mesh, Vertex};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Per-frame uniforms passed to `forward.wgsl`. Layout pinned with size
/// assertions; vec3s align on 16-byte boundaries with an explicit pad.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FrameUniforms {
    view_proj: [[f32; 4]; 4],
    sun_dir: [f32; 3],
    _pad0: f32,
    sun_color: [f32; 3],
    _pad1: f32,
    ambient: [f32; 3],
    _pad2: f32,
}

/// The rasterized renderer.
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    frame_uniform_buf: wgpu::Buffer,

    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,

    depth_view: wgpu::TextureView,

    pub camera: OrbitCamera,
}

fn create_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raster-depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

impl State {
    pub async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
    ) -> State {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable GPU adapter found");
        log::info!("raster adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("quasi-raster-device"),
                required_features: wgpu::Features::empty(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                ..Default::default()
            })
            .await
            .expect("failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        // Non-sRGB surface — the shader applies gamma itself, matching the
        // path tracer's present.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raster-frame-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("forward"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raster/shaders/forward.wgsl").into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raster-pipeline-layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // Test mesh: a unit cube colored a warm gray.
        let mesh = cube_mesh(1.0, [0.85, 0.7, 0.55]);
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster-vertices"),
            size: (mesh.vertices.len() as u64) * Vertex::STRIDE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&mesh.vertices));

        let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster-indices"),
            size: (mesh.indices.len() as u64) * 2,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&index_buf, 0, bytemuck::cast_slice(&mesh.indices));
        let index_count = mesh.index_count();

        let frame_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster-frame-uniforms"),
            size: std::mem::size_of::<FrameUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raster-frame-bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_uniform_buf.as_entire_binding(),
            }],
        });

        let depth_view = create_depth(&device, config.width, config.height);

        State {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_group,
            frame_uniform_buf,
            vertex_buf,
            index_buf,
            index_count,
            depth_view,
            camera: OrbitCamera::new(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 && (width != self.config.width || height != self.config.height) {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth(&self.device, width, height);
        }
    }

    pub fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                log::warn!("surface validation error acquiring frame");
                return;
            }
        };

        self.camera.dirty = false;

        let aspect = self.config.width as f32 / self.config.height as f32;
        let view_proj = view_projection_matrix(&self.camera, aspect);
        let uniforms = FrameUniforms {
            view_proj,
            sun_dir: normalize3([0.3, -0.8, -0.5]),
            _pad0: 0.0,
            sun_color: [1.0, 0.95, 0.9],
            _pad1: 0.0,
            ambient: [0.12, 0.13, 0.16],
            _pad2: 0.0,
        };
        self.queue
            .write_buffer(&self.frame_uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("raster-frame-encoder"),
            });
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("forward-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.09,
                            b: 0.11,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rp.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
            rp.draw_indexed(0..self.index_count, 0, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

// ---------------------------------------------------------------------------
// Matrix math: just enough for a view-projection. Kept inline to avoid
// pulling in a linear-algebra crate for this small need. Native + WGSL agree
// on column-major mat4x4<f32> with the array-of-columns convention.
// ---------------------------------------------------------------------------

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / len, v[1] / len, v[2] / len]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Right-handed look-at, output is column-major, view-space looks down -Z.
fn look_at_rh(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
    let f = normalize3([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let s = normalize3(cross3(f, up));
    let u = cross3(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

/// Right-handed perspective producing the WebGPU clip-space convention
/// (z ∈ [0, 1], y up, looking down -Z). Column-major.
fn perspective_rh_zo(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov_y_rad * 0.5).tan();
    let nf = 1.0 / (near - far);
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far * nf, -1.0],
        [0.0, 0.0, far * near * nf, 0.0],
    ]
}

/// Column-major 4x4 multiply.
fn mul_mat4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[k][row] * b[col][k];
            }
            out[col][row] = s;
        }
    }
    out
}

fn view_projection_matrix(camera: &OrbitCamera, aspect: f32) -> [[f32; 4]; 4] {
    let eye = camera.position();
    let target = camera.target;
    let view = look_at_rh(eye, target, [0.0, 1.0, 0.0]);
    let proj = perspective_rh_zo(camera.fov.to_radians(), aspect, 0.1, 100.0);
    mul_mat4(proj, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_uniforms_size_matches_wgsl_layout() {
        // mat4 (64) + (vec3+pad, 16) * 3 = 112 bytes.
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 112);
    }

    #[test]
    fn look_at_eye_maps_to_origin() {
        let m = look_at_rh([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Apply view to eye -> should land at origin in view space.
        let p = transform([0.0, 0.0, 5.0, 1.0], m);
        for k in 0..3 {
            assert!(p[k].abs() < 1e-5, "eye -> {p:?}");
        }
    }

    #[test]
    fn look_at_target_lies_along_negative_z_in_view_space() {
        let m = look_at_rh([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let p = transform([0.0, 0.0, 0.0, 1.0], m);
        assert!(p[0].abs() < 1e-5 && p[1].abs() < 1e-5);
        assert!(p[2] < 0.0, "target should be down -Z: {p:?}");
    }

    #[test]
    fn perspective_keeps_centered_point_at_origin_in_clip_xy() {
        let p = perspective_rh_zo(60f32.to_radians(), 1.0, 0.1, 100.0);
        let q = transform([0.0, 0.0, -5.0, 1.0], p);
        assert!(q[0].abs() < 1e-5 && q[1].abs() < 1e-5);
    }

    /// Column-major matrix * column vector.
    fn transform(v: [f32; 4], m: [[f32; 4]; 4]) -> [f32; 4] {
        let mut out = [0.0; 4];
        for row in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += m[k][row] * v[k];
            }
            out[row] = s;
        }
        out
    }
}
