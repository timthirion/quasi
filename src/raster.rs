//! Real-time rasterized renderer.
//!
//! Forward-shaded triangle pipeline with per-instance model transforms,
//! targeting 60fps interactive scenes. R2 lands the instanced scene
//! shape; R3 adds line / point overlays; R4 ingests motum's serialized
//! scene + trajectory and adds a draggable goal handle.
//!
//! Shares nothing with [`pathtrace`](crate::pathtrace) below the
//! [`gpu`](crate::gpu) seam.

pub mod mesh;
pub mod scene;

#[cfg(target_arch = "wasm32")]
pub mod web;

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};

use crate::gpu::OrbitCamera;
use mesh::{cube_mesh, cylinder_mesh, sphere_mesh, Mesh, Vertex};
use scene::{translation, Instance, InstanceRaw, MeshHandle, OverlayVertex, Scene};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Per-frame uniforms passed to `forward.wgsl`.
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

struct GpuMesh {
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

/// Live raster renderer. Owns a geometry library and a [`Scene`] of
/// instances; the caller mutates the scene each frame as needed.
pub struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pipeline: wgpu::RenderPipeline,
    /// R3 overlay pipelines: `[topology][depth_mode]` where
    /// topology 0 = LineList, 1 = PointList; depth_mode 0 = Less
    /// (occluded by geometry), 1 = Always (drawn on top of everything).
    overlay_pipelines: [[wgpu::RenderPipeline; 2]; 2],
    bind_group: wgpu::BindGroup,
    frame_uniform_buf: wgpu::Buffer,

    meshes: Vec<GpuMesh>,
    instance_buf: wgpu::Buffer,
    instance_capacity: u64,
    /// Concat(depth_tested.lines, on_top.lines) packed each frame.
    overlay_line_buf: wgpu::Buffer,
    overlay_line_capacity: u64,
    /// Concat(depth_tested.points, on_top.points).
    overlay_point_buf: wgpu::Buffer,
    overlay_point_capacity: u64,

    depth_view: wgpu::TextureView,

    pub camera: OrbitCamera,
    pub scene: Scene,
}

const OVERLAY_LINE_LIST: usize = 0;
const OVERLAY_POINT_LIST: usize = 1;
const OVERLAY_DEPTH_TESTED: usize = 0;
const OVERLAY_ON_TOP: usize = 1;

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

/// Builds one overlay pipeline. Topology picks line-list vs point-list;
/// `depth_compare` picks depth-tested (Less) vs on-top (Always).
/// Depth writes are disabled either way so overlays don't occlude
/// triangle geometry drawn in later frames or passes.
fn build_overlay_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    topology: wgpu::PrimitiveTopology,
    depth_compare: wgpu::CompareFunction,
    color_format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[OverlayVertex::layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // Lines and points have no winding; back-face culling would
            // either be ignored or (on some backends) drop everything.
            cull_mode: None,
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn upload_mesh(device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh, label: &str) -> GpuMesh {
    let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (mesh.vertices.len() as u64) * Vertex::STRIDE,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&vertex_buf, 0, bytemuck::cast_slice(&mesh.vertices));

    let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (mesh.indices.len() as u64) * 2,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&index_buf, 0, bytemuck::cast_slice(&mesh.indices));

    GpuMesh {
        vertex_buf,
        index_buf,
        index_count: mesh.index_count(),
    }
}

/// Default mesh handles produced by the renderer's seeded geometry library
/// — small set of useful primitives, the order is stable.
#[derive(Copy, Clone, Debug)]
pub struct DefaultMeshes {
    pub cube: MeshHandle,
    pub sphere: MeshHandle,
    pub cylinder: MeshHandle,
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

        // R3 overlay shader — shared by all four overlay pipelines.
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raster/shaders/overlay.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("forward"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout(), InstanceRaw::layout()],
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

        let overlay_pipelines = [
            [
                build_overlay_pipeline(
                    &device,
                    &overlay_shader,
                    &layout,
                    wgpu::PrimitiveTopology::LineList,
                    wgpu::CompareFunction::Less,
                    config.format,
                    "overlay-line-depth-tested",
                ),
                build_overlay_pipeline(
                    &device,
                    &overlay_shader,
                    &layout,
                    wgpu::PrimitiveTopology::LineList,
                    wgpu::CompareFunction::Always,
                    config.format,
                    "overlay-line-on-top",
                ),
            ],
            [
                build_overlay_pipeline(
                    &device,
                    &overlay_shader,
                    &layout,
                    wgpu::PrimitiveTopology::PointList,
                    wgpu::CompareFunction::Less,
                    config.format,
                    "overlay-point-depth-tested",
                ),
                build_overlay_pipeline(
                    &device,
                    &overlay_shader,
                    &layout,
                    wgpu::PrimitiveTopology::PointList,
                    wgpu::CompareFunction::Always,
                    config.format,
                    "overlay-point-on-top",
                ),
            ],
        ];

        // Seed the geometry library with a small useful set: cube, sphere,
        // cylinder. Callers wanting custom meshes can extend it via
        // `register_mesh`.
        let cube = Mesh {
            ..cube_mesh(1.0, [1.0, 1.0, 1.0])
        };
        let sphere = sphere_mesh(0.5, 12, 24, [1.0, 1.0, 1.0]);
        let cylinder = cylinder_mesh(0.5, 1.0, 24, [1.0, 1.0, 1.0]);
        let meshes = vec![
            upload_mesh(&device, &queue, &cube, "mesh-cube"),
            upload_mesh(&device, &queue, &sphere, "mesh-sphere"),
            upload_mesh(&device, &queue, &cylinder, "mesh-cylinder"),
        ];
        let defaults = DefaultMeshes {
            cube: MeshHandle(0),
            sphere: MeshHandle(1),
            cylinder: MeshHandle(2),
        };

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

        let instance_capacity: u64 = 256;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raster-instances"),
            size: instance_capacity * InstanceRaw::STRIDE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let overlay_line_capacity: u64 = 256;
        let overlay_line_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay-lines"),
            size: overlay_line_capacity * OverlayVertex::STRIDE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_point_capacity: u64 = 256;
        let overlay_point_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay-points"),
            size: overlay_point_capacity * OverlayVertex::STRIDE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_view = create_depth(&device, config.width, config.height);

        let mut scene = Scene::new();
        // Default demo scene: a ground plane and three colored cubes in
        // a row, plus a sphere — gives `cargo run -- raster` something
        // immediately useful to look at.
        scene.push(Instance {
            mesh: defaults.cube,
            model: mul(translation(0.0, -0.5, 0.0), scene::scale(6.0, 0.05, 6.0)),
            tint: [0.25, 0.27, 0.32, 1.0],
        });
        for (i, color) in [
            [0.9, 0.3, 0.3, 1.0],
            [0.3, 0.85, 0.4, 1.0],
            [0.35, 0.55, 0.95, 1.0],
        ]
        .iter()
        .enumerate()
        {
            scene.push(Instance {
                mesh: defaults.cube,
                model: translation(-1.6 + 1.6 * i as f32, 0.0, 0.0),
                tint: *color,
            });
        }
        scene.push(Instance {
            mesh: defaults.sphere,
            model: translation(0.0, 1.2, -1.4),
            tint: [0.95, 0.9, 0.6, 1.0],
        });
        scene.push(Instance {
            mesh: defaults.cylinder,
            model: translation(2.0, 0.0, -1.0),
            tint: [0.75, 0.6, 0.95, 1.0],
        });

        // R3 demo overlay: world-space coordinate axes at the origin
        // (depth-tested so they hide behind geometry), plus bright dots
        // above each cube (on top of everything for goal-marker feel).
        scene
            .depth_tested_overlay
            .line([0.0, 0.01, 0.0], [2.0, 0.01, 0.0], [1.0, 0.2, 0.2, 1.0]);
        scene
            .depth_tested_overlay
            .line([0.0, 0.01, 0.0], [0.0, 2.0, 0.0], [0.2, 1.0, 0.2, 1.0]);
        scene
            .depth_tested_overlay
            .line([0.0, 0.01, 0.0], [0.0, 0.01, 2.0], [0.2, 0.5, 1.0, 1.0]);
        for x in [-1.6_f32, 0.0, 1.6] {
            scene
                .on_top_overlay
                .point([x, 1.2, 0.0], [1.0, 1.0, 0.4, 1.0]);
        }

        State {
            surface,
            device,
            queue,
            config,
            pipeline,
            overlay_pipelines,
            bind_group,
            frame_uniform_buf,
            meshes,
            instance_buf,
            instance_capacity,
            overlay_line_buf,
            overlay_line_capacity,
            overlay_point_buf,
            overlay_point_capacity,
            depth_view,
            camera: OrbitCamera::new(),
            scene,
        }
    }

    /// Register an additional mesh and return its handle.
    pub fn register_mesh(&mut self, mesh: &Mesh, label: &str) -> MeshHandle {
        let gpu = upload_mesh(&self.device, &self.queue, mesh, label);
        self.meshes.push(gpu);
        MeshHandle(self.meshes.len() as u32 - 1)
    }

    /// Handles for the three default meshes (cube, sphere, cylinder) seeded
    /// by `State::new`.
    pub fn default_meshes(&self) -> DefaultMeshes {
        DefaultMeshes {
            cube: MeshHandle(0),
            sphere: MeshHandle(1),
            cylinder: MeshHandle(2),
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

        // Bucket the scene's instances by mesh handle and pack them into a
        // single buffer so each mesh becomes one instanced draw.
        let mut by_mesh: BTreeMap<u32, Vec<InstanceRaw>> = BTreeMap::new();
        for inst in self.scene.instances() {
            by_mesh
                .entry(inst.mesh.0)
                .or_default()
                .push(InstanceRaw::from(inst));
        }
        let total: u64 = by_mesh.values().map(|v| v.len() as u64).sum();
        if total > self.instance_capacity {
            let new_capacity = total.next_power_of_two().max(self.instance_capacity * 2);
            self.instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raster-instances"),
                size: new_capacity * InstanceRaw::STRIDE,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_capacity;
        }
        // Pack: each mesh's instances are contiguous in the buffer. Record
        // the (mesh_handle, offset, count) draws so we can issue them in the
        // render pass.
        let mut packed: Vec<InstanceRaw> = Vec::with_capacity(total as usize);
        let mut draws: Vec<(u32, u32, u32)> = Vec::with_capacity(by_mesh.len());
        for (handle, list) in by_mesh {
            let start = packed.len() as u32;
            let count = list.len() as u32;
            packed.extend(list);
            draws.push((handle, start, count));
        }
        if !packed.is_empty() {
            self.queue
                .write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&packed));
        }

        // ---- Overlay buffer packing (R3) ----
        // Pack into `[depth_tested, on_top]` so each pipeline draws a
        // contiguous range. The buffer slice (rather than a dynamic
        // first-vertex offset) keeps wgpu's validation happy across
        // backends without a "min vertex" feature.
        let dt_lines = &self.scene.depth_tested_overlay.lines;
        let top_lines = &self.scene.on_top_overlay.lines;
        let line_total: usize = dt_lines.len() + top_lines.len();
        if line_total as u64 > self.overlay_line_capacity {
            let new_cap = (line_total as u64)
                .next_power_of_two()
                .max(self.overlay_line_capacity * 2);
            self.overlay_line_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay-lines"),
                size: new_cap * OverlayVertex::STRIDE,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.overlay_line_capacity = new_cap;
        }
        if line_total > 0 {
            let mut packed_lines: Vec<OverlayVertex> = Vec::with_capacity(line_total);
            packed_lines.extend_from_slice(dt_lines);
            packed_lines.extend_from_slice(top_lines);
            self.queue.write_buffer(
                &self.overlay_line_buf,
                0,
                bytemuck::cast_slice(&packed_lines),
            );
        }
        let line_dt_count = dt_lines.len() as u32;
        let line_top_count = top_lines.len() as u32;

        let dt_points = &self.scene.depth_tested_overlay.points;
        let top_points = &self.scene.on_top_overlay.points;
        let point_total: usize = dt_points.len() + top_points.len();
        if point_total as u64 > self.overlay_point_capacity {
            let new_cap = (point_total as u64)
                .next_power_of_two()
                .max(self.overlay_point_capacity * 2);
            self.overlay_point_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("overlay-points"),
                size: new_cap * OverlayVertex::STRIDE,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.overlay_point_capacity = new_cap;
        }
        if point_total > 0 {
            let mut packed_points: Vec<OverlayVertex> = Vec::with_capacity(point_total);
            packed_points.extend_from_slice(dt_points);
            packed_points.extend_from_slice(top_points);
            self.queue.write_buffer(
                &self.overlay_point_buf,
                0,
                bytemuck::cast_slice(&packed_points),
            );
        }
        let point_dt_count = dt_points.len() as u32;
        let point_top_count = top_points.len() as u32;

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
            for (handle, start, count) in draws {
                let mesh = match self.meshes.get(handle as usize) {
                    Some(m) => m,
                    None => continue,
                };
                rp.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
                let inst_byte_start = start as u64 * InstanceRaw::STRIDE;
                let inst_byte_end = inst_byte_start + count as u64 * InstanceRaw::STRIDE;
                rp.set_vertex_buffer(1, self.instance_buf.slice(inst_byte_start..inst_byte_end));
                rp.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint16);
                rp.draw_indexed(0..mesh.index_count, 0, 0..count);
            }

            // ---- Overlay passes (R3) ----
            // Order: depth-tested lines, depth-tested points,
            // on-top lines, on-top points. All inside the same render
            // pass so each reads the depth buffer the triangle pass
            // wrote; none of them write back, so successive overlay
            // passes don't occlude each other.
            let dt_line_range = 0..line_dt_count;
            let top_line_range = line_dt_count..(line_dt_count + line_top_count);
            let dt_point_range = 0..point_dt_count;
            let top_point_range = point_dt_count..(point_dt_count + point_top_count);

            if !dt_line_range.is_empty() {
                rp.set_pipeline(&self.overlay_pipelines[OVERLAY_LINE_LIST][OVERLAY_DEPTH_TESTED]);
                rp.set_vertex_buffer(0, self.overlay_line_buf.slice(..));
                rp.draw(dt_line_range, 0..1);
            }
            if !dt_point_range.is_empty() {
                rp.set_pipeline(&self.overlay_pipelines[OVERLAY_POINT_LIST][OVERLAY_DEPTH_TESTED]);
                rp.set_vertex_buffer(0, self.overlay_point_buf.slice(..));
                rp.draw(dt_point_range, 0..1);
            }
            if !top_line_range.is_empty() {
                rp.set_pipeline(&self.overlay_pipelines[OVERLAY_LINE_LIST][OVERLAY_ON_TOP]);
                rp.set_vertex_buffer(0, self.overlay_line_buf.slice(..));
                rp.draw(top_line_range, 0..1);
            }
            if !top_point_range.is_empty() {
                rp.set_pipeline(&self.overlay_pipelines[OVERLAY_POINT_LIST][OVERLAY_ON_TOP]);
                rp.set_vertex_buffer(0, self.overlay_point_buf.slice(..));
                rp.draw(top_point_range, 0..1);
            }
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

// ---------------------------------------------------------------------------
// Matrix math: hand-rolled to keep deps light. Native + WGSL agree on
// column-major mat4x4<f32> with the array-of-columns convention.
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
pub fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
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
    mul(proj, view)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_uniforms_size_matches_wgsl_layout() {
        assert_eq!(std::mem::size_of::<FrameUniforms>(), 112);
    }

    #[test]
    fn look_at_eye_maps_to_origin() {
        let m = look_at_rh([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
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

    #[test]
    fn mul_with_identity_is_a_noop() {
        let a = [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ];
        let id = scene::IDENTITY_MAT4;
        let ab = mul(a, id);
        let ba = mul(id, a);
        assert_eq!(ab, a);
        assert_eq!(ba, a);
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
