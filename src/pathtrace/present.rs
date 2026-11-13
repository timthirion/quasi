//! Where the interactive renderer's present pass writes.
//!
//! [`State`](super::State) normally presents to a window or canvas
//! surface. For the widget smoke test it can instead present into an
//! offscreen texture that the test reads back, so the exact pipelines,
//! bind groups and passes the widget uses get exercised without a
//! window. That test exists because the widget once rendered nothing for
//! three months while CI stayed green (R0003 findings, 2026-09-15).

/// Format of the offscreen present target. Non-sRGB, matching the surface
/// path: the present shader applies gamma itself.
pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The present pass's destination: a configured surface, or an offscreen
/// texture with the same role.
pub(crate) enum PresentTarget {
    Surface {
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    },
    Offscreen {
        texture: wgpu::Texture,
    },
}

/// One acquired frame. Dropping an offscreen frame is a no-op; a surface
/// frame must be [`present`](Frame::present)ed.
pub(crate) enum Frame {
    Surface(wgpu::SurfaceTexture),
    Offscreen(wgpu::TextureView),
}

impl Frame {
    /// A view of the texture the present pass should write.
    pub fn view(&self) -> wgpu::TextureView {
        match self {
            Frame::Surface(t) => t
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
            Frame::Offscreen(v) => v.clone(),
        }
    }

    /// Hand the frame to the compositor (surface) or do nothing (offscreen).
    pub fn present(self) {
        if let Frame::Surface(t) = self {
            t.present();
        }
    }
}

fn create_offscreen(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("present-offscreen"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OFFSCREEN_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

impl PresentTarget {
    /// Configure `surface` for `device` at `width`×`height`, preferring a
    /// non-sRGB format.
    pub fn surface(
        surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> Self {
        let caps = surface.get_capabilities(adapter);
        // A non-sRGB surface: the present shader applies gamma itself.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(device, &config);
        PresentTarget::Surface { surface, config }
    }

    /// An offscreen texture of `width`×`height` in [`OFFSCREEN_FORMAT`].
    pub fn offscreen(device: &wgpu::Device, width: u32, height: u32) -> Self {
        PresentTarget::Offscreen {
            texture: create_offscreen(device, width, height),
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        match self {
            PresentTarget::Surface { config, .. } => config.format,
            PresentTarget::Offscreen { .. } => OFFSCREEN_FORMAT,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        match self {
            PresentTarget::Surface { config, .. } => (config.width, config.height),
            PresentTarget::Offscreen { texture } => (texture.width(), texture.height()),
        }
    }

    /// Reconfigure the surface or reallocate the texture at the new size.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        match self {
            PresentTarget::Surface { surface, config } => {
                config.width = width;
                config.height = height;
                surface.configure(device, config);
            }
            PresentTarget::Offscreen { texture } => {
                *texture = create_offscreen(device, width, height);
            }
        }
    }

    /// Acquire the next frame, or `None` when this frame should be skipped
    /// (surface outdated, lost, occluded, or timed out).
    pub fn acquire(&self, device: &wgpu::Device) -> Option<Frame> {
        match self {
            PresentTarget::Surface { surface, config } => match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t)
                | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(Frame::Surface(t)),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    surface.configure(device, config);
                    None
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                    None
                }
                wgpu::CurrentSurfaceTexture::Validation => {
                    log::warn!("surface validation error acquiring frame");
                    None
                }
            },
            PresentTarget::Offscreen { texture } => Some(Frame::Offscreen(
                texture.create_view(&wgpu::TextureViewDescriptor::default()),
            )),
        }
    }

    /// Read the offscreen texture back as tightly packed RGBA8 rows.
    /// Returns `None` for a surface target, which cannot be read back.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_rgba8(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Vec<u8>> {
        let PresentTarget::Offscreen { texture } = self else {
            return None;
        };
        let (w, h) = (texture.width(), texture.height());
        let unpadded = w * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present-readback"),
            size: u64::from(padded) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("present-readback"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            texture.size(),
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async failed"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mapped = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded * h) as usize);
        for row in mapped.chunks(padded as usize) {
            out.extend_from_slice(&row[..unpadded as usize]);
        }
        drop(mapped);
        buf.unmap();
        Some(out)
    }
}

/// Summary of a presented RGBA8 frame — the "is it actually rendering?"
/// probe. Mirrors the browser harness's `nonzeroFrac` / max / mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameStats {
    /// Fraction of pixels whose RGB is not all zero.
    pub nonzero_frac: f64,
    /// Mean of the R, G and B channels over all pixels, in 0..=255.
    pub mean: f64,
    /// Largest R, G or B value in the frame.
    pub max: u8,
}

impl FrameStats {
    /// Compute stats over tightly packed RGBA8 pixels; alpha is ignored.
    ///
    /// # Panics
    /// If `rgba.len()` is not a multiple of 4.
    pub fn from_rgba8(rgba: &[u8]) -> Self {
        assert!(
            rgba.len().is_multiple_of(4),
            "RGBA8 buffer length {} is not a multiple of 4",
            rgba.len()
        );
        let pixels = rgba.len() / 4;
        if pixels == 0 {
            return FrameStats {
                nonzero_frac: 0.0,
                mean: 0.0,
                max: 0,
            };
        }
        let (mut nonzero, mut sum, mut max) = (0usize, 0u64, 0u8);
        for px in rgba.chunks_exact(4) {
            let rgb = &px[..3];
            if rgb.iter().any(|&c| c != 0) {
                nonzero += 1;
            }
            sum += rgb.iter().map(|&c| u64::from(c)).sum::<u64>();
            max = max.max(*rgb.iter().max().unwrap_or(&0));
        }
        FrameStats {
            nonzero_frac: nonzero as f64 / pixels as f64,
            mean: sum as f64 / (pixels * 3) as f64,
            max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_frame_is_all_zero() {
        let s = FrameStats::from_rgba8(&[0, 0, 0, 255].repeat(16));
        assert_eq!(s.nonzero_frac, 0.0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.max, 0);
    }

    #[test]
    fn alpha_is_ignored() {
        // Opaque black must still count as black: a cleared target with
        // alpha 255 is exactly what a no-op renderer produces.
        let s = FrameStats::from_rgba8(&[0, 0, 0, 255, 0, 0, 0, 0]);
        assert_eq!(s.nonzero_frac, 0.0);
    }

    #[test]
    fn mixed_frame_stats() {
        // One black pixel, one (30, 60, 90) pixel.
        let s = FrameStats::from_rgba8(&[0, 0, 0, 255, 30, 60, 90, 255]);
        assert_eq!(s.nonzero_frac, 0.5);
        assert_eq!(s.mean, 30.0); // (30 + 60 + 90) / 6
        assert_eq!(s.max, 90);
    }

    #[test]
    fn empty_frame_is_zero() {
        let s = FrameStats::from_rgba8(&[]);
        assert_eq!(s.nonzero_frac, 0.0);
        assert_eq!(s.max, 0);
    }

    #[test]
    #[should_panic(expected = "not a multiple of 4")]
    fn ragged_buffer_panics() {
        FrameStats::from_rgba8(&[0, 0, 0]);
    }
}
