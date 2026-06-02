// M0 sanity shader: a fullscreen triangle with a UV gradient.
//
// No vertex buffer — the vertex shader synthesizes an oversized triangle from
// the vertex index, the same trick we'll reuse for the path-tracing passes.

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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Red/green gradient with a constant blue — easy to eyeball that the pipeline,
    // surface format, and UV orientation are all wired up correctly.
    return vec4<f32>(in.uv.x, in.uv.y, 0.5, 1.0);
}
