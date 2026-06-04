# Textures

Reference test textures used by the path tracer once the
`PT-textures` milestone wires up UV-attributed `Vertex`es and a
`baseColorTexture` sampler. Until then these are inert — they sit on
disk waiting to be referenced.

## Files

- **`uv_checker_color.png`** (1024 × 1024, RGBA)
  Rainbow UV grid. Eight rows × eight columns of saturated checker
  cells; cell hue cycles around the colour wheel. The right choice
  when you want a visual eye-test for the texture-sampling code:
  rotating the texture, scaling it, or sampling it with the wrong
  filtering all show up immediately because the colours are
  asymmetric.

- **`uv_checker_mono.png`** (1024 × 1024, RGBA)
  Dark-gray-on-light-gray checker. The right choice when you want
  texture detail to read as surface shading rather than to dominate
  the rendered image — e.g. a Lambertian bunny with a subtle checker
  pattern that doesn't fight the red / green colour bleeding from
  the Cornell walls.

Both are 1 K (1024 × 1024) — enough resolution to keep alias-free at
roughly the same scales the existing demo scenes occupy. Higher-res
mip pyramids can be generated at load time if the path tracer ever
asks for them.

## Attribution

Both textures are from **Valle**'s *Custom UV Checker* set,
distributed publicly. The original filenames were
`CustomUVChecker_byValle_1K.png` (color) and
`CustomUVChecker_byValle_1K (1).png` (monochrome). Renamed for
clarity inside this repo; the bytes are unchanged.
