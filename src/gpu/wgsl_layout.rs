//! Test helpers that read struct layouts straight out of WGSL source.
//!
//! A hand-written `assert_eq!(size_of::<T>(), N)` pins the Rust side only.
//! If the WGSL struct drifts (a `vec3` pad that forces 16-byte alignment,
//! a new field on one side), `N` still matches and the bug ships. These
//! helpers parse the shader with naga and report the size naga computed,
//! so a test can compare the two sides directly.

/// Byte size (naga's `span`) of the struct named `name` in `src`.
///
/// Panics if the source does not parse or has no struct with that name,
/// which is the right failure for a test.
pub(crate) fn wgsl_struct_size(src: &str, name: &str) -> u32 {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("WGSL parse failed while looking for `{name}`: {e}"));
    let span = module
        .types
        .iter()
        .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span),
            _ => None,
        });
    span.unwrap_or_else(|| panic!("no struct `{name}` in WGSL source"))
}

/// Assert that the Rust type's size equals the WGSL struct's size.
pub(crate) fn assert_matches_wgsl<T>(src: &str, wgsl_name: &str) {
    let rust = std::mem::size_of::<T>();
    let wgsl = wgsl_struct_size(src, wgsl_name) as usize;
    assert_eq!(
        rust,
        wgsl,
        "{} is {rust} bytes but WGSL `{wgsl_name}` is {wgsl} bytes",
        std::any::type_name::<T>(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_pad_rounds_struct_to_32_bytes() {
        // The plan 0028 `PresentU` bug: a u32 followed by vec3<u32> is 32
        // bytes, not 16. If naga ever stops reporting it that way, every
        // layout test built on this helper would be meaningless.
        let src = "struct P { a: u32, b: vec3<u32>, }";
        assert_eq!(wgsl_struct_size(src, "P"), 32);
        let src = "struct P { a: u32, b: u32, c: u32, d: u32, }";
        assert_eq!(wgsl_struct_size(src, "P"), 16);
    }

    #[test]
    #[should_panic(expected = "no struct `Missing`")]
    fn missing_struct_panics() {
        wgsl_struct_size("struct P { a: u32, }", "Missing");
    }
}
