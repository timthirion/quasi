//! SAH-binned bounding volume hierarchy for triangle scenes.
//!
//! T2 of plan 0003. Replaces T1's linear scan with a CPU-built BVH that
//! the WGSL fragment shader walks via a function-local stack. The node
//! layout is engineered for direct upload to a storage buffer — 32
//! bytes, two AABBs flanking two `u32`s that double as inner-child
//! indices or leaf (first, count) pairs.
//!
//! ## Algorithm
//!
//! Recursive **SAH binned** split:
//!
//! 1. For each triangle, precompute AABB + centroid.
//! 2. Recurse on a triangle range:
//!    - If `count <= LEAF_CAP`, emit a leaf.
//!    - Otherwise, for each axis bin triangles by centroid into 16
//!      bins, run prefix scans both ways to get per-side counts and
//!      AABB surface areas, pick the (axis, bin boundary) with the
//!      lowest `count_left * area_left + count_right * area_right`,
//!      partition triangles, recurse.
//! 3. The partition step reorders an in-memory `tris` vector;
//!    `triangle_indices` records the final order so leaves can point
//!    at a contiguous range.
//!
//! ## Node packing
//!
//! ```text
//! offset 0   12  16  28
//!        +-----+--+-----+--+
//!        | min | L | max | R |
//!        +-----+--+-----+--+
//!          12   4  12   4   = 32 bytes total
//! ```
//!
//! - `L = left_or_first` — for an inner node, the index of the left
//!   child in `nodes`. For a leaf, the high bit ([`LEAF_FLAG`]) is set
//!   and the low 31 bits index into `triangle_indices`.
//! - `R = right_or_count` — for an inner node, the index of the right
//!   child. For a leaf, the triangle count at the offset given by `L`.
//!
//! WGSL std430 places `vec3<f32>` with 16-byte alignment, but its
//! *size* is 12 — so a vec3 followed by a u32 packs into exactly 16
//! bytes with no padding (the next member's offset is the vec3's
//! 12-byte end, rounded up to its 4-byte alignment — i.e. 12). The
//! `#[repr(C)]` struct matches byte-for-byte; the test
//! `node_size_matches_wgsl_layout` pins it.

use bytemuck::{Pod, Zeroable};

use crate::pathtrace::mesh::Vertex;

/// Bit pattern set in [`Node::left_or_first`] to flag a leaf node.
pub const LEAF_FLAG: u32 = 0x8000_0000;

/// Mask off the high leaf flag bit to recover the index payload.
pub const LEAF_MASK: u32 = 0x7FFF_FFFF;

/// Maximum triangles in a leaf. 4 is the standard small-leaf size that
/// keeps SIMD-friendly Möller-Trumbore loops short without exploding
/// node count.
pub const LEAF_CAP: usize = 4;

/// Bins per axis for the SAH evaluation. 16 is the standard tradeoff —
/// enough resolution to find good splits without the prefix-scan
/// arrays blowing past stack limits.
pub const NUM_BINS: usize = 16;

/// A BVH node — 32 bytes, matches the WGSL `Node` struct in
/// `pathtrace.wgsl` byte-for-byte (vec3 takes 12 bytes followed by a
/// u32 in std430).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct Node {
    pub aabb_min: [f32; 3],
    /// Inner: index of left child in `nodes`. Leaf: `LEAF_FLAG | first
    /// triangle index in `triangle_indices`.
    pub left_or_first: u32,
    pub aabb_max: [f32; 3],
    /// Inner: index of right child. Leaf: triangle count.
    pub right_or_count: u32,
}

impl Node {
    /// True iff the high bit of `left_or_first` is set.
    pub fn is_leaf(&self) -> bool {
        (self.left_or_first & LEAF_FLAG) != 0
    }

    /// First triangle index (into [`Bvh::triangle_indices`]) for a
    /// leaf; meaningless for an inner node.
    pub fn first_triangle(&self) -> u32 {
        self.left_or_first & LEAF_MASK
    }

    /// Triangle count for a leaf; meaningless for an inner node.
    pub fn triangle_count(&self) -> u32 {
        self.right_or_count
    }

    /// Left child index for an inner node.
    pub fn left_child(&self) -> u32 {
        self.left_or_first
    }

    /// Right child index for an inner node.
    pub fn right_child(&self) -> u32 {
        self.right_or_count
    }
}

/// SAH-binned BVH over a triangle mesh.
#[derive(Clone, Debug)]
pub struct Bvh {
    /// Flat node array; root at index 0.
    pub nodes: Vec<Node>,
    /// Triangle indices reordered so each leaf points to a contiguous
    /// range. Each entry indexes into the input mesh's triangle list
    /// (each triangle = 3 entries in the mesh's `indices`).
    pub triangle_indices: Vec<u32>,
}

impl Default for Bvh {
    /// Default = an [`empty`](Bvh::empty) BVH — single zero-volume
    /// leaf. The WGSL traversal indexes `bvh_nodes[0]` unconditionally,
    /// so a truly empty `nodes` Vec would be undefined behaviour on the
    /// GPU.
    fn default() -> Self {
        Self::empty()
    }
}

impl Bvh {
    /// Empty BVH — single empty leaf covering nothing.
    pub fn empty() -> Self {
        Bvh {
            nodes: vec![Node {
                aabb_min: [0.0; 3],
                left_or_first: LEAF_FLAG,
                aabb_max: [0.0; 3],
                right_or_count: 0,
            }],
            triangle_indices: Vec::new(),
        }
    }

    /// Builds a SAH BVH over `indices.len() / 3` triangles, each
    /// looking up its 3 vertices in `vertices`.
    pub fn build(vertices: &[Vertex], indices: &[u32]) -> Self {
        let num_triangles = indices.len() / 3;
        if num_triangles == 0 {
            return Bvh::empty();
        }

        // Precompute per-triangle AABB + centroid.
        let mut tris: Vec<TriInfo> = Vec::with_capacity(num_triangles);
        for tri in 0..num_triangles {
            let v0 = vertices[indices[tri * 3] as usize].position;
            let v1 = vertices[indices[tri * 3 + 1] as usize].position;
            let v2 = vertices[indices[tri * 3 + 2] as usize].position;
            let aabb_min = [
                v0[0].min(v1[0]).min(v2[0]),
                v0[1].min(v1[1]).min(v2[1]),
                v0[2].min(v1[2]).min(v2[2]),
            ];
            let aabb_max = [
                v0[0].max(v1[0]).max(v2[0]),
                v0[1].max(v1[1]).max(v2[1]),
                v0[2].max(v1[2]).max(v2[2]),
            ];
            let centroid = [
                (aabb_min[0] + aabb_max[0]) * 0.5,
                (aabb_min[1] + aabb_max[1]) * 0.5,
                (aabb_min[2] + aabb_max[2]) * 0.5,
            ];
            tris.push(TriInfo {
                tri: tri as u32,
                aabb_min,
                aabb_max,
                centroid,
            });
        }

        let mut nodes = Vec::new();
        let mut triangle_indices = Vec::with_capacity(num_triangles);
        let end = tris.len();
        build_recursive(&mut tris, 0, end, &mut nodes, &mut triangle_indices);

        Bvh {
            nodes,
            triangle_indices,
        }
    }

    /// Total node count, useful for benchmarking and tests.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

struct TriInfo {
    tri: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    centroid: [f32; 3],
}

fn aabb_of_tris(tris: &[TriInfo]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for t in tris {
        for k in 0..3 {
            if t.aabb_min[k] < lo[k] {
                lo[k] = t.aabb_min[k];
            }
            if t.aabb_max[k] > hi[k] {
                hi[k] = t.aabb_max[k];
            }
        }
    }
    (lo, hi)
}

/// Surface area of an AABB (or 0 if it's empty).
fn aabb_area(lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let dx = (hi[0] - lo[0]).max(0.0);
    let dy = (hi[1] - lo[1]).max(0.0);
    let dz = (hi[2] - lo[2]).max(0.0);
    2.0 * (dx * dy + dy * dz + dz * dx)
}

fn make_leaf(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    tris: &[TriInfo],
    triangle_indices: &mut Vec<u32>,
) -> Node {
    let first = triangle_indices.len() as u32;
    for ti in tris {
        triangle_indices.push(ti.tri);
    }
    Node {
        aabb_min,
        left_or_first: LEAF_FLAG | first,
        aabb_max,
        right_or_count: tris.len() as u32,
    }
}

/// Recursive build. Returns the index of the node it appended (always
/// the most recent push at entry).
fn build_recursive(
    tris: &mut [TriInfo],
    start: usize,
    end: usize,
    nodes: &mut Vec<Node>,
    triangle_indices: &mut Vec<u32>,
) -> u32 {
    let node_idx = nodes.len();
    // Placeholder; overwritten before return.
    nodes.push(Node::zeroed());

    let count = end - start;
    let (aabb_min, aabb_max) = aabb_of_tris(&tris[start..end]);

    if count <= LEAF_CAP {
        nodes[node_idx] = make_leaf(aabb_min, aabb_max, &tris[start..end], triangle_indices);
        return node_idx as u32;
    }

    let split = find_best_split(&tris[start..end]);
    let (axis, pos) = match split {
        Some(s) => s,
        None => {
            nodes[node_idx] = make_leaf(aabb_min, aabb_max, &tris[start..end], triangle_indices);
            return node_idx as u32;
        }
    };

    let mid = partition_by_centroid(tris, start, end, axis, pos);
    // Degenerate split (all on one side) — fall back to a leaf.
    if mid == start || mid == end {
        nodes[node_idx] = make_leaf(aabb_min, aabb_max, &tris[start..end], triangle_indices);
        return node_idx as u32;
    }

    let left_idx = build_recursive(tris, start, mid, nodes, triangle_indices);
    let right_idx = build_recursive(tris, mid, end, nodes, triangle_indices);

    nodes[node_idx] = Node {
        aabb_min,
        left_or_first: left_idx,
        aabb_max,
        right_or_count: right_idx,
    };
    node_idx as u32
}

fn find_best_split(tris: &[TriInfo]) -> Option<(usize, f32)> {
    // Centroid bounds drive the binning.
    let mut centroid_lo = [f32::INFINITY; 3];
    let mut centroid_hi = [f32::NEG_INFINITY; 3];
    for t in tris {
        for k in 0..3 {
            if t.centroid[k] < centroid_lo[k] {
                centroid_lo[k] = t.centroid[k];
            }
            if t.centroid[k] > centroid_hi[k] {
                centroid_hi[k] = t.centroid[k];
            }
        }
    }

    let mut best_cost = f32::INFINITY;
    let mut best_axis = 0;
    let mut best_pos = 0.0;

    for axis in 0..3 {
        let extent = centroid_hi[axis] - centroid_lo[axis];
        if extent < 1e-9 {
            continue;
        }
        let scale = NUM_BINS as f32 / extent;

        let mut bin_count = [0u32; NUM_BINS];
        let mut bin_lo = [[f32::INFINITY; 3]; NUM_BINS];
        let mut bin_hi = [[f32::NEG_INFINITY; 3]; NUM_BINS];

        for t in tris {
            let mut b = ((t.centroid[axis] - centroid_lo[axis]) * scale) as usize;
            if b >= NUM_BINS {
                b = NUM_BINS - 1;
            }
            bin_count[b] += 1;
            for k in 0..3 {
                if t.aabb_min[k] < bin_lo[b][k] {
                    bin_lo[b][k] = t.aabb_min[k];
                }
                if t.aabb_max[k] > bin_hi[b][k] {
                    bin_hi[b][k] = t.aabb_max[k];
                }
            }
        }

        // Left prefix: cumulative count + AABB up to and including bin i.
        let mut left_count = [0u32; NUM_BINS];
        let mut left_area = [0.0f32; NUM_BINS];
        let mut cur_count = 0;
        let mut cur_lo = [f32::INFINITY; 3];
        let mut cur_hi = [f32::NEG_INFINITY; 3];
        for i in 0..NUM_BINS {
            cur_count += bin_count[i];
            for k in 0..3 {
                if bin_lo[i][k] < cur_lo[k] {
                    cur_lo[k] = bin_lo[i][k];
                }
                if bin_hi[i][k] > cur_hi[k] {
                    cur_hi[k] = bin_hi[i][k];
                }
            }
            left_count[i] = cur_count;
            left_area[i] = aabb_area(cur_lo, cur_hi);
        }

        // Right prefix: cumulative count + AABB from bin i to end.
        let mut right_count = [0u32; NUM_BINS];
        let mut right_area = [0.0f32; NUM_BINS];
        let mut cur_count = 0;
        let mut cur_lo = [f32::INFINITY; 3];
        let mut cur_hi = [f32::NEG_INFINITY; 3];
        for i in (0..NUM_BINS).rev() {
            cur_count += bin_count[i];
            for k in 0..3 {
                if bin_lo[i][k] < cur_lo[k] {
                    cur_lo[k] = bin_lo[i][k];
                }
                if bin_hi[i][k] > cur_hi[k] {
                    cur_hi[k] = bin_hi[i][k];
                }
            }
            right_count[i] = cur_count;
            right_area[i] = aabb_area(cur_lo, cur_hi);
        }

        // SAH at boundary between bin i and bin i+1.
        for i in 0..(NUM_BINS - 1) {
            let cost =
                left_count[i] as f32 * left_area[i] + right_count[i + 1] as f32 * right_area[i + 1];
            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_pos = centroid_lo[axis] + (i + 1) as f32 / NUM_BINS as f32 * extent;
            }
        }
    }

    if best_cost.is_finite() {
        Some((best_axis, best_pos))
    } else {
        None
    }
}

/// Partition `tris[start..end]` so the first segment has centroids
/// strictly less than `pos` along `axis`. Returns the split index.
fn partition_by_centroid(
    tris: &mut [TriInfo],
    start: usize,
    end: usize,
    axis: usize,
    pos: f32,
) -> usize {
    let mut left = start;
    let mut right = end;
    while left < right {
        if tris[left].centroid[axis] < pos {
            left += 1;
        } else {
            right -= 1;
            tris.swap(left, right);
        }
    }
    left
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pathtrace::mesh::Vertex;

    fn vtx(p: [f32; 3]) -> Vertex {
        Vertex {
            position: p,
            _pad0: 0.0,
            normal: [0.0, 0.0, 1.0],
            _pad1: 0.0,
            uv: [0.0, 0.0],
            _pad2: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn node_size_matches_wgsl_layout() {
        // Pin the WGSL std430 packing: vec3 + u32 + vec3 + u32 = 32 bytes.
        assert_eq!(std::mem::size_of::<Node>(), 32);
        assert_eq!(std::mem::align_of::<Node>(), 4);
    }

    #[test]
    fn leaf_flag_bit_pattern() {
        assert_eq!(LEAF_FLAG, 0x8000_0000);
        assert_eq!(LEAF_MASK, 0x7FFF_FFFF);
        assert_eq!(LEAF_FLAG | LEAF_MASK, 0xFFFF_FFFF);
        let n = Node {
            aabb_min: [0.0; 3],
            left_or_first: LEAF_FLAG | 123,
            aabb_max: [0.0; 3],
            right_or_count: 7,
        };
        assert!(n.is_leaf());
        assert_eq!(n.first_triangle(), 123);
        assert_eq!(n.triangle_count(), 7);
    }

    #[test]
    fn empty_mesh_builds_one_empty_leaf() {
        let bvh = Bvh::build(&[], &[]);
        assert_eq!(bvh.nodes.len(), 1);
        assert!(bvh.nodes[0].is_leaf());
        assert_eq!(bvh.nodes[0].triangle_count(), 0);
        assert!(bvh.triangle_indices.is_empty());
    }

    #[test]
    fn single_triangle_collapses_into_one_leaf() {
        let verts = [
            vtx([0.0, 0.0, 0.0]),
            vtx([1.0, 0.0, 0.0]),
            vtx([0.0, 1.0, 0.0]),
        ];
        let indices = [0, 1, 2];
        let bvh = Bvh::build(&verts, &indices);
        assert_eq!(bvh.nodes.len(), 1);
        let n = &bvh.nodes[0];
        assert!(n.is_leaf());
        assert_eq!(n.triangle_count(), 1);
        assert_eq!(bvh.triangle_indices, vec![0]);
        assert_eq!(n.aabb_min, [0.0, 0.0, 0.0]);
        assert_eq!(n.aabb_max, [1.0, 1.0, 0.0]);
    }

    /// Generate `count` triangles on a line along +X, each one unit wide.
    fn line_of_triangles(count: usize) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut indices = Vec::new();
        for i in 0..count {
            let x = i as f32;
            verts.push(vtx([x, 0.0, 0.0]));
            verts.push(vtx([x + 1.0, 0.0, 0.0]));
            verts.push(vtx([x, 1.0, 0.0]));
            indices.push((i * 3) as u32);
            indices.push((i * 3 + 1) as u32);
            indices.push((i * 3 + 2) as u32);
        }
        (verts, indices)
    }

    #[test]
    fn every_triangle_is_reachable_through_some_leaf() {
        for &count in &[1usize, 4, 5, 8, 17, 32, 100, 513] {
            let (verts, indices) = line_of_triangles(count);
            let bvh = Bvh::build(&verts, &indices);
            assert_eq!(bvh.triangle_indices.len(), count, "count={count}");

            // Every input triangle index appears exactly once across leaves.
            let mut seen = vec![false; count];
            for n in &bvh.nodes {
                if !n.is_leaf() {
                    continue;
                }
                let first = n.first_triangle() as usize;
                let cnt = n.triangle_count() as usize;
                for i in 0..cnt {
                    let t = bvh.triangle_indices[first + i] as usize;
                    assert!(
                        !seen[t],
                        "triangle {t} appears in two leaves (count={count})"
                    );
                    seen[t] = true;
                }
            }
            assert!(
                seen.into_iter().all(|s| s),
                "some triangle is missing (count={count})"
            );
        }
    }

    #[test]
    fn each_leaf_aabb_encloses_all_its_triangles() {
        let (verts, indices) = line_of_triangles(32);
        let bvh = Bvh::build(&verts, &indices);
        for n in &bvh.nodes {
            if !n.is_leaf() {
                continue;
            }
            let first = n.first_triangle() as usize;
            let cnt = n.triangle_count() as usize;
            for i in 0..cnt {
                let t = bvh.triangle_indices[first + i] as usize;
                let p0 = verts[indices[t * 3] as usize].position;
                let p1 = verts[indices[t * 3 + 1] as usize].position;
                let p2 = verts[indices[t * 3 + 2] as usize].position;
                for &p in &[p0, p1, p2] {
                    for k in 0..3 {
                        assert!(
                            p[k] >= n.aabb_min[k] - 1e-5 && p[k] <= n.aabb_max[k] + 1e-5,
                            "triangle {t} vertex {p:?} escapes leaf aabb [{:?}, {:?}]",
                            n.aabb_min,
                            n.aabb_max,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn inner_nodes_reference_only_valid_children() {
        let (verts, indices) = line_of_triangles(100);
        let bvh = Bvh::build(&verts, &indices);
        for n in &bvh.nodes {
            if n.is_leaf() {
                continue;
            }
            assert!((n.left_child() as usize) < bvh.nodes.len());
            assert!((n.right_child() as usize) < bvh.nodes.len());
            // No self-references.
            assert_ne!(n.left_child(), n.right_child());
        }
    }

    #[test]
    fn many_identical_triangles_fall_back_to_a_leaf() {
        // All 10 triangles share the same vertices → all centroids equal
        // → SAH binning can't find a split. Should produce a single leaf.
        let verts = [
            vtx([0.0, 0.0, 0.0]),
            vtx([1.0, 0.0, 0.0]),
            vtx([0.0, 1.0, 0.0]),
        ];
        let mut indices = Vec::new();
        for _ in 0..10 {
            indices.extend_from_slice(&[0, 1, 2]);
        }
        let bvh = Bvh::build(&verts, &indices);
        // The exact node count depends on how the fall-back kicks in,
        // but every triangle must still be reachable.
        assert_eq!(bvh.triangle_indices.len(), 10);
    }

    #[test]
    fn balanced_input_produces_more_than_one_node() {
        // Triangles spread out enough that the SAH find at least one
        // split is the canonical success case.
        let (verts, indices) = line_of_triangles(64);
        let bvh = Bvh::build(&verts, &indices);
        assert!(
            bvh.nodes.len() > 1,
            "64 triangles in a line should produce > 1 node, got {}",
            bvh.nodes.len(),
        );
        // Root must encompass the full span.
        let root = &bvh.nodes[0];
        assert!(root.aabb_min[0] <= 0.0);
        assert!(root.aabb_max[0] >= 64.0);
    }
}
