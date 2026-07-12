//! 3D mesh block partitioning and per-block padded accumulators — direct
//! generalization of `block.rs` to a `Grid3D`. Same false-sharing padding
//! discipline (`#[repr(align(64))]`) and static-spatial-partition-plus-
//! dtact-deflection design as the 2D `block::Block`/`partition_grid`.

use janus_core::grid3d::Grid3D;

/// Classification of a block's workload character (same semantics as the
/// 2D `block::BlockKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind3D {
    WaveDominated,
    ParticleDominated,
}

/// A rectangular-prism sub-region of a `Grid3D`, in cell-index space
/// `[i0,i1) x [j0,j1) x [k0,k1)`.
#[derive(Clone, Copy, Debug)]
pub struct Block3D {
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub k0: usize,
    pub k1: usize,
    /// Last-measured cost proxy (particle count), same role as
    /// `block::Block::last_particle_count`.
    pub last_particle_count: u64,
}

impl Block3D {
    #[inline]
    pub fn ncells(&self) -> usize {
        (self.i1 - self.i0) * (self.j1 - self.j0) * (self.k1 - self.k0)
    }

    /// Iterate the linear cell indices covered by this block on `grid`.
    pub fn cell_indices(&self, grid: &Grid3D) -> impl Iterator<Item = usize> + '_ {
        let nx = grid.nx;
        let plane = grid.plane_cells();
        (self.k0..self.k1).flat_map(move |k| {
            (self.j0..self.j1)
                .flat_map(move |j| (self.i0..self.i1).map(move |i| k * plane + j * nx + i))
        })
    }

    /// Classify this block from its last-measured particle count and a
    /// threshold (identical logic to `block::Block::classify`).
    pub fn classify(&self, particles_per_cell_threshold: f64) -> BlockKind3D {
        let ncells = self.ncells().max(1) as f64;
        let avg = self.last_particle_count as f64 / ncells;
        if avg > particles_per_cell_threshold {
            BlockKind3D::ParticleDominated
        } else {
            BlockKind3D::WaveDominated
        }
    }
}

/// Partition a `Grid3D` into a roughly-even `blocks_x * blocks_y * blocks_z`
/// tiling of rectangular-prism blocks (last block in each row/column/layer
/// absorbs any remainder) — 3D generalization of `block::partition_grid`.
/// Static spatial decomposition; dtact's work-deflection (not spatial
/// re-partitioning) adapts to load imbalance at runtime (same rationale as
/// the 2D partitioner, ENGINEERING_SPEC.md §7).
pub fn partition_grid_3d(
    grid: &Grid3D,
    blocks_x: usize,
    blocks_y: usize,
    blocks_z: usize,
) -> Vec<Block3D> {
    assert!(
        blocks_x > 0 && blocks_y > 0 && blocks_z > 0,
        "block counts must be positive"
    );
    let mut blocks = Vec::with_capacity(blocks_x * blocks_y * blocks_z);
    let base_w = grid.nx / blocks_x;
    let base_h = grid.ny / blocks_y;
    let base_d = grid.nz / blocks_z;
    for bz in 0..blocks_z {
        let k0 = bz * base_d;
        let k1 = if bz + 1 == blocks_z {
            grid.nz
        } else {
            k0 + base_d
        };
        for by in 0..blocks_y {
            let j0 = by * base_h;
            let j1 = if by + 1 == blocks_y {
                grid.ny
            } else {
                j0 + base_h
            };
            for bx in 0..blocks_x {
                let i0 = bx * base_w;
                let i1 = if bx + 1 == blocks_x {
                    grid.nx
                } else {
                    i0 + base_w
                };
                if i1 > i0 && j1 > j0 && k1 > k0 {
                    blocks.push(Block3D {
                        i0,
                        i1,
                        j0,
                        j1,
                        k0,
                        k1,
                        last_particle_count: 0,
                    });
                }
            }
        }
    }
    blocks
}

/// Per-block accumulator padded to a 64-byte cache line, 3D-block-scheduler
/// analog of `block::PaddedAccumulator` (same layout — a separate type only
/// because it is namespaced with `Block3D`'s telemetry role; the memory
/// layout requirement it satisfies is identical).
#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct PaddedAccumulator3D {
    pub particle_count: u64,
    pub wall_time_ns: u64,
    _pad: [u8; 48],
}

impl PaddedAccumulator3D {
    pub const fn new() -> Self {
        Self {
            particle_count: 0,
            wall_time_ns: 0,
            _pad: [0; 48],
        }
    }
}

impl Default for PaddedAccumulator3D {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_is_64_byte_aligned_and_sized_3d() {
        assert_eq!(std::mem::align_of::<PaddedAccumulator3D>(), 64);
        assert_eq!(std::mem::size_of::<PaddedAccumulator3D>(), 64);
    }

    #[test]
    fn partition_covers_every_cell_exactly_once_3d() {
        let grid = Grid3D::new(10, 7, 5, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        let blocks = partition_grid_3d(&grid, 3, 2, 2);
        let mut covered = vec![0u32; grid.ncells()];
        for b in &blocks {
            for c in b.cell_indices(&grid) {
                covered[c] += 1;
            }
        }
        assert!(
            covered.iter().all(|&c| c == 1),
            "every cell must be covered exactly once"
        );
    }

    #[test]
    fn classify_by_particle_density_3d() {
        let mut b = Block3D {
            i0: 0,
            i1: 4,
            j0: 0,
            j1: 4,
            k0: 0,
            k1: 4,
            last_particle_count: 0,
        };
        assert_eq!(b.classify(1.0), BlockKind3D::WaveDominated);
        b.last_particle_count = 10_000; // 10000/64 cells ~ 156 avg
        assert_eq!(b.classify(1.0), BlockKind3D::ParticleDominated);
    }
}
