//! Mesh block partitioning and per-block padded accumulators.

use janus_core::grid::Grid2D;

/// Classification of a block's workload character, used to choose dtact
/// scheduling hints (ENGINEERING_SPEC.md §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// Predominantly continuum cells: cheap, regular wave-only work.
    WaveDominated,
    /// Predominantly rarefied cells: expensive, irregular particle work.
    ParticleDominated,
}

/// A rectangular sub-region of a `Grid2D`, in cell-index space
/// `[i0, i1) x [j0, j1)`.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    /// Last-measured cost proxy (particle count in this block, per spec
    /// §7: "particle count is a good proxy"). Updated every step by the
    /// caller after advancing the block; used to choose `BlockKind` for the
    /// *next* step's spawn.
    pub last_particle_count: u64,
}

impl Block {
    #[inline]
    pub fn ncells(&self) -> usize {
        (self.i1 - self.i0) * (self.j1 - self.j0)
    }

    /// Iterate the linear cell indices covered by this block on `grid`.
    pub fn cell_indices(&self, grid: &Grid2D) -> impl Iterator<Item = usize> + '_ {
        let nx = grid.nx;
        (self.j0..self.j1).flat_map(move |j| (self.i0..self.i1).map(move |i| j * nx + i))
    }

    /// Classify this block from its last-measured particle count and a
    /// threshold (particle-dominated if the *average* particle count per
    /// cell in the block exceeds `particles_per_cell_threshold`).
    pub fn classify(&self, particles_per_cell_threshold: f64) -> BlockKind {
        let ncells = self.ncells().max(1) as f64;
        let avg = self.last_particle_count as f64 / ncells;
        if avg > particles_per_cell_threshold {
            BlockKind::ParticleDominated
        } else {
            BlockKind::WaveDominated
        }
    }
}

/// Partition a grid into a roughly-even `blocks_x * blocks_y` tiling of
/// rectangular blocks (last block in each row/column absorbs any
/// remainder). Simple static decomposition — dtact's work-deflection
/// (not spatial re-partitioning) is what adapts to load imbalance at
/// runtime, per ENGINEERING_SPEC.md §7 ("dynamic, spatially-drifting load
/// is exactly what static MPI partitioning fails at and where dtact should
/// shine" — i.e. blocks stay spatially fixed; scheduling adapts around
/// them).
pub fn partition_grid(grid: &Grid2D, blocks_x: usize, blocks_y: usize) -> Vec<Block> {
    assert!(
        blocks_x > 0 && blocks_y > 0,
        "block counts must be positive"
    );
    let mut blocks = Vec::with_capacity(blocks_x * blocks_y);
    let base_w = grid.nx / blocks_x;
    let base_h = grid.ny / blocks_y;
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
            if i1 > i0 && j1 > j0 {
                blocks.push(Block {
                    i0,
                    i1,
                    j0,
                    j1,
                    last_particle_count: 0,
                });
            }
        }
    }
    blocks
}

/// Per-block accumulator padded to a 64-byte cache line
/// (ENGINEERING_SPEC.md §8: "pad per-block accumulators to 64 bytes
/// (`#[repr(align(64))]`), never let two fibers write adjacent cache lines
/// of the same array"). Used to accumulate per-block telemetry (e.g. step
/// wall-time, particle count) written concurrently by different fibers
/// without false-sharing one another's cache lines.
#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct PaddedAccumulator {
    pub particle_count: u64,
    pub wall_time_ns: u64,
    _pad: [u8; 48],
}

impl PaddedAccumulator {
    pub const fn new() -> Self {
        Self {
            particle_count: 0,
            wall_time_ns: 0,
            _pad: [0; 48],
        }
    }
}

// `[u8; 48]` does not implement `Default` (arrays > 32 elements only get
// element-wise trait impls up to length 32 in this edition/std version), so
// `#[derive(Default)]` on a struct containing it fails to compile
// (E0277). Mirrors the identical fix applied to `PaddedCounter` in
// `janus-kinetic/src/coupled.rs`.
impl Default for PaddedAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_is_64_byte_aligned_and_sized() {
        assert_eq!(std::mem::align_of::<PaddedAccumulator>(), 64);
        assert_eq!(std::mem::size_of::<PaddedAccumulator>(), 64);
    }

    #[test]
    fn partition_covers_every_cell_exactly_once() {
        let grid = Grid2D::new(10, 7, 1.0, 1.0, [0.0, 0.0]);
        let blocks = partition_grid(&grid, 3, 2);
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
    fn classify_by_particle_density() {
        let mut b = Block {
            i0: 0,
            i1: 4,
            j0: 0,
            j1: 4,
            last_particle_count: 0,
        };
        assert_eq!(b.classify(1.0), BlockKind::WaveDominated);
        b.last_particle_count = 1000; // 1000/16 cells = 62.5 avg
        assert_eq!(b.classify(1.0), BlockKind::ParticleDominated);
    }
}
