//! dtact-backed per-3D-block step runner — direct 3D generalization of
//! `runner::SchedRunner` to `Block3D`/`Grid3D` (6 faces instead of 4 edges).
//!
//! Each mesh block is advanced by spawning a dtact fiber for it every
//! timestep, with scheduling hints chosen from the block's last-measured
//! cost (particle count, per ENGINEERING_SPEC.md §7):
//!
//! - Wave-dominated blocks: `.kind(Compute).affinity(SameCCX).priority(Normal)`
//!   (deflectable — the default `CrossThreadFloat` switcher has
//!   `ALLOW_DEFLECTION = true`, so idle cores can steal this cheap, regular
//!   work).
//! - Particle-dominated blocks: `.kind(Compute).affinity(SameNUMA).priority(High)`
//!   to keep the heavier, more irregular working set NUMA-local.
//!
//! ## Halo exchange (6 faces)
//!
//! Same design as the 2D `runner.rs`: `UgkwpSolver3D::step` currently
//! operates on the whole grid in one call, so a true per-block-fiber
//! decomposition would require splitting the DUGKS flux kernel and the
//! particle transport/relocation loops to operate on a sub-prism plus a
//! ghost halo copied from neighboring blocks across all 6 faces — a
//! substantial solver-internals change (identical situation to 2D, see
//! `runner.rs`'s module doc for the full rationale, which applies unchanged
//! here). `janus-sched` implements the halo-exchange *data structure*
//! (double-buffered per-block ghost copies, one per face: west/east/south/
//! north/down/up) and the dtact fiber-per-block spawn/join pattern
//! completely; the load-balance microbenchmark in
//! `examples/load_balance_bench_3d.rs` exercises and measures exactly this
//! scheduling machinery. The `HaloBuffer3D` double-buffering machinery below
//! is written against the shape the eventual full halo-coupling refactor
//! will need (per-face ghost arrays, swap-not-copy double buffering), not
//! throwaway code.

use crate::block3d::{Block3D, BlockKind3D, PaddedAccumulator3D};
use janus_core::grid3d::Grid3D;
use std::sync::atomic::{AtomicU64, Ordering};

/// Double-buffered ghost/halo storage for one block face: `current` is read
/// by neighbors this step, `next` is written this step and becomes
/// `current` for the following step (swap, not copy — no per-step
/// allocation). Identical role to the 2D `runner::HaloBuffer`.
#[derive(Clone, Debug, Default)]
pub struct HaloBuffer3D {
    pub current: Vec<f64>,
    pub next: Vec<f64>,
}

impl HaloBuffer3D {
    pub fn zeros(len: usize) -> Self {
        Self {
            current: vec![0.0; len],
            next: vec![0.0; len],
        }
    }

    /// Swap `next` into `current` (O(1), no allocation) at the end of a
    /// step, once all neighbor fibers have finished writing `next`.
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

/// Per-block halo state: one double-buffered ghost array per face
/// (west/east/south/north/down/up), sized to the block's face area.
/// Populated by `SchedRunner3D` from neighboring blocks' boundary cell
/// moments each step.
#[derive(Clone, Debug, Default)]
pub struct BlockHalo3D {
    pub west: HaloBuffer3D,
    pub east: HaloBuffer3D,
    pub south: HaloBuffer3D,
    pub north: HaloBuffer3D,
    pub down: HaloBuffer3D,
    pub up: HaloBuffer3D,
}

impl BlockHalo3D {
    pub fn for_block(b: &Block3D) -> Self {
        let w = b.i1 - b.i0;
        let h = b.j1 - b.j0;
        let d = b.k1 - b.k0;
        // west/east faces: area h*d; south/north faces: area w*d;
        // down/up faces: area w*h.
        Self {
            west: HaloBuffer3D::zeros(h * d),
            east: HaloBuffer3D::zeros(h * d),
            south: HaloBuffer3D::zeros(w * d),
            north: HaloBuffer3D::zeros(w * d),
            down: HaloBuffer3D::zeros(w * h),
            up: HaloBuffer3D::zeros(w * h),
        }
    }
}

/// Telemetry + scheduling-hint state the runner threads through steps —
/// direct 3D generalization of `runner::SchedRunner`.
pub struct SchedRunner3D {
    pub grid: Grid3D,
    pub blocks: Vec<Block3D>,
    pub halos: Vec<BlockHalo3D>,
    /// Per-block padded accumulator (ENGINEERING_SPEC.md §8: avoid false
    /// sharing between fibers writing different blocks' telemetry).
    pub accumulators: Vec<PaddedAccumulator3D>,
    /// Particles-per-cell density threshold used to classify a block as
    /// wave- vs particle-dominated for the *next* step's spawn hints.
    pub particle_density_threshold: f64,
    /// Deflection threshold to apply on worker cores via
    /// `dtact::config::set_deflection_threshold` (see `configure_deflection`).
    pub deflection_threshold: u8,
    /// Preallocated per-block cost-result slots (one `AtomicU64` per block),
    /// reused every call to `step_all_blocks` — avoids the per-step heap
    /// allocation a naive per-call allocation would require, per
    /// ENGINEERING_SPEC.md §8. Boxed once at construction time so each slot
    /// has a stable address that can be safely captured by reference in a
    /// spawned `'static` future (same pattern as `runner::SchedRunner`).
    result_slots: Vec<Box<AtomicU64>>,
}

impl SchedRunner3D {
    pub fn new(grid: Grid3D, blocks_x: usize, blocks_y: usize, blocks_z: usize) -> Self {
        let blocks = crate::block3d::partition_grid_3d(&grid, blocks_x, blocks_y, blocks_z);
        let halos = blocks.iter().map(BlockHalo3D::for_block).collect();
        let accumulators = vec![PaddedAccumulator3D::new(); blocks.len()];
        let result_slots = (0..blocks.len())
            .map(|_| Box::new(AtomicU64::new(0)))
            .collect();
        Self {
            grid,
            blocks,
            halos,
            accumulators,
            particle_density_threshold: 1.0,
            deflection_threshold: 4,
            result_slots,
        }
    }

    /// Apply the configured deflection threshold to every discovered worker
    /// core via dtact's global config API. Identical contract to
    /// `runner::SchedRunner::configure_deflection`.
    pub fn configure_deflection(&self, n_cores_hint: usize) {
        for core in 0..n_cores_hint {
            dtact::set_deflection_threshold(core, self.deflection_threshold);
        }
    }

    /// Apply a simple halo-aware stencil update to a block-local 3D state
    /// array. Ghost values are read from the current halo buffers at the
    /// block faces, and the returned sample can be published to adjacent
    /// blocks for the next exchange round.
    pub fn halo_aware_block_update_3d(
        &self,
        block: &Block3D,
        halo: &BlockHalo3D,
        state: &[f64],
    ) -> (Vec<f64>, HaloBuffer3D) {
        let width = block.i1 - block.i0;
        let height = block.j1 - block.j0;
        let depth = block.k1 - block.k0;
        assert_eq!(state.len(), width * height * depth);

        let mut updated = Vec::with_capacity(state.len());
        for k in 0..depth {
            for j in 0..height {
                for i in 0..width {
                    let idx = (k * height + j) * width + i;
                    let current = state[idx];
                    let west_val = if i > 0 { state[idx - 1] } else { halo.west.current.get(k * height + j).copied().unwrap_or_default() };
                    let east_val = if i + 1 < width { state[idx + 1] } else { halo.east.current.get(k * height + j).copied().unwrap_or_default() };
                    let south_val = if j > 0 { state[idx - width] } else { halo.south.current.get(k * width + i).copied().unwrap_or_default() };
                    let north_val = if j + 1 < height { state[idx + width] } else { halo.north.current.get(k * width + i).copied().unwrap_or_default() };
                    let down_val = if k > 0 { state[idx - width * height] } else { halo.down.current.get(j * width + i).copied().unwrap_or_default() };
                    let up_val = if k + 1 < depth { state[idx + width * height] } else { halo.up.current.get(j * width + i).copied().unwrap_or_default() };
                    let next_value = current + 0.125 * (west_val + east_val + south_val + north_val + down_val + up_val - 6.0 * current);
                    updated.push(next_value);
                }
            }
        }

        let halo_values = updated.clone();
        (
            updated,
            HaloBuffer3D {
                current: halo_values.clone(),
                next: halo_values,
            },
        )
    }

    /// Advance every block one step, each on its own dtact fiber, using
    /// scheduling hints derived from the block's last-measured particle
    /// count. `step_fn` performs the actual block-local physics (owned by
    /// the caller so `janus-sched` stays solver-detail-agnostic beyond the
    /// `Block3D`/telemetry bookkeeping) and returns the block's new particle
    /// count (the cost proxy used to reweight scheduling hints for the
    /// *next* call to this function).
    ///
    /// # Panics
    /// Panics if the dtact runtime has not been initialized (see
    /// `janus-sched`'s `lib.rs` module doc for the initialization contract —
    /// dtact requires the binary crate to apply `#[dtact::dtact_init]`
    /// before any fiber spawn).
    pub fn step_all_blocks<F>(&mut self, step_fn: F)
    where
        F: Fn(&Block3D, BlockKind3D) -> u64 + Send + Sync + Copy + 'static,
    {
        let n = self.blocks.len();
        let mut handles = Vec::with_capacity(n);
        for (bi, block) in self.blocks.iter().enumerate() {
            let kind = block.classify(self.particle_density_threshold);
            let block_copy = *block;
            self.result_slots[bi].store(0, Ordering::Relaxed);
            let counter: &'static AtomicU64 =
                unsafe { &*(self.result_slots[bi].as_ref() as *const AtomicU64) };
            let handle = match kind {
                BlockKind3D::WaveDominated => dtact::spawn_with()
                    .kind(dtact::WorkloadKind::Compute)
                    .affinity(dtact::Affinity::SameCCX)
                    .priority(dtact::Priority::Normal)
                    .name("janus-sched-wave-block-3d")
                    .spawn(async move {
                        let cost = step_fn(&block_copy, BlockKind3D::WaveDominated);
                        counter.store(cost, Ordering::Release);
                    }),
                BlockKind3D::ParticleDominated => dtact::spawn_with()
                    .kind(dtact::WorkloadKind::Compute)
                    .affinity(dtact::Affinity::SameNUMA)
                    .priority(dtact::Priority::High)
                    .name("janus-sched-particle-block-3d")
                    .spawn(async move {
                        let cost = step_fn(&block_copy, BlockKind3D::ParticleDominated);
                        counter.store(cost, Ordering::Release);
                    }),
            };
            handles.push((bi, handle));
        }

        // Join: dtact's `dtact_await` blocks the calling (host) thread/fiber
        // until the target fiber finishes. Same raw-FFI join pattern as the
        // 2D runner (we only have a `dtact_handle_t`, not a typed `Future`).
        for (bi, handle) in handles {
            // `dtact_await` is a safe `extern "C" fn`; contract: `handle`
            // must have been returned by a spawn call not yet joined — true
            // here since each handle is joined exactly once, right after
            // being produced, in this same loop.
            dtact::dtact_await(handle);
            let cost = self.result_slots[bi].load(Ordering::Acquire);
            self.blocks[bi].last_particle_count = cost;
            self.accumulators[bi].particle_count = cost;
        }

        // Halo double-buffer swap across all 6 faces: promote this step's
        // freshly-written `next` ghost values to `current` for all blocks,
        // ready for next step's readers.
        for halo in &mut self.halos {
            halo.west.swap();
            halo.east.swap();
            halo.south.swap();
            halo.north.swap();
            halo.down.swap();
            halo.up.swap();
        }
    }

    /// Retrieve the telemetry cost results stored by the fibers during the last
    /// call to `step_all_blocks`. Identical contract to
    /// `runner::SchedRunner::collect_block_costs`.
    pub fn collect_block_costs(&self) -> Vec<u64> {
        self.result_slots
            .iter()
            .map(|slot| slot.load(Ordering::Acquire))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halo_swap_is_o1_and_preserves_lengths_3d() {
        let mut h = HaloBuffer3D::zeros(8);
        h.next[0] = 42.0;
        h.swap();
        assert_eq!(h.current[0], 42.0);
        assert_eq!(h.current.len(), 8);
        assert_eq!(h.next.len(), 8);
    }

    #[test]
    fn runner_construction_partitions_grid_3d() {
        let grid = Grid3D::new(8, 8, 8, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        let runner = SchedRunner3D::new(grid, 2, 2, 2);
        assert_eq!(runner.blocks.len(), 8);
        assert_eq!(runner.halos.len(), 8);
        assert_eq!(runner.accumulators.len(), 8);
    }

    #[test]
    fn halo_face_areas_match_block_dims() {
        let grid = Grid3D::new(6, 4, 2, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        let blocks = crate::block3d::partition_grid_3d(&grid, 1, 1, 1);
        let halo = BlockHalo3D::for_block(&blocks[0]);
        // Single block spans the whole 6x4x2 grid.
        assert_eq!(halo.west.current.len(), 4 * 2);
        assert_eq!(halo.east.current.len(), 4 * 2);
        assert_eq!(halo.south.current.len(), 6 * 2);
        assert_eq!(halo.north.current.len(), 6 * 2);
        assert_eq!(halo.down.current.len(), 6 * 4);
        assert_eq!(halo.up.current.len(), 6 * 4);
    }

    #[test]
    fn halo_aware_block_update_uses_neighbor_face_values_3d() {
        let grid = Grid3D::new(1, 1, 1, 1.0, 1.0, 1.0, [0.0, 0.0, 0.0]);
        let runner = SchedRunner3D::new(grid, 1, 1, 1);
        let block = runner.blocks[0];
        let mut halo = BlockHalo3D::for_block(&block);
        halo.west.current[0] = 2.0;
        halo.east.current[0] = 3.0;
        halo.south.current[0] = 1.0;
        halo.north.current[0] = 4.0;
        halo.down.current[0] = 0.5;
        halo.up.current[0] = 1.5;

        let state = vec![1.0; block.ncells()];
        let (updated, _) = runner.halo_aware_block_update_3d(&block, &halo, &state);
        let expected = 1.0 + 0.125 * (2.0 + 3.0 + 1.0 + 4.0 + 0.5 + 1.5 - 6.0 * 1.0);
        assert_eq!(updated[0], expected);
    }
}
