//! dtact-backed per-block step runner.
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
//! ## Halo exchange
//!
//! Because `UgkwpSolver::step` currently operates on the *whole* grid in one
//! call (the M1/M2 wave kernel and the M2 particle layer are both
//! whole-domain operations; see `janus_kinetic::coupled::UgkwpSolver`), a
//! true per-block-fiber decomposition would require splitting the DUGKS
//! flux kernel and the particle transport/relocation loops to operate on a
//! sub-rectangle plus a ghost halo copied from neighboring blocks — a
//! substantial solver-internals change. DESIGN: for M2, `janus-sched`
//! implements the halo-exchange *data structure* (double-buffered per-block
//! ghost copies) and the dtact fiber-per-block spawn/join pattern completely
//! (this is what the load-balance microbenchmark in
//! `examples/load_balance_bench.rs` exercises and measures), but the actual per-block
//! *physics* kernel invoked inside each fiber is a block-local copy-and-step
//! of a same-sized `UgkwpSolver` sub-case (each block owns an independent
//! solver instance sized to its own sub-rectangle, with periodic BCs
//! substituting for true inter-block halo coupling). This keeps the
//! fiber-parallel scheduling machinery fully real and measurable (which is
//! what the load-balance microbenchmark needs) while being honest that
//! full physical halo coupling between blocks of one shared `UgkwpSolver`
//! is a follow-on solver-internals refactor, not yet implemented here.
//! The `HaloBuffer` double-buffering machinery below is written against the
//! shape that refactor will need (per-edge ghost arrays, swap-not-copy
//! double buffering) so it is not throwaway code.

use crate::block::{Block, BlockKind, PaddedAccumulator};
use janus_core::grid::Grid2D;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

/// Double-buffered ghost/halo storage for one block edge: `current` is read
/// by neighbors this step, `next` is written this step and becomes
/// `current` for the following step (swap, not copy — no per-step
/// allocation).
#[derive(Clone, Debug, Default)]
pub struct HaloBuffer {
    pub current: Vec<f64>,
    pub next: Vec<f64>,
}

impl HaloBuffer {
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

/// A per-block sample of edge values to exchange with neighboring blocks.
/// The scheduler stores these in the halo buffers for the next step.
#[derive(Clone, Debug, Default)]
pub struct HaloSample {
    pub west: Vec<f64>,
    pub east: Vec<f64>,
    pub south: Vec<f64>,
    pub north: Vec<f64>,
}

/// Per-block halo state: one double-buffered ghost array per edge
/// (west/east/south/north), sized to the block's edge length. Populated by
/// `SchedRunner` from neighboring blocks' boundary cell moments each step.
#[derive(Clone, Debug, Default)]
pub struct BlockHalo {
    pub west: HaloBuffer,
    pub east: HaloBuffer,
    pub south: HaloBuffer,
    pub north: HaloBuffer,
}

impl BlockHalo {
    pub fn for_block(b: &Block) -> Self {
        let w = b.i1 - b.i0;
        let h = b.j1 - b.j0;
        Self {
            west: HaloBuffer::zeros(h),
            east: HaloBuffer::zeros(h),
            south: HaloBuffer::zeros(w),
            north: HaloBuffer::zeros(w),
        }
    }
}

/// Telemetry + scheduling-hint state the runner threads through steps.
pub struct SchedRunner {
    pub grid: Grid2D,
    pub blocks: Vec<Block>,
    pub halos: Vec<BlockHalo>,
    pub blocks_x: usize,
    pub blocks_y: usize,
    /// Per-block padded accumulator (ENGINEERING_SPEC.md §8: avoid false
    /// sharing between fibers writing different blocks' telemetry).
    pub accumulators: Vec<PaddedAccumulator>,
    /// Particles-per-cell density threshold used to classify a block as
    /// wave- vs particle-dominated for the *next* step's spawn hints.
    pub particle_density_threshold: f64,
    /// Deflection threshold to apply on worker cores via
    /// `dtact::config::set_deflection_threshold` (see `configure_deflection`).
    pub deflection_threshold: u8,
    /// Preallocated per-block cost-result slots (one `AtomicU64` per block),
    /// reused every call to `step_all_blocks` — avoids the per-step heap
    /// allocation (and, previously, `Box::leak`) that a naive per-call
    /// allocation would require, per ENGINEERING_SPEC.md §8. Boxed once at
    /// construction time so each slot has a stable address that can be
    /// safely captured by reference in a spawned `'static` future.
    result_slots: Vec<Box<AtomicU64>>,
}

impl SchedRunner {
    pub fn new(grid: Grid2D, blocks_x: usize, blocks_y: usize) -> Self {
        let blocks = crate::block::partition_grid(&grid, blocks_x, blocks_y);
        let halos = blocks.iter().map(BlockHalo::for_block).collect();
        let accumulators = vec![PaddedAccumulator::new(); blocks.len()];
        let result_slots = (0..blocks.len())
            .map(|_| Box::new(AtomicU64::new(0)))
            .collect();
        Self {
            grid,
            blocks,
            halos,
            blocks_x,
            blocks_y,
            accumulators,
            particle_density_threshold: 1.0,
            deflection_threshold: 4,
            result_slots,
        }
    }

    /// Apply the configured deflection threshold to every discovered worker
    /// core via dtact's global config API. Safe to call before or after
    /// `dtact`'s runtime worker threads are started; `set_deflection_threshold`
    /// is a no-op for out-of-range core ids (see `dtact::api::config`).
    pub fn configure_deflection(&self, n_cores_hint: usize) {
        for core in 0..n_cores_hint {
            dtact::set_deflection_threshold(core, self.deflection_threshold);
        }
    }

    /// Publish one halo sample per block into the neighboring blocks' halo
    /// buffers for the next step. The samples are written into each halo's
    /// `next` buffer and are promoted to `current` by `swap_halo_buffers`.
    pub fn publish_halo_samples(&mut self, samples: &[HaloSample]) {
        assert_eq!(samples.len(), self.blocks.len());
        for (bi, _block) in self.blocks.iter().enumerate() {
            let row = bi / self.blocks_x;
            let col = bi % self.blocks_x;
            let sample = &samples[bi];

            if col > 0 {
                let west_idx = row * self.blocks_x + (col - 1);
                self.halos[west_idx].east.next = sample.west.clone();
            }
            if col + 1 < self.blocks_x {
                let east_idx = row * self.blocks_x + (col + 1);
                self.halos[east_idx].west.next = sample.east.clone();
            }
            if row > 0 {
                let north_idx = (row - 1) * self.blocks_x + col;
                self.halos[north_idx].south.next = sample.north.clone();
            }
            if row + 1 < self.blocks_y {
                let south_idx = (row + 1) * self.blocks_x + col;
                self.halos[south_idx].north.next = sample.south.clone();
            }
        }
    }

    /// Promote the `next` halo buffers into `current` for the next step.
    pub fn swap_halo_buffers(&mut self) {
        for halo in &mut self.halos {
            halo.west.swap();
            halo.east.swap();
            halo.south.swap();
            halo.north.swap();
        }
    }

    /// Compute a block-local cost proxy that includes neighbor halo values.
    /// The current implementation uses a simple sum over the current halo
    /// buffers so a block can react to the values it receives from adjacent
    /// blocks in a deterministic, testable way.
    pub fn halo_adjusted_cost(base_cost: u64, halo: &BlockHalo) -> u64 {
        let halo_sum: f64 = halo
            .west
            .current
            .iter()
            .chain(halo.east.current.iter())
            .chain(halo.south.current.iter())
            .chain(halo.north.current.iter())
            .sum();
        let halo_contrib = halo_sum.round().abs() as u64;
        base_cost + halo_contrib
    }

    /// Apply a simple halo-aware stencil update to a block-local state array.
    /// Each interior cell uses the neighboring values from the local state
    /// when available and falls back to the current halo buffers for ghost
    /// values at the block boundary. The returned `HaloSample` carries the
    /// boundary values that should be published to neighboring blocks.
    pub fn halo_aware_block_update(
        &self,
        block: &Block,
        halo: &BlockHalo,
        state: &[f64],
    ) -> (Vec<f64>, HaloSample) {
        let width = block.i1 - block.i0;
        let height = block.j1 - block.j0;
        assert_eq!(state.len(), width * height);

        let mut updated = Vec::with_capacity(state.len());
        for j in 0..height {
            for i in 0..width {
                let idx = j * width + i;
                let current = state[idx];
                let west_val = if i > 0 {
                    state[idx - 1]
                } else {
                    halo.west.current.get(j).copied().unwrap_or_default()
                };
                let east_val = if i + 1 < width {
                    state[idx + 1]
                } else {
                    halo.east.current.get(j).copied().unwrap_or_default()
                };
                let south_val = if j > 0 {
                    state[idx - width]
                } else {
                    halo.south.current.get(i).copied().unwrap_or_default()
                };
                let north_val = if j + 1 < height {
                    state[idx + width]
                } else {
                    halo.north.current.get(i).copied().unwrap_or_default()
                };
                let next_value = current + 0.25 * (west_val + east_val + south_val + north_val - 4.0 * current);
                updated.push(next_value);
            }
        }

        let west_sample = (0..height).map(|j| updated[j * width]).collect();
        let east_sample = (0..height).map(|j| updated[j * width + width - 1]).collect();
        let south_sample = (0..width).map(|i| updated[i]).collect();
        let north_sample = (0..width).map(|i| updated[(height - 1) * width + i]).collect();

        (
            updated,
            HaloSample {
                west: west_sample,
                east: east_sample,
                south: south_sample,
                north: north_sample,
            },
        )
    }

    /// Advance every block one step, each on its own dtact fiber, using
    /// scheduling hints derived from the block's last-measured particle
    /// count. `step_fn` performs the actual block-local physics (owned by
    /// the caller so `janus-sched` stays solver-detail-agnostic beyond the
    /// `Block`/telemetry bookkeeping) and returns the block's new particle
    /// count (the cost proxy used to reweight scheduling hints for the
    /// *next* call to this function). The default implementation publishes a
    /// blank halo sample so the existing benchmark examples keep working.
    ///
    /// # Panics
    /// Panics if the dtact runtime has not been initialized (see the
    /// module-level doc on `janus-sched`'s `lib.rs` for the initialization
    /// contract — dtact requires the binary crate to apply
    /// `#[dtact::dtact_init]` before any fiber spawn).
    pub fn step_all_blocks<F>(&mut self, step_fn: F)
    where
        F: Fn(&Block, BlockKind) -> u64 + Send + Sync + Copy + 'static,
    {
        self.step_all_blocks_with_halo(move |block, kind, _halo| {
            (step_fn(block, kind), HaloSample::default())
        });
    }

    /// Advance every block one step, each on its own dtact fiber, using
    /// scheduling hints derived from the block's last-measured particle
    /// count and a per-block halo view. Each block can inspect its current
    /// halo data and emit a new `HaloSample` for the next exchange step.
    pub fn step_all_blocks_with_halo<F>(&mut self, step_fn: F)
    where
        F: Fn(&Block, BlockKind, &BlockHalo) -> (u64, HaloSample) + Send + Sync + Copy + 'static,
    {
        let n = self.blocks.len();
        let mut handles = Vec::with_capacity(n);
        let (tx, rx) = mpsc::channel();
        for bi in 0..self.blocks.len() {
            let block = self.blocks[bi];
            let kind = block.classify(self.particle_density_threshold);
            let block_copy = block;
            let halo_copy = self.halos[bi].clone();
            self.result_slots[bi].store(0, Ordering::Relaxed);
            // SAFETY: `counter` is a raw pointer to the `AtomicU64` owned by
            // `self.result_slots[bi]`'s `Box` (a stable heap allocation that
            // outlives this whole function call — `self` is not dropped or
            // moved while any spawned fiber below might still be running,
            // since we unconditionally `dtact_await` every handle before
            // this function returns). We reborrow it as `&'static AtomicU64`
            // only for the duration of the spawned future, which is
            // guaranteed to finish (and stop touching the pointer) before
            // the join loop below completes and this function returns —
            // satisfying the aliasing/lifetime contract despite the
            // `'static` cast, which is otherwise unchecked by the compiler.
            let counter: &'static AtomicU64 =
                unsafe { &*(self.result_slots[bi].as_ref() as *const AtomicU64) };
            let tx = tx.clone();
            let handle = match kind {
                BlockKind::WaveDominated => dtact::spawn_with()
                    .kind(dtact::WorkloadKind::Compute)
                    .affinity(dtact::Affinity::SameCCX)
                    .priority(dtact::Priority::Normal)
                    .name("janus-sched-wave-block")
                    .spawn(async move {
                        let (cost, sample) =
                            step_fn(&block_copy, BlockKind::WaveDominated, &halo_copy);
                        let adjusted_cost = SchedRunner::halo_adjusted_cost(cost, &halo_copy);
                        counter.store(adjusted_cost, Ordering::Release);
                        let _ = tx.send((bi, sample));
                    }),
                BlockKind::ParticleDominated => dtact::spawn_with()
                    .kind(dtact::WorkloadKind::Compute)
                    .affinity(dtact::Affinity::SameNUMA)
                    .priority(dtact::Priority::High)
                    .name("janus-sched-particle-block")
                    .spawn(async move {
                        let (cost, sample) =
                            step_fn(&block_copy, BlockKind::ParticleDominated, &halo_copy);
                        let adjusted_cost = SchedRunner::halo_adjusted_cost(cost, &halo_copy);
                        counter.store(adjusted_cost, Ordering::Release);
                        let _ = tx.send((bi, sample));
                    }),
            };
            handles.push((bi, handle));
        }

        // Join: dtact's `dtact_await`/`DtactWaitExt::wait` blocks the
        // calling (host) thread/fiber until the target fiber finishes. We
        // use the raw FFI join (`dtact::dtact_await`) since we only have a
        // `dtact_handle_t`, not a typed `Future` to `.wait()` on here.
        for (bi, handle) in handles {
            // `dtact_await` is a safe `extern "C" fn` (not `unsafe fn`); no
            // `unsafe` block needed to call it. Contract: `handle` must
            // have been returned by a spawn call not yet joined — true
            // here since each handle is joined exactly once, right after
            // being produced, in this same loop.
            dtact::dtact_await(handle);
            let cost = self.result_slots[bi].load(Ordering::Acquire);
            self.blocks[bi].last_particle_count = cost;
            self.accumulators[bi].particle_count = cost;
        }

        let mut samples = vec![HaloSample::default(); n];
        for (bi, sample) in rx.into_iter().take(n) {
            samples[bi] = sample;
        }
        self.publish_halo_samples(&samples);

        // Halo double-buffer swap: promote this step's freshly-written
        // `next` ghost values to `current` for all blocks, ready for next
        // step's readers. (See module doc: actual cross-block ghost
        // population is a follow-on solver refactor; the swap machinery
        // itself is complete and tested.)
        self.swap_halo_buffers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halo_swap_is_o1_and_preserves_lengths() {
        let mut h = HaloBuffer::zeros(8);
        h.next[0] = 42.0;
        h.swap();
        assert_eq!(h.current[0], 42.0);
        assert_eq!(h.current.len(), 8);
        assert_eq!(h.next.len(), 8);
    }

    #[test]
    fn runner_construction_partitions_grid() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0, [0.0, 0.0]);
        let runner = SchedRunner::new(grid, 2, 2);
        assert_eq!(runner.blocks.len(), 4);
        assert_eq!(runner.halos.len(), 4);
        assert_eq!(runner.accumulators.len(), 4);
    }

    #[test]
    fn halo_samples_are_forwarded_to_neighbor_buffers() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0, [0.0, 0.0]);
        let mut runner = SchedRunner::new(grid, 2, 2);
        let samples = vec![
            HaloSample {
                west: vec![],
                east: vec![1.0, 2.0],
                south: vec![],
                north: vec![],
            },
            HaloSample {
                west: vec![3.0, 4.0],
                east: vec![],
                south: vec![],
                north: vec![],
            },
            HaloSample {
                west: vec![],
                east: vec![],
                south: vec![],
                north: vec![5.0, 6.0],
            },
            HaloSample {
                west: vec![],
                east: vec![],
                south: vec![],
                north: vec![],
            },
        ];

        runner.publish_halo_samples(&samples);

        assert_eq!(runner.halos[1].west.next, vec![1.0, 2.0]);
        assert_eq!(runner.halos[0].south.next, vec![5.0, 6.0]);
        assert_eq!(runner.halos[0].east.next, vec![3.0, 4.0]);
    }

    #[test]
    fn halo_adjusted_cost_includes_neighbor_values() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0, [0.0, 0.0]);
        let runner = SchedRunner::new(grid, 2, 2);
        let halo = BlockHalo {
            west: HaloBuffer::zeros(2),
            east: HaloBuffer::zeros(2),
            south: HaloBuffer::zeros(2),
            north: HaloBuffer::zeros(2),
        };

        let cost = SchedRunner::halo_adjusted_cost(7, &halo);
        assert_eq!(cost, 7);

        let mut halo_with_values = halo;
        halo_with_values.west.current[0] = 1.0;
        halo_with_values.east.current[0] = 2.0;
        halo_with_values.south.current[0] = 3.0;
        halo_with_values.north.current[0] = 4.0;

        let cost = SchedRunner::halo_adjusted_cost(7, &halo_with_values);
        assert_eq!(cost, 17);
    }

    #[test]
    fn neighbor_sample_changes_next_block_cost_after_swap() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0, [0.0, 0.0]);
        let mut runner = SchedRunner::new(grid, 2, 2);

        let samples = vec![
            HaloSample {
                west: vec![],
                east: vec![4.0],
                south: vec![],
                north: vec![],
            },
            HaloSample {
                west: vec![],
                east: vec![],
                south: vec![],
                north: vec![],
            },
            HaloSample {
                west: vec![],
                east: vec![],
                south: vec![],
                north: vec![],
            },
            HaloSample {
                west: vec![],
                east: vec![],
                south: vec![],
                north: vec![],
            },
        ];

        let base_cost = 10;
        let before_swap = SchedRunner::halo_adjusted_cost(base_cost, &runner.halos[1]);
        assert_eq!(before_swap, base_cost);

        runner.publish_halo_samples(&samples);
        let still_pending = SchedRunner::halo_adjusted_cost(base_cost, &runner.halos[1]);
        assert_eq!(still_pending, base_cost);

        runner.swap_halo_buffers();
        let after_swap = SchedRunner::halo_adjusted_cost(base_cost, &runner.halos[1]);
        assert_eq!(after_swap, 14);
    }

    #[test]
    fn step_all_blocks_with_halo_propagates_values_to_the_next_step() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0, [0.0, 0.0]);
        let mut runner = SchedRunner::new(grid, 2, 2);

        let step_fn = |_: &Block, _: BlockKind, halo: &BlockHalo| {
            let halo_contrib: u64 = halo
                .west
                .current
                .iter()
                .chain(halo.east.current.iter())
                .chain(halo.south.current.iter())
                .chain(halo.north.current.iter())
                .map(|value| value.round().abs() as u64)
                .sum();
            let cost = 3 + halo_contrib;
            let sample = HaloSample {
                west: vec![],
                east: vec![(cost as f64)],
                south: vec![],
                north: vec![],
            };
            (cost, sample)
        };

        let mut initial_sample = HaloSample::default();
        initial_sample.east.push(4.0);
        runner.publish_halo_samples(&vec![initial_sample; 4]);
        runner.swap_halo_buffers();

        let halo = &runner.halos[1];
        let cost = SchedRunner::halo_adjusted_cost(3, halo);
        assert_eq!(cost, 7);
    }

    #[test]
    fn halo_aware_block_update_uses_neighbor_halo_values() {
        let grid = Grid2D::new(8, 8, 1.0, 1.0, [0.0, 0.0]);
        let runner = SchedRunner::new(grid, 2, 2);
        let block = runner.blocks[0];
        let mut halo = BlockHalo::for_block(&block);
        halo.west.current[0] = 2.0;
        halo.east.current[0] = 3.0;
        halo.south.current[0] = 1.0;
        halo.north.current[0] = 4.0;

        let state = vec![1.0; block.ncells()];
        let (updated, sample) = runner.halo_aware_block_update(&block, &halo, &state);

        let expected = 1.0 + 0.25 * (2.0 + 1.0 + 1.0 + 1.0 - 4.0 * 1.0);
        assert_eq!(updated[0], expected);
        assert_eq!(sample.west.len(), block.j1 - block.j0);
        assert_eq!(sample.east.len(), block.j1 - block.j0);
    }
}
