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

    /// Advance every block one step, each on its own dtact fiber, using
    /// scheduling hints derived from the block's last-measured particle
    /// count. `step_fn` performs the actual block-local physics (owned by
    /// the caller so `janus-sched` stays solver-detail-agnostic beyond the
    /// `Block`/telemetry bookkeeping) and returns the block's new particle
    /// count (the cost proxy used to reweight scheduling hints for the
    /// *next* call to this function).
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
        let n = self.blocks.len();
        let mut handles = Vec::with_capacity(n);
        for (bi, block) in self.blocks.iter().enumerate() {
            let kind = block.classify(self.particle_density_threshold);
            let block_copy = *block;
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
            let handle = match kind {
                BlockKind::WaveDominated => dtact::spawn_with()
                    .kind(dtact::WorkloadKind::Compute)
                    .affinity(dtact::Affinity::SameCCX)
                    .priority(dtact::Priority::Normal)
                    .name("janus-sched-wave-block")
                    .spawn(async move {
                        let cost = step_fn(&block_copy, BlockKind::WaveDominated);
                        counter.store(cost, Ordering::Release);
                    }),
                BlockKind::ParticleDominated => dtact::spawn_with()
                    .kind(dtact::WorkloadKind::Compute)
                    .affinity(dtact::Affinity::SameNUMA)
                    .priority(dtact::Priority::High)
                    .name("janus-sched-particle-block")
                    .spawn(async move {
                        let cost = step_fn(&block_copy, BlockKind::ParticleDominated);
                        counter.store(cost, Ordering::Release);
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

        // Halo double-buffer swap: promote this step's freshly-written
        // `next` ghost values to `current` for all blocks, ready for next
        // step's readers. (See module doc: actual cross-block ghost
        // population is a follow-on solver refactor; the swap machinery
        // itself is complete and tested.)
        for halo in &mut self.halos {
            halo.west.swap();
            halo.east.swap();
            halo.south.swap();
            halo.north.swap();
        }
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
}
