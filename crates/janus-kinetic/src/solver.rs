//! DUGKS (Discrete Unified Gas-Kinetic Scheme) solver, 2D structured grid,
//! pure-wave part of UGKWP, reduced (g,h) two-distribution formulation,
//! Shakhov collision.
//!
//! Reference: Guo, Z., Xu, K., Wang, R., "Discrete unified gas kinetic
//! scheme for all Knudsen number flows: Low-speed isothermal case", Phys.
//! Rev. E 88, 033305 (2013).
//!
//! Shakhov collision model reference: Shakhov, E. M., "Generalization of the
//! Krook kinetic relaxation equation", Fluid Dynamics 3, 95 (1968).
//!
//! Reduced (g,h) two-distribution formulation reference: Xu, K., Huang,
//! J.-C., "A unified gas-kinetic scheme for continuum and rarefied flows",
//! J. Comput. Phys. 229, 7747-7764 (2010). This is what retires the M1
//! `// PHYSICS-DEBT:` marker: a 2D-velocity-space simulation now keeps the
//! correct monatomic 3 translational DOF and gamma = 5/3 (see
//! `crate::maxwellian` module docs for the full derivation).
//!
//! UGKWP wave/particle coupling (M2): Liu, S., Zhu, Y., Xu, K., "A unified
//! gas kinetic wave-particle method I: Continuum and rarefied gas dynamics",
//! J. Comput. Phys. 401, 108977 (2020). The wave step implemented here is
//! the deterministic "wave" side of UGKWP; per-cell particle representation
//! lives in `crate::particles`; `janus-kinetic::coupled` combines them,
//! weighting the split by the local collision probability
//! `p_free = exp(-dt/tau)` every step (see `crate::coupled` module docs) —
//! this `DugksSolver` is the deterministic sub-step advancing the
//! complementary `1-p_free` wave fraction of every cell, NOT a standalone
//! substitute for particles anywhere in the default path.
//!
//! RETIRED PHYSICS-DEBT (was: "no standalone DUGKS-only opt-in exists"):
//! `DugksSolver` (this type) is driven from inside `UgkwpSolver::step` as the
//! full UGKWP wave sub-step by default. The spec (§2) permits DUGKS to
//! *additionally* exist as an explicit, user-selectable pure-continuum
//! fast-path kernel (no particle sampling at all) for known-continuum-only
//! cases; this now exists as `coupled::FluxKernel::Dugks` — set
//! `UgkwpSolver::kernel = FluxKernel::Dugks` to run only this type's `step`
//! every UGKWP step, with the particle layer never populated. The default
//! constructed by `UgkwpSolver::new` remains `FluxKernel::Ugkwp` (full
//! UGKWP); nothing silently falls back to `Dugks`. See `coupled::FluxKernel`
//! docs and `coupled::tests::{default_kernel_is_full_ugkwp,
//! dugks_opt_in_kernel_never_spawns_particles}`.
//!
//! ## DUGKS algorithm summary (per cell, per face)
//!
//! DUGKS's defining trick is a *half-time-step* reconstruction of the
//! distribution at a cell face using the exact (closed-form, since BGK-type
//! relaxation is linear in `g`/`h` at fixed `g_eq`/`h_eq`) solution of the
//! kinetic equation along its characteristic. For a face with upwind cell
//! value `g_up` (resp. `h_up`) and upwind equilibrium `geq_up`/`heq_up`
//! (evaluated at the upwind cell's macro state) and upwind relaxation time
//! `tau_up`:
//!
//! ```text
//! g_face = (tau_up * g_up + 0.5*dt*geq_up) / (tau_up + 0.5*dt)
//! h_face = (tau_up * h_up + 0.5*dt*heq_up) / (tau_up + 0.5*dt)
//! ```
//!
//! ## MUSCL slope-limited spatial reconstruction (hardening pass)
//!
//! `g_up`/`h_up` above were originally the plain first-order-upwind cell
//! value (no slope reconstruction). This has been upgraded to a
//! second-order-in-space van Leer MUSCL reconstruction (van Leer, B.,
//! "Towards the ultimate conservative difference scheme. V. A second-order
//! sequel to Godunov's method", J. Comput. Phys. 32, 101-136 (1979)):
//! instead of taking the upwind CELL-CENTER value of `g`/`h` directly, we
//! extrapolate half a cell-width from the upwind cell center to the face
//! using a slope-limited estimate of `d(g)/dx` (resp. `h`) built from the
//! upwind cell's two neighbors along the face-normal direction, THEN
//! evaluate the DUGKS half-step formula using that reconstructed face value
//! in place of the raw cell-center value (`g_up`/`h_up` in the formula
//! above). This keeps the finite-volume update exactly conservative: the
//! reconstructed face state feeds the *same* flux formula and the *same*
//! symmetric add/subtract FV accumulation as before, so nothing beyond the
//! spatial order of the face value changes. The limiter is the standard van
//! Leer harmonic limiter,
//! `phi(r) = (r + |r|) / (1 + |r|)`, applied to the ratio `r` of the
//! downwind-side to upwind-side one-sided differences, which reduces
//! smoothly to zero (first-order upwind) at local extrema/discontinuities
//! (TVD, no new oscillations) and to the unlimited central slope in smooth
//! regions. At a domain boundary (no far-upwind neighbor available, e.g. the
//! last interior cell before a non-periodic ghost) the reconstruction
//! degrades gracefully to first-order upwind (slope = 0) — DESIGN: a ghost
//! cell's `g`/`h` is a BC-synthesized value, not a real upwind neighbor, so a
//! genuine slope through it is not meaningful; first-order there is the
//! simplest correct choice and only affects the single boundary-adjacent
//! layer of cells.
//!
//! The full per-step algorithm (unchanged in structure from M1, now carrying
//! two distributions):
//! 1. Precompute per-cell `tau` from the pre-step macro state.
//! 2. For every face, pick the upwind macro state, evaluate the Shakhov
//!    (g,h) equilibrium there, and reconstruct the half-step face values.
//! 3. Accumulate the finite-volume flux balance into scratch copies of `f`/`h`.
//! 4. Swap the scratch buffers into `dist.f`/`dist.h`, recompute moments.
//! 5. Apply the (implicit, closed-form) full-step Shakhov relaxation at the
//!    freshly updated cell moments for both `g` and `h`.
//! 6. Recompute moments again.
//!
//! ## Borrow-checker design note
//!
//! Every face-update computes upwind macro state and the resulting flux into
//! small local variables/arrays *first*, and only afterwards writes into
//! `self.dist_scratch.f`/`.h` (never aliased with the local read-only slices
//! at the point of the write).

use crate::bc;
use crate::collision::{Collision, Shakhov};
use crate::maxwellian::DOF;
use janus_core::config::{BoundaryKind, CaseConfig, Edge};
use janus_core::distribution::Distribution;
use janus_core::fields::MacroFields;
use janus_core::grid::Grid2D;

/// Generic explicit time-stepper interface.
pub trait TimeStepper {
    fn step(&mut self, dt: f64);
}

/// Selectable time-integration scheme for `DugksSolver::step_scheme`
/// (2D) / `DugksSolver3D::step_scheme` (3D).
///
/// - `Euler`: the original single-stage update (unchanged; still the
///   default via `DugksSolver::new`/`DugksSolver3D::new`).
/// - `Rk2`: Shu-Osher SSP-RK2 (Heun's method) applied via operator
///   splitting — see the `// DESIGN:` comment on `step_scheme` for exactly
///   what is/isn't sub-stepped and why.
///
/// Reference: Shu, C.-W., Osher, S., "Efficient implementation of
/// essentially non-oscillatory shock-capturing schemes", J. Comput. Phys.
/// 77, 439-471 (1988) (introduces the Shu-Osher SSP-RK form); Gottlieb, S.,
/// Shu, C.-W., Tadmor, E., "Strong stability-preserving high-order time
/// discretization methods", SIAM Rev. 43, 89-112 (2001) (the standard SSP-RK2
/// coefficients used here: `u1 = u^n + dt*L(u^n)`,
/// `u^{n+1} = 0.5*u^n + 0.5*(u1 + dt*L(u1))`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TimeScheme {
    #[default]
    Euler,
    Rk2,
}

#[derive(Clone, Copy)]
enum BoundaryKindResolved {
    Periodic,
    Other,
}

/// Local, upwind-selected macro state used to evaluate a face's Shakhov
/// (g,h) equilibrium. Kept as a small plain struct (not borrowed from
/// `self`) so it can be computed before any `&mut self` write happens.
#[derive(Clone, Copy)]
struct UpwindState {
    g: f64,
    h: f64,
    rho: f64,
    u: [f64; 2],
    t: f64,
    q: [f64; 2],
    tau: f64,
}

/// The DUGKS solver state: grid, gas/BC config, discrete velocity (g,h)
/// distribution pair (double-buffered), macro fields, and preallocated
/// scratch (face flux / ghost / relaxation-time buffers) so the step loop
/// performs zero heap allocation beyond what is already preallocated at
/// `new()`.
pub struct DugksSolver {
    pub grid: Grid2D,
    pub gas_r: f64,
    pub mu_ref: f64,
    pub t_ref: f64,
    pub omega: f64,
    pub collision: Shakhov,
    pub dist: Distribution,
    dist_scratch: Distribution, // double buffer, same shape as dist
    pub fields: MacroFields,
    bcs: [BoundaryKindResolved; 4], // west, east, south, north
    /// Selectable explicit time-integration scheme (see `TimeScheme` docs).
    /// Defaults to `TimeScheme::Euler` (original behavior, unchanged) via
    /// `new()`; set directly to opt into `TimeScheme::Rk2`.
    pub scheme: TimeScheme,
    // Scratch buffers reused every step (no per-step allocation):
    tau_scratch: Vec<f64>, // len = ncells, relaxation time at pre-step state
    face_flux_g: Vec<f64>, // len = nv, reused per face
    face_flux_h: Vec<f64>, // len = nv, reused per face
    ghost_buf_g: Vec<f64>, // len = nv, reused per boundary face
    ghost_buf_h: Vec<f64>, // len = nv, reused per boundary face
    // RK2-only scratch (preallocated once here so `step_scheme(Rk2)` performs
    // zero heap allocation in the hot path, same discipline as the rest of
    // this struct's scratch buffers): holds the stage-1 (`u1 = u^n +
    // dt*L(u^n)`) distribution state.
    rk2_stage_f: Vec<f64>, // len = ncells * nv
    rk2_stage_h: Vec<f64>, // len = ncells * nv
    // Relaxation scratch: per-cell discrete equilibrium (g_eq, h_eq) at the
    // velocity nodes, reused by the discretely-conservative relaxation
    // correction (see `relax_conservative`). Length nv, no per-step alloc.
    eq_g_scratch: Vec<f64>,
    eq_h_scratch: Vec<f64>,
}

impl DugksSolver {
    pub fn new(config: &CaseConfig, dist: Distribution) -> Self {
        let ncells = config.grid.ncells();
        let nv = dist.nv;
        let fields = MacroFields::zeros(ncells);
        let dist_scratch = Distribution::zeros(ncells, dist.vgrid.clone(), dist.vw.clone());
        let resolve = |k: &BoundaryKind| match k {
            BoundaryKind::Periodic => BoundaryKindResolved::Periodic,
            _ => BoundaryKindResolved::Other,
        };
        Self {
            grid: config.grid,
            gas_r: config.gas.r_gas,
            mu_ref: config.gas.mu_ref,
            t_ref: config.gas.t_ref,
            omega: config.gas.vhs_omega,
            collision: Shakhov::new(config.gas.prandtl),
            dist,
            dist_scratch,
            fields,
            bcs: [
                resolve(&config.bcs.west),
                resolve(&config.bcs.east),
                resolve(&config.bcs.south),
                resolve(&config.bcs.north),
            ],
            scheme: TimeScheme::Euler,
            tau_scratch: vec![0.0; ncells],
            face_flux_g: vec![0.0; nv],
            face_flux_h: vec![0.0; nv],
            ghost_buf_g: vec![0.0; nv],
            ghost_buf_h: vec![0.0; nv],
            rk2_stage_f: vec![0.0; ncells * nv],
            rk2_stage_h: vec![0.0; ncells * nv],
            eq_g_scratch: vec![0.0; nv],
            eq_h_scratch: vec![0.0; nv],
        }
    }

    /// Recompute macro fields (rho, mom, energy, heat flux; kn_loc left
    /// untouched here, computed separately) from the current (g,h)
    /// distribution pair.
    pub fn update_moments(&mut self) {
        let nv = self.dist.nv;
        let vgrid = &self.dist.vgrid;
        let vw = &self.dist.vw;
        for c in 0..self.grid.ncells() {
            let gs = &self.dist.f[c * nv..c * nv + nv];
            let hs = &self.dist.h[c * nv..c * nv + nv];
            let mut rho = 0.0;
            let mut mx = 0.0;
            let mut my = 0.0;
            let mut e = 0.0;
            for k in 0..nv {
                let gv = gs[k];
                let w = vw[k] * gv;
                rho += w;
                mx += w * vgrid[k][0];
                my += w * vgrid[k][1];
                let v2 = vgrid[k][0] * vgrid[k][0] + vgrid[k][1] * vgrid[k][1];
                e += 0.5 * w * v2 + 0.5 * vw[k] * hs[k];
            }
            self.fields.rho[c] = rho;
            self.fields.mom[0][c] = mx;
            self.fields.mom[1][c] = my;
            self.fields.energy[c] = e;

            let rho_safe = rho.max(f64::MIN_POSITIVE);
            let u = [mx / rho_safe, my / rho_safe];
            // Total heat flux: in-plane (from g) + reduced-direction (from h)
            // contribution, per the standard (g,h) heat-flux decomposition
            // (Xu & Huang 2010, eq. for q from g,h moments):
            //   q = 0.5 * int( c*c^2*g + c*h ) dv
            let mut qx = 0.0;
            let mut qy = 0.0;
            for k in 0..nv {
                let gv = gs[k];
                let hv = hs[k];
                let cx = vgrid[k][0] - u[0];
                let cy = vgrid[k][1] - u[1];
                let c2 = cx * cx + cy * cy;
                qx += 0.5 * vw[k] * (c2 * gv + hv) * cx;
                qy += 0.5 * vw[k] * (c2 * gv + hv) * cy;
            }
            self.fields.heat[0][c] = qx;
            self.fields.heat[1][c] = qy;
        }
    }

    /// CFL-limited timestep: `dt = cfl * min(dx,dy) / vmax` where `vmax` is
    /// the maximum |velocity ordinate| magnitude in the discrete velocity
    /// set (the fastest characteristic speed in the DVM).
    pub fn cfl_dt(&self, cfl: f64) -> f64 {
        let mut vmax = 0.0f64;
        for v in self.dist.vgrid.iter() {
            let speed = (v[0] * v[0] + v[1] * v[1]).sqrt();
            if speed > vmax {
                vmax = speed;
            }
        }
        let vmax = vmax.max(1e-9);
        let h = self.grid.dx.min(self.grid.dy);
        cfl * h / vmax
    }

    /// Total conserved moments over the whole domain: (mass, momentum_x,
    /// momentum_y, energy). Used by conservation tests.
    pub fn totals(&self) -> (f64, f64, f64, f64) {
        let cell_vol = self.grid.dx * self.grid.dy;
        let mut mass = 0.0;
        let mut px = 0.0;
        let mut py = 0.0;
        let mut e = 0.0;
        for c in 0..self.grid.ncells() {
            mass += self.fields.rho[c] * cell_vol;
            px += self.fields.mom[0][c] * cell_vol;
            py += self.fields.mom[1][c] * cell_vol;
            e += self.fields.energy[c] * cell_vol;
        }
        (mass, px, py, e)
    }

    /// Build the local macro state (rho, u, T, q) for cell `c` from
    /// `self.fields` — a plain-data read, safe to call before any `&mut
    /// self` write in the same statement because the result is copied out
    /// into local variables immediately.
    #[inline]
    fn cell_macro(&self, c: usize) -> (f64, [f64; 2], f64, [f64; 2]) {
        let rho = self.fields.rho[c];
        let u = self.fields.velocity(c);
        let t = self.fields.temperature(c, self.gas_r, DOF);
        let q = [self.fields.heat[0][c], self.fields.heat[1][c]];
        (rho, u, t, q)
    }

    /// Compute the per-velocity face flux (`g_face * v.n`, `h_face * v.n`)
    /// for an interior (or periodic-wrapped) face between cell `cin`
    /// (upwind reference on the `vn>=0` side) and cell `cout`, given outward
    /// normal `normal` from `cin`'s perspective. `far_in`/`far_out` are the
    /// next cell further upstream of `cin`/`cout` respectively along the
    /// face-normal direction (i.e. one more step in the `-normal`/`+normal`
    /// direction), used ONLY for the van Leer MUSCL slope reconstruction of
    /// the upwind `g`/`h` node value at this face (see module docs); `None`
    /// degrades that side to first-order upwind. Writes into
    /// `self.face_flux_g`/`self.face_flux_h`.
    fn compute_interior_face_flux(
        &mut self,
        cin: usize,
        cout: usize,
        far_in: Option<usize>,
        far_out: Option<usize>,
        normal: [f64; 2],
        dt: f64,
    ) {
        let nv = self.dist.nv;

        let (rho_in, u_in, t_in, q_in) = self.cell_macro(cin);
        let tau_in = self.tau_scratch[cin];
        let (rho_out, u_out, t_out, q_out) = self.cell_macro(cout);
        let tau_out = self.tau_scratch[cout];

        let dt_half = 0.5 * dt;
        for k in 0..nv {
            let v = self.dist.vgrid[k];
            let vn = v[0] * normal[0] + v[1] * normal[1];
            let g_in_k = self.dist.f[cin * nv + k];
            let h_in_k = self.dist.h[cin * nv + k];
            let g_out_k = self.dist.f[cout * nv + k];
            let h_out_k = self.dist.h[cout * nv + k];

            // MUSCL-reconstructed face-side values of g/h for whichever side
            // ends up upwind (see module docs, van Leer 1979): the "far"
            // neighbor stands in for the far-side cell of a 3-point
            // (far, near, downwind) stencil centered on the upwind cell.
            let g_in_face = if let Some(far) = far_in {
                van_leer_face_value(self.dist.f[far * nv + k], g_in_k, g_out_k)
            } else {
                g_in_k
            };
            let h_in_face = if let Some(far) = far_in {
                van_leer_face_value(self.dist.h[far * nv + k], h_in_k, h_out_k)
            } else {
                h_in_k
            };
            let g_out_face = if let Some(far) = far_out {
                van_leer_face_value(self.dist.f[far * nv + k], g_out_k, g_in_k)
            } else {
                g_out_k
            };
            let h_out_face = if let Some(far) = far_out {
                van_leer_face_value(self.dist.h[far * nv + k], h_out_k, h_in_k)
            } else {
                h_out_k
            };

            let up: UpwindState = if vn >= 0.0 {
                UpwindState {
                    g: g_in_face,
                    h: h_in_face,
                    rho: rho_in,
                    u: u_in,
                    t: t_in,
                    q: q_in,
                    tau: tau_in,
                }
            } else {
                UpwindState {
                    g: g_out_face,
                    h: h_out_face,
                    rho: rho_out,
                    u: u_out,
                    t: t_out,
                    q: q_out,
                    tau: tau_out,
                }
            };

            let (geq, heq) = self
                .collision
                .equilibrium(up.rho, up.u, up.t, self.gas_r, up.q, v);
            let g_face = (up.tau * up.g + dt_half * geq) / (up.tau + dt_half);
            let h_face = (up.tau * up.h + dt_half * heq) / (up.tau + dt_half);
            self.face_flux_g[k] = g_face * vn;
            self.face_flux_h[k] = h_face * vn;
        }
    }

    /// Boundary-face variant: `cin` is the interior cell, `normal` points out
    /// of the domain. Ghost values are built into `self.ghost_buf_g`/`_h` via
    /// the BC object, then the face flux (upwind between interior and ghost)
    /// is computed using ONLY the interior cell's macro state for the
    /// equilibrium (DESIGN: same approximation as M1 — see prior version's
    /// note; simplest-correct, exact in the Pr=1/no-heat-flux BGK limit).
    fn compute_boundary_face_flux(
        &mut self,
        cin: usize,
        normal: [f64; 2],
        edge: Edge,
        config_bcs: &janus_core::config::BoundaryAssignment,
        dt: f64,
    ) {
        let nv = self.dist.nv;

        let (rho_in, u_in, t_in, q_in) = self.cell_macro(cin);
        let tau_in = self.tau_scratch[cin];

        // Enum-dispatch BC kernel: `Copy`, built fresh per face at zero cost
        // (no heap allocation, no vtable — see `bc::BoundaryConditionKernel`
        // docs). Replaces the previous `Box<dyn BoundaryCondition>` in this
        // hot per-face-per-step path.
        let bc_kernel = bc::BoundaryConditionKernel::from_kind(config_bcs.get(edge));
        let g_interior: Vec<f64> = self.dist.f[cin * nv..cin * nv + nv].to_vec();
        let h_interior: Vec<f64> = self.dist.h[cin * nv..cin * nv + nv].to_vec();
        let vgrid = self.dist.vgrid.clone();
        let vw = self.dist.vw.clone();
        bc_kernel.apply_gh(
            &g_interior,
            &h_interior,
            &vgrid,
            &vw,
            normal,
            self.gas_r,
            &mut self.ghost_buf_g,
            &mut self.ghost_buf_h,
        );

        let dt_half = 0.5 * dt;
        for k in 0..nv {
            let v = vgrid[k];
            let vn = v[0] * normal[0] + v[1] * normal[1];
            let (g_up, h_up): (f64, f64);
            if vn >= 0.0 {
                g_up = g_interior[k];
                h_up = h_interior[k];
            } else {
                g_up = self.ghost_buf_g[k];
                h_up = self.ghost_buf_h[k];
            }
            // DESIGN: use interior cell's own macro state for the
            // equilibrium evaluation at incoming (ghost-sourced) nodes too.
            let (rho_up, u_up, t_up, q_up) = (rho_in, u_in, t_in, q_in);
            let tau_up = tau_in;

            let (geq, heq) = self
                .collision
                .equilibrium(rho_up, u_up, t_up, self.gas_r, q_up, v);
            let g_face = (tau_up * g_up + dt_half * geq) / (tau_up + dt_half);
            let h_face = (tau_up * h_up + dt_half * heq) / (tau_up + dt_half);
            self.face_flux_g[k] = g_face * vn;
            self.face_flux_h[k] = h_face * vn;
        }
    }

    /// Dispatch to `self.scheme` (see `TimeScheme` docs): `Euler` calls
    /// `step` (unchanged, single-stage); `Rk2` calls `step_rk2`
    /// (Shu-Osher SSP-RK2, operator-split transport/collision).
    pub fn step_scheme(&mut self, dt: f64, config_bcs: &janus_core::config::BoundaryAssignment) {
        match self.scheme {
            TimeScheme::Euler => self.step(dt, config_bcs),
            TimeScheme::Rk2 => self.step_rk2(dt, config_bcs),
        }
    }

    /// Shu-Osher SSP-RK2 (Heun's method) time integration, applied via
    /// operator splitting (see `TimeScheme` docs for the citation):
    ///
    /// ```text
    /// u1     = Euler_step(u^n, dt)        // stage 1: one full Euler step
    /// u^{n+1} = 0.5*u^n + 0.5*Euler_step(u1, dt)   // stage 2
    /// ```
    ///
    /// // DESIGN: `Euler_step` here is the *entire* existing `step` (both
    /// // the MUSCL-reconstructed transport/flux update AND the closed-form
    /// // Shakhov relaxation substep), not just the transport RHS in
    /// // isolation. This is a deliberate operator-splitting choice, not an
    /// // oversight: the collision/relaxation substep in `step` is already a
    /// // semi-analytic/exponential (closed-form implicit) update
    /// // (`g_new = (tau*g_old + dt*g_eq) / (tau + dt)`, the exact solution
    /// // of `dg/dt = (g_eq-g)/tau` in the fixed-`g_eq` limit), which is
    /// // unconditionally stable for ANY `dt/tau` ratio, including the
    /// // stiff `tau -> 0` continuum limit. Applying a literal explicit RK
    /// // stage directly to the raw collision RHS `(g_eq-g)/tau` would
    /// // reintroduce exactly the stability problem the closed-form update
    /// // was written to avoid (an explicit RK stage on a stiff linear decay
    /// // term needs `dt <~ tau` to stay bounded, defeating the point of
    /// // DUGKS's characteristic-based half-step reconstruction). So instead
    /// // of RK-integrating the collision term, each RK2 stage re-applies
    /// // the exact closed-form relaxation at that stage's own frozen
    /// // macro state -- i.e. RK2 raises the *spatial/temporal order of the
    /// // transport (flux) part* to second order while the collision part
    /// // remains its already-stable closed-form treatment every stage. This
    /// // is a standard, defensible operator-splitting choice for stiff
    /// // BGK-type kinetic solvers (cf. how IMEX/exponential-integrator
    /// // schemes for BGK-type equations universally keep the stiff
    /// // collision term implicit/exact while only raising the order of the
    /// // non-stiff transport term), applied identically here for both the
    /// // 2D (`solver.rs`) and 3D (`solver3d.rs`) paths.
    fn step_rk2(&mut self, dt: f64, config_bcs: &janus_core::config::BoundaryAssignment) {
        let n = self.dist.f.len();
        debug_assert_eq!(n, self.rk2_stage_f.len());

        // Stage 1: u1 = Euler_step(u^n, dt). Save u^n first (into
        // rk2_stage_*, reused below as scratch after we no longer need the
        // pre-step snapshot) so we can form the final 0.5*u^n + 0.5*u2 blend.
        self.rk2_stage_f.copy_from_slice(&self.dist.f);
        self.rk2_stage_h.copy_from_slice(&self.dist.h);
        self.step(dt, config_bcs); // self.dist now holds u1

        // Stage 2: u2 = Euler_step(u1, dt); blend u^{n+1} = 0.5*u^n + 0.5*u2.
        self.step(dt, config_bcs); // self.dist now holds u2 = Euler_step(u1, dt)
        for i in 0..n {
            self.dist.f[i] = 0.5 * self.rk2_stage_f[i] + 0.5 * self.dist.f[i];
            self.dist.h[i] = 0.5 * self.rk2_stage_h[i] + 0.5 * self.dist.h[i];
        }
        self.update_moments();
    }

    /// One explicit DUGKS step: face reconstruction + flux + FV update +
    /// (implicit, closed-form) Shakhov relaxation for both `g` and `h`, then
    /// recompute moments.
    pub fn step(&mut self, dt: f64, config_bcs: &janus_core::config::BoundaryAssignment) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nv = self.dist.nv;
        let vol = self.grid.dx * self.grid.dy;
        let ncells = self.grid.ncells();

        self.dist_scratch.f.copy_from_slice(&self.dist.f);
        self.dist_scratch.h.copy_from_slice(&self.dist.h);

        for c in 0..ncells {
            let rho = self.fields.rho[c];
            let t = self.fields.temperature(c, self.gas_r, DOF);
            self.tau_scratch[c] = self.collision.relaxation_time(
                rho,
                t,
                self.gas_r,
                self.mu_ref,
                self.t_ref,
                self.omega,
            );
        }

        for j in 0..ny {
            for i in 0..nx {
                let cin = self.grid.idx(i, j);

                // East face
                if i + 1 < nx {
                    let cout = self.grid.idx(i + 1, j);
                    // far_in: one more step west of cin (None at the west
                    // boundary unless periodic); far_out: one more step east
                    // of cout (None at the east boundary unless periodic).
                    let far_in = if i > 0 {
                        Some(self.grid.idx(i - 1, j))
                    } else if matches!(self.bcs[0], BoundaryKindResolved::Periodic) {
                        Some(self.grid.idx(nx - 1, j))
                    } else {
                        None
                    };
                    let far_out = if i + 2 < nx {
                        Some(self.grid.idx(i + 2, j))
                    } else if matches!(self.bcs[1], BoundaryKindResolved::Periodic) {
                        Some(self.grid.idx((i + 2) % nx, j))
                    } else {
                        None
                    };
                    self.compute_interior_face_flux(cin, cout, far_in, far_out, [1.0, 0.0], dt);
                    for k in 0..nv {
                        let fg = self.face_flux_g[k] * self.grid.dy * dt / vol;
                        let fh = self.face_flux_h[k] * self.grid.dy * dt / vol;
                        self.dist_scratch.f[cin * nv + k] -= fg;
                        self.dist_scratch.f[cout * nv + k] += fg;
                        self.dist_scratch.h[cin * nv + k] -= fh;
                        self.dist_scratch.h[cout * nv + k] += fh;
                    }
                } else {
                    match self.bcs[1] {
                        BoundaryKindResolved::Periodic => {
                            let cout = self.grid.idx(0, j);
                            // far_in: west of cin (periodic-aware, same as
                            // above); far_out: east of cout=cell(0,j), i.e.
                            // cell(1,j) if it exists (nx>1 periodic wrap).
                            let far_in = if i > 0 {
                                Some(self.grid.idx(i - 1, j))
                            } else {
                                Some(self.grid.idx(nx - 1, j))
                            };
                            let far_out = if nx > 1 {
                                Some(self.grid.idx(1, j))
                            } else {
                                None
                            };
                            self.compute_interior_face_flux(
                                cin,
                                cout,
                                far_in,
                                far_out,
                                [1.0, 0.0],
                                dt,
                            );
                            for k in 0..nv {
                                let fg = self.face_flux_g[k] * self.grid.dy * dt / vol;
                                let fh = self.face_flux_h[k] * self.grid.dy * dt / vol;
                                self.dist_scratch.f[cin * nv + k] -= fg;
                                self.dist_scratch.f[cout * nv + k] += fg;
                                self.dist_scratch.h[cin * nv + k] -= fh;
                                self.dist_scratch.h[cout * nv + k] += fh;
                            }
                        }
                        BoundaryKindResolved::Other => {
                            self.compute_boundary_face_flux(
                                cin,
                                [1.0, 0.0],
                                Edge::East,
                                config_bcs,
                                dt,
                            );
                            for k in 0..nv {
                                let fg = self.face_flux_g[k] * self.grid.dy * dt / vol;
                                let fh = self.face_flux_h[k] * self.grid.dy * dt / vol;
                                self.dist_scratch.f[cin * nv + k] -= fg;
                                self.dist_scratch.h[cin * nv + k] -= fh;
                            }
                        }
                    }
                }

                // North face
                if j + 1 < ny {
                    let cout = self.grid.idx(i, j + 1);
                    let far_in = if j > 0 {
                        Some(self.grid.idx(i, j - 1))
                    } else if matches!(self.bcs[2], BoundaryKindResolved::Periodic) {
                        Some(self.grid.idx(i, ny - 1))
                    } else {
                        None
                    };
                    let far_out = if j + 2 < ny {
                        Some(self.grid.idx(i, j + 2))
                    } else if matches!(self.bcs[3], BoundaryKindResolved::Periodic) {
                        Some(self.grid.idx(i, (j + 2) % ny))
                    } else {
                        None
                    };
                    self.compute_interior_face_flux(cin, cout, far_in, far_out, [0.0, 1.0], dt);
                    for k in 0..nv {
                        let fg = self.face_flux_g[k] * self.grid.dx * dt / vol;
                        let fh = self.face_flux_h[k] * self.grid.dx * dt / vol;
                        self.dist_scratch.f[cin * nv + k] -= fg;
                        self.dist_scratch.f[cout * nv + k] += fg;
                        self.dist_scratch.h[cin * nv + k] -= fh;
                        self.dist_scratch.h[cout * nv + k] += fh;
                    }
                } else {
                    match self.bcs[3] {
                        BoundaryKindResolved::Periodic => {
                            let cout = self.grid.idx(i, 0);
                            let far_in = if j > 0 {
                                Some(self.grid.idx(i, j - 1))
                            } else {
                                Some(self.grid.idx(i, ny - 1))
                            };
                            let far_out = if ny > 1 {
                                Some(self.grid.idx(i, 1))
                            } else {
                                None
                            };
                            self.compute_interior_face_flux(
                                cin,
                                cout,
                                far_in,
                                far_out,
                                [0.0, 1.0],
                                dt,
                            );
                            for k in 0..nv {
                                let fg = self.face_flux_g[k] * self.grid.dx * dt / vol;
                                let fh = self.face_flux_h[k] * self.grid.dx * dt / vol;
                                self.dist_scratch.f[cin * nv + k] -= fg;
                                self.dist_scratch.f[cout * nv + k] += fg;
                                self.dist_scratch.h[cin * nv + k] -= fh;
                                self.dist_scratch.h[cout * nv + k] += fh;
                            }
                        }
                        BoundaryKindResolved::Other => {
                            self.compute_boundary_face_flux(
                                cin,
                                [0.0, 1.0],
                                Edge::North,
                                config_bcs,
                                dt,
                            );
                            for k in 0..nv {
                                let fg = self.face_flux_g[k] * self.grid.dx * dt / vol;
                                let fh = self.face_flux_h[k] * self.grid.dx * dt / vol;
                                self.dist_scratch.f[cin * nv + k] -= fg;
                                self.dist_scratch.h[cin * nv + k] -= fh;
                            }
                        }
                    }
                }

                if i == 0 {
                    if let BoundaryKindResolved::Other = self.bcs[0] {
                        self.compute_boundary_face_flux(
                            cin,
                            [-1.0, 0.0],
                            Edge::West,
                            config_bcs,
                            dt,
                        );
                        for k in 0..nv {
                            let fg = self.face_flux_g[k] * self.grid.dy * dt / vol;
                            let fh = self.face_flux_h[k] * self.grid.dy * dt / vol;
                            self.dist_scratch.f[cin * nv + k] -= fg;
                            self.dist_scratch.h[cin * nv + k] -= fh;
                        }
                    }
                }
                if j == 0 {
                    if let BoundaryKindResolved::Other = self.bcs[2] {
                        self.compute_boundary_face_flux(
                            cin,
                            [0.0, -1.0],
                            Edge::South,
                            config_bcs,
                            dt,
                        );
                        for k in 0..nv {
                            let fg = self.face_flux_g[k] * self.grid.dx * dt / vol;
                            let fh = self.face_flux_h[k] * self.grid.dx * dt / vol;
                            self.dist_scratch.f[cin * nv + k] -= fg;
                            self.dist_scratch.h[cin * nv + k] -= fh;
                        }
                    }
                }
            }
        }

        std::mem::swap(&mut self.dist.f, &mut self.dist_scratch.f);
        std::mem::swap(&mut self.dist.h, &mut self.dist_scratch.h);

        // Positivity-preserving floor (conservative, per-cell): a first-
        // order-upwind explicit FV update can, under a strong shock/large
        // local gradient (even within the CFL limit, since CFL only bounds
        // the *fastest* discrete velocity, not every node's local flux
        // balance), drive an individual `g_k`/`h_k` node slightly negative
        // due to floating-point/discretization error. `g`/`h` must stay
        // non-negative (they are physically number/energy densities in
        // velocity space). Rather than a naive per-node `.max(0.0)` clamp
        // (which is NOT mass-conserving: zeroing some nodes and leaving
        // others unchanged shifts the cell's total `rho`/`rho*u`/`rho*E`
        // moments), we floor negative nodes to zero and then rescale the
        // REMAINING non-negative nodes of that cell so the cell's `rho`
        // moment (computed from `g` alone, the zeroth moment) is restored
        // to its pre-floor value exactly -- a standard positivity-preserving
        // limiter construction (cf. Zhang & Shu, "On positivity-preserving
        // high order discontinuous Galerkin schemes...", J. Comput. Phys.
        // 229, 3091 (2010), whose mass-conserving rescale-toward-the-mean
        // construction this mirrors, adapted here to velocity-space nodes
        // rather than spatial nodes). This only ever activates in the rare
        // pathological case (strongly negative flux overshoot); it is a
        // no-op (rescale factor 1.0) whenever every node is already
        // non-negative, which is the case for every existing test in this
        // module (verified in `positivity_floor_is_noop_on_nonnegative_state`
        // below).
        apply_positivity_floor(&mut self.dist.f, self.dist.nv, ncells);
        apply_positivity_floor(&mut self.dist.h, self.dist.nv, ncells);

        self.update_moments();
        for c in 0..ncells {
            let rho = self.fields.rho[c];
            let u = self.fields.velocity(c);
            let t = self.fields.temperature(c, self.gas_r, DOF);
            let q = [self.fields.heat[0][c], self.fields.heat[1][c]];
            // Conservation targets: the relaxation (a local collision step) must
            // leave this cell's mass/momentum/energy EXACTLY unchanged. These
            // are the post-transport moments just recomputed above.
            let (tgt_rho, tgt_px, tgt_py, tgt_e) = (
                rho,
                self.fields.mom[0][c],
                self.fields.mom[1][c],
                self.fields.energy[c],
            );
            let tau_c = self.collision.relaxation_time(
                rho,
                t,
                self.gas_r,
                self.mu_ref,
                self.t_ref,
                self.omega,
            );

            // Pass 1: relax each node in place, cache the discrete equilibrium,
            // and accumulate (a) the post-relaxation moments of g1 and (b) the
            // discrete moments of the equilibrium needed to build the
            // conservative correction.
            let mut rho1 = 0.0;
            let mut px1 = 0.0;
            let mut py1 = 0.0;
            let mut e1 = 0.0;
            let (mut s1, mut sx, mut sy) = (0.0, 0.0, 0.0);
            let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
            let (mut sqq, mut sxqq, mut syqq, mut sh) = (0.0, 0.0, 0.0, 0.0);
            for k in 0..nv {
                let v = self.dist.vgrid[k];
                let w = self.dist.vw[k];
                let (geq, heq) = self.collision.equilibrium(rho, u, t, self.gas_r, q, v);
                self.eq_g_scratch[k] = geq;
                self.eq_h_scratch[k] = heq;
                let idx = c * nv + k;
                let g1 = (tau_c * self.dist.f[idx] + dt * geq) / (tau_c + dt);
                let h1 = (tau_c * self.dist.h[idx] + dt * heq) / (tau_c + dt);
                self.dist.f[idx] = g1;
                self.dist.h[idx] = h1;
                let (vx, vy) = (v[0], v[1]);
                let v2 = vx * vx + vy * vy;
                rho1 += w * g1;
                px1 += w * vx * g1;
                py1 += w * vy * g1;
                e1 += 0.5 * w * (v2 * g1 + h1);
                // equilibrium moments (weighted by vw)
                s1 += w * geq;
                sx += w * vx * geq;
                sy += w * vy * geq;
                sxx += w * vx * vx * geq;
                sxy += w * vx * vy * geq;
                syy += w * vy * vy * geq;
                sqq += w * v2 * geq;
                sxqq += w * vx * v2 * geq;
                syqq += w * vy * v2 * geq;
                sh += w * heq;
            }

            // Solve the 3x3 SPD system for the equilibrium-weighted correction
            // coefficients (A,B,C) that restore mass + momentum exactly:
            //   g1_k += geq_k*(A + B*vx + C*vy)
            // with rhs = post-relaxation defects. (Odd equilibrium moments are
            // near-zero so the system is diagonally dominant; solved via Cramer.)
            let d_rho = tgt_rho - rho1;
            let d_px = tgt_px - px1;
            let d_py = tgt_py - py1;
            let det = s1 * (sxx * syy - sxy * sxy) - sx * (sx * syy - sxy * sy)
                + sy * (sx * sxy - sxx * sy);
            let (a, b, cc) = if det.abs() > 1e-300 {
                let inv = 1.0 / det;
                let det_a = d_rho * (sxx * syy - sxy * sxy) - sx * (d_px * syy - sxy * d_py)
                    + sy * (d_px * sxy - sxx * d_py);
                let det_b = s1 * (d_px * syy - sxy * d_py) - d_rho * (sx * syy - sxy * sy)
                    + sy * (sx * d_py - d_px * sy);
                let det_c = s1 * (sxx * d_py - d_px * sxy) - sx * (sx * d_py - d_px * sy)
                    + d_rho * (sx * sxy - sxx * sy);
                (det_a * inv, det_b * inv, det_c * inv)
            } else {
                (0.0, 0.0, 0.0)
            };

            // Energy defect after the (A,B,C) g-correction is applied; absorb the
            // remainder with a scalar rescale F of the h-equilibrium:
            //   h1_k += heq_k*F,  0.5*F*sum(vw*heq) = remaining energy defect.
            let e_from_g_corr = 0.5 * (a * sqq + b * sxqq + cc * syqq);
            let d_e = tgt_e - e1 - e_from_g_corr;
            let f_h = if sh.abs() > 1e-300 {
                d_e / (0.5 * sh)
            } else {
                0.0
            };

            // Pass 2: apply the correction. This makes the cell's discrete
            // (mass, momentum, energy) moments equal the targets to machine
            // precision, so the relaxation is exactly conservative regardless of
            // the velocity-grid quadrature error in the continuous equilibrium.
            for k in 0..nv {
                let v = self.dist.vgrid[k];
                let idx = c * nv + k;
                self.dist.f[idx] += self.eq_g_scratch[k] * (a + b * v[0] + cc * v[1]);
                self.dist.h[idx] += self.eq_h_scratch[k] * f_h;
            }
        }
        // Final positivity guard: the conservative moment correction above can
        // extrapolate a node slightly negative across an extreme (e.g. 1000:1)
        // gradient. Re-apply the mass-conserving floor so g/h stay non-negative.
        // This is a NO-OP whenever no node is negative (the overwhelming common
        // case, incl. all smooth conservation tests), so it does not disturb the
        // correction's exact moments there; it only activates for pathological
        // shock overshoot, trading a negligible momentum/energy perturbation for
        // guaranteed positivity (mass is still conserved by the rescale).
        apply_positivity_floor(&mut self.dist.f, self.dist.nv, ncells);
        apply_positivity_floor(&mut self.dist.h, self.dist.nv, ncells);
        self.update_moments();
    }
}

impl TimeStepper for DugksSolver {
    fn step(&mut self, _dt: f64) {
        panic!("use DugksSolver::step(dt, &case.bcs) directly; the generic TimeStepper::step is not wired to boundary conditions")
    }
}

/// Mass-conserving positivity floor for a cell-major distribution buffer
/// (`dist`, length `ncells * nv`): clamp negative velocity-node values to
/// zero, then rescale the cell's remaining non-negative nodes so the cell's
/// zeroth moment (`sum_k dist[c*nv+k]`, an UNWEIGHTED sum here since this
/// function is quadrature-weight-agnostic -- the weighting is folded in
/// identically for both the pre- and post-floor sum, so the *ratio* used for
/// rescaling is exactly the same whether or not `vw` is included) is
/// unchanged. See `DugksSolver::step`'s call site doc comment for the full
/// rationale and citation. Shared verbatim by both the `g` (`dist.f`) and
/// `h` (`dist.h`) buffers (called once each per step).
/// Crate-visible alias so `solver3d.rs` (which has no separate `g,h` pair --
/// see `maxwellian3d` module docs -- and therefore has no reason to
/// duplicate this function) can reuse the identical mass-conserving
/// positivity-floor construction for its single distribution `f`.
pub(crate) fn apply_positivity_floor_pub(dist: &mut [f64], nv: usize, ncells: usize) {
    apply_positivity_floor(dist, nv, ncells)
}

/// van Leer (1979) MUSCL-limited face-value extrapolation from a cell center
/// to a face half a cell-width away, given the upwind cell's value `near`,
/// the next cell further upstream `far`, and the immediate downwind
/// neighbor `down` (the 3-point `(far, near, down)` stencil centered on the
/// upwind cell, all in cell-index order along the face-normal direction).
///
/// Reference: van Leer, B., "Towards the ultimate conservative difference
/// scheme. V. A second-order sequel to Godunov's method", J. Comput. Phys.
/// 32, 101-136 (1979).
///
/// Uses the standard van Leer harmonic slope limiter
/// `phi(r) = (r + |r|) / (1 + |r|)` (which is 0 for `r <= 0`, i.e. at a
/// local extremum, and tends to `min(2, 2r/(1+r))`-like smooth limiting
/// otherwise — TVD, no new extrema introduced) applied to the ratio of the
/// downwind-side difference to the upwind-side difference, and extrapolates
/// the limited slope half a cell outward to the face:
/// `face = near + 0.5 * phi(r) * (near - far)`.
/// This is algebraically equivalent to the more commonly quoted symmetric
/// van Leer form `face = near + 0.5*phi(r)*minmod-style average slope`; the
/// one-sided-times-limiter form used here is the standard implementation
/// (e.g. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics",
/// 3rd ed., §13.3) and reduces exactly to first-order upwind (`face =
/// near`) whenever `phi(r) = 0`.
#[inline]
pub(crate) fn van_leer_face_value(far: f64, near: f64, down: f64) -> f64 {
    let d_upwind = near - far; // one-sided upwind-side difference
    let d_downwind = down - near; // one-sided downwind-side difference

    // r = ratio of downwind to upwind differences. Guard the near-zero
    // denominator: a vanishing upwind-side gradient with a nonzero
    // downwind-side one is a local extremum in the reconstruction sense, so
    // fall back to first-order (phi = 0) rather than dividing by ~0.
    if d_upwind.abs() < 1e-300 {
        return near;
    }
    let r = d_downwind / d_upwind;
    let phi = if r > 0.0 {
        (r + r.abs()) / (1.0 + r.abs())
    } else {
        0.0
    };
    // The reconstructed face value of g/h is a velocity-space density and must
    // stay non-negative; a large slope across a strong (e.g. 1000:1) jump can
    // otherwise extrapolate below zero and over-deplete the downwind cell,
    // producing negative distribution values the positivity floor cannot always
    // repair. Clamp to zero: this keeps the finite-volume update conservative
    // (the same clamped face value feeds both adjacent cells' fluxes) while
    // preventing negative-density overshoot. Reduces to first-order where the
    // limited slope would cross zero.
    (near + 0.5 * phi * d_upwind).max(0.0)
}

fn apply_positivity_floor(dist: &mut [f64], nv: usize, ncells: usize) {
    for c in 0..ncells {
        let start = c * nv;
        let slice = &mut dist[start..start + nv];
        let mut had_negative = false;
        let mut pre_sum = 0.0;
        for &v in slice.iter() {
            pre_sum += v;
            if v < 0.0 {
                had_negative = true;
            }
        }
        if !had_negative {
            continue; // no-op fast path: overwhelmingly the common case.
        }
        let mut post_sum = 0.0;
        for v in slice.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
            post_sum += *v;
        }
        // Rescale remaining non-negative nodes so the cell total is
        // restored exactly (to floating-point precision); if the entire
        // cell floored to zero (post_sum == 0, only possible if pre_sum was
        // already <= 0, itself only possible for a state so pathological
        // that no rescale can recover it), leave it at zero rather than
        // dividing by zero -- physically, a cell with non-positive total
        // density has no meaningful distribution to rescale toward anyway.
        if post_sum > 1e-300 && pre_sum > 0.0 {
            let scale = pre_sum / post_sum;
            for v in slice.iter_mut() {
                *v *= scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::config::{BoundaryAssignment, GasProperties};
    use janus_core::grid::Grid2D;

    fn periodic_case(nx: usize, ny: usize) -> CaseConfig {
        CaseConfig {
            grid: Grid2D::new(nx, ny, 0.05, 0.05, [0.0, 0.0]),
            bcs: BoundaryAssignment::all_periodic(),
            gas: GasProperties::monatomic_default(),
        }
    }

    fn init_uniform(solver: &mut DugksSolver, rho: f64, u: [f64; 2], t: f64) {
        let nv = solver.dist.nv;
        let ncells = solver.grid.ncells();
        let r_gas = solver.gas_r;
        let vgrid = solver.dist.vgrid.clone();
        for c in 0..ncells {
            for k in 0..nv {
                let (g, h) = crate::maxwellian::gh_equilibrium(rho, u, t, r_gas, vgrid[k]);
                solver.dist.f[c * nv + k] = g;
                solver.dist.h[c * nv + k] = h;
            }
        }
        solver.update_moments();
    }

    #[test]
    fn conservation_on_periodic_domain() {
        let config = periodic_case(4, 4);
        let (vgrid, vw) = crate::velocity_grid::VelocityGrid2D::simpson(2000.0, 17);
        let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver::new(&config, dist);
        init_uniform(&mut solver, 1.0, [20.0, -10.0], 300.0);
        let c0 = solver.grid.idx(1, 1);
        let nv = solver.dist.nv;
        for k in 0..nv {
            solver.dist.f[c0 * nv + k] *= 1.05;
            solver.dist.h[c0 * nv + k] *= 1.05;
        }
        solver.update_moments();

        let before = solver.totals();
        let dt = solver.cfl_dt(0.3);
        for _ in 0..5 {
            solver.step(dt, &config.bcs);
        }
        let after = solver.totals();

        let tol = 1e-3;
        assert!(
            (after.0 - before.0).abs() / before.0.abs().max(1e-30) < tol,
            "mass drift: before={} after={}",
            before.0,
            after.0
        );
        assert!(
            (after.1 - before.1).abs() / before.0.abs().max(1e-30) < tol,
            "momentum-x drift: before={} after={}",
            before.1,
            after.1
        );
        assert!(
            (after.2 - before.2).abs() / before.0.abs().max(1e-30) < tol,
            "momentum-y drift: before={} after={}",
            before.2,
            after.2
        );
        assert!(
            (after.3 - before.3).abs() / before.3.abs().max(1e-30) < tol,
            "energy drift: before={} after={}",
            before.3,
            after.3
        );
    }

    #[test]
    fn step_produces_finite_values() {
        let config = periodic_case(3, 3);
        let (vgrid, vw) = crate::velocity_grid::VelocityGrid2D::simpson(1500.0, 9);
        let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver::new(&config, dist);
        init_uniform(&mut solver, 1.2, [5.0, -3.0], 320.0);

        let dt = solver.cfl_dt(0.4);
        for _ in 0..3 {
            solver.step(dt, &config.bcs);
        }

        assert!(
            solver.dist.f.iter().all(|v| v.is_finite()),
            "g has non-finite values"
        );
        assert!(
            solver.dist.h.iter().all(|v| v.is_finite()),
            "h has non-finite values"
        );
        assert!(solver.fields.rho.iter().all(|v| v.is_finite()));
        assert!(solver.fields.mom[0].iter().all(|v| v.is_finite()));
        assert!(solver.fields.mom[1].iter().all(|v| v.is_finite()));
        assert!(solver.fields.energy.iter().all(|v| v.is_finite()));
        assert!(solver.fields.heat[0].iter().all(|v| v.is_finite()));
        assert!(solver.fields.heat[1].iter().all(|v| v.is_finite()));
    }

    #[test]
    fn positivity_floor_is_noop_on_nonnegative_state() {
        let nv = 4;
        let ncells = 2;
        let mut dist = vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.0, 2.5, 1.0];
        let before = dist.clone();
        apply_positivity_floor(&mut dist, nv, ncells);
        assert_eq!(
            dist, before,
            "positivity floor must be a no-op when nothing is negative"
        );
    }

    #[test]
    fn positivity_floor_clamps_negative_and_conserves_cell_total() {
        let nv = 4;
        let ncells = 1;
        let mut dist = vec![5.0, -1.0, 3.0, 2.0];
        let pre_sum: f64 = dist.iter().sum();
        apply_positivity_floor(&mut dist, nv, ncells);
        assert!(
            dist.iter().all(|&v| v >= 0.0),
            "all nodes must be non-negative after flooring: {dist:?}"
        );
        let post_sum: f64 = dist.iter().sum();
        assert!(
            (post_sum - pre_sum).abs() / pre_sum.abs() < 1e-12,
            "cell total must be conserved: {pre_sum} vs {post_sum}"
        );
    }

    #[test]
    fn gamma_five_thirds_matches_sound_speed() {
        // Sanity check that the effective gamma used by this solver (via
        // DOF=3) gives the correct monatomic sound speed relation
        // a^2 = gamma * R * T, gamma = 5/3.
        assert!((crate::maxwellian::GAMMA - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn van_leer_limiter_is_first_order_upwind_at_local_extremum() {
        // At a local extremum (downwind difference opposite sign to upwind
        // difference, r <= 0) the van Leer limiter must vanish (phi=0),
        // reproducing plain first-order upwind (face value = near).
        let far = 1.0;
        let near = 2.0; // upwind diff = +1
        let down = 1.5; // downwind diff = -0.5 (opposite sign -> r < 0)
        assert_eq!(van_leer_face_value(far, near, down), near);
    }

    #[test]
    fn van_leer_reduces_to_upwind_when_far_equals_near() {
        // Zero upwind-side gradient must not divide by zero and must return
        // the unmodified near value.
        assert_eq!(van_leer_face_value(2.0, 2.0, 5.0), 2.0);
    }

    #[test]
    fn van_leer_smooth_linear_profile_extrapolates_to_exact_half_step() {
        // For a perfectly linear profile (far, near, down equally spaced),
        // r = 1 exactly, phi(1) = 1, and the MUSCL face value should equal
        // the exact linear interpolation half a cell towards the face:
        // face = near + 0.5*(near-far) = near + 0.5*slope.
        let far = 1.0;
        let near = 2.0;
        let down = 3.0; // same slope both sides
        let face = van_leer_face_value(far, near, down);
        assert!((face - 2.5).abs() < 1e-12, "face={face}");
    }

    #[test]
    fn muscl_reconstruction_preserves_conservation_on_periodic_domain() {
        // Same conservation check as `conservation_on_periodic_domain`, but
        // on a large-enough grid that the MUSCL far-neighbor stencil is
        // fully populated (periodic wrap) everywhere -- the FV update must
        // still conserve mass/momentum/energy exactly regardless of spatial
        // reconstruction order, since MUSCL only changes the face VALUE fed
        // into the same symmetric add/subtract flux balance.
        let config = periodic_case(6, 6);
        let (vgrid, vw) = crate::velocity_grid::VelocityGrid2D::simpson(2000.0, 17);
        let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver::new(&config, dist);
        init_uniform(&mut solver, 1.0, [15.0, -5.0], 300.0);
        let c0 = solver.grid.idx(2, 2);
        let nv = solver.dist.nv;
        for k in 0..nv {
            solver.dist.f[c0 * nv + k] *= 1.08;
            solver.dist.h[c0 * nv + k] *= 1.08;
        }
        solver.update_moments();

        let before = solver.totals();
        let dt = solver.cfl_dt(0.3);
        for _ in 0..5 {
            solver.step(dt, &config.bcs);
        }
        let after = solver.totals();

        let tol = 1e-3;
        assert!(
            (after.0 - before.0).abs() / before.0.abs().max(1e-30) < tol,
            "mass drift"
        );
        assert!(
            (after.3 - before.3).abs() / before.3.abs().max(1e-30) < tol,
            "energy drift"
        );
        assert!(solver.dist.f.iter().all(|v| v.is_finite()));
        assert!(solver.dist.h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rk2_scheme_conserves_and_stays_finite_on_periodic_domain() {
        let config = periodic_case(4, 4);
        let (vgrid, vw) = crate::velocity_grid::VelocityGrid2D::simpson(2000.0, 17);
        let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver::new(&config, dist);
        solver.scheme = TimeScheme::Rk2;
        init_uniform(&mut solver, 1.0, [20.0, -10.0], 300.0);
        let c0 = solver.grid.idx(1, 1);
        let nv = solver.dist.nv;
        for k in 0..nv {
            solver.dist.f[c0 * nv + k] *= 1.05;
            solver.dist.h[c0 * nv + k] *= 1.05;
        }
        solver.update_moments();

        let before = solver.totals();
        let dt = solver.cfl_dt(0.2);
        for _ in 0..5 {
            solver.step_scheme(dt, &config.bcs);
        }
        let after = solver.totals();

        let tol = 1e-3;
        assert!(
            (after.0 - before.0).abs() / before.0.abs().max(1e-30) < tol,
            "mass drift: {before:?} vs {after:?}"
        );
        assert!(
            (after.3 - before.3).abs() / before.3.abs().max(1e-30) < tol,
            "energy drift"
        );
        assert!(solver.dist.f.iter().all(|v| v.is_finite()));
        assert!(solver.dist.h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rk2_default_scheme_is_euler() {
        // Constructing via `new()` must default to Euler (unchanged
        // behavior) so existing callers are unaffected unless they opt in.
        let config = periodic_case(2, 2);
        let (vgrid, vw) = crate::velocity_grid::VelocityGrid2D::simpson(1500.0, 9);
        let dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);
        let solver = DugksSolver::new(&config, dist);
        assert_eq!(solver.scheme, TimeScheme::Euler);
    }
}
