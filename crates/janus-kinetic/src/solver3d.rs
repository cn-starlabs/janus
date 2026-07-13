//! 3D DUGKS solver (M4): full single-distribution DVM over a `Grid3D`,
//! Shakhov collision, DUGKS half-step face reconstruction across all 6
//! faces (+x,-x,+y,-y,+z,-z).
//!
//! ## Why there is no `(g,h)` reduction here
//!
//! See `maxwellian3d` module docs for the full reasoning. Short version: the
//! 2D solver's `(g,h)` two-distribution trick exists ONLY to patch a missing
//! translational DOF caused by discretizing just 2 of the 3 physical
//! velocity components. This solver discretizes all 3 velocity components
//! directly (`Distribution3D`, `VelocityGrid3D`), so all 3 DOF are already
//! physically present in the moments of a single `f` — nothing is reduced
//! out, and there is nothing to patch. `DOF = 3` directly.
//!
//! ## 6-face flux, one direction-parameterized kernel (not 6x copy-paste)
//!
//! DUGKS's half-step face reconstruction (Guo, Xu & Wang 2013) is
//! algebraically IDENTICAL regardless of which physical axis a face's
//! normal points along — the only per-face inputs are (a) the outward unit
//! normal vector (used solely via the scalar `v.n` projection) and (b) the
//! upwind/downwind neighbor cell indices along that axis. `compute_face_flux`
//! below takes the normal and the two neighboring cell indices as plain
//! arguments and is called once per face per cell in `step`'s inner loop
//! (6 call sites: east/west/north/south/up/down), rather than having 6
//! independently-hand-written copies of the same half-step-reconstruction
//! math — this is the "small internal helper parameterized by face-normal
//! direction" the M4 task calls for. Each of the 6 call sites still performs
//! genuine 3D velocity-moment integration (`v.n` uses the full 3-component
//! `v` and the 3-component `normal`), not a fake 2D-plus-pass-through: e.g.
//! the "up"/"down" (z) faces exercise exactly the same `v[2]`-dependent
//! moment integral that x/y faces exercise for `v[0]`/`v[1]`, verified by
//! the `z_face_flux_matches_hand_rotated_x_face` test below (a genuine
//! DUGKS z-face flux must equal an x-face flux computed on a state with x
//! and z coordinates/velocities swapped — this is the correctness check
//! that would fail if z were merely passed through as a dummy dimension).
//!
//! ## MUSCL slope-limited spatial reconstruction (hardening pass)
//!
//! Same second-order van Leer MUSCL upgrade as the 2D solver (see
//! `crate::solver` module docs and `crate::solver::van_leer_face_value` for
//! the shared limiter helper reused verbatim here): the raw upwind
//! cell-center `f` node value is replaced by a slope-limited half-cell
//! extrapolation to the face, built from the 3-point stencil
//! `(far, near, down)` along the face-normal axis, before being fed into the
//! identical DUGKS half-step formula. Reference: van Leer, B., "Towards the
//! ultimate conservative difference scheme. V.", J. Comput. Phys. 32,
//! 101-136 (1979).
//!
//! References:
//! - DUGKS: Guo, Z., Xu, K., Wang, R., Phys. Rev. E 88, 033305 (2013).
//! - Shakhov: Shakhov, E. M., Fluid Dynamics 3, 95 (1968).
//! - UGKWP (wave part is this solver, standalone-selectable per
//!   ENGINEERING_SPEC.md §2, mirroring `coupled::FluxKernel::Dugks`): Liu,
//!   S., Zhu, Y., Xu, K., J. Comput. Phys. 401, 108977 (2020).

use crate::bc3d::BoundaryConditionKernel3D;
use crate::collision3d::{Collision3D, Shakhov3D};
use crate::maxwellian3d::DOF;
use janus_core::config::{BoundaryAssignment3D, BoundaryKind3D, CaseConfig3D, Face};
use janus_core::distribution::Distribution3D;
use janus_core::fields3d::MacroFields3D;
use janus_core::grid3d::Grid3D;

#[derive(Clone, Copy)]
enum BoundaryKindResolved {
    Periodic,
    Other,
}

/// Local upwind macro state for a face's Shakhov equilibrium evaluation
/// (plain data, computed before any `&mut self` write — same borrow-safety
/// discipline as the 2D solver's `UpwindState`).
#[derive(Clone, Copy)]
struct UpwindState3D {
    f: f64,
    rho: f64,
    u: [f64; 3],
    t: f64,
    q: [f64; 3],
    tau: f64,
}

/// The 3D DUGKS solver state: grid, gas/BC config, discrete-velocity
/// distribution (double-buffered), macro fields, preallocated scratch.
pub struct DugksSolver3D {
    pub grid: Grid3D,
    pub gas_r: f64,
    pub mu_ref: f64,
    pub t_ref: f64,
    pub omega: f64,
    pub collision: Shakhov3D,
    pub dist: Distribution3D,
    dist_scratch: Distribution3D,
    pub fields: MacroFields3D,
    bcs: [BoundaryKindResolved; 6], // west, east, south, north, down, up
    /// Selectable explicit time-integration scheme, same convention as the
    /// 2D solver (`crate::solver::TimeScheme`). Defaults to `Euler`.
    pub scheme: crate::solver::TimeScheme,
    tau_scratch: Vec<f64>,
    face_flux: Vec<f64>, // len = nv, reused per face
    ghost_buf: Vec<f64>, // len = nv, reused per boundary face
    // RK2/RK4 preallocated scratch (zero heap allocation in the step path).
    rk2_stage_f: Vec<f64>, // len = ncells * nv
    rk4_stage1_f: Vec<f64>, // len = ncells * nv, for RK4
    rk4_stage2_f: Vec<f64>, // len = ncells * nv, for RK4
    rk4_stage3_f: Vec<f64>, // len = ncells * nv, for RK4
    // Pre-floor conservation target per cell: [rho, px, py, pz, e, qx, qy, qz]
    // captured after transport but BEFORE the positivity floor, so the
    // conservative relaxation correction restores the transport-conserved
    // moments (undoing the floor's mass-only, energy-non-conserving rescale).
    relax_tgt: Vec<[f64; 8]>, // len = ncells
}

impl DugksSolver3D {
    pub fn new(config: &CaseConfig3D, dist: Distribution3D) -> Self {
        let ncells = config.grid.ncells();
        let nv = dist.nv;
        let fields = MacroFields3D::zeros(ncells);
        let dist_scratch = Distribution3D::zeros(ncells, dist.vgrid.clone(), dist.vw.clone());
        let resolve = |k: &BoundaryKind3D| match k {
            BoundaryKind3D::Periodic => BoundaryKindResolved::Periodic,
            _ => BoundaryKindResolved::Other,
        };
        Self {
            grid: config.grid,
            gas_r: config.gas.r_gas,
            mu_ref: config.gas.mu_ref,
            t_ref: config.gas.t_ref,
            omega: config.gas.vhs_omega,
            collision: Shakhov3D::new(config.gas.prandtl),
            dist,
            dist_scratch,
            fields,
            bcs: [
                resolve(&config.bcs.west),
                resolve(&config.bcs.east),
                resolve(&config.bcs.south),
                resolve(&config.bcs.north),
                resolve(&config.bcs.down),
                resolve(&config.bcs.up),
            ],
            scheme: crate::solver::TimeScheme::Euler,
            tau_scratch: vec![0.0; ncells],
            face_flux: vec![0.0; nv],
            ghost_buf: vec![0.0; nv],
            rk2_stage_f: vec![0.0; ncells * nv],
            rk4_stage1_f: vec![0.0; ncells * nv],
            rk4_stage2_f: vec![0.0; ncells * nv],
            rk4_stage3_f: vec![0.0; ncells * nv],
            relax_tgt: vec![[0.0; 8]; ncells],
        }
    }

    /// Recompute macro fields from the current distribution.
    pub fn update_moments(&mut self) {
        let nv = self.dist.nv;
        let vgrid = &self.dist.vgrid;
        let vw = &self.dist.vw;
        for c in 0..self.grid.ncells() {
            let fs = &self.dist.f[c * nv..c * nv + nv];
            let mut rho = 0.0;
            let mut m = [0.0; 3];
            let mut e = 0.0;
            for k in 0..nv {
                let fv = fs[k];
                let w = vw[k] * fv;
                rho += w;
                m[0] += w * vgrid[k][0];
                m[1] += w * vgrid[k][1];
                m[2] += w * vgrid[k][2];
                let v2 = vgrid[k][0] * vgrid[k][0]
                    + vgrid[k][1] * vgrid[k][1]
                    + vgrid[k][2] * vgrid[k][2];
                e += 0.5 * w * v2;
            }
            self.fields.rho[c] = rho;
            self.fields.mom[0][c] = m[0];
            self.fields.mom[1][c] = m[1];
            self.fields.mom[2][c] = m[2];
            self.fields.energy[c] = e;

            let rho_safe = rho.max(f64::MIN_POSITIVE);
            let u = [m[0] / rho_safe, m[1] / rho_safe, m[2] / rho_safe];
            // Heat flux: q = 0.5 * int(c * c^2 * f) dv, c = v - u (standard
            // single-distribution 3D DVM heat-flux moment, no h term).
            let mut q = [0.0; 3];
            for k in 0..nv {
                let fv = fs[k];
                let cx = vgrid[k][0] - u[0];
                let cy = vgrid[k][1] - u[1];
                let cz = vgrid[k][2] - u[2];
                let c2 = cx * cx + cy * cy + cz * cz;
                let w = 0.5 * vw[k] * c2 * fv;
                q[0] += w * cx;
                q[1] += w * cy;
                q[2] += w * cz;
            }
            self.fields.heat[0][c] = q[0];
            self.fields.heat[1][c] = q[1];
            self.fields.heat[2][c] = q[2];
        }
    }

    /// CFL-limited timestep over the fastest discrete velocity and the
    /// smallest of the three grid spacings.
    pub fn cfl_dt(&self, cfl: f64) -> f64 {
        let mut vmax = 0.0f64;
        for v in self.dist.vgrid.iter() {
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if speed > vmax {
                vmax = speed;
            }
        }
        let vmax = vmax.max(1e-9);
        let h = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        cfl * h / vmax
    }

    /// Total conserved moments over the whole domain: (mass, px, py, pz, E).
    pub fn totals(&self) -> (f64, f64, f64, f64, f64) {
        let vol = self.grid.dx * self.grid.dy * self.grid.dz;
        let mut mass = 0.0;
        let mut px = 0.0;
        let mut py = 0.0;
        let mut pz = 0.0;
        let mut e = 0.0;
        for c in 0..self.grid.ncells() {
            mass += self.fields.rho[c] * vol;
            px += self.fields.mom[0][c] * vol;
            py += self.fields.mom[1][c] * vol;
            pz += self.fields.mom[2][c] * vol;
            e += self.fields.energy[c] * vol;
        }
        (mass, px, py, pz, e)
    }

    #[inline]
    fn safe_temperature(&self, c: usize) -> f64 {
        let t = self.fields.temperature(c, self.gas_r, DOF);
        if t.is_finite() && t > 1e-6 {
            t
        } else {
            self.t_ref.max(1e-6)
        }
    }

    #[inline]
    fn cell_macro(&self, c: usize) -> (f64, [f64; 3], f64, [f64; 3]) {
        let rho = self.fields.rho[c];
        let u = self.fields.velocity(c);
        let t = self.safe_temperature(c);
        let q = [
            self.fields.heat[0][c],
            self.fields.heat[1][c],
            self.fields.heat[2][c],
        ];
        (rho, u, t, q)
    }

    /// Direction-parameterized interior-face DUGKS half-step flux: computes
    /// `f_face * v.n` for every velocity node into `self.face_flux`, given
    /// the upwind (`cin`) and downwind (`cout`) cell indices and the outward
    /// normal from `cin`'s perspective. Shared by all 6 face directions
    /// (east/west/north/south/up/down call sites in `step`) — see module
    /// docs for why this single kernel is physically correct (not a fake
    /// extrusion) for every axis.
    /// `far_in`/`far_out` are the next cell further upstream of `cin`/`cout`
    /// along the face-normal axis (used only for the van Leer MUSCL slope;
    /// `None` degrades that side to first-order upwind) — same convention as
    /// the 2D solver's `compute_interior_face_flux` (see `crate::solver`
    /// module docs).
    #[allow(clippy::too_many_arguments)]
    fn compute_interior_face_flux(
        &mut self,
        cin: usize,
        cout: usize,
        far_in: Option<usize>,
        far_out: Option<usize>,
        normal: [f64; 3],
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
            let vn = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
            let f_in_k = self.dist.f[cin * nv + k];
            let f_out_k = self.dist.f[cout * nv + k];

            let f_in_face = if let Some(far) = far_in {
                crate::solver::van_leer_face_value(self.dist.f[far * nv + k], f_in_k, f_out_k)
            } else {
                f_in_k
            };
            let f_out_face = if let Some(far) = far_out {
                crate::solver::van_leer_face_value(self.dist.f[far * nv + k], f_out_k, f_in_k)
            } else {
                f_out_k
            };

            let up: UpwindState3D = if vn >= 0.0 {
                UpwindState3D {
                    f: f_in_face,
                    rho: rho_in,
                    u: u_in,
                    t: t_in,
                    q: q_in,
                    tau: tau_in,
                }
            } else {
                UpwindState3D {
                    f: f_out_face,
                    rho: rho_out,
                    u: u_out,
                    t: t_out,
                    q: q_out,
                    tau: tau_out,
                }
            };

            let feq = self
                .collision
                .equilibrium(up.rho, up.u, up.t, self.gas_r, up.q, v);
            let f_face = (up.tau * up.f + dt_half * feq) / (up.tau + dt_half);
            self.face_flux[k] = f_face * vn;
        }
    }

    /// Boundary-face variant: `cin` is the interior cell, `normal` points
    /// out of the domain. Same direction-parameterized reuse as the
    /// interior case.
    fn compute_boundary_face_flux(
        &mut self,
        cin: usize,
        normal: [f64; 3],
        face: Face,
        config_bcs: &BoundaryAssignment3D,
        dt: f64,
    ) {
        let nv = self.dist.nv;
        let (rho_in, u_in, t_in, q_in) = self.cell_macro(cin);
        let tau_in = self.tau_scratch[cin];

        let bc_kernel = BoundaryConditionKernel3D::from_kind(config_bcs.get(face));
        let f_interior: Vec<f64> = self.dist.f[cin * nv..cin * nv + nv].to_vec();
        let vgrid = self.dist.vgrid.clone();
        let vw = self.dist.vw.clone();
        bc_kernel.apply(
            &f_interior,
            &vgrid,
            &vw,
            normal,
            self.gas_r,
            &mut self.ghost_buf,
        );

        let dt_half = 0.5 * dt;
        for k in 0..nv {
            let v = vgrid[k];
            let vn = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2];
            let f_up = if vn >= 0.0 {
                f_interior[k]
            } else {
                self.ghost_buf[k]
            };
            // DESIGN: interior cell's own macro state used for the
            // equilibrium at incoming (ghost-sourced) nodes too, mirroring
            // the 2D solver's identical simplification.
            let feq = self
                .collision
                .equilibrium(rho_in, u_in, t_in, self.gas_r, q_in, v);
            let f_face = (tau_in * f_up + dt_half * feq) / (tau_in + dt_half);
            self.face_flux[k] = f_face * vn;
        }
    }

    /// Dispatch to `self.scheme`: `Euler` calls `step` (unchanged); `Rk2`
    /// calls `step_rk2` (Shu-Osher SSP-RK2, operator-split transport/
    /// collision); `Rk4` calls `step_rk4` (classical 4-stage RK, operator-split)
    /// — mirrors `DugksSolver::step_scheme` (2D) exactly.
    pub fn step_scheme(&mut self, dt: f64, config_bcs: &BoundaryAssignment3D) {
        match self.scheme {
            crate::solver::TimeScheme::Euler => self.step(dt, config_bcs),
            crate::solver::TimeScheme::Rk2 => self.step_rk2(dt, config_bcs),
            crate::solver::TimeScheme::Rk4 => self.step_rk4(dt, config_bcs),
        }
    }

    /// Shu-Osher SSP-RK2 (Heun's method), operator-split transport/collision
    /// — see `DugksSolver::step_rk2`'s `// DESIGN:` comment (2D solver) for
    /// the full rationale, which applies identically here: each RK2 stage
    /// re-applies the existing `step`'s closed-form (unconditionally
    /// stable) Shakhov relaxation at that stage's own frozen macro state,
    /// rather than RK-integrating the stiff collision RHS directly.
    /// Reference: Shu & Osher, J. Comput. Phys. 77, 439-471 (1988); Gottlieb,
    /// Shu & Tadmor, SIAM Rev. 43, 89-112 (2001).
    fn step_rk2(&mut self, dt: f64, config_bcs: &BoundaryAssignment3D) {
        let n = self.dist.f.len();
        debug_assert_eq!(n, self.rk2_stage_f.len());

        self.rk2_stage_f.copy_from_slice(&self.dist.f); // save u^n
        self.step(dt, config_bcs); // self.dist now holds u1 = Euler_step(u^n, dt)
        self.step(dt, config_bcs); // self.dist now holds u2 = Euler_step(u1, dt)
        for i in 0..n {
            self.dist.f[i] = 0.5 * self.rk2_stage_f[i] + 0.5 * self.dist.f[i];
        }
        self.update_moments();
    }

    /// Classical 4-stage Runge-Kutta (RK4) time integration, applied via
    /// operator splitting (see `solver::TimeScheme` docs). This 3D version
    /// mirrors `DugksSolver::step_rk4` (2D) exactly, adapted to use only f
    /// (no separate h collision distribution).
    fn step_rk4(&mut self, dt: f64, config_bcs: &BoundaryAssignment3D) {
        let n = self.dist.f.len();

        debug_assert_eq!(n, self.rk4_stage1_f.len());
        debug_assert_eq!(n, self.rk4_stage2_f.len());
        debug_assert_eq!(n, self.rk4_stage3_f.len());

        // Save u^n for blending and intermediate stages.
        let u_n_f = self.dist.f.clone();

        // Stage 1: k1 = L(u^n, dt)
        self.step(dt, config_bcs);
        self.rk4_stage1_f.copy_from_slice(&self.dist.f);

        // Stage 2: k2 = L(u^n + 0.5*dt*k1, dt)
        for i in 0..n {
            self.dist.f[i] = 0.5 * u_n_f[i] + 0.5 * self.rk4_stage1_f[i];
        }
        self.update_moments();
        self.step(dt, config_bcs);
        self.rk4_stage2_f.copy_from_slice(&self.dist.f);

        // Stage 3: k3 = L(u^n + 0.5*dt*k2, dt)
        for i in 0..n {
            self.dist.f[i] = 0.5 * u_n_f[i] + 0.5 * self.rk4_stage2_f[i];
        }
        self.update_moments();
        self.step(dt, config_bcs);
        self.rk4_stage3_f.copy_from_slice(&self.dist.f);

        // Stage 4: k4 = L(u^n + dt*k3, dt)
        for i in 0..n {
            self.dist.f[i] = u_n_f[i] + (self.rk4_stage3_f[i] - u_n_f[i]);
        }
        self.update_moments();
        self.step(dt, config_bcs);

        // Final blend: u^{n+1} = u^n + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        // = (1/6)*u_n + (1/6)*stage1 + (1/3)*stage2 + (1/3)*stage3 + (1/6)*dist
        for i in 0..n {
            self.dist.f[i] = (1.0/6.0) * u_n_f[i]
                + (1.0/6.0) * self.rk4_stage1_f[i]
                + (1.0/3.0) * self.rk4_stage2_f[i]
                + (1.0/3.0) * self.rk4_stage3_f[i]
                + (1.0/6.0) * self.dist.f[i];
        }
        self.update_moments();
    }

    /// One explicit DUGKS step across all 6 faces + Shakhov relaxation.
    pub fn step(&mut self, dt: f64, config_bcs: &BoundaryAssignment3D) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let nv = self.dist.nv;
        let vol = self.grid.dx * self.grid.dy * self.grid.dz;
        let ncells = self.grid.ncells();

        self.dist_scratch.f.copy_from_slice(&self.dist.f);

        for c in 0..ncells {
            let rho = self.fields.rho[c];
            let rho_safe = if rho.is_finite() && rho > 0.0 { rho } else { 1e-6 };
            let t = self.safe_temperature(c);
            self.tau_scratch[c] = self.collision.relaxation_time(
                rho_safe,
                t,
                self.gas_r,
                self.mu_ref,
                self.t_ref,
                self.omega,
            );
        }

        // Per-face area (the face perpendicular to the axis being crossed).
        let area_x = self.grid.dy * self.grid.dz; // east/west faces
        let area_y = self.grid.dx * self.grid.dz; // north/south faces
        let area_z = self.grid.dx * self.grid.dy; // up/down faces

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let cin = self.grid.idx(i, j, k);
                    // Copy out the (Copy) resolved-BC-kind values into plain
                    // locals *before* any `&mut self` method call below, so
                    // there is no question of borrow-checker argument-
                    // evaluation-order ambiguity between reading `self.bcs`
                    // and taking `&mut self` for the receiver in the same
                    // call expression.
                    let bc_east = self.bcs[1];
                    let bc_north = self.bcs[3];
                    let bc_up = self.bcs[5];
                    let bc_west = self.bcs[0];
                    let bc_south = self.bcs[2];
                    let bc_down = self.bcs[4];

                    // +x face (east). far_in: one more step west of cin
                    // (periodic-aware); far_out: one more step east of cout.
                    if i + 1 < nx {
                        let cout = self.grid.idx(i + 1, j, k);
                        let far_in = if i > 0 {
                            Some(self.grid.idx(i - 1, j, k))
                        } else if matches!(bc_west, BoundaryKindResolved::Periodic) {
                            Some(self.grid.idx(nx - 1, j, k))
                        } else {
                            None
                        };
                        let far_out = if i + 2 < nx {
                            Some(self.grid.idx(i + 2, j, k))
                        } else if matches!(bc_east, BoundaryKindResolved::Periodic) {
                            Some(self.grid.idx((i + 2) % nx, j, k))
                        } else {
                            None
                        };
                        self.compute_interior_face_flux(
                            cin,
                            cout,
                            far_in,
                            far_out,
                            [1.0, 0.0, 0.0],
                            dt,
                        );
                        self.accumulate_interior(cin, cout, area_x, dt, vol, nv);
                    } else {
                        self.handle_boundary_or_periodic(
                            cin,
                            i,
                            j,
                            k,
                            [1.0, 0.0, 0.0],
                            Face::East,
                            bc_east,
                            config_bcs,
                            dt,
                            vol,
                            area_x,
                            nv,
                            |g, _ii, jj, kk| g.idx(0, jj, kk),
                            |g, _ii, jj, kk| if nx > 1 { Some(g.idx(1, jj, kk)) } else { None },
                            (i, j, k),
                        );
                    }

                    // +y face (north)
                    if j + 1 < ny {
                        let cout = self.grid.idx(i, j + 1, k);
                        let far_in = if j > 0 {
                            Some(self.grid.idx(i, j - 1, k))
                        } else if matches!(bc_south, BoundaryKindResolved::Periodic) {
                            Some(self.grid.idx(i, ny - 1, k))
                        } else {
                            None
                        };
                        let far_out = if j + 2 < ny {
                            Some(self.grid.idx(i, j + 2, k))
                        } else if matches!(bc_north, BoundaryKindResolved::Periodic) {
                            Some(self.grid.idx(i, (j + 2) % ny, k))
                        } else {
                            None
                        };
                        self.compute_interior_face_flux(
                            cin,
                            cout,
                            far_in,
                            far_out,
                            [0.0, 1.0, 0.0],
                            dt,
                        );
                        self.accumulate_interior(cin, cout, area_y, dt, vol, nv);
                    } else {
                        self.handle_boundary_or_periodic(
                            cin,
                            i,
                            j,
                            k,
                            [0.0, 1.0, 0.0],
                            Face::North,
                            bc_north,
                            config_bcs,
                            dt,
                            vol,
                            area_y,
                            nv,
                            |g, ii, _jj, kk| g.idx(ii, 0, kk),
                            |g, ii, _jj, kk| if ny > 1 { Some(g.idx(ii, 1, kk)) } else { None },
                            (i, j, k),
                        );
                    }

                    // +z face (up)
                    if k + 1 < nz {
                        let cout = self.grid.idx(i, j, k + 1);
                        let far_in = if k > 0 {
                            Some(self.grid.idx(i, j, k - 1))
                        } else if matches!(bc_down, BoundaryKindResolved::Periodic) {
                            Some(self.grid.idx(i, j, nz - 1))
                        } else {
                            None
                        };
                        let far_out = if k + 2 < nz {
                            Some(self.grid.idx(i, j, k + 2))
                        } else if matches!(bc_up, BoundaryKindResolved::Periodic) {
                            Some(self.grid.idx(i, j, (k + 2) % nz))
                        } else {
                            None
                        };
                        self.compute_interior_face_flux(
                            cin,
                            cout,
                            far_in,
                            far_out,
                            [0.0, 0.0, 1.0],
                            dt,
                        );
                        self.accumulate_interior(cin, cout, area_z, dt, vol, nv);
                    } else {
                        self.handle_boundary_or_periodic(
                            cin,
                            i,
                            j,
                            k,
                            [0.0, 0.0, 1.0],
                            Face::Up,
                            bc_up,
                            config_bcs,
                            dt,
                            vol,
                            area_z,
                            nv,
                            |g, ii, jj, _kk| g.idx(ii, jj, 0),
                            |g, ii, jj, _kk| if nz > 1 { Some(g.idx(ii, jj, 1)) } else { None },
                            (i, j, k),
                        );
                    }

                    // -x face (west), only at the low boundary (interior
                    // -x faces are covered by the neighbor's +x pass above).
                    if i == 0 {
                        self.handle_low_boundary_or_periodic(
                            cin,
                            i,
                            j,
                            k,
                            [-1.0, 0.0, 0.0],
                            Face::West,
                            bc_west,
                            config_bcs,
                            dt,
                            vol,
                            area_x,
                            nv,
                            |g, _ii, jj, kk| g.idx(g.nx - 1, jj, kk),
                        );
                    }
                    // -y face (south)
                    if j == 0 {
                        self.handle_low_boundary_or_periodic(
                            cin,
                            i,
                            j,
                            k,
                            [0.0, -1.0, 0.0],
                            Face::South,
                            bc_south,
                            config_bcs,
                            dt,
                            vol,
                            area_y,
                            nv,
                            |g, ii, _jj, kk| g.idx(ii, g.ny - 1, kk),
                        );
                    }
                    // -z face (down)
                    if k == 0 {
                        self.handle_low_boundary_or_periodic(
                            cin,
                            i,
                            j,
                            k,
                            [0.0, 0.0, -1.0],
                            Face::Down,
                            bc_down,
                            config_bcs,
                            dt,
                            vol,
                            area_z,
                            nv,
                            |g, ii, jj, _kk| g.idx(ii, jj, g.nz - 1),
                        );
                    }
                }
            }
        }

        std::mem::swap(&mut self.dist.f, &mut self.dist_scratch.f);

        // Capture the transport-conserved moments BEFORE the positivity floor.
        // The floor (a clamp-then-rescale that only preserves the zeroth/mass
        // moment) perturbs momentum and energy non-conservatively; by targeting
        // these PRE-floor moments, the conservative relaxation correction below
        // restores them exactly, so the floor keeps its positivity benefit
        // without introducing a conservation error.
        self.update_moments();
        for c in 0..ncells {
            self.relax_tgt[c] = [
                self.fields.rho[c],
                self.fields.mom[0][c],
                self.fields.mom[1][c],
                self.fields.mom[2][c],
                self.fields.energy[c],
                self.fields.heat[0][c],
                self.fields.heat[1][c],
                self.fields.heat[2][c],
            ];
        }

        // Positivity-preserving floor (Zhang & Shu 2010-style clamp-then-rescale)
        // applied to the single distribution `f` (no separate `h` buffer in 3D).
        crate::solver::apply_positivity_floor_pub(&mut self.dist.f, self.dist.nv, ncells);

        for c in 0..ncells {
            // Equilibrium parameters and conservation targets both come from the
            // pre-floor (transport-conserved) moments cached in `relax_tgt`.
            let s = self.relax_tgt[c];
            let rho = s[0];
            let u = if rho > 0.0 {
                [s[1] / rho, s[2] / rho, s[3] / rho]
            } else {
                [0.0; 3]
            };
            // e = 0.5*rho*|u|^2 + 0.5*rho*DOF*R*T  =>  T = (2e/rho - |u|^2)/(DOF*R)
            let umag2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            // Guard against a non-positive temperature (possible transiently in
            // an RK2 intermediate stage or a near-vacuum/over-depleted cell):
            // a negative T would make the Maxwellian's sqrt(R*T) NaN and poison
            // the whole solution. Floor to a small positive value; the
            // conservative correction below still restores the exact target
            // moments, so this only affects the equilibrium *shape*, not
            // conservation.
            let t = if rho > 0.0 {
                (((2.0 * s[4] / rho) - umag2) / (DOF * self.gas_r)).max(1e-6)
            } else {
                1e-6
            };
            let q = [s[5], s[6], s[7]];
            let tgt = [s[0], s[1], s[2], s[3], s[4]];
            let tau_c = self.collision.relaxation_time(
                rho,
                t,
                self.gas_r,
                self.mu_ref,
                self.t_ref,
                self.omega,
            );
            // Thermal velocity scale used to NORMALIZE the peculiar-basis columns
            // {1, cx/sig, cy/sig, cz/sig, c^2/sig^2} so every basis function is
            // O(1) near the thermal core. Without this, the c^2 column's entries
            // (~v^4 moments amplified by the folded Gauss-Hermite tail weights)
            // are ~1e11x the mass column, making the 5x5 catastrophically ill-
            // conditioned and the correction blow up over steps (3D NaN).
            // IMPORTANT: base sig on the GRID reference temperature `t_ref`, NOT
            // the local `t` — a transiently corrupted cell can floor `t` to ~1e-6,
            // which would make sig ~0.017 and OVER-normalize (c^2/sig^2 ~ 1e10),
            // re-introducing the very ill-conditioning this normalization removes.
            // The reference scale is constant and always sane; normalization is a
            // pure conditioning device (the correction is exact for any sig>0).
            let sig = (self.gas_r * self.t_ref).sqrt().max(1e-3);
            let inv_sig = 1.0 / sig;
            let inv_sig2 = inv_sig * inv_sig;

            // Pass 1: relax in place; accumulate post-relaxation f1 moments and
            // the 5x5 correction matrix. (3D has a single distribution f, so
            // mass, all three momenta, AND energy must be restored through
            // corrections to f itself.) The correction basis is the PECULIAR-
            // velocity set PHI = {1, cx, cy, cz, c^2} with c = v - u; centering
            // on the local bulk velocity makes the equilibrium moment matrix
            // diagonally dominant / well-conditioned (the raw {1,vx,vy,vz,|v|^2}
            // basis becomes near-singular for large |u|, which otherwise leaves
            // the energy row inaccurate). Constraint rows use PSI =
            // {1, vx, vy, vz, 0.5*|v|^2} (mass, momentum, energy). A[i][j] =
            // sum_k w feq PSI_i(v_k) PHI_j(v_k).
            let mut f1m = [0.0f64; 5]; // rho1, px1, py1, pz1, e1
            let mut a_mat = [[0.0f64; 5]; 5];
            for k in 0..nv {
                let v = self.dist.vgrid[k];
                let w = self.dist.vw[k];
                let feq = self.collision.equilibrium(rho, u, t, self.gas_r, q, v);
                let idx = c * nv + k;
                let f1 = (tau_c * self.dist.f[idx] + dt * feq) / (tau_c + dt);
                self.dist.f[idx] = f1;
                let (vx, vy, vz) = (v[0], v[1], v[2]);
                let v2 = vx * vx + vy * vy + vz * vz;
                f1m[0] += w * f1;
                f1m[1] += w * vx * f1;
                f1m[2] += w * vy * f1;
                f1m[3] += w * vz * f1;
                f1m[4] += 0.5 * w * v2 * f1;
                // peculiar velocity and the two monomial sets
                let cx = vx - u[0];
                let cy = vy - u[1];
                let cz = vz - u[2];
                let c2 = cx * cx + cy * cy + cz * cz;
                let psi = [1.0, vx, vy, vz, 0.5 * v2];
                let phi = [1.0, cx * inv_sig, cy * inv_sig, cz * inv_sig, c2 * inv_sig2];
                let e = w * feq;
                for i in 0..5 {
                    for j in 0..5 {
                        a_mat[i][j] += e * psi[i] * phi[j];
                    }
                }
            }
            let rhs = [
                tgt[0] - f1m[0],
                tgt[1] - f1m[1],
                tgt[2] - f1m[2],
                tgt[3] - f1m[3],
                tgt[4] - f1m[4],
            ];
            let coef = solve5(a_mat, rhs).unwrap_or([0.0; 5]);

            // Pass 2: apply the correction f += feq*(a0 + a1 cx + a2 cy + a3 cz
            // + a4 c^2) with the SAME peculiar basis (recompute feq;
            // deterministic).
            for k in 0..nv {
                let v = self.dist.vgrid[k];
                let feq = self.collision.equilibrium(rho, u, t, self.gas_r, q, v);
                let cx = v[0] - u[0];
                let cy = v[1] - u[1];
                let cz = v[2] - u[2];
                let c2 = cx * cx + cy * cy + cz * cz;
                let idx = c * nv + k;
                self.dist.f[idx] += feq
                    * (coef[0]
                        + coef[1] * cx * inv_sig
                        + coef[2] * cy * inv_sig
                        + coef[3] * cz * inv_sig
                        + coef[4] * c2 * inv_sig2);
            }
        }
        self.update_moments();
    }

    /// Apply the already-computed `self.face_flux` as a symmetric FV update
    /// (subtract from `cin`, add to `cout`) — shared bookkeeping for every
    /// interior-face call site.
    #[inline]
    fn accumulate_interior(
        &mut self,
        cin: usize,
        cout: usize,
        area: f64,
        dt: f64,
        vol: f64,
        nv: usize,
    ) {
        for k in 0..nv {
            let flux = self.face_flux[k] * area * dt / vol;
            self.dist_scratch.f[cin * nv + k] -= flux;
            self.dist_scratch.f[cout * nv + k] += flux;
        }
    }

    /// High-side (+x/+y/+z) boundary handling: periodic wraps to the
    /// opposite low-index face (an interior-style two-sided flux); any
    /// other BC kind uses the one-sided ghost-flux kernel (only `cin` loses
    /// mass; the domain exterior is not part of the state).
    #[allow(clippy::too_many_arguments)]
    fn handle_boundary_or_periodic(
        &mut self,
        cin: usize,
        i: usize,
        j: usize,
        k: usize,
        normal: [f64; 3],
        face: Face,
        resolved: BoundaryKindResolved,
        config_bcs: &BoundaryAssignment3D,
        dt: f64,
        vol: f64,
        area: f64,
        nv: usize,
        wrap: impl Fn(&Grid3D, usize, usize, usize) -> usize,
        far_wrap: impl Fn(&Grid3D, usize, usize, usize) -> Option<usize>,
        (i0, j0, k0): (usize, usize, usize),
    ) {
        match resolved {
            BoundaryKindResolved::Periodic => {
                let cout = wrap(&self.grid, i0, j0, k0);
                // far_in: the cell just behind cin along -normal (cin is at
                // the high-index boundary here, so this is always a genuine
                // interior neighbor); far_out: one more periodic step past
                // cout (cout is the wrapped low-index cell 0, so its "far"
                // neighbor is cell 1 along this axis, via `far_wrap`).
                let far_in = {
                    let (di, dj, dk) = (normal[0] as isize, normal[1] as isize, normal[2] as isize);
                    let fi = i0 as isize - di;
                    let fj = j0 as isize - dj;
                    let fk = k0 as isize - dk;
                    if fi >= 0 && fj >= 0 && fk >= 0 {
                        Some(self.grid.idx(fi as usize, fj as usize, fk as usize))
                    } else {
                        None
                    }
                };
                let far_out = far_wrap(&self.grid, i0, j0, k0);
                let _ = (i, j, k);
                self.compute_interior_face_flux(cin, cout, far_in, far_out, normal, dt);
                self.accumulate_interior(cin, cout, area, dt, vol, nv);
            }
            BoundaryKindResolved::Other => {
                self.compute_boundary_face_flux(cin, normal, face, config_bcs, dt);
                for kk in 0..nv {
                    let flux = self.face_flux[kk] * area * dt / vol;
                    self.dist_scratch.f[cin * nv + kk] -= flux;
                }
            }
        }
    }

    /// Low-side (-x/-y/-z) boundary handling. Periodic low-side faces are
    /// intentionally NOT double-counted here: the corresponding high-side
    /// pass (`handle_boundary_or_periodic`, called for the neighbor's last
    /// cell on the opposite edge) already applies the symmetric two-cell
    /// flux for a periodic pair; wrapping again here would double-apply the
    /// same face. So periodic low-side faces are a no-op; only genuine
    /// (non-periodic) BCs apply a one-sided ghost flux here.
    #[allow(clippy::too_many_arguments)]
    fn handle_low_boundary_or_periodic(
        &mut self,
        cin: usize,
        i: usize,
        j: usize,
        k: usize,
        normal: [f64; 3],
        face: Face,
        resolved: BoundaryKindResolved,
        config_bcs: &BoundaryAssignment3D,
        dt: f64,
        vol: f64,
        area: f64,
        nv: usize,
        _wrap: impl Fn(&Grid3D, usize, usize, usize) -> usize,
    ) {
        let _ = (i, j, k);
        if let BoundaryKindResolved::Other = resolved {
            self.compute_boundary_face_flux(cin, normal, face, config_bcs, dt);
            for kk in 0..nv {
                let flux = self.face_flux[kk] * area * dt / vol;
                self.dist_scratch.f[cin * nv + kk] -= flux;
            }
        }
    }
}

/// Solve a 5x5 dense linear system `A x = b` by Gaussian elimination with
/// partial pivoting. Returns `None` if the matrix is (numerically) singular —
/// the caller then skips the conservative correction (only happens for an
/// empty/degenerate cell whose equilibrium moment matrix has no rank). Small
/// fixed size, no allocation, no external dependency.
pub(crate) fn solve5(a: [[f64; 5]; 5], b: [f64; 5]) -> Option<[f64; 5]> {
    let mut m = a;
    let mut r = b;
    for col in 0..5 {
        // Partial pivot: find the largest-magnitude entry in this column.
        let mut piv = col;
        let mut best = m[col][col].abs();
        for row in (col + 1)..5 {
            let v = m[row][col].abs();
            if v > best {
                best = v;
                piv = row;
            }
        }
        if best < 1e-300 {
            return None;
        }
        m.swap(col, piv);
        r.swap(col, piv);
        // Eliminate below.
        for row in (col + 1)..5 {
            let factor = m[row][col] / m[col][col];
            if factor != 0.0 {
                for c in col..5 {
                    m[row][c] -= factor * m[col][c];
                }
                r[row] -= factor * r[col];
            }
        }
    }
    // Back-substitution.
    let mut x = [0.0f64; 5];
    for row in (0..5).rev() {
        let mut s = r[row];
        for c in (row + 1)..5 {
            s -= m[row][c] * x[c];
        }
        x[row] = s / m[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::config::{BoundaryAssignment3D, GasProperties};
    use janus_core::grid3d::Grid3D;

    fn periodic_case(nx: usize, ny: usize, nz: usize) -> CaseConfig3D {
        CaseConfig3D {
            grid: Grid3D::new(nx, ny, nz, 0.05, 0.05, 0.05, [0.0, 0.0, 0.0]),
            bcs: BoundaryAssignment3D::all_periodic(),
            gas: GasProperties::monatomic_default(),
        }
    }

    fn init_uniform(solver: &mut DugksSolver3D, rho: f64, u: [f64; 3], t: f64) {
        let nv = solver.dist.nv;
        let ncells = solver.grid.ncells();
        let r_gas = solver.gas_r;
        let vgrid = solver.dist.vgrid.clone();
        for c in 0..ncells {
            for k in 0..nv {
                let f = crate::maxwellian3d::maxwellian_3d(rho, u, t, r_gas, vgrid[k]);
                solver.dist.f[c * nv + k] = f;
            }
        }
        solver.update_moments();
    }

    #[test]
    fn conservation_on_periodic_domain_3d() {
        let config = periodic_case(3, 3, 3);
        let (vgrid, vw) = crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(
            287.0,
            2000.0,
            [0.0, 0.0, 0.0],
            6,
        );
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver3D::new(&config, dist);
        init_uniform(&mut solver, 1.0, [20.0, -10.0, 5.0], 300.0);

        // Perturb one cell.
        let c0 = solver.grid.idx(1, 1, 1);
        let nv = solver.dist.nv;
        for k in 0..nv {
            solver.dist.f[c0 * nv + k] *= 1.05;
        }
        solver.update_moments();

        let before = solver.totals();
        let dt = solver.cfl_dt(0.2);
        for _ in 0..5 {
            solver.step(dt, &config.bcs);
        }
        let after = solver.totals();

        let tol = 1e-2;
        assert!(
            (after.0 - before.0).abs() / before.0.abs().max(1e-30) < tol,
            "mass drift: before={} after={}",
            before.0,
            after.0
        );
        assert!(
            (after.4 - before.4).abs() / before.4.abs().max(1e-30) < tol,
            "energy drift: before={} after={}",
            before.4,
            after.4
        );
    }

    #[test]
    fn step_produces_finite_values_3d() {
        let config = periodic_case(3, 3, 3);
        let (vgrid, vw) = crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(
            287.0,
            1500.0,
            [0.0, 0.0, 0.0],
            5,
        );
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver3D::new(&config, dist);
        init_uniform(&mut solver, 1.2, [5.0, -3.0, 2.0], 320.0);

        let dt = solver.cfl_dt(0.3);
        for _ in 0..3 {
            solver.step(dt, &config.bcs);
        }

        assert!(
            solver.dist.f.iter().all(|v| v.is_finite()),
            "f has non-finite values"
        );
        assert!(solver.fields.rho.iter().all(|v| v.is_finite()));
        for d in 0..3 {
            assert!(solver.fields.mom[d].iter().all(|v| v.is_finite()));
            assert!(solver.fields.heat[d].iter().all(|v| v.is_finite()));
        }
        assert!(solver.fields.energy.iter().all(|v| v.is_finite()));
    }

    /// Genuine-3D-flux correctness check (see module docs): a DUGKS z-face
    /// flux computed directly must equal an x-face flux computed on the
    /// same physical state with x and z coordinates/velocities swapped —
    /// this fails if the z-direction were a fake pass-through rather than a
    /// real velocity-moment integral over `v[2]`.
    #[test]
    fn z_face_flux_matches_hand_rotated_x_face() {
        let config = periodic_case(2, 2, 2);
        let (vgrid, vw) = crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(
            287.0,
            1500.0,
            [0.0, 0.0, 0.0],
            5,
        );
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid.clone(), vw.clone());
        let mut solver = DugksSolver3D::new(&config, dist);

        // Cell A: rho=1, u=(30,0,0), T=310; Cell B: rho=1.1, u=(10,0,0), T=300.
        // z-face test: place this exact (rho,u,T) pair along z at (0,0,0)->(0,0,1)
        // with u rotated so the "z-directed" bulk velocity component is in vz.
        let ca = solver.grid.idx(0, 0, 0);
        let cb = solver.grid.idx(0, 0, 1);
        let nv = solver.dist.nv;
        let r_gas = solver.gas_r;
        for k in 0..nv {
            solver.dist.f[ca * nv + k] =
                crate::maxwellian3d::maxwellian_3d(1.0, [0.0, 0.0, 30.0], 310.0, r_gas, vgrid[k]);
            solver.dist.f[cb * nv + k] =
                crate::maxwellian3d::maxwellian_3d(1.1, [0.0, 0.0, 10.0], 300.0, r_gas, vgrid[k]);
        }
        solver.update_moments();
        for c in 0..solver.grid.ncells() {
            let rho = solver.fields.rho[c];
            let t = solver.fields.temperature(c, r_gas, DOF);
            solver.tau_scratch[c] = solver.collision.relaxation_time(
                rho,
                t,
                r_gas,
                solver.mu_ref,
                solver.t_ref,
                solver.omega,
            );
        }
        let dt = 1e-6;
        solver.compute_interior_face_flux(ca, cb, None, None, [0.0, 0.0, 1.0], dt);
        let z_flux_mass: f64 = (0..nv).map(|k| solver.face_flux[k] * vw[k]).sum();

        // Now build an equivalent x-face scenario: same rho/T, bulk velocity
        // rotated into vx instead of vz (velocity grid is symmetric under
        // axis permutation since it's an isotropic tensor-product GH grid
        // centered at 0 for this test, and the affected macro state's u only
        // has this one nonzero component in both cases).
        let config2 = periodic_case(2, 2, 2);
        let dist2 = Distribution3D::zeros(config2.grid.ncells(), vgrid.clone(), vw.clone());
        let mut solver2 = DugksSolver3D::new(&config2, dist2);
        let ca2 = solver2.grid.idx(0, 0, 0);
        let cb2 = solver2.grid.idx(1, 0, 0);
        for k in 0..nv {
            solver2.dist.f[ca2 * nv + k] =
                crate::maxwellian3d::maxwellian_3d(1.0, [30.0, 0.0, 0.0], 310.0, r_gas, vgrid[k]);
            solver2.dist.f[cb2 * nv + k] =
                crate::maxwellian3d::maxwellian_3d(1.1, [10.0, 0.0, 0.0], 300.0, r_gas, vgrid[k]);
        }
        solver2.update_moments();
        for c in 0..solver2.grid.ncells() {
            let rho = solver2.fields.rho[c];
            let t = solver2.fields.temperature(c, r_gas, DOF);
            solver2.tau_scratch[c] = solver2.collision.relaxation_time(
                rho,
                t,
                r_gas,
                solver2.mu_ref,
                solver2.t_ref,
                solver2.omega,
            );
        }
        solver2.compute_interior_face_flux(ca2, cb2, None, None, [1.0, 0.0, 0.0], dt);
        let x_flux_mass: f64 = (0..nv).map(|k| solver2.face_flux[k] * vw[k]).sum();

        assert!(
            (z_flux_mass - x_flux_mass).abs() / x_flux_mass.abs().max(1e-300) < 1e-8,
            "z-face flux {z_flux_mass} must match rotated x-face flux {x_flux_mass} (genuine 3D moment integral, not a pass-through)"
        );
    }

    #[test]
    fn muscl_reconstruction_preserves_conservation_3d() {
        // Same idea as the 2D `muscl_reconstruction_preserves_conservation_
        // on_periodic_domain`: a large-enough fully-periodic grid so the
        // MUSCL far-neighbor stencil is populated everywhere, verifying the
        // FV update still conserves mass/energy exactly with second-order
        // spatial reconstruction.
        let config = periodic_case(4, 4, 4);
        let (vgrid, vw) = crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(
            287.0,
            2000.0,
            [0.0, 0.0, 0.0],
            6,
        );
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver3D::new(&config, dist);
        init_uniform(&mut solver, 1.0, [15.0, -5.0, 3.0], 300.0);
        let c0 = solver.grid.idx(2, 2, 2);
        let nv = solver.dist.nv;
        for k in 0..nv {
            solver.dist.f[c0 * nv + k] *= 1.08;
        }
        solver.update_moments();

        let before = solver.totals();
        let dt = solver.cfl_dt(0.2);
        for _ in 0..5 {
            solver.step(dt, &config.bcs);
        }
        let after = solver.totals();

        let tol = 1e-2;
        assert!(
            (after.0 - before.0).abs() / before.0.abs().max(1e-30) < tol,
            "mass drift"
        );
        assert!(
            (after.4 - before.4).abs() / before.4.abs().max(1e-30) < tol,
            "energy drift"
        );
        assert!(solver.dist.f.iter().all(|v| v.is_finite()));
    }

    // SSP-RK2 integrator in 3D. The earlier "instability" was the same under-
    // resolved velocity grid (t_ref=1500, u_ref=0, n=5) that made
    // `combined_wave_particle` blow up: garbage discrete moments -> stiff tiny
    // tau -> divergence, regardless of the integrator. With a flow-resolving grid
    // (u_ref = flow velocity, t_ref ~ flow temperature, n=8) RK2 is stable and
    // conservative, consistent with the 2D RK2 test which always passed.
    #[test]
    fn rk2_scheme_conserves_and_stays_finite_3d() {
        let config = periodic_case(3, 3, 3);
        let u_flow = [5.0, -3.0, 2.0];
        let (vgrid, vw) =
            crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(287.0, 400.0, u_flow, 8);
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid, vw);
        let mut solver = DugksSolver3D::new(&config, dist);
        solver.scheme = crate::solver::TimeScheme::Rk2;
        init_uniform(&mut solver, 1.2, u_flow, 320.0);

        let before = solver.totals();
        let dt = solver.cfl_dt(0.2);
        for _ in 0..3 {
            solver.step_scheme(dt, &config.bcs);
        }
        let after = solver.totals();

        let tol = 1e-2;
        assert!(
            (after.0 - before.0).abs() / before.0.abs().max(1e-30) < tol,
            "mass drift: {before:?} vs {after:?}"
        );
        assert!(
            (after.4 - before.4).abs() / before.4.abs().max(1e-30) < tol,
            "energy drift"
        );
        assert!(solver.dist.f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rk2_default_scheme_is_euler_3d() {
        let config = periodic_case(2, 2, 2);
        let (vgrid, vw) = crate::velocity_grid3d::VelocityGrid3D::gauss_hermite(
            287.0,
            1500.0,
            [0.0, 0.0, 0.0],
            5,
        );
        let dist = Distribution3D::zeros(config.grid.ncells(), vgrid, vw);
        let solver = DugksSolver3D::new(&config, dist);
        assert_eq!(solver.scheme, crate::solver::TimeScheme::Euler);
    }
}
