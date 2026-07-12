//! Macroscopic field storage, Structure-of-Arrays, one entry per cell —
//! 3D-space generalization of `fields::MacroFields` (3-component momentum
//! and heat flux, matching a `Grid3D`'s cell count/indexing).
//!
//! DESIGN: kept as a separate type from `MacroFields` (rather than changing
//! `MacroFields.mom`/`heat` from `[Vec<f64>; 2]` to `[Vec<f64>; 3]` in
//! place) for the same reason `Grid3D` is separate from `Grid2D`: existing
//! 2D solver code (`janus-kinetic::solver`, `coupled`, `bc`) indexes
//! `fields.mom[0]`/`fields.mom[1]` and constructs `MacroFields` with 2D
//! moment math throughout; changing the array arity in place would silently
//! break that code's semantics (not a compile error necessarily, since
//! `[Vec<f64>; 2]` -> `[Vec<f64>; 3]` does not affect `mom[0]`/`mom[1]`
//! access, but every energy/temperature moment computation that assumes 2
//! translational in-plane components + the reduced `h`-carried third
//! component would become physically wrong without every call site being
//! re-derived and re-verified by hand). `MacroFields3D` is the type new
//! 3D-space-aware code should use; `MacroFields` (2D) is untouched.

/// Cell-centered macroscopic (moment) fields over a `Grid3D`, SoA layout,
/// full 3-component momentum/heat-flux (unlike `MacroFields`'s 2-component
/// + reduced-`h` scheme, since a genuine 3D *physical-space* solver has no
/// need to reduce out a velocity-space direction — all 3 velocity
/// components are physically resolved directions here, not a DVM reduction
/// artifact). `energy` remains a scalar (total energy per volume).
#[derive(Clone, Debug)]
pub struct MacroFields3D {
    pub rho: Vec<f64>,
    pub mom: [Vec<f64>; 3],
    pub energy: Vec<f64>,
    /// Optional higher moments (viscous stress tensor components: xx, yy,
    /// zz, xy, xz, yz — 6 independent components for a symmetric 3x3
    /// tensor) for the transition regime / diagnostics, mirroring
    /// `MacroFields::stress`'s 3-component (2D-symmetric) analog.
    pub stress: Vec<[f64; 6]>,
    pub heat: [Vec<f64>; 3],
    /// Local Knudsen number, drives the wave/particle split (same role as
    /// `MacroFields::kn_loc`).
    pub kn_loc: Vec<f64>,
}

impl MacroFields3D {
    pub fn zeros(ncells: usize) -> Self {
        Self {
            rho: vec![0.0; ncells],
            mom: [vec![0.0; ncells], vec![0.0; ncells], vec![0.0; ncells]],
            energy: vec![0.0; ncells],
            stress: vec![[0.0; 6]; ncells],
            heat: [vec![0.0; ncells], vec![0.0; ncells], vec![0.0; ncells]],
            kn_loc: vec![0.0; ncells],
        }
    }

    pub fn ncells(&self) -> usize {
        self.rho.len()
    }

    /// Bulk velocity `u = mom / rho` for cell `c`.
    #[inline]
    pub fn velocity(&self, c: usize) -> [f64; 3] {
        let rho = self.rho[c].max(f64::MIN_POSITIVE);
        [
            self.mom[0][c] / rho,
            self.mom[1][c] / rho,
            self.mom[2][c] / rho,
        ]
    }

    /// Temperature from ideal-gas EOS: `E = rho*(|u|^2/2) + rho*dof/2*R*T`.
    /// For a genuine 3D physical-space monatomic gas (all 3 translational
    /// DOF physically resolved, no reduced-`h` carrier needed), `dof = 3`
    /// is the physically-correct value a caller should pass (mirroring how
    /// `janus_kinetic::maxwellian::DOF == 3` for the 2D-space reduced (g,h)
    /// formulation) — kept as a parameter here (not hardcoded) so this type
    /// stays agnostic to which kinetic formulation (full 3D DVM vs. some
    /// other reduction) produced the moments, exactly like `MacroFields`
    /// does for the 2D case.
    #[inline]
    pub fn temperature(&self, c: usize, r_gas: f64, dof: f64) -> f64 {
        let rho = self.rho[c].max(f64::MIN_POSITIVE);
        let u = self.velocity(c);
        let kinetic = 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let internal = self.energy[c] - kinetic;
        (2.0 * internal) / (rho * dof * r_gas)
    }

    /// Pressure from ideal gas law `p = rho * r_gas * T`.
    #[inline]
    pub fn pressure(&self, c: usize, r_gas: f64, dof: f64) -> f64 {
        self.rho[c] * r_gas * self.temperature(c, r_gas, dof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_consistent_lengths() {
        let f = MacroFields3D::zeros(10);
        assert_eq!(f.rho.len(), 10);
        assert_eq!(f.mom[0].len(), 10);
        assert_eq!(f.mom[1].len(), 10);
        assert_eq!(f.mom[2].len(), 10);
        assert_eq!(f.energy.len(), 10);
        assert_eq!(f.stress.len(), 10);
        assert_eq!(f.heat[0].len(), 10);
        assert_eq!(f.heat[2].len(), 10);
        assert_eq!(f.kn_loc.len(), 10);
    }

    #[test]
    fn velocity_and_temperature_3d() {
        let mut f = MacroFields3D::zeros(1);
        f.rho[0] = 2.0;
        f.mom[0][0] = 4.0; // u_x = 2
        f.mom[1][0] = 0.0;
        f.mom[2][0] = 6.0; // u_z = 3
        let r_gas = 287.0;
        let dof = 3.0;
        let u = f.velocity(0);
        assert!((u[0] - 2.0).abs() < 1e-12);
        assert!((u[2] - 3.0).abs() < 1e-12);
        let kinetic = 0.5 * f.rho[0] * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        let target_t = 300.0;
        let internal = 0.5 * f.rho[0] * dof * r_gas * target_t;
        f.energy[0] = kinetic + internal;
        let t = f.temperature(0, r_gas, dof);
        assert!((t - target_t).abs() < 1e-9);
    }
}
