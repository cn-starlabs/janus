//! C-ABI exports for Blender / Python (`ctypes`) integration.
//!
//! All functions are single-threaded: one handle must not be used concurrently
//! from multiple threads. Blender's addon runs the solver on a worker thread and
//! only reads field pointers on the main thread between steps.

use janus_core::config::{BoundaryKind, CaseConfig, Edge};
use janus_core::distribution::Distribution;
use janus_io::writer::{FieldData, NamedField, ParticleBlock};
use janus_io::JvtkWriter;
use janus_kinetic::coupled::{FluxKernel, UgkwpSolver};
use janus_kinetic::kn::update_kn_loc;
use janus_kinetic::maxwellian::{gh_equilibrium, DOF};
use janus_kinetic::solver::TimeScheme;
use janus_kinetic::velocity_grid::VelocityGrid2D;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::os::raw::c_int;
use std::ptr;

/// ABI version — bump when breaking the C layout or semantics.
pub const JANUS_FFI_API_VERSION: u32 = 1;

#[repr(C)]
pub struct JanusFieldView {
    pub data: *const f64,
    pub len: usize,
}

#[repr(C)]
pub struct JanusGridInfo {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub origin_x: f64,
    pub origin_y: f64,
    pub ncells: usize,
}

/// Opaque handle used by the C ABI. The actual solver state remains private
/// to Rust, but the type must be public so public FFI entry points do not
/// trigger Rust's private-interface warnings.
pub struct SolverHandle {
    pub solver: UgkwpSolver,
    pub config: CaseConfig,
    pub mu_scratch: Vec<f64>,
    pub particle_density_scratch: Vec<f64>,
    pub temperature_scratch: Vec<f64>,
    pub output_dir: Option<std::path::PathBuf>,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<String>) {
    let c =
        CString::new(msg.into()).unwrap_or_else(|_| CString::new("invalid error string").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

#[derive(Serialize, Deserialize)]
struct SolverCreateJson {
    config: CaseConfig,
    #[serde(default)]
    initial: InitialConditionJson,
    #[serde(default)]
    velocity_grid: VelocityGridJson,
    #[serde(default)]
    scene: Option<SceneJson>,
    #[serde(default)]
    time_scheme: TimeSchemeJson,
    #[serde(default = "default_kernel")]
    kernel: String,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_kn_threshold")]
    kn_threshold: f64,
    #[serde(default)]
    output_dir: Option<String>,
}

fn default_kernel() -> String {
    "ugkwp".to_string()
}

fn default_seed() -> u64 {
    12345
}

fn default_kn_threshold() -> f64 {
    janus_kinetic::coupled::DEFAULT_KN_THRESHOLD
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct SceneJson {
    #[serde(default)]
    boundary_tags: Vec<BoundaryTagJson>,
}

/// Time integration scheme selector (passed from Blender configuration).
#[derive(Serialize, Deserialize, Default, Clone, Debug)]
struct TimeSchemeJson {
    /// One of: "euler" (default), "rk2", "rk4".
    #[serde(default)]
    scheme: String,
}

impl TimeSchemeJson {
    fn to_time_scheme(&self) -> TimeScheme {
        match self.scheme.to_lowercase().as_str() {
            "rk2" => TimeScheme::Rk2,
            "rk4" => TimeScheme::Rk4,
            _ => TimeScheme::Euler, // default
        }
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct BoundaryTagJson {
    #[serde(default)]
    role: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    velocity: Option<[f64; 2]>,
}

#[derive(Serialize, Deserialize, Default)]
struct InitialConditionJson {
    #[serde(default = "default_rho")]
    rho: f64,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default)]
    velocity: [f64; 2],
}

#[derive(Serialize, Deserialize)]
struct VelocityGridJson {
    #[serde(default = "default_vmax")]
    v_max: f64,
    #[serde(default = "default_nv")]
    n_per_dim: usize,
}

impl Default for VelocityGridJson {
    fn default() -> Self {
        Self {
            v_max: default_vmax(),
            n_per_dim: default_nv(),
        }
    }
}

fn default_rho() -> f64 {
    1.0
}
fn default_temperature() -> f64 {
    300.0
}
fn default_vmax() -> f64 {
    1800.0
}
fn default_nv() -> usize {
    25
}

fn build_solver(json: &SolverCreateJson) -> Result<SolverHandle, String> {
    let mut config = json.config.clone();
    apply_scene_boundary_tags(&mut config, json.scene.as_ref());

    // Security Hardening: parameter caps & input validation
    let ncells = config.grid.ncells();
    if ncells == 0 || ncells > 4096 * 4096 {
        return Err(format!("grid cell count exceeds safety limit: {ncells}"));
    }
    if json.velocity_grid.n_per_dim == 0 || json.velocity_grid.n_per_dim > 128 {
        return Err(format!(
            "velocity grid resolution exceeds safety limit: {}",
            json.velocity_grid.n_per_dim
        ));
    }
    let rho0 = json.initial.rho;
    if !rho0.is_finite() || rho0 <= 0.0 {
        return Err(format!(
            "initial density must be positive and finite: {rho0}"
        ));
    }
    let t0 = json.initial.temperature;
    if !t0.is_finite() || t0 <= 0.0 {
        return Err(format!(
            "initial temperature must be positive and finite: {t0}"
        ));
    }
    let u0 = json.initial.velocity;
    if !u0[0].is_finite() || !u0[1].is_finite() {
        return Err(format!("initial velocity must be finite: {:?}", u0));
    }

    let (vgrid, vw) =
        VelocityGrid2D::simpson(json.velocity_grid.v_max, json.velocity_grid.n_per_dim);
    let mut dist = Distribution::zeros(ncells, vgrid, vw);
    let r_gas = config.gas.r_gas;
    for c in 0..ncells {
        for (k, v) in dist.vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(rho0, u0, t0, r_gas, *v);
            dist.f[c * dist.nv + k] = g;
            dist.h[c * dist.nv + k] = h;
        }
    }
    let mut solver = UgkwpSolver::new(&config, dist, json.seed);
    solver.kn_threshold = json.kn_threshold;

    let kernel_str = json.kernel.trim().to_lowercase();
    if kernel_str == "dugks" {
        solver.kernel = FluxKernel::Dugks;
    } else if kernel_str == "ugkwp" || kernel_str.is_empty() {
        solver.kernel = FluxKernel::Ugkwp;
    } else {
        return Err(format!(
            "unknown flux kernel '{}' (supported: 'ugkwp', 'dugks')",
            json.kernel
        ));
    }

    solver.wave.update_moments();
    // Apply selected time scheme (Euler, RK2, RK4)
    solver.wave.scheme = json.time_scheme.to_time_scheme();
    let mu_scratch = vec![0.0; ncells];
    let particle_density_scratch = vec![0.0; ncells];
    let temperature_scratch = vec![0.0; ncells];

    // Path sandboxing setup
    let output_dir: Option<std::path::PathBuf> = if let Some(ref dir) = json.output_dir {
        let p = std::path::Path::new(dir);
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_default().join(p)
        };
        let _ = std::fs::create_dir_all(&abs);
        Some(abs.canonicalize().ok().unwrap_or(abs))
    } else {
        None
    };

    Ok(SolverHandle {
        solver,
        config,
        mu_scratch,
        particle_density_scratch,
        temperature_scratch,
        output_dir,
    })
}

fn apply_scene_boundary_tags(config: &mut CaseConfig, scene: Option<&SceneJson>) {
    let Some(scene) = scene else {
        return;
    };

    for tag in &scene.boundary_tags {
        let Some(edge) = (match tag.role.as_str() {
            "west" => Some(Edge::West),
            "east" => Some(Edge::East),
            "south" => Some(Edge::South),
            "north" => Some(Edge::North),
            _ => None,
        }) else {
            continue;
        };

        let kind = match tag.kind.trim().to_lowercase().as_str() {
            "diffusewall" | "wall" => BoundaryKind::DiffuseWall {
                temperature: tag.temperature.unwrap_or(300.0),
                wall_velocity: tag.velocity.unwrap_or([0.0, 0.0]),
                accommodation: 1.0,
            },
            "specularwall" => BoundaryKind::SpecularWall,
            "velocityinlet" => BoundaryKind::VelocityInlet {
                velocity: tag.velocity.unwrap_or([0.0, 0.0]),
                density: 1.0,
                temperature: tag.temperature.unwrap_or(300.0),
            },
            "pressureinlet" => BoundaryKind::PressureInlet {
                pressure: 1.0,
                temperature: tag.temperature.unwrap_or(300.0),
            },
            "outlet" => BoundaryKind::Outlet,
            "symmetry" => BoundaryKind::Symmetry,
            "periodic" => BoundaryKind::Periodic,
            _ => continue,
        };

        match edge {
            Edge::West => config.bcs.west = kind,
            Edge::East => config.bcs.east = kind,
            Edge::South => config.bcs.south = kind,
            Edge::North => config.bcs.north = kind,
        }
    }
}

fn update_kn_and_mu(handle: &mut SolverHandle) {
    let ncells = handle.solver.wave.grid.ncells();
    if handle.mu_scratch.len() != ncells {
        handle.mu_scratch.resize(ncells, 0.0);
    }
    let r_gas = handle.solver.wave.gas_r;
    for c in 0..ncells {
        let t = handle.solver.wave.fields.temperature(c, r_gas, DOF);
        handle.mu_scratch[c] = janus_core::units::vhs_viscosity(
            t,
            handle.solver.wave.mu_ref,
            handle.solver.wave.t_ref,
            handle.solver.wave.omega,
        );
    }
    update_kn_loc(
        &handle.solver.wave.grid,
        &mut handle.solver.wave.fields,
        &handle.mu_scratch,
        r_gas,
    );
}

fn kn_range(fields: &janus_core::fields::MacroFields) -> [f64; 2] {
    let mut min_kn = f64::INFINITY;
    let mut max_kn = 0.0f64;
    for &k in &fields.kn_loc {
        if k.is_finite() {
            min_kn = min_kn.min(k);
            max_kn = max_kn.max(k);
        }
    }
    if min_kn.is_infinite() {
        [0.0, 0.0]
    } else {
        [min_kn, max_kn]
    }
}

fn temperature_field(solver: &UgkwpSolver) -> Vec<f64> {
    let n = solver.wave.grid.ncells();
    let mut t = vec![0.0; n];
    for c in 0..n {
        t[c] = solver.wave.fields.temperature(c, solver.wave.gas_r, DOF);
    }
    t
}

/// Returns `JANUS_FFI_API_VERSION`.
#[no_mangle]
pub extern "C" fn janus_ffi_api_version() -> u32 {
    JANUS_FFI_API_VERSION
}

/// Thread-local last error message, or null if none.
#[no_mangle]
pub extern "C" fn janus_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

/// Create a solver from a UTF-8 JSON blob (see `SolverCreateJson`). Returns null on error.
#[no_mangle]
pub extern "C" fn janus_solver_create(config_json: *const c_char) -> *mut SolverHandle {
    clear_error();
    if config_json.is_null() {
        set_error("config_json is null");
        return ptr::null_mut();
    }
    let json_str = unsafe {
        match CStr::from_ptr(config_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_error(format!("invalid UTF-8 in config_json: {e}"));
                return ptr::null_mut();
            }
        }
    };
    let parsed: SolverCreateJson = match serde_json::from_str(json_str) {
        Ok(p) => p,
        Err(e) => {
            set_error(format!("failed to parse config JSON: {e}"));
            return ptr::null_mut();
        }
    };
    match build_solver(&parsed) {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            set_error(e);
            ptr::null_mut()
        }
    }
}

/// Destroy a solver handle created by `janus_solver_create`.
#[no_mangle]
pub extern "C" fn janus_solver_destroy(handle: *mut SolverHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// Advance one time step. Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn janus_solver_step(handle: *mut SolverHandle, dt: f64) -> c_int {
    clear_error();
    if handle.is_null() {
        set_error("handle is null");
        return -1;
    }
    if !(dt > 0.0) {
        set_error("dt must be positive");
        return -1;
    }
    let h = unsafe { &mut *handle };
    h.solver.step(dt, &h.config.bcs);
    update_kn_and_mu(h);
    0
}

/// CFL-limited timestep suggestion.
#[no_mangle]
pub extern "C" fn janus_solver_cfl_dt(handle: *const SolverHandle, cfl: f64) -> f64 {
    if handle.is_null() || !(cfl > 0.0) {
        return 0.0;
    }
    let h = unsafe { &*handle };
    h.solver.wave.cfl_dt(cfl)
}

/// Fill grid metadata. Returns 0 on success.
#[no_mangle]
pub extern "C" fn janus_solver_grid_info(
    handle: *const SolverHandle,
    out: *mut JanusGridInfo,
) -> c_int {
    clear_error();
    if handle.is_null() || out.is_null() {
        set_error("handle or out is null");
        return -1;
    }
    let h = unsafe { &*handle };
    let g = h.solver.wave.grid;
    unsafe {
        *out = JanusGridInfo {
            nx: g.nx,
            ny: g.ny,
            dx: g.dx,
            dy: g.dy,
            origin_x: g.origin[0],
            origin_y: g.origin[1],
            ncells: g.ncells(),
        };
    }
    0
}

/// Set time integrator: 0 = Euler, 1 = RK2, 2 = RK4. Returns 0 on success.
#[no_mangle]
pub extern "C" fn janus_solver_set_scheme(handle: *mut SolverHandle, scheme: c_int) -> c_int {
    clear_error();
    if handle.is_null() {
        set_error("handle is null");
        return -1;
    }
    let h = unsafe { &mut *handle };
    h.solver.wave.scheme = match scheme {
        0 => TimeScheme::Euler,
        1 => TimeScheme::Rk2,
        2 => TimeScheme::Rk4,
        _ => {
            set_error(format!(
                "unknown scheme id {scheme} (supported: 0=Euler, 1=RK2, 2=RK4)"
            ));
            return -1;
        }
    };
    0
}

/// Zero-copy view of a named cell field (`rho`, `mom_x`, `mom_y`, `energy`, `kn_loc`, `temperature`, `p_free`, `particle_count_density`).
/// Pointers are valid until the next `janus_solver_step` or `janus_solver_destroy`.
#[no_mangle]
pub extern "C" fn janus_solver_field_view(
    handle: *const SolverHandle,
    name: *const c_char,
    out: *mut JanusFieldView,
) -> c_int {
    clear_error();
    if handle.is_null() || name.is_null() || out.is_null() {
        set_error("handle, name, or out is null");
        return -1;
    }
    let h = unsafe { &mut *(handle as *mut SolverHandle) };
    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_error(format!("invalid field name UTF-8: {e}"));
                return -1;
            }
        }
    };
    let (ptr, len): (*const f64, usize) = match name_str {
        "rho" => {
            let s = &h.solver.wave.fields.rho;
            (s.as_ptr(), s.len())
        }
        "mom_x" => {
            let s = &h.solver.wave.fields.mom[0];
            (s.as_ptr(), s.len())
        }
        "mom_y" => {
            let s = &h.solver.wave.fields.mom[1];
            (s.as_ptr(), s.len())
        }
        "energy" => {
            let s = &h.solver.wave.fields.energy;
            (s.as_ptr(), s.len())
        }
        "kn_loc" => {
            let s = &h.solver.wave.fields.kn_loc;
            (s.as_ptr(), s.len())
        }
        "temperature" => {
            let ncells = h.solver.wave.grid.ncells();
            if h.temperature_scratch.len() != ncells {
                h.temperature_scratch.resize(ncells, 0.0);
            }
            h.temperature_scratch = temperature_field(&h.solver);
            (h.temperature_scratch.as_ptr(), ncells)
        }
        "p_free" => {
            let s = h.solver.p_free();
            (s.as_ptr(), s.len())
        }
        "particle_count_density" => {
            let ncells = h.solver.wave.grid.ncells();
            if h.particle_density_scratch.len() != ncells {
                h.particle_density_scratch.resize(ncells, 0.0);
            }
            h.solver
                .particle_count_density(&mut h.particle_density_scratch);
            (h.particle_density_scratch.as_ptr(), ncells)
        }
        _ => {
            set_error(format!("unknown field '{name_str}'"));
            return -1;
        }
    };
    unsafe {
        *out = JanusFieldView { data: ptr, len };
    }
    0
}

/// Write the current state to a `.jvtk` file. Returns 0 on success.
#[no_mangle]
pub extern "C" fn janus_solver_write_jvtk(
    handle: *const SolverHandle,
    path: *const c_char,
    time: f64,
    step: u64,
) -> c_int {
    clear_error();
    if handle.is_null() || path.is_null() {
        set_error("handle or path is null");
        return -1;
    }
    let h = unsafe { &mut *(handle as *mut SolverHandle) };
    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(e) => {
                set_error(format!("invalid path UTF-8: {e}"));
                return -1;
            }
        }
    };

    // Security Hardening: path sandboxing validation
    let target_path = std::path::Path::new(path_str);
    if target_path
        .components()
        .any(|c| c == std::path::Component::ParentDir)
    {
        set_error("path contains parent directory traversal (..)");
        return -1;
    }
    if let Some(ref allowed) = h.output_dir {
        let abs_target = if target_path.is_absolute() {
            target_path.to_path_buf()
        } else {
            allowed.join(target_path)
        };
        if !abs_target.starts_with(allowed) {
            set_error(format!(
                "path is outside the configured output directory: {path_str}"
            ));
            return -1;
        }
    }

    let temp = temperature_field(&h.solver);
    let rho = &h.solver.wave.fields.rho;
    let mom_x = &h.solver.wave.fields.mom[0];
    let mom_y = &h.solver.wave.fields.mom[1];
    let energy = &h.solver.wave.fields.energy;
    let kn_loc = &h.solver.wave.fields.kn_loc;
    let p_free = h.solver.p_free();

    let mut particle_density = vec![0.0; h.solver.wave.grid.ncells()];
    h.solver.particle_count_density(&mut particle_density);

    let fields = vec![
        NamedField {
            name: "rho".into(),
            comps: 1,
            data: FieldData::F64(rho),
        },
        NamedField {
            name: "mom_x".into(),
            comps: 1,
            data: FieldData::F64(mom_x),
        },
        NamedField {
            name: "mom_y".into(),
            comps: 1,
            data: FieldData::F64(mom_y),
        },
        NamedField {
            name: "energy".into(),
            comps: 1,
            data: FieldData::F64(energy),
        },
        NamedField {
            name: "kn_loc".into(),
            comps: 1,
            data: FieldData::F64(kn_loc),
        },
        NamedField {
            name: "temperature".into(),
            comps: 1,
            data: FieldData::F64(&temp),
        },
        NamedField {
            name: "p_free".into(),
            comps: 1,
            data: FieldData::F64(p_free),
        },
        NamedField {
            name: "particle_count_density".into(),
            comps: 1,
            data: FieldData::F64(&particle_density),
        },
    ];

    let particle_bytes = if h.solver.particles.len() > 0 {
        let n = h.solver.particles.len();
        let mut bytes = Vec::with_capacity(n * 40);
        for i in 0..n {
            let pos = h.solver.particles.pos[i];
            let vel = h.solver.particles.vel[i];
            let w = h.solver.particles.weight[i];
            bytes.extend_from_slice(bytemuck::bytes_of(&pos));
            bytes.extend_from_slice(bytemuck::bytes_of(&vel));
            bytes.extend_from_slice(bytemuck::bytes_of(&w));
        }
        Some(bytes)
    } else {
        None
    };

    let pb = particle_bytes.as_ref().map(|b| ParticleBlock {
        count: h.solver.particles.len() as u64,
        stride: 40,
        layout: vec!["pos2".into(), "vel2".into(), "weight".into()],
        bytes: b,
    });

    let g = h.solver.wave.grid;
    let kn_rng = kn_range(&h.solver.wave.fields);
    if let Err(e) = JvtkWriter::write_file(
        path_str,
        [g.nx, g.ny, 1],
        [g.dx, g.dy, 1.0],
        [g.origin[0], g.origin[1], 0.0],
        time,
        step,
        kn_rng,
        &fields,
        &[],
        pb,
    ) {
        set_error(format!("jvtk write failed: {e}"));
        return -1;
    }
    0
}

/// Returns a JSON string describing the default Couette-like demo case. Caller must not free.
#[no_mangle]
pub extern "C" fn janus_default_case_json() -> *const c_char {
    thread_local! {
        static DEFAULT_JSON: RefCell<Option<CString>> = const { RefCell::new(None) };
    }
    DEFAULT_JSON.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            use janus_core::config::{BoundaryAssignment, BoundaryKind, GasProperties};
            use janus_core::grid::Grid2D;
            let nx = 20usize;
            let ny = 20usize;
            let gas = GasProperties::monatomic_default();
            let grid = Grid2D::new(nx, ny, 1.0e-3 / nx as f64, 1.0e-3 / ny as f64, [0.0, 0.0]);
            let bcs = BoundaryAssignment {
                west: BoundaryKind::Periodic,
                east: BoundaryKind::Periodic,
                south: BoundaryKind::DiffuseWall {
                    temperature: 300.0,
                    wall_velocity: [0.0, 0.0],
                    accommodation: 1.0,
                },
                north: BoundaryKind::DiffuseWall {
                    temperature: 300.0,
                    wall_velocity: [50.0, 0.0],
                    accommodation: 1.0,
                },
            };
            let cfg = SolverCreateJson {
                config: CaseConfig { grid, bcs, gas },
                initial: InitialConditionJson::default(),
                velocity_grid: VelocityGridJson::default(),
                scene: None,
                time_scheme: TimeSchemeJson::default(),
                kernel: default_kernel(),
                seed: default_seed(),
                kn_threshold: default_kn_threshold(),
                output_dir: None,
            };
            let json = serde_json::to_string_pretty(&cfg).expect("default case serializes");
            *borrow = Some(CString::new(json).expect("no interior nul in json"));
        }
        borrow.as_ref().unwrap().as_ptr()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_boundary_tags_override_edge_config() {
        let payload = r#"{
            "config": {
                "grid": {"nx": 2, "ny": 2, "dx": 0.5, "dy": 0.5, "origin": [0.0, 0.0]},
                "bcs": {
                    "west": "Periodic",
                    "east": "Periodic",
                    "south": "Periodic",
                    "north": "Periodic"
                },
                "gas": {
                    "r_gas": 208.13,
                    "molar_mass": 0.039948,
                    "vhs_omega": 0.81,
                    "mu_ref": 2.117e-5,
                    "t_ref": 273.15,
                    "prandtl": 0.6666666666666666
                }
            },
            "scene": {
                "boundary_tags": [
                    {
                        "role": "north",
                        "kind": "DiffuseWall",
                        "temperature": 400.0,
                        "velocity": [2.0, 0.0]
                    }
                ]
            }
        }"#;

        let parsed: SolverCreateJson = serde_json::from_str(payload).unwrap();
        let mut config = parsed.config.clone();
        apply_scene_boundary_tags(&mut config, parsed.scene.as_ref());

        match config.bcs.north {
            janus_core::config::BoundaryKind::DiffuseWall {
                temperature,
                wall_velocity,
            } => {
                assert!((temperature - 400.0).abs() < 1e-12);
                assert_eq!(wall_velocity, [2.0, 0.0]);
            }
            other => panic!("expected diffuse wall, got {other:?}"),
        }
    }
}
