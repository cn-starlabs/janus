//! janus-cli: headless batch runner for the janus-kinetic UGKWP / DUGKS solver.
//!
//! Loads a case.json or runs a default Couette-like case, runs the
//! time-stepper for N steps, and writes a `.jvtk` time series every M steps
//! containing both wave fields and particles (when UGKWP is selected).

use janus_core::config::{BoundaryAssignment, BoundaryKind, CaseConfig, GasProperties};
use janus_core::distribution::Distribution;
use janus_core::grid::Grid2D;
use janus_io::writer::{FieldData, NamedField};
use janus_io::JvtkWriter;
use janus_kinetic::maxwellian::{gh_equilibrium, DOF};
use janus_kinetic::velocity_grid::VelocityGrid2D;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize)]
struct ManifestFrame {
    index: u32,
    step: u64,
    time: f64,
    file: String,
}

#[derive(Serialize)]
struct JvtkManifest {
    case_name: String,
    write_every: u32,
    dims: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    frames: Vec<ManifestFrame>,
}

#[derive(Deserialize)]
struct CaseJson {
    config: CaseConfig,
    #[serde(default)]
    initial: InitialConditionJson,
    #[serde(default)]
    velocity_grid: VelocityGridJson,
    #[serde(default = "default_kernel")]
    kernel: String,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_kn_threshold")]
    kn_threshold: f64,
}

#[derive(Deserialize, Default)]
struct InitialConditionJson {
    #[serde(default = "default_rho")]
    rho: f64,
    #[serde(default = "default_temperature")]
    temperature: f64,
    #[serde(default)]
    velocity: [f64; 2],
}

#[derive(Deserialize)]
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

fn default_kernel() -> String {
    "ugkwp".to_string()
}
fn default_seed() -> u64 {
    12345
}
fn default_kn_threshold() -> f64 {
    janus_kinetic::coupled::DEFAULT_KN_THRESHOLD
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

fn main() {
    let mut config_path: Option<String> = None;
    let mut out_dir = "janus_output".to_string();
    let mut kernel = "ugkwp".to_string();
    let mut n_steps: u32 = 2000;
    let mut write_every: u32 = 200;
    let mut seed: u64 = 12345;
    let mut kn_threshold = janus_kinetic::coupled::DEFAULT_KN_THRESHOLD;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" if i + 1 < args.len() => {
                config_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_dir = args[i + 1].clone();
                i += 2;
            }
            "--kernel" if i + 1 < args.len() => {
                kernel = args[i + 1].to_lowercase();
                i += 2;
            }
            "--steps" if i + 1 < args.len() => {
                n_steps = args[i + 1].parse().expect("invalid --steps number");
                i += 2;
            }
            "--write-every" if i + 1 < args.len() => {
                write_every = args[i + 1].parse().expect("invalid --write-every number");
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                seed = args[i + 1].parse().expect("invalid --seed number");
                i += 2;
            }
            "--kn-threshold" if i + 1 < args.len() => {
                kn_threshold = args[i + 1].parse().expect("invalid --kn-threshold number");
                i += 2;
            }
            // Compatibility with legacy positional argument (out_dir):
            other if !other.starts_with('-') && i == 1 => {
                out_dir = other.to_string();
                i += 1;
            }
            other => {
                eprintln!("Unknown or malformed argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

    let (
        config,
        initial_rho,
        initial_temp,
        initial_u,
        v_max,
        nv,
        final_kernel,
        final_seed,
        final_kn_threshold,
    ) = if let Some(ref path) = config_path {
        let content = std::fs::read_to_string(path).expect("failed to read config file");
        let parsed: CaseJson = serde_json::from_str(&content).expect("failed to parse config JSON");

        let k = if args.iter().any(|a| a == "--kernel") {
            kernel
        } else {
            parsed.kernel
        };
        let s = if args.iter().any(|a| a == "--seed") {
            seed
        } else {
            parsed.seed
        };
        let kn = if args.iter().any(|a| a == "--kn-threshold") {
            kn_threshold
        } else {
            parsed.kn_threshold
        };

        (
            parsed.config,
            parsed.initial.rho,
            parsed.initial.temperature,
            parsed.initial.velocity,
            parsed.velocity_grid.v_max,
            parsed.velocity_grid.n_per_dim,
            k,
            s,
            kn,
        )
    } else {
        let nx = 20;
        let ny = 20;
        let gas = GasProperties::monatomic_default();
        let grid = Grid2D::new(nx, ny, 1.0e-3 / nx as f64, 1.0e-3 / ny as f64, [0.0, 0.0]);
        let bcs = BoundaryAssignment {
            west: BoundaryKind::Periodic,
            east: BoundaryKind::Periodic,
            south: BoundaryKind::DiffuseWall {
                temperature: 300.0,
                wall_velocity: [0.0, 0.0],
            },
            north: BoundaryKind::DiffuseWall {
                temperature: 300.0,
                wall_velocity: [50.0, 0.0],
            },
        };
        let config = CaseConfig { grid, bcs, gas };
        (
            config,
            1.0,
            300.0,
            [0.0, 0.0],
            1800.0,
            25,
            kernel,
            seed,
            kn_threshold,
        )
    };

    let (vgrid, vw) = VelocityGrid2D::simpson(v_max, nv);
    let mut dist = Distribution::zeros(config.grid.ncells(), vgrid, vw);

    let ncells = config.grid.ncells();
    for c in 0..ncells {
        for (k, v) in dist.vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(initial_rho, initial_u, initial_temp, config.gas.r_gas, *v);
            dist.f[c * dist.nv + k] = g;
            dist.h[c * dist.nv + k] = h;
        }
    }

    let mut solver = janus_kinetic::coupled::UgkwpSolver::new(&config, dist, final_seed);
    solver.kn_threshold = final_kn_threshold;
    match final_kernel.as_str() {
        "dugks" => solver.kernel = janus_kinetic::coupled::FluxKernel::Dugks,
        "ugkwp" => solver.kernel = janus_kinetic::coupled::FluxKernel::Ugkwp,
        other => {
            eprintln!("Unknown kernel '{}' (supported: 'ugkwp', 'dugks')", other);
            std::process::exit(1);
        }
    }
    solver.wave.update_moments();

    let dt = solver.wave.cfl_dt(0.4);
    let case_name = "case";

    println!("janus-cli: running {n_steps} steps, dt={dt:.6e}, writing every {write_every} steps to {out_dir}");

    let mut manifest_frames = Vec::new();
    let mut frame_index: u32 = 0;
    let path = write_frame(&out_dir, case_name, frame_index, &solver, 0.0, 0);
    manifest_frames.push(ManifestFrame {
        index: frame_index,
        step: 0,
        time: 0.0,
        file: path.file_name().unwrap().to_string_lossy().into_owned(),
    });

    for step in 1..=n_steps {
        solver.step(dt, &config.bcs);
        if step % write_every == 0 {
            frame_index += 1;
            let time = dt * step as f64;
            let path = write_frame(&out_dir, case_name, frame_index, &solver, time, step as u64);
            manifest_frames.push(ManifestFrame {
                index: frame_index,
                step: step as u64,
                time,
                file: path.file_name().unwrap().to_string_lossy().into_owned(),
            });
        }
    }

    let manifest = JvtkManifest {
        case_name: case_name.into(),
        write_every,
        dims: [solver.wave.grid.nx, solver.wave.grid.ny, 1],
        spacing: [solver.wave.grid.dx, solver.wave.grid.dy, 1.0],
        origin: [solver.wave.grid.origin[0], solver.wave.grid.origin[1], 0.0],
        frames: manifest_frames,
    };
    let manifest_path = Path::new(&out_dir).join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest).expect("manifest serializes");
    std::fs::write(&manifest_path, manifest_json).expect("failed to write manifest.json");

    println!(
        "janus-cli: done, wrote {} frames + manifest.json",
        frame_index + 1
    );
}

fn kn_range(solver: &janus_kinetic::coupled::UgkwpSolver) -> [f64; 2] {
    let mut min_kn = f64::INFINITY;
    let mut max_kn = 0.0f64;
    for &k in &solver.wave.fields.kn_loc {
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

fn write_frame(
    out_dir: &str,
    case_name: &str,
    frame_index: u32,
    solver: &janus_kinetic::coupled::UgkwpSolver,
    time: f64,
    step: u64,
) -> std::path::PathBuf {
    let rho = solver.wave.fields.rho.clone();
    let mom_x = solver.wave.fields.mom[0].clone();
    let mom_y = solver.wave.fields.mom[1].clone();
    let energy = solver.wave.fields.energy.clone();
    let kn_loc = solver.wave.fields.kn_loc.clone();
    let temperature: Vec<f64> = (0..solver.wave.grid.ncells())
        .map(|c| solver.wave.fields.temperature(c, solver.wave.gas_r, DOF))
        .collect();

    let p_free = solver.p_free().to_vec();
    let mut particle_density = vec![0.0; solver.wave.grid.ncells()];
    solver.particle_count_density(&mut particle_density);

    let fields = vec![
        NamedField {
            name: "rho".into(),
            comps: 1,
            data: FieldData::F64(&rho),
        },
        NamedField {
            name: "mom_x".into(),
            comps: 1,
            data: FieldData::F64(&mom_x),
        },
        NamedField {
            name: "mom_y".into(),
            comps: 1,
            data: FieldData::F64(&mom_y),
        },
        NamedField {
            name: "energy".into(),
            comps: 1,
            data: FieldData::F64(&energy),
        },
        NamedField {
            name: "kn_loc".into(),
            comps: 1,
            data: FieldData::F64(&kn_loc),
        },
        NamedField {
            name: "temperature".into(),
            comps: 1,
            data: FieldData::F64(&temperature),
        },
        NamedField {
            name: "p_free".into(),
            comps: 1,
            data: FieldData::F64(&p_free),
        },
        NamedField {
            name: "particle_count_density".into(),
            comps: 1,
            data: FieldData::F64(&particle_density),
        },
    ];

    let particle_bytes = if solver.particles.len() > 0 {
        let n = solver.particles.len();
        let mut bytes = Vec::with_capacity(n * 40);
        for i in 0..n {
            let pos = solver.particles.pos[i];
            let vel = solver.particles.vel[i];
            let w = solver.particles.weight[i];
            bytes.extend_from_slice(bytemuck::bytes_of(&pos));
            bytes.extend_from_slice(bytemuck::bytes_of(&vel));
            bytes.extend_from_slice(bytemuck::bytes_of(&w));
        }
        Some(bytes)
    } else {
        None
    };

    let pb = particle_bytes
        .as_ref()
        .map(|b| janus_io::writer::ParticleBlock {
            count: solver.particles.len() as u64,
            stride: 40,
            layout: vec!["pos2".into(), "vel2".into(), "weight".into()],
            bytes: b,
        });

    let g = solver.wave.grid;

    JvtkWriter::write_series_step(
        out_dir,
        case_name,
        frame_index,
        [g.nx, g.ny, 1],
        [g.dx, g.dy, 1.0],
        [g.origin[0], g.origin[1], 0.0],
        time,
        step,
        kn_range(solver),
        &fields,
        &[],
        pb,
    )
    .expect("failed to write jvtk frame")
}
