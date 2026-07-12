//! janus-cli: headless batch runner for the janus-kinetic DUGKS solver.
//!
//! Builds a hardcoded case (grid + BCs + initial condition), runs the
//! time-stepper for N steps, and writes a `.jvtk` time series every M steps.

use janus_core::config::{BoundaryAssignment, BoundaryKind, CaseConfig, GasProperties};
use janus_core::distribution::Distribution;
use janus_core::grid::Grid2D;
use janus_io::writer::{FieldData, NamedField};
use janus_io::JvtkWriter;
use janus_kinetic::kn::update_kn_loc;
use janus_kinetic::maxwellian::{gh_equilibrium, DOF};
use janus_kinetic::solver::DugksSolver;
use janus_kinetic::velocity_grid::VelocityGrid2D;
use serde::Serialize;
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

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "janus_output".to_string());
    std::fs::create_dir_all(&out_dir).expect("failed to create output directory");

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

    let (vgrid, vw) = VelocityGrid2D::simpson(1800.0, 25);
    let mut dist = Distribution::zeros(config.grid.ncells(), vgrid.clone(), vw.clone());

    let rho0 = 1.0;
    let t0 = 300.0;
    let ncells = config.grid.ncells();
    for c in 0..ncells {
        for (k, v) in vgrid.iter().enumerate() {
            let (g, h) = gh_equilibrium(rho0, [0.0, 0.0], t0, config.gas.r_gas, *v);
            dist.f[c * dist.nv + k] = g;
            dist.h[c * dist.nv + k] = h;
        }
    }

    let mut solver = DugksSolver::new(&config, dist);
    solver.update_moments();

    let n_steps: u32 = 2000;
    let write_every: u32 = 200;
    let dt = solver.cfl_dt(0.4);
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
        update_solver_kn(&mut solver);
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
        dims: [solver.grid.nx, solver.grid.ny, 1],
        spacing: [solver.grid.dx, solver.grid.dy, 1.0],
        origin: [solver.grid.origin[0], solver.grid.origin[1], 0.0],
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

fn update_solver_kn(solver: &mut DugksSolver) {
    let ncells = solver.grid.ncells();
    let mut mu = vec![0.0; ncells];
    for c in 0..ncells {
        let t = solver.fields.temperature(c, solver.gas_r, DOF);
        mu[c] = janus_core::units::vhs_viscosity(t, solver.mu_ref, solver.t_ref, solver.omega);
    }
    update_kn_loc(&solver.grid, &mut solver.fields, &mu, solver.gas_r);
}

fn kn_range(solver: &DugksSolver) -> [f64; 2] {
    let mut min_kn = f64::INFINITY;
    let mut max_kn = 0.0f64;
    for &k in &solver.fields.kn_loc {
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
    solver: &DugksSolver,
    time: f64,
    step: u64,
) -> std::path::PathBuf {
    let rho = solver.fields.rho.clone();
    let mom_x = solver.fields.mom[0].clone();
    let mom_y = solver.fields.mom[1].clone();
    let energy = solver.fields.energy.clone();
    let kn_loc = solver.fields.kn_loc.clone();
    let temperature: Vec<f64> = (0..solver.grid.ncells())
        .map(|c| solver.fields.temperature(c, solver.gas_r, DOF))
        .collect();

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
    ];

    JvtkWriter::write_series_step(
        out_dir,
        case_name,
        frame_index,
        [solver.grid.nx, solver.grid.ny, 1],
        [solver.grid.dx, solver.grid.dy, 1.0],
        [solver.grid.origin[0], solver.grid.origin[1], 0.0],
        time,
        step,
        kn_range(solver),
        &fields,
        &[],
        None,
    )
    .expect("failed to write jvtk frame")
}
