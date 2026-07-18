//! Round-trip test: write a small MacroFields-like snapshot to `.jvtk`, read
//! it back via mmap, assert bit-exact equality.

use janus_io::writer::{FieldData, NamedField};
use janus_io::{JvtkReader, JvtkWriter};

#[test]
fn roundtrip_bit_exact() {
    let dir = std::env::temp_dir().join(format!("janus_io_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip.jvtk");

    let nx = 4usize;
    let ny = 3usize;
    let ncells = nx * ny;
    let rho: Vec<f64> = (0..ncells).map(|i| 1.0 + i as f64 * 0.1).collect();
    let mom_x: Vec<f64> = (0..ncells).map(|i| (i as f64).sin()).collect();
    let energy: Vec<f32> = (0..ncells).map(|i| 300.0 + i as f32).collect();

    let cell_fields = vec![
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
            name: "energy".into(),
            comps: 1,
            data: FieldData::F32(&energy),
        },
    ];

    JvtkWriter::write_file(
        &path,
        [nx, ny, 1],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        1.234,
        42,
        [0.001, 1.0],
        &cell_fields,
        &[],
        None,
    )
    .unwrap();

    let reader = JvtkReader::open(&path).unwrap();
    let h = reader.header();
    assert_eq!(h.dims, [nx, ny, 1]);
    assert_eq!(h.step, 42);
    assert!((h.time - 1.234).abs() < 1e-15);
    assert_eq!(h.kn_range, [0.001, 1.0]);

    let rho_back = reader.cell_field_f64("rho").unwrap();
    assert_eq!(rho_back, rho.as_slice());

    let mom_back = reader.cell_field_f64("mom_x").unwrap();
    assert_eq!(mom_back, mom_x.as_slice());

    let energy_back = reader.cell_field_f32("energy").unwrap();
    assert_eq!(energy_back, energy.as_slice());

    // All block offsets must be 64-byte aligned.
    for d in &h.cell_fields {
        assert_eq!(d.offset % 64, 0, "field {} not 64-byte aligned", d.name);
    }

    std::fs::remove_dir_all(&dir).ok();
}

/// M4 3D extension: a `.jvtk` file with `nz > 1` round-trips exactly like
/// the `nz == 1` case above — the format (header `dims`/`spacing`/`origin`
/// are already `[usize; 3]`/`[f64; 3]` triples, and the writer/reader never
/// assume anything about `dims[2]`) needed NO changes for this; this test
/// exists to make that fact explicit and regression-proof. Field data is
/// generated in `Grid3D`'s C-order linear indexing (`k*nx*ny + j*nx + i`)
/// so this also exercises that `Grid3D::idx` produces exactly the layout
/// `.jvtk` expects (ENGINEERING_SPEC.md §4: "Field arrays are plain
/// [f64]/[f32] in C order").
#[test]
fn roundtrip_3d_nz_greater_than_one() {
    use janus_core::grid3d::Grid3D;

    let dir = std::env::temp_dir().join(format!("janus_io_test_3d_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("roundtrip3d.jvtk");

    let grid = Grid3D::new(3, 2, 4, 0.1, 0.2, 0.3, [0.0, 0.0, 0.0]);
    let ncells = grid.ncells();
    assert_eq!(ncells, 24);

    // Fill rho with a value that encodes (i,j,k) so we can verify C-order
    // placement survives the round trip bit-exactly.
    let mut rho = vec![0.0f64; ncells];
    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let c = grid.idx(i, j, k);
                rho[c] = (i as f64) + 100.0 * (j as f64) + 10_000.0 * (k as f64);
            }
        }
    }

    let cell_fields = vec![NamedField {
        name: "rho".into(),
        comps: 1,
        data: FieldData::F64(&rho),
    }];

    JvtkWriter::write_file(
        &path,
        grid.dims(),
        grid.spacing(),
        grid.origin,
        0.5,
        3,
        [0.0, 10.0],
        &cell_fields,
        &[],
        None,
    )
    .unwrap();

    let reader = JvtkReader::open(&path).unwrap();
    let h = reader.header();
    assert_eq!(h.dims, [3, 2, 4]);
    assert_eq!(h.dims[2], 4, "nz must round-trip as dims[2]");

    let rho_back = reader.cell_field_f64("rho").unwrap();
    assert_eq!(
        rho_back,
        rho.as_slice(),
        "3D C-order field data must round-trip bit-exactly"
    );

    // Spot-check a few specific (i,j,k) decode correctly through Grid3D.
    for &(i, j, k) in &[(0usize, 0usize, 0usize), (2, 1, 3), (1, 0, 2)] {
        let c = grid.idx(i, j, k);
        let expected = (i as f64) + 100.0 * (j as f64) + 10_000.0 * (k as f64);
        assert_eq!(rho_back[c], expected);
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn series_naming() {
    let dir = std::env::temp_dir().join(format!("janus_io_series_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let rho = vec![1.0f64; 4];
    let fields = vec![NamedField {
        name: "rho".into(),
        comps: 1,
        data: FieldData::F64(&rho),
    }];
    let path = JvtkWriter::write_series_step(
        &dir,
        "case",
        7,
        [2, 2, 1],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        0.0,
        7,
        [0.0, 0.0],
        &fields,
        &[],
        None,
    )
    .unwrap();
    assert_eq!(
        path.file_name().unwrap().to_str().unwrap(),
        "case.0007.jvtk"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn roundtrip_particles() {
    use janus_io::writer::ParticleBlock;

    let dir = std::env::temp_dir().join(format!("janus_io_particles_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("particles.jvtk");

    let p_bytes = vec![1.23f64, 4.56, 7.89, -1.2, 0.5, 9.87, 6.54, 3.21, 0.0, 1.5]; // two particles
    let p_bytes_u8 = bytemuck::cast_slice::<f64, u8>(&p_bytes);

    let pb = ParticleBlock {
        count: 2,
        stride: 40,
        layout: vec!["pos2".into(), "vel2".into(), "weight".into()],
        bytes: p_bytes_u8,
    };

    JvtkWriter::write_file(
        &path,
        [2, 2, 1],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        0.0,
        1,
        [0.0, 0.0],
        &[],
        &[],
        Some(pb),
    )
    .unwrap();

    let reader = JvtkReader::open(&path).unwrap();
    let h = reader.header();
    let pd = h.particles.as_ref().unwrap();
    assert_eq!(pd.count, 2);
    assert_eq!(pd.stride, 40);
    assert_eq!(pd.layout, vec!["pos2", "vel2", "weight"]);

    let read_u8 = reader.particle_bytes().unwrap();
    let read_f64 = bytemuck::cast_slice::<u8, f64>(read_u8);
    assert_eq!(read_f64, p_bytes.as_slice());

    std::fs::remove_dir_all(&dir).ok();
}
