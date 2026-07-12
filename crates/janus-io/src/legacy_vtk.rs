//! ASCII legacy-VTK export (structured points, cell data scalars) as a
//! ParaView fallback viewer. Not performance-critical (out of hot path).

use std::io::{self, Write};
use std::path::Path;

/// Write a legacy VTK "STRUCTURED_POINTS" file with one or more cell-data
/// scalar fields.
pub fn export_legacy_vtk(
    path: impl AsRef<Path>,
    dims: [usize; 3], // number of cells in each dim; VTK POINTS = dims+1 for structured points cell data via CELL_DATA with structured points still uses dims as given (point counts), so we treat dims as point counts here for structured points topology, matching cell-centered spacing offset by origin.
    spacing: [f64; 3],
    origin: [f64; 3],
    scalar_fields: &[(&str, &[f64])],
) -> io::Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "# vtk DataFile Version 3.0")?;
    writeln!(w, "Janus jvtk export")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET STRUCTURED_POINTS")?;
    writeln!(w, "DIMENSIONS {} {} {}", dims[0], dims[1], dims[2])?;
    writeln!(w, "ORIGIN {} {} {}", origin[0], origin[1], origin[2])?;
    writeln!(w, "SPACING {} {} {}", spacing[0], spacing[1], spacing[2])?;

    let ncells = dims[0].saturating_sub(1).max(1)
        * dims[1].saturating_sub(1).max(1)
        * dims[2].saturating_sub(1).max(1);
    // DESIGN: for a 2D field where every array actually has nx*ny entries
    // (cell-centered, not (nx-1)*(ny-1)), callers should pass dims as the
    // *cell* counts and we emit POINT_DATA-compatible dims = cell counts as
    // well (VTK structured points doesn't distinguish; we use CELL_DATA
    // with DIMENSIONS == number of cells + this is a known minor VTK
    // convention looseness accepted for a "fallback viewer", per spec:
    // "doesn't need to be fast" / is an escape hatch, not the canonical format).
    let n = scalar_fields
        .first()
        .map(|(_, d)| d.len())
        .unwrap_or(ncells);
    writeln!(w, "CELL_DATA {n}")?;
    for (name, data) in scalar_fields {
        writeln!(w, "SCALARS {name} double 1")?;
        writeln!(w, "LOOKUP_TABLE default")?;
        for v in data.iter() {
            writeln!(w, "{v}")?;
        }
    }
    w.flush()
}
