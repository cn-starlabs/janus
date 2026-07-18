//! `.jvtk` reader: mmap the file, expose zero-copy `&[f64]`/`&[f32]` views
//! via `bytemuck::cast_slice`.

use crate::header::{FieldDesc, JvtkHeader, ParticleDesc};
use crate::MAGIC;
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

/// A memory-mapped `.jvtk` file with the parsed header and zero-copy field
/// accessors.
#[derive(Debug)]
pub struct JvtkReader {
    mmap: Mmap,
    header: JvtkHeader,
}

impl JvtkReader {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: memmap2::Mmap::map requires the backing file not be mutated
        // by another process concurrently in a way that violates Rust's
        // aliasing rules for the returned immutable byte slice. We only ever
        // open `.jvtk` files as read-only snapshots produced by JvtkWriter
        // (which is done writing and has closed its handle by the time a
        // reader opens the file in this crate's usage), so no writer holds a
        // live mutable mapping concurrently. This is the standard caveat
        // documented by memmap2 itself.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too short for jvtk header",
            ));
        }
        if mmap[0..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad jvtk magic bytes",
            ));
        }
        let header_len = u64::from_le_bytes(mmap[8..16].try_into().unwrap()) as usize;
        let header_start: usize = 16;
        let header_end = header_start
            .checked_add(header_len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "header_len overflow"))?;
        if mmap.len() < header_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file too short for declared header_len",
            ));
        }
        // The header JSON may be shorter than header_len (padding); find the
        // JSON's actual extent by trimming trailing NUL padding bytes.
        let raw = &mmap[header_start..header_end];
        let trimmed_len = raw
            .iter()
            .rposition(|&b| b != 0)
            .map(|p| p + 1)
            .unwrap_or(0);
        let header = JvtkHeader::from_json_bytes(&raw[..trimmed_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        validate_header_ranges(&header, mmap.len())?;

        Ok(Self { mmap, header })
    }

    pub fn header(&self) -> &JvtkHeader {
        &self.header
    }

    /// Zero-copy `f64` view of a cell field block by name.
    pub fn cell_field_f64(&self, name: &str) -> Option<&[f64]> {
        let d = self.header.cell_fields.iter().find(|d| d.name == name)?;
        if d.dtype != "f64" {
            return None;
        }
        let (start, end) = field_byte_range(d, 8)?;
        Some(bytemuck::cast_slice(&self.mmap[start..end]))
    }

    /// Zero-copy `f32` view of a cell field block by name.
    pub fn cell_field_f32(&self, name: &str) -> Option<&[f32]> {
        let d = self.header.cell_fields.iter().find(|d| d.name == name)?;
        if d.dtype != "f32" {
            return None;
        }
        let (start, end) = field_byte_range(d, 4)?;
        Some(bytemuck::cast_slice(&self.mmap[start..end]))
    }

    /// Zero-copy `f64` view of a point field block by name.
    pub fn point_field_f64(&self, name: &str) -> Option<&[f64]> {
        let d = self.header.point_fields.iter().find(|d| d.name == name)?;
        if d.dtype != "f64" {
            return None;
        }
        let (start, end) = field_byte_range(d, 8)?;
        Some(bytemuck::cast_slice(&self.mmap[start..end]))
    }

    /// Raw particle block bytes, if present.
    pub fn particle_bytes(&self) -> Option<&[u8]> {
        let pd = self.header.particles.as_ref()?;
        let (start, end) = particle_byte_range(pd)?;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }
}

fn dtype_elem_size(dtype: &str) -> Option<usize> {
    match dtype {
        "f64" => Some(8),
        "f32" => Some(4),
        _ => None,
    }
}

fn field_byte_range(d: &FieldDesc, elem_size: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(d.offset).ok()?;
    let n = usize::try_from(d.len).ok()?;
    let byte_len = n.checked_mul(elem_size)?;
    let end = start.checked_add(byte_len)?;
    Some((start, end))
}

fn particle_byte_range(pd: &ParticleDesc) -> Option<(usize, usize)> {
    let start = usize::try_from(pd.offset).ok()?;
    let count = usize::try_from(pd.count).ok()?;
    let stride = usize::try_from(pd.stride).ok()?;
    let byte_len = count.checked_mul(stride)?;
    let end = start.checked_add(byte_len)?;
    Some((start, end))
}

/// Reject crafted headers whose declared field/particle ranges fall outside
/// the mapped file (or overflow). Called once at open so later accessors can
/// slice without panicking on OOB.
fn validate_header_ranges(header: &JvtkHeader, file_len: usize) -> io::Result<()> {
    for d in header.cell_fields.iter().chain(header.point_fields.iter()) {
        let elem = dtype_elem_size(&d.dtype).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported field dtype '{}'", d.dtype),
            )
        })?;
        let (start, end) = field_byte_range(d, elem).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("field '{}' range overflow", d.name),
            )
        })?;
        if end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "field '{}' extends past end of file (end={end}, file_len={file_len})",
                    d.name
                ),
            ));
        }
        if start < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("field '{}' offset lands inside magic/header_len", d.name),
            ));
        }
    }
    if let Some(pd) = header.particles.as_ref() {
        let (start, end) = particle_byte_range(pd).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "particle block range overflow")
        })?;
        if end > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("particle block extends past end of file (end={end}, file_len={file_len})"),
            ));
        }
        if start < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "particle block offset lands inside magic/header_len",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{FieldData, NamedField, ParticleBlock};
    use crate::JvtkWriter;
    use std::io::Write;

    #[test]
    fn rejects_field_range_past_eof() {
        let dir = std::env::temp_dir().join(format!("janus_io_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.jvtk");

        let rho = vec![1.0_f64; 4];
        JvtkWriter::write_file(
            &path,
            [2, 2, 1],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            0.0,
            0,
            [0.0, 0.0],
            &[NamedField {
                name: "rho".into(),
                comps: 1,
                data: FieldData::F64(&rho),
            }],
            &[],
            None,
        )
        .unwrap();

        // Truncate after writing so the declared field range no longer fits.
        let meta = std::fs::metadata(&path).unwrap();
        let truncated = meta.len().saturating_sub(16);
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(truncated).unwrap();
        drop(f);

        let err = JvtkReader::open(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn particle_block_roundtrip_readable() {
        let dir = std::env::temp_dir().join(format!("janus_io_part_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("parts.jvtk");

        let rho = vec![1.0_f64; 4];
        // two particles: pos2 + vel2 + weight = 5 f64 each
        let mut bytes = Vec::new();
        for vals in [[0.1f64, 0.2, 1.0, 0.0, 0.5], [0.3, 0.4, -1.0, 0.1, 0.25]] {
            for v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let particles = ParticleBlock {
            count: 2,
            stride: 40,
            layout: vec!["pos2".into(), "vel2".into(), "weight".into()],
            bytes: &bytes,
        };

        JvtkWriter::write_file(
            &path,
            [2, 2, 1],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            0.0,
            0,
            [0.0, 0.0],
            &[NamedField {
                name: "rho".into(),
                comps: 1,
                data: FieldData::F64(&rho),
            }],
            &[],
            Some(particles),
        )
        .unwrap();

        let reader = JvtkReader::open(&path).unwrap();
        let pb = reader.particle_bytes().unwrap();
        assert_eq!(pb.len(), 80);
        assert_eq!(reader.header().particles.as_ref().unwrap().count, 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[allow(unused_must_use)]
    fn silence_unused_write_import() {
        // keep Write in scope for potential future tests
        let mut v = Vec::new();
        v.write_all(b"x");
    }
}
