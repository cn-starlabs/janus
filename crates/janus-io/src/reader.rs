//! `.jvtk` reader: mmap the file, expose zero-copy `&[f64]`/`&[f32]` views
//! via `bytemuck::cast_slice`.

use crate::header::JvtkHeader;
use crate::MAGIC;
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

/// A memory-mapped `.jvtk` file with the parsed header and zero-copy field
/// accessors.
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
        let header_start = 16;
        let header_end = header_start + header_len;
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

        Ok(Self { mmap, header })
    }

    pub fn header(&self) -> &JvtkHeader {
        &self.header
    }

    /// Zero-copy `f64` view of a cell field block by name.
    pub fn cell_field_f64(&self, name: &str) -> Option<&[f64]> {
        let d = self.header.cell_fields.iter().find(|d| d.name == name)?;
        assert_eq!(d.dtype, "f64");
        let start = d.offset as usize;
        let end = start + d.len as usize * 8;
        Some(bytemuck::cast_slice(&self.mmap[start..end]))
    }

    /// Zero-copy `f32` view of a cell field block by name.
    pub fn cell_field_f32(&self, name: &str) -> Option<&[f32]> {
        let d = self.header.cell_fields.iter().find(|d| d.name == name)?;
        assert_eq!(d.dtype, "f32");
        let start = d.offset as usize;
        let end = start + d.len as usize * 4;
        Some(bytemuck::cast_slice(&self.mmap[start..end]))
    }

    /// Zero-copy `f64` view of a point field block by name.
    pub fn point_field_f64(&self, name: &str) -> Option<&[f64]> {
        let d = self.header.point_fields.iter().find(|d| d.name == name)?;
        assert_eq!(d.dtype, "f64");
        let start = d.offset as usize;
        let end = start + d.len as usize * 8;
        Some(bytemuck::cast_slice(&self.mmap[start..end]))
    }

    /// Raw particle block bytes, if present.
    pub fn particle_bytes(&self) -> Option<&[u8]> {
        let pd = self.header.particles.as_ref()?;
        let start = pd.offset as usize;
        let end = start + (pd.count * pd.stride) as usize;
        Some(&self.mmap[start..end])
    }
}
