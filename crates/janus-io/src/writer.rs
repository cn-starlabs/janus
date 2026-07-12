//! Streaming `.jvtk` writer.
//!
//! Usage pattern: register field blocks up front (name + data), then call
//! `write_to_file` (or `write_series_step` for a time series) which computes
//! offsets, writes magic + header + padding + aligned raw blocks in one pass.
//!
//! ## Streaming design
//!
//! `write_file` never materializes the file's bytes (or any field's data) in
//! memory as a single buffer: it writes magic + header + padding, then
//! streams each field's data directly from the caller's borrowed `&[f64]`/
//! `&[f32]` slice into a `BufWriter` (bounded internal buffer, flushed
//! incrementally by `std::io::Write`), one 64-byte-aligned block at a time.
//! The only thing computed "up front" is the *header metadata* (field names,
//! dtypes, byte offsets/lengths) -- small, O(num_fields) JSON, not the bulk
//! field data itself, which is exactly what ENGINEERING_SPEC.md section 4
//! means by mmap-friendly/aligned blocks. The two-pass offset computation
//! below (serialize header with placeholder offsets, compute lengths, then
//! serialize again with final offsets) only ever touches header-sized JSON
//! bytes, never the (potentially large) field arrays, so it does not
//! reintroduce the "buffer everything" problem it might look like at a
//! glance -- the actual bulk data write loop (`for f in cell_fields { ...
//! f.data.write_bytes(&mut w) ... }` below) is a single streamed pass with
//! zero extra copies of the field data.

use crate::header::{FieldDesc, JvtkHeader, ParticleDesc};
use crate::{align_up, MAGIC};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// One named field's raw data, borrowed for the duration of a write call.
pub enum FieldData<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}

impl<'a> FieldData<'a> {
    fn dtype_str(&self) -> &'static str {
        match self {
            FieldData::F64(_) => "f64",
            FieldData::F32(_) => "f32",
        }
    }
    fn byte_len(&self) -> usize {
        match self {
            FieldData::F64(s) => std::mem::size_of_val(*s),
            FieldData::F32(s) => std::mem::size_of_val(*s),
        }
    }
    fn elem_len(&self) -> usize {
        match self {
            FieldData::F64(s) => s.len(),
            FieldData::F32(s) => s.len(),
        }
    }
    fn write_bytes(&self, w: &mut impl Write) -> io::Result<()> {
        match self {
            FieldData::F64(s) => w.write_all(bytemuck::cast_slice(s)),
            FieldData::F32(s) => w.write_all(bytemuck::cast_slice(s)),
        }
    }
}

/// A named field entry to register with the writer.
pub struct NamedField<'a> {
    pub name: String,
    pub comps: usize,
    pub data: FieldData<'a>,
}

/// Metadata describing an optional particle point-cloud block. The actual
/// bytes are supplied by the caller as a raw `&[u8]` already laid out per
/// `layout` (M1 does not populate particles; this exists so the format is
/// forward-compatible with M2 without a breaking change).
pub struct ParticleBlock<'a> {
    pub count: u64,
    pub stride: u64,
    pub layout: Vec<String>,
    pub bytes: &'a [u8],
}

/// Streaming `.jvtk` file writer.
pub struct JvtkWriter;

impl JvtkWriter {
    /// Write a single `.jvtk` file containing the given cell fields, point
    /// fields, and (optional) particle block.
    #[allow(clippy::too_many_arguments)]
    pub fn write_file(
        path: impl AsRef<Path>,
        dims: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
        time: f64,
        step: u64,
        kn_range: [f64; 2],
        cell_fields: &[NamedField],
        point_fields: &[NamedField],
        particles: Option<ParticleBlock>,
    ) -> io::Result<()> {
        // Pass 1: compute offsets. Offsets are absolute byte offsets from the
        // start of the file, which is why we need to know header_len before
        // we can finalize them -- but header_len depends on the offsets
        // (they're embedded in the JSON!). We break the chicken-and-egg by
        // serializing the header twice: once with placeholder (zero) offsets
        // to measure its length with the correct field *count* and name
        // lengths, then again with the real offsets substituted in-place.
        // Since offsets are u64 and JSON numbers can vary in digit width,
        // we instead reserve worst-case width by formatting offsets as fixed
        // width using serde_json's number formatting only after the header
        // length is fixed: we pad the header itself (not just the block
        // region) so a change in digit count of `header_len` cannot occur
        // from the offset values, because header_len is computed from the
        // *final* header bytes directly (single serialization), see below.
        let magic_and_len_size = 8 + 8; // magic + u64 header_len

        // Build header with placeholder offsets first to get a stable byte
        // layout for names/dtypes (offsets are u64, so digit-width changes
        // are possible only across power-of-10 boundaries; we avoid the
        // whole issue by doing a fixed-point iteration: serialize once with
        // offset=0 placeholders, compute header_len from THAT, then compute
        // real offsets, re-serialize, and if the new header_len differs
        // (digit width changed), redo once more. Two iterations always
        // suffice in practice; loop defensively.
        let mut cell_descs: Vec<FieldDesc> = cell_fields
            .iter()
            .map(|f| FieldDesc {
                name: f.name.clone(),
                comps: f.comps,
                dtype: f.data.dtype_str().to_string(),
                offset: 0,
                len: f.data.elem_len() as u64,
            })
            .collect();
        let mut point_descs: Vec<FieldDesc> = point_fields
            .iter()
            .map(|f| FieldDesc {
                name: f.name.clone(),
                comps: f.comps,
                dtype: f.data.dtype_str().to_string(),
                offset: 0,
                len: f.data.elem_len() as u64,
            })
            .collect();
        let mut particle_desc = particles.as_ref().map(|p| ParticleDesc {
            count: p.count,
            offset: 0,
            stride: p.stride,
            layout: p.layout.clone(),
        });

        let mut header_len_padded;
        for _ in 0..4 {
            let header = JvtkHeader {
                dims,
                spacing,
                origin,
                time,
                step,
                kn_range,
                cell_fields: cell_descs.clone(),
                point_fields: point_descs.clone(),
                particles: particle_desc.clone(),
            };
            let json = header.to_json_bytes();
            let unpadded_total = magic_and_len_size + json.len();
            header_len_padded = align_up(unpadded_total) - magic_and_len_size;

            // Now assign real offsets: block region starts right after
            // magic+len+header_len_padded bytes, and each block is 64-byte
            // aligned relative to file start.
            let mut cursor = magic_and_len_size + header_len_padded;
            for d in cell_descs.iter_mut() {
                cursor = align_up(cursor);
                d.offset = cursor as u64;
                cursor += (d.len as usize) * dtype_size(&d.dtype);
            }
            for d in point_descs.iter_mut() {
                cursor = align_up(cursor);
                d.offset = cursor as u64;
                cursor += (d.len as usize) * dtype_size(&d.dtype);
            }
            if let (Some(pd), Some(pb)) = (particle_desc.as_mut(), particles.as_ref()) {
                cursor = align_up(cursor);
                pd.offset = cursor as u64;
                let _ = pb;
            }

            // Check whether re-serializing with real offsets changes the
            // header length (digit-width change in a u64 offset). If not,
            // we're done.
            let header2 = JvtkHeader {
                dims,
                spacing,
                origin,
                time,
                step,
                kn_range,
                cell_fields: cell_descs.clone(),
                point_fields: point_descs.clone(),
                particles: particle_desc.clone(),
            };
            let json2 = header2.to_json_bytes();
            if json2.len() == json.len() {
                break;
            }
        }

        // Final header with settled offsets.
        let header = JvtkHeader {
            dims,
            spacing,
            origin,
            time,
            step,
            kn_range,
            cell_fields: cell_descs,
            point_fields: point_descs,
            particles: particle_desc,
        };
        let json = header.to_json_bytes();
        let header_len = align_up(magic_and_len_size + json.len()) - magic_and_len_size;

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);
        w.write_all(&MAGIC)?;
        w.write_all(&(header_len as u64).to_le_bytes())?;
        w.write_all(&json)?;
        // pad header to header_len
        let pad = header_len - json.len();
        w.write_all(&vec![0u8; pad])?;

        let mut written = magic_and_len_size + header_len;
        for f in cell_fields {
            let target = align_up(written);
            w.write_all(&vec![0u8; target - written])?;
            f.data.write_bytes(&mut w)?;
            written = target + f.data.byte_len();
        }
        for f in point_fields {
            let target = align_up(written);
            w.write_all(&vec![0u8; target - written])?;
            f.data.write_bytes(&mut w)?;
            written = target + f.data.byte_len();
        }
        if let Some(pb) = particles {
            let target = align_up(written);
            w.write_all(&vec![0u8; target - written])?;
            w.write_all(pb.bytes)?;
        }

        w.flush()
    }

    /// Write one frame of a time series to `case.NNNN.jvtk` in `dir`.
    #[allow(clippy::too_many_arguments)]
    pub fn write_series_step(
        dir: impl AsRef<Path>,
        case_name: &str,
        step_index: u32,
        dims: [usize; 3],
        spacing: [f64; 3],
        origin: [f64; 3],
        time: f64,
        step: u64,
        kn_range: [f64; 2],
        cell_fields: &[NamedField],
        point_fields: &[NamedField],
        particles: Option<ParticleBlock>,
    ) -> io::Result<std::path::PathBuf> {
        let path = dir
            .as_ref()
            .join(format!("{case_name}.{step_index:04}.jvtk"));
        Self::write_file(
            &path,
            dims,
            spacing,
            origin,
            time,
            step,
            kn_range,
            cell_fields,
            point_fields,
            particles,
        )?;
        Ok(path)
    }
}

fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "f64" => 8,
        "f32" => 4,
        _ => panic!("unknown dtype {dtype}"),
    }
}
