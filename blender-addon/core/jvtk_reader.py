"""Pure-Python .jvtk reader (mirrors janus-io layout)."""

from __future__ import annotations

import json
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"JVTK\x01\x00\x00\x00"
BLOCK_ALIGN = 64


@dataclass(frozen=True)
class FieldDesc:
    name: str
    comps: int
    dtype: str
    offset: int
    length: int


@dataclass(frozen=True)
class JvtkHeader:
    dims: tuple[int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    time: float
    step: int
    kn_range: tuple[float, float]
    cell_fields: tuple[FieldDesc, ...]

    @property
    def ncells(self) -> int:
        nx, ny, nz = self.dims
        return nx * ny * max(nz, 1)


def _parse_header(raw: dict[str, Any]) -> JvtkHeader:
    cell_fields = tuple(
        FieldDesc(
            name=d["name"],
            comps=int(d["comps"]),
            dtype=d["dtype"],
            offset=int(d["offset"]),
            length=int(d["len"]),
        )
        for d in raw.get("cell_fields", [])
    )
    dims = tuple(int(x) for x in raw["dims"])
    spacing = tuple(float(x) for x in raw["spacing"])
    origin = tuple(float(x) for x in raw["origin"])
    kn = raw.get("kn_range", [0.0, 0.0])
    return JvtkHeader(
        dims=(dims[0], dims[1], dims[2]),
        spacing=(spacing[0], spacing[1], spacing[2]),
        origin=(origin[0], origin[1], origin[2]),
        time=float(raw.get("time", 0.0)),
        step=int(raw.get("step", 0)),
        kn_range=(float(kn[0]), float(kn[1])),
        cell_fields=cell_fields,
    )


class JvtkReader:
    """Memory-mapped .jvtk reader with zero-copy NumPy views."""

    def __init__(self, path: Path, mm: mmap.mmap, header: JvtkHeader):
        self.path = path
        self._mm = mm
        self.header = header

    @classmethod
    def open(cls, path: str | Path) -> "JvtkReader":
        path = Path(path)
        f = path.open("rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()
        if len(mm) < 16:
            raise OSError("file too short for jvtk header")
        if mm[0:8] != MAGIC:
            raise OSError("bad jvtk magic")
        header_len = struct.unpack_from("<Q", mm, 8)[0]
        header_start = 16
        header_end = header_start + header_len
        raw = mm[header_start:header_end]
        trimmed = raw[: raw.rfind(b"}") + 1] if b"}" in raw else raw.rstrip(b"\x00")
        header_json = json.loads(trimmed.decode("utf-8"))
        return cls(path, mm, _parse_header(header_json))

    def cell_field(self, name: str) -> np.ndarray:
        desc = next((d for d in self.header.cell_fields if d.name == name), None)
        if desc is None:
            raise KeyError(f"cell field '{name}' not found in {self.path}")
        start = desc.offset
        if desc.dtype == "f64":
            count = desc.length
            return np.frombuffer(self._mm, dtype=np.float64, count=count, offset=start)
        if desc.dtype == "f32":
            count = desc.length
            return np.frombuffer(self._mm, dtype=np.float32, count=count, offset=start).astype(np.float64)
        raise ValueError(f"unsupported dtype {desc.dtype}")

    def available_fields(self) -> list[str]:
        return [d.name for d in self.header.cell_fields]

    def close(self):
        self._mm.close()


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
