"""Helpers for manifest-based timeline playback."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def frame_index_from_blender_frame(blender_frame: int, frame_offset: int) -> int:
    return blender_frame - frame_offset


def resolve_manifest_frame_path(
    manifest: dict[str, Any],
    manifest_path: str | None,
    data_root: str | None,
    frame_index: int,
) -> str:
    frames = manifest.get("frames", [])
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError("frame index out of range")

    frame = frames[frame_index]
    raw_path = frame.get("file") or frame.get("path") or ""
    if not raw_path:
        raise ValueError("manifest frame has no file entry")

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)

    base_dir = None
    if data_root:
        base_dir = Path(data_root)
    elif manifest_path:
        base_dir = Path(manifest_path).resolve().parent
    if base_dir is None:
        base_dir = Path.cwd()
    return str((base_dir / candidate).resolve())
