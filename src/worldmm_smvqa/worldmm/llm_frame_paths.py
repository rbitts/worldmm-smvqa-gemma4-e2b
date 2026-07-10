from __future__ import annotations

from pathlib import Path

from worldmm_smvqa.video_frames import FRAME_EXTENSIONS
from worldmm_smvqa.worldmm.llm_errors import LLMMemoryError


def frame_file(frame_root: Path, video_id: str, frame_ref: str) -> Path:
    base = _safe_frame_base(frame_root, video_id, frame_ref)
    for suffix in FRAME_EXTENSIONS:
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise LLMMemoryError(stage="visual", detail=f"missing frame asset: {base}")


def _safe_frame_base(frame_root: Path, video_id: str, frame_ref: str) -> Path:
    if _unsafe_path_part(video_id) or _unsafe_path_part(frame_ref):
        raise LLMMemoryError(
            stage="visual",
            detail=f"frame path escapes frame root: {video_id}/{frame_ref}",
        )
    root = frame_root.resolve()
    base = root / video_id / frame_ref
    try:
        _ = base.resolve().relative_to(root)
    except ValueError as exc:
        raise LLMMemoryError(
            stage="visual",
            detail=f"frame path escapes frame root: {video_id}/{frame_ref}",
        ) from exc
    return base


def _unsafe_path_part(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or ".." in path.parts
