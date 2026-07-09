from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, cast, override

if TYPE_CHECKING:
    from worldmm_smvqa.video_frames import QAVideoFrame


@dataclass(frozen=True, slots=True)
class TransformersGenerationError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TransformersGenerationError: {self.detail}"


class _GenerationPipeline(Protocol):
    def __call__(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Sequence[Mapping[str, str]]: ...


class _ContentPart(TypedDict):
    type: str
    image: NotRequired[str]
    text: NotRequired[str]


class _ChatMessage(TypedDict):
    role: str
    content: Sequence[_ContentPart]


class _MultimodalPipeline(Protocol):
    def __call__(
        self,
        messages: Sequence[_ChatMessage],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> Sequence[Mapping[str, str]]: ...


@lru_cache(maxsize=1)
def _pipeline(model_ref: str) -> _GenerationPipeline:
    """Load the Transformers pipeline once per process.

    Under torch.distributed.run each rank pins to its LOCAL_RANK GPU so ranks
    do not shard one copy of the model across every visible device.
    """
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise TransformersGenerationError(
            detail="transformers is not installed; install the 'remote' extra",
        ) from exc
    factory = cast(
        "Callable[..., _GenerationPipeline]",
        transformers.pipeline,
    )
    try:
        return factory(
            "text-generation",
            model=model_ref,
            dtype="auto",
            **_placement(),
        )
    except Exception as exc:
        raise TransformersGenerationError(detail=str(exc)) from exc


@lru_cache(maxsize=1)
def _multimodal_pipeline(model_ref: str) -> _MultimodalPipeline:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise TransformersGenerationError(
            detail="transformers is not installed; install the 'remote' extra",
        ) from exc
    factory = cast(
        "Callable[..., _MultimodalPipeline]",
        transformers.pipeline,
    )
    try:
        return factory(
            "image-text-to-text",
            model=model_ref,
            dtype="auto",
            **_placement(),
        )
    except Exception as exc:
        raise TransformersGenerationError(detail=str(exc)) from exc


def generate_transformers_text(prompt: str, model_ref: str) -> str:
    pipe = _pipeline(model_ref)
    try:
        rows = pipe(prompt, max_new_tokens=256, do_sample=False)
        text = rows[0]["generated_text"]
    except Exception as exc:
        raise TransformersGenerationError(detail=str(exc)) from exc
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def generate_transformers_multimodal(
    prompt: str,
    model_ref: str,
    video_frames: Sequence[QAVideoFrame],
) -> str:
    paths = tuple(_required_frame_path(frame) for frame in video_frames)
    pipe = _multimodal_pipeline(model_ref)
    content: tuple[_ContentPart, ...] = (
        *({"type": "image", "image": str(path)} for path in paths),
        {"type": "text", "text": prompt},
    )
    messages: tuple[_ChatMessage, ...] = (
        {
            "role": "user",
            "content": content,
        },
    )
    try:
        rows = pipe(messages, max_new_tokens=256, do_sample=False)
        text = rows[0]["generated_text"]
    except Exception as exc:
        raise TransformersGenerationError(detail=str(exc)) from exc
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def _placement() -> Mapping[str, int | str]:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank:
        return {"device": int(local_rank)}
    return {"device_map": "auto"}


def _required_frame_path(frame: QAVideoFrame) -> Path:
    if frame.path is None:
        raise TransformersGenerationError(
            detail=f"missing frame path: {frame.frame_ref}",
        )
    if not frame.path.is_file():
        raise TransformersGenerationError(detail=f"missing frame asset: {frame.path}")
    return frame.path
