from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, cast, override


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
    local_rank = os.environ.get("LOCAL_RANK")
    placement: dict[str, object] = (
        {"device": int(local_rank)} if local_rank else {"device_map": "auto"}
    )
    factory = cast(
        "Callable[..., _GenerationPipeline]",
        transformers.pipeline,
    )
    try:
        return factory("text-generation", model=model_ref, dtype="auto", **placement)
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
