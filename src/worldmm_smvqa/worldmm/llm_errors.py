from __future__ import annotations

from dataclasses import dataclass
from typing import override


@dataclass(frozen=True, slots=True)
class LLMMemoryError(Exception):
    stage: str
    detail: str

    @override
    def __str__(self) -> str:
        return f"LLMMemoryError: {self.stage}: {self.detail}"
