from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import override


@dataclass(frozen=True, slots=True)
class TransformersGenerationError(Exception):
    detail: str

    @override
    def __str__(self) -> str:
        return f"TransformersGenerationError: {self.detail}"


def generate_transformers_text(prompt: str, model_ref: str) -> str:
    script = (
        "import sys\n"
        "from transformers import pipeline\n"
        "prompt = sys.stdin.read()\n"
        "pipe = pipeline('text-generation', model=sys.argv[1], device_map='auto')\n"
        "rows = pipe(prompt, max_new_tokens=256, do_sample=False)\n"
        "text = rows[0]['generated_text']\n"
        "sys.stdout.write(text[len(prompt):].strip() if "
        "text.startswith(prompt) else text.strip())\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script, model_ref],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "Transformers generation failed"
        raise TransformersGenerationError(detail=detail)
    return result.stdout
