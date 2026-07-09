from __future__ import annotations

import importlib.util

import pytest

from worldmm_smvqa.transformers_backend import (
    TransformersGenerationError,
    generate_transformers_text,
)


def test_generate_reports_missing_transformers_dependency() -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed; remote environment")
    with pytest.raises(TransformersGenerationError, match="remote"):
        _ = generate_transformers_text("prompt", "model-ref")
