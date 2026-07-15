from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NotRequired, TypedDict

import pytest

from worldmm_smvqa.transformers_backend import (
    TransformersGenerationError,
    embed_transformers_image,
    generate_transformers_multimodal,
    generate_transformers_text,
)
from worldmm_smvqa.video_frames import QAVideoFrame


class _ContentPart(TypedDict):
    type: str
    image: NotRequired[str]
    text: NotRequired[str]


class _ChatMessage(TypedDict):
    role: str
    content: Sequence[_ContentPart]


def test_generate_reports_missing_transformers_dependency() -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers is installed; remote environment")
    with pytest.raises(TransformersGenerationError, match="remote"):
        _ = generate_transformers_text("prompt", "model-ref")


def test_generate_multimodal_uses_image_text_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a fake remote Transformers module and one real sampled frame file.
    frame = tmp_path / "frame.jpg"
    _ = frame.write_bytes(b"fake")
    calls: list[str] = []
    images: list[str] = []

    def fake_pipeline(task: str, **_kwargs: str) -> _FakePipeline:
        calls.append(task)
        return _FakePipeline(images)

    fake_transformers = SimpleNamespace(pipeline=fake_pipeline)

    def fake_import_module(name: str) -> SimpleNamespace:
        assert name == "transformers"
        return fake_transformers

    monkeypatch.setattr(
        "worldmm_smvqa.transformers_backend.importlib.import_module",
        fake_import_module,
    )

    # When: multimodal generation runs.
    output = generate_transformers_multimodal(
        "prompt",
        "model-ref-test-multimodal",
        (QAVideoFrame(video_id="v", frame_ref="f", timestamp=1.0, path=frame),),
    )

    # Then: image-text-to-text receives the frame path and prompt together.
    assert output == "answer"
    assert calls == ["image-text-to-text"]
    assert images == [str(frame)]


def test_generate_multimodal_reads_chat_message_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: real pipeline-style generated chat messages.
    class ChatPipeline:
        def __call__(
            self,
            _messages: Sequence[_ChatMessage],
            *,
            max_new_tokens: int,
            do_sample: bool,
        ) -> tuple[dict[str, object], ...]:
            assert max_new_tokens == 256
            assert not do_sample
            return (
                {
                    "generated_text": [
                        {"role": "user", "content": "prompt"},
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": '{"answer":"A"}'}],
                        },
                    ],
                },
            )

    def chat_pipeline(_model_ref: str) -> ChatPipeline:
        return ChatPipeline()

    monkeypatch.setattr(
        "worldmm_smvqa.transformers_backend._multimodal_pipeline",
        chat_pipeline,
    )

    # When / Then: assistant text is extracted from message content.
    assert (
        generate_transformers_multimodal("prompt", "chat-model", ()) == '{"answer":"A"}'
    )


def test_embed_transformers_image_flattens_feature_pipeline_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: a fake feature-extraction pipeline with nested numeric output.
    frame = tmp_path / "frame.jpg"
    _ = frame.write_bytes(b"fake")
    calls: list[tuple[str, dict[str, str]]] = []
    images: list[str] = []

    def fake_pipeline(task: str, **kwargs: str) -> _FakeEmbeddingPipeline:
        calls.append((task, kwargs))
        return _FakeEmbeddingPipeline(images)

    fake_transformers = SimpleNamespace(pipeline=fake_pipeline)

    def fake_import_module(name: str) -> SimpleNamespace:
        assert name == "transformers"
        return fake_transformers

    monkeypatch.setattr(
        "worldmm_smvqa.transformers_backend.importlib.import_module",
        fake_import_module,
    )

    # When: the image embedder runs.
    embedding = embed_transformers_image(frame, "model-ref-test-embedding")

    # Then: it uses image-feature-extraction and returns a flat tuple.
    assert embedding == (1.0, 2.0, 3.5)
    assert images == [str(frame)]
    assert calls[0][0] == "image-feature-extraction"
    assert "trust_remote_code" not in calls[0][1]


class _FakePipeline:
    def __init__(self, images: list[str]) -> None:
        self._images: list[str] = images

    def __call__(
        self,
        messages: Sequence[_ChatMessage],
        *,
        max_new_tokens: int,
        do_sample: bool,
    ) -> tuple[dict[str, str], ...]:
        self._images.extend(
            part["image"]
            for message in messages
            for part in message["content"]
            if part["type"] == "image" and "image" in part
        )
        assert max_new_tokens == 256
        assert not do_sample
        return ({"generated_text": "answer"},)


class _FakeEmbeddingPipeline:
    def __init__(self, images: list[str]) -> None:
        self._images: list[str] = images

    def __call__(self, image: str) -> tuple[tuple[int, float], tuple[float, ...]]:
        self._images.append(image)
        return ((1, 2.0), (3.5,))
