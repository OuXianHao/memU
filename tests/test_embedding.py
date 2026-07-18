"""Unit tests for the standalone ``memu.embedding`` package.

These pin the embedding module's contract:

- per-provider backends (openai/jina/voyage/openrouter/doubao) build the right
  payload/endpoint and parse the ``data[].embedding`` response shape.
- the HTTP client falls back to an OpenAI-compatible backend for unknown
  providers and returns ``(vectors, raw_response)``.
- the gateway dispatches on ``client_backend``.
- ``EmbeddingConfig`` resolves per-provider base_url/api_key/model defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest  # noqa: E402

from memu.app.settings import EmbeddingConfig  # noqa: E402
from memu.embedding.backends import (  # noqa: E402
    JinaEmbeddingBackend,
    OpenAIEmbeddingBackend,
    OpenRouterEmbeddingBackend,
    VoyageEmbeddingBackend,
)
from memu.embedding.gateway import build_embedding_client  # noqa: E402
from memu.embedding.http_client import HTTPEmbeddingClient  # noqa: E402
from memu.embedding.openai_sdk import OpenAIEmbeddingSDKClient  # noqa: E402


@pytest.mark.parametrize(
    ("backend", "endpoint"),
    [
        (OpenAIEmbeddingBackend(), "/embeddings"),
        (JinaEmbeddingBackend(), "/embeddings"),
        (VoyageEmbeddingBackend(), "/embeddings"),
        (OpenRouterEmbeddingBackend(), "/api/v1/embeddings"),
    ],
)
def test_backend_payload_and_parse(backend, endpoint):
    assert backend.embedding_endpoint == endpoint
    assert backend.default_headers("k") == {"Authorization": "Bearer k"}

    payload = backend.build_embedding_payload(inputs=["a", "b"], embed_model="m")
    assert payload["model"] == "m"
    assert payload["input"] == ["a", "b"]

    parsed = backend.parse_embedding_response({"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]})
    assert parsed == [[0.1, 0.2], [0.3, 0.4]]


def test_http_client_unknown_provider_falls_back_to_openai():
    client = HTTPEmbeddingClient(base_url="https://x/v1", api_key="k", embed_model="m", provider="grok")
    assert isinstance(client.backend, OpenAIEmbeddingBackend)


def test_http_client_selects_registered_backend():
    client = HTTPEmbeddingClient(base_url="https://api.jina.ai/v1", api_key="k", embed_model="m", provider="jina")
    assert isinstance(client.backend, JinaEmbeddingBackend)


async def test_http_client_embed_returns_vectors_and_raw(monkeypatch):
    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"embedding": [1.0, 2.0]}], "usage": {"total_tokens": 3}}

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, endpoint, json, headers):
            captured["endpoint"] = endpoint
            captured["json"] = json
            captured["headers"] = headers
            return _FakeResponse()

    import memu.embedding.http_client as http_mod

    monkeypatch.setattr(http_mod.httpx, "AsyncClient", _FakeAsyncClient)

    client = HTTPEmbeddingClient(
        base_url="https://api.voyageai.com/v1", api_key="key", embed_model="voyage-3.5", provider="voyage"
    )
    vectors, raw = await client.embed(["hello"])

    assert vectors == [[1.0, 2.0]]
    assert raw["usage"]["total_tokens"] == 3
    assert captured["endpoint"] == "embeddings"  # leading slash stripped
    assert captured["headers"] == {"Authorization": "Bearer key"}
    assert captured["json"] == {"model": "voyage-3.5", "input": ["hello"]}


def test_gateway_builds_sdk_and_httpx_clients():
    sdk = build_embedding_client(EmbeddingConfig(provider="openai", api_key="k", client_backend="sdk"))
    assert isinstance(sdk, OpenAIEmbeddingSDKClient)

    httpx_client = build_embedding_client(EmbeddingConfig(provider="jina", api_key="k", client_backend="httpx"))
    assert isinstance(httpx_client, HTTPEmbeddingClient)
    assert isinstance(httpx_client.backend, JinaEmbeddingBackend)


def test_gateway_rejects_unknown_backends():
    with pytest.raises(ValueError, match="Unknown embedding client_backend"):
        build_embedding_client(EmbeddingConfig(client_backend="nope"))


def test_embedding_config_provider_defaults():
    jina = EmbeddingConfig(provider="jina")
    assert jina.base_url == "https://api.jina.ai/v1"
    assert jina.api_key == "JINA_API_KEY"
    assert jina.embed_model == "jina-embeddings-v3"

    voyage = EmbeddingConfig(provider="voyage")
    assert voyage.base_url == "https://api.voyageai.com/v1"
    assert voyage.api_key == "VOYAGE_API_KEY"
    assert voyage.embed_model == "voyage-3.5"

    # Explicit values always survive the provider-default merge.
    explicit = EmbeddingConfig(provider="jina", base_url="https://proxy/v1", api_key="real", embed_model="custom")
    assert explicit.base_url == "https://proxy/v1"
    assert explicit.api_key == "real"
    assert explicit.embed_model == "custom"


def test_local_embedding_config_accepts_aliases_and_selects_local_backend():
    cfg = EmbeddingConfig(provider="local", model="/models/bge-base-zh-v1.5", batch_size=32)
    assert cfg.embed_model == "/models/bge-base-zh-v1.5"
    assert cfg.embed_batch_size == 32
    assert cfg.client_backend == "local"


async def test_local_embedding_client_loads_sentence_transformer_and_normalizes(monkeypatch):
    captured = {"calls": 0}

    class _FakeSentenceTransformer:
        def __init__(self, model_path):
            captured["model_path"] = model_path

        def encode(self, texts, batch_size, normalize_embeddings):
            captured["calls"] += 1
            captured["texts"] = texts
            captured["batch_size"] = batch_size
            captured["normalize_embeddings"] = normalize_embeddings
            return [[1.0, 0.0], [0.0, 1.0]]

    class _FakeModule:
        SentenceTransformer = _FakeSentenceTransformer

    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeModule())

    from memu.embedding.local import LocalEmbeddingClient

    client = LocalEmbeddingClient(embed_model="/models/bge", batch_size=2)
    vectors, raw = await client.embed(["alpha", "beta"])

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert raw is None
    assert captured == {
        "calls": 1,
        "model_path": "/models/bge",
        "texts": ["alpha", "beta"],
        "batch_size": 2,
        "normalize_embeddings": True,
    }


def test_gateway_builds_local_client(monkeypatch):
    class _FakeSentenceTransformer:
        def __init__(self, model_path):
            self.model_path = model_path

    class _FakeModule:
        SentenceTransformer = _FakeSentenceTransformer

    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeModule())

    from memu.embedding.local import LocalEmbeddingClient

    client = build_embedding_client(EmbeddingConfig(provider="local", model="/models/bge", batch_size=16))
    assert isinstance(client, LocalEmbeddingClient)
    assert client.embed_model == "/models/bge"
    assert client.batch_size == 16
