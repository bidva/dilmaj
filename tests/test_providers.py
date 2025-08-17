"""Tests for provider factory and OpenAI provider wrapper (no network)."""

import asyncio
from typing import Any

import pytest

from dilmaj.config import Config
from dilmaj.providers.factory import ProviderFactory
from dilmaj.providers.openai_provider import OpenAIProvider


def test_factory_creates_openai_provider():
    p = ProviderFactory.from_config(Config())
    assert isinstance(p, OpenAIProvider)


@pytest.mark.asyncio
async def test_openai_provider_init_and_invoke(monkeypatch):
    # Patch ChatOpenAI to avoid network and create a stub response
    calls: dict[str, Any] = {"messages": None}

    class DummyLLM:
        def __init__(self, *a, **k):
            pass

        async def ainvoke(self, messages):
            calls["messages"] = messages

            class Resp:
                content = "ok"

            return Resp()

    import dilmaj.providers.openai_provider as mod

    monkeypatch.setattr(mod, "ChatOpenAI", DummyLLM)

    prov = OpenAIProvider(
        config=mod.ProviderConfig(model="gpt-4o-mini", max_tokens=128)
    )
    prov.init()
    out = await prov.ainvoke("hello")
    assert out == "ok"
    assert calls["messages"] is not None
