"""Additional coverage for provider base behaviors."""

import pytest

from dilmaj.providers.base import LLMProvider, ProviderConfig


class DummyProvider(LLMProvider):
    def init(self) -> None:  # pragma: no cover - trivial
        pass

    async def ainvoke(self, prompt: str) -> str:  # pragma: no cover - not used
        return prompt


def test_provider_base_invoke_raises():
    p = DummyProvider(ProviderConfig(model="x"))
    with pytest.raises(NotImplementedError):
        p.invoke("hello")


def test_provider_close_noop():
    p = DummyProvider(ProviderConfig(model="x"))
    assert p.close() is None
