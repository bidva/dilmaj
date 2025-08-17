"""Factory for creating LLM providers based on app Config."""

from __future__ import annotations

from typing import Literal

from ..config import Config
from .base import LLMProvider, ProviderConfig
from .openai_provider import OpenAIProvider

ProviderKind = Literal["openai"]


class ProviderFactory:
    @staticmethod
    def from_config(config: Config) -> LLMProvider:
        """Create an appropriate provider based on Config."""
        provider_config = ProviderConfig(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        provider: LLMProvider = OpenAIProvider(provider_config)

        provider.init()
        return provider
