"""Factory for creating LLM providers based on app Config."""

from __future__ import annotations

from typing import Literal

from ..config import Config
from .base import LLMProvider, ProviderConfig
from .local_provider import LocalLlamaProvider
from .openai_provider import OpenAIProvider

ProviderKind = Literal["openai", "local"]


class ProviderFactory:
    @staticmethod
    def from_config(config: Config) -> LLMProvider:
        """Create an appropriate provider based on Config."""
        provider_config = ProviderConfig(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            model_path=config.model_path,
            n_gpu_layers=config.n_gpu_layers,
            n_ctx=config.n_ctx,
        )

        if config.model_type == "local" or config.is_local_model:
            provider: LLMProvider = LocalLlamaProvider(provider_config)
        else:
            provider = OpenAIProvider(provider_config)

        provider.init()
        return provider
