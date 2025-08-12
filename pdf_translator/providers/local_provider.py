"""Local llama-cpp provider using langchain-community LlamaCpp."""

from __future__ import annotations

from typing import Optional

from langchain_community.llms import LlamaCpp

from .base import LLMProvider, ProviderConfig


class LocalLlamaProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._llm: Optional[LlamaCpp] = None

    def init(self) -> None:
        if not self.config.model_path:
            raise ValueError("model_path must be specified for local models")

        self._llm = LlamaCpp(
            model_path=self.config.model_path,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens or 512,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            verbose=False,
        )

    async def ainvoke(self, prompt: str) -> str:
        # wrap sync llama-cpp call in a thread for async compatibility
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, prompt)

    def invoke(self, prompt: str) -> str:
        if not self._llm:
            raise RuntimeError("LocalLlamaProvider not initialized")
        result = self._llm.invoke(prompt)
        return str(result) if not isinstance(result, str) else result
