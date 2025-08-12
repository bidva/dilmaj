"""OpenAI provider implementation using langchain-openai."""

from __future__ import annotations

from typing import Optional

from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

from .base import LLMProvider, ProviderConfig


class OpenAIProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._client: Optional[ChatOpenAI] = None

    def init(self) -> None:
        # Create ChatOpenAI instance
        if self.config.max_tokens is not None:
            self._client = ChatOpenAI(  # type: ignore
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,  # type: ignore
            )
        else:
            self._client = ChatOpenAI(  # type: ignore
                model=self.config.model, temperature=self.config.temperature
            )

    async def ainvoke(self, prompt: str) -> str:
        if not self._client:
            raise RuntimeError("OpenAIProvider not initialized")
        message = HumanMessage(content=prompt)
        response = await self._client.ainvoke([message])
        return str(response.content) if response.content else ""
