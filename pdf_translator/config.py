"""Configuration module for PDF Translator."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for PDF processing."""

    model: str = "gpt-4o-mini"
    prompt: str = (
        "Please translate this text to English and provide a clean, formatted version."
    )
    max_retries: int = 3
    rate_limit_rpm: int = 60
    concurrent_requests: int = 3
    verbose: bool = False
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    model_path: Optional[str] = None  # Path to local model file for llama-cpp
    model_type: str = "openai"  # "openai" or "local"
    n_gpu_layers: int = 0  # Number of layers to offload to GPU (for llama-cpp)
    n_ctx: int = 2048  # Context size for llama-cpp models
    prompt_template: str = "standard"  # "standard", "persian", or "custom"
    preprocess_text: bool = True  # Enable text preprocessing
    remove_headers_footers: bool = True  # Remove headers/footers during preprocessing
    chunk_paragraphs: bool = True  # Chunk text into paragraphs during preprocessing

    @property
    def rate_limit_delay(self) -> float:
        """Calculate delay between requests in seconds based on rate limit."""
        if self.model_type == "local":
            # Local models don't need rate limiting, but we might want some
            # delay for processing
            return 0.1
        return 60.0 / self.rate_limit_rpm

    @property
    def is_local_model(self) -> bool:
        """Check if this is a local model configuration."""
        return self.model_type == "local" or self.model_path is not None

    def format_prompt(self, paragraph: str) -> str:
        """Format the prompt with paragraph content using the specified template."""
        if self.prompt_template == "persian":
            # Enhanced format for Persian translation with local models
            return f"{self.prompt}\n\n{paragraph}\n\nترجمه فارسی:"
        elif self.prompt_template == "standard":
            # Standard format with a neutral label
            return f"{self.prompt}\n\nText:\n{paragraph}"
        else:  # custom or any other value
            # Simple format: just prompt + content
            return f"{self.prompt}\n\n{paragraph}"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "prompt": self.prompt,
            "max_retries": self.max_retries,
            "rate_limit_rpm": self.rate_limit_rpm,
            "concurrent_requests": self.concurrent_requests,
            "verbose": self.verbose,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "prompt_template": self.prompt_template,
            "preprocess_text": self.preprocess_text,
            "remove_headers_footers": self.remove_headers_footers,
            "chunk_paragraphs": self.chunk_paragraphs,
        }
