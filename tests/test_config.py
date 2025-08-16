"""Tests for the configuration module (paragraph-only)."""

import pytest

from pdf_translator.config import Config


class TestConfig:
    """Test cases for Config class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.model == "gpt-4o-mini"
        assert config.max_retries == 3
        assert config.rate_limit_rpm == 60
        assert config.concurrent_requests == 3
        assert config.verbose is False
        assert config.temperature == 0.1
        assert config.max_tokens is None
        # Paragraph-related defaults
        assert config.preprocess_text is True
        assert config.remove_headers_footers is True
        assert config.chunk_paragraphs is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = Config(
            model="gpt-4",
            max_retries=5,
            rate_limit_rpm=30,
            concurrent_requests=2,
            verbose=True,
            temperature=0.5,
            max_tokens=1000,
        )

        assert config.model == "gpt-4"
        assert config.max_retries == 5
        assert config.rate_limit_rpm == 30
        assert config.concurrent_requests == 2
        assert config.verbose is True
        assert config.temperature == 0.5
        assert config.max_tokens == 1000

    def test_rate_limit_delay(self):
        """Test rate limit delay calculation."""
        config = Config(rate_limit_rpm=60)
        assert config.rate_limit_delay == 1.0

        config = Config(rate_limit_rpm=30)
        assert config.rate_limit_delay == 2.0

        config = Config(rate_limit_rpm=120)
        assert config.rate_limit_delay == 0.5

    def test_to_dict(self):
        """Test configuration to dictionary conversion."""
        config = Config(
            model="gpt-4",
            prompt="Test prompt",
            max_retries=5,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model"] == "gpt-4"
        assert config_dict["prompt"] == "Test prompt"
        assert config_dict["max_retries"] == 5
        assert "rate_limit_rpm" in config_dict
        assert "concurrent_requests" in config_dict

    def test_to_dict_keys(self):
        """Config to_dict contains expected keys and no page range."""
        config = Config()
        d = config.to_dict()
        assert "model" in d
        assert "prompt" in d
        assert "preprocess_text" in d
        assert "remove_headers_footers" in d
        assert "chunk_paragraphs" in d
        assert "start_page" not in d
        assert "end_page" not in d
