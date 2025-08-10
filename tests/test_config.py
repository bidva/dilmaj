"""Tests for the configuration module."""

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
        assert config.start_page is None
        assert config.end_page is None

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
            start_page=10,
            end_page=20,
        )

        assert config.model == "gpt-4"
        assert config.max_retries == 5
        assert config.rate_limit_rpm == 30
        assert config.concurrent_requests == 2
        assert config.verbose is True
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.start_page == 10
        assert config.end_page == 20

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

    def test_page_range_defaults(self):
        """Test default page range values."""
        config = Config()

        assert config.start_page is None
        assert config.end_page is None

    def test_custom_page_range(self):
        """Test custom page range values."""
        config = Config(start_page=5, end_page=10)

        assert config.start_page == 5
        assert config.end_page == 10

    def test_get_page_range_start_only(self):
        """Test page range with only start page specified."""
        config = Config(start_page=5)

        start, end = config.get_page_range(20)
        assert start == 5
        assert end == 20

    def test_get_page_range_end_only(self):
        """Test page range with only end page specified."""
        config = Config(end_page=15)

        start, end = config.get_page_range(20)
        assert start == 1
        assert end == 15

    def test_get_page_range_both_specified(self):
        """Test page range with both start and end specified."""
        config = Config(start_page=5, end_page=15)

        start, end = config.get_page_range(20)
        assert start == 5
        assert end == 15

    def test_get_page_range_invalid_start(self):
        """Test page range with invalid start page."""
        config = Config(start_page=0)

        with pytest.raises(ValueError, match="Start page must be >= 1"):
            config.get_page_range(20)

    def test_get_page_range_invalid_end(self):
        """Test page range with invalid end page."""
        config = Config(end_page=25)

        with pytest.raises(ValueError, match="End page 25 exceeds total pages 20"):
            config.get_page_range(20)

    def test_get_page_range_start_greater_than_end(self):
        """Test page range where start is greater than end."""
        config = Config(start_page=15, end_page=10)

        with pytest.raises(
            ValueError, match="Start page 15 cannot be greater than end page 10"
        ):
            config.get_page_range(20)

    def test_to_dict_includes_page_range(self):
        """Test that to_dict includes page range configuration."""
        config = Config(start_page=5, end_page=10)

        config_dict = config.to_dict()

        assert config_dict["start_page"] == 5
        assert config_dict["end_page"] == 10
        assert "start_page" in config_dict
        assert "end_page" in config_dict

    def test_get_page_range_defaults(self):
        """Test page range with default values (None)."""
        config = Config()
        start, end = config.get_page_range(100)

        assert start == 1
        assert end == 100

    def test_get_page_range_custom_start(self):
        """Test page range with custom start page."""
        config = Config(start_page=10)
        start, end = config.get_page_range(100)

        assert start == 10
        assert end == 100

    def test_get_page_range_custom_end(self):
        """Test page range with custom end page."""
        config = Config(end_page=50)
        start, end = config.get_page_range(100)

        assert start == 1
        assert end == 50

    def test_get_page_range_custom_both(self):
        """Test page range with custom start and end pages."""
        config = Config(start_page=10, end_page=20)
        start, end = config.get_page_range(100)

        assert start == 10
        assert end == 20

    def test_get_page_range_validation_start_too_low(self):
        """Test page range validation for start page < 1."""
        config = Config(start_page=0)

        with pytest.raises(ValueError, match="Start page must be >= 1"):
            config.get_page_range(100)

    def test_get_page_range_validation_end_too_high(self):
        """Test page range validation for end page > total pages."""
        config = Config(end_page=150)

        with pytest.raises(ValueError, match="End page 150 exceeds total pages 100"):
            config.get_page_range(100)

    def test_get_page_range_validation_start_greater_than_end(self):
        """Test page range validation for start > end."""
        config = Config(start_page=20, end_page=10)

        with pytest.raises(
            ValueError, match="Start page 20 cannot be greater than end page 10"
        ):
            config.get_page_range(100)

    def test_get_page_range_edge_case_single_page(self):
        """Test page range for single page PDF."""
        config = Config()
        start, end = config.get_page_range(1)

        assert start == 1
        assert end == 1

    def test_get_page_range_edge_case_same_start_end(self):
        """Test page range where start and end are the same."""
        config = Config(start_page=5, end_page=5)
        start, end = config.get_page_range(10)

        assert start == 5
        assert end == 5
