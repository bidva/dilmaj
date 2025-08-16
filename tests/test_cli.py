"""Tests for the CLI module."""

import os
import tempfile

from click.testing import CliRunner

from pdf_translator.cli import cli


class TestCLI:
    """Test cases for CLI functionality."""

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "PDF Translator CLI" in result.output

    def test_process_command_help(self):
        """Test process command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "--help"])
        assert result.exit_code == 0
        assert "Process a PDF file" in result.output

    def test_process_command_missing_api_key(self):
        """Test process command without API key."""
        runner = CliRunner()

        # Temporarily remove API key
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                result = runner.invoke(cli, ["process", tmp_pdf.name])
                assert result.exit_code == 1
                assert "OPENAI_API_KEY not found" in result.output
        finally:
            # Restore API key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_process_command_placeholder_api_key(self):
        """Test process command with placeholder API key."""
        runner = CliRunner()

        # Temporarily set placeholder API key
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                result = runner.invoke(cli, ["process", tmp_pdf.name])
                assert result.exit_code == 1
                assert "placeholder value" in result.output
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_process_command_empty_api_key(self):
        """Test process command with empty API key."""
        runner = CliRunner()

        # Temporarily set empty API key
        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "   "  # Just whitespace

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                result = runner.invoke(cli, ["process", tmp_pdf.name])
                assert result.exit_code == 1
                assert "is empty" in result.output
        finally:
            # Restore original key
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def test_process_command_nonexistent_file(self):
        """Test process command with nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process", "nonexistent.pdf"])
        assert result.exit_code == 2  # Click's file not found exit code

    def test_process_local_help(self):
        """Test process-local command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["process-local", "--help"])  # type: ignore
        assert result.exit_code == 0
        assert (
            "Process a PDF file" in result.output
            or "Process a PDF file using a local" in result.output
        )

    def test_process_local_requires_model_path(self):
        """process-local should require --model-path."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
            result = runner.invoke(cli, ["process-local", tmp_pdf.name])
            # Missing required option -> Click error code 2
            assert result.exit_code == 2
