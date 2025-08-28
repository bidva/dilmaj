# PDF Translator CLI

[![Tests](https://github.com/bidva/dilmaj/actions/workflows/test-and-coverage.yml/badge.svg)](https://github.com/bidva/dilmaj/actions/workflows/test-and-coverage.yml)
[![Coverage](https://raw.githubusercontent.com/bidva/dilmaj/main/coverage-badge.svg)](https://github.com/bidva/dilmaj/actions/workflows/coverage-badge.yml)
[![codecov](https://codecov.io/gh/bidva/dilmaj/branch/main/graph/badge.svg)](https://codecov.io/gh/bidva/dilmaj)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency--manager-poetry-blue)](https://python-poetry.org/)

A CLI tool that extracts paragraphs from documents and processes them with OpenAI GPT models. It includes robust error handling, retries, and clear outputs.

## Features

- 📄 Extract paragraphs from documents
- 🤖 Process each paragraph with OpenAI GPT models
- 🔄 Intelligent retry policy with exponential backoff
- ⚡ Rate limiting and HTTP error handling
- 📁 Organized output directory structure
- 🎨 Beautiful CLI interface with Rich
- ⚙️ Configurable processing options


## Model Options

### OpenAI Models (API-based)

- `gpt-4o-mini` (recommended for cost/quality balance)
- `gpt-3.5-turbo` (fast and low cost)
- `gpt-4` (highest quality)



## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd translator
```

2. Install Poetry (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Add Poetry to your PATH:

```bash
export PATH="/Users/$(whoami)/.local/bin:$PATH"
```

   Add this line to your shell profile (`.zshrc`, `.bashrc`, etc.) to make it permanent.

4. Install dependencies using Poetry:

```bash
poetry install
```

5. Set up your environment variables:

```bash
cp .env.example .env
```

   Then edit `.env` and replace `your_openai_api_key_here` with your actual OpenAI API key from [OpenAI API Keys](https://platform.openai.com/account/api-keys).

## Quick Start

### Using OpenAI Models (API-based)

1. Set up your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

2. Process a PDF:

```bash
poetry run dilmaj process document.pdf --model gpt-4o-mini
```



## Usage Examples

### Dry Run Preview

```bash
# See what would be processed without making any API calls
poetry run dilmaj process document.pdf --dry-run
```

### Processing PDFs

```bash
# Basic OpenAI processing
poetry run dilmaj process document.pdf

# With specific model and output directory
poetry run dilmaj process document.pdf --model gpt-3.5-turbo --output-dir ./results



# Process paragraphs (full document)
poetry run dilmaj process document.pdf

# Custom prompt
poetry run dilmaj process document.pdf --prompt "Translate to French and summarize"

# Dry run (see what would be processed without API calls)
poetry run dilmaj process document.pdf --dry-run
```



## Command Reference

### `process` - Process PDF files

**OpenAI Models:**
```bash
dilmaj process FILE.pdf [OPTIONS]
  --model TEXT              Model name (gpt-4o-mini, gpt-3.5-turbo, etc.)
  --output-dir PATH         Output directory (default: ./output)
  --prompt TEXT             Custom processing prompt
  --rate-limit INTEGER      Requests per minute (default: 60)
  --concurrent INTEGER      Concurrent requests (default: 3)
```

**Common Options:**
```bash
  --dry-run                 Show what would be processed
  --verbose                 Enable verbose output
  --no-preprocess           Disable text preprocessing
  --keep-headers-footers    Keep headers and footers during preprocessing
  --no-paragraph-chunking   Disable paragraph chunking during preprocessing
  --from-extracted-dir PATH Process pre-extracted paragraph_*.txt in this directory
```






## Configuration

### Environment Variables (.env file)

```env
# For OpenAI models
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Troubleshooting

### API Key Issues

If you get API key related errors, the application now validates your OpenAI API key and will stop execution with helpful error messages:

- **`OPENAI_API_KEY not found`**: You haven't set the API key at all
- **`OPENAI_API_KEY is empty`**: The key is set but contains only whitespace
- **`placeholder value`**: You're using a placeholder value like `your_openai_api_key_here`

To fix API key issues:

1. Make sure you have a valid OpenAI API key from [OpenAI API Keys](https://platform.openai.com/account/api-keys)
2. Edit your `.env` file and replace `your_openai_api_key_here` with your actual API key
3. Ensure your API key has sufficient credits and permissions
4. Verify your API key starts with `sk-` (OpenAI's standard format)



### Poetry Command Not Found

If you get `zsh: command not found: poetry`:

1. Make sure Poetry is installed: `curl -sSL https://install.python-poetry.org | python3 -`
2. Add Poetry to your PATH: `export PATH="/Users/$(whoami)/.local/bin:$PATH"`
3. Or use the full path: `/Users/$(whoami)/.local/bin/poetry install`

## Development

### Running Tests

```bash
# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=dilmaj --cov-report=html --cov-report=term

# Or use make targets
make test
make coverage
make coverage-badge
```

### Code Quality

Format code:

```bash
poetry run black dilmaj/
poetry run isort dilmaj/
```

Type checking:

```bash
poetry run mypy dilmaj/
```

### Coverage

The project maintains high test coverage. You can view the coverage report by running:

```bash
make coverage
```

This will generate an HTML coverage report in the `htmlcov/` directory. Open `htmlcov/index.html` in your browser to view detailed coverage information.

The coverage badge in this README is automatically updated by GitHub Actions on each commit to the main branch.

### Continuous Integration

This project uses GitHub Actions for continuous integration:

- **Tests**: Run automatically on Python 3.9, 3.10, 3.11, and 3.12 for all pull requests and pushes to main/develop branches
- **Coverage**: Coverage reports are uploaded to Codecov
- **Coverage Badge**: Automatically updated on each commit to main branch
- **Code Quality**: Pre-commit hooks ensure code formatting and quality

### Local Development Workflow

1. Make your changes
2. Run tests locally: `make test`
3. Check coverage: `make coverage`
4. Update coverage badge: `make coverage-badge`
5. Commit and push your changes

The GitHub Actions will automatically run tests and update the coverage badge when you push to the main branch.

## License

MIT License
