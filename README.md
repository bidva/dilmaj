# PDF Translator CLI

A powerful CLI tool that slices PDF files into individual pages and processes each page using GPT models with robust error handling and retry mechanisms.

## Features

- 📄 Slice PDF files into individual pages
- 🤖 Process each page with GPT models using LangChain
- 🔄 Intelligent retry policy with exponential backoff
- ⚡ Rate limiting and HTTP error handling
- 📁 Organized output directory structure
- 🎨 Beautiful CLI interface with Rich
- ⚙️ Configurable processing options

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

## Usage

### Basic Usage

Process a PDF file:
```bash
poetry run pdf-translator process input.pdf --output-dir results
```

### Advanced Options

```bash
poetry run pdf-translator process input.pdf \
    --output-dir results \
    --model gpt-4o-mini \
    --prompt "Translate this text to Spanish" \
    --max-retries 5 \
    --rate-limit 10
```

### Command Options

- `--output-dir`: Directory to save results (default: ./output)
- `--model`: GPT model to use (default: gpt-4o-mini)
- `--prompt`: Custom prompt for processing pages
- `--max-retries`: Maximum retry attempts (default: 3)
- `--rate-limit`: Requests per minute (default: 60)
- `--concurrent`: Number of concurrent requests (default: 3)

## Configuration

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key_here
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

Run tests:
```bash
poetry run pytest
```

Format code:
```bash
poetry run black pdf_translator/
poetry run isort pdf_translator/
```

Type checking:
```bash
poetry run mypy pdf_translator/
```

## License

MIT License
