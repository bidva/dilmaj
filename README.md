# PDF Translator CLI

A powerful CLI tool that extracts paragraphs from documents and processes them with AI models (OpenAI GPT or local LLaMA models) with robust error handling and retry mechanisms.

## Features

- 📄 Extract paragraphs from documents
- 🤖 Process each paragraph with AI models:
  - **OpenAI GPT models** (GPT-4, GPT-3.5-turbo, GPT-4o-mini)
  - **Local LLaMA models** (FREE - no API costs!)
- 💰 **Cost-free processing** with local models via llama-cpp-python
- 🔄 Intelligent retry policy with exponential backoff
- ⚡ Rate limiting and HTTP error handling
- 📁 Organized output directory structure
- 🎨 Beautiful CLI interface with Rich
- ⚙️ Configurable processing options
- 🔒 **Privacy-focused**: Local models keep your documents on your machine

## Model Options

### OpenAI Models (API-based)
- `gpt-4o-mini` (recommended for cost/quality balance)
- `gpt-3.5-turbo` (fastest, cheapest)
- `gpt-4` (highest quality, most expensive)

### Local Models (FREE!)
- **Mistral 7B Instruct** - Excellent for text processing
- **LLaMA 2 7B/13B Chat** - Great general purpose models
- **OpenChat 3.5** - Fast and efficient
- Any `.gguf` model from Hugging Face

> 💡 **Local models run entirely on your hardware with zero API costs!**

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

### Option 1: Using OpenAI Models (API-based)

1. Set up your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

2. Process a PDF:

```bash
poetry run pdf-translator process document.pdf --model gpt-4o-mini
```

### Option 2: Using Local Models (FREE!)

1. Download a local model:

```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

2. Process a PDF with the local model:

```bash
poetry run pdf-translator process-local document.pdf --model-path ./mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

> 💡 **See [LLAMA_SETUP.md](LLAMA_SETUP.md) for detailed local model setup guide**

## Usage Examples

### Dry Run Preview

```bash
# See what would be processed without making any API calls
poetry run pdf-translator process document.pdf --dry-run
```

### Processing PDFs

```bash
# Basic OpenAI processing
poetry run pdf-translator process document.pdf

# With specific model and output directory
poetry run pdf-translator process document.pdf --model gpt-3.5-turbo --output-dir ./results

# Local model processing (FREE!)
poetry run pdf-translator process-local document.pdf --model-path /path/to/model.gguf

# Process paragraphs (full document)
poetry run pdf-translator process document.pdf

# Custom prompt
poetry run pdf-translator process document.pdf --prompt "Translate to French and summarize"

# Dry run (see what would be processed without API calls)
poetry run pdf-translator process document.pdf --dry-run
```

### Managing Local Models

```bash
# List available and suggested models
poetry run pdf-translator models

# Search custom directories for models
poetry run pdf-translator models --search-paths ~/my-models --search-paths /shared/models
```

## Command Reference

### `process` - Process PDF files

**OpenAI Models:**
```bash
pdf-translator process FILE.pdf [OPTIONS]
  --model TEXT              Model name (gpt-4o-mini, gpt-3.5-turbo, etc.)
  --output-dir PATH         Output directory (default: ./output)
  --prompt TEXT             Custom processing prompt
  --rate-limit INTEGER      Requests per minute (default: 60)
  --concurrent INTEGER      Concurrent requests (default: 3)
```

**Local Models:**
```bash
pdf-translator process FILE.pdf --local --model-path PATH [OPTIONS]
  --local                   Use local model instead of OpenAI
  --model-path PATH         Path to .gguf model file (required with --local)
  --n-gpu-layers INTEGER    GPU layers to offload (default: 0)
  --n-ctx INTEGER          Context size (default: 2048)
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


### `models` - Manage Local Models

```bash
pdf-translator models [--search-paths PATH]
```

## Local Model Setup

This section explains how to set up and use local LLaMA models with the PDF Translator.

### Benefits of Local Models

- 🆓 **FREE**: No API costs - run unlimited translations locally
- 🔒 **Private**: Your documents never leave your machine
- 🌐 **Offline**: Works without internet connection
- ⚡ **Fast**: No network latency for requests

### Quick Setup Guide

#### 1. Download a Model

Download a `.gguf` model file from Hugging Face. Here are some recommended models:

**For Translation Tasks:**

```bash
# Mistral 7B (Good for text processing)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# LLaMA 2 7B Chat (Balanced quality/speed)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

#### 2. Organize Your Models

```bash
# Create models directory
mkdir ~/models
mv *.gguf ~/models/
```

#### 3. Use with PDF Translator

```bash
# Process PDF with local model
poetry run pdf-translator process-local document.pdf --model-path ~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# With GPU acceleration (if available)
poetry run pdf-translator process-local document.pdf --model-path ~/models/model.gguf --n-gpu-layers 20

# Preview settings without processing (no API calls)
poetry run pdf-translator process document.pdf --dry-run
```

### Model Recommendations

| Model | Size | Use Case | Quality | Speed |
|-------|------|----------|---------|--------|
| **Mistral 7B Instruct** | ~4GB | Text processing, translation | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **LLaMA 2 7B Chat** | ~4GB | General purpose | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **LLaMA 2 13B Chat** | ~7GB | Higher quality | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **OpenChat 3.5** | ~4GB | Fast responses | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Performance Optimization

#### GPU Acceleration

If you have a compatible GPU, use `--n-gpu-layers` to offload computation:

```bash
# For modern GPUs (RTX 3060+, M1 Mac+)
poetry run pdf-translator process document.pdf --local --model-path model.gguf --n-gpu-layers 32

# For older/limited VRAM
poetry run pdf-translator process document.pdf --local --model-path model.gguf --n-gpu-layers 10
```

#### Memory Management

- **Q4_K_M**: Good balance of quality and size
- **Q8_0**: Higher quality, larger size
- **Q2_K**: Smallest size, lower quality

#### Context Size

Adjust `--n-ctx` based on your document content size:

```bash
# Typical documents (default)
poetry run pdf-translator process-local document.pdf --model-path model.gguf --n-ctx 2048

# Long documents
poetry run pdf-translator process-local document.pdf --model-path model.gguf --n-ctx 4096
```

### Common Model Locations

The tool automatically searches these directories:

- `~/models/`
- `~/.cache/huggingface/transformers/`
- `~/.ollama/models/`
- `/usr/local/share/models/`
- `./models/`

### Cost Comparison

| Processing Method | 100 paragraphs | 1000 paragraphs | Notes |
|-------------------|-----------|------------|-------|
| **Local Model** | $0.00 | $0.00 | Always free! |
| GPT-4o-mini | ~$0.50 | ~$5.00 | API costs |
| GPT-4 | ~$5.00 | ~$50.00 | High API costs |

### Complete Example

```bash
# 1. Download a model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# 2. Create models directory
mkdir ~/models
mv mistral-7b-instruct-v0.1.Q4_K_M.gguf ~/models/

# 3. Process your PDF
poetry run pdf-translator process-local my_document.pdf --model-path ~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf

# 4. Check results in ./output/
ls output/
```

## Configuration

### Environment Variables (.env file)

```env
# For OpenAI models
OPENAI_API_KEY=sk-your-actual-api-key-here
```

> **Note:** Local models don't require any API keys or environment variables!

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

### Local Model Issues

#### Model Loading Problems

1. **File not found**: Check that the model path is correct and the file exists
2. **Out of memory**: Try a smaller model (Q2_K instead of Q8_0) or reduce `--n-ctx`
3. **Slow processing**: Add GPU layers with `--n-gpu-layers` if you have a compatible GPU
4. **Model format error**: Ensure you're using a `.gguf` format model file

#### Performance Issues

```bash
# If model is running slowly, try GPU acceleration
poetry run pdf-translator process document.pdf --local --model-path model.gguf --n-gpu-layers 20

# If running out of memory, reduce context size
poetry run pdf-translator process document.pdf --local --model-path model.gguf --n-ctx 1024

# Try a smaller quantized model
# Q2_K < Q4_K_M < Q8_0 (size and quality)
```

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
