#!/bin/bash

# PDF Translator Development Setup Script

set -e

echo "Setting up PDF Translator development environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install dependencies
echo "Installing dependencies with Poetry..."
poetry install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env file and add your OpenAI API key"
fi

# Create output directory
mkdir -p output

# Run tests to ensure everything is working
echo "Running tests..."
poetry run pytest tests/ -v

echo "Setup complete!"
echo ""
echo "To use the CLI tool, run:"
echo "  poetry run pdf-translator --help"
echo ""
echo "Example usage:"
echo "  poetry run pdf-translator process your-file.pdf --output-dir results"
echo ""
echo "Don't forget to set your OPENAI_API_KEY in the .env file!"
