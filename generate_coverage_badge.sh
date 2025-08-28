#!/bin/bash

# Generate initial coverage badge
echo "🧪 Running tests with coverage..."

# Run tests with coverage
poetry run pytest tests/ --cov=dilmaj --cov-report=json --cov-report=term

# Check if coverage.json exists
if [ ! -f "coverage.json" ]; then
    echo "❌ Error: coverage.json not found. Make sure tests ran successfully."
    exit 1
fi

# Generate coverage badge
echo "🎨 Generating coverage badge..."
poetry run coverage-badge -o coverage-badge.svg

if [ -f "coverage-badge.svg" ]; then
    echo "✅ Coverage badge generated: coverage-badge.svg"

    # Extract coverage percentage from coverage.json
    if command -v jq &> /dev/null; then
        COVERAGE=$(jq -r '.totals.percent_covered' coverage.json)
        echo "📊 Current coverage: ${COVERAGE}%"
    fi
else
    echo "❌ Error: Failed to generate coverage badge"
    exit 1
fi
