#!/bin/bash

# Enhanced coverage badge generator with color thresholds
echo "🧪 Running tests with coverage..."

# Configuration for color thresholds
EXCELLENT_THRESHOLD=90
GOOD_THRESHOLD=80
FAIR_THRESHOLD=70

# Run tests with coverage
poetry run pytest tests/ --cov=dilmaj --cov-report=json --cov-report=term --cov-report=html

# Check if coverage.json exists
if [ ! -f "coverage.json" ]; then
    echo "❌ Error: coverage.json not found. Make sure tests ran successfully."
    exit 1
fi

# Extract coverage percentage from coverage.json
if command -v jq &> /dev/null; then
    COVERAGE=$(jq -r '.totals.percent_covered' coverage.json)
    COVERAGE_INT=$(printf "%.0f" "$COVERAGE")
    echo "📊 Current coverage: ${COVERAGE}%"

    # Determine badge color based on coverage thresholds
    if (( $(echo "$COVERAGE >= $EXCELLENT_THRESHOLD" | bc -l) )); then
        COLOR="brightgreen"
        STATUS="🟢 Excellent"
    elif (( $(echo "$COVERAGE >= $GOOD_THRESHOLD" | bc -l) )); then
        COLOR="green"
        STATUS="🟡 Good"
    elif (( $(echo "$COVERAGE >= $FAIR_THRESHOLD" | bc -l) )); then
        COLOR="yellow"
        STATUS="🟠 Fair"
    else
        COLOR="red"
        STATUS="🔴 Needs Improvement"
    fi

    echo "📈 Coverage status: $STATUS"

    # Generate coverage badge with custom color
    echo "🎨 Generating coverage badge with color: $COLOR..."
    poetry run coverage-badge -o coverage-badge.svg -f

    # Alternatively, use shields.io for more control over colors
    BADGE_URL="https://img.shields.io/badge/coverage-${COVERAGE_INT}%25-${COLOR}"
    echo "🔗 Badge URL: $BADGE_URL"

    # Download badge using curl if available
    if command -v curl &> /dev/null; then
        curl -s "$BADGE_URL" -o coverage-badge-custom.svg
        if [ -f "coverage-badge-custom.svg" ]; then
            mv coverage-badge-custom.svg coverage-badge.svg
            echo "✅ Custom coverage badge generated with $COLOR color"
        fi
    fi
else
    echo "⚠️  Warning: jq not found. Installing jq for JSON parsing..."
    if command -v brew &> /dev/null; then
        brew install jq
        echo "✅ jq installed via brew"
    else
        echo "❌ Please install jq manually: https://stedolan.github.io/jq/download/"
        exit 1
    fi
fi

# Generate coverage badge (fallback)
echo "🎨 Generating fallback coverage badge..."
poetry run coverage-badge -o coverage-badge.svg

if [ -f "coverage-badge.svg" ]; then
    echo "✅ Coverage badge generated: coverage-badge.svg"

    # Show coverage thresholds
    echo ""
    echo "� Coverage Thresholds:"
    echo "   🟢 Excellent: ≥${EXCELLENT_THRESHOLD}%"
    echo "   🟡 Good:      ≥${GOOD_THRESHOLD}%"
    echo "   🟠 Fair:      ≥${FAIR_THRESHOLD}%"
    echo "   🔴 Poor:      <${FAIR_THRESHOLD}%"
    echo ""

    # Coverage analysis
    if [ -n "$COVERAGE" ]; then
        if (( $(echo "$COVERAGE < $GOOD_THRESHOLD" | bc -l) )); then
            echo "💡 Tip: Consider adding more tests to reach the target of ${GOOD_THRESHOLD}% coverage"
        fi
    fi
else
    echo "❌ Error: Failed to generate coverage badge"
    exit 1
fi
