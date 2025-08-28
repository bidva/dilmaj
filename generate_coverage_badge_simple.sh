#!/bin/bash

# Simple coverage badge generator with color thresholds
echo "🧪 Running tests with coverage..."

# Configuration for color thresholds
EXCELLENT_THRESHOLD=90
GOOD_THRESHOLD=80
FAIR_THRESHOLD=70

# Run tests with coverage (skip if already done)
if [ ! -f "coverage.json" ]; then
    echo "📊 Running coverage tests..."
    poetry run pytest tests/ --cov=dilmaj --cov-report=json --cov-report=term --cov-report=html
fi

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

    # Determine badge color based on coverage thresholds using Python for comparison
    COLOR=$(python3 -c "
coverage = float('$COVERAGE')
if coverage >= $EXCELLENT_THRESHOLD:
    print('brightgreen')
elif coverage >= $GOOD_THRESHOLD:
    print('green')
elif coverage >= $FAIR_THRESHOLD:
    print('yellow')
else:
    print('red')
")

    STATUS=$(python3 -c "
coverage = float('$COVERAGE')
if coverage >= $EXCELLENT_THRESHOLD:
    print('🟢 Excellent')
elif coverage >= $GOOD_THRESHOLD:
    print('🟡 Good')
elif coverage >= $FAIR_THRESHOLD:
    print('🟠 Fair')
else:
    print('🔴 Needs Improvement')
")

    echo "📈 Coverage status: $STATUS"

    # Use shields.io for badge generation with custom colors
    BADGE_URL="https://img.shields.io/badge/coverage-${COVERAGE_INT}%25-${COLOR}"
    echo "🔗 Badge URL: $BADGE_URL"

    # Download badge using curl if available
    if command -v curl &> /dev/null; then
        echo "🎨 Downloading custom coverage badge..."
        curl -s -L "$BADGE_URL" -o coverage-badge.svg
        if [ -f "coverage-badge.svg" ] && [ -s "coverage-badge.svg" ]; then
            echo "✅ Custom coverage badge generated with $COLOR color"
        else
            echo "⚠️  Fallback to coverage-badge tool..."
            poetry run coverage-badge -o coverage-badge.svg -f
        fi
    else
        echo "🎨 Generating coverage badge with coverage-badge tool..."
        poetry run coverage-badge -o coverage-badge.svg -f
    fi
else
    echo "⚠️  jq not found. Using coverage-badge tool only..."
    poetry run coverage-badge -o coverage-badge.svg -f
    COVERAGE="Unknown"
fi

if [ -f "coverage-badge.svg" ]; then
    echo "✅ Coverage badge generated: coverage-badge.svg"

    # Show coverage thresholds
    echo ""
    echo "📋 Coverage Thresholds:"
    echo "   🟢 Excellent: ≥${EXCELLENT_THRESHOLD}%"
    echo "   🟡 Good:      ≥${GOOD_THRESHOLD}%"
    echo "   🟠 Fair:      ≥${FAIR_THRESHOLD}%"
    echo "   🔴 Poor:      <${FAIR_THRESHOLD}%"
    echo ""

    # Coverage analysis using Python for comparison
    if [ "$COVERAGE" != "Unknown" ]; then
        NEEDS_IMPROVEMENT=$(python3 -c "print('1' if float('$COVERAGE') < $GOOD_THRESHOLD else '0')")
        if [ "$NEEDS_IMPROVEMENT" = "1" ]; then
            echo "💡 Tip: Consider adding more tests to reach the target of ${GOOD_THRESHOLD}% coverage"
        fi
    fi
else
    echo "❌ Error: Failed to generate coverage badge"
    exit 1
fi
