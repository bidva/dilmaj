#!/usr/bin/env python3
"""
Simple coverage badge generator with color thresholds.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    # Load coverage data
    try:
        with open("coverage.json", "r") as f:
            data = json.load(f)
            coverage = data["totals"]["percent_covered"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"❌ Error reading coverage data: {e}")
        sys.exit(1)

    # Determine color based on coverage
    if coverage >= 90:
        color = "brightgreen"
        status = "🟢 Excellent"
    elif coverage >= 75:
        color = "green"
        status = "🟡 Good"
    elif coverage >= 65:
        color = "yellow"
        status = "🟠 Fair"
    else:
        color = "red"
        status = "🔴 Needs Improvement"

    coverage_int = int(round(coverage))

    print(f"📊 Current coverage: {coverage:.2f}%")
    print(f"📈 Coverage status: {status}")
    print(f"🎨 Badge color: {color}")

    # Generate badge URL
    badge_url = f"https://img.shields.io/badge/coverage-{coverage_int}%25-{color}"
    print(f"🔗 Badge URL: {badge_url}")

    # Try to download badge
    try:
        import urllib.request

        with urllib.request.urlopen(badge_url) as response:
            with open("coverage-badge.svg", "wb") as f:
                f.write(response.read())
        print("✅ Coverage badge generated: coverage-badge.svg")
    except Exception as e:
        print(f"⚠️  Could not download badge: {e}")
        print("🔧 Falling back to coverage-badge tool...")
        import subprocess

        subprocess.run(
            ["poetry", "run", "coverage-badge", "-o", "coverage-badge.svg", "-f"]
        )

    # Show thresholds
    print("\n📋 Coverage Thresholds:")
    print("   🟢 Excellent: ≥90%")
    print("   🟡 Good:      ≥75%")
    print("   🟠 Fair:      ≥65%")
    print("   🔴 Poor:      <65%")


if __name__ == "__main__":
    main()
