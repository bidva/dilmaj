#!/usr/bin/env python3
"""
Enhanced coverage badge generator with color thresholds and custom styling.
"""

import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Tuple


class CoverageThresholds:
    """Coverage thresholds configuration."""

    EXCELLENT = 90.0
    GOOD = 80.0
    FAIR = 70.0
    POOR = 0.0

    @classmethod
    def get_color_and_status(cls, coverage: float) -> Tuple[str, str, str]:
        """
        Get badge color, status emoji, and description based on coverage.

        Args:
            coverage: Coverage percentage (0-100)

        Returns:
            Tuple of (color, emoji, description)
        """
        if coverage >= cls.EXCELLENT:
            return "brightgreen", "🟢", "Excellent"
        elif coverage >= cls.GOOD:
            return "green", "🟡", "Good"
        elif coverage >= cls.FAIR:
            return "yellow", "🟠", "Fair"
        else:
            return "red", "🔴", "Needs Improvement"


def load_coverage_data(coverage_file: Path = Path("coverage.json")) -> Optional[float]:
    """
    Load coverage percentage from coverage.json file.

    Args:
        coverage_file: Path to coverage.json file

    Returns:
        Coverage percentage or None if file not found/invalid
    """
    try:
        with open(coverage_file, "r") as f:
            data = json.load(f)
            coverage_value = data["totals"]["percent_covered"]
            return float(coverage_value)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"❌ Error reading coverage data: {e}")
        return None


def generate_badge_url(coverage: float, style: str = "flat") -> str:
    """
    Generate shields.io badge URL with appropriate color.

    Args:
        coverage: Coverage percentage
        style: Badge style (flat, plastic, flat-square, for-the-badge, social)

    Returns:
        Badge URL
    """
    color, _, _ = CoverageThresholds.get_color_and_status(coverage)
    coverage_int = int(round(coverage))

    return (
        f"https://img.shields.io/badge/coverage-{coverage_int}%25-{color}"
        f"?style={style}&logo=pytest&logoColor=white"
    )


def download_badge(url: str, output_file: Path = Path("coverage-badge.svg")) -> bool:
    """
    Download badge from URL and save to file.

    Args:
        url: Badge URL
        output_file: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        with urllib.request.urlopen(url) as response:
            with open(output_file, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"❌ Error downloading badge: {e}")
        return False


def print_coverage_report(coverage: float) -> None:
    """
    Print detailed coverage report with thresholds.

    Args:
        coverage: Coverage percentage
    """
    color, emoji, status = CoverageThresholds.get_color_and_status(coverage)

    print(f"\n📊 Coverage Report")
    print(f"{'='*50}")
    print(f"Current Coverage: {coverage:.2f}%")
    print(f"Status: {emoji} {status}")
    print(f"Badge Color: {color}")

    print(f"\n📋 Thresholds:")
    print(f"   🟢 Excellent: ≥{CoverageThresholds.EXCELLENT}%")
    print(f"   🟡 Good:      ≥{CoverageThresholds.GOOD}%")
    print(f"   🟠 Fair:      ≥{CoverageThresholds.FAIR}%")
    print(f"   🔴 Poor:      <{CoverageThresholds.FAIR}%")

    # Suggestions
    if coverage < CoverageThresholds.GOOD:
        needed = CoverageThresholds.GOOD - coverage
        print(f"\n💡 Tip: Add {needed:.1f}% more coverage to reach 'Good' status")
    elif coverage < CoverageThresholds.EXCELLENT:
        needed = CoverageThresholds.EXCELLENT - coverage
        print(f"\n💡 Tip: Add {needed:.1f}% more coverage to reach 'Excellent' status")
    else:
        print(f"\n🎉 Congratulations! You have excellent test coverage!")


def main() -> None:
    """Main function to generate coverage badge with colors."""
    print("🎨 Generating enhanced coverage badge...")

    # Load coverage data
    coverage = load_coverage_data()
    if coverage is None:
        print(
            "❌ Could not load coverage data. Make sure to run tests with coverage first."
        )
        sys.exit(1)

    # Print coverage report
    print_coverage_report(coverage)

    # Generate and download badge
    badge_styles = ["flat", "flat-square", "plastic"]

    for style in badge_styles:
        url = generate_badge_url(coverage, style)
        filename = f"coverage-badge-{style}.svg"

        print(f"\n🔗 Generating {style} badge...")
        print(f"URL: {url}")

        if download_badge(url, Path(filename)):
            print(f"✅ Badge saved: {filename}")
        else:
            print(f"❌ Failed to generate {filename}")

    # Generate default badge (flat style)
    default_url = generate_badge_url(coverage, "flat")
    if download_badge(default_url, Path("coverage-badge.svg")):
        print(f"\n✅ Default badge saved: coverage-badge.svg")

    # Check if coverage meets minimum threshold
    if coverage < CoverageThresholds.GOOD:
        print(
            f"\n⚠️  Warning: Coverage ({coverage:.2f}%) is below the recommended threshold ({CoverageThresholds.GOOD}%)"
        )
        sys.exit(1)
    else:
        print(f"\n✅ Coverage ({coverage:.2f}%) meets the minimum threshold!")


if __name__ == "__main__":
    main()
