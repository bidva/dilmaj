#!/usr/bin/env python3
"""Simple test script to check pre-commit setup."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔍 {description}")
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print(f"Output: {result.stdout[:200]}...")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False


def main() -> None:
    """Test pre-commit hooks individually."""
    print("Testing pre-commit hooks individually...")

    # Change to the project directory
    project_dir = Path(__file__).parent
    print(f"Working in: {project_dir}")

    tests = [
        (
            "poetry run pre-commit run trailing-whitespace --all-files",
            "Trailing whitespace check",
        ),
        (
            "poetry run pre-commit run end-of-file-fixer --all-files",
            "End of file fixer",
        ),
        ("poetry run pre-commit run check-yaml --all-files", "YAML check"),
        ("poetry run pre-commit run black --all-files", "Black formatting"),
        ("poetry run pre-commit run isort --all-files", "Import sorting"),
        ("poetry run pre-commit run flake8 --all-files", "Flake8 linting"),
    ]

    results = []
    for cmd, desc in tests:
        success = run_command(cmd, desc)
        results.append((desc, success))

    print("\n" + "=" * 50)
    print("SUMMARY:")
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {desc}")

    # Test mypy separately (might take longer)
    print("\n🔍 Testing mypy (may take longer)...")
    mypy_success = run_command(
        "poetry run pre-commit run mypy --all-files", "MyPy type checking"
    )

    all_passed = all(success for _, success in results) and mypy_success

    if all_passed:
        print("\n🎉 All tests passed! Pre-commit setup is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
