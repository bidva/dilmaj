# Coverage Setup Summary

## What was added:

### 1. GitHub Actions Workflows
- **`.github/workflows/test-and-coverage.yml`**: Main CI/CD workflow that:
  - Runs tests on Python 3.9, 3.10, 3.11, and 3.12
  - Generates coverage reports
  - Uploads coverage to Codecov
  - Automatically updates the coverage badge on main branch commits

### 2. Coverage Configuration
- **`pyproject.toml`**: Added coverage settings and `coverage-badge` dependency
- **`.codecov.yml`**: Codecov configuration with 80% target coverage
- **`.gitignore`**: Updated to exclude `coverage.json` but keep the SVG badge

### 3. README Updates
- Added status badges including:
  - Tests status badge
  - Coverage badge (auto-updated)
  - Codecov badge
  - Python version badge
  - Poetry badge
- Added comprehensive development section with coverage information
- Added CI/CD documentation

### 4. Local Development Tools
- **`Makefile`**: Added `test`, `coverage`, and `coverage-badge` targets
- **`generate_coverage_badge.sh`**: Script for local badge generation
- **`coverage-badge.svg`**: Initial coverage badge (81.59%)

## How to use:

### Local Development
```bash
# Run tests
make test

# Run tests with coverage
make coverage

# Generate coverage badge locally
make coverage-badge
```

### GitHub Integration
- The coverage badge will be automatically updated on each push to main
- Coverage reports are uploaded to Codecov
- Tests run automatically on all pull requests

### Next Steps
1. Push these changes to GitHub
2. Set up Codecov integration (if desired) at https://codecov.io
3. The coverage badge will start updating automatically

## Current Coverage: 81.59%

The project has good test coverage! Areas for improvement (if desired):
- `dilmaj/cli.py`: 61.27% coverage - could add more CLI tests
- `dilmaj/extractors/pdf_extractor.py`: 70.73% coverage
- `dilmaj/extractors/docx_extractor.py`: 83.05% coverage
