# Makefile for running pre-commit hooks individually or all at once
# Requires: Poetry (uses the project's virtualenv)

SHELL := /bin/zsh
POETRY ?= poetry
PRE_COMMIT := $(POETRY) run pre-commit
CONFIG := .pre-commit-config.yaml

.PHONY: help pre-commit-help list-hooks pre-commit-install pre-commit-update pre-commit-all \
	pre-commit-trailing-whitespace pre-commit-end-of-file-fixer pre-commit-check-yaml \
	pre-commit-check-added-large-files pre-commit-check-merge-conflict pre-commit-debug-statements \
	pre-commit-black pre-commit-isort pre-commit-mypy

help: pre-commit-help

pre-commit-help:
	@echo "Pre-commit targets:"
	@echo "  make pre-commit-install       # Install git hooks"
	@echo "  make pre-commit-update        # Autoupdate hook revisions"
	@echo "  make pre-commit-all           # Run all hooks on all files"
	@echo "  make list-hooks               # Show hook IDs from $(CONFIG)"
	@echo "  make pre-commit-<hook-id>     # Run a single hook on all files"
	@echo ""
	@echo "Examples:"
	@echo "  make pre-commit-black"
	@echo "  make pre-commit-isort"
	@echo "  make pre-commit-mypy"
	@echo "  make pre-commit-trailing-whitespace"

# Utility: list configured hook ids from the YAML
list-hooks:
	@awk '/^[[:space:]]*-[[:space:]]*id:[[:space:]]/{print $$NF}' $(CONFIG)

# Install/Update
pre-commit-install:
	$(PRE_COMMIT) install --install-hooks

pre-commit-update:
	$(PRE_COMMIT) autoupdate

# Run all hooks on all files
pre-commit-all:
	$(PRE_COMMIT) run --all-files --show-diff-on-failure

# Pattern rule to run any hook by id: make pre-commit-<hook-id>
# e.g., make pre-commit-black, make pre-commit-isort, etc.
pre-commit-%:
	$(PRE_COMMIT) run $* --all-files --show-diff-on-failure

# Explicit convenience targets (documented) matching current config
pre-commit-trailing-whitespace: ; $(PRE_COMMIT) run trailing-whitespace --all-files --show-diff-on-failure
pre-commit-end-of-file-fixer: ; $(PRE_COMMIT) run end-of-file-fixer --all-files --show-diff-on-failure
pre-commit-check-yaml: ; $(PRE_COMMIT) run check-yaml --all-files --show-diff-on-failure
pre-commit-check-added-large-files: ; $(PRE_COMMIT) run check-added-large-files --all-files --show-diff-on-failure
pre-commit-check-merge-conflict: ; $(PRE_COMMIT) run check-merge-conflict --all-files --show-diff-on-failure
pre-commit-debug-statements: ; $(PRE_COMMIT) run debug-statements --all-files --show-diff-on-failure
pre-commit-black: ; $(PRE_COMMIT) run black --all-files --show-diff-on-failure
pre-commit-isort: ; $(PRE_COMMIT) run isort --all-files --show-diff-on-failure
pre-commit-mypy: ; $(PRE_COMMIT) run mypy --all-files --show-diff-on-failure
