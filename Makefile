.PHONY: sync run run-dev test test-cov lint lint-fix format format-check typecheck security check build install init uninstall docker-build docker-push docker run-up run-down clean clean-all info help

# Project / package names
PROJECT_NAME=llm-bench
PACKAGE_NAME=llm_bench
SRC_DIR=src

# Version derived from git tags (overridable)
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# Python version check
PYTHON_VERSION=$(shell python3 --version 2>/dev/null | cut -d' ' -f2)

# Detect if uv is available
HAS_UV=$(shell command -v uv >/dev/null 2>&1 && echo "yes" || echo "no")

# Docker configuration
MAKE_DOCKER_PREFIX ?=
DOCKER_TAG ?= latest

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

## sync: Install/update project dependencies using uv
sync:
ifeq ($(HAS_UV),yes)
	@echo "Syncing dependencies with uv..."
	@uv sync
	@echo "Dependencies synced!"
else
	@echo "Error: uv not found. Install it from https://docs.astral.sh/uv/"
	@exit 1
endif

# ============================================================================
# RUNNING
# ============================================================================

## run: Run the CLI application via uv
run: sync
ifdef ARGS
	@echo "Running $(PROJECT_NAME) with args: $(ARGS)..."
	@uv run $(PROJECT_NAME) $(ARGS)
else
	@echo "Running $(PROJECT_NAME)..."
	@uv run $(PROJECT_NAME)
endif

## run-dev: Run entry point module directly (useful during development)
run-dev:
ifdef ARGS
	@echo "Running $(PACKAGE_NAME).$(PACKAGE_NAME) with args: $(ARGS)..."
	@uv run python -m $(PACKAGE_NAME).$(PACKAGE_NAME) $(ARGS)
else
	@echo "Running $(PACKAGE_NAME).$(PACKAGE_NAME)..."
	@uv run python -m $(PACKAGE_NAME).$(PACKAGE_NAME)
endif

# ============================================================================
# TESTING
# ============================================================================

## test: Run tests with pytest (supports ARGS='...' for extra arguments)
test:
	@echo "Running tests..."
ifdef ARGS
	@uv run pytest -v $(ARGS)
else
	@uv run pytest -v
endif
	@echo "Tests complete!"

## test-cov: Run tests with coverage report
test-cov:
	@echo "Running tests with coverage..."
	@uv run pytest -v --cov=$(SRC_DIR) --cov-report=term-missing
	@echo "Tests complete!"

# ============================================================================
# CODE QUALITY
# ============================================================================

## lint: Check code style with Ruff
lint:
	@echo "Running Ruff linter..."
	@uv run ruff check .
	@echo "Lint check complete!"

## lint-fix: Auto-fix lint issues with Ruff
lint-fix:
	@echo "Running Ruff linter with auto-fix..."
	@uv run ruff check --fix .
	@echo "Lint fix complete!"

## format: Format code with Ruff
format:
	@echo "Formatting code with Ruff..."
	@uv run ruff format .
	@echo "Format complete!"

## format-check: Check code formatting without changes
format-check:
	@echo "Checking code format..."
	@uv run ruff format --check .
	@echo "Format check complete!"

## typecheck: Run type checking with mypy
typecheck:
	@echo "Running mypy type checker..."
	@uv run mypy $(SRC_DIR)/
	@echo "Type check complete!"

## security: Run bandit security scanner
security:
	@echo "Running bandit security scanner..."
	@uv run bandit -r $(SRC_DIR)/ -c pyproject.toml 2>/dev/null || uv run bandit -r $(SRC_DIR)/
	@echo "Security scan complete!"

## check: Run all quality checks (lint, format, typecheck, security, tests+coverage)
check: lint format-check typecheck security test-cov
	@echo "All checks passed!"

# ============================================================================
# BUILD & INSTALL
# ============================================================================

## build: Inject version, then build wheel and sdist packages
build: sync
	@echo "Injecting version $(VERSION)..."
	@echo '__version__ = "$(VERSION)"' > $(SRC_DIR)/$(PACKAGE_NAME)/version.py
	@echo "Building package..."
	@uv build
	@echo "Build complete! Artifacts in dist/"

## install: Install as a uv tool (available system-wide) and scaffold config + prompts
install:
	@echo "Installing $(PROJECT_NAME) as uv tool..."
	@uv tool install . --reinstall --force
	@$(MAKE) --no-print-directory init
	@echo "Install complete! Run '$(PROJECT_NAME)' from anywhere."

## init: Scaffold ~/.config/llm-bench (config.yaml + prompts/ + dashboards/) and the runs data dir
init:
	@uv run $(PROJECT_NAME) init

## uninstall: Remove uv tool
uninstall:
	@echo "Uninstalling $(PROJECT_NAME)..."
	@uv tool uninstall $(PROJECT_NAME) 2>/dev/null || echo "Not installed"
	@echo "Uninstall complete!"

# ============================================================================
# CLEANUP
# ============================================================================

## clean: Remove caches and build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@rm -rf $(SRC_DIR)/__pycache__ tests/__pycache__
	@rm -rf dist build *.egg-info $(SRC_DIR)/*.egg-info
	@rm -rf .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Clean complete!"

## clean-all: Remove everything including venv and lock file
clean-all: clean
	@echo "Removing virtual environment and lock file..."
	@rm -rf .venv
	@rm -f uv.lock
	@echo "Full clean complete!"

# ============================================================================
# DOCKER
# ============================================================================

## docker-build: Build Docker image
docker-build:
	@echo "Building Docker image: $(MAKE_DOCKER_PREFIX)$(PROJECT_NAME):$(DOCKER_TAG)..."
	@docker build --build-arg APP_VERSION=$(VERSION) -t $(MAKE_DOCKER_PREFIX)$(PROJECT_NAME):$(DOCKER_TAG) .
	@echo "Docker image built!"

## docker-push: Push Docker image to registry
docker-push:
	@echo "Pushing Docker image: $(MAKE_DOCKER_PREFIX)$(PROJECT_NAME):$(DOCKER_TAG)..."
	@docker push $(MAKE_DOCKER_PREFIX)$(PROJECT_NAME):$(DOCKER_TAG)
	@echo "Docker image pushed!"

## docker: Build and push Docker image
docker: docker-build docker-push

## run-up: Build Docker image and start docker compose
run-up: docker-build
	@echo "Starting services..."
	@PROJECT_NAME=$(PROJECT_NAME) DOCKER_PREFIX=$(MAKE_DOCKER_PREFIX) DOCKER_TAG=$(DOCKER_TAG) docker compose up -d
	@echo "Services started!"

## run-down: Stop docker compose services
run-down:
	@echo "Stopping services..."
	@PROJECT_NAME=$(PROJECT_NAME) DOCKER_PREFIX=$(MAKE_DOCKER_PREFIX) DOCKER_TAG=$(DOCKER_TAG) docker compose down
	@echo "Services stopped!"

# ============================================================================
# INFORMATION
# ============================================================================

## info: Show project information
info:
	@echo "Project Information"
	@echo "==================="
	@echo "Project name:    $(PROJECT_NAME)"
	@echo "Package name:    $(PACKAGE_NAME)"
	@echo "Source dir:      $(SRC_DIR)/"
	@echo "Version:         $(VERSION)"
	@echo "Python version:  $(PYTHON_VERSION)"
	@echo "uv available:    $(HAS_UV)"
	@echo "Layout:          src/ package ($(SRC_DIR)/$(PACKAGE_NAME))"

## help: Show this help message
help:
	@echo "llm-bench Project Makefile"
	@echo "=========================="
	@echo ""
	@echo "Dependency Management:"
	@echo "  sync             - Install/update dependencies with uv"
	@echo ""
	@echo "Running:"
	@echo "  run              - Run the CLI application via uv"
	@echo "  run-dev          - Run entry point module directly (development)"
	@echo ""
	@echo "Testing:"
	@echo "  test             - Run tests with pytest"
	@echo "  test-cov         - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             - Check code style with Ruff"
	@echo "  lint-fix         - Auto-fix lint issues"
	@echo "  format           - Format code with Ruff"
	@echo "  format-check     - Check formatting without changes"
	@echo "  typecheck        - Run mypy type checking"
	@echo "  security         - Run bandit security scanner"
	@echo "  check            - Run all quality checks (lint, format, typecheck, security, tests+coverage)"
	@echo ""
	@echo "Build & Install:"
	@echo "  build            - Inject version and build wheel and sdist packages"
	@echo "  install          - Install as a uv tool (system-wide) and scaffold config + prompts"
	@echo "  init             - Scaffold ~/.config/llm-bench (config.yaml + prompts/ + dashboards/) and runs dir"
	@echo "  uninstall        - Remove uv tool"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-push      - Push Docker image to registry"
	@echo "  docker           - Build and push Docker image"
	@echo "  run-up           - Build Docker image and start docker compose"
	@echo "  run-down         - Stop docker compose services"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            - Remove caches and build artifacts"
	@echo "  clean-all        - Remove everything including venv"
	@echo ""
	@echo "Information:"
	@echo "  info             - Show project information"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make run ARGS='--help'       - Run with --help flag"
	@echo "  make test ARGS='-k test_foo' - Run specific tests"
	@echo "  make check                   - Run all quality checks before commit"
