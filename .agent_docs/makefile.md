# Makefile Documentation

## Overview

Generic Makefile for Python projects with uv-based dependency management and Docker support. Auto-detects project name from `pyproject.toml` and entry point from `src/`.

## Key Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_NAME` | Auto-detected from `pyproject.toml` | — |
| `ENTRY_POINT` | First `.py` in `src/` (not `__init__.py`) | `app` |
| `SRC_DIR` | Source directory | `src` or `.` |
| `MAKE_DOCKER_PREFIX` | Docker registry prefix | empty |
| `DOCKER_TAG` | Docker image tag | `latest` |

## Dependency Management

| Target | Description |
|--------|-------------|
| `sync` | Install/update dependencies with `uv sync` |

## Run Targets

```bash
make run                    # Run via uv
make run ARGS='--help'      # Run with arguments
make run-dev                # Run entry point directly
```

## Test Targets

| Target | Description |
|--------|-------------|
| `test` | Run tests with pytest |
| `test-cov` | Run tests with coverage report |

## Code Quality Targets

| Target | Description |
|--------|-------------|
| `lint` | Check code style with Ruff |
| `lint-fix` | Auto-fix lint issues |
| `format` | Format code with Ruff |
| `format-check` | Check formatting without changes |
| `typecheck` | Run mypy type checking |
| `check` | Run all quality checks (lint, format-check, typecheck, test) |

## Build & Install Targets

| Target | Description |
|--------|-------------|
| `build` | Build wheel and sdist packages |
| `install` | Install as a uv tool (system-wide); also runs `init` to scaffold config + prompts |
| `init` | Scaffold `~/.config/llm-bench` (starter `config.yaml` + a `prompts/` dir with `short.yaml` + `long.yaml`) and the `~/.local/share/llm-bench/runs` data dir (via `llm-bench init`) |
| `uninstall` | Remove the uv tool |

## Docker Targets

| Target | Description |
|--------|-------------|
| `docker-build` | Build Docker image |
| `docker-push` | Push Docker image to registry |
| `docker` | Build and push Docker image |
| `run-up` | Build Docker image and start docker compose |
| `run-down` | Stop docker compose services |

Example with custom registry:
```bash
MAKE_DOCKER_PREFIX=gcr.io/my-project/ DOCKER_TAG=v1.0.0 make docker
```

## Cleanup Targets

| Target | Description |
|--------|-------------|
| `clean` | Remove caches and build artifacts |
| `clean-all` | Remove everything including venv and lock file |

## Other Targets

| Target | Description |
|--------|-------------|
| `info` | Show project information |
| `help` | Show help message |
