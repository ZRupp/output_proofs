# Output Proofs

Adversarial benchmark for code generation with formal verification.

## Project Structure

```
src/output_proofs/     # Main package
  transforms/          # Code transformation utilities
  inference/           # Model inference and prompts
  translation/         # Code translation between languages
  verification/        # Formal verification components
  reporting/           # Benchmark reporting
  tasks/               # Task definitions
  pipeline.py          # Main pipeline orchestration
tests/                 # Test suite
scripts/               # Utility scripts
vendor/                # Vendored dependencies (verina submodule)
```

## Development

- Python 3.10+
- Install: `pip install -e ".[dev]"`
- Run tests: `pytest`
- Format: `black .` and `ruff check .`

## Claude Code

This project includes Claude Code automations in `.claude/`:

- **`settings.json`** — Shared hooks (auto-format with black/ruff) and permissions
- **`skills/`** — `/run-benchmark` and `/new-transform` slash commands
- **`agents/`** — `benchmark-analyzer` and `security-reviewer` sub-agents
- **`.github/workflows/ci.yml`** — CI pipeline (lint, typecheck, test)

New contributors: clone the repo and Claude Code auto-discovers everything. See `CLAUDE.local.md` for personal overrides (auto-gitignored).

## Docker

Build and run with Docker Compose:
```bash
docker-compose up --build
```
