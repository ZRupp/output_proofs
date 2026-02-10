# Output Proofs

Adversarial benchmark for code generation with formal verification.

Takes MBPP coding tasks, generates adversarial variants via LibCST transforms (variable renaming, parameter reordering), runs code generation with a 3B model, translates to Lean 4 with an 8B formalizer, then verifies the Lean output using [Verina](https://github.com/Goedel-LM/verina). Measures how robust code generation models are to superficial input perturbations.

## Pipeline

```
MBPP Tasks ──► Adversarial Variants (LibCST) ──► Code Generation (3B)
                                                        │
                                                        ▼
                                          Lean Translation (8B) ──► Verification (Verina)
```

The pipeline runs in three sequential phases to manage GPU memory:

1. **Generate** — Load generation model, batch-generate Python for all variants, unload
2. **Translate** — Load translation model, batch-translate to Lean 4, unload
3. **Verify** — Run Lean verification (CPU-only)

Results are cached to disk by default (SHA256-keyed), so interrupted runs resume from where they left off.

## Quickstart

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd output_proofs

# Install
pip install -e ".[dev]"

# Run tests
pytest

# Dry run (no inference, just loads tasks and shows variants)
python scripts/run_benchmark.py --dry-run

# Full benchmark
python scripts/run_benchmark.py --model meta-llama/Llama-3.2-3B --variants 5
```

## Docker

Build and run with Docker Compose (defaults to ROCm 7.2 + PyTorch 2.9.1):

```bash
# GPU (AMD ROCm)
docker compose up --build benchmark

# CPU-only (for testing/dry-runs)
docker compose --profile cpu-only up --build benchmark-cpu
```

Override models and device via environment variables:

```bash
INFERENCE_MODEL=meta-llama/Llama-3.2-3B \
FORMALIZER_MODEL=Goedel-LM/Goedel-Formalizer-V2-8B \
HF_TOKEN=hf_... \
docker compose up benchmark
```

For NVIDIA GPUs, override the base image:

```bash
BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime docker compose up benchmark
```

## CLI Reference

```
python scripts/run_benchmark.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt2` | Code generation model (env: `INFERENCE_MODEL`) |
| `--formalizer` | `Goedel-LM/Goedel-Formalizer-V2-8B` | Translation model (env: `FORMALIZER_MODEL`) |
| `--variants` | `5` | Adversarial variants per task |
| `--tasks` | all | Specific MBPP task IDs to run |
| `--device` | auto-detect | `cuda`, `cpu`, or `auto` |
| `--quantize` | off | `4` or `8` bit quantization (requires `bitsandbytes`) |
| `--generation-batch-size` | `8` | Batch size for code generation |
| `--translation-batch-size` | `4` | Batch size for Lean translation |
| `--no-cache` | off | Disable disk-based result caching |
| `--clear-cache` | off | Clear cache before running |
| `--cache-dir` | `results/cache` | Custom cache directory |
| `--output` | `results/benchmark_report.json` | Report output path |
| `--use-fim` | off | Use fill-in-the-middle format |
| `--dry-run` | off | Load tasks without running inference |
| `-v` | off | Verbose logging |

## Project Structure

```
src/output_proofs/
  config.py              # All configuration dataclasses
  pipeline.py            # Phased pipeline orchestration
  cache.py               # SHA256-keyed disk cache
  inference/             # Code generation model wrapper
  translation/           # Goedel-Formalizer wrapper
  transforms/            # LibCST adversarial transforms
  verification/          # Lean verification via Verina
  reporting/             # Result aggregation and reports
  tasks/                 # MBPP + Verina task loading
scripts/
  run_benchmark.py       # CLI entrypoint
tests/                   # Test suite (pytest)
vendor/verina/           # Vendored Verina submodule
```

## Development

```bash
pip install -e ".[dev]"
pytest                   # Run tests
black .                  # Format
ruff check .             # Lint
```

## License

MIT
