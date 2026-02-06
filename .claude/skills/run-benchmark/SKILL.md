# Run Benchmark

Run the output_proofs benchmark pipeline with common presets.

## Usage

Invoke with `/run-benchmark` followed by a preset name or custom options.

## Presets

### Dry Run (no GPU needed)
```bash
python scripts/run_benchmark.py --dry-run --device cpu -v
```

### Quick Test (single task, CPU)
```bash
python scripts/run_benchmark.py --tasks 101 --variants 1 --device cpu -v
```

### Full Benchmark (GPU)
```bash
python scripts/run_benchmark.py --variants 5 --device auto
```

### FIM Mode
```bash
python scripts/run_benchmark.py --use-fim --variants 5 --device auto
```

### Custom Model
```bash
INFERENCE_MODEL=codellama/CodeLlama-7b-hf python scripts/run_benchmark.py --variants 3
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `INFERENCE_MODEL` | Code generation model | `gpt2` |
| `FORMALIZER_MODEL` | Python-to-Lean model | `Goedel-LM/Goedel-Formalizer-V2-8B` |
| `DEVICE` | Compute device (`cuda`, `cpu`, `auto`) | auto-detect |
| `HF_TOKEN` | Hugging Face auth token | (none) |
| `VERINA_PATH` | Path to Verina benchmark data | `vendor/verina/src` |

## Docker

```bash
# GPU (CUDA)
docker-compose up benchmark

# GPU (ROCm)
BASE_IMAGE=rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.2 docker-compose up benchmark

# CPU only
docker-compose up benchmark-cpu
```

## Instructions

1. Ask the user which preset they want, or if they have custom requirements
2. Set environment variables if needed (model overrides, HF_TOKEN)
3. Run the appropriate command
4. After completion, check `./results/` for the JSON report
