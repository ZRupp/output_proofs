ARG BASE_IMAGE=rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY vendor/ vendor/

# Fail fast if verina submodule wasn't initialized before build
RUN test -f vendor/verina/src/verina/__init__.py || \
    (echo "ERROR: vendor/verina is empty. Run 'git submodule update --init' first." && exit 1)

RUN pip install --no-cache-dir -e ".[gpu]"

# tomllib is stdlib in 3.11+; install backport for 3.10 and shim it
RUN pip install --no-cache-dir tomli && \
    python -c 'import tomllib' 2>/dev/null || \
    printf 'from tomli import *\n' > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/tomllib.py"

# Install elan (Lean version manager) — reads lean-toolchain to get Lean 4 v4.18.0
RUN curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
ENV PATH="/root/.elan/bin:${PATH}"

# Resolve Verina's Lake dependencies and download prebuilt Mathlib cache.
# This avoids compiling Mathlib from source (saves ~30min).
# Note: this adds ~3-4GB to the image for Lean toolchain + Mathlib oleans.
WORKDIR /app/vendor/verina
RUN lake update && lake exe cache get
WORKDIR /app

# Absolute imports for both output_proofs and verina
ENV PYTHONPATH="/app/src:/app/vendor/verina/src"

# Model defaults (override via docker-compose or -e)
ENV INFERENCE_MODEL="gpt2"
ENV FORMALIZER_MODEL="Goedel-LM/Goedel-Formalizer-V2-8B"

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
