ARG BASE_IMAGE=rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.9.1
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

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

# Absolute imports for both output_proofs and verina
ENV PYTHONPATH="/app/src:/app/vendor/verina/src"

# Model defaults (override via docker-compose or -e)
ENV INFERENCE_MODEL="gpt2"
ENV FORMALIZER_MODEL="Goedel-LM/Goedel-Formalizer-V2-8B"

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
