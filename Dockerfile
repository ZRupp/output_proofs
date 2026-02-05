FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY vendor/ vendor/

RUN pip install --no-cache-dir -e .

# Absolute imports for both output_proofs and verina
ENV PYTHONPATH="/app/src:/app/vendor/verina/src"

# Model defaults (override via docker-compose or -e)
ENV INFERENCE_MODEL="gpt2"
ENV FORMALIZER_MODEL="Goedel-LM/Goedel-Formalizer-V2-8B"

ENTRYPOINT ["python", "scripts/run_benchmark.py"]
