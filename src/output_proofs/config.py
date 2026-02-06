"""Configuration management for output_proofs.

All paths are relative or configurable via environment variables.
No hardcoded absolute paths.
"""

from pathlib import Path
from dataclasses import dataclass, field
import os
from typing import Optional


def detect_device() -> str:
    """Detect the best available compute device.

    Checks DEVICE env var first, then probes for GPU availability.
    Works with both CUDA and ROCm (torch.cuda.is_available() returns
    True for both backends).
    """
    env = os.environ.get("DEVICE")
    if env:
        return env.lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class PathConfig:
    """Path configuration - all paths relative to project root or from env vars."""

    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    verina_path: Optional[Path] = None

    def __post_init__(self):
        # Use env var if set, otherwise use vendor submodule
        verina_env = os.environ.get("VERINA_PATH")
        if verina_env:
            self.verina_path = Path(verina_env)
        else:
            self.verina_path = self.project_root / "vendor" / "verina" / "src"

    def validate(self) -> None:
        """Validate that required paths exist."""
        if not self.verina_path.exists():
            raise FileNotFoundError(
                f"Verina not found at {self.verina_path}. "
                "Either run 'git submodule update --init' or set VERINA_PATH env var."
            )

    @property
    def results_dir(self) -> Path:
        """Directory for benchmark results."""
        path = self.project_root / "results"
        path.mkdir(exist_ok=True)
        return path


@dataclass
class ModelConfig:
    """Configuration for the code generation model.

    Model name can be overridden via INFERENCE_MODEL environment variable.
    """

    model_name: str = field(default_factory=lambda: os.environ.get("INFERENCE_MODEL", "gpt2"))
    device: str = field(default_factory=detect_device)
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class FormalizerConfig:
    """Configuration for the Goedel-Formalizer translation model.

    Model name can be overridden via FORMALIZER_MODEL environment variable.
    """

    model_name: str = field(
        default_factory=lambda: os.environ.get(
            "FORMALIZER_MODEL", "Goedel-LM/Goedel-Formalizer-V2-8B"
        )
    )
    device: str = field(default_factory=detect_device)
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = True


@dataclass
class TransformConfig:
    """Configuration for LibCST transformations."""

    # V_noise variable names for obfuscation
    v_noise_names: tuple = (
        "__tmp0",
        "__tmp1",
        "__tmp2",
        "var_x",
        "var_y",
        "var_z",
        "_arg0",
        "_arg1",
        "_arg2",
        "_v0",
        "_v1",
        "_v2",
    )

    # Number of variants to generate per task
    num_variants: int = 5

    # Enable specific transforms
    enable_variable_rename: bool = True
    enable_param_reorder: bool = True


@dataclass
class PipelineConfig:
    """Configuration for the full benchmark pipeline."""

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    formalizer: FormalizerConfig = field(default_factory=FormalizerConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)

    # Pipeline settings
    num_variants_per_task: int = 5
    batch_size: int = 4

    @property
    def output_dir(self) -> Path:
        """Output directory for results."""
        return self.paths.results_dir
