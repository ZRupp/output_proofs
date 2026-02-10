"""Data schemas for MBPP tasks and Verina specifications."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MBPPTask:
    """Python task from MBPP dataset."""

    task_id: int
    text: str  # Task description
    code: str  # Reference Python solution
    test_list: List[str]  # Python test assertions
    test_setup_code: str = ""  # Setup code for tests
    challenge_test_list: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MBPPTask":
        """Create MBPPTask from dataset row."""
        return cls(
            task_id=data["task_id"],
            text=data["text"],
            code=data["code"],
            test_list=data.get("test_list", []),
            test_setup_code=data.get("test_setup_code", ""),
            challenge_test_list=data.get("challenge_test_list", []),
        )


@dataclass
class VerinaSpec:
    """Lean specification from Verina dataset."""

    data_id: str  # e.g., "verina_basic_46"
    lean_code: str  # Annotated Lean source (with !benchmark markers)
    signature: Dict[str, Any]  # Function signature dict {name, parameters, return_type}
    description: str  # Task description
    tests: List[Dict[str, Any]] = field(default_factory=list)  # Test cases from HF
    reject_inputs: List[Dict[str, Any]] = field(default_factory=list)  # Reject inputs
    metadata: Dict[str, Any] = field(default_factory=dict)  # Upstream info
    difficulty: str = "basic"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerinaSpec":
        """Create VerinaSpec from HuggingFace dataset row."""
        return cls(
            data_id=data.get("id", data.get("data_id", "")),
            lean_code=data.get("lean_code", ""),
            signature=data.get("signature", {}),
            description=data.get("description", ""),
            tests=data.get("tests", []),
            reject_inputs=data.get("reject_inputs", []),
            metadata=data.get("metadata", {}),
            difficulty=data.get("difficulty", "basic"),
        )

    @property
    def func_name(self) -> str:
        """Extract function name from signature."""
        if isinstance(self.signature, dict):
            return self.signature.get("name", "")
        return ""

    @property
    def mbpp_task_id(self) -> Optional[int]:
        """Extract MBPP task ID from metadata if available."""
        upstream = self.metadata.get("upstream", {})
        task_id_str = upstream.get("task_id", "")
        if task_id_str and task_id_str.startswith("task_id_"):
            try:
                return int(task_id_str.replace("task_id_", ""))
            except ValueError:
                pass
        return None


@dataclass
class CombinedTask:
    """Combined MBPP Python task + Verina Lean spec."""

    mbpp: MBPPTask
    verina: VerinaSpec
    task_id: int  # Shared identifier

    @property
    def description(self) -> str:
        """Get task description from MBPP."""
        return self.mbpp.text

    @property
    def reference_python(self) -> str:
        """Get reference Python solution."""
        return self.mbpp.code

    @property
    def reference_lean(self) -> str:
        """Get reference Lean implementation."""
        return self.verina.lean_code


@dataclass
class TaskVariant:
    """An adversarial variant of a task."""

    original_task: CombinedTask
    variant_id: str
    transformed_description: str  # Modified description with V_noise
    transformed_signature: str  # Function signature with noisy params
    func_name: str  # Function name
    noisy_params: str  # Parameter string with noisy names
    transforms_applied: List[str]
    param_mapping: Dict[str, str]  # original_name -> noisy_name
    reverse_mapping: Dict[str, str]  # noisy_name -> original_name

    @property
    def task_id(self) -> int:
        """Get original task ID."""
        return self.original_task.task_id
