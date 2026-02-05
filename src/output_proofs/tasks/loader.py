"""Task loading from MBPP and Verina datasets."""

from typing import List, Dict, Optional
from datasets import load_dataset
import re

from .schema import MBPPTask, VerinaSpec, CombinedTask


# 49 MBPP task_ids that have Verina Lean specifications
# Note: task_id 644 is excluded - it exists in MBPP but was not translated to Lean in Verina
MBPP_TASK_IDS = [
    807, 760, 755, 743, 732, 616, 741, 474, 733,
    632, 629, 627, 625, 624, 610, 605, 602, 600, 599,
    594, 588, 579, 576, 803, 573, 567, 566, 477, 472,
    798, 454, 447, 441, 435, 433, 793, 431, 404, 784,
    267, 775, 227, 127, 770, 101, 77, 764, 62, 58
]


def _extract_function_name(code: str) -> Optional[str]:
    """Extract function name from Python code."""
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None


def _build_verina_mapping(verina_dataset) -> Dict[str, VerinaSpec]:
    """Build mapping from task descriptions to Verina specs.

    Since Verina uses data_id like 'verina_basic_1' and MBPP uses task_id,
    we need to match them based on task content.
    """
    verina_specs = {}

    for row in verina_dataset:
        data_id = row.get("data_id", "")
        if "verina_basic" not in data_id:
            continue

        spec = VerinaSpec.from_dict(row)
        verina_specs[data_id] = spec

    return verina_specs


def _match_mbpp_to_verina(
    mbpp_task: MBPPTask,
    verina_specs: Dict[str, VerinaSpec],
    task_id_to_verina: Dict[int, str],
) -> Optional[VerinaSpec]:
    """Match an MBPP task to its Verina specification.

    Uses pre-computed mapping if available, otherwise attempts heuristic matching.
    """
    # Check pre-computed mapping first
    if mbpp_task.task_id in task_id_to_verina:
        verina_id = task_id_to_verina[mbpp_task.task_id]
        if verina_id in verina_specs:
            return verina_specs[verina_id]

    # Fallback: try to match by index (verina_basic_N corresponds to Nth MBPP task)
    # This requires knowledge of the ordering in Verina
    return None


# Pre-computed mapping from MBPP task_id to Verina data_id
# This mapping was derived from analyzing the Verina dataset
# The verina_basic tasks are ordered sequentially and correspond to MBPP tasks
MBPP_TO_VERINA_MAPPING: Dict[int, str] = {
    # This mapping needs to be populated based on actual Verina dataset analysis
    # For now, we use a heuristic based on task order
}


def load_mbpp_tasks() -> Dict[int, MBPPTask]:
    """Load MBPP tasks filtered to those with Verina specs."""
    # Try google-research-datasets/mbpp first (standard format)
    # Fall back to Muennighoff/mbpp with trust_remote_code if needed
    try:
        mbpp_dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
    except Exception:
        mbpp_dataset = load_dataset("Muennighoff/mbpp", split="test", trust_remote_code=True)

    task_ids_set = set(MBPP_TASK_IDS)
    mbpp_by_id = {}

    for row in mbpp_dataset:
        task_id = row["task_id"]
        if task_id in task_ids_set:
            mbpp_by_id[task_id] = MBPPTask.from_dict(row)

    return mbpp_by_id


def load_verina_specs() -> Dict[int, VerinaSpec]:
    """Load Verina specifications for the MBPP-derived problems.

    Returns a dict mapping MBPP task_id -> VerinaSpec.
    """
    verina_dataset = load_dataset("sunblaze-ucb/verina", split="train")

    # Map by MBPP task ID extracted from metadata
    verina_by_mbpp_id = {}
    for row in verina_dataset:
        data_id = row.get("id", "")
        if "verina_basic" not in data_id:
            continue

        # Check if this is from dafny-synthesis (MBPP-derived)
        metadata = row.get("metadata", {})
        upstream = metadata.get("upstream", {})
        if upstream.get("name") != "dafny-synthesis":
            continue

        # Extract MBPP task ID from metadata
        task_id_str = upstream.get("task_id", "")
        if task_id_str and task_id_str.startswith("task_id_"):
            try:
                mbpp_task_id = int(task_id_str.replace("task_id_", ""))
                verina_by_mbpp_id[mbpp_task_id] = VerinaSpec.from_dict(row)
            except ValueError:
                continue

    return verina_by_mbpp_id


def build_task_mapping(
    mbpp_tasks: Dict[int, MBPPTask],
    verina_specs: Dict[int, VerinaSpec],
) -> Dict[int, str]:
    """Build mapping from MBPP task_id to Verina data_id.

    Since verina_specs is now keyed by MBPP task_id, this just extracts data_ids.
    """
    mapping = {}
    for task_id in mbpp_tasks:
        if task_id in verina_specs:
            mapping[task_id] = verina_specs[task_id].data_id
    return mapping


def load_combined_tasks() -> List[CombinedTask]:
    """Load both MBPP Python tasks and Verina Lean specs, mapped by task_id."""
    # Load both datasets
    mbpp_tasks = load_mbpp_tasks()
    verina_specs = load_verina_specs()  # Now keyed by MBPP task_id

    # Combine tasks - verina_specs is already keyed by MBPP task_id
    combined = []
    for task_id, mbpp_task in mbpp_tasks.items():
        if task_id in verina_specs:
            combined.append(CombinedTask(
                mbpp=mbpp_task,
                verina=verina_specs[task_id],
                task_id=task_id,
            ))

    return combined


def load_mbpp_only() -> List[MBPPTask]:
    """Load just the MBPP tasks (for testing without Verina)."""
    mbpp_tasks = load_mbpp_tasks()
    return list(mbpp_tasks.values())
