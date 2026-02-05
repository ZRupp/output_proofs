"""Task loading and variant generation."""

from .schema import MBPPTask, VerinaSpec, CombinedTask, TaskVariant

# Lazy imports for modules requiring external dependencies
def load_combined_tasks():
    """Load combined tasks from MBPP and Verina datasets."""
    from .loader import load_combined_tasks as _load
    return _load()

def get_mbpp_task_ids():
    """Get the list of MBPP task IDs."""
    from .loader import MBPP_TASK_IDS
    return MBPP_TASK_IDS

# For backwards compatibility
try:
    from .loader import MBPP_TASK_IDS
except ImportError:
    MBPP_TASK_IDS = []  # Will be populated when datasets is available

__all__ = [
    "MBPPTask",
    "VerinaSpec",
    "CombinedTask",
    "TaskVariant",
    "load_combined_tasks",
    "get_mbpp_task_ids",
    "MBPP_TASK_IDS",
]
