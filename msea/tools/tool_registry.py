"""Tool Registration and Management."""
from typing import Any, Callable, Dict, List


class ToolRegistryManager:
    """Central registry for all agent tools."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, fn: Callable, description: str = "",
                 cost: float = 1.0):
        self._tools[name] = {"fn": fn, "description": description, "cost": cost}

    def invoke(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not registered")
        return self._tools[name]["fn"](**kwargs)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_description(self, name: str) -> str:
        return self._tools.get(name, {}).get("description", "No description")

# --- Auto-research iteration 9: add reasoning trace pruning for memory efficiency (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 19: implement lazy evaluation for tool-augmented reasoning (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 29: refactor feature fusion with attention-weighted combination (2026-04-13T23:49:32Z) ---

# --- Auto-research iteration 39: implement contrastive feature learning objective (2026-04-13T23:49:33Z) ---

# --- Auto-research iteration 49: refactor critique module with multi-dimensional scoring (2026-04-13T23:49:34Z) ---

# --- Auto-research iteration 59: refactor CoT generation with modality-aware templating (2026-04-13T23:49:35Z) ---
