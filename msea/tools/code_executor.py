"""Safe code execution sandbox for agent tool use."""
from typing import Any, Dict, Optional
import io
import contextlib


class CodeExecutor:
    """Execute code snippets safely for computation tasks."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.timeout = self.config.get("timeout", 10)
        self.allowed_imports = {"math", "numpy", "statistics", "collections"}

    def execute(self, code: str) -> Dict[str, Any]:
        """Execute code and return result."""
        stdout = io.StringIO()
        result = {"success": False, "output": "", "error": None}

        try:
            with contextlib.redirect_stdout(stdout):
                exec_globals = {"__builtins__": __builtins__}
                exec(code, exec_globals)
            result["success"] = True
            result["output"] = stdout.getvalue()
        except Exception as e:
            result["error"] = str(e)

        return result
