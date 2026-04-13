"""Experiment logging utilities."""
import json
import os
import time
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Log experiment results to TSV and JSON."""

    def __init__(self, log_dir: str = "results"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.tsv_path = os.path.join(log_dir, "results.tsv")
        self._init_tsv()

    def _init_tsv(self):
        if not os.path.exists(self.tsv_path):
            with open(self.tsv_path, "w") as f:
                f.write("commit\tself_eval_acc\treasoning_q\ttool_f1\tstatus\tdescription\n")

    def log(self, commit: str, metrics: Dict[str, float], status: str, description: str):
        """Log a single experiment result."""
        with open(self.tsv_path, "a") as f:
            f.write(f"{commit}\t{metrics.get('self_eval_accuracy', 0):.6f}\t"
                   f"{metrics.get('reasoning_quality', 0):.6f}\t"
                   f"{metrics.get('tool_f1', 0):.6f}\t"
                   f"{status}\t{description}\n")

    def log_json(self, experiment_id: str, data: Dict):
        """Log detailed experiment data as JSON."""
        path = os.path.join(self.log_dir, f"{experiment_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_best_result(self) -> Optional[Dict]:
        """Get the best result from the TSV log."""
        if not os.path.exists(self.tsv_path):
            return None
        best = None
        with open(self.tsv_path, "r") as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) >= 5 and parts[4] == "keep":
                    score = float(parts[1])
                    if best is None or score > best["self_eval_acc"]:
                        best = {"commit": parts[0], "self_eval_acc": score}
        return best
