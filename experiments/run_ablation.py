"""Ablation study runner."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msea.utils.config import load_config, merge_configs
from experiments.run_benchmark import run_benchmark

if __name__ == "__main__":
    print("Running ablation studies...")
    config = load_config("configs/ablation_config.yaml")
    for ablation in config.get("ablations", []):
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation['name']}")
        print(f"{'='*60}")
        # Each ablation overrides the base config
        run_benchmark("configs/base_config.yaml")
