"""Metacognition-specific experiments."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_benchmark import run_benchmark

if __name__ == "__main__":
    print("Running metacognition experiments...")
    run_benchmark("configs/metacognition_config.yaml")
