"""
Autonomous Research Loop — Karpathy-style auto-experimentation.
Reads program.md, runs experiments, keeps/discards, repeats forever.

Usage: python run_agent.py --program program.md
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="Run autonomous research agent")
    parser.add_argument("--program", type=str, default="program.md",
                       help="Path to agent program file")
    parser.add_argument("--max-experiments", type=int, default=-1,
                       help="Max experiments to run (-1 = infinite)")
    args = parser.parse_args()

    print("=" * 60)
    print("MULTIMODAL SELF-EVALUATING AGENTS — Autonomous Research")
    print("=" * 60)
    print(f"Program: {args.program}")
    print(f"Max experiments: {'infinite' if args.max_experiments < 0 else args.max_experiments}")
    print()

    # Read program
    if os.path.exists(args.program):
        with open(args.program, "r") as f:
            program = f.read()
        print(f"Loaded program ({len(program)} chars)")
    else:
        print(f"Program file not found: {args.program}")
        sys.exit(1)

    # Initialize results
    os.makedirs("results", exist_ok=True)
    results_path = "results/results.tsv"
    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("commit\tself_eval_acc\treasoning_q\ttool_f1\tstatus\tdescription\n")

    print("\nReady to begin autonomous experimentation.")
    print("The agent will modify msea/, run benchmarks, and iterate.")
    print("Press Ctrl+C to stop.\n")

    experiment_count = 0
    while args.max_experiments < 0 or experiment_count < args.max_experiments:
        experiment_count += 1
        print(f"\n--- Experiment {experiment_count} ---")

        # Import and run benchmark
        from experiments.run_benchmark import run_benchmark
        try:
            metrics = run_benchmark("configs/base_config.yaml")
            self_eval = metrics.get("self_eval_accuracy", 0)
            print(f"\nResult: self_eval_accuracy = {self_eval:.6f}")
        except Exception as e:
            print(f"Experiment crashed: {e}")

        # In autonomous mode, the agent would modify code here
        # For now, just demonstrate the loop
        if experiment_count >= 3:
            print("\nDemo mode: stopping after 3 experiments.")
            print("In production, the agent would modify msea/ and continue forever.")
            break

    print(f"\nCompleted {experiment_count} experiments.")


if __name__ == "__main__":
    main()
