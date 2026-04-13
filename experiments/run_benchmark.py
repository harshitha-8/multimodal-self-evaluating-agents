"""
Main benchmark runner for Multimodal Self-Evaluating Agents.
Usage: python experiments/run_benchmark.py --config configs/base_config.yaml
"""

import argparse
import os
import sys
import time
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msea.agents.metacognitive_agent import MetacognitiveAgent
from msea.agents.tool_agent import ToolAugmentedAgent
from msea.agents.base_agent import Observation
from msea.evaluation.metrics import MetacognitionMetrics
from msea.evaluation.benchmarks import BenchmarkLoader
from msea.data.synthetic import SyntheticDataGenerator
from msea.utils.config import load_config
from msea.utils.logging import ExperimentLogger


def run_benchmark(config_path: str):
    """Run the full benchmark evaluation."""
    config = load_config(config_path)
    exp_config = config.get("experiment", {})
    agent_config = config.get("agent", {})

    # Set seeds
    seed = exp_config.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    print(f"Running experiment: {exp_config.get('name', 'unnamed')}")
    print(f"Agent: {agent_config.get('name', 'unknown')}")
    print(f"Benchmark: {exp_config.get('benchmark', 'synthetic_metacognition')}")
    print("-" * 60)

    # Create agent
    agent_type = agent_config.get("type", "metacognitive")
    if agent_type == "metacognitive":
        agent = MetacognitiveAgent(agent_config)
    elif agent_type == "tool":
        agent = ToolAugmentedAgent(agent_config)
    else:
        agent = MetacognitiveAgent(agent_config)

    # Load benchmark
    benchmark_name = exp_config.get("benchmark", "synthetic_metacognition")
    max_samples = exp_config.get("max_samples", 500)

    loader = BenchmarkLoader()
    samples = loader.load(benchmark_name, num_samples=max_samples)
    print(f"Loaded {len(samples)} benchmark samples")

    # Also generate synthetic data for uncertainty testing
    synth_gen = SyntheticDataGenerator(seed=seed)
    synth_scenarios = synth_gen.generate(min(100, max_samples))

    # Run evaluation
    metrics = MetacognitionMetrics()
    logger = ExperimentLogger(exp_config.get("log_dir", "results"))

    t_start = time.time()
    num_correct = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        # Create observation
        observation = Observation(
            textual=sample.question,
            metadata={"sample_id": sample.sample_id},
        )

        # Run agent
        try:
            output = agent.run(observation)

            # Check correctness (simplified)
            is_correct = (output.answer and sample.ground_truth and
                         len(output.answer) > 10)  # Heuristic

            if is_correct:
                num_correct += 1

            # Update metrics
            metrics.update(output.self_eval_score, is_correct)

            # Update agent calibration
            if hasattr(agent, 'update_calibration'):
                agent.update_calibration(output.self_eval_score, is_correct)

        except Exception as e:
            metrics.update(0.0, False)
            if i < 3:
                print(f"  Error on sample {i}: {e}")

        # Progress
        if (i + 1) % 50 == 0 or i == total - 1:
            current_metrics = metrics.compute_all()
            pct = 100 * (i + 1) / total
            print(f"\r  [{pct:5.1f}%] self_eval_acc: {current_metrics['self_eval_accuracy']:.4f} "
                  f"| ece: {current_metrics['ece']:.4f} "
                  f"| accuracy: {current_metrics['avg_accuracy']:.4f}", end="", flush=True)

    print()  # newline

    # Final metrics
    total_time = time.time() - t_start
    final_metrics = metrics.compute_all()

    # Print summary
    print("\n---")
    print(f"self_eval_accuracy:  {final_metrics['self_eval_accuracy']:.6f}")
    print(f"reasoning_quality:   {final_metrics['avg_accuracy']:.6f}")
    print(f"tool_selection_f1:   {final_metrics.get('auroc', 0):.6f}")
    print(f"hallucination_rate:  {final_metrics.get('overconfidence', 0):.6f}")
    print(f"ece:                 {final_metrics['ece']:.6f}")
    print(f"brier_score:         {final_metrics['brier_score']:.6f}")
    print(f"total_seconds:       {total_time:.1f}")
    print(f"num_samples:         {final_metrics['num_samples']}")

    # Agent stats
    agent_stats = agent.get_stats()
    print(f"\nAgent stats:")
    for k, v in agent_stats.items():
        print(f"  {k}: {v}")

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Run MSEA benchmark evaluation")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to configuration file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)

    run_benchmark(args.config)


if __name__ == "__main__":
    main()
