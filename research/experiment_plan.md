# Low-Cost Experiment Plan

## Experiment 1: Baseline Self-Evaluation Accuracy (Day 1)
**Goal**: Establish baseline metacognitive accuracy on synthetic benchmark.
**Setup**: Run MetacognitiveAgent on 500 synthetic samples with default config.
**Compute**: CPU only, <5 min.
**Command**: `python experiments/run_benchmark.py --config configs/base_config.yaml`
**Expected output**: self_eval_accuracy ~ 0.5-0.6 (random baseline)

## Experiment 2: Reflection Token Ablation (Day 1-2)
**Goal**: Measure the impact of reflection tokens on self-evaluation quality.
**Setup**: Compare agent with/without ReflectionToken detection.
**Compute**: CPU only, ~10 min total.
**Variation**: Disable reflection by setting `reflection_frequency: 999`.

## Experiment 3: Multi-Encoder Uncertainty (Day 2-3)
**Goal**: Test if ensemble disagreement between CLIP and DINOv2 predicts agent errors.
**Setup**: Extract features from both encoders, compute uncertainty, correlate with errors.
**Compute**: Single GPU, <30 min for 1K images.
**Dataset**: Use CIFAR-10 or a small subset of ImageNet.

## Experiment 4: Calibration Improvement Loop (Day 3-5)
**Goal**: Test if calibration updates improve self-evaluation over time.
**Setup**: Run 500 samples sequentially, updating calibration after each.
**Metric**: Plot ECE over time (should decrease).
**Compute**: CPU only, ~15 min.

## Experiment 5: Process Supervision vs. Outcome Supervision (Day 5-7)
**Goal**: Compare step-level vs. final-answer evaluation.
**Setup**: Evaluate same reasoning chains with ProcessSupervisor and final accuracy.
**Metric**: Correlation between process reward and outcome.
**Compute**: CPU only.

## Experiment 6: Tool Selection Under Budget Constraints (Day 7-10)
**Goal**: Test ToolAugmentedAgent with varying tool budgets.
**Setup**: Run with tool_budget = [0, 1, 3, 5, 10].
**Metric**: accuracy vs. tool cost Pareto curve.
**Compute**: CPU only.

## Compute Requirements Summary
| Experiment | GPU | Time | Dataset |
|-----------|-----|------|---------|
| Baseline | No | 5 min | Synthetic |
| Reflection | No | 10 min | Synthetic |
| Multi-encoder | 1× GPU | 30 min | CIFAR-10 |
| Calibration | No | 15 min | Synthetic |
| Process sup. | No | 10 min | Synthetic |
| Tool budget | No | 20 min | Synthetic |
| **Total** | **<1 GPU-hour** | **~90 min** | **Open datasets** |
