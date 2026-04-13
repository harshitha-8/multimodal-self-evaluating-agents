"""Tests for evaluation modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msea.evaluation.metrics import (
    MetacognitionMetrics, compute_ece, compute_self_eval_accuracy, compute_brier_score
)
from msea.evaluation.benchmarks import BenchmarkLoader
from msea.evaluation.self_eval import SelfEvaluationScorer


def test_self_eval_accuracy():
    preds = [0.9, 0.8, 0.3, 0.2, 0.7]
    actuals = [True, True, False, False, True]
    acc = compute_self_eval_accuracy(preds, actuals)
    assert acc == 1.0  # All predictions match at threshold 0.5
    print("✓ test_self_eval_accuracy passed")


def test_ece():
    preds = [0.9, 0.1, 0.8, 0.2]
    actuals = [True, False, True, False]
    ece = compute_ece(preds, actuals)
    assert 0 <= ece <= 1
    print("✓ test_ece passed")


def test_brier_score():
    preds = [0.9, 0.1]
    actuals = [True, False]
    brier = compute_brier_score(preds, actuals)
    assert brier < 0.05  # Should be low for good predictions
    print("✓ test_brier_score passed")


def test_metacognition_metrics():
    metrics = MetacognitionMetrics()
    for i in range(100):
        pred = 0.7 + 0.1 * (i % 3)
        actual = i % 2 == 0
        metrics.update(pred, actual)
    result = metrics.compute_all()
    assert "self_eval_accuracy" in result
    assert "ece" in result
    assert result["num_samples"] == 100
    print("✓ test_metacognition_metrics passed")


def test_benchmark_loader():
    loader = BenchmarkLoader()
    samples = loader.load("synthetic_metacognition", num_samples=10)
    assert len(samples) == 10
    assert samples[0].sample_id.startswith("synth_")
    print("✓ test_benchmark_loader passed")


def test_self_evaluation_scorer():
    scorer = SelfEvaluationScorer()
    result = scorer.score(0.8, "cat", "cat")
    assert result["is_correct"] == 1.0
    result = scorer.score(0.9, "dog", "cat")
    assert result["is_correct"] == 0.0
    assert result["overconfident"] == 1.0
    print("✓ test_self_evaluation_scorer passed")


if __name__ == "__main__":
    test_self_eval_accuracy()
    test_ece()
    test_brier_score()
    test_metacognition_metrics()
    test_benchmark_loader()
    test_self_evaluation_scorer()
    print("\nAll evaluation tests passed! ✓")
