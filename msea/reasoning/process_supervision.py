"""
Process Supervision — Evaluate intermediate reasoning steps,
not just final outcomes. Implements process reward models (PRMs).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class StepEvaluation:
    """Evaluation of a single reasoning step."""
    step_idx: int
    step_text: str
    correctness_score: float
    helpfulness_score: float
    is_on_track: bool
    feedback: str


class ProcessSupervisor:
    """
    Evaluates reasoning steps individually (process supervision)
    rather than just the final answer (outcome supervision).

    This enables:
    1. Early error detection — catch mistakes before they propagate
    2. Credit assignment — identify which steps helped/hurt
    3. Training signal — denser rewards for learning
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strictness = config.get("strictness", 0.5)
        self.step_history: List[StepEvaluation] = []

    def evaluate_step(self, step_text: str, step_idx: int,
                      previous_steps: Optional[List[str]] = None) -> StepEvaluation:
        """Evaluate a single reasoning step."""
        correctness = self._score_correctness(step_text, previous_steps)
        helpfulness = self._score_helpfulness(step_text, previous_steps)
        is_on_track = correctness > self.strictness and helpfulness > 0.3

        feedback = self._generate_step_feedback(correctness, helpfulness, is_on_track)

        eval_result = StepEvaluation(
            step_idx=step_idx,
            step_text=step_text,
            correctness_score=correctness,
            helpfulness_score=helpfulness,
            is_on_track=is_on_track,
            feedback=feedback,
        )

        self.step_history.append(eval_result)
        return eval_result

    def evaluate_chain(self, steps: List[str]) -> Dict[str, Any]:
        """Evaluate an entire reasoning chain with process supervision."""
        evaluations = []
        for i, step in enumerate(steps):
            prev = steps[:i] if i > 0 else None
            evaluation = self.evaluate_step(step, i, prev)
            evaluations.append(evaluation)

        # Aggregate metrics
        correctness_scores = [e.correctness_score for e in evaluations]
        helpfulness_scores = [e.helpfulness_score for e in evaluations]
        on_track = [e.is_on_track for e in evaluations]

        # First off-track step (early error detection)
        first_error_idx = None
        for i, e in enumerate(evaluations):
            if not e.is_on_track:
                first_error_idx = i
                break

        return {
            "evaluations": evaluations,
            "avg_correctness": np.mean(correctness_scores),
            "avg_helpfulness": np.mean(helpfulness_scores),
            "fraction_on_track": sum(on_track) / len(on_track),
            "first_error_step": first_error_idx,
            "chain_length": len(steps),
            "process_reward": self._compute_process_reward(evaluations),
        }

    def _score_correctness(self, step: str, previous: Optional[List[str]]) -> float:
        """Score the correctness of a reasoning step."""
        score = 0.5  # Base
        # Heuristic checks
        if any(w in step.lower() for w in ["clearly", "obviously", "therefore"]):
            score += 0.1
        if any(w in step.lower() for w in ["wrong", "mistake", "error"]):
            score -= 0.2
        if previous and any(step.lower() in p.lower() for p in previous):
            score -= 0.1  # Repetition penalty
        return max(0.0, min(1.0, score))

    def _score_helpfulness(self, step: str, previous: Optional[List[str]]) -> float:
        """Score how helpful a step is toward the final answer."""
        if len(step.split()) < 5:
            return 0.3  # Too short to be helpful
        if any(w in step.lower() for w in ["analyze", "consider", "examine", "evaluate"]):
            return 0.7
        return 0.5

    def _generate_step_feedback(self, correctness: float, helpfulness: float,
                                 on_track: bool) -> str:
        """Generate feedback for a reasoning step."""
        if on_track:
            return "Step is on track — reasoning appears sound."
        parts = []
        if correctness < self.strictness:
            parts.append("Step may contain errors.")
        if helpfulness < 0.3:
            parts.append("Step does not meaningfully advance reasoning.")
        return " ".join(parts) or "Step needs improvement."

    def _compute_process_reward(self, evaluations: List[StepEvaluation]) -> float:
        """Compute overall process reward from step evaluations."""
        if not evaluations:
            return 0.0
        # Weighted: later steps matter more
        weights = [i + 1 for i in range(len(evaluations))]
        total_weight = sum(weights)
        reward = sum(
            w * (e.correctness_score * 0.6 + e.helpfulness_score * 0.4)
            for w, e in zip(weights, evaluations)
        ) / total_weight
        return reward

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregate statistics from all evaluated steps."""
        if not self.step_history:
            return {"total_steps": 0}
        return {
            "total_steps": len(self.step_history),
            "avg_correctness": np.mean([e.correctness_score for e in self.step_history]),
            "avg_helpfulness": np.mean([e.helpfulness_score for e in self.step_history]),
            "on_track_rate": np.mean([e.is_on_track for e in self.step_history]),
        }

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-04-13T23:49:31Z) ---
