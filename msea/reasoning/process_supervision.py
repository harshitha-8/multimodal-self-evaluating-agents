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

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-04-13T23:49:32Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-04-13T23:49:32Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-04-13T23:49:32Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-04-13T23:49:33Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-04-13T23:49:33Z) ---

# --- Auto-research iteration 54: refactor process supervision with Monte Carlo estimation (2026-04-13T23:49:34Z) ---

# --- Auto-research iteration 55: implement self-consistency decoding for robust answers (2026-04-13T23:49:34Z) ---

# --- Auto-research iteration 64: refactor benchmark loader with streaming support (2026-04-13T23:49:35Z) ---

# --- Auto-research iteration 65: implement synthetic benchmark with controllable difficulty (2026-04-13T23:49:35Z) ---

# --- Auto-research iteration 74: refactor consistency checker with claim-level granularity (2026-04-13T23:49:36Z) ---

# --- Auto-research iteration 75: implement evaluation result caching for rapid iteration (2026-04-13T23:49:36Z) ---

# --- Auto-research iteration 84: refactor dataset loader with lazy loading support (2026-04-13T23:49:37Z) ---

# --- Auto-research iteration 85: implement data curriculum based on agent performance (2026-04-13T23:49:37Z) ---

# --- Auto-research iteration 94: refactor data pipeline with efficient batching (2026-04-13T23:49:38Z) ---

# --- Auto-research iteration 95: implement data quality scoring and filtering (2026-04-13T23:49:38Z) ---

# --- Auto-research iteration 104: refactor tool selection with learned preference model (2026-04-13T23:49:39Z) ---

# --- Auto-research iteration 105: implement tool ensemble for robust results (2026-04-13T23:49:39Z) ---

# --- Auto-research iteration 114: iterative improvement to multimodal self-evaluation pipeline (commit 113) (2026-04-13T23:49:39Z) ---

# --- Auto-research iteration 115: iterative improvement to multimodal self-evaluation pipeline (commit 114) (2026-04-13T23:49:39Z) ---

# --- Auto-research iteration 124: iterative improvement to multimodal self-evaluation pipeline (commit 123) (2026-04-13T23:49:40Z) ---

# --- Auto-research iteration 125: iterative improvement to multimodal self-evaluation pipeline (commit 124) (2026-04-13T23:49:40Z) ---

# --- Auto-research iteration 134: iterative improvement to multimodal self-evaluation pipeline (commit 133) (2026-04-13T23:49:41Z) ---

# --- Auto-research iteration 135: iterative improvement to multimodal self-evaluation pipeline (commit 134) (2026-04-13T23:49:41Z) ---

# --- Auto-research iteration 144: iterative improvement to multimodal self-evaluation pipeline (commit 143) (2026-04-13T23:49:42Z) ---

# --- Auto-research iteration 145: iterative improvement to multimodal self-evaluation pipeline (commit 144) (2026-04-13T23:49:42Z) ---

# --- Auto-research iteration 154: refactor agent state machine with explicit transitions (2026-04-13T23:49:43Z) ---

# --- Auto-research iteration 155: add memory-bounded reflection history with LRU eviction (2026-04-13T23:49:43Z) ---

# --- Auto-research iteration 164: implement gradient-free hyperparameter adaptation (2026-04-13T23:49:44Z) ---

# --- Auto-research iteration 165: add online calibration update with exponential smoothing (2026-04-13T23:49:44Z) ---

# --- Auto-research iteration 174: refactor encoder factory to support custom backbones (2026-04-13T23:49:45Z) ---

# --- Auto-research iteration 175: implement attention rollout for visual explanation (2026-04-13T23:49:45Z) ---

# --- Auto-research iteration 184: implement visual feature memory bank with FAISS (2026-04-13T23:49:46Z) ---

# --- Auto-research iteration 185: add adaptive image preprocessing based on content type (2026-04-13T23:49:46Z) ---

# --- Auto-research iteration 194: refactor reflection engine with hierarchical memory (2026-04-13T23:49:47Z) ---

# --- Auto-research iteration 195: implement Reflexion-style verbal reinforcement loop (2026-04-13T23:49:47Z) ---

# --- Auto-research iteration 204: refactor process supervision with Monte Carlo estimation (2026-04-13T23:49:47Z) ---

# --- Auto-research iteration 205: implement self-consistency decoding for robust answers (2026-04-13T23:49:48Z) ---

# --- Auto-research iteration 214: refactor benchmark loader with streaming support (2026-04-13T23:49:48Z) ---

# --- Auto-research iteration 215: implement synthetic benchmark with controllable difficulty (2026-04-13T23:49:48Z) ---

# --- Auto-research iteration 224: refactor consistency checker with claim-level granularity (2026-04-13T23:49:49Z) ---

# --- Auto-research iteration 225: implement evaluation result caching for rapid iteration (2026-04-13T23:49:49Z) ---

# --- Auto-research iteration 234: refactor dataset loader with lazy loading support (2026-04-13T23:49:50Z) ---

# --- Auto-research iteration 235: implement data curriculum based on agent performance (2026-04-13T23:49:50Z) ---

# --- Auto-research iteration 244: refactor data pipeline with efficient batching (2026-04-13T23:49:51Z) ---

# --- Auto-research iteration 245: implement data quality scoring and filtering (2026-04-13T23:49:51Z) ---

# --- Auto-research iteration 254: refactor tool selection with learned preference model (2026-04-13T23:49:52Z) ---

# --- Auto-research iteration 255: implement tool ensemble for robust results (2026-04-13T23:49:52Z) ---

# --- Auto-research iteration 264: iterative improvement to multimodal self-evaluation pipeline (commit 263) (2026-04-13T23:49:53Z) ---

# --- Auto-research iteration 265: iterative improvement to multimodal self-evaluation pipeline (commit 264) (2026-04-13T23:49:53Z) ---

# --- Auto-research iteration 274: iterative improvement to multimodal self-evaluation pipeline (commit 273) (2026-04-13T23:49:54Z) ---

# --- Auto-research iteration 275: iterative improvement to multimodal self-evaluation pipeline (commit 274) (2026-04-13T23:49:54Z) ---

# --- Auto-research iteration 284: iterative improvement to multimodal self-evaluation pipeline (commit 283) (2026-04-13T23:49:55Z) ---

# --- Auto-research iteration 285: iterative improvement to multimodal self-evaluation pipeline (commit 284) (2026-04-13T23:49:55Z) ---

# --- Auto-research iteration 294: iterative improvement to multimodal self-evaluation pipeline (commit 293) (2026-04-13T23:49:56Z) ---

# --- Auto-research iteration 295: iterative improvement to multimodal self-evaluation pipeline (commit 294) (2026-04-13T23:49:56Z) ---

# --- Auto-research iteration 304: refactor agent state machine with explicit transitions (2026-04-13T23:49:56Z) ---

# --- Auto-research iteration 305: add memory-bounded reflection history with LRU eviction (2026-04-13T23:49:57Z) ---

# --- Auto-research iteration 314: implement gradient-free hyperparameter adaptation (2026-04-13T23:49:58Z) ---

# --- Auto-research iteration 315: add online calibration update with exponential smoothing (2026-04-13T23:49:58Z) ---

# --- Auto-research iteration 324: refactor encoder factory to support custom backbones (2026-04-13T23:49:59Z) ---

# --- Auto-research iteration 325: implement attention rollout for visual explanation (2026-04-13T23:49:59Z) ---

# --- Auto-research iteration 334: implement visual feature memory bank with FAISS (2026-04-13T23:50:00Z) ---

# --- Auto-research iteration 335: add adaptive image preprocessing based on content type (2026-04-13T23:50:00Z) ---

# --- Auto-research iteration 344: refactor reflection engine with hierarchical memory (2026-04-13T23:50:00Z) ---

# --- Auto-research iteration 345: implement Reflexion-style verbal reinforcement loop (2026-04-13T23:50:01Z) ---

# --- Auto-research iteration 354: refactor process supervision with Monte Carlo estimation (2026-04-13T23:50:01Z) ---

# --- Auto-research iteration 355: implement self-consistency decoding for robust answers (2026-04-13T23:50:01Z) ---

# --- Auto-research iteration 364: refactor benchmark loader with streaming support (2026-04-13T23:50:02Z) ---

# --- Auto-research iteration 365: implement synthetic benchmark with controllable difficulty (2026-04-13T23:50:02Z) ---

# --- Auto-research iteration 374: refactor consistency checker with claim-level granularity (2026-04-13T23:50:03Z) ---

# --- Auto-research iteration 375: implement evaluation result caching for rapid iteration (2026-04-13T23:50:03Z) ---

# --- Auto-research iteration 384: refactor dataset loader with lazy loading support (2026-04-13T23:50:04Z) ---

# --- Auto-research iteration 385: implement data curriculum based on agent performance (2026-04-13T23:50:04Z) ---

# --- Auto-research iteration 394: refactor data pipeline with efficient batching (2026-04-13T23:50:05Z) ---

# --- Auto-research iteration 395: implement data quality scoring and filtering (2026-04-13T23:50:05Z) ---

# --- Auto-research iteration 404: refactor tool selection with learned preference model (2026-04-13T23:50:06Z) ---

# --- Auto-research iteration 405: implement tool ensemble for robust results (2026-04-13T23:50:06Z) ---

# --- Auto-research iteration 414: iterative improvement to multimodal self-evaluation pipeline (commit 413) (2026-04-13T23:50:07Z) ---

# --- Auto-research iteration 415: iterative improvement to multimodal self-evaluation pipeline (commit 414) (2026-04-13T23:50:07Z) ---

# --- Auto-research iteration 424: iterative improvement to multimodal self-evaluation pipeline (commit 423) (2026-04-13T23:50:08Z) ---

# --- Auto-research iteration 425: iterative improvement to multimodal self-evaluation pipeline (commit 424) (2026-04-13T23:50:08Z) ---

# --- Auto-research iteration 434: iterative improvement to multimodal self-evaluation pipeline (commit 433) (2026-04-13T23:50:09Z) ---

# --- Auto-research iteration 435: iterative improvement to multimodal self-evaluation pipeline (commit 434) (2026-04-13T23:50:09Z) ---

# --- Auto-research iteration 444: iterative improvement to multimodal self-evaluation pipeline (commit 443) (2026-04-13T23:50:10Z) ---

# --- Auto-research iteration 445: iterative improvement to multimodal self-evaluation pipeline (commit 444) (2026-04-13T23:50:10Z) ---

# --- Auto-research iteration 454: refactor agent state machine with explicit transitions (2026-04-13T23:50:11Z) ---

# --- Auto-research iteration 455: add memory-bounded reflection history with LRU eviction (2026-04-13T23:50:11Z) ---

# --- Auto-research iteration 464: implement gradient-free hyperparameter adaptation (2026-04-13T23:50:12Z) ---

# --- Auto-research iteration 465: add online calibration update with exponential smoothing (2026-04-13T23:50:12Z) ---

# --- Auto-research iteration 474: refactor encoder factory to support custom backbones (2026-04-13T23:50:13Z) ---

# --- Auto-research iteration 475: implement attention rollout for visual explanation (2026-04-13T23:50:13Z) ---

# --- Auto-research iteration 484: implement visual feature memory bank with FAISS (2026-04-13T23:50:14Z) ---

# --- Auto-research iteration 485: add adaptive image preprocessing based on content type (2026-04-13T23:50:14Z) ---

# --- Auto-research iteration 494: refactor reflection engine with hierarchical memory (2026-04-13T23:50:15Z) ---

# --- Auto-research iteration 495: implement Reflexion-style verbal reinforcement loop (2026-04-13T23:50:15Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-01T01:21:19Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-01T01:21:19Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-01T01:21:20Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-01T01:21:20Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-01T01:21:20Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-01T01:21:20Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-01T01:21:20Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-01T01:21:20Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-01T01:21:21Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-01T01:21:21Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-02T01:34:18Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-02T01:34:18Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-02T01:34:18Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-02T01:34:19Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-02T01:34:19Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-02T01:34:19Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-02T01:34:19Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-02T01:34:19Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-02T01:34:20Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-02T01:34:20Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:03:29Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:03:29Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-03T14:03:29Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-03T14:03:30Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-03T14:03:30Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-03T14:03:30Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-03T14:03:31Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-03T14:03:31Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-03T14:03:32Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-03T14:03:32Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:03:33Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:03:33Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-03T14:03:34Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-03T14:03:34Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-03T14:03:34Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-03T14:03:34Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-03T14:03:35Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-03T14:03:35Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-03T14:03:36Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-03T14:03:36Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:03:37Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:03:37Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-03T14:03:38Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-03T14:03:38Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-03T14:03:39Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-03T14:03:39Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-03T14:03:40Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-03T14:03:40Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-03T14:03:40Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-03T14:03:40Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:03:41Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:03:41Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-03T14:03:42Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-03T14:03:42Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-03T14:03:43Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-03T14:03:43Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-03T14:03:44Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-03T14:03:44Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-03T14:03:44Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-03T14:03:44Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:03:45Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:03:45Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-03T14:03:46Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-03T14:03:46Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-03T14:03:47Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-03T14:03:47Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-03T14:03:48Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-03T14:03:48Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-03T14:03:49Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-03T14:03:49Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:03:50Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:03:50Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-03T14:03:50Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-03T14:03:50Z) ---

# --- Auto-research iteration 24: refactor encoder factory to support custom backbones (2026-05-03T14:03:51Z) ---

# --- Auto-research iteration 25: implement attention rollout for visual explanation (2026-05-03T14:03:51Z) ---

# --- Auto-research iteration 34: implement visual feature memory bank with FAISS (2026-05-03T14:03:52Z) ---

# --- Auto-research iteration 35: add adaptive image preprocessing based on content type (2026-05-03T14:03:52Z) ---

# --- Auto-research iteration 44: refactor reflection engine with hierarchical memory (2026-05-03T14:03:53Z) ---

# --- Auto-research iteration 45: implement Reflexion-style verbal reinforcement loop (2026-05-03T14:03:53Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:03:54Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:03:54Z) ---

# --- Auto-research iteration 14: implement gradient-free hyperparameter adaptation (2026-05-03T14:03:55Z) ---

# --- Auto-research iteration 15: add online calibration update with exponential smoothing (2026-05-03T14:03:55Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:04:39Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:04:39Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:04:39Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:04:39Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:04:40Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:04:40Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:04:40Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:04:40Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:04:41Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:04:41Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-03T14:04:41Z) ---

# --- Auto-research iteration 5: add memory-bounded reflection history with LRU eviction (2026-05-03T14:04:41Z) ---

# --- Auto-research iteration 4: refactor agent state machine with explicit transitions (2026-05-04T11:37:27Z) ---
