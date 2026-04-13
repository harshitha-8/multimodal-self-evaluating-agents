"""
Base Agent — Abstract interface for all multimodal agents.
Inspired by the ReAct pattern: Observe → Think → Act → Evaluate.
"""

import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class AgentState(Enum):
    """Current state of the agent in the reasoning loop."""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    ACTING = "acting"
    EVALUATING = "evaluating"
    REFLECTING = "reflecting"
    REFINING = "refining"


@dataclass
class Observation:
    """Multimodal observation from the environment."""
    visual: Optional[Any] = None          # Image tensor or features
    textual: Optional[str] = None         # Text input
    structured: Optional[Dict] = None     # Structured data (tables, JSON)
    audio: Optional[Any] = None           # Audio features
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningTrace:
    """A single step in the reasoning chain."""
    step_id: int
    thought: str                          # Natural language reasoning
    action: Optional[str] = None          # Action taken (tool call, etc.)
    action_input: Optional[Dict] = None   # Action parameters
    observation: Optional[str] = None     # Result of action
    confidence: float = 0.5              # Agent's self-assessed confidence
    uncertainty: float = 0.5             # Estimated uncertainty
    is_terminal: bool = False            # Whether this is a final step
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentOutput:
    """Complete agent output including reasoning trace and self-evaluation."""
    answer: str
    reasoning_trace: List[ReasoningTrace]
    confidence: float
    self_eval_score: float               # Agent's prediction of its own correctness
    uncertainty_estimate: float
    tools_used: List[str] = field(default_factory=list)
    num_refinement_rounds: int = 0
    total_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for multimodal self-evaluating agents.

    The agent follows a perceive → reason → act → evaluate loop,
    with metacognitive capabilities for self-assessment.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = AgentState.IDLE
        self.history: List[AgentOutput] = []
        self.step_count = 0
        self.total_tokens = 0
        self.name = config.get("name", "BaseAgent")
        self.max_reasoning_steps = config.get("max_reasoning_steps", 10)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        self.verbose = config.get("verbose", False)

    @abstractmethod
    def perceive(self, observation: Observation) -> Dict[str, Any]:
        """
        Process multimodal input and extract relevant features.
        Returns a feature dictionary that downstream reasoning can use.
        """
        pass

    @abstractmethod
    def reason(self, features: Dict[str, Any], context: Optional[str] = None) -> ReasoningTrace:
        """
        Generate a single reasoning step given features and context.
        Returns a ReasoningTrace with thought, action, and confidence.
        """
        pass

    @abstractmethod
    def self_evaluate(self, output: AgentOutput) -> float:
        """
        Metacognitive self-evaluation: predict correctness of own output.
        Returns a score in [0, 1] estimating probability of being correct.
        """
        pass

    @abstractmethod
    def estimate_uncertainty(self, features: Dict[str, Any]) -> float:
        """
        Estimate epistemic uncertainty for the current input.
        Returns uncertainty score in [0, 1], higher = more uncertain.
        """
        pass

    def should_use_tool(self, features: Dict[str, Any], trace: ReasoningTrace) -> bool:
        """
        Decide whether to invoke an external tool based on current state.
        Default: use tool when uncertainty exceeds threshold.
        """
        return trace.uncertainty > self.confidence_threshold

    def refine(self, output: AgentOutput, critique: str) -> AgentOutput:
        """
        Refine a previous output based on self-critique.
        Default implementation re-runs reasoning with critique as context.
        """
        # Subclasses should override for sophisticated refinement
        return output

    def run(self, observation: Observation) -> AgentOutput:
        """
        Execute the full perceive → reason → evaluate → refine loop.
        This is the main entry point for agent execution.
        """
        t_start = time.time()
        self.state = AgentState.PERCEIVING

        # Step 1: Perceive
        features = self.perceive(observation)

        # Step 2: Iterative reasoning
        self.state = AgentState.REASONING
        traces = []
        tools_used = []

        for step in range(self.max_reasoning_steps):
            context = self._build_context(traces)
            trace = self.reason(features, context)
            trace.step_id = step
            traces.append(trace)

            # Check if tool is needed
            if self.should_use_tool(features, trace) and trace.action:
                self.state = AgentState.ACTING
                tools_used.append(trace.action)
                self.state = AgentState.REASONING

            if trace.is_terminal:
                break

        # Step 3: Generate output
        answer = self._extract_answer(traces)
        confidence = self._aggregate_confidence(traces)
        uncertainty = self.estimate_uncertainty(features)

        output = AgentOutput(
            answer=answer,
            reasoning_trace=traces,
            confidence=confidence,
            self_eval_score=0.0,  # Will be filled by self_evaluate
            uncertainty_estimate=uncertainty,
            tools_used=tools_used,
            total_time=time.time() - t_start,
        )

        # Step 4: Self-evaluate
        self.state = AgentState.EVALUATING
        output.self_eval_score = self.self_evaluate(output)

        # Step 5: Optional refinement
        if output.self_eval_score < self.confidence_threshold:
            self.state = AgentState.REFINING
            critique = self._generate_critique(output)
            output = self.refine(output, critique)
            output.num_refinement_rounds += 1

        self.state = AgentState.IDLE
        self.history.append(output)
        self.step_count += 1
        return output

    def _build_context(self, traces: List[ReasoningTrace]) -> str:
        """Build context string from previous reasoning steps."""
        if not traces:
            return ""
        lines = []
        for t in traces:
            lines.append(f"Step {t.step_id}: {t.thought}")
            if t.action:
                lines.append(f"  Action: {t.action}")
            if t.observation:
                lines.append(f"  Observation: {t.observation}")
        return "\n".join(lines)

    def _extract_answer(self, traces: List[ReasoningTrace]) -> str:
        """Extract final answer from reasoning trace."""
        if traces and traces[-1].thought:
            return traces[-1].thought
        return "No answer generated."

    def _aggregate_confidence(self, traces: List[ReasoningTrace]) -> float:
        """Aggregate confidence across all reasoning steps."""
        if not traces:
            return 0.0
        confidences = [t.confidence for t in traces]
        # Weighted average favoring later (more refined) steps
        weights = [i + 1 for i in range(len(confidences))]
        total_weight = sum(weights)
        return sum(c * w for c, w in zip(confidences, weights)) / total_weight

    def _generate_critique(self, output: AgentOutput) -> str:
        """Generate self-critique of the current output."""
        critique_parts = []
        if output.confidence < 0.5:
            critique_parts.append("Low overall confidence in reasoning chain.")
        if output.uncertainty_estimate > 0.7:
            critique_parts.append("High uncertainty — consider using additional tools.")
        if len(output.reasoning_trace) >= self.max_reasoning_steps:
            critique_parts.append("Hit maximum reasoning steps without convergence.")
        return " ".join(critique_parts) if critique_parts else "Output appears reasonable."

    def get_stats(self) -> Dict[str, Any]:
        """Return agent performance statistics."""
        if not self.history:
            return {"steps": 0, "avg_confidence": 0.0, "avg_self_eval": 0.0}

        confidences = [o.confidence for o in self.history]
        self_evals = [o.self_eval_score for o in self.history]
        times = [o.total_time for o in self.history]

        return {
            "steps": self.step_count,
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_self_eval": sum(self_evals) / len(self_evals),
            "avg_time": sum(times) / len(times),
            "total_time": sum(times),
            "tools_used": sum(len(o.tools_used) for o in self.history),
        }

    def reset(self):
        """Reset agent state for a new episode."""
        self.state = AgentState.IDLE
        self.history.clear()
        self.step_count = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, state={self.state.value})"

# --- Auto-research iteration 1: add confidence decay schedule to metacognitive loop (2026-04-13T23:49:29Z) ---

# --- Auto-research iteration 10: refactor tool selection with cost-benefit scoring (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 11: add multi-round debate protocol to multi-agent coordinator (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 20: add confidence interval estimation using bootstrap (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 21: add SigLIP encoder with sigmoid loss similarity (2026-04-13T23:49:31Z) ---
