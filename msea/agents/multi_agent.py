"""
Multi-Agent Coordinator — Orchestrates multiple specialized agents
for collaborative multimodal reasoning.

Architecture: Manager agent dispatches tasks to specialist agents,
aggregates results, and resolves conflicts through voting or debate.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from msea.agents.base_agent import (
    BaseAgent, AgentOutput, Observation, ReasoningTrace
)


@dataclass
class AgentVote:
    """A vote from a specialist agent."""
    agent_name: str
    answer: str
    confidence: float
    reasoning_summary: str


class MultiAgentCoordinator(BaseAgent):
    """
    Coordinates multiple agents for ensemble-style multimodal reasoning.

    Strategies:
    1. Parallel — All agents reason independently, vote on answer
    2. Sequential — Agents build on each other's reasoning
    3. Debate — Agents critique each other's outputs
    4. Specialization — Route to the most appropriate agent
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config.get("name", "MultiAgentCoordinator")
        self.agents: Dict[str, BaseAgent] = {}
        self.strategy = config.get("strategy", "parallel")
        self.agreement_threshold = config.get("agreement_threshold", 0.6)
        self.debate_rounds = config.get("debate_rounds", 2)

    def register_agent(self, name: str, agent: BaseAgent):
        """Register a specialist agent."""
        self.agents[name] = agent

    def perceive(self, observation: Observation) -> Dict[str, Any]:
        """Distribute perception to all agents."""
        features = {
            "observation": observation,
            "num_agents": len(self.agents),
            "strategy": self.strategy,
            "modalities": [],
        }
        if observation.visual is not None:
            features["modalities"].append("visual")
        if observation.textual is not None:
            features["modalities"].append("textual")
        return features

    def reason(self, features: Dict[str, Any], context: Optional[str] = None) -> ReasoningTrace:
        """Coordinate multi-agent reasoning."""
        observation = features.get("observation")

        if self.strategy == "parallel":
            votes = self._parallel_reasoning(observation)
        elif self.strategy == "sequential":
            votes = self._sequential_reasoning(observation)
        elif self.strategy == "debate":
            votes = self._debate_reasoning(observation)
        else:
            votes = self._parallel_reasoning(observation)

        # Aggregate votes
        answer, confidence = self._aggregate_votes(votes)

        return ReasoningTrace(
            step_id=0,
            thought=f"Multi-agent consensus ({self.strategy}): {answer}",
            confidence=confidence,
            uncertainty=1.0 - confidence,
            is_terminal=True,
        )

    def self_evaluate(self, output: AgentOutput) -> float:
        """Evaluate based on inter-agent agreement."""
        return output.confidence

    def estimate_uncertainty(self, features: Dict[str, Any]) -> float:
        """Lower uncertainty with more agents and agreement."""
        n = features.get("num_agents", 1)
        return 0.5 / max(1, n)

    def _parallel_reasoning(self, observation: Observation) -> List[AgentVote]:
        """All agents reason independently."""
        votes = []
        for name, agent in self.agents.items():
            try:
                output = agent.run(observation)
                votes.append(AgentVote(
                    agent_name=name,
                    answer=output.answer,
                    confidence=output.confidence,
                    reasoning_summary=output.answer[:100],
                ))
            except Exception as e:
                votes.append(AgentVote(
                    agent_name=name, answer="", confidence=0.0,
                    reasoning_summary=f"Error: {str(e)}",
                ))
        return votes

    def _sequential_reasoning(self, observation: Observation) -> List[AgentVote]:
        """Agents build on each other sequentially."""
        votes = []
        accumulated_context = ""
        for name, agent in self.agents.items():
            try:
                features = agent.perceive(observation)
                trace = agent.reason(features, accumulated_context)
                votes.append(AgentVote(
                    agent_name=name,
                    answer=trace.thought,
                    confidence=trace.confidence,
                    reasoning_summary=trace.thought[:100],
                ))
                accumulated_context += f"\n{name}: {trace.thought}"
            except Exception:
                pass
        return votes

    def _debate_reasoning(self, observation: Observation) -> List[AgentVote]:
        """Agents debate with each other."""
        # First round: independent reasoning
        votes = self._parallel_reasoning(observation)

        # Subsequent rounds: critique and revise
        for round_num in range(self.debate_rounds):
            debate_context = "\n".join(
                f"{v.agent_name}: {v.answer} (confidence: {v.confidence:.2f})"
                for v in votes
            )
            revised_votes = []
            for name, agent in self.agents.items():
                try:
                    features = agent.perceive(observation)
                    trace = agent.reason(features, f"Debate round {round_num + 1}:\n{debate_context}")
                    revised_votes.append(AgentVote(
                        agent_name=name,
                        answer=trace.thought,
                        confidence=trace.confidence,
                        reasoning_summary=trace.thought[:100],
                    ))
                except Exception:
                    pass
            if revised_votes:
                votes = revised_votes

        return votes

    def _aggregate_votes(self, votes: List[AgentVote]) -> tuple:
        """Aggregate votes using confidence-weighted voting."""
        if not votes:
            return "No consensus", 0.0

        # Confidence-weighted selection
        best_vote = max(votes, key=lambda v: v.confidence)
        avg_confidence = sum(v.confidence for v in votes) / len(votes)

        return best_vote.answer, avg_confidence

# --- Auto-research iteration 1: add confidence decay schedule to metacognitive loop (2026-04-13T23:49:29Z) ---

# --- Auto-research iteration 10: refactor tool selection with cost-benefit scoring (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 11: add multi-round debate protocol to multi-agent coordinator (2026-04-13T23:49:30Z) ---

# --- Auto-research iteration 20: add confidence interval estimation using bootstrap (2026-04-13T23:49:31Z) ---

# --- Auto-research iteration 21: add SigLIP encoder with sigmoid loss similarity (2026-04-13T23:49:31Z) ---
