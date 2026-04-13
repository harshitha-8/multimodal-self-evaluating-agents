"""Tests for agent modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msea.agents.base_agent import Observation, ReasoningTrace, AgentOutput
from msea.agents.metacognitive_agent import MetacognitiveAgent, ReflectionToken
from msea.agents.tool_agent import ToolAugmentedAgent
from msea.agents.multi_agent import MultiAgentCoordinator


def test_observation_creation():
    obs = Observation(textual="What is in the image?")
    assert obs.textual == "What is in the image?"
    assert obs.visual is None
    print("✓ test_observation_creation passed")


def test_metacognitive_agent_creation():
    config = {"name": "TestAgent", "max_reasoning_steps": 5}
    agent = MetacognitiveAgent(config)
    assert agent.name == "TestAgent"
    assert agent.max_reasoning_steps == 5
    print("✓ test_metacognitive_agent_creation passed")


def test_metacognitive_agent_run():
    config = {"name": "TestAgent", "max_reasoning_steps": 3, "confidence_threshold": 0.8}
    agent = MetacognitiveAgent(config)
    obs = Observation(textual="What is shown in the image?")
    output = agent.run(obs)
    assert isinstance(output, AgentOutput)
    assert len(output.reasoning_trace) > 0
    assert 0 <= output.self_eval_score <= 1
    print("✓ test_metacognitive_agent_run passed")


def test_reflection_token_detection():
    assert "UNCERTAIN" in str(ReflectionToken.detect("I'm not sure about this"))
    assert "CONFIDENT" in str(ReflectionToken.detect("This is obviously correct"))
    assert "CONTRADICTION" in str(ReflectionToken.detect("However this contradicts"))
    print("✓ test_reflection_token_detection passed")


def test_tool_agent():
    config = {"name": "ToolTest", "tool_budget": 3}
    agent = ToolAugmentedAgent(config)
    obs = Observation(textual="Analyze this data")
    output = agent.run(obs)
    assert isinstance(output, AgentOutput)
    print("✓ test_tool_agent passed")


def test_multi_agent():
    config = {"name": "MultiTest", "strategy": "parallel"}
    coordinator = MultiAgentCoordinator(config)
    agent1 = MetacognitiveAgent({"name": "Agent1"})
    agent2 = MetacognitiveAgent({"name": "Agent2"})
    coordinator.register_agent("agent1", agent1)
    coordinator.register_agent("agent2", agent2)
    obs = Observation(textual="Test question")
    output = coordinator.run(obs)
    assert isinstance(output, AgentOutput)
    print("✓ test_multi_agent passed")


def test_calibration_update():
    config = {"name": "CalibTest"}
    agent = MetacognitiveAgent(config)
    for i in range(20):
        agent.update_calibration(0.7, i % 2 == 0)
    assert len(agent.meta_state.calibration_history) == 20
    print("✓ test_calibration_update passed")


if __name__ == "__main__":
    test_observation_creation()
    test_metacognitive_agent_creation()
    test_metacognitive_agent_run()
    test_reflection_token_detection()
    test_tool_agent()
    test_multi_agent()
    test_calibration_update()
    print("\nAll agent tests passed! ✓")
