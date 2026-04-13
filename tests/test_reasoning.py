"""Tests for reasoning modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msea.reasoning.chain_of_thought import ChainOfThought
from msea.reasoning.reflection import ReflectionEngine
from msea.reasoning.critique import SelfCritique
from msea.reasoning.process_supervision import ProcessSupervisor


def test_chain_of_thought():
    cot = ChainOfThought({"max_steps": 5})
    chain = cot.generate("What color is the sky?")
    assert chain.length > 0
    assert chain.total_confidence > 0
    print("✓ test_chain_of_thought passed")


def test_cot_coherence():
    cot = ChainOfThought({})
    chain = cot.generate("Test question", visual_features="mock")
    score = cot.evaluate_coherence(chain)
    assert 0 <= score <= 1
    print("✓ test_cot_coherence passed")


def test_reflection_engine():
    engine = ReflectionEngine({"max_memory": 50})
    reflection = engine.reflect(0, 0.8, False, "I think the answer is correct")
    assert reflection.episode_id == 0
    assert not reflection.success
    assert len(reflection.identified_errors) > 0
    print("✓ test_reflection_engine passed")


def test_self_critique():
    critic = SelfCritique({})
    result = critic.critique(
        "The image shows a cat. Therefore the answer is cat.",
        "cat", confidence=0.7, has_visual=True
    )
    assert 0 <= result.overall_score <= 1
    assert isinstance(result.actionable_feedback, list)
    print("✓ test_self_critique passed")


def test_process_supervisor():
    supervisor = ProcessSupervisor({"strictness": 0.4})
    steps = [
        "Let me analyze the image carefully.",
        "I can see objects in the scene. Therefore the answer involves spatial reasoning.",
        "Based on careful examination, the answer is B."
    ]
    result = supervisor.evaluate_chain(steps)
    assert result["chain_length"] == 3
    assert 0 <= result["process_reward"] <= 1
    print("✓ test_process_supervisor passed")


if __name__ == "__main__":
    test_chain_of_thought()
    test_cot_coherence()
    test_reflection_engine()
    test_self_critique()
    test_process_supervisor()
    print("\nAll reasoning tests passed! ✓")
