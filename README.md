# 🧠 Multimodal Self-Evaluating Agents

> *Autonomous research at the intersection of Multimodal AI × Agent Metacognition × Annotation-Efficient Learning*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org)

---

One day, multimodal AI research used to be done by human researchers painstakingly labeling data,
tuning hyperparameters, and reading papers between coffee breaks. That era is ending. This repo
explores what happens when you give AI agents the ability to **reason about their own reasoning**
over multimodal inputs — and let them improve autonomously. Inspired by
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch), but focused on the frontier
of **multimodal metacognition**. — @harshitha-8, April 2026.

## Core Idea

Give a multimodal agent a **self-evaluation loop**: it processes visual + textual inputs, generates
reasoning traces, critiques its own outputs, estimates uncertainty, and iteratively refines its
strategy — all without human annotation. The agent modifies its own reasoning pipeline, evaluates
on benchmarks, keeps improvements, discards regressions, and repeats.

```
Visual Input → Perception → Reasoning → Self-Critique → Tool Use → Refinement → Output
                    ↑                         |
                    └─────── Metacognitive Feedback Loop ───────┘
```

## Research Directions

This framework explores 5 key research themes:

| # | Theme | Description |
|---|-------|-------------|
| 1 | **Metacognitive Multimodal Agents** | Agents that evaluate their own VL reasoning quality |
| 2 | **Annotation-Free Visual Learning** | Learning robust vision representations without labels |
| 3 | **Uncertainty-Aware Tool Selection** | Agents that know when to use tools vs. reason internally |
| 4 | **Self-Improving Reasoning Chains** | Iterative refinement of chain-of-thought over visual inputs |
| 5 | **Cross-Modal Consistency Checking** | Detecting hallucinations via vision-language agreement |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/harshitha-8/multimodal-self-evaluating-agents.git
cd multimodal-self-evaluating-agents

# 2. Install dependencies
pip install -e .

# 3. Run the self-evaluation benchmark
python experiments/run_benchmark.py --config configs/base_config.yaml

# 4. Run the autonomous research loop
python run_agent.py --program program.md
```

## Project Structure

```
multimodal-self-evaluating-agents/
├── program.md                    # Agent instructions (Karpathy-style)
├── run_agent.py                  # Main autonomous research loop
├── setup.py                      # Package installation
├── configs/                      # Experiment configurations
│   ├── base_config.yaml
│   ├── metacognition_config.yaml
│   └── ablation_config.yaml
├── msea/                         # Core library (Multimodal Self-Evaluating Agents)
│   ├── __init__.py
│   ├── agents/                   # Agent implementations
│   │   ├── __init__.py
│   │   ├── base_agent.py         # Abstract agent interface
│   │   ├── metacognitive_agent.py # Self-evaluating agent
│   │   ├── tool_agent.py         # Tool-augmented agent
│   │   └── multi_agent.py        # Multi-agent coordination
│   ├── perception/               # Visual perception modules
│   │   ├── __init__.py
│   │   ├── encoders.py           # Vision encoders (CLIP, SigLIP, DINOv2)
│   │   ├── features.py           # Feature extraction pipeline
│   │   └── uncertainty.py        # Perceptual uncertainty estimation
│   ├── reasoning/                # Reasoning modules
│   │   ├── __init__.py
│   │   ├── chain_of_thought.py   # CoT generation and evaluation
│   │   ├── reflection.py         # Self-reflection mechanisms
│   │   ├── critique.py           # Self-critique scoring
│   │   └── process_supervision.py # Process reward models
│   ├── tools/                    # Tool-use interfaces
│   │   ├── __init__.py
│   │   ├── tool_registry.py      # Tool registration and dispatch
│   │   ├── retrieval.py          # RAG tool implementation
│   │   ├── code_executor.py      # Code execution sandbox
│   │   └── visual_tools.py       # Visual analysis tools
│   ├── evaluation/               # Evaluation framework
│   │   ├── __init__.py
│   │   ├── metrics.py            # Evaluation metrics
│   │   ├── benchmarks.py         # Benchmark loaders
│   │   ├── self_eval.py          # Self-evaluation scoring
│   │   └── consistency.py        # Cross-modal consistency
│   ├── data/                     # Data handling
│   │   ├── __init__.py
│   │   ├── datasets.py           # Dataset loaders
│   │   ├── preprocessing.py      # Data preprocessing
│   │   └── synthetic.py          # Synthetic data generation
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── logging.py            # Experiment logging
│       ├── config.py             # Configuration management
│       └── visualization.py      # Result visualization
├── experiments/                  # Experiment scripts
│   ├── run_benchmark.py          # Main benchmark runner
│   ├── run_metacognition.py      # Metacognition experiments
│   ├── run_ablation.py           # Ablation studies
│   └── analyze_results.py        # Result analysis
├── research/                     # Research outputs
│   ├── landscape_scan.md         # Literature landscape
│   ├── gap_analysis.md           # Research gap identification
│   ├── project_ideas.md          # Publishable project ideas
│   └── experiment_plan.md        # Low-cost experiment plan
├── results/                      # Experiment results (auto-generated)
│   └── results.tsv
├── tests/                        # Unit tests
│   ├── test_agents.py
│   ├── test_perception.py
│   ├── test_reasoning.py
│   └── test_evaluation.py
└── scripts/                      # Utility scripts
    ├── auto_commit.sh            # Automated daily commits
    └── setup_environment.sh      # Environment setup
```

## How It Works

The repo follows the Karpathy autoresearch pattern but adapted for multimodal metacognition:

1. **`program.md`** — Instructions for the autonomous agent (you don't touch the Python, you program the program)
2. **`run_agent.py`** — The main loop: modify reasoning pipeline → evaluate → keep/discard → repeat
3. **`msea/`** — The core library that the agent modifies and experiments with

### The Experiment Loop

```
LOOP FOREVER:
1. Read current state of the reasoning pipeline
2. Propose a modification (architecture, prompts, tool selection, etc.)
3. Run evaluation on multimodal benchmarks
4. If metric improved → keep changes, advance branch
5. If metric worsened → revert, log result
6. Record everything in results.tsv
7. Never stop until manually interrupted
```

## Research Output

### Section 1 — Research Clusters
See [research/landscape_scan.md](research/landscape_scan.md)

### Section 2 — Research Gaps
See [research/gap_analysis.md](research/gap_analysis.md)

### Section 3 — Project Ideas
See [research/project_ideas.md](research/project_ideas.md)

### Section 4 — Experiment Plan
See [research/experiment_plan.md](research/experiment_plan.md)

## Design Principles

- **Single metric**: Self-evaluation accuracy (does the agent correctly predict its own performance?)
- **Fixed compute budget**: Each experiment runs for a fixed time window
- **Simplicity criterion**: Simpler is better, all else equal
- **Zero annotation**: No human labels required for training or evaluation
- **Open everything**: Open datasets, open models, open benchmarks

## Supported Models

| Model | Type | Use Case |
|-------|------|----------|
| CLIP ViT-B/16 | Vision Encoder | Feature extraction |
| SigLIP | Vision-Language | Cross-modal alignment |
| DINOv2 | Self-supervised Vision | Annotation-free features |
| LLaVA-1.5 7B | Multimodal LLM | Reasoning backbone |
| Phi-3 Vision | Compact VLM | Edge deployment |
| Qwen2-VL 2B | Small VLM | Rapid iteration |

## Citation

```bibtex
@software{multimodal_self_eval_agents_2026,
  author = {Harshitha M},
  title = {Multimodal Self-Evaluating Agents: A Research Framework for Agent Metacognition},
  year = {2026},
  url = {https://github.com/harshitha-8/multimodal-self-evaluating-agents}
}
```

## License

MIT

## Acknowledgments

Inspired by [Karpathy/autoresearch](https://github.com/karpathy/autoresearch), [Reflexion](https://arxiv.org/abs/2303.11366), [ReAct](https://arxiv.org/abs/2210.03629), and the broader multimodal AI research community.
