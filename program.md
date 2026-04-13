This is an experiment to have an AI agent do its own multimodal metacognition research.

## Setup
To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr13`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo has a clean structure. Read these for full context:
   - `README.md` — repository context and research directions.
   - `msea/` — the core library. Everything here is fair game for modification.
   - `configs/base_config.yaml` — experiment configuration.
4. **Verify environment**: Check that dependencies are installed. If not, tell the human to run `pip install -e .`
5. **Initialize results.tsv**: Create `results/results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation
Each experiment evaluates the agent's metacognitive ability. The evaluation runs on available compute (CPU or single GPU). You launch it as: `python experiments/run_benchmark.py --config configs/base_config.yaml`.

**What you CAN do:**
- Modify anything in `msea/` — this is the core library. Everything is fair game: agent architecture, reasoning strategies, tool selection, uncertainty estimation, self-critique mechanisms.
- Modify experiment configs in `configs/`.
- Add new evaluation metrics in `msea/evaluation/`.

**What you CANNOT do:**
- Modify the benchmark datasets themselves.
- Install new packages beyond what's in `setup.py`.
- Fabricate evaluation results.

**The goal is simple: maximize self-evaluation accuracy.** The agent should correctly predict when its own reasoning will succeed or fail on multimodal tasks. Higher self-eval accuracy = better metacognition.

**Key metrics:**
- `self_eval_accuracy` — Does the agent correctly estimate its own confidence? (primary)
- `reasoning_quality` — Quality of generated reasoning chains (secondary)
- `tool_selection_f1` — Does the agent pick the right tool? (secondary)
- `hallucination_rate` — Rate of cross-modal inconsistencies (lower is better)

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results = simplification win.

**The first run**: Your very first run should always establish the baseline.

## Output format
Once the evaluation finishes it prints a summary:

```
---
self_eval_accuracy:  0.7234
reasoning_quality:   0.6891
tool_selection_f1:   0.8123
hallucination_rate:  0.0456
total_seconds:       125.3
num_samples:         500
```

## Logging results
When an experiment is done, log it to `results/results.tsv` (tab-separated).

The TSV has a header row and 6 columns:

```
commit	self_eval_acc	reasoning_q	tool_f1	status	description
```

1. git commit hash (short, 7 chars)
2. self_eval_accuracy (e.g. 0.723400)
3. reasoning_quality (e.g. 0.689100)
4. tool_selection_f1 (e.g. 0.812300)
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

## Experiment Loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Modify `msea/` with an experimental idea
3. git commit
4. Run evaluation: `python experiments/run_benchmark.py --config configs/base_config.yaml > run.log 2>&1`
5. Read results: `grep "^self_eval_accuracy:\|^reasoning_quality:" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the traceback.
7. Record results in results.tsv (NOTE: do not commit results.tsv)
8. If self_eval_accuracy improved → keep changes, advance branch
9. If equal or worse → git reset back

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — read the research papers referenced in the code, try combining approaches, try more radical changes. The loop runs until the human stops you.

## Research Focus Areas

When generating experimental ideas, focus on:

1. **Reflection tokens** — Adding special tokens that trigger self-evaluation
2. **Uncertainty calibration** — Making confidence scores match actual accuracy
3. **Process supervision** — Evaluating intermediate reasoning steps, not just final answers
4. **Cross-modal consistency** — Using vision-language agreement as a self-check
5. **Active tool selection** — Learning when to use retrieval vs. code execution vs. visual analysis
6. **Failure mode detection** — Teaching the agent to recognize its own failure patterns
7. **Iterative refinement** — Multi-round reasoning with self-critique between rounds
