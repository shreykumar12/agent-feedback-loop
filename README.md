<!--
WHY THIS FILE EXISTS:
  Build step 8. The README is where the project becomes defensible: it explains
  the design decisions, not just how to run it. Reviewers read this first.

WHAT IT NEEDS (fill in as you build):
  - One-liner + "the agent is a component; the engineering is the loop"
  - Self-correcting vs self-improving distinction
  - Architecture diagram (copy from Spec.md)
  - Why correctness is deterministic and the LLM judge is secondary
  - Setup + usage commands
  - Results: pass rate, tries-to-pass distribution, feedback-level ablation,
    an example regression comparison
  - Limitations: subprocess sandbox isn't strong isolation, small suite = noisy
    metrics, possible benchmark contamination in HumanEval-style tasks
  Delete this comment block when done.
-->

# AgentEval: Self-Correcting Coding Agent

TODO: one-paragraph overview.

## Architecture

TODO

## Design decisions

TODO

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY
python main.py init-db
```

## Usage

```bash
python main.py run --model gemini-2.5-flash --prompt-version v1
python main.py runs
python main.py compare <baseline_run_id> <candidate_run_id>
streamlit run dashboard/app.py
pytest
```

## Results

TODO

## Limitations

TODO
