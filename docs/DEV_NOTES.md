# Dev notes (working memory for building AgentEval)

Running log of requirements, decisions, and findings. Keep updated as work progresses.

## Original README requirements (verbatim from the template, build step 8)

- One-liner + "the agent is a component; the engineering is the loop"
- Self-correcting vs self-improving distinction
- Architecture diagram (copy from Spec.md)
- Why correctness is deterministic and the LLM judge is secondary
- Setup + usage commands
- Results: pass rate, tries-to-pass distribution, feedback-level ablation,
  an example regression comparison
- Limitations: subprocess sandbox isn't strong isolation, small suite = noisy
  metrics, possible benchmark contamination in HumanEval-style tasks

Original usage block:
```
python main.py init-db
python main.py run --model gemini-2.5-flash --prompt-version v1
python main.py runs
python main.py compare <baseline_run_id> <candidate_run_id>
streamlit run dashboard/app.py
pytest
```

## Spec pillars (from module docstrings; Spec.md itself is gitignored / not in repo)
1. Safe execution of untrusted code (subprocess, rlimits, stripped env)
2. Deterministic correctness via real tests (never ask an LLM if code is correct)
3. Actionable feedback (levels minimal/names/full -> ablation)
4. Bounded retry loop (LangGraph, conditional edge, MAX_TRIES ~3-4)
5. Measurement & regression detection (SQLite history, compare runs)
Optional: LLM judge for readability/approach only, on passing code only.

## Environment findings (session 2026-09-29)
- System python is 3.11; requirements.txt pins (numpy 2.5.3 etc.) need >=3.12.
  CI uses 3.13. Local venv: `uv venv -p /usr/bin/python3.13 venv && uv pip install -r requirements.txt`.
- No GEMINI_API_KEY in this container and generativelanguage.googleapis.com is
  not reachable -> real LLM runs cannot be executed here. Built an offline
  simulated agent (`sim-*` models) so the whole pipeline produces metrics
  without a key. Simulated results demonstrate the pipeline; they are NOT
  claims about a real model.
- Spec.md / structure.md / agents.md are gitignored (user keeps them locally).

## Decisions
(see below, appended as made)

## User instructions (mid-session)
- "make branches for all the work" + "dont merge them i will merge them, just tell me
  when done and how to test". => Commit per component on the integration branch
  `claude/sleepy-noether-1nmbe3`, then create stacked branches `claude/0N-<name>`
  pointing at each milestone commit. Push all. NO merges, NO PRs. Final message:
  merge order + how to test.

## Task suite (built by a subagent, verified by `python main.py validate-tasks`)
- 20 original tasks (6 easy / 8 medium / 6 hard), 164 hidden tests, 88 mutants,
  mutation score 100%, suite hash e0aacf5367b3 (changes if prompts/tests change).
- Mutant verdict mix: 64 wrong_answer, 19 runtime_error, 2 syntax, 2 timeout, 1 import.
  Each loadable mutant fails 1-3 tests (subtle bugs -> retries are informative).
- Trap tasks (non-obvious conventions): rounded_mean (half away from zero),
  merge_intervals (half-open), rank_players (competition ranking + case-sensitive ties),
  summarize_ranges (runs of 2 are not ranges), lru_simulate (put refreshes recency),
  evaluate_expression (/ truncates toward zero).
- Generator script lived in the session scratchpad (not committed); tasks.json is the source of truth.

## Findings from the simulated ablation (60 runs: 3 sim models x 4 feedback levels x 5 seeds)
- Sim calibration bug found + fixed: "names" feedback scored BELOW "minimal" because
  the blind-retry branch had a higher fix probability than the names branch.
  Now all retries share a blind floor p_blind = first_try*0.5 and info only adds.
- Final pass rate (mean of 5 seeds), minimal -> names -> full:
  sim-strong 85 -> 97 -> 100, sim-base 82 -> 92 -> 96, sim-weak 59 -> 77 -> 87.
  Richer feedback helps the weak model most (+28pp vs +15pp).
- pass@1 is identical across feedback levels for the same seed (the first try never
  sees feedback) -> the ablation is a controlled, paired experiment.
- Self-correction MASKS model regressions: strong->weak single-seed swap drops
  final pass rate 10pp but pass@1 40pp. => compare reports McNemar on pass@1 too.
- Noise: an A/A comparison (same config, seed 0 vs 1) trips the pass@1 threshold
  on a 20-task suite. => repeated-run group comparison with permutation test and
  pooled (seed, task) McNemar. full->minimal: 14 pairs lost / 0 gained, p=1.2e-4.
