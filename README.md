# AgentEval: Self-Correcting Coding Agent

An LLM writes a Python function, a sandbox runs hidden tests against it, and the failures are turned into feedback for the next attempt. The loop stops after a fixed try budget. Every attempt is stored so runs can be measured and compared.

**The agent is a component; the engineering is the loop.** The model is swappable by config. The real work is in five parts:

1. Running untrusted code safely.
2. Scoring correctness deterministically.
3. Turning test failures into feedback the agent can act on.
4. Bounding the retry loop.
5. Measuring whether a change to the model, prompt or feedback made things better or worse.

**Headline result (measured on a real model):** self-training Qwen2.5-Coder-0.5B-Instruct with LoRA on its own sandbox-verified solutions raised held-out pass@1 from **5% to 35%** (+6 tasks gained, 0 lost, exact McNemar p = 0.031). The ablation also showed what did *not* happen: the 0.5B model almost never repaired its code from test feedback, so the gain came from training on verified solutions, not from feedback. See [Self-training results](#self-training-results-measured).

![Evaluation report](docs/img/report-overview.png)

## Self-correcting, and optionally self-improving

Within one task the agent is **self-correcting**: it uses test feedback to repair its own answer over a few bounded tries. A single run is never **self-improving**: nothing learned on one task carries over to the next. There is no memory or prompt rewriting between tasks, so every run stays a clean, comparable measurement.

Self-improvement is a separate, explicit step on top of that, using PyTorch (see [Self-improvement with PyTorch](#self-improvement-with-pytorch)). A local model runs the loop on *training* tasks, and its sandbox-verified successes and repairs become fine-tuning data for a LoRA adapter. The next round's model is then measured on the held-out eval suites. Each round is a new, versioned model, and the same regression detector judges whether it actually got better.

## Architecture

```mermaid
flowchart LR
    T[tasks.json<br/>prompt + hidden tests] --> G
    subgraph Loop["LangGraph loop (graph.py), bounded by max_tries"]
        G[generate<br/>agent.py] --> X[test<br/>sandbox.py]
        X -- all tests pass --> E((END))
        X -- budget spent --> E
        X -- failed --> F[feedback<br/>feedback.py]
        F --> G
    end
    X -. per-test verdicts .-> DB[(SQLite<br/>storage.py)]
    G -. code, tokens, latency .-> DB
    DB --> M[metrics.py]
    M --> R[regression.py<br/>McNemar / permutation]
    M --> H[report.py<br/>HTML report]
    M --> D[dashboard/app.py<br/>Streamlit]
```

| Module | Role |
|---|---|
| `agent_eval/sandbox.py`, `_harness.py` | Runs candidate code in an isolated subprocess and returns a verdict per test |
| `agent_eval/feedback.py` | Turns test results into feedback at level `none` / `minimal` / `names` / `full` |
| `agent_eval/prompts.py` | Versioned prompt sets (`v1`, `v2`) |
| `agent_eval/agent.py` | Provider-agnostic `generate()`: any OpenAI-compatible endpoint (Gemini by default) or the offline simulator |
| `agent_eval/simulated.py` | Deterministic offline `sim-*` models, so the pipeline runs without an API key |
| `agent_eval/graph.py` | The generate → test → feedback state machine, with the stopping rule as a conditional edge |
| `agent_eval/storage.py` | SQLite history of runs, attempts, feedback and per-test results |
| `agent_eval/metrics.py` | pass@k, lift, recovery, error transitions, retry dynamics, cost |
| `agent_eval/regression.py` | Paired run comparison with significance tests |
| `agent_eval/tasks.py` | Task loader, suite fingerprint, mutation-score validator |
| `agent_eval/judge.py` | Optional LLM judge for readability and approach (never correctness) |
| `agent_eval/report.py` | Self-contained HTML report |
| `dashboard/app.py` | Interactive Streamlit dashboard |
| `agent_eval/training_tasks.py` | Procedural training-task generator (38 families), kept separate from the eval suites |
| `agent_eval/ml/local_model.py` | Local Hugging Face/PyTorch models served through LangChain (`hf:<model>[+<adapter>]`) |
| `agent_eval/ml/verifier.py` | A pass/fail verifier written from scratch in PyTorch, used to rerank best-of-N candidates |
| `agent_eval/ml/sft.py`, `finetune.py`, `selftrain.py` | Verified trajectories → LoRA fine-tuning → self-training rounds with a learning curve |

## Design decisions

**Correctness is deterministic, and the LLM judge is secondary.** A task passes only if every hidden test passes in the sandbox. An LLM is never asked whether code is correct: that would be slow, costly, inconsistent between runs, and easy to game. The optional judge (`--judge`) scores readability and approach on an anchored 1–5 rubric with structured output. It runs only on code that already passed, and it returns nothing on any error. Cheap, reproducible static metrics (lines of code, cyclomatic complexity, nesting) are always recorded as well.

**The sandbox reports each failure as a result, never an exception.** Each attempt runs in a fresh temp directory under `python -I -S`:
- isolated mode, with no site-packages, so the code gets the stdlib only;
- an environment stripped to an allowlist, so API keys never reach generated code;
- POSIX rlimits on CPU, memory and file size;
- its own process group, killed as a whole on the global timeout.

Each test gets its own SIGALRM timeout, raised as a `BaseException` so `except Exception` in generated code can't swallow it. Results stream out as JSON lines, so partial results survive a killed process. Every `assert actual == expected` is rewritten through the AST so failures report expected vs. actual values, as pytest does. Every failure is classified: `syntax_error`, `import_error`, `missing_entry_point`, `timeout`, `memory_error`, `runtime_error`, `wrong_answer` or `crash`.

**Hidden-test policy.** The agent never sees the test file. Full feedback shows, for up to 5 failing tests, the test name, the single failing assertion line, expected vs. actual values, and a trimmed traceback. That is what a developer sees in a pytest report. There is special guidance for timeouts ("likely an infinite loop"), syntax errors and import errors ("stdlib only"). Feedback is truncated to about 3.5k characters.

**Feedback is an experimental variable.** The four levels (`none`, `minimal`, `names`, `full`) make "does richer feedback help?" a measured result instead of a claim. The first attempt never sees feedback, so for a fixed seed pass@1 is identical across levels and the ablation is a paired experiment.

**The loop is an explicit graph.** LangGraph models `generate → test → (passed | tries ≥ max_tries ? END : feedback → generate)`. The stopping rule is a named function (`route_after_test`) with its own unit test, not a condition buried in a while-loop.

**The suite is validated too.** Each of the 20 original tasks ships a reference solution and 3–6 plausible buggy *mutants* (off-by-one errors, ignored edge cases, wrong tie-breaking, crashes, infinite loops, syntax or import errors). `validate-tasks` and the test suite require every reference to pass and every mutant to be caught: **mutation score 100% (88/88)**. If the tests can't tell a subtle bug from a fix, the pass rates mean nothing. `suite_hash` fingerprints prompts and tests, and `compare` warns if two runs weren't scored on the same suite.

**Comparisons are paired and carry statistics.** Runs are compared task by task. The detector flags a regression if any task newly fails, if pass rate drops more than 5 pp, if pass@1 drops more than 10 pp, or if mean tries rises by more than 0.5. Every flag comes with an exact McNemar test on the discordant tasks, for both final pass and first-try pass, and a paired bootstrap interval. Repeated runs (`--seeds N`) can be compared as groups, using an exact permutation test and a pooled (seed, task) McNemar test.

**Offline simulator.** No API key? The `sim-strong`, `sim-base` and `sim-weak` models "write" either a task's reference solution or one of its mutants. A failure lowers the chance of fixing the task on a blind retry, and feedback can only raise it. The simulator reads the feedback text: with full feedback it turns the shown assertions into probe tests and runs its candidate fixes against them in the sandbox. Its results are a demonstration of the measurement system, **not claims about any real LLM**.

## Task suites

The easy suite is useful as a baseline, but classic "write a function from a docstring" problems are saturated: frontier models pass nearly all of them on the first try, which leaves the retry loop nothing to measure. The hard suite targets the ways strong models actually fail.

| Suite | Tasks | Hidden tests | Mutants | What it contains |
|---|---|---|---|---|
| `easy` (`tasks/tasks.json`) | 20 (6 easy, 8 medium, 6 hard) | 164 | 88 | Function-from-docstring problems with a few trap conventions |
| `hard` (`tasks/tasks_hard.json`) | 20 (16 hard, 4 medium) | 269 | 102 | Five tasks in each of four categories, below |

Hard-suite categories:
- **Bug-fixing:** 40–120 lines of realistic, buggy production-style code with a symptom-only bug report. Examples: refund cent allocation, keyset pagination, semver ranges, SLA business-hour deadlines, config deep merge. The original buggy code and plausible partial fixes are among the mutants.
- **Stateful classes:** a token-bucket rate limiter, a limit order book, a nested-transaction key-value store, a coalescing undo buffer, a card-hold ledger. Tests drive long call sequences with an injected clock.
- **Long specs:** a semver range matcher, a template engine, RFC 6902 JSON Patch, cron next-fire time, a TOML-subset parser. They have 15+ interacting rules, each one tested.
- **Performance:** inputs of 10⁵–10⁶ under a 1.5 s per-test limit. A correct but quadratic solution times out, so the agent has to read the timeout feedback and change algorithms.

Every task in both suites has a reference solution and mutants. `pytest` and CI require every reference to pass and every mutant to be caught, so both suites have a 100% mutation score. Runs record their suite, and every comparison, report and dashboard view keys on it, so easy and hard results never mix. Each run's summary also breaks pass@1 and the final pass rate down by category.

## Self-improvement with PyTorch

The loop's own ground truth (sandbox verdicts on thousands of attempts) is training data. Three PyTorch components use it, alongside the LangChain/LangGraph stack, not instead of it:

| Layer | Tool | Role |
|---|---|---|
| Orchestration | LangGraph | The bounded generate → test → feedback loop (unchanged) |
| Model interface | LangChain | One `invoke()` over Gemini (`ChatOpenAI`) and local PyTorch models (`ChatHuggingFace`) |
| Models and training | PyTorch + Hugging Face + PEFT | Local inference, LoRA self-training, the from-scratch verifier |
| Correctness | Sandbox + hidden tests | The ground truth every label and metric comes from |

**1. Local models (`--model hf:<repo-or-path>[+<adapter>]`).** Any Hugging Face causal LM, e.g. `hf:Qwen/Qwen2.5-Coder-0.5B-Instruct`, loads once per process on CUDA, Apple Silicon (MPS) or CPU. It runs through LangChain's `ChatHuggingFace`, so prompts, extraction, the sandbox and every metric work unchanged. A `+<adapter-dir>` suffix loads a LoRA adapter from self-training.

**2. A learned verifier, written from scratch.** `agent_eval/ml/verifier.py` predicts whether code will pass the hidden tests *without running it*. It uses no pretrained weights:
- a byte tokenizer
- a byte-patch embedding that groups 4 bytes into one token, making attention about 16× cheaper
- hand-written multi-head self-attention blocks
- [CLS] + mean + max pooling into a single logit

Training uses a pairwise ranking loss (a task's passing program should score above its failing ones), plus a little class-balanced BCE to keep scores calibrated. Code is normalized through the AST (docstrings and comments stripped) so the byte budget goes to logic. Without that, 22–73% of pass/fail pairs truncated to *identical* inputs with opposite labels, a data bug found while debugging a flat loss curve. Splits are by task, and the test set is the eval suites, which it never sees in training. In the loop, `--candidates N --verifier PATH` samples N programs per attempt and sends only the verifier's top pick to the sandbox. Candidate 1 is always the normal single-sample output, so best-of-N vs. single-sample is a paired comparison.

**3. Self-training (expert iteration).** `python main.py selftrain` runs rounds of:
1. evaluate on the held-out suites
2. run the loop on training tasks
3. turn verified successes into fine-tuning data
4. train a fresh LoRA adapter
5. evaluate again

The fine-tuning data has three kinds of example: first-try passes, **repairs** (failing code + feedback → the fix that passed, which teaches the model to use feedback), and **distilled retries** (the first prompt → code that only passed later). Only sandbox-verified code is ever a training target. The training suite (`tasks/tasks_train.json`, 346 tasks, 1,141 mutants) and the held-out suite (`tasks/tasks_heldout.json`, 110 tasks, 365 mutants) are built by `build-suites` from one pool of procedurally generated tasks across 38 families. The pool is deduplicated by solution fingerprint, so training has no repeated problems and no held-out problem appears in training. Every task is validated like the eval suites (reference passes, every mutant caught). `selftrain` refuses to run if the training and eval suites share a task. Each round is stored, and `python main.py curve`, the report and the dashboard's Self-training tab show the learning curve.

### Verifier results (measured)

> **Note:** the tables below were measured on the *earlier* 400-task training suite, before it was deduplicated. That suite repeated near-identical problems, which made "unseen generated tasks" easier than they look. Retrained on the deduplicated suite, the verifier scores ROC-AUC **0.65** on the held-out suite, **0.57** on easy and **0.50** on hard. Read the numbers below as an upper bound; the in-loop best-of-N run has not been repeated on the new suites.

The verifier was trained on the 400 generated training tasks (1,477 labeled programs from 340 tasks). It was validated on 60 *unseen* generated tasks and tested on the 40 hand-written benchmark tasks, none of which it saw in training.

| | Plain bytes | Identifiers canonicalized (default) |
|---|---|---|
| Unseen generated tasks: ROC-AUC | 0.73 | **0.84** |
| Unseen generated tasks: calibration error (ECE) | 0.21 | **0.13** |
| Benchmark suites: ROC-AUC | 0.45 (reversed) | 0.50 (chance) |

**In the loop** (sim-weak agent, one try, best-of-5 vs. single sample, 3 seeds, paired by seed and task):

| Tasks | Single sample | Best-of-5 + verifier | Paired McNemar |
|---|---|---|---|
| 60 unseen generated tasks | 43.3% | **66.7% (+23.3 pp)** | 60 gained / 18 lost, p = 2×10⁻⁶ |
| Easy benchmark suite | 38.3% | 31.7% (−6.7 pp) | p = 0.48 (n.s.) |
| Hard benchmark suite | 20.0% | 16.7% (−3.3 pp) | p = 0.79 (n.s.) |

What this shows:
- **A from-scratch verifier generalizes to unseen tasks from its training distribution,** and reranking with it raises pass@1 by 23 points.
- **It doesn't transfer to differently-written code.** Before identifier canonicalization it actively *hurt* the benchmark suites (−20 pp, p = 0.002): it had learned a vocabulary shortcut that holds in generated code and reverses in human-written code. Canonicalization removed the harm, but not the gap.
- **Validate a learned reranker on the distribution you'll use it on.** The project's own paired regression test is what caught the harmful version. Training on real model attempts (`--with-attempts`) is the natural next step to close the gap.

(The simulated agent's candidates are the tasks' reference solutions and mutants, so these numbers measure the verifier's ranking quality in the real loop. They are not claims about an LLM's raw ability.)

### Self-training results (measured)

Qwen2.5-Coder-0.5B-Instruct, 2 rounds, 20 training tasks collected per round, evaluated on the first 20 tasks of the held-out and easy suites, full feedback, 3 tries, `--warm-start`, `TEMPERATURE=0.7`, `LOCAL_MAX_NEW_TOKENS=1024`. Trained on a free Colab T4 GPU (LoRA r=16, 8.8M trainable of 503M parameters, about 8 minutes per round).

```
round  suite    examples repairs   pass@1    final  recovery    loss   vs round 0 (pass@1: +gained/-lost, McNemar p)
    0  heldout         0       0     5.0%     5.0%      0.0%       -
    0  easy            0       0     5.0%    10.0%      5.3%       -
    1  heldout       350       0    35.0%    35.0%      0.0%   0.045   +6/-0  p=0.0312
    1  easy          350       0     5.0%    10.0%      5.3%   0.045   +1/-1  p=1
    2  heldout       355       0    35.0%    35.0%      0.0%   0.002   +6/-0  p=0.0312
    2  easy          355       0    10.0%    10.0%      0.0%   0.002   +2/-1  p=1
```

| | Round 0 | Round 1 | Round 2 |
|---|---|---|---|
| Training tasks solved during collection | 20% | 75% | - |
| Model's own verified solutions in the next round's data | 4 | 19 | - |

What this shows:
- **Self-training works on held-out problems.** pass@1 went from 5% to 35% on tasks the model never trained on, with every flipped task in the right direction (p = 0.031). Round 2 held the gain without adding to it.
- **The loop feeds itself.** The trained model solved far more training tasks, so round 2 trained on 19 of its own verified solutions instead of 4.
- **The gain did not come from feedback.** The `repairs` column is 0 and held-out recovery is 0%: the 0.5B model never fixed its code after reading a test failure. Round 1's data was 346 warm-start reference solutions plus 4 of the model's own passes, so this run shows *rejection-sampling fine-tuning works*, not *feedback teaches the model*.
- **It did not transfer to the hand-written easy suite** (5% → 10%, p = 1). The held-out tasks come from the same 38 generated families as training; the easy suite is written differently.
- **The 75% collection rate is inflated:** with `--warm-start` the model trained on those tasks' reference solutions.
- **The sample is small** (20 tasks per suite), so treat the size of the jump as rough even though it is significant.

Next steps to test the feedback claim: a larger model that can actually use feedback (e.g. Qwen2.5-Coder-1.5B on a GPU), more training tasks per round so repairs can appear, and repair examples built from each task's mutants (buggy code + its real test failure → the reference fix) so the model is trained on fixing code directly.

### Proving the loop teaches a small model

Frontier models already pass most of these tasks, so they can't show learning. A small local model can: it fails often, has room to improve, and is cheap to fine-tune. The claim to test is: **"self-training on its own feedback-driven successes makes it better on problems it has never seen, and the feedback is what does it."**

1. **Build the data.**
   ```bash
   python main.py build-suites --pool 3000 --heldout-per-family 3
   ```
   This writes `tasks/tasks_train.json` and `tasks/tasks_heldout.json` from one pool with one task per distinct problem. Training has no repeats, and no held-out problem appears in training.

2. **Treatment run:** self-train with full feedback.
   ```bash
   python main.py selftrain --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --experiment fb-full \
       --rounds 3 --train-tasks 100 --feedback-level full
   ```

3. **Control run:** identical, but the loop's retries get no information.
   ```bash
   python main.py selftrain --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --experiment fb-none \
       --rounds 3 --train-tasks 100 --feedback-level none
   ```
   An alternative control keeps the feedback but drops the repair examples from training: `--kinds direct distill`.

4. **Read the result.**
   ```bash
   python main.py curve fb-full        # pass@1 per round, each round tested against round 0
   python main.py curve fb-none
   python main.py compare-experiments fb-full fb-none   # final models, paired task by task
   ```

The loop has been shown to teach the model if `fb-full` improves held-out pass@1 over round 0 with a small McNemar p-value **and** beats `fb-none` in `compare-experiments`. Round 0 runs the same base model in both experiments, so the two curves share a starting point.

Practical notes:
- Run a tiny version first to measure speed: `--rounds 1 --train-tasks 20 --eval-tasks 20`. Then size the real run from how long that took.
- Don't cut generations short. With `LOCAL_MAX_NEW_TOKENS=384` the code got truncated and most attempts were syntax errors; use `LOCAL_MAX_NEW_TOKENS=1024`.
- Use `TEMPERATURE=0.7` for local models. At temperature 0, retries resubmit identical code (35 of 40 retries were stuck in one run), so the loop can't help.
- If round 0 solves almost nothing on the training tasks, add `--warm-start`. It supervises round 1 on the training tasks' reference solutions to bootstrap. That's supervised data, not self-generated, so report it.
- `TRANSFORMERS_VERBOSITY=error` hides the library's warning spam.

**Hardware.** Generation with a 0.5B model works on an 8 GB Apple Silicon Mac (about 17 s per task on MPS), but **LoRA training did not**: it stalled on the first optimizer step for over an hour, swapping, because the fp32 model, Qwen's 151k-token vocabulary logits and macOS share 8 GB. Train on a CUDA GPU instead. A free Google Colab T4 runs a round in about 8 minutes. The verifier trains in minutes on CPU.

**Running on Google Colab** (Runtime → Change runtime type → T4 GPU):

```bash
!git clone https://github.com/shreykumar12/agent-feedback-loop   # private repo: https://<user>:<token>@github.com/...
%cd agent-feedback-loop
!pip install -q -r requirements.txt -r requirements-ml.txt
# Colab's preinstalled torch add-ons are built for its older torch and break transformers' imports
!pip uninstall -y torchvision torchaudio torchcodec torchao
!python -c "from transformers import BloomPreTrainedModel; import peft; print('imports OK')"
!python main.py init-db
!TRANSFORMERS_VERBOSITY=error LOCAL_MAX_NEW_TOKENS=1024 TEMPERATURE=0.7 python main.py selftrain \
    --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --experiment my-run --rounds 2 \
    --train-tasks 20 --eval-tasks 20 --warm-start
!python main.py curve my-run
```

pip prints dependency-conflict warnings about Colab's own packages; they don't matter. Results live in `data/agent_eval.db` inside the Colab session, so download it before the session ends, then put it in `data/` locally to use `curve`, `report` or the dashboard.

```bash
pip install -r requirements.txt -r requirements-ml.txt

# Learned verifier: train on generated tasks, test on the held-out eval suites
python main.py train-verifier --train-suites train --eval-suites easy hard --out models/verifier.pt
python main.py run --model sim-weak --candidates 5 --verifier models/verifier.pt --seeds 5

# Local model through LangChain, then self-training rounds
python main.py run --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --tasks caesar_shift clamp_all
python main.py selftrain --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --experiment qwen05-r3 \
    --rounds 3 --train-tasks 200 --warm-start
python main.py curve qwen05-r3

# Regenerate the training suite (deterministic)
python main.py gen-train-tasks --n 400 --seed 0
```

## Setup

Requires Python ≥ 3.12, because the pinned dependencies need it.

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY for real runs (optional)
python main.py init-db
```

## Usage

```bash
# Offline, no key needed: the simulated agent runs the full pipeline in about 2 s
python main.py run --model sim-base
python main.py show latest                 # full metrics summary
python main.py attempts latest rank_players  # attempts, feedback and code for one task

# Real model (Gemini through its OpenAI-compatible endpoint)
python main.py run --model gemini-3.8-flash --prompt-version v1 --feedback-level full
python main.py run --model gemini-3.8-flash --prompt-version v2 --seeds 3 --judge
python main.py run --model gemini-3.8-flash --rpm 5          # pace a rate-limited (free-tier) key

# The hard suite (bug-fixing, stateful classes, long specs, performance limits)
python main.py run --model gemini-3.8-flash --suite hard
python main.py show latest                                   # includes the by-category breakdown
python main.py compare @sim-strong/full/v1/hard @sim-weak/full/v1/hard   # compare within one suite

# Feedback-level ablation: models x levels x seeds
python main.py ablation --models sim-strong sim-base sim-weak --seeds 5

# Regression detection (exit code 1 if a regression is flagged, for CI gating)
python main.py runs
python main.py compare <baseline_run_id> <candidate_run_id>   # prefixes, latest, latest~1 work
python main.py compare @sim-strong/full @sim-weak/full        # all seeds of two configs

# Reporting
python main.py report --compare <baseline> <candidate>        # writes reports/report.html
streamlit run dashboard/app.py

# Tests and the suite's own validity check
pytest
python main.py validate-tasks --suite easy
python main.py validate-tasks --suite hard
```

![Streamlit dashboard](docs/img/dashboard-overview.png)

The dashboard has seven tabs: run overview, configuration leaderboard and ablation, regression comparison, per-attempt task drill-down, per-task solve matrix, code-quality metrics, and the self-training learning curve.

Any OpenAI-compatible endpoint works (OpenAI, OpenRouter, Ollama, vLLM): set `LLM_BASE_URL` and `LLM_API_KEY`. The sandbox limits, try budget and paths are all env-overridable (see `.env.example`).

## Metrics

| Metric | Meaning |
|---|---|
| **pass@1** | First-attempt pass rate, with no feedback |
| **pass@k** | Cumulative pass rate if the loop stopped after k tries |
| **final pass rate** | pass@max_tries, with a Wilson 95% interval |
| **self-correction lift** | final − pass@1: what the loop buys |
| **recovery rate** | Share of first-try failures that were later fixed |
| **tries-to-pass** | Distribution and mean among solved tasks |
| **marginal gain per try** | pass@k − pass@(k−1): do retries plateau? |
| **error-state transitions** | Verdict on try k → verdict on try k+1 (e.g. `runtime_error → wrong_answer`) |
| **partial credit per try** | Mean share of tests passed at each try: is the agent converging? |
| **retry outcomes** | Whether each retry improved, stayed the same or regressed the test-pass share; **stuck** = resubmitted identical code |
| **edit ratio** | How much code changed between tries: a targeted fix or a rewrite |
| **cost** | Tokens in/out, estimated USD (priced models), tokens per solved task, latency p50/p95 |

## Results

> **Read this first:** there was no API key in the environment where these numbers were produced, so they come from the offline **simulated** agent (60 suite runs: 3 sim models × 4 feedback levels × 5 seeds). They show what the system measures and how the analysis reads; they are not benchmark results for a real LLM. `python main.py ablation --models gemini-3.8-flash --seeds 3` produces the real version. The real-model self-training numbers are in [Self-training results](#self-training-results-measured).

### Feedback-level ablation (final pass rate, mean ± sd over 5 seeds, max 3 tries)

| model | pass@1 | none | minimal | names | full |
|---|---|---|---|---|---|
| sim-strong | 73.0% ± 11.0 | 85.0% ± 11.7 | 85.0% ± 11.7 | 97.0% ± 4.5 | **100.0% ± 0.0** |
| sim-base | 60.0% ± 14.1 | 82.0% ± 11.0 | 82.0% ± 11.0 | 92.0% ± 7.6 | **96.0% ± 6.5** |
| sim-weak | 41.0% ± 9.6 | 59.0% ± 10.8 | 59.0% ± 10.8 | 77.0% ± 12.0 | **87.0% ± 11.0** |

- Richer feedback raises the final pass rate at every model strength, and it matters most for the weakest model: +28 pp from minimal to full for sim-weak, against +15 pp for sim-strong.
- With content-free feedback (`none` / `minimal`), 41% of retries resubmit identical code. With `names` or `full`, almost none do.
- `none` and `minimal` coincide here because neither carries information the simulator can use. With a real model, the difference measures whether knowing "tests failed" changes behavior at all.
- Paired over (seed, task), downgrading sim-base from `full` to `minimal` loses 14 tasks and gains 0 (McNemar p = 1.2e-4).

### Tries-to-pass and error transitions (sim-base, full feedback, 5 seeds × 20 tasks)

- Solved on try 1: 60, try 2: 28, try 3: 8, never: 4. Most of the lift arrives on the first retry; the third try adds 8 pp.
- Most common transitions: `wrong_answer → passed` 28, `runtime_error → passed` 7, `wrong_answer → runtime_error` 5, `runtime_error → wrong_answer` 4. Of 52 retries, 41 improved the test-pass share, 7 stayed flat and 4 regressed.
- Hardest tasks by first-try rate across all 60 runs: `max_booking_value` 20%, `evaluate_expression` 27%, `shortest_path` 27%, `lru_simulate` 33%, `window_maxima` 33%.

### Example regression comparison: model swap sim-strong → sim-weak (seed 0, full feedback)

```
metric                    baseline   candidate       delta
pass_rate                   100.0%       90.0%      -10.0%
pass_at_1                    70.0%       30.0%      -40.0%
self_correction_lift         30.0%       60.0%      +30.0%
mean_tries_to_pass            1.30        1.83       +0.53
McNemar exact p (final pass): 0.500 (not significant)
McNemar exact p (pass@1):     0.021 (significant; first-try lost 9, gained 1)
newly failing (2): parse_duration, shortest_path
VERDICT: REGRESSION
```

**The loop masks regressions.** The final pass rate fell only 10 pp, because retries rescued most of the weaker model's failures, while pass@1 fell 40 pp. A detector that watched only the final pass rate would call this a small, insignificant change. That is why `compare` tests first-try outcomes separately. Over all 5 seeds (`compare @sim-strong/full @sim-weak/full`), pass@1 drops 32 pp with permutation p = 0.016.

**Noise is real.** An A/A comparison of *the same configuration* with seed 0 vs. seed 1 trips the single-run pass@1 threshold (−20 pp) purely by chance. The group comparison of seeds {0, 1} vs. {2, 3, 4} correctly reports no significant regression. On a 20-task suite, don't call a regression from one run: use `--seeds`.

## Testing

`pytest` runs 176 tests in about 3.5 minutes, with no API key and no model download, so CI runs them on every push. The 28 ML tests skip automatically if PyTorch isn't installed.
- **ML:** a tiny locally built Llama runs through LangChain and the loop; SFT export only targets sandbox-verified code; LoRA lowers the loss and its adapter loads back; the verifier learns and round-trips; AUC and selection math (ties never broken by label); best-of-N is paired with single-sample and beats it with an oracle verifier; self-training rounds record a curve and refuse train/eval overlap; the training generator is deterministic, varied, valid, and disjoint from the eval suites.
- **Sandbox:** timeouts that `except Exception` can't swallow, the global kill backstop, memory limit, secret stripping, isolation between runs, `sys.exit`.
- **Feedback:** content at each level, truncation, and the hidden-test policy.
- **Graph:** a scripted fake agent covers first-try pass, pass after retries, the max-tries stop, and that retry prompts receive the previous code and feedback.
- **Storage:** round-trips.
- **Metrics and regression:** hand-built runs with known answers, plus the McNemar, permutation and bootstrap math.
- **Task suite validity:** in both suites, every reference solution passes and every mutant is caught; performance tasks must declare a time limit; task ids are unique across suites.
- **Suites:** easy-suite prompts render exactly as before, old databases migrate in place, and per-task time limits apply.
- **Simulator:** more information never lowers the pass rate.
- **Report and dashboard:** smoke tests.

## Limitations

- **The subprocess sandbox is not strong isolation.** rlimits and a stripped environment stop accidents (infinite loops, memory blowups, leaked keys), not an adversary. The code can still read the filesystem and open network sockets. `RLIMIT_NPROC` is not set because it is per-user. For untrusted models, run each attempt in Docker, gVisor or Firecracker.
- **A small suite means noisy metrics.** With 20 tasks, one flipped task moves the pass rate by 5 pp and the 95% interval on a single run is about ±20 pp (13/20 → 43–82%). Even at temperature 0, provider-side nondeterminism means two identical runs can differ. Repeat runs, and prefer the paired and group tests over raw deltas.
- **Benchmark contamination.** The tasks were written for this project rather than copied from HumanEval or MBPP, but the easy suite is HumanEval-*style*, and close variants likely exist in training data. Pass@1 on classic problems overstates ability on novel ones. The hard suite reduces this, since its bug-fix code, class specs and rule sets are original, but it doesn't eliminate it: semver, cron and JSON Patch are well-known standards.
- **The verifier sees code, not behavior.** It never executes anything, so it can only learn bug patterns that show up in the text. Long programs are truncated at 2,048 normalized bytes, which still makes 28% of the hard suite's pass/fail pairs indistinguishable to it. Treat it as a reranker that biases sampling, never as a replacement for the hidden tests.
- **The self-training result is small-scale.** One 0.5B model, 2 rounds, 20 eval tasks per suite. The held-out gain is significant, but it relied on `--warm-start` (supervised reference solutions) and didn't transfer to the hand-written easy suite. The claim that *feedback* drives the improvement is untested so far: this model produced no feedback repairs, so a feedback vs. no-feedback control (`fb-full` vs. `fb-none`) needs a model that can use feedback first.
- **Small models barely use feedback.** At 0.5B, recovery after a failed first try was 0–5%. Earlier runs also showed habits that waste retries: calling undefined helpers, printing at module level, and pasting the error text back into the code.
- **The hard suite is not a repository benchmark.** Each task is still a single module. Multi-file, SWE-bench-style repair tasks would need the sandbox to copy a repo and run its own test suite.
- **Simulated results are illustrative.** The `sim-*` models' behavior is designed. Only how feedback is used on a retry emerges from the feedback content, via the sandbox probes. The ablation shape above validates the pipeline, not a hypothesis about LLMs.
- **The judge is optional and unvalidated.** LLM quality scores are not checked against human ratings. Treat them as a weak signal; correctness never depends on them.
- **Cost figures** use a static price table in `config.py`. Check them against current provider pricing.

## Project layout

```
agent_eval/        package: sandbox, harness, feedback, prompts, agent, simulator, graph,
                   storage, metrics, regression, tasks, judge, report (+ HTML template)
agent_eval/ml/     PyTorch: local models, LoRA fine-tuning, self-training rounds, verifier
dashboard/app.py   Streamlit dashboard
tasks/tasks.json   easy suite: 20 tasks (prompt, hidden tests, reference solution, mutants)
tasks/tasks_hard.json  hard suite: 20 bug-fix / stateful / spec / performance tasks
tasks/tasks_train.json    training suite: 346 generated tasks (self-training and the verifier only)
tasks/tasks_heldout.json  held-out suite: 110 generated tasks, never trained on
tests/             pytest suite (no API key needed)
docs/DEV_NOTES.md  design log and findings
main.py            CLI
```
