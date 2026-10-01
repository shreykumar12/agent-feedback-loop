# agent-feedback-loop

Big models like GPT and Claude already solve most coding problems on the first try, so they don't have much left to learn. I wanted to see whether a small model could get better at coding by training on its own work.

The setup: Qwen2.5-Coder-0.5B writes a Python function, a sandbox runs hidden tests on it, and if it fails, the test output goes back to the model for another try. Any code that passes gets saved. After each round I fine-tune the model on everything it got right, then test it on problems it has never seen.

## Result

One round of self-training took held-out pass@1 from **5% to 35%**. Six tasks flipped from fail to pass and none went the other way (McNemar p = 0.031).

```
round  suite    pass@1   final   vs round 0
    0  heldout    5.0%    5.0%
    1  heldout   35.0%   35.0%   +6/-0  p=0.031
    2  heldout   35.0%   35.0%   +6/-0  p=0.031
    0  easy       5.0%   10.0%
    2  easy      10.0%   10.0%   +2/-1  p=1
```

The trained model also produced more of its own training data: it solved 19 training tasks in round 1, up from 4 in round 0.

The model never actually fixed its code from test feedback (0 repairs, 0% recovery). So the improvement came from training on verified solutions, not from the feedback itself. The gain also didn't carry over to the hand-written `easy` suite. My next step is testing whether a bigger model (1.5B) can learn to use feedback. I also plan to generate repair examples from each task's buggy mutants, so the model gets trained on fixing code directly.

## How it's built

- **Sandbox:** each attempt runs in an isolated Python process with CPU and memory limits and a timeout on every test. Failures are classified (syntax error, wrong answer, timeout, ...) and turned into pytest-style feedback.
- **Loop:** a LangGraph state machine (generate → test → feedback → retry), capped at 3 tries.
- **Training:** LoRA on PyTorch + PEFT. Only about 2% of the weights are trained, and the base model stays frozen. Only code that passed every hidden test ever becomes a training example.
- **Benchmark:** 456 generated tasks across 38 problem families, split into 346 train and 110 held-out. They're deduplicated so no held-out problem ever shows up in training. Every task has buggy mutants the tests are required to catch, so a passing score means something.
- **Stats:** every round is compared task by task against the base model with a paired significance test.

It also has a Streamlit dashboard and HTML reports, plus 176 tests.

## Running it

Training needs a GPU. My 8 GB Mac could generate code but stalled on training, so I used a free Colab T4, where each round trains in about 8 minutes.

On Colab (Runtime → Change runtime type → T4 GPU):

```bash
!git clone https://github.com/shreykumar12/agent-feedback-loop
%cd agent-feedback-loop
!pip install -q -r requirements.txt -r requirements-ml.txt
!pip uninstall -y torchvision torchaudio torchcodec torchao   # Colab's versions break transformers
!python main.py init-db
!TRANSFORMERS_VERBOSITY=error LOCAL_MAX_NEW_TOKENS=1024 TEMPERATURE=0.7 python main.py selftrain \
    --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --experiment my-run --rounds 2 \
    --train-tasks 20 --eval-tasks 20 --warm-start
!python main.py curve my-run
```

A few things I learned the hard way:
- Use `LOCAL_MAX_NEW_TOKENS=1024`. Shorter limits cut the code off mid-function.
- Use `TEMPERATURE=0.7`. At 0, every retry is the exact same code.
- Keep `--warm-start`. The base model solves almost nothing alone, so round 1 also trains on the training tasks' reference solutions to get it started.
- Download `data/agent_eval.db` before the Colab session ends. That's where all the results live.

Other commands:

```bash
python main.py run --model hf:Qwen/Qwen2.5-Coder-0.5B-Instruct --suite heldout   # one eval run
python main.py show latest                                                        # metrics
python main.py compare-experiments fb-full fb-none                                # feedback vs. no feedback
streamlit run dashboard/app.py
pytest
```
