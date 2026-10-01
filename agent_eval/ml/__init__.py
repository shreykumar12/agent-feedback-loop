"""PyTorch side of AgentEval: local models, self-training, and the learned verifier.

Everything here needs the optional ML dependencies (``pip install -r
requirements-ml.txt``). The rest of the package imports these modules lazily,
so the core loop, sandbox and metrics work without PyTorch installed.
"""
