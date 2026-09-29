"""AgentEval: a self-correcting coding agent with deterministic evaluation.

The public entry points are ``run_task`` (one task through the retry loop) and
``run_suite`` (a whole suite as one stored, comparable run). They are imported
lazily so that light modules (sandbox, feedback, metrics) stay importable
without pulling in LangGraph.
"""

__all__ = ["run_task", "run_suite"]


def __getattr__(name):
    if name == "run_task":
        from agent_eval.graph import run_task

        return run_task
    if name == "run_suite":
        from agent_eval.run_suite import run_suite

        return run_suite
    raise AttributeError(name)
