# WHY THIS FILE EXISTS:
#   Verifies the loop's control flow -- the stopping conditions are the core
#   engineering claim of the project. Uses a FAKE agent so it's deterministic,
#   free, and runs in CI without an API key.
#
# WHAT IT NEEDS:
#   Monkeypatch agent.generate_code to return scripted code sequences:
#   - correct on try 1 -> passed, tries_taken=1, generate called once
#   - wrong, wrong, correct -> passed, tries_taken=3, feedback passed on retries
#   - always wrong -> stops at MAX_TRIES, passed=False, tries_taken=MAX_TRIES
#   - retry prompt receives previous code + feedback

import pytest


@pytest.mark.skip(reason="TODO: implement once graph.run_task exists")
def test_passes_first_try():
    pass


@pytest.mark.skip(reason="TODO")
def test_passes_after_retries():
    pass


@pytest.mark.skip(reason="TODO")
def test_stops_at_max_tries():
    pass
