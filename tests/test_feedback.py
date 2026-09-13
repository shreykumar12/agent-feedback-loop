# WHY THIS FILE EXISTS:
#   Feedback is what the agent learns from on each retry. These tests lock in
#   that the right information (and only that) reaches the prompt.
#
# WHAT IT NEEDS:
#   Build TestRunResult objects by hand and assert:
#   - "full" feedback contains failing test names, errors, expected vs actual
#   - passing cases are not listed as failures
#   - timeouts / syntax errors get their special-case messages
#   - very long tracebacks are truncated
#   - each feedback level ("minimal", "names", "full") includes what it should

import pytest


@pytest.mark.skip(reason="TODO: implement once feedback.build_feedback exists")
def test_full_feedback_includes_failures():
    pass


@pytest.mark.skip(reason="TODO")
def test_timeout_feedback():
    pass
