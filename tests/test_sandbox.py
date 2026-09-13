# WHY THIS FILE EXISTS:
#   The sandbox is the source of truth for correctness. If it mis-scores code,
#   every metric downstream is wrong -- so it needs the strongest tests. These
#   need no API key, so they run in CI for free.
#
# WHAT IT NEEDS:
#   Hand-written code strings (no LLM) covering:
#   - correct solution -> passed=True, all cases pass
#   - wrong answer -> passed=False, failing case names + expected/actual populated
#   - syntax error -> passed=False, error captured, no exception raised
#   - infinite loop -> timed_out=True within the timeout
#   - code that tries to read os.environ secrets -> key not present
#   - partial pass -> correct per-test breakdown

import pytest


@pytest.mark.skip(reason="TODO: implement once sandbox.run_tests exists")
def test_correct_solution_passes():
    pass


@pytest.mark.skip(reason="TODO")
def test_wrong_answer_fails_with_details():
    pass


@pytest.mark.skip(reason="TODO")
def test_syntax_error_is_a_result_not_exception():
    pass


@pytest.mark.skip(reason="TODO")
def test_infinite_loop_times_out():
    pass
