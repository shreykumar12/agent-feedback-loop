# WHY THIS FILE EXISTS:
#   The regression comparator decides what "got worse" means. A bug here
#   silently hides regressions, which defeats the point of the system.
#
# WHAT IT NEEDS:
#   Seed a temp DB with two hand-built runs and assert:
#   - pass rate and mean tries deltas are computed correctly
#   - newly_failing / newly_passing / more_tries lists are correct
#   - tasks present in only one run are reported, not silently compared

import pytest


@pytest.mark.skip(reason="TODO: implement once regression.compare_runs exists")
def test_detects_newly_failing_task():
    pass


@pytest.mark.skip(reason="TODO")
def test_pass_rate_delta():
    pass
