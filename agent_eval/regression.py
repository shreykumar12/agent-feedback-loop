# WHY THIS FILE EXISTS:
#   Build step 5: answer "after changing the model or prompt, did things get
#   worse?" Comparing two stored runs and flagging regressions is what turns
#   this from a demo into an evaluation system.
#
# WHAT IT NEEDS:
#   - compare_runs(baseline_run_id, candidate_run_id) -> dict with:
#       * aggregate: pass rate (both runs + delta), mean tries-to-pass (+ delta)
#       * newly_failing: tasks that passed in baseline, fail in candidate
#       * newly_passing: the reverse
#       * more_tries: tasks that passed in both but took more tries
#   - Only compare tasks present in BOTH runs (report any mismatch).
#   - Thresholds for "flag as regression" (e.g. any newly failing task, or
#     pass-rate drop > X%) -- keep them explicit constants.
#   - A human-readable report formatter for the CLI.
#   - Note in the README: with a small suite, run-to-run noise is real even at
#     temperature 0; consider repeated runs before calling a regression.

def compare_runs(baseline_run_id: str, candidate_run_id: str) -> dict:
    raise NotImplementedError


def format_report(comparison: dict) -> str:
    raise NotImplementedError
