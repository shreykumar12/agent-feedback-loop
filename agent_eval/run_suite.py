# WHY THIS FILE EXISTS:
#   Ties everything together for build step 3+4: create a run record, loop
#   over every task, execute the graph, optionally judge quality, and persist
#   it all. This is the unit of comparison for regression detection -- one
#   call = one run_id under a fixed (model, prompt_version) configuration.
#
# WHAT IT NEEDS:
#   - run_suite(model, prompt_version, task_ids=None, use_judge=False) -> run_id
#       1. storage.init_db()
#       2. Generate run_id (uuid) + timestamp, storage.create_run(...)
#       3. For each task from tasks.load_tasks():
#            - storage.save_task(task)
#            - result, attempts = graph.run_task(task)
#            - if use_judge and result.passed: judge.score_quality(...)
#            - storage.save_attempt(...) for each attempt, storage.save_result(...)
#       4. Print a summary: pass rate, mean tries-to-pass, failed task ids.
#   - Catch unexpected per-task exceptions, record the task as failed, and keep
#     going -- one broken task must not kill a whole suite run.

def run_suite(
    model: str,
    prompt_version: str,
    task_ids: list[str] | None = None,
    use_judge: bool = False,
) -> str:
    raise NotImplementedError
