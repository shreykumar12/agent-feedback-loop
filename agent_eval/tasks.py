# WHY THIS FILE EXISTS:
#   Loads the task suite (build step 3) from tasks/tasks.json into Task
#   objects. Separating data from code means you can add/remove problems
#   without touching the loop, and keeps the suite identical across runs --
#   a requirement for fair regression comparison.
#
# WHAT IT NEEDS:
#   - load_tasks(path=config.TASKS_PATH, task_ids=None) -> list[Task]
#   - Validate each entry has task_id, prompt, entry_point, test_code.
#   - Optional filter by task_ids so you can debug a single task quickly.
#   - Optionally: a helper to import a subset of HumanEval/MBPP problems into
#     the JSON format (keep the suite small: 15-20 tasks).

from pathlib import Path

from agent_eval import config
from agent_eval.models import Task


def load_tasks(path: Path = config.TASKS_PATH, task_ids: list[str] | None = None) -> list[Task]:
    raise NotImplementedError
