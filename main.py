# WHY THIS FILE EXISTS:
#   The command-line entry point. Gives one obvious way to run the system
#   (and for CI/README instructions to reference) without importing modules
#   by hand.
#
# WHAT IT NEEDS:
#   argparse subcommands:
#   - `python main.py init-db`
#       -> storage.init_db()
#   - `python main.py run --model gemini-2.5-flash --prompt-version v1 [--tasks id1 id2] [--judge]`
#       -> run_suite.run_suite(...), print the run_id
#   - `python main.py compare <baseline_run_id> <candidate_run_id>`
#       -> regression.compare_runs + format_report; exit code 1 if a
#          regression is flagged (useful for CI gating)
#   - `python main.py runs`
#       -> storage.list_runs() as a table
#   Keep logic out of here -- it should only parse args and call the package.


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
