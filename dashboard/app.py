# WHY THIS FILE EXISTS:
#   Build step 7: a metrics UI so results are visible at a glance instead of
#   buried in SQLite. Streamlit keeps the frontend to a single Python script,
#   so effort stays on the loop, not UI plumbing.
#
# WHAT IT NEEDS:
#   Run with: `streamlit run dashboard/app.py`
#   Read ONLY through agent_eval.storage (no raw SQL here). Sections:
#   - Run selector: list runs (model, prompt_version, timestamp).
#   - Headline metrics (st.metric): pass rate, mean tries-to-pass, # tasks.
#   - Tries-to-pass distribution: bar chart of how many tasks passed on try
#     1, 2, 3... and how many never passed. Shows whether retries help or plateau.
#   - Pass rate by attempt number (cumulative): the "did feedback help" curve.
#   - Regression comparison: pick baseline + candidate run, show
#     regression.compare_runs output, highlight newly failing tasks.
#   - Task drill-down: pick a task, show each attempt's code, test results,
#     and the feedback that was given -- the best debugging view.
#   - Quality scores table if the judge was used.
