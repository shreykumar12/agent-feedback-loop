"""Streamlit dashboard for AgentEval (build step 7).

Why this exists: results are only useful if they are visible at a glance
instead of buried in SQLite. Streamlit keeps the UI to one Python script so the
engineering effort stays on the generate -> test -> feedback -> retry loop.

It reads ONLY through ``agent_eval.storage`` / ``metrics`` / ``regression`` (no
raw SQL), so every number matches the CLI and the HTML report.

Run with:  streamlit run dashboard/app.py
The DB is ``agent_eval.config.DB_PATH`` (override with the AGENT_EVAL_DB env var).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

# `streamlit run dashboard/app.py` puts dashboard/ (not the project root) on
# sys.path, so make `import agent_eval` work regardless of the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from agent_eval import config, metrics, regression, storage  # noqa: E402
from agent_eval.feedback import FEEDBACK_LEVELS  # noqa: E402
from agent_eval.models import ERROR_TYPES  # noqa: E402
from agent_eval.report import config_key  # noqa: E402

# --- palette -----------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]
GRAY = "#b9b7ae"
SEQ_RAMP = ["#cde2fb", "#0d366b"]
BAR = 22  # bar thickness, px
POINT = 70  # point area (~9px marker)
CACHE_TTL = 30  # seconds
SIM_NOTICE = "Simulated agent — demonstrates the pipeline, not a real LLM result."
EMPTY_HINT = "python main.py run --model sim-base"
DIFF_ORDER = ["easy", "medium", "hard", "unknown"]


# --- small helpers -----------------------------------------------------------

def pct(v, digits: int = 1) -> str:
    return "–" if v is None else f"{v:.{digits}%}"


def pp(v) -> str:
    return "–" if v is None else f"{v * 100:+.1f} pp"


def num(v, digits: int = 2) -> str:
    return "–" if v is None else f"{v:,.{digits}f}"


def is_sim(model: str | None) -> bool:
    return bool(model) and model.startswith("sim")


def cat_scale(domain) -> alt.Scale:
    """Categorical hues in fixed order; anything past slot 4 folds to gray."""
    domain = list(domain)
    return alt.Scale(domain=domain, range=[CATEGORICAL[i] if i < len(CATEGORICAL) else GRAY
                                           for i in range(len(domain))])


def show_chart(chart: alt.Chart, data: pd.DataFrame, label: str = "Data") -> None:
    st.altair_chart(chart, width="stretch")
    with st.expander(label):
        st.dataframe(data, hide_index=True)


def run_label(r: dict) -> str:
    rate = "–" if r["pass_rate"] is None else f"{r['pass_rate']:.0%}"
    status = "" if r.get("status") in (None, "completed") else f" ({r['status']})"
    return (f"{r['run_id'][:8]} · {r['model']} · {r['prompt_version']} · fb={r['feedback_level']}"
            f" · s{r['seed']} · {rate}{status}")


def config_tuple(r: dict) -> tuple:
    return (r["model"], r["prompt_version"], r["feedback_level"], r["max_tries"])


def series_label(model: str, prompt: str, k: int) -> str:
    return f"{model} · {prompt} · k={k}"


def fb_rank(level: str) -> int:
    return FEEDBACK_LEVELS.index(level) if level in FEEDBACK_LEVELS else len(FEEDBACK_LEVELS)


def state_rank(state: str) -> int:
    if state == metrics.TERMINAL_PASS:
        return len(ERROR_TYPES) + 1
    return ERROR_TYPES.index(state) if state in ERROR_TYPES else len(ERROR_TYPES)


# --- cached data access (keyed on DB path + mtime so a new run shows up) -----

def db_key() -> str:
    path = Path(config.DB_PATH)
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        mtime = 0
    return f"{path.resolve()}:{mtime}"


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_runs(dbk: str) -> list[dict]:
    storage.init_db()
    return storage.list_runs()


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_results(dbk: str, run_id: str) -> list[dict]:
    return storage.get_results(run_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_attempts(dbk: str, run_id: str) -> list[dict]:
    return storage.get_attempts(run_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_task_attempts(dbk: str, run_id: str, task_id: str) -> list[dict]:
    return storage.get_attempts(run_id, task_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_task(dbk: str, task_id: str) -> dict | None:
    return storage.get_task(task_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_summary(dbk: str, run_id: str) -> dict:
    return metrics.load_summary(run_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def pooled_summary(dbk: str, run_ids: tuple[str, ...]) -> dict:
    """One summary over every seed of a configuration: task ids are suffixed with
    the run id so per-task grouping (tries, transitions) stays per run."""
    results, attempts, first_run = [], [], None
    for rid in run_ids:
        run = storage.get_run(rid)
        first_run = first_run or run
        tag = "@" + rid[:8]
        results += [{**r, "task_id": r["task_id"] + tag} for r in load_results(dbk, rid)]
        attempts += [{**a, "task_id": a["task_id"] + tag} for a in load_attempts(dbk, rid)]
    return metrics.summarize_run(first_run, results, attempts)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def compare_single(dbk: str, base_id: str, cand_id: str) -> dict:
    return regression.compare_runs(base_id, cand_id)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def compare_group(dbk: str, base_ids: tuple[str, ...], cand_ids: tuple[str, ...]) -> dict:
    return regression.compare_groups(list(base_ids), list(cand_ids))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def config_table(dbk: str) -> list[dict]:
    """Every configuration with its seeds' summaries aggregated (mean ± std)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in load_runs(dbk):
        groups[config_tuple(r)].append(r)
    rows = []
    for (model, prompt, fb, k), runs in groups.items():
        summaries = [load_summary(dbk, r["run_id"]) for r in runs]
        agg = metrics.aggregate_summaries(summaries)
        tps = [s["tokens_per_solved"] for s in summaries if s["tokens_per_solved"] is not None]
        rows.append({
            "key": (model, prompt, fb, k), "label": config_key(runs[0]),
            "model": model, "prompt_version": prompt, "feedback_level": fb, "max_tries": k,
            "run_ids": [r["run_id"] for r in runs], "seeds": sorted(r["seed"] for r in runs),
            "agg": agg, "tokens_per_solved": sum(tps) / len(tps) if tps else None,
        })
    rows.sort(key=lambda c: (c["model"], c["prompt_version"], c["max_tries"], fb_rank(c["feedback_level"])))
    return rows


# --- sections ------------------------------------------------------------------

def headline(summary: dict, agg: dict | None) -> None:
    """Six st.metric tiles. In configuration view the value carries ± std across seeds."""
    def tile(col, label, key, fmt, help_text):
        if agg is not None and key in agg and agg[key]["mean"] is not None:
            m = agg[key]
            std_fmt = (lambda v: f"{v * 100:.1f} pp") if fmt in (pct, pp) else (lambda v: f"{v:.2f}")
            value = f"{fmt(m['mean'])} ± {std_fmt(m['std'])}"
        else:
            value = fmt(summary.get(key))
        col.metric(label, value, help=help_text, border=True)

    lo, hi = summary["pass_rate_ci95"]
    ci_note = f"95% Wilson CI of the {'pooled' if agg else ''} pass rate: [{lo:.1%}, {hi:.1%}] over {summary['num_tasks']} task outcomes."
    cols = st.columns(3) + st.columns(3)
    tile(cols[0], "Final pass rate", "pass_rate", pct,
         f"Tasks solved within max_tries ({summary['max_tries']}). {ci_note}")
    tile(cols[1], "pass@1", "pass_at_1", pct, "Solved on the first try, before any feedback.")
    tile(cols[2], "Self-correction lift", "self_correction_lift", pp,
         "Final pass rate minus pass@1: what the feedback loop buys.")
    tile(cols[3], "Recovery rate", "recovery_rate", pct,
         f"Of {summary['num_failed_first']} first-try failures, {summary['num_recovered']} were fixed by a retry"
         + (" (pooled)." if agg else "."))
    tile(cols[4], "Mean tries-to-pass", "mean_tries_to_pass", num, "Mean attempt number among solved tasks.")
    tps = summary.get("tokens_per_solved")
    cols[5].metric("Tokens / solved task", "–" if tps is None else f"{tps:,.0f}",
                   help=f"(tokens in + out) / solved tasks. Total {summary['tokens_in']:,} in, "
                        f"{summary['tokens_out']:,} out; est. cost ${summary['est_cost_usd']:.4f}.",
                   border=True)
    rd = summary["retry_dynamics"]
    st.caption(
        f"{summary['num_tasks']} task outcomes · {summary['total_attempts']} attempts · "
        f"latency p50 {summary['latency_p50_s']:.2f}s / p95 {summary['latency_p95_s']:.2f}s · "
        f"retries: {rd['retries']} ({pct(rd['improved_rate'], 0)} improved, "
        f"{pct(rd['regression_rate'], 0)} regressed, {pct(rd['stuck_rate'], 0)} resubmitted identical code) · "
        f"infra errors: {summary['infra_errors']}"
    )


def chart_tries(summary: dict) -> None:
    rows = []
    for bucket, n in summary["tries_distribution"].items():
        failed = bucket == "failed"
        rows.append({"bucket": "never passed" if failed else f"try {bucket}",
                     "outcome": "never passed" if failed else "solved", "tasks": n,
                     "share": n / summary["num_tasks"] if summary["num_tasks"] else 0.0})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar(size=BAR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("bucket:N", sort=list(df["bucket"]), title="solved on", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("tasks:Q", title="tasks"),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["solved", "never passed"], range=[BLUE, GRAY]),
                        legend=None),
        tooltip=[alt.Tooltip("bucket:N", title="bucket"), alt.Tooltip("tasks:Q"),
                 alt.Tooltip("share:Q", format=".1%")],
    ).properties(height=260, title="Tries-to-pass distribution")
    show_chart(chart, df)


def line_chart(df: pd.DataFrame, x: str, y: str, title: str, y_title: str, color: str | None = None,
               color_scale: alt.Scale | None = None, extra_tooltip: list | None = None,
               height: int = 260) -> alt.Chart:
    enc = {
        "x": alt.X(f"{x}:O", title="try k", axis=alt.Axis(labelAngle=0)),
        "y": alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        "tooltip": [alt.Tooltip(f"{x}:O", title="try k"), alt.Tooltip(f"{y}:Q", title=y_title, format=".1%")]
        + (extra_tooltip or []),
    }
    base = alt.Chart(df)
    if color:
        # Streamlit fits legend + axes inside `height`, so give the legend its own room.
        height += 50
        enc["color"] = alt.Color(f"{color}:N", scale=color_scale, title=None, legend=alt.Legend(orient="top"))
        enc["tooltip"] = [alt.Tooltip(f"{color}:N")] + enc["tooltip"]
        line = base.mark_line(strokeWidth=2).encode(**enc)
        pts = base.mark_point(filled=True, size=POINT).encode(**enc)
    else:
        line = base.mark_line(strokeWidth=2, color=BLUE).encode(**enc)
        pts = base.mark_point(filled=True, size=POINT, color=BLUE).encode(**enc)
    return (line + pts).properties(height=height, title=title)


def chart_pass_at_k(summary: dict) -> None:
    df = pd.DataFrame({"k": list(range(1, len(summary["pass_at_k"]) + 1)), "pass_at_k": summary["pass_at_k"],
                       "marginal_gain": summary["marginal_gain"]})
    show_chart(line_chart(df, "k", "pass_at_k", "pass@k (cumulative)", "pass rate",
                          extra_tooltip=[alt.Tooltip("marginal_gain:Q", title="gain vs k-1", format="+.1%")]), df)


def chart_partial_credit(summary: dict) -> None:
    df = pd.DataFrame({"k": list(range(1, len(summary["partial_credit"]) + 1)),
                       "partial_credit": summary["partial_credit"]})
    show_chart(line_chart(df, "k", "partial_credit", "Partial credit per try",
                          "tests passed"), df)


def chart_errors(summary: dict) -> None:
    first = summary["first_attempt_errors"]
    rows = []
    for err, total in summary["error_counts"].items():
        n_first = first.get(err, 0)
        rows.append({"error_type": err, "phase": "first attempt", "order": 0, "attempts": n_first, "total": total})
        rows.append({"error_type": err, "phase": "retries", "order": 1, "attempts": total - n_first, "total": total})
    if not rows:
        st.info("No failed attempts in this selection — nothing to break down.")
        return
    df = pd.DataFrame(rows)
    order = list(df.drop_duplicates("error_type").sort_values("total", ascending=False)["error_type"])
    chart = alt.Chart(df).mark_bar(size=BAR, cornerRadiusEnd=4).encode(
        y=alt.Y("error_type:N", sort=order, title=None),
        x=alt.X("attempts:Q", title="failed attempts", stack="zero"),
        color=alt.Color("phase:N", scale=cat_scale(["first attempt", "retries"]), title=None,
                        legend=alt.Legend(orient="bottom")),
        order=alt.Order("order:Q"),
        tooltip=["error_type:N", "phase:N", "attempts:Q", alt.Tooltip("total:Q", title="all attempts")],
    ).properties(height=alt.Step(34), title="Error types of failed attempts")
    show_chart(chart, df.drop(columns="order"))


def chart_transitions(summary: dict) -> None:
    rows = []
    for key, n in summary["transitions"].items():
        src, dst = key.split(" -> ")
        rows.append({"from": src, "to": dst, "count": n})
    if not rows:
        st.info("No retries in this selection — no error-state transitions to show.")
        return
    df = pd.DataFrame(rows)
    states_from = sorted(set(df["from"]), key=state_rank)
    states_to = sorted(set(df["to"]), key=state_rank)
    base = alt.Chart(df).encode(
        x=alt.X("to:N", sort=states_to, title="state of try k+1",
                axis=alt.Axis(labelAngle=-30, labelAlign="right", labelOverlap=False)),
        y=alt.Y("from:N", sort=states_from, title="state of try k"),
    )
    rect = base.mark_rect(cornerRadius=3).encode(
        color=alt.Color("count:Q", scale=alt.Scale(range=SEQ_RAMP), title="retries"),
        tooltip=["from:N", "to:N", "count:Q"],
    )
    max_n = int(df["count"].max())
    text = base.mark_text(fontSize=12).encode(
        text="count:Q",
        color=alt.condition(alt.datum.count > max_n / 2, alt.value("white"), alt.value("#1a1a1a")),
        tooltip=["from:N", "to:N", "count:Q"],
    )
    chart = (rect + text).properties(height=alt.Step(40), title="Error-state transitions")
    show_chart(chart, df.sort_values("count", ascending=False))


def chart_difficulty(summary: dict) -> None:
    rows = []
    for diff, d in summary["by_difficulty"].items():
        rows.append({"difficulty": diff, "outcome": "first try", "order": 0, "share": d["pass_at_1"],
                     "tasks": d["n"], "final_pass_rate": d["pass_rate"]})
        rows.append({"difficulty": diff, "outcome": "rescued by retry", "order": 1, "share": d["lift"],
                     "tasks": d["n"], "final_pass_rate": d["pass_rate"]})
    if not rows:
        st.info("No task results yet.")
        return
    df = pd.DataFrame(rows)
    order = sorted(set(df["difficulty"]), key=lambda d: DIFF_ORDER.index(d) if d in DIFF_ORDER else 99)
    chart = alt.Chart(df).mark_bar(size=BAR * 2).encode(
        x=alt.X("difficulty:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("share:Q", stack="zero", title="share of tasks", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%")),
        color=alt.Color("outcome:N", scale=cat_scale(["first try", "rescued by retry"]), title=None,
                        legend=alt.Legend(orient="bottom")),
        order=alt.Order("order:Q"),
        tooltip=["difficulty:N", "outcome:N", alt.Tooltip("share:Q", format=".1%"), "tasks:Q",
                 alt.Tooltip("final_pass_rate:Q", title="final pass rate", format=".1%")],
    ).properties(height=260, title="Pass rate by difficulty")
    show_chart(chart, df.drop(columns="order"))


def section_overview(summary: dict, agg: dict | None, scope: str) -> None:
    st.subheader(f"Run overview — {scope}")
    headline(summary, agg)
    c1, c2, c3 = st.columns(3)
    with c1:
        chart_tries(summary)
    with c2:
        chart_pass_at_k(summary)
    with c3:
        chart_partial_credit(summary)
    c1, c2, c3 = st.columns(3)
    with c1:
        chart_errors(summary)
    with c2:
        chart_transitions(summary)
    with c3:
        chart_difficulty(summary)


def section_leaderboard(configs: list[dict]) -> None:
    st.subheader("Configuration leaderboard & feedback-level ablation")
    st.caption("Runs grouped by (model, prompt version, feedback level, max tries); mean ± std across seeds.")
    table = []
    for c in configs:
        a = c["agg"]
        table.append({
            "model": c["model"], "prompt": c["prompt_version"], "feedback": c["feedback_level"],
            "max_tries": c["max_tries"], "runs": a["num_runs"], "seeds": ", ".join(map(str, c["seeds"])),
            "pass_rate": a["pass_rate"]["mean"], "pass_rate_std": a["pass_rate"]["std"],
            "pass@1": a["pass_at_1"]["mean"], "pass@1_std": a["pass_at_1"]["std"],
            "lift": a["self_correction_lift"]["mean"], "lift_std": a["self_correction_lift"]["std"],
            "recovery": a["recovery_rate"]["mean"], "mean_tries": a["mean_tries_to_pass"]["mean"],
            "tokens/solved": c["tokens_per_solved"],
        })
    df = pd.DataFrame(table).sort_values("pass_rate", ascending=False)
    pct_cols = ["pass_rate", "pass_rate_std", "pass@1", "pass@1_std", "lift", "lift_std", "recovery"]
    st.dataframe(df, hide_index=True, column_config={
        **{col: st.column_config.NumberColumn(col, format="percent") for col in pct_cols},
        "mean_tries": st.column_config.NumberColumn(format="%.2f"),
        "tokens/solved": st.column_config.NumberColumn(format="%.0f"),
    })

    metric_choice = st.radio("Metric", ["final pass rate", "pass@1", "self-correction lift"], horizontal=True,
                             key="ablation_metric")
    agg_key = {"final pass rate": "pass_rate", "pass@1": "pass_at_1",
               "self-correction lift": "self_correction_lift"}[metric_choice]
    rows = []
    for c in configs:
        m = c["agg"][agg_key]
        if m["mean"] is None:
            continue
        rows.append({"series": series_label(c["model"], c["prompt_version"], c["max_tries"]),
                     "feedback_level": c["feedback_level"], "mean": m["mean"], "std": m["std"],
                     "lo": m["mean"] - m["std"], "hi": m["mean"] + m["std"], "runs": c["agg"]["num_runs"]})
    df = pd.DataFrame(rows)
    levels = sorted(set(df["feedback_level"]), key=fb_rank)
    series = sorted(set(df["series"]))
    scale = cat_scale(series)
    tip = ["series:N", "feedback_level:N", alt.Tooltip("mean:Q", format=".1%"),
           alt.Tooltip("std:Q", format=".1%"), "runs:Q"]
    x = alt.X("feedback_level:N", sort=levels, title="feedback level", axis=alt.Axis(labelAngle=0))
    color = alt.Color("series:N", scale=scale, title=None, legend=alt.Legend(orient="top"))
    base = alt.Chart(df)
    y_domain = [min(0.0, float(df["lo"].min())), 1]
    lines = base.mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("mean:Q", title=metric_choice, scale=alt.Scale(domain=y_domain), axis=alt.Axis(format="%")),
        color=color, tooltip=tip)
    pts = base.mark_point(filled=True, size=POINT).encode(x=x, y="mean:Q", color=color, tooltip=tip)
    bars = base.mark_rule(strokeWidth=2, opacity=0.6).encode(x=x, y="lo:Q", y2="hi:Q", color=color, tooltip=tip)
    show_chart((bars + lines + pts).properties(height=340, title=f"{metric_choice} vs feedback level (± std)"),
               df.drop(columns=["lo", "hi"]))
    st.markdown("**pass@k per configuration** (mean across seeds)")
    by_series: dict[str, list[dict]] = defaultdict(list)
    for c in configs:
        by_series[series_label(c["model"], c["prompt_version"], c["max_tries"])].append(c)
    all_levels = sorted({c["feedback_level"] for c in configs}, key=fb_rank)
    # Same entity -> color mapping as the HTML report: full=slot 1, names=2, minimal=3, none=4.
    color_order = ["full", "names", "minimal", "none"]
    lvl_scale = cat_scale(sorted(all_levels, key=lambda lv: color_order.index(lv) if lv in color_order else 9))
    curve_rows = []
    for s, cs in by_series.items():
        for c in cs:
            for k, v in enumerate(c["agg"].get("pass_at_k_mean", []), 1):
                curve_rows.append({"series": s, "feedback_level": c["feedback_level"], "k": k, "pass_at_k": v})
    curves = pd.DataFrame(curve_rows)
    if curves.empty:
        st.info("No pass@k curves yet.")
    else:
        names = list(by_series)
        enc = {
            "x": alt.X("k:O", title="try k", axis=alt.Axis(labelAngle=0)),
            "y": alt.Y("pass_at_k:Q", title="pass rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            "color": alt.Color("feedback_level:N", scale=lvl_scale, title="feedback", legend=alt.Legend(orient="top")),
            "tooltip": ["series:N", "feedback_level:N", alt.Tooltip("k:O", title="try k"),
                        alt.Tooltip("pass_at_k:Q", title="pass rate", format=".1%")],
        }
        base = alt.Chart().encode(**enc)
        layered = alt.layer(base.mark_line(strokeWidth=2), base.mark_point(filled=True, size=POINT),
                            data=curves).properties(width=300, height=220)
        facet = layered.facet(column=alt.Column("series:N", sort=names, title=None,
                                                header=alt.Header(labelFontSize=12, labelLimit=260)))
        st.altair_chart(facet.properties(title=""), width="content")
        with st.expander("Data"):
            st.dataframe(curves, hide_index=True)


def section_regression(runs: list[dict], configs: list[dict]) -> None:
    st.subheader("Regression comparison")
    mode = st.radio("Compare", ["Configurations (all seeds)", "Single runs"], horizontal=True, key="cmp_mode",
                    help="Configurations use compare_groups (permutation / paired McNemar across seeds) — "
                         "far less noisy than one run vs one run.")
    if mode == "Single runs":
        if len(runs) < 2:
            st.info("Need at least two runs to compare.")
            return
        labels = {r["run_id"]: run_label(r) for r in runs}
        ids = list(labels)
        c1, c2 = st.columns(2)
        base = c1.selectbox("Baseline run", ids, index=1, format_func=labels.get, key="cmp_base_run")
        cand = c2.selectbox("Candidate run", ids, index=0, format_func=labels.get, key="cmp_cand_run")
        if base == cand:
            st.info("Pick two different runs.")
            return
        show_single_comparison(compare_single(db_key(), base, cand))
    else:
        if len(configs) < 2:
            st.info("Need at least two configurations to compare — try `python main.py ablation`.")
            return
        labels = {i: f"{c['label']} ({c['agg']['num_runs']} runs)" for i, c in enumerate(configs)}
        # Default to the most informative pair: best vs worst mean final pass rate.
        by_rate = sorted(labels, key=lambda i: configs[i]["agg"]["pass_rate"]["mean"] or 0.0, reverse=True)
        c1, c2 = st.columns(2)
        base = c1.selectbox("Baseline configuration", list(labels), index=by_rate[0], format_func=labels.get,
                            key="cmp_base_cfg")
        cand = c2.selectbox("Candidate configuration", list(labels), index=by_rate[-1], format_func=labels.get,
                            key="cmp_cand_cfg")
        if base == cand:
            st.info("Pick two different configurations.")
            return
        show_group_comparison(compare_group(db_key(), tuple(configs[base]["run_ids"]),
                                            tuple(configs[cand]["run_ids"])), configs[base], configs[cand])


def verdict(cmp: dict, significant: bool | None = None) -> None:
    if cmp["regression"]:
        msg = "REGRESSION — " + "; ".join(cmp["reasons"])
        if significant is False:
            msg += " (not statistically significant: confirm with repeated runs)"
        st.error(msg)
    else:
        st.success("No regression detected.")
    for w in cmp["warnings"]:
        st.warning(w)


def show_single_comparison(cmp: dict) -> None:
    verdict(cmp, significant=cmp["significant"] or cmp["significant_at_1"])
    if cmp["changed"]:
        st.caption("Changed: " + ", ".join(f"{k}: {a} → {b}" for k, (a, b) in cmp["changed"].items()))
    lo, hi = cmp["pass_rate_delta_ci95"]
    c = st.columns(4)
    c[0].metric("Tasks compared", cmp["num_common"], border=True)
    c[1].metric("McNemar p (final pass)", f"{cmp['mcnemar_p']:.3f}", border=True,
                help="Exact test on newly failing vs newly passing tasks.")
    c[2].metric("McNemar p (pass@1)", f"{cmp['mcnemar_p_at_1']:.3f}", border=True,
                help=f"First-try lost {len(cmp['first_try_lost'])}, gained {len(cmp['first_try_gained'])}.")
    c[3].metric("Δ pass rate 95% CI", f"[{lo:+.1%}, {hi:+.1%}]", border=True, help="Paired bootstrap over tasks.")

    higher_is_better = {"pass_rate", "pass_at_1", "self_correction_lift", "recovery_rate"}
    pct_keys = higher_is_better
    rows = []
    for key, row in cmp["aggregate"].items():
        digits = 4 if key == "est_cost_usd" else 2
        fmt = pct if key in pct_keys else (lambda v, d=digits: num(v, d))
        d = row["delta"]
        if d is None or abs(d) < 1e-12:
            change = "same"
        else:
            better = (d > 0) == (key in higher_is_better)
            change = "better" if better else "worse"
        rows.append({"metric": key, "baseline": fmt(row["baseline"]), "candidate": fmt(row["candidate"]),
                     "delta": pp(d) if key in pct_keys else ("–" if d is None else f"{d:+,.2f}"), "change": change})
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Metric deltas**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with c2:
        k = cmp["pass_at_k"]
        curve = pd.DataFrame(
            [{"side": "baseline", "k": i + 1, "pass_at_k": v} for i, v in enumerate(k["baseline"])]
            + [{"side": "candidate", "k": i + 1, "pass_at_k": v} for i, v in enumerate(k["candidate"])])
        show_chart(line_chart(curve, "k", "pass_at_k", "pass@k: baseline vs candidate", "pass rate", color="side",
                              color_scale=cat_scale(["baseline", "candidate"]), height=240), curve)

    if cmp["newly_failing"]:
        st.error(f"Newly failing ({len(cmp['newly_failing'])}): " + ", ".join(cmp["newly_failing"]))
    task_rows = (
        [{"task_id": t, "change": "newly failing"} for t in cmp["newly_failing"]]
        + [{"task_id": t, "change": "first-try lost"} for t in cmp["first_try_lost"]]
        + [{"task_id": m["task_id"], "change": f"more tries ({m['baseline']}→{m['candidate']})"}
           for m in cmp["more_tries"]]
        + [{"task_id": t, "change": "newly passing"} for t in cmp["newly_passing"]]
        + [{"task_id": t, "change": "first-try gained"} for t in cmp["first_try_gained"]]
        + [{"task_id": m["task_id"], "change": f"fewer tries ({m['baseline']}→{m['candidate']})"}
           for m in cmp["fewer_tries"]]
    )
    if task_rows:
        st.markdown("**Per-task changes** (worst first)")
        st.dataframe(pd.DataFrame(task_rows), hide_index=True)
    else:
        st.caption("No per-task changes between the two runs.")
    with st.expander("Plain-text report"):
        st.code(regression.format_report(cmp), language=None)


def show_group_comparison(cmp: dict, base_cfg: dict, cand_cfg: dict) -> None:
    verdict(cmp)
    b, c = cmp["baseline"], cmp["candidate"]
    st.caption(f"Baseline: {base_cfg['label']} — seeds {b['seeds']}  ·  Candidate: {cand_cfg['label']} — "
               f"seeds {c['seeds']}  ·  {cmp['num_common']} common tasks")
    rows = []
    for key, m in cmp["metrics"].items():
        is_pct = key != "mean_tries_to_pass"
        f = pct if is_pct else num
        sd = (lambda v: f"{v * 100:.1f} pp") if is_pct else (lambda v: f"{v:.2f}")
        rows.append({"metric": key, "baseline": f"{f(m['baseline_mean'])} ± {sd(m['baseline_std'])}",
                     "candidate": f"{f(m['candidate_mean'])} ± {sd(m['candidate_std'])}",
                     "delta": (pp(m["delta"]) if is_pct else ("–" if m["delta"] is None else f"{m['delta']:+.2f}")),
                     "permutation p": m["p_value"], "paired McNemar p": m.get("paired_p_value")})
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Metric deltas** (mean ± std across seeds)")
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "permutation p": st.column_config.NumberColumn(format="%.3f"),
            "paired McNemar p": st.column_config.NumberColumn(format="%.3g"),
        })
        if cmp.get("paired"):
            pr = cmp["paired"]
            pc = st.columns(2)
            for col, key in zip(pc, ("pass_rate", "pass_at_1"), strict=True):
                col.metric(f"Paired McNemar p ({key})", f"{pr[key]['p_value']:.3g}", border=True,
                           help=f"{pr['num_pairs']} (seed, task) pairs: lost {pr[key]['lost']}, "
                                f"gained {pr[key]['gained']}.")
        else:
            st.caption("Seeds differ between groups, so outcomes are not paired; only the permutation test applies.")
    with c2:
        curve = pd.DataFrame(
            [{"side": "baseline", "k": i + 1, "pass_at_k": v}
             for i, v in enumerate(base_cfg["agg"].get("pass_at_k_mean", []))]
            + [{"side": "candidate", "k": i + 1, "pass_at_k": v}
               for i, v in enumerate(cand_cfg["agg"].get("pass_at_k_mean", []))])
        if not curve.empty:
            show_chart(line_chart(curve, "k", "pass_at_k", "pass@k (mean): baseline vs candidate", "pass rate",
                                  color="side", color_scale=cat_scale(["baseline", "candidate"]), height=240), curve)

    solve_cols = {col: st.column_config.NumberColumn(format="percent")
                  for col in ("baseline_solve", "candidate_solve", "baseline_first", "candidate_first")}
    if cmp["degraded_tasks"]:
        st.error(f"Degraded tasks ({len(cmp['degraded_tasks'])}): "
                 + ", ".join(t["task_id"] for t in cmp["degraded_tasks"]))
        st.dataframe(pd.DataFrame(cmp["degraded_tasks"]), hide_index=True, column_config=solve_cols)
    else:
        st.caption("No degraded tasks (solve rate or first-try rate down ≥ 50 pp).")
    if cmp["improved_tasks"]:
        st.markdown(f"**Improved tasks ({len(cmp['improved_tasks'])})**")
        st.dataframe(pd.DataFrame(cmp["improved_tasks"]), hide_index=True, column_config=solve_cols)
    with st.expander("Plain-text report"):
        st.code(regression.format_group_report(cmp), language=None)


def section_drilldown(run: dict) -> None:
    st.subheader(f"Task drill-down — run {run['run_id'][:8]}")
    key = db_key()
    results = load_results(key, run["run_id"])
    if not results:
        st.info("This run has no task results yet.")
        return
    by_id = {r["task_id"]: r for r in results}

    def label(tid: str) -> str:
        r = by_id[tid]
        outcome = f"passed on try {r['tries_taken']}" if r["passed"] else f"FAILED after {r['tries_taken']} tries"
        return f"{tid} · {outcome} · {r.get('difficulty') or '?'}"

    ids = sorted(by_id, key=lambda t: (by_id[t]["passed"], -by_id[t]["tries_taken"], t))
    task_id = st.selectbox("Task (failures first)", ids, format_func=label, key=f"task_select_{run['run_id']}")
    res = by_id[task_id]
    task = load_task(key, task_id) or {}
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Prompt**")
        st.code(task.get("prompt", "(task not stored)"), language="python")
    with c2:
        (st.success if res["passed"] else st.error)(
            f"{'Passed' if res['passed'] else 'Failed'} after {res['tries_taken']} tr{'y' if res['tries_taken'] == 1 else 'ies'}"
            f" (max {run['max_tries']}) · feedback level: {run['feedback_level']}")
        st.caption(f"difficulty: {res.get('difficulty')} · tags: {', '.join(res.get('tags') or []) or '–'} · "
                   f"tokens {res['tokens_in']:,} in / {res['tokens_out']:,} out · latency {res['latency_s']:.2f}s")
        if res.get("error"):
            st.warning(f"Infrastructure error: {res['error']}")
        with st.expander("Hidden test file"):
            st.code(task.get("test_code", ""), language="python")

    for a in load_task_attempts(key, run["run_id"], task_id):
        tr = a["test_results"] or {}
        with st.container(border=True):
            n = a["attempt_number"]
            if a["passed"]:
                st.success(f"Attempt {n} — PASSED · {a['num_passed']}/{a['num_total']} tests passed")
            else:
                st.error(f"Attempt {n} — FAILED ({a.get('error_type') or 'wrong_answer'}) · "
                         f"{a['num_passed']}/{a['num_total']} tests passed")
            st.caption(f"tokens {a['tokens_in']:,} in / {a['tokens_out']:,} out · LLM {a['latency_s']:.2f}s · "
                       f"sandbox {a['sandbox_s']:.2f}s")
            t_tests, t_fb, t_code, t_out, t_raw = st.tabs(
                ["Tests", "Feedback given to this attempt", "Code", "Feedback produced", "Raw response"])
            with t_tests:
                if tr.get("error"):
                    st.markdown(f"**Module-level error** ({tr.get('error_type') or 'error'})")
                    st.code(tr["error"], language=None)
                cases = tr.get("cases") or []
                if cases:
                    st.dataframe(pd.DataFrame([{
                        "name": c.get("name"), "passed": bool(c.get("passed")), "error_type": c.get("error_type"),
                        "expected": c.get("expected"), "actual": c.get("actual"), "assertion": c.get("assertion"),
                        "error": c.get("error"),
                    } for c in cases]), hide_index=True)
                elif not tr.get("error"):
                    st.caption("No per-test results recorded.")
                if tr.get("stdout"):
                    st.markdown("**stdout**")
                    st.code(tr["stdout"], language=None)
            with t_fb:
                if n == 1:
                    st.caption("First attempt — no feedback yet (prompt only).")
                elif a.get("feedback_given"):
                    st.code(a["feedback_given"], language=None)
                else:
                    st.caption(f"No feedback text was given (feedback level: {run['feedback_level']}).")
            with t_code:
                st.code(a["code"], language="python")
            with t_out:
                if a.get("feedback_produced"):
                    st.code(a["feedback_produced"], language=None)
                else:
                    st.caption("No feedback produced (attempt passed, or loop ended).")
            with t_raw:
                st.code(a.get("raw_response") or "", language=None)


def section_solve_matrix(configs: list[dict]) -> None:
    st.subheader("Per-task solve matrix across configurations")
    which = st.radio("Cell value", ["final solve rate", "first-try solve rate"], horizontal=True, key="matrix_metric")
    key = db_key()
    uniform = len({(c["prompt_version"], c["max_tries"]) for c in configs}) == 1
    rows = []
    for c in configs:
        label = f"{c['model']} · fb={c['feedback_level']}" if uniform else c["label"]
        per_task: dict[str, dict] = defaultdict(lambda: {"solved": 0, "first": 0, "n": 0, "difficulty": None})
        for rid in c["run_ids"]:
            by_task = metrics.group_attempts(load_attempts(key, rid))
            for r in load_results(key, rid):
                t = per_task[r["task_id"]]
                t["n"] += 1
                t["solved"] += int(r["passed"])
                t["first"] += int(metrics.first_pass_attempt(by_task.get(r["task_id"], [])) == 1)
                t["difficulty"] = r.get("difficulty")
        for tid, t in per_task.items():
            rows.append({"task_id": tid, "config": label, "difficulty": t["difficulty"], "runs": t["n"],
                         "solve_rate": t["solved"] / t["n"], "first_try_rate": t["first"] / t["n"]})
    if not rows:
        st.info("No task results yet.")
        return
    df = pd.DataFrame(rows)
    field = "solve_rate" if which == "final solve rate" else "first_try_rate"
    task_order = list(df.groupby("task_id")[field].mean().sort_values().index)  # hardest first
    config_order = list(dict.fromkeys(df["config"]))
    chart = alt.Chart(df).mark_rect(cornerRadius=2).encode(
        x=alt.X("config:N", sort=config_order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=220, labelOverlap=False)),
        y=alt.Y("task_id:N", sort=task_order, title=None),
        color=alt.Color(f"{field}:Q", scale=alt.Scale(domain=[0, 1], range=SEQ_RAMP), title=which,
                        legend=alt.Legend(format="%")),
        tooltip=["task_id:N", "config:N", "difficulty:N", "runs:Q",
                 alt.Tooltip("solve_rate:Q", title="final solve rate", format=".0%"),
                 alt.Tooltip("first_try_rate:Q", title="first-try rate", format=".0%")],
    ).properties(height=alt.Step(20), title=f"{which} per task (across seeds; hardest on top)")
    st.altair_chart(chart, width="stretch")
    with st.expander("Data"):
        st.dataframe(df.pivot_table(index="task_id", columns="config", values=field, sort=False)
                     .reindex(index=task_order, columns=config_order).reset_index(), hide_index=True)


def section_quality(run: dict, summary: dict) -> None:
    st.subheader(f"Judge quality & static code metrics — run {run['run_id'][:8]}")
    results = load_results(db_key(), run["run_id"])
    rows = []
    for r in results:
        row = {"task_id": r["task_id"], "passed": r["passed"], "tries": r["tries_taken"],
               "difficulty": r.get("difficulty")}
        row.update(r.get("code_metrics") or {})
        q = r.get("quality_score") or {}
        for k, v in q.items():
            if isinstance(v, int | float | str | bool) or v is None:
                row[f"judge_{k}"] = v
        rows.append(row)
    has_quality = any(r.get("quality_score") for r in results)
    has_code = any(r.get("code_metrics") for r in results)
    if not (has_quality or has_code):
        st.info("No judge scores or static code metrics stored for this run.")
        return
    if summary.get("mean_quality"):
        cols = st.columns(len(summary["mean_quality"]))
        for col, (k, v) in zip(cols, summary["mean_quality"].items(), strict=True):
            col.metric(f"Mean judge {k}", f"{v:.2f}", border=True)
    else:
        st.caption("LLM judge not used for this run (run with `--judge` to score passing code). "
                   "Static code metrics below are computed deterministically.")
    if summary.get("mean_code_metrics"):
        cols = st.columns(len(summary["mean_code_metrics"]))
        for col, (k, v) in zip(cols, summary["mean_code_metrics"].items(), strict=True):
            col.metric(f"Mean {k} (solved)", f"{v:.1f}", border=True)
    st.dataframe(pd.DataFrame(rows), hide_index=True)


# --- page ------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="AgentEval dashboard", layout="wide")
    st.title("AgentEval: self-correcting coding agent")
    st.caption("generate → sandbox-test → feedback → retry. Every number comes from agent_eval.metrics, "
               "so it matches the CLI and HTML report.")

    try:
        runs = load_runs(db_key())
    except Exception as exc:  # corrupt / unreadable DB: say so instead of a stack trace
        st.error(f"Could not read the database at {config.DB_PATH}: {exc}")
        st.stop()
    if not runs:
        st.info(f"No runs stored in {config.DB_PATH} yet. Run the loop first, e.g. `{EMPTY_HINT}` "
                "(offline simulated agent, no API key needed), then refresh this page.")
        st.stop()

    with st.sidebar:
        st.header("Run")
        labels = {r["run_id"]: run_label(r) for r in runs}
        run_id = st.selectbox("Selected run (newest first)", list(labels), format_func=labels.get, key="run_select")
        config_view = st.toggle("Configuration view", key="config_view",
                                help="Aggregate the overview over every seed of the selected run's configuration.")
        st.caption(f"{len(runs)} runs · DB: `{config.DB_PATH}`")
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()

    run = next(r for r in runs if r["run_id"] == run_id)
    configs = config_table(db_key())
    cfg = next(c for c in configs if c["key"] == config_tuple(run))

    sim_models = sorted({r["model"] for r in runs if is_sim(r["model"])})
    if sim_models:
        st.warning(f"**{SIM_NOTICE}** Runs from {', '.join(sim_models)} use the offline simulated agent "
                   "(canned solutions and mutants), not a real LLM.")

    if config_view:
        summary = pooled_summary(db_key(), tuple(cfg["run_ids"]))
        agg = cfg["agg"]
        scope = f"configuration {cfg['label']} ({agg['num_runs']} runs, seeds {cfg['seeds']})"
    else:
        summary = load_summary(db_key(), run_id)
        agg = None
        scope = f"{run['run_id'][:8]} · {config_key(run)} · seed {run['seed']}"
        if run.get("notes"):
            scope += f" · “{run['notes']}”"

    tabs = st.tabs(["Overview", "Leaderboard & ablation", "Regression", "Task drill-down", "Solve matrix",
                    "Quality & code metrics"])
    with tabs[0]:
        section_overview(summary, agg, scope)
    with tabs[1]:
        section_leaderboard(configs)
    with tabs[2]:
        section_regression(runs, configs)
    with tabs[3]:
        section_drilldown(run)
    with tabs[4]:
        section_solve_matrix(configs)
    with tabs[5]:
        section_quality(run, load_summary(db_key(), run_id))


main()
