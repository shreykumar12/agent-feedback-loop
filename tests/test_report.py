import json

from agent_eval import report
from tests.helpers import store_run


def test_report_payload_and_html(tmp_db, tmp_path):
    store_run("a1", {"e1": [True], "h1": [False, True]}, model="sim-base", seed=0)
    store_run("a2", {"e1": [True], "h1": [False, False, False]}, model="sim-base", seed=1)
    store_run("b1", {"e1": [False, True], "h1": [False, False, False]}, model="sim-weak", seed=0)

    payload = report.build_payload(compare=("a1", "b1"))
    assert payload["simulated"] is True
    assert payload["num_runs"] == 3
    configs = {c["model"]: c for c in payload["configs"]}
    assert configs["sim-base"]["n_runs"] == 2
    assert configs["sim-base"]["pass_rate"] == 0.75
    assert payload["comparison"]["newly_failing"] == ["h1"]
    assert {g["task_id"] for g in payload["grid"]} == {"e1", "h1"}

    out = report.write_report(out_path=tmp_path / "r.html", compare=("a1", "b1"))
    text = out.read_text()
    assert text.startswith("<!doctype html>")
    assert "<title>AgentEval Report</title>" in text
    embedded = text.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["focus"]["num_tasks"] == 2

    fragment = report.render_html(payload, standalone=False)
    assert not fragment.startswith("<!doctype") and "<body>" not in fragment
