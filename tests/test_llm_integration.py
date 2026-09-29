"""Exercise the real LLM code path (langchain-openai ChatOpenAI -> an
OpenAI-compatible HTTP endpoint) against a local fake server, so provider
wiring, usage accounting, extraction and the judge's structured output are
tested without a key or network access."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent_eval import agent, config, judge
from agent_eval.graph import LoopSettings, run_task
from agent_eval.models import Task

TASK = Task(task_id="add", prompt="def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n", entry_point="add",
            test_code="def test_sum():\n    assert add(2, 3) == 5\n")


class FakeOpenAI(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append(body)
        last = body["messages"][-1]["content"]
        message = {"role": "assistant"}
        if body.get("response_format") or body.get("tools"):
            args = json.dumps({"readability": 4, "readability_reason": "clear", "approach": 5,
                               "approach_reason": "direct"})
            if body.get("tools"):
                message["content"] = None
                message["tool_calls"] = [{"id": "c1", "type": "function",
                                          "function": {"name": body["tools"][0]["function"]["name"], "arguments": args}}]
            else:
                message["content"] = args
        elif "previous attempt" in last.lower():
            message["content"] = "Fixed:\n```python\ndef add(a, b):\n    return a + b\n```"
        else:
            message["content"] = "```python\ndef add(a, b):\n    return a - b\n```"
        payload = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_llm(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    FakeOpenAI.requests = []
    monkeypatch.setattr(config, "LLM_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    agent._chat_client.cache_clear()
    yield FakeOpenAI.requests
    server.shutdown()
    agent._chat_client.cache_clear()


def test_real_client_generates_and_counts_tokens(fake_llm):
    gen = agent.generate(TASK, model="fake-model", prompt_version="v2")
    assert gen.code == "def add(a, b):\n    return a - b\n"
    assert gen.tokens_in == 42 and gen.tokens_out == 7
    request = fake_llm[0]
    assert request["model"] == "fake-model"
    assert request["temperature"] == 0
    assert request["messages"][0]["role"] == "system"


def test_full_loop_over_http(fake_llm):
    result, attempts = run_task(TASK, LoopSettings(model="fake-model", max_tries=3))
    assert result.passed and result.tries_taken == 2
    assert "expected: 5" in fake_llm[1]["messages"][-1]["content"]  # feedback reached the model
    assert result.total_tokens_in == 84


def test_judge_structured_output(fake_llm):
    score = judge.score_quality(TASK, "def add(a, b):\n    return a + b\n", model="fake-judge")
    assert score is not None
    assert score["readability"] == 4 and score["approach"] == 5


def test_judge_returns_none_on_api_error(monkeypatch):
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://127.0.0.1:9/v1")  # nothing listens here
    monkeypatch.setattr(config, "LLM_API_KEY", "k")
    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 2)
    assert judge.score_quality(TASK, "def add(a, b): return a + b", model="x") is None


def test_missing_key_is_a_clear_agent_error(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    agent._chat_client.cache_clear()
    with pytest.raises(agent.AgentError, match="API key"):
        agent.generate(TASK, model="gemini-2.5-flash")
