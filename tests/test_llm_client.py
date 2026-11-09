import pytest
from pydantic import BaseModel

from app.config import LLMConfig
from app.llm.client import LLMError, OllamaClient, ensure_local
from fakes import FakeOllama


class Pair(BaseModel):
    title: str | None = None
    org: str | None = None


def test_structured_output_request_and_parse():
    fake = FakeOllama([{"title": "Engineer", "org": "Acme"}])
    client = OllamaClient(LLMConfig(), client=fake)
    assert client.generate_json("block", Pair, system="sys") == Pair(title="Engineer", org="Acme")
    req = fake.requests[0]
    assert req["format"] == Pair.model_json_schema()
    assert req["options"]["temperature"] == 0 and req["options"]["num_ctx"] == 2048
    assert req["think"] is False
    assert [m["role"] for m in req["messages"]] == ["system", "user"]


def test_retries_with_validation_feedback():
    fake = FakeOllama(["not json", {"title": 42}, {"title": "Engineer"}])
    client = OllamaClient(LLMConfig(max_retries=2), client=fake)
    assert client.generate_json("x", Pair).title == "Engineer"
    assert client.stats.retries == 2 and client.stats.calls == 1
    assert "not valid for the schema" in fake.requests[1]["messages"][-1]["content"]


def test_gives_up_after_max_retries():
    client = OllamaClient(LLMConfig(max_retries=1), client=FakeOllama(["bad", "bad"]))
    with pytest.raises(LLMError, match="invalid JSON 2 times"):
        client.generate_json("x", Pair)
    assert client.stats.failures == 1


def test_think_flag_dropped_for_models_without_it():
    class NoThink(FakeOllama):
        def chat(self, **kwargs):
            if "think" in kwargs:
                raise RuntimeError('"qwen" does not support thinking')
            return super().chat(**kwargs)

    fake = NoThink([{"title": "A"}, {"title": "B"}])
    client = OllamaClient(LLMConfig(), client=fake)
    assert client.generate_json("x", Pair).title == "A"
    assert client.generate_json("x", Pair).title == "B"
    assert all("think" not in r for r in fake.requests)


def test_available():
    assert OllamaClient(LLMConfig(), client=FakeOllama()).available()
    assert not OllamaClient(LLMConfig(model="other:1b"), client=FakeOllama()).available()
    assert not OllamaClient(LLMConfig(), client=FakeOllama(fail_with=ConnectionError("down"))).available()


def test_unreachable_server_raises_llm_error():
    client = OllamaClient(LLMConfig(), client=FakeOllama(fail_with=ConnectionError("down")))
    with pytest.raises(LLMError, match="request failed"):
        client.generate_json("x", Pair)


@pytest.mark.parametrize("host", ["http://127.0.0.1:11434", "localhost:11434", "http://[::1]:11434"])
def test_local_hosts_allowed(host):
    ensure_local(host)


def test_remote_host_refused_unless_allowed():
    with pytest.raises(LLMError, match="non-local"):
        OllamaClient(LLMConfig(host="http://gpu-box.example.com:11434"), client=FakeOllama())
    OllamaClient(LLMConfig(host="http://gpu-box.example.com:11434", allow_remote=True), client=FakeOllama())
