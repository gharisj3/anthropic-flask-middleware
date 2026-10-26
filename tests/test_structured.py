from types import SimpleNamespace

import pytest

from anthropic_service import AnthropicService, TaskExtractionResult, create_app


class FakeStructuredService:
    def call_structured(self, prompt, schema, system="", session_id=None):
        result = TaskExtractionResult(
            title="Ship API wrapper",
            priority="high",
            due_date=None,
            tags=["flask", "anthropic"],
            summary=f"Task extracted from: {prompt}",
        )
        return result, {
            "session_id": session_id or "mock-session",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "correction_attempts": 0,
        }

    def stream_claude(self, messages, system):
        yield 'data: {"delta": "mocked"}\n\n'


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("APP_API_KEY", "test-key")
    app = create_app(service=FakeStructuredService())
    app.config["TESTING"] = True
    return app.test_client()


def test_health_works(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ok",
        "service": "anthropic-flask-middleware",
    }


def test_missing_x_api_key_is_rejected(client):
    response = client.post("/structured", json={"prompt": "Extract this task"})

    assert response.status_code == 401
    assert response.get_json()["code"] == "missing_api_key"


def test_invalid_structured_data_fails_validation():
    with pytest.raises(ValueError):
        TaskExtractionResult.model_validate(
            {
                "title": "",
                "priority": "urgent",
                "due_date": None,
                "tags": ["api"],
                "summary": "",
            }
        )


def test_valid_structured_data_passes_pydantic_validation():
    result = TaskExtractionResult.model_validate(
        {
            "title": "Write docs",
            "priority": "medium",
            "due_date": "2026-06-30",
            "tags": ["docs", "flask"],
            "summary": "Create clear project documentation.",
        }
    )

    assert result.title == "Write docs"
    assert result.priority == "medium"
    assert result.tags == ["docs", "flask"]


def test_structured_endpoint_uses_mock_and_makes_no_real_anthropic_call(
    client, monkeypatch
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "Real Anthropic client should not be constructed during tests"
        )

    monkeypatch.setattr("anthropic_service.anthropic.Anthropic", fail_if_called)

    response = client.post(
        "/structured",
        headers={"X-API-Key": "test-key"},
        json={"prompt": "Prepare the middleware repo by Friday"},
    )

    assert response.status_code == 200
    assert response.get_json()["result"]["priority"] == "high"


def test_structured_correction_loop_uses_mocked_anthropic_messages(monkeypatch):
    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return SimpleNamespace(content=[SimpleNamespace(text='{"title": ""}')])
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        text=(
                            '{"title": "Follow up", "priority": "low", '
                            '"due_date": null, "tags": ["ops"], '
                            '"summary": "Follow up with the client."}'
                        )
                    )
                ]
            )

    fake_client = SimpleNamespace(messages=FakeMessages())
    service = AnthropicService(api_key="unused", client=fake_client)

    result, meta = service.call_structured(
        "Follow up with the client",
        schema=TaskExtractionResult,
    )

    assert result.title == "Follow up"
    assert meta["correction_attempts"] == 1
    assert len(calls) == 2
