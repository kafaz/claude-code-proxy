import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.conversion.response_converter import (
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest


class FakeChunk:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


class FakeRequest:
    async def is_disconnected(self):
        return False


class FakeLogger:
    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


@pytest.mark.asyncio
async def test_stream_stops_cleanly_when_cancel_event_set_before_yield():
    request_id = "req-1"

    class FakeStream:
        def __init__(self, client):
            self.client = client
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            self.client.active_requests[request_id].set()
            return FakeChunk({"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]})

    class FakeCompletions:
        def __init__(self, client):
            self.client = client

        async def create(self, **_kwargs):
            return FakeStream(self.client)

    client = OpenAIClient(api_key="k", base_url="http://localhost")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(client)))

    output = [item async for item in client.create_chat_completion_stream({"model": "x", "messages": []}, request_id)]

    assert output == []
    assert request_id not in client.active_requests


@pytest.mark.asyncio
async def test_stream_swallows_cancelled_error_from_upstream():
    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise asyncio.CancelledError

    class FakeCompletions:
        async def create(self, **_kwargs):
            return FakeStream()

    client = OpenAIClient(api_key="k", base_url="http://localhost")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    output = [item async for item in client.create_chat_completion_stream({"model": "x", "messages": []})]

    assert output == []


@pytest.mark.asyncio
async def test_converter_handles_http_exception_as_sse_error():
    async def failing_stream():
        raise HTTPException(status_code=500, detail="boom")
        yield ""  # pragma: no cover

    original_request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )

    events = [
        item
        async for item in convert_openai_streaming_to_claude_with_cancellation(
            failing_stream(),
            original_request,
            FakeLogger(),
            FakeRequest(),
            openai_client=SimpleNamespace(cancel_request=lambda _request_id: True),
            request_id="req-2",
        )
    ]

    assert any("event: error" in event for event in events)
    assert any("boom" in event for event in events)


@pytest.mark.asyncio
async def test_converter_accepts_dict_stream_chunks():
    async def dict_stream():
        yield {
            "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    original_request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )

    events = [
        item
        async for item in convert_openai_streaming_to_claude_with_cancellation(
            dict_stream(),
            original_request,
            FakeLogger(),
            FakeRequest(),
            openai_client=SimpleNamespace(cancel_request=lambda _request_id: True),
            request_id="req-3",
        )
    ]

    assert any("hello" in event for event in events)
    assert any("event: message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_converter_disconnect_check_interval_controls_frequency():
    class CountingRequest:
        def __init__(self):
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return False

    async def dict_stream():
        for _ in range(5):
            yield {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}

    original_request = ClaudeMessagesRequest(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    request = CountingRequest()

    _ = [
        item
        async for item in convert_openai_streaming_to_claude_with_cancellation(
            dict_stream(),
            original_request,
            FakeLogger(),
            request,
            openai_client=SimpleNamespace(cancel_request=lambda _request_id: True),
            request_id="req-4",
            disconnect_check_interval=2,
        )
    ]

    assert request.calls == 2
