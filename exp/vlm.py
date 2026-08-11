from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union


ToolChoice = Optional[Union[str, Dict[str, Any]]]


@dataclass
class ToolCall:
    """A tool call requested by an assistant message."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str = ""
    provider: str = ""
    raw: Any = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "provider": self.provider,
        }


@dataclass
class ChatMessage:
    """Provider-neutral chat message.

    Conventions:
    - assistant tool calls live in role="assistant" messages.
    - tool feedback lives in role="tool" messages and refers to tool_call_id.
    """

    role: str
    content: str = ""
    images: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, images: list[str] | None = None) -> "ChatMessage":
        return cls(role="user", content=content, images=list(images or []))

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> "ChatMessage":
        return cls(role="assistant", content=content, tool_calls=list(tool_calls or []))

    @classmethod
    def assistant_tool_call(cls, tool_call: ToolCall) -> "ChatMessage":
        return cls.assistant(tool_calls=[tool_call])

    @classmethod
    def tool_result(cls, tool_call: ToolCall, content: str | dict[str, Any]) -> "ChatMessage":
        return cls(
            role="tool",
            content=_content_to_text(content),
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )

    def to_debug_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.images:
            payload["images"] = list(self.images)
        if self.tool_calls:
            payload["tool_calls"] = [tool_call.to_debug_dict() for tool_call in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.name:
            payload["name"] = self.name
        return payload


@dataclass
class VLMRequest:
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: ToolChoice = None
    max_new_tokens: int = 512


@dataclass
class ModelResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    allow_text_fallback: bool = True


class BaseVLM(Protocol):
    def predict(self, request: VLMRequest) -> ModelResponse:
        raise NotImplementedError


class MockVLM:
    """Fixed-response VLM for wiring agent/runner without loading a real model."""

    def __init__(self, response: ModelResponse | None = None) -> None:
        self.response = response or ModelResponse(
            tool_calls=[
                ToolCall(
                    id="mock_finish",
                    name="finish_inspection",
                    arguments={"reason": "mock finished"},
                    provider="mock",
                )
            ],
            allow_text_fallback=False,
        )
        self.requests: list[VLMRequest] = []

    def predict(self, request: VLMRequest) -> ModelResponse:
        self.requests.append(request)
        return self.response


class SequenceVLM:
    """Return predefined responses in order.

    After the sequence is exhausted, the last response is repeated. This makes
    multi-step tests deterministic without special stop handling.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        if not responses:
            raise ValueError("SequenceVLM requires at least one response")
        self.responses = list(responses)
        self.index = 0
        self.requests: list[VLMRequest] = []

    def predict(self, request: VLMRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model: str
    base_url: str | None = None
    timeout: float | None = None
    temperature: float = 0.6
    top_p: float = 0.95
    supports_strict_tools: bool = False
    supports_parallel_tool_calls: bool = False


class OpenAICompatibleVLM:
    """OpenAI-compatible chat-completions adapter.

    This adapter converts the internal conversation-first message format into
    Chat Completions messages. Assistant tool calls stay in assistant messages;
    tool feedback becomes role="tool" messages.
    """

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("OpenAICompatibleVLM requires the openai package.") from exc

        api_key = os.getenv("VLM_API_KEY")
        if not api_key:
            raise RuntimeError("VLM_API_KEY is required for OpenAI-compatible VLMs.")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.timeout is not None:
            kwargs["timeout"] = config.timeout
        self.client = OpenAI(**kwargs)
        self.config = config

    def predict(self, request: VLMRequest) -> ModelResponse:
        tools = [_tool_payload(tool, strict=self.config.supports_strict_tools) for tool in request.tools]
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [_message_payload(message) for message in request.messages],
            "max_tokens": request.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if tools:
            kwargs["tools"] = tools
            if request.tool_choice is not None:
                kwargs["tool_choice"] = request.tool_choice
            if self.config.supports_parallel_tool_calls:
                kwargs["parallel_tool_calls"] = False
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        return ModelResponse(
            text=message.content or "",
            tool_calls=[
                _convert_tool_call(tool_call, provider="openai-compatible")
                for tool_call in (message.tool_calls or [])
            ],
            raw=response,
            allow_text_fallback=False,
        )


def _tool_payload(tool: dict[str, Any], strict: bool = False) -> dict[str, Any]:
    payload = json.loads(json.dumps(tool, ensure_ascii=False))
    if not strict:
        payload.get("function", {}).pop("strict", None)
    return payload


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        payload = {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": message.content,
        }
        if message.name:
            payload["name"] = message.name
        return payload

    payload: dict[str, Any] = {
        "role": message.role,
        "content": _message_content(message),
    }
    if message.tool_calls:
        payload["tool_calls"] = [_tool_call_payload(tool_call) for tool_call in message.tool_calls]
        if message.role == "assistant" and not message.content:
            payload["content"] = None
    return payload


def _message_content(message: ChatMessage) -> Any:
    if message.role != "user" or not message.images:
        return message.content

    content: list[dict[str, Any]] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    for image_path in message.images:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
    return content


def _tool_call_payload(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments or {}, ensure_ascii=False),
        },
    }


def _convert_tool_call(item: Any, provider: str) -> ToolCall:
    function = item.function
    arguments = function.arguments
    if isinstance(arguments, str):
        arguments_dict = json.loads(arguments or "{}")
    else:
        arguments_dict = dict(arguments or {})
    return ToolCall(
        id=getattr(item, "id", ""),
        name=function.name,
        arguments=arguments_dict,
        provider=provider,
        raw=item,
    )


def _image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _content_to_text(content: str | dict[str, Any]) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)
