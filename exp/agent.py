from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from vlm import BaseVLM, ChatMessage, ModelResponse, ToolCall, VLMRequest

WALK_TOOL_NAME = "walk"
TURN_TOOL_NAME = "turn"
LOOK_UP_TOOL_NAME = "look_up"
LOOK_DOWN_TOOL_NAME = "look_down"
REPORT_HAZARD_TOOL_NAME = "report_hazard"
FINISH_INSPECTION_TOOL_NAME = "finish_inspection"

DECISION_ENVIRONMENT_ACTION = "environment_action"
DECISION_REPORT_HAZARD = "report_hazard"
DECISION_FINISH = "finish"
DECISION_ERROR = "error"

VALID_ENV_ACTIONS = {"walk straight", "turn left", "turn right", "look up", "look down"}

SYSTEM_PROMPT = """You are a home safety inspection agent in a VirtualHome scene.
Use the provided tools to inspect the room step by step.
Call exactly one tool at a time.
For each step, reason briefly: at most 3 short sentences.
Then immediately call exactly one tool.
Do not spend the whole response on reasoning; the tool call is required.
Use report_hazard when you identify visible hazards.
Use walk, turn, look_up, or look_down when more inspection is needed.
Use finish_inspection only when the current sample is complete.
Hazard category definitions:
- Fire Hazard: Conditions that could start, intensify, or spread a fire.
- Electrical Hazard: Conditions involving electrical sources or powered items that could cause shock, short circuit, overheating, or electrical damage.
- Falling-Object Hazard: Conditions where an item could drop, tip over, or strike a person or nearby object.
- Trip Hazard: Conditions that could obstruct normal movement, reduce footing, or cause a person to trip.
- Child-Safety Hazard: Conditions that could put a child at risk because a child may access, misuse, ingest, touch, climb on, or become trapped by something in the scene.
Report a category only when the visible scene provides plausible evidence of that risk, not merely because an object type can be dangerous in general.
"""

SYSTEM_PROMPT_VIEW_REPORT = """You are a home safety inspection agent in a VirtualHome scene.
This analysis setting does not provide movement tools. Do not try to walk or turn.
Use only the provided view-adjustment and reporting tools to inspect from the current position.
Call exactly one tool at a time.
For each step, reason briefly: at most 3 short sentences.
Then immediately call exactly one tool.
Do not spend the whole response on reasoning; the tool call is required.
Use look_up or look_down when changing the camera angle may reveal hazards.
Use report_hazard when you identify visible hazards.
Use finish_inspection only when no more useful view adjustment or hazard report is needed for the current sample.
Hazard category definitions:
- Fire Hazard: Conditions that could start, intensify, or spread a fire.
- Electrical Hazard: Conditions involving electrical sources or powered items that could cause shock, short circuit, overheating, or electrical damage.
- Falling-Object Hazard: Conditions where an item could drop, tip over, or strike a person or nearby object.
- Trip Hazard: Conditions that could obstruct normal movement, reduce footing, or cause a person to trip.
- Child-Safety Hazard: Conditions that could put a child at risk because a child may access, misuse, ingest, touch, climb on, or become trapped by something in the scene.
Report a category only when the visible scene provides plausible evidence of that risk, not merely because an object type can be dangerous in general.
"""

SYSTEM_PROMPT_REPORT_ONLY = """You are a home safety inspection agent in a VirtualHome scene.
This analysis setting provides exactly one first-person image and no navigation or view-adjustment tools.
Do not try to walk, turn, look up, or look down.
Call exactly one tool: use report_hazard if the single visible image provides evidence of one or more hazards; otherwise use finish_inspection.
For each step, reason briefly: at most 3 short sentences.
Then immediately call exactly one tool.
Do not spend the whole response on reasoning; the tool call is required.
Hazard category definitions:
- Fire Hazard: Conditions that could start, intensify, or spread a fire.
- Electrical Hazard: Conditions involving electrical sources or powered items that could cause shock, short circuit, overheating, or electrical damage.
- Falling-Object Hazard: Conditions where an item could drop, tip over, or strike a person or nearby object.
- Trip Hazard: Conditions that could obstruct normal movement, reduce footing, or cause a person to trip.
- Child-Safety Hazard: Conditions that could put a child at risk because a child may access, misuse, ingest, touch, climb on, or become trapped by something in the scene.
Report a category only when the visible scene provides plausible evidence of that risk, not merely because an object type can be dangerous in general.
"""

@dataclass
class AgentDecision:
    kind: str
    tool_call: ToolCall | None = None
    action: str = ""
    movement_steps: int | None = None
    turn_angle: int | None = None
    camera_angle: int | None = None
    hazards: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    raw_response: str = ""
    ok: bool = True
    error: str = ""

    @classmethod
    def error_decision(cls, message: str, raw_response: str = "", tool_call: ToolCall | None = None) -> "AgentDecision":
        return cls(
            kind=DECISION_ERROR,
            tool_call=tool_call,
            raw_response=raw_response,
            ok=False,
            error=message,
        )

class SafeAgent:
    """Conversation-first VLM agent.

    The agent owns conversation history and tool schemas. It does not execute
    VirtualHome actions. Runner/env execute decisions and feed results back via
    add_tool_feedback().
    """

    def __init__(
        self,
        vlm: BaseVLM,
        system_prompt: str = SYSTEM_PROMPT,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_new_tokens: int = 512,
    ) -> None:
        self.vlm = vlm
        self.tools = tools if tools is not None else build_tools()
        self.tool_choice = tool_choice
        self.max_new_tokens = max_new_tokens
        self.messages: list[ChatMessage] = [ChatMessage.system(system_prompt)]

    def decide(self, observation: Any) -> AgentDecision:
        self.messages.append(observation_to_message(observation))
        response = self.vlm.predict(
            VLMRequest(
                messages=list(self.messages),
                tools=self.tools,
                tool_choice=self.tool_choice,
                max_new_tokens=self.max_new_tokens,
            )
        )

        decision = parse_model_response(response)
        if decision.tool_call is not None:
            self.messages.append(
                ChatMessage.assistant(
                    content=response.text,
                    tool_calls=[decision.tool_call],
                )
            )
        elif response.text:
            self.messages.append(ChatMessage.assistant(content=response.text))
        return decision

    def add_tool_feedback(self, tool_call: ToolCall, feedback: dict[str, Any] | str) -> None:
        self.messages.append(ChatMessage.tool_result(tool_call, feedback))

    def add_retry_instruction(self, message: str) -> None:
        self.messages.append(ChatMessage.user(message))

def build_tools() -> list[dict[str, Any]]:
    return [
        _walk_tool_schema(),
        _turn_tool_schema(),
        _look_up_tool_schema(),
        _look_down_tool_schema(),
        _report_hazard_tool_schema(),
        _finish_inspection_tool_schema(),
    ]

def build_view_report_tools() -> list[dict[str, Any]]:
    return [
        _look_up_tool_schema(),
        _look_down_tool_schema(),
        _report_hazard_tool_schema(),
        _finish_inspection_tool_schema(),
    ]

def build_report_only_tools() -> list[dict[str, Any]]:
    return [
        _report_hazard_tool_schema(),
        _finish_inspection_tool_schema(),
    ]

def parse_model_response(response: ModelResponse) -> AgentDecision:
    if response.tool_calls:
        if len(response.tool_calls) != 1:
            return AgentDecision.error_decision(
                f"Expected exactly one tool call, got {len(response.tool_calls)}",
                raw_response=response.text,
                tool_call=response.tool_calls[0],
            )
        return parse_tool_call(response.tool_calls[0], raw_response=response.text)

    if not response.allow_text_fallback:
        return AgentDecision.error_decision("VLM did not return a tool call.", raw_response=response.text)

    tool_call = parse_text_tool_call(response.text)
    if tool_call is None:
        return AgentDecision.error_decision("Failed to parse VLM text response as a tool call.", response.text)
    return parse_tool_call(tool_call, raw_response=response.text)

def parse_tool_call(tool_call: ToolCall, raw_response: str = "") -> AgentDecision:
    name = tool_call.name
    arguments = tool_call.arguments or {}
    if not isinstance(arguments, dict):
        return AgentDecision.error_decision(f"{name} arguments must be a JSON object", raw_response, tool_call)

    try:
        if name == WALK_TOOL_NAME:
            steps = int(arguments.get("steps", 1))
            if steps < 1:
                raise ValueError("walk.steps must be >= 1")
            if steps > 5:
                raise ValueError("walk.steps must be <= 5")
            return AgentDecision(
                kind=DECISION_ENVIRONMENT_ACTION,
                tool_call=tool_call,
                action="walk straight",
                movement_steps=steps,
                raw_response=raw_response,
            )

        if name == TURN_TOOL_NAME:
            direction = str(arguments.get("direction", "")).strip().lower()
            if direction not in {"left", "right"}:
                raise ValueError("turn.direction must be left or right")
            angle = int(arguments.get("angle", 30))
            if angle <= 0:
                raise ValueError("turn.angle must be positive")
            if angle > 180:
                raise ValueError("turn.angle must be <= 180 degrees")
            if angle % 30 != 0:
                raise ValueError("turn.angle must be a multiple of 30 degrees")
            action = f"turn {direction}"
            return AgentDecision(
                kind=DECISION_ENVIRONMENT_ACTION,
                tool_call=tool_call,
                action=action,
                turn_angle=angle,
                raw_response=raw_response,
            )

        if name == LOOK_UP_TOOL_NAME:
            angle = int(arguments.get("angle", 15))
            if angle <= 0:
                raise ValueError("look_up.angle must be positive")
            if angle > 75:
                raise ValueError("look_up.angle must be <= 75 degrees")
            return AgentDecision(
                kind=DECISION_ENVIRONMENT_ACTION,
                tool_call=tool_call,
                action="look up",
                camera_angle=angle,
                raw_response=raw_response,
            )

        if name == LOOK_DOWN_TOOL_NAME:
            angle = int(arguments.get("angle", 30))
            if angle <= 0:
                raise ValueError("look_down.angle must be positive")
            if angle > 75:
                raise ValueError("look_down.angle must be <= 75 degrees")
            return AgentDecision(
                kind=DECISION_ENVIRONMENT_ACTION,
                tool_call=tool_call,
                action="look down",
                camera_angle=angle,
                raw_response=raw_response,
            )

        if name == REPORT_HAZARD_TOOL_NAME:
            hazards = _parse_hazards(arguments.get("hazards", []))
            return AgentDecision(
                kind=DECISION_REPORT_HAZARD,
                tool_call=tool_call,
                hazards=hazards,
                raw_response=raw_response,
            )

        if name == FINISH_INSPECTION_TOOL_NAME:
            return AgentDecision(
                kind=DECISION_FINISH,
                tool_call=tool_call,
                finish_reason=str(arguments.get("reason", "")).strip(),
                raw_response=raw_response,
            )

        raise ValueError(f"Unsupported tool call: {name}")
    except Exception as exc:
        return AgentDecision.error_decision(str(exc), raw_response, tool_call)

def parse_text_tool_call(text: str) -> ToolCall | None:
    stripped = text.strip().strip("` ").lower()
    alias_to_tool: dict[str, tuple[str, dict[str, Any]]] = {
        "walk straight": (WALK_TOOL_NAME, {"steps": 1}),
        "turn left": (TURN_TOOL_NAME, {"direction": "left", "angle": 90}),
        "turn right": (TURN_TOOL_NAME, {"direction": "right", "angle": 90}),
        "look up": (LOOK_UP_TOOL_NAME, {"angle": 15}),
        "look down": (LOOK_DOWN_TOOL_NAME, {"angle": 30}),
        "finish inspection": (FINISH_INSPECTION_TOOL_NAME, {"reason": "Finished inspection."}),
    }
    if stripped in alias_to_tool:
        name, arguments = alias_to_tool[stripped]
        return ToolCall(
            id=f"text_fallback_{name}",
            name=name,
            arguments=arguments,
            provider="text-fallback",
            raw=text,
        )
    bare_tool_names = {
        WALK_TOOL_NAME,
        TURN_TOOL_NAME,
        LOOK_UP_TOOL_NAME,
        LOOK_DOWN_TOOL_NAME,
        REPORT_HAZARD_TOOL_NAME,
        FINISH_INSPECTION_TOOL_NAME,
    }
    if stripped in bare_tool_names:
        defaults: dict[str, dict[str, Any]] = {
            WALK_TOOL_NAME: {"steps": 1},
            TURN_TOOL_NAME: {"direction": "right", "angle": 90},
            LOOK_UP_TOOL_NAME: {"angle": 15},
            LOOK_DOWN_TOOL_NAME: {"angle": 30},
            REPORT_HAZARD_TOOL_NAME: {"hazards": []},
            FINISH_INSPECTION_TOOL_NAME: {"reason": "Finished inspection."},
        }
        return ToolCall(
            id=f"text_fallback_{stripped}",
            name=stripped,
            arguments=defaults.get(stripped, {}),
            provider="text-fallback",
            raw=text,
        )

    payload = _load_json(text)
    if isinstance(payload, dict):
        if isinstance(payload.get("tool_call"), dict):
            payload = payload["tool_call"]

        name = payload.get("tool") or payload.get("name")
        arguments = payload.get("arguments", payload.get("args", {}))
        if name and isinstance(arguments, dict):
            return ToolCall(
                id=str(payload.get("id", "text_fallback_call")),
                name=str(name),
                arguments=arguments,
                provider="text-fallback",
                raw=payload,
            )

        legacy_action = str(payload.get("action", "")).strip().lower()
        if legacy_action in {"walk straight", "turn left", "turn right", "look up", "look down"}:
            return _legacy_action_to_tool_call(legacy_action, payload)

    # GLM Base sometimes emits truncated JSON such as {"tool":"walk",...
    # that is unparseable but still contains an unambiguous tool name.
    repaired = _fallback_tool_call_from_text(text)
    if repaired is not None:
        return repaired
    return None

def _fallback_tool_call_from_text(text: str) -> ToolCall | None:
    lowered = text.lower()
    defaults: dict[str, dict[str, Any]] = {
        WALK_TOOL_NAME: {"steps": 1},
        TURN_TOOL_NAME: {"direction": "right", "angle": 90},
        LOOK_UP_TOOL_NAME: {"angle": 15},
        LOOK_DOWN_TOOL_NAME: {"angle": 30},
        REPORT_HAZARD_TOOL_NAME: {"hazards": _extract_hazards_from_text(text)},
        FINISH_INSPECTION_TOOL_NAME: {"reason": "Finished inspection."},
    }
    aliases: list[tuple[str, str]] = [
        ('"tool":"walk"', WALK_TOOL_NAME),
        ('"tool": "walk"', WALK_TOOL_NAME),
        ('"tool":"turn"', TURN_TOOL_NAME),
        ('"tool": "turn"', TURN_TOOL_NAME),
        ('"tool":"look_up"', LOOK_UP_TOOL_NAME),
        ('"tool": "look_up"', LOOK_UP_TOOL_NAME),
        ('"tool":"look_down"', LOOK_DOWN_TOOL_NAME),
        ('"tool": "look_down"', LOOK_DOWN_TOOL_NAME),
        ('"tool":"report_hazard"', REPORT_HAZARD_TOOL_NAME),
        ('"tool": "report_hazard"', REPORT_HAZARD_TOOL_NAME),
        ('"tool":"finish_inspection"', FINISH_INSPECTION_TOOL_NAME),
        ('"tool": "finish_inspection"', FINISH_INSPECTION_TOOL_NAME),
        ('turn left', TURN_TOOL_NAME),
        ('turn right', TURN_TOOL_NAME),
        ('look up', LOOK_UP_TOOL_NAME),
        ('look down', LOOK_DOWN_TOOL_NAME),
    ]
    for marker, name in aliases:
        if marker in lowered:
            arguments = dict(defaults[name])
            if name == TURN_TOOL_NAME and 'turn left' in lowered:
                arguments["direction"] = "left"
            return ToolCall(
                id=f"text_fallback_repaired_{name}",
                name=name,
                arguments=arguments,
                provider="text-fallback-repaired",
                raw=text,
            )
    return None

def _extract_hazards_from_text(text: str) -> list[dict[str, Any]]:
    category = ""
    objects: list[str] = []
    if '"category"' in text:
        try:
            category = text.split('"category"', 1)[1].split(':', 1)[1].split('"', 2)[1]
        except Exception:
            category = ""
    if '"objects"' in text:
        try:
            inside = text.split('"objects"', 1)[1].split('[', 1)[1].split(']', 1)[0]
            objects = [part.split('"', 1)[0] for part in inside.split('"')[1::2] if part.strip()]
        except Exception:
            objects = []
    if category or objects:
        return [{"category": category or "Unknown Hazard", "objects": objects or ["object"]}]
    return []

def observation_to_message(observation: Any) -> ChatMessage:
    image_path = str(getattr(observation, "image_path", "") or "")
    step = int(getattr(observation, "step", 0))
    return ChatMessage.user(
        content=f"Observation step {step}. Inspect the attached first-person image and call exactly one tool.",
        images=[image_path] if image_path else [],
    )

def _walk_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": WALK_TOOL_NAME,
            "description": "Move forward in the current facing direction. This changes the environment state.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Number of cautious forward movement steps. Use 1 unless a longer movement is clearly needed.",
                    }
                },
                "required": ["steps"],
            },
            "strict": True,
        },
    }

def _turn_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TURN_TOOL_NAME,
            "description": "Turn left or right to inspect a different view. This changes the environment state.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Turn direction.",
                    },
                    "angle": {
                        "type": "integer",
                        "minimum": 30,
                        "maximum": 180,
                        "multipleOf": 30,
                        "description": "Turn angle in degrees. Turn actions are applied in about 30-degree increments.",
                    },
                },
                "required": ["direction", "angle"],
            },
            "strict": True,
        },
    }

def _look_up_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": LOOK_UP_TOOL_NAME,
            "description": "Look upward in the next observation without moving the character.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "angle": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 75,
                        "description": "Pitch angle in degrees. Use 15 for a mild upward glance unless a stronger angle is needed.",
                    }
                },
                "required": ["angle"],
            },
            "strict": True,
        },
    }

def _look_down_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": LOOK_DOWN_TOOL_NAME,
            "description": "Look downward in the next observation without moving the character.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "angle": {
                        "type": "integer",
                        "minimum": 5,
                        "maximum": 75,
                        "description": "Pitch angle in degrees. Use 30 for looking at floor-level hazards unless a stronger angle is needed.",
                    }
                },
                "required": ["angle"],
            },
            "strict": True,
        },
    }

def _report_hazard_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": REPORT_HAZARD_TOOL_NAME,
            "description": "Record hazards visible in the current observation without moving the character.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "hazards": {
                        "type": "array",
                        "items": _hazard_schema(),
                    }
                },
                "required": ["hazards"],
            },
            "strict": True,
        },
    }

def _finish_inspection_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": FINISH_INSPECTION_TOOL_NAME,
            "description": "Finish the current inspection sample when no more useful inspection is needed.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief reason for finishing the sample.",
                    }
                },
                "required": ["reason"],
            },
            "strict": True,
        },
    }

def _hazard_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "category": {
                "type": "string",
                "description": "Hazard category, such as fire, electrical, trip, falling-object, or child-safety.",
            },
            "objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Visible objects related to the hazard.",
            },
        },
        "required": ["category", "objects"],
    }

def _parse_hazards(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", [], "none", "None"):
        return []
    if not isinstance(value, list):
        raise ValueError("hazards must be a list")

    hazards: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"invalid hazard item: {item!r}")
        category = str(item.get("category", "")).strip()
        objects_value = item.get("objects", item.get("object_names"))
        if objects_value is None and item.get("object") is not None:
            objects_value = item.get("object")
        if objects_value is None and item.get("object_name") is not None:
            objects_value = item.get("object_name")
        if isinstance(objects_value, str):
            objects_value = [objects_value]
        if not isinstance(objects_value, list):
            raise ValueError("hazard.objects must be a list")
        objects = [str(object_name).strip() for object_name in objects_value if str(object_name).strip()]
        if category and objects:
            hazards.append(
                {
                    "category": category,
                    "objects": objects,
                }
            )
    return hazards

def _load_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

def _legacy_action_to_tool_call(action: str, payload: dict[str, Any]) -> ToolCall:
    if action == "walk straight":
        name = WALK_TOOL_NAME
        arguments: dict[str, Any] = {"steps": 1}
    elif action == "look up":
        name = LOOK_UP_TOOL_NAME
        arguments = {"angle": 15}
    elif action == "look down":
        name = LOOK_DOWN_TOOL_NAME
        arguments = {"angle": 30}
    else:
        name = TURN_TOOL_NAME
        arguments = {
            "direction": "left" if action == "turn left" else "right",
            "angle": 30,
        }
    return ToolCall(
        id=str(payload.get("id", f"text_fallback_{name}")),
        name=name,
        arguments=arguments,
        provider="text-fallback",
        raw=payload,
    )
