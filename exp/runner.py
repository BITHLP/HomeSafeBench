from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from itertools import islice
from pathlib import Path
from typing import Any

from agent import (
    DECISION_ENVIRONMENT_ACTION,
    DECISION_ERROR,
    DECISION_FINISH,
    DECISION_REPORT_HAZARD,
    SafeAgent,
)
from env import EnvConfig, VirtualHomeEnv
from vlm import (
    MockVLM,
    ModelResponse,
    OpenAICompatibleConfig,
    OpenAICompatibleVLM,
    SequenceVLM,
    ToolCall,
)


def wait_until_ready(health_check: Callable[[], None], timeout: int = 120, interval: float = 2.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            health_check()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(interval)

    raise RuntimeError(f"Unity simulator is not ready after {timeout}s: {last_error}")


def iter_samples(input_path: str | Path):
    path = Path(input_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            yield path.name, json.load(f)
        return

    for item in sorted(path.iterdir()):
        if item.suffix != ".json":
            continue
        with item.open("r", encoding="utf-8") as f:
            yield item.name, json.load(f)


def output_path_for(output_dir: str | Path, filename: str) -> Path:
    return Path(output_dir) / f"output_{filename}"


def image_path_for(image_dir: str | Path, filename: str, step: int) -> Path:
    stem = Path(filename).stem
    return Path(image_dir) / stem / f"{step:03d}.jpg"


def build_vlm(args: argparse.Namespace):
    if args.vlm == "mock":
        if args.mock_plan:
            return SequenceVLM([_mock_response(item, index) for index, item in enumerate(args.mock_plan.split(","))])
        return MockVLM()
    if args.vlm == "openai-compatible":
        if not args.model:
            raise ValueError("--model is required for --vlm openai-compatible")
        return OpenAICompatibleVLM(
            OpenAICompatibleConfig(
                model=args.model,
                base_url=args.base_url,
                timeout=args.timeout,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        )
    raise ValueError(f"Unsupported vlm: {args.vlm}")


def run(args: argparse.Namespace) -> None:
    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    Path(args.image_path).mkdir(parents=True, exist_ok=True)

    env = VirtualHomeEnv(
        EnvConfig(
            port=args.port,
            timeout_wait=args.timeout_wait,
            image_width=args.image_width,
            image_height=args.image_height,
            character_resource=args.character_resource,
            action_repeats=args.action_repeats,
            look_down_camera_offset=args.look_down_camera_offset,
        )
    )
    try:
        env.connect()
        wait_until_ready(env.health_check, timeout=args.startup_timeout)

        samples = iter_samples(args.input_path)
        if args.limit is not None:
            samples = islice(samples, args.limit)

        consecutive_failures = 0
        for filename, sample in samples:
            print(f"[runner] start sample={filename}", flush=True)
            out_path = output_path_for(args.output_path, filename)
            if out_path.exists() and not args.overwrite:
                continue

            try:
                agent = SafeAgent(
                    build_vlm(args),
                    tool_choice=args.tool_choice,
                    max_new_tokens=args.max_new_tokens,
                )
                result = run_sample(args, env, agent, filename, sample)
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                result = {
                    "info": {
                        "filename": filename,
                        "status": "error",
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                    "steps": [],
                }
                try:
                    env.reconnect()
                except Exception as reconnect_exc:
                    result["info"]["reconnect_error"] = str(reconnect_exc)

            write_json(out_path, result)
            print(
                f"[runner] wrote sample={filename} status={result['info'].get('status')} "
                f"steps={len(result.get('steps', []))}",
                flush=True,
            )
            if consecutive_failures >= args.max_consecutive_failures:
                raise RuntimeError(f"Reached max_consecutive_failures={args.max_consecutive_failures}")
    finally:
        env.close()


def run_sample(
    args: argparse.Namespace,
    env: VirtualHomeEnv,
    agent: SafeAgent,
    filename: str,
    sample: dict[str, Any],
) -> dict[str, Any]:
    env.reset(sample)

    status = "max_turns"
    finish_reason = ""
    previous_action = None
    steps: list[dict[str, Any]] = []
    reported_hazards: list[dict[str, Any]] = []
    consecutive_tool_errors = 0

    for step in range(args.run_turns):
        step_start = time.time()
        print(f"[runner] step={step} begin", flush=True)
        image_path = image_path_for(args.image_path, filename, step)
        observe_start = time.time()
        observation = env.observe(step=step, image_path=image_path, previous_action=previous_action)
        print(
            f"[runner] step={step} observe_sec={time.time() - observe_start:.2f} image={image_path}",
            flush=True,
        )
        decide_start = time.time()
        decision = agent.decide(observation)
        print(
            f"[runner] step={step} agent_sec={time.time() - decide_start:.2f} "
            f"kind={decision.kind} action={decision.action} "
            f"tool={decision.tool_call.name if decision.tool_call else ''} "
            f"messages={len(agent.messages)}",
            flush=True,
        )

        feedback: dict[str, Any]
        tool_error: dict[str, Any] | None = None
        action_state = ""
        env_action_start = time.time()
        if decision.kind == DECISION_ERROR:
            consecutive_tool_errors += 1
            retryable = consecutive_tool_errors < args.max_consecutive_tool_errors
            feedback = tool_error_feedback(decision.error, retryable=retryable)
            tool_error = {
                "retryable": retryable,
                "consecutive_tool_errors": consecutive_tool_errors,
                "max_consecutive_tool_errors": args.max_consecutive_tool_errors,
            }
            if decision.tool_call is not None:
                agent.add_tool_feedback(decision.tool_call, feedback)
            elif retryable:
                agent.add_retry_instruction(tool_error_retry_instruction(decision.error))
            if not retryable:
                status = "error"
        elif decision.kind == DECISION_ENVIRONMENT_ACTION:
            action_state = env.step(
                decision.action,
                angle=decision.camera_angle,
                movement_steps=decision.movement_steps,
                turn_angle=decision.turn_angle,
            )
            consecutive_tool_errors = 0
            previous_action = action_state
            feedback = {"ok": True, "action_state": action_state}
            if decision.movement_steps is not None:
                feedback["movement_steps"] = decision.movement_steps
            if decision.turn_angle is not None:
                feedback["turn_angle"] = decision.turn_angle
            if decision.camera_angle is not None:
                feedback["camera_angle"] = decision.camera_angle
            agent.add_tool_feedback(decision.tool_call, feedback)
        elif decision.kind == DECISION_REPORT_HAZARD:
            consecutive_tool_errors = 0
            reported_hazards.extend(decision.hazards)
            feedback = {
                "ok": True,
                "recorded": True,
                "count": len(decision.hazards),
            }
            agent.add_tool_feedback(decision.tool_call, feedback)
        elif decision.kind == DECISION_FINISH:
            consecutive_tool_errors = 0
            status = "success"
            finish_reason = decision.finish_reason
            feedback = {
                "ok": True,
                "finished": True,
                "reason": decision.finish_reason,
            }
            agent.add_tool_feedback(decision.tool_call, feedback)
        else:
            status = "error"
            feedback = {"ok": False, "error": f"Unsupported decision kind: {decision.kind}"}
        print(
            f"[runner] step={step} feedback_sec={time.time() - env_action_start:.2f} "
            f"status={status} feedback={feedback}",
            flush=True,
        )

        step_record = {
            "step": step,
            "image_path": str(image_path),
            "seen_objects": make_jsonable(observation.visible_objects),
            "character_nodes": make_jsonable(observation.character_nodes),
            "character_bbox_center": character_bbox_center(observation.character_nodes),
            "character_obj_position": character_obj_position(observation.character_nodes),
            "decision": decision_to_record(decision),
            "feedback": feedback,
            "action_state": action_state,
        }
        if tool_error is not None:
            step_record["tool_error"] = tool_error
        steps.append(step_record)

        if args.write_each_step:
            partial_result = build_sample_result(
                args=args,
                filename=filename,
                status=status,
                finish_reason=finish_reason,
                reported_hazards=reported_hazards,
                steps=steps,
            )
            write_json(output_path_for(args.output_path, filename), partial_result)

        if status in {"success", "error"}:
            print(f"[runner] step={step} end_sec={time.time() - step_start:.2f}", flush=True)
            break
        print(f"[runner] step={step} end_sec={time.time() - step_start:.2f}", flush=True)

    result = build_sample_result(
        args=args,
        filename=filename,
        status=status,
        finish_reason=finish_reason,
        reported_hazards=reported_hazards,
        steps=steps,
    )
    return result


def build_sample_result(
    args: argparse.Namespace,
    filename: str,
    status: str,
    finish_reason: str,
    reported_hazards: list[dict[str, Any]],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "info": {
            "filename": filename,
            "status": status,
            "finish_reason": finish_reason,
            "vlm": args.vlm,
            "model": args.model or args.vlm,
            "sim_manager": "external",
            "port": args.port,
            "max_consecutive_tool_errors": args.max_consecutive_tool_errors,
            "reported_hazards": reported_hazards,
        },
        "steps": steps,
    }


def tool_error_feedback(
    error: str,
    retryable: bool,
) -> dict[str, Any]:
    feedback = {
        "ok": False,
        "error": error,
    }
    if retryable:
        feedback["instruction"] = "Retry with exactly one valid tool call. The previous tool call was not executed."
    else:
        feedback["instruction"] = "Maximum consecutive invalid tool calls reached. This sample is marked as error."
    return feedback


def tool_error_retry_instruction(error: str) -> str:
    return (
        "Your previous response did not produce one valid tool call and no environment action was executed. "
        f"Error: {error}. Retry now with exactly one valid tool call."
    )


def decision_to_record(decision) -> dict[str, Any]:
    tool_call = decision.tool_call.to_debug_dict() if decision.tool_call is not None else None
    return {
        "kind": decision.kind,
        "ok": decision.ok,
        "action": decision.action,
        "movement_steps": decision.movement_steps,
        "turn_angle": decision.turn_angle,
        "camera_angle": decision.camera_angle,
        "hazards": decision.hazards,
        "finish_reason": decision.finish_reason,
        "raw_response": decision.raw_response,
        "error": decision.error,
        "tool_call": tool_call,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def make_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, tuple):
            return [make_jsonable(item) for item in value]
        if isinstance(value, list):
            return [make_jsonable(item) for item in value]
        if isinstance(value, dict):
            return {str(key): make_jsonable(item) for key, item in value.items()}
        return repr(value)


def character_bbox_center(character_nodes: list[dict[str, Any]]) -> list[Any] | None:
    node = first_character_node(character_nodes)
    if not node:
        return None
    bounding_box = node.get("bounding_box")
    if not isinstance(bounding_box, dict):
        return None
    center = bounding_box.get("center")
    return list(center) if isinstance(center, list) else None


def character_obj_position(character_nodes: list[dict[str, Any]]) -> list[Any] | None:
    node = first_character_node(character_nodes)
    if not node:
        return None
    obj_transform = node.get("obj_transform")
    if not isinstance(obj_transform, dict):
        return None
    position = obj_transform.get("position")
    return list(position) if isinstance(position, list) else None


def first_character_node(character_nodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not character_nodes:
        return None
    node = character_nodes[0]
    return node if isinstance(node, dict) else None


def resolve_tool_choice(value: str) -> str | dict[str, Any] | None:
    if value in {"", "default"}:
        return "required"
    if value == "none":
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight SafeAgent experiments.")
    parser.add_argument("--input_path", required=True, help="Dataset JSON file or directory.")
    parser.add_argument("--output_path", required=True, help="Directory for result JSON files.")
    parser.add_argument("--image_path", default="tmp/images", help="Directory for captured observation images.")
    parser.add_argument("--run_turns", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--write_each_step",
        action="store_true",
        help="Write the result after each completed step.",
    )
    parser.add_argument("--max_consecutive_failures", type=int, default=5)
    parser.add_argument(
        "--max_consecutive_tool_errors",
        type=int,
        default=3,
        help="Stop the current sample after this many consecutive invalid tool calls.",
    )

    parser.add_argument("--vlm", choices=["mock", "openai-compatible"], default="mock")
    parser.add_argument(
        "--mock_plan",
        default="",
        help="Comma-separated mock tool plan, e.g. walk,turn_left,report,finish.",
    )
    parser.add_argument("--model", default=None, help="Model identifier for an external OpenAI-compatible endpoint.")
    parser.add_argument("--base_url", default=None, help="Base URL for an external OpenAI-compatible endpoint.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--tool_choice",
        default="required",
        help="Tool choice passed to the VLM: required, auto, none, or default.",
    )

    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout_wait", type=int, default=60)
    parser.add_argument("--startup_timeout", type=int, default=120)

    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=360)
    parser.add_argument("--character_resource", default="Chars/Female2")
    parser.add_argument("--action_repeats", type=int, default=1)
    parser.add_argument(
        "--look_down_camera_offset",
        type=int,
        default=-1,
        help="Fallback camera offset for look_down when character camera names cannot be resolved.",
    )
    args = parser.parse_args(argv)
    if args.max_consecutive_tool_errors < 1:
        parser.error("--max_consecutive_tool_errors must be >= 1")
    args.tool_choice = resolve_tool_choice(args.tool_choice)
    return args


def main() -> None:
    run(parse_args())


def _mock_response(name: str, index: int) -> ModelResponse:
    item = name.strip().lower().replace("-", "_").replace(" ", "_")
    call_id = f"mock_{index}_{item}"
    if item in {"walk", "walk_straight"}:
        tool_call = ToolCall(id=call_id, name="walk", arguments={"steps": 1}, provider="mock")
    elif item in {"turn_left", "left"}:
        tool_call = ToolCall(
            id=call_id,
            name="turn",
            arguments={"direction": "left", "angle": 30},
            provider="mock",
        )
    elif item in {"turn_right", "right"}:
        tool_call = ToolCall(
            id=call_id,
            name="turn",
            arguments={"direction": "right", "angle": 30},
            provider="mock",
        )
    elif item in {"look_up", "lookup"} or item.startswith("look_up_") or item.startswith("lookup_"):
        tool_call = ToolCall(
            id=call_id,
            name="look_up",
            arguments={"angle": _mock_angle(item, default=15)},
            provider="mock",
        )
    elif item in {"look_down", "lookdown"} or item.startswith("look_down_") or item.startswith("lookdown_"):
        tool_call = ToolCall(
            id=call_id,
            name="look_down",
            arguments={"angle": _mock_angle(item, default=30)},
            provider="mock",
        )
    elif item in {"report", "report_hazard"}:
        tool_call = ToolCall(
            id=call_id,
            name="report_hazard",
            arguments={
                "hazards": [
                    {
                        "category": "mock",
                        "objects": ["mock object"],
                    }
                ]
            },
            provider="mock",
        )
    elif item in {"finish", "finish_inspection"}:
        tool_call = ToolCall(
            id=call_id,
            name="finish_inspection",
            arguments={"reason": "mock finished"},
            provider="mock",
        )
    else:
        raise ValueError(f"Unsupported mock plan item: {name}")
    return ModelResponse(tool_calls=[tool_call], allow_text_fallback=False)


def _mock_angle(item: str, default: int) -> int:
    try:
        return int(item.rsplit("_", 1)[-1])
    except ValueError:
        return default


if __name__ == "__main__":
    main()
