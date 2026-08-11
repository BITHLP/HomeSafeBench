"""Core metrics for HomeSafeBench evaluation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


DANGER_TYPES = ("trip", "falling", "fire", "children", "electric")
CATEGORY_ALIASES = {
    "fire": "fire", "burn": "fire", "burning": "fire", "flammable": "fire", "fire hazard": "fire",
    "electric": "electric", "electrical": "electric", "electricity": "electric", "electrocution": "electric", "electrical hazard": "electric",
    "fall": "falling", "falling": "falling", "falling object": "falling", "falling objects": "falling", "falling object hazard": "falling",
    "trip": "trip", "tripping": "trip", "trip hazard": "trip", "slip": "trip", "slipping": "trip",
    "children": "children", "child": "children", "child safety": "children", "child safety hazard": "children", "child-safety": "children",
}
JUDGE_RULES = """Judging rules:
- Each predicted hazard can match at most one golden hazard, and each golden hazard can match at most one predicted hazard.
- The hazard type must be the same or clearly semantically equivalent.
- The reported objects and golden objects should be semantically the same, closely related, or visually similar.
- The golden location may be used as supporting context.
- If the hazard type matches but the objects clearly refer to different items, mark the prediction as unmatched.
- If multiple matches seem plausible, choose the best one-to-one matching."""
JUDGE_PROMPT_TEMPLATE = """You are a senior evaluator for home safety inspection datasets.
You are strict about matching reports to the provided golden annotations and do not invent hazards that are not listed.

You will be given predicted hazards and golden hazards. Each includes a hazard type and the objects that cause the hazard. Each golden hazard also includes a semantic location.

Predicted hazards:

{predictions}

Golden hazards:

{golden_hazards}

Return one entry for every predicted hazard. Answer JSON only:
{{
  "predictions": [
{answer_items}
  ]
}}

Use null when a prediction does not match a golden hazard. Do not include explanations, markdown, or any text outside the JSON object.

{rules}"""


class JudgeParseError(ValueError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())


def normalize_category(value: Any) -> str:
    text = normalize_text(value)
    if text in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[text]
    if "electric" in text or "electrocution" in text:
        return "electric"
    if "falling" in text or text == "fall":
        return "falling"
    if "trip" in text or "slip" in text:
        return "trip"
    if "child" in text or "children" in text:
        return "children"
    if "fire" in text or "burn" in text or "flammable" in text:
        return "fire"
    return f"unknown:{text}" if text else "unknown:"


def prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def extract_gt_hazards(sample: dict[str, Any]) -> list[dict[str, Any]]:
    hazards = []
    for danger in sample.get("dangers", []):
        if not isinstance(danger, dict):
            continue
        objects = [item for item in danger.get("objects", []) if isinstance(item, dict)]
        hazards.append({"hazard_id": danger.get("id"), "category": normalize_category(danger.get("danger_type", "")), "raw_category": danger.get("danger_type", ""), "location": danger.get("name", ""), "objects": [str(item.get("class_name", "")) for item in objects]})
    return hazards


def extract_pred_hazards(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hazards = []
    for step in steps:
        decision = step.get("decision", {})
        if not isinstance(decision, dict) or decision.get("kind") != "report_hazard":
            continue
        for item in decision.get("hazards", []) or []:
            if not isinstance(item, dict):
                continue
            value = item.get("objects", item.get("object_names", item.get("object", item.get("object_name", []))))
            objects = [str(value)] if isinstance(value, str) else [str(entry) for entry in value or []]
            hazards.append({"category": normalize_category(item.get("category", "")), "raw_category": str(item.get("category", "")), "objects": [entry for entry in objects if entry.strip()]})
    return hazards


def compute_category_metrics(golden: list[str], predictions: list[str]) -> dict[str, float | int]:
    gt_counts, pred_counts = Counter(golden), Counter(predictions)
    types = set(gt_counts) | set(pred_counts)
    return prf(sum(min(gt_counts[key], pred_counts[key]) for key in types), sum(max(0, pred_counts[key] - gt_counts[key]) for key in types), sum(max(0, gt_counts[key] - pred_counts[key]) for key in types))


def build_judge_prompt(predictions: list[dict[str, Any]], golden: list[dict[str, Any]]) -> str:
    answer_items = ",\n".join(f'    {{"pred_id": "{item["pred_id"]}", "matched_golden_id": "Gn or null"}}' for item in predictions)
    pred_text = "\n\n".join(f'{item["pred_id"]}: type={item["category"]}; objects={", ".join(item["objects"])}' for item in predictions) or "(none)"
    golden_text = "\n\n".join(f'{item["golden_id"]}: type={item["category"]}; objects={", ".join(item["objects"])}; location={item["location"]}' for item in golden) or "(none)"
    return JUDGE_PROMPT_TEMPLATE.format(predictions=pred_text, golden_hazards=golden_text, answer_items=answer_items, rules=JUDGE_RULES)


def mock_judge_client(request: dict[str, Any]) -> dict[str, Any]:
    used, rows = set(), []
    for prediction in request["predictions"]:
        match = None
        objects = normalize_text(" ".join(prediction["objects"]))
        for golden in request["golden_hazards"]:
            if golden["golden_id"] in used or golden["category"] != prediction["category"]:
                continue
            if any(item and (item in objects or objects in item) for item in map(normalize_text, golden["objects"])):
                match = golden["golden_id"]
                used.add(match)
                break
        rows.append({"pred_id": prediction["pred_id"], "matched_golden_id": match})
    return {"predictions": rows}


def call_llm_judge(request: dict[str, Any], backend: str) -> Any:
    if backend == "mock":
        return mock_judge_client(request)
    raise NotImplementedError("Implement call_llm_judge() for your LLM provider, or use --judge_backend mock.")


def parse_judge_response(raw: Any, pred_ids: list[str], golden_ids: list[str]) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError as error:
            raise JudgeParseError("judge_invalid_json", str(error)) from error
    if not isinstance(raw, dict) or not isinstance(raw.get("predictions"), list):
        raise JudgeParseError("judge_invalid_schema", "response must contain a predictions list")
    seen_pred, seen_golden, result = set(), set(), []
    for row in raw["predictions"]:
        pred_id, golden_id = row.get("pred_id"), row.get("matched_golden_id") if isinstance(row, dict) else (None, None)
        if pred_id not in pred_ids or pred_id in seen_pred:
            raise JudgeParseError("judge_invalid_prediction_id", f"invalid pred_id: {pred_id}")
        if golden_id is not None and (golden_id not in golden_ids or golden_id in seen_golden):
            raise JudgeParseError("judge_invalid_golden_id", f"invalid golden ID: {golden_id}")
        seen_pred.add(pred_id)
        if golden_id is not None:
            seen_golden.add(golden_id)
        result.append({"pred_id": pred_id, "matched_golden_id": golden_id})
    if set(pred_ids) != seen_pred:
        raise JudgeParseError("judge_missing_prediction", "response omitted predictions")
    return sorted(result, key=lambda row: pred_ids.index(row["pred_id"]))


def compute_judge_metrics(pred_ids: list[str], golden_ids: list[str], matches: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["matched_golden_id"] is not None for row in matches)
    return prf(tp, len(pred_ids) - tp, len(golden_ids) - tp)
