"""Evaluate HomeSafeBench runner results."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from eval_utils import DANGER_TYPES, build_judge_prompt, call_llm_judge, compute_category_metrics, compute_judge_metrics, extract_gt_hazards, extract_pred_hazards, parse_judge_response, prf, read_json, write_json, write_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--use_judge", action="store_true")
    parser.add_argument("--judge_backend", choices=["mock", "custom"], default="mock")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_eval_step", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def result_path(result_dir: Path, filename: str) -> Path:
    return result_dir / f"output_{filename}"


def empty_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def add_counts(total: dict[str, int], value: dict[str, Any]) -> None:
    for key in total:
        total[key] += int(value.get(key, 0))


def type_metrics(golden: list[dict[str, Any]], predictions: list[dict[str, Any]], matches: list[dict[str, Any]] | None) -> dict[str, dict[str, dict[str, float | int]]]:
    result = {kind: {"category_level": empty_counts(), "hazard_level": empty_counts()} for kind in DANGER_TYPES}
    for kind in DANGER_TYPES:
        add_counts(result[kind]["category_level"], compute_category_metrics([item["category"] for item in golden if item["category"] == kind], [item["category"] for item in predictions if item["category"] == kind]))
    if matches is None:
        return result
    pred_by_id = {f"P{index + 1}": item for index, item in enumerate(predictions)}
    golden_by_id = {f"G{index + 1}": item for index, item in enumerate(golden)}
    matched_golden = {row["matched_golden_id"] for row in matches if row["matched_golden_id"]}
    for row in matches:
        prediction = pred_by_id[row["pred_id"]]
        golden_id = row["matched_golden_id"]
        if golden_id is None:
            kind = prediction["category"]
            if kind in result:
                result[kind]["hazard_level"]["fp"] += 1
        else:
            golden_item = golden_by_id[golden_id]
            if prediction["category"] != golden_item["category"]:
                raise ValueError("judge matched hazards with different canonical categories")
            result[golden_item["category"]]["hazard_level"]["tp"] += 1
    for golden_id, golden_item in golden_by_id.items():
        if golden_id not in matched_golden:
            result[golden_item["category"]]["hazard_level"]["fn"] += 1
    return result


def evaluate_sample(sample_path: Path, result_dir: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample = read_json(sample_path)
    golden = extract_gt_hazards(sample)
    result_file = result_path(result_dir, sample_path.name)
    issues: list[str] = []
    predictions: list[dict[str, Any]] = []
    if result_file.is_file():
        try:
            result = read_json(result_file)
            predictions = extract_pred_hazards(result.get("steps", [])[:args.max_eval_step])
        except Exception as error:
            issues.append(f"invalid_result: {error}")
    else:
        issues.append("missing_result")
    category = compute_category_metrics([item["category"] for item in golden], [item["category"] for item in predictions])
    matches = None
    judge = None
    judge_records: list[dict[str, Any]] = []
    if args.use_judge:
        judge_predictions = [{**item, "pred_id": f"P{index + 1}"} for index, item in enumerate(predictions)]
        judge_golden = [{**item, "golden_id": f"G{index + 1}"} for index, item in enumerate(golden)]
        if judge_predictions:
            prompt = build_judge_prompt(judge_predictions, judge_golden)
            raw = call_llm_judge({"prompt": prompt, "predictions": judge_predictions, "golden_hazards": judge_golden}, args.judge_backend)
            matches = parse_judge_response(raw, [item["pred_id"] for item in judge_predictions], [item["golden_id"] for item in judge_golden])
            judge_records = [{"filename": sample_path.name, **row} for row in matches]
        else:
            matches = []
        judge = compute_judge_metrics([item["pred_id"] for item in judge_predictions], [item["golden_id"] for item in judge_golden], matches)
    return {"filename": sample_path.name, "category_level": category, "hazard_level": judge, "by_danger_type": type_metrics(golden, predictions, matches), "issues": issues}, judge_records


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    category, hazard = empty_counts(), empty_counts()
    by_type = {kind: {"category_level": empty_counts(), "hazard_level": empty_counts()} for kind in DANGER_TYPES}
    judge_enabled = any(record["hazard_level"] is not None for record in records)
    for record in records:
        add_counts(category, record["category_level"])
        if record["hazard_level"] is not None:
            add_counts(hazard, record["hazard_level"])
        for kind in DANGER_TYPES:
            for metric in by_type[kind]:
                add_counts(by_type[kind][metric], record["by_danger_type"][kind][metric])
    for kind in DANGER_TYPES:
        by_type[kind] = {metric: prf(**counts) if metric == "category_level" or judge_enabled else None for metric, counts in by_type[kind].items()}
    return {"category_level": prf(**category), "hazard_level": prf(**hazard) if judge_enabled else None, "by_danger_type": by_type}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    data_dir, result_dir, output_dir = Path(args.data_dir), Path(args.result_dir), Path(args.output_dir)
    paths = sorted(path for path in data_dir.glob("*.json") if path.name.partition("_")[0].isdigit())
    if args.limit is not None:
        paths = paths[:args.limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not (args.resume or args.overwrite):
        raise FileExistsError("output_dir is not empty; pass --overwrite")
    config = {"use_judge": args.use_judge, "judge_backend": args.judge_backend, "max_eval_step": args.max_eval_step}
    checkpoint = output_dir / "checkpoint.json"
    state = read_json(checkpoint) if args.resume and checkpoint.is_file() else {"config": config, "records": {}}
    if state.get("config") != config or not isinstance(state.get("records"), dict):
        raise ValueError("checkpoint does not match this evaluation; pass --overwrite")
    for sample_path in paths:
        if sample_path.name in state["records"]:
            continue
        record, matches = evaluate_sample(sample_path, result_dir, args)
        state["records"][sample_path.name] = {"record": record, "judge_records": matches}
        write_json(checkpoint, state)
    records = [state["records"][name]["record"] for name in sorted(state["records"])]
    judge_records = [row for name in sorted(state["records"]) for row in state["records"][name]["judge_records"]]
    overall = aggregate(records)
    summary = {"config": config, "overall": {key: overall[key] for key in ("category_level", "hazard_level")}, "by_danger_type": overall["by_danger_type"], "sample_count": len(records), "issue_counts": dict(Counter(issue for record in records for issue in record["issues"]))}
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "samples.jsonl", records)
    write_jsonl(output_dir / "judge_predictions.jsonl", judge_records)
    print("Hazard-level:", summary["overall"]["hazard_level"])


if __name__ == "__main__":
    main()
