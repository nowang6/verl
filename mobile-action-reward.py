"""Reward for mobile tool-calling: match tool name + arguments against expected list."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_json_obj(raw: Any) -> Any:
    """Parse raw value as JSON; returns dict, list, or str, or None on failure."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    return None


def _normalize_arguments(args: Any) -> str:
    if args is None:
        return json.dumps({}, sort_keys=True)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return json.dumps(args, sort_keys=True)
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def _tool_from_gt_item(item: Any) -> Optional[tuple[str, str]]:
    """Extract (name, canonical_args_json) from a single expected-tool item.

    Supports:
    - ``{"function": {"name": "...", "arguments": ...}}`` (OpenAI-like)
    - ``{"name": "...", "arguments": ...}`` (Hermes-style)
    """
    gt = _parse_json_obj(item)
    if not isinstance(gt, dict):
        return None
    if "function" in gt and isinstance(gt["function"], dict):
        fn = gt["function"]
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        return (str(name), _normalize_arguments(args)) if name is not None else None
    name = gt.get("name") or gt.get("function")
    args = gt.get("arguments")
    if args is None and "parameters" in gt:
        args = gt["parameters"]
    return (str(name), _normalize_arguments(args)) if name is not None else None


def _extract_expected_tools(ground_truth: Any) -> list[tuple[str, str]]:
    """Return list of (tool_name, canonical_args_json) from ground truth.

    Supports the same shapes as before, PLUS a JSON list of tool descriptors
    (the format used by the td-mobile dataset for multi-step tasks).
    """
    gt = _parse_json_obj(ground_truth)
    if gt is None:
        if isinstance(ground_truth, str) and ground_truth.strip():
            return [(ground_truth.strip(), json.dumps({}, sort_keys=True))]
        return []

    # List of tool descriptors
    if isinstance(gt, list):
        tools = []
        for item in gt:
            t = _tool_from_gt_item(item)
            if t is not None:
                tools.append(t)
        return tools

    # Single tool descriptor
    t = _tool_from_gt_item(gt)
    if t is not None:
        return [t]
    return []


def _all_predicted_hermes_tools(solution_str: str) -> list[tuple[str, str]]:
    """Parse ALL Hermes ``<tool_call>{...}</tool_call>`` blocks from model output."""
    if not solution_str:
        return []
    tools = []
    for m in _TOOL_CALL_BLOCK.finditer(solution_str):
        inner = m.group(1).strip()
        obj = _parse_json_obj(inner)
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if name is None:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        tools.append((str(name), _normalize_arguments(args)))
    return tools


def _match_multi(expected: list[tuple[str, str]], predicted: list[tuple[str, str]]) -> float:
    """Proportional match: return fraction of expected tools that are matched.

    Each expected tool (name + canonical arguments JSON) is matched against
    unused predicted tools. Returns matched_count / len(expected), or 0.0 if
    expected is empty.
    """
    if not expected:
        return 0.0
    used = [False] * len(predicted)
    matched = 0
    for exp_name, exp_args in expected:
        for i, (pred_name, pred_args) in enumerate(predicted):
            if used[i]:
                continue
            if pred_name == exp_name and pred_args == exp_args:
                used[i] = True
                matched += 1
                break
    return matched / len(expected)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    method="strict",
    format_score=0.0,
    score=1.0,
    **kwargs,
):
    """Score proportionally by fraction of expected tool calls matched.

    ``ground_truth`` can be a single tool descriptor or a JSON list of them.
    The model output may contain multiple ``<tool_call>`` blocks. Score is
    matched_count / len(expected), giving partial credit for partially
    correct tool-calling sequences.
    """
    del data_source, method, format_score, score, kwargs

    expected = _extract_expected_tools(ground_truth)
    predicted = _all_predicted_hermes_tools(solution_str or "")

    if not expected:
        result = {"score": 0.0, "acc": False, "reason": "invalid_ground_truth", "n_expected": 0, "n_matched": 0}
        print(f"[score] {result}")
        print(f"[response] {solution_str[:500]}")
        print(f"[ground_truth] {ground_truth}")
        return result

    if not predicted:
        result = {"score": 0.0, "acc": False, "reason": "no_tool_call_in_completion", "n_expected": len(expected), "n_matched": 0}
        print(f"[score] {result}")
        print(f"[response] {solution_str[:500]}")
        print(f"[ground_truth] {ground_truth}")
        return result

    ratio = _match_multi(expected, predicted)
    result = {
        "score": ratio,
        "acc": ratio >= 1.0,
        "n_expected": len(expected),
        "n_matched": int(ratio * len(expected)),
        "expected": [{"name": n, "arguments": a} for n, a in expected],
        "predicted": [{"name": n, "arguments": a} for n, a in predicted],
    }
    print(f"[score] ratio={ratio:.2f} ({result['n_matched']}/{result['n_expected']})")
    print(f"[response] {solution_str[:500]}")
    print(f"[ground_truth] {ground_truth}")
    return result
