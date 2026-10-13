"""Offline audit of recorded native runs; never captures or controls the game."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def recorded_scores(root: Path) -> dict[str, int]:
    scores: dict[str, int] = {}
    for path in (root / "evaluation").glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for episode in data.get("episodes", []):
            if not isinstance(episode, dict):
                continue
            name, score = episode.get("name"), episode.get("score")
            if isinstance(name, str) and isinstance(score, int):
                scores[name] = score
    return scores


def crossing_times(frames: list[dict], offset: float = 14, width: float = 15) -> list[float]:
    """Count observed lip crossings, not a claim of exact game score.

    Debouncing relies on the website's approximately 1.42 s pipe interval.
    The visual lip and scoring rectangle need not share the same right edge.
    """
    events: list[float] = []
    for frame in frames:
        x, gates, t = frame.get("x"), frame.get("gates"), frame.get("t")
        if x is None or t is None or not gates:
            continue
        threshold = x - offset
        if any(threshold - width <= gate["right"] <= threshold for gate in gates):
            if not events or t - events[-1] > .9:
                events.append(t)
    return events


def crossing_transitions(frames: list[dict], offset: float = 14, width: float = 15) -> Counter:
    """Distribution of next visible gap changes at scored-pipe crossings."""
    result: Counter = Counter()
    last_event = float("-inf")
    for frame in frames:
        x, gates, t = frame.get("x"), frame.get("gates"), frame.get("t")
        if x is None or t is None or not gates:
            continue
        threshold = x - offset
        candidates = [gate for gate in gates if threshold - width <= gate["right"] <= threshold]
        if not candidates or t - last_event <= .9:
            continue
        current = min(candidates, key=lambda gate: abs(gate["right"] - threshold))
        next_gates = [gate for gate in gates if gate["x"] > current["right"] + 30]
        if next_gates:
            next_gate = min(next_gates, key=lambda gate: gate["x"])
            change = next_gate["top"] - current["top"]
            result["high_to_low" if change >= 60 else "low_to_high" if change <= -60 else "other"] += 1
        else:
            result["UNKNOWN"] += 1
        last_event = t
    return result


def last_geometry(frames: list[dict]) -> dict:
    if not frames:
        return {"candidate": "UNKNOWN"}
    frame = frames[-1]
    gates = sorted(frame.get("gates", []), key=lambda gate: gate["x"])
    x, y = frame.get("x"), frame.get("y")
    if x is None or y is None:
        return {"candidate": "UNKNOWN"}
    hw, hh = frame.get("w", 0) / 2, frame.get("h", 0) / 2
    overlapping = [gate for gate in gates if gate["x"] < x + hw and gate["right"] > x - hw]
    candidate = "not_observed"
    top_margin = bottom_margin = None
    if overlapping:
        gate = overlapping[0]
        top_margin = y - hh - gate["top"]
        bottom_margin = gate["bottom"] - y - hh
        if top_margin < 0:
            candidate = "upper_overlap_last_frame"
        elif bottom_margin < 0:
            candidate = "lower_overlap_last_frame"
        else:
            candidate = "inside_gap_last_frame"
    transition = "UNKNOWN"
    if len(gates) >= 2:
        change = gates[1]["top"] - gates[0]["top"]
        transition = "high_to_low" if change >= 60 else "low_to_high" if change <= -60 else "other"
    return {"candidate": candidate, "top_margin_bbox_px": top_margin,
            "bottom_margin_bbox_px": bottom_margin, "transition": transition,
            "last_bird": {k: frame.get(k) for k in ("t", "x", "y", "w", "h", "v")},
            "last_gates": gates[:2]}


def audit(root: Path) -> dict:
    known = recorded_scores(root)
    episodes: list[dict] = []
    for path in sorted((root / "logs").glob("local-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "frames" not in data:
            continue
        name = data.get("summary", {}).get("name") or path.stem.removeprefix("local-")
        frames = data["frames"]
        crossings = crossing_times(frames)
        actual = known.get(name)
        summary = data.get("summary", {})
        episodes.append({"name": name, "source_file": str(path.relative_to(root)),
                         "score_recorded": actual, "crossing_estimate": len(crossings),
                         "crossing_times_s": crossings,
                         "survival_s": summary.get("observed_span_s"),
                         "end_reason": summary.get("reason"),
                         "frames": len(frames), "flaps": summary.get("flaps"),
                         "final_observation": last_geometry(frames)})
    validated = [e for e in episodes if e["score_recorded"] is not None]
    errors = [e["crossing_estimate"] - e["score_recorded"] for e in validated]
    finished = [e for e in episodes if e["end_reason"] == "modal/end"]
    exposure: Counter = Counter()
    for episode in finished:
        data = json.loads((root / episode["source_file"]).read_text(encoding="utf-8"))
        exposure.update(crossing_transitions(data["frames"]))
    score_bins = ((0, 0), (1, 9), (10, 49), (50, 99), (100, 249), (250, 499), (500, 999), (1000, 10**9))
    bins = []
    for lo, hi in score_bins:
        group = [e for e in finished if lo <= (e["score_recorded"] if e["score_recorded"] is not None else e["crossing_estimate"]) <= hi]
        bins.append({"score_range": [lo, hi], "episodes": len(group),
                     "recorded_scores": sum(e["score_recorded"] is not None for e in group),
                     "geometry_candidates": dict(Counter(e["final_observation"]["candidate"] for e in group)),
                     "transitions": dict(Counter(e["final_observation"].get("transition", "UNKNOWN") for e in group))})
    return {"method": "observed gate right <= bird x - 14, within previous 15 px; 0.9 s debounce",
            "score_is_estimate": True, "local_files": len(episodes),
            "ended_episodes": len(finished), "validated": len(validated),
            "exact_matches": sum(error == 0 for error in errors),
            "absolute_error_max": max(map(abs, errors), default=None),
            "absolute_error_mean": sum(map(abs, errors)) / len(errors) if errors else None,
            "score_bins": bins, "crossing_transition_exposure": dict(exposure),
            "episodes": episodes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"},
                     ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
