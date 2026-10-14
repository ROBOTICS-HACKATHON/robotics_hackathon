"""Evaluate saved reports against human photo labels, never passed to the solver."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from typed_errs import Err, Ok, Result

from camera_experiment.geometry import Failure, attempt
from camera_experiment.run import Record, save_json


def evaluate(reports: list[Path]) -> Result[Record, Failure]:
    """Compare exact selections and coordinate invariants with visible photo content."""
    return attempt("evaluate reports", lambda: _evaluate(reports))


def _evaluate(reports: list[Path]) -> Record:
    expected = cast(dict[str, Record], json.loads(Path(__file__).with_name("expected.json").read_text()))
    groups: dict[str, Record] = {}
    rows: list[Record] = []
    for path in reports:
        report = cast(Record, json.loads(path.read_text()))
        for record in cast(list[Record], report["records"]):
            name = Path(cast(str, record["source"])).name
            truth = expected[name]
            plan = cast(Record, record.get("solver", {}))
            has_plan = record["status"] == "solved"
            matches = has_plan and plan.get("target") == truth["target"] and plan.get("tiles") == truth["tiles"]
            actions = cast(list[Record], plan.get("actions", []))
            width, height = cast(list[int], plan.get("image_size", [1, 1]))
            valid = has_plan and bool(actions) and actions[-1].get("role") == "verify"
            for action in actions:
                x, y = float(cast(float, action["x"])), float(cast(float, action["y"]))
                u, v = float(cast(float, action["x_norm"])), float(cast(float, action["y_norm"]))
                valid = valid and 0 <= u <= 1 and 0 <= v <= 1
                valid = valid and abs(u - x / width) < 1e-5 and abs(v - y / height) < 1e-5
            key = f"{path.parent.name}/{record['variant']}"
            group = groups.setdefault(key, {"model": report["model"], "cases": 0, "visible": 0,
                                           "layouts": 0, "plans": 0, "exact": 0, "valid_coordinates": 0,
                                           "correct_absence": 0, "geometry_errors": 0})
            increments = {"cases": True, "visible": truth["visible"], "layouts": "layout" in record,
                          "plans": has_plan, "exact": matches, "valid_coordinates": valid,
                          "correct_absence": not truth["visible"] and record["status"] == "no_dialog",
                          "geometry_errors": record["status"] == "geometry_error"}
            for field, increment in increments.items():
                group[field] = cast(int, group[field]) + int(bool(increment))
            rows.append({"run": path.parent.name, "source": name, "variant": record["variant"],
                         "status": record["status"], "expected_target": truth["target"],
                         "expected_tiles": truth["tiles"], "target": plan.get("target", ""),
                         "tiles": plan.get("tiles", []), "exact": matches, "coordinates_valid": valid})
    return {"label_source": "human inspection of supplied photos; tile 0 of dogs is partly occluded by debug labels",
            "groups": groups, "records": rows}


def main() -> Result[int, Failure]:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("reports", type=Path, nargs="+")
    _ = parser.add_argument("--out", type=Path, default=Path(__file__).with_name("comparison.json"))
    parsed = parser.parse_args()
    result = evaluate(cast(list[Path], parsed.reports))
    if isinstance(result, Err):
        return result
    saved = save_json(cast(Path, parsed.out), result.value)
    if isinstance(saved, Err):
        return saved
    print(json.dumps(result.value["groups"], indent=2))
    return Ok(0)


if __name__ == "__main__":
    try:
        outcome = main()
        if isinstance(outcome, Err):
            print(outcome.context_msg, file=sys.stderr)
            sys.exit(1)
        sys.exit(outcome.value)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - last-resort process boundary
        print(f"unexpected process boundary failure: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
