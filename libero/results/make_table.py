#!/usr/bin/env python3
"""Summarize the released LIBERO-Goal eval logs in this directory.

    python results/make_table.py            # per-variant success rate
    python results/make_table.py --per-task # add the per-task breakdown

Each JSON is one variant's closed-loop evaluation of the released seed-3072 checkpoint:
10 LIBERO-Goal tasks x 50 rollouts, eval seed 4242, max_steps 600, action horizon 8. The
reported number is the mean over the 10 task success rates.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

VARIANTS = [
    ("lewm", "LeWM"),
    ("lewm_actionidm", "LeWM_ActionIDM"),
    ("dinov2", "DINOv2"),
    ("psgjepa", "PSG-JEPA (ours)"),
]


def load(variant: str) -> dict:
    return json.loads((HERE / f"{variant}.json").read_text())


def render() -> None:
    print("\nLIBERO-Goal, released seed-3072 checkpoints")
    print(f"{'Method':<22}{'Success (%)':>13}")
    print("-" * 35)
    for variant, label in VARIANTS:
        print(f"{label:<22}{load(variant)['mean_success_rate'] * 100:>13.1f}")


def render_per_task() -> None:
    print("\nPer-task success rate (%)")
    print(f"{'Method':<22}" + "".join(f"{'T' + str(i):>6}" for i in range(10)))
    print("-" * (22 + 60))
    for variant, label in VARIANTS:
        tasks = load(variant)["tasks"]
        print(f"{label:<22}" + "".join(f"{t['success_rate'] * 100:>6.1f}" for t in tasks))

    print("\nTasks")
    for task in load("psgjepa")["tasks"]:
        print(f"  T{task['task_id']}  {task['language']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", action="store_true")
    args = parser.parse_args()
    render()
    if args.per_task:
        render_per_task()


if __name__ == "__main__":
    main()
