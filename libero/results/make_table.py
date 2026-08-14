#!/usr/bin/env python3
"""Summarize the released PSG-JEPA evaluation log in this directory.

    python results/make_table.py            # overall success rate
    python results/make_table.py --per-task # per-task breakdown

`psgjepa.json` is the closed-loop evaluation of the released seed-3072 checkpoint: 10
LIBERO-Goal tasks x 50 rollouts, eval seed 4242, max_steps 600, action horizon 8. The headline
number is the mean over the 10 task success rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-task", action="store_true")
    args = parser.parse_args()

    run = json.loads((HERE / "psgjepa.json").read_text())
    print(f"\nPSG-JEPA on LIBERO-Goal, seed {run['train_seed']}: "
          f"{run['mean_success_rate'] * 100:.1f}% mean success "
          f"({len(run['tasks'])} tasks x {run['n_eval']} rollouts)")

    if args.per_task:
        print()
        for task in run["tasks"]:
            print(f"  T{task['task_id']}  {task['success_rate'] * 100:>5.1f}%  {task['language']}")


if __name__ == "__main__":
    main()
