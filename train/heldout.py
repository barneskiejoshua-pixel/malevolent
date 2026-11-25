"""The frozen held-out split: tasks that never enter training -- standard library only.

    python tools/train/heldout.py --freeze      # writes tools/train/heldout.json once; refuses to overwrite
    python tools/train/heldout.py               # prints what the frozen file holds

Two kinds of id, frozen 2026-11-23 before any training (Malevolent brief, Phase 0):

  contracts   a third of the 60 benchmark contracts, per category (round(n/3) of each, seeded). Every
              benchmark receipt of one of these, and every rewording made from one (a task that names it
              as its base), is evaluation data. Acceptance runs on them.
  goals       20% of the distinct live goals in data/traces.jsonl(.1) at freeze time (cli, chat, ...), by a
              seeded hash of the normalised goal -- the same key export.task_key computes.

export.py drops every example these name, and train.py calls refuse() on what it loads, so a dataset
built some other way still cannot train on them. A goal first seen after the freeze is not in the file;
it can be held out by a later, separately named freeze, never by editing this one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DROCK = HERE.parent.parent
FILE = HERE / "heldout.json"
SEED = 20261123
GOAL_PCT = 20
LIVE_SOURCES = ("cli", "chat", "trial", "capbench", "reflexbench", "bench")


def task_key(goal: str) -> str:
    """Same formula as export.task_key (a test holds them equal)."""
    return hashlib.sha256(" ".join(goal.lower().split()).encode("utf-8")).hexdigest()[:16]


def pick_contracts(tasks: list[dict], seed: int = SEED) -> list[str]:
    by_cat: dict[str, list[str]] = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t["id"])
    rng = random.Random(seed)
    out = []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        out += rng.sample(ids, round(len(ids) / 3))
    return sorted(out)


def pick_goals(goals: list[str], seed: int = SEED, pct: int = GOAL_PCT) -> list[str]:
    keys = {task_key(g) for g in goals if g and g.strip()}
    return sorted(k for k in keys if int(hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()[:8], 16) % 100 < pct)


def live_goals(trace_files: list[Path]) -> list[str]:
    out = []
    for p in trace_files:
        try:
            f = open(p, encoding="utf-8")
        except OSError:
            continue
        with f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("source") in LIVE_SOURCES and r.get("goal"):
                    out.append(str(r["goal"]))
    return out


def freeze(path: Path = FILE, *, tasks: list[dict], goals: list[str], when: str) -> dict:
    if path.exists():
        raise FileExistsError(f"{path} is frozen; a new split needs a new file name, never an edit of this one")
    data = {"v": 1, "frozen": when, "seed": SEED, "goal_pct": GOAL_PCT,
            "contracts": pick_contracts(tasks), "goals": pick_goals(goals)}
    data["sha256"] = hashlib.sha256(json.dumps({k: data[k] for k in ("contracts", "goals")}, sort_keys=True).encode()).hexdigest()
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


def load(path: Path = FILE) -> dict:
    """The frozen split. Missing file = nothing may be trained yet (Phase 0 comes first)."""
    if not path.exists():
        raise SystemExit(f"no held-out split at {path}: run `python tools/train/heldout.py --freeze` before any training")
    return json.loads(path.read_text(encoding="utf-8"))


def is_heldout(ex: dict, split: dict) -> bool:
    """An export example (or a receipt/trace row) that belongs to the held-out split."""
    contracts, goals = set(split["contracts"]), set(split["goals"])
    if ex.get("contract") in contracts or ex.get("task_id") in contracts or ex.get("base") in contracts:
        return True
    key = ex.get("task") or (task_key(ex["goal"]) if ex.get("goal") else None)
    return key in goals


def refuse(examples: list[dict], split: dict) -> None:
    bad = [ex.get("id") or ex.get("task") for ex in examples if is_heldout(ex, split)]
    if bad:
        raise SystemExit(f"refusing to train: {len(bad)} example(s) are in the held-out split "
                         f"(first: {bad[:3]}); re-export with tools/train/export.py")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)
    if args.freeze:
        sys.path.insert(0, str(DROCK.parent))
        import time
        from drock.benchmark import contracts
        data = freeze(tasks=contracts.TASKS, goals=live_goals([DROCK / "data" / "traces.jsonl.1", DROCK / "data" / "traces.jsonl"]),
                      when=time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        data = load()
    print(json.dumps({"frozen": data["frozen"], "contracts": data["contracts"], "goals": len(data["goals"]),
                      "sha256": data["sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
