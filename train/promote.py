"""The promotion gate for a learned reflex model -- standard library only, runs on the core python.

    python tools/train/promote.py --before runs/A1 runs/A2 --after runs/B1 runs/B2 \
        --manifest tools/train/adapters/round1/adapter_manifest.json [--model drock-reflex-<hash>] [--dry-run]

Refuses to decide on anything less than two BEFORE runs (the current reflex tier) and two AFTER
runs (the candidate) over the same task set, plus the adapter manifest. The rule is BUILD-PLAN's
adoption rule, the one benchmark/paired.py measures with:

    flips  = number of tasks whose result differs between the two BEFORE runs (the noise floor)
    ADOPT  iff  min(after pass) - max(before pass) > flips
           and  max(after false_success) <= min(before false_success)

When benchmark/paired.py exposes `decide` (another session is adding a --decide mode), that is
called and its verdict recorded next to this file's; the two must agree or the gate refuses (a
disagreement is a bug in one of them, not a result). Without it, the rule above is applied here.

On ADOPT (and not --dry-run) the reflex entry of data/models.json gains
    "adapter": "<adapter sha256>", "adapter_manifest": "<path>", "adapter_dataset": "<dataset sha256>"
and, with --model, its "model" becomes the Ollama name export_gguf.py created (models.json is the
only place a tier's model is named; an adapter hash alone changes nothing at run time). Every
decision, ADOPT or REJECT, is appended to tools/train/PROMOTIONS.md with the numbers it was made on.

Nothing here runs a model or a benchmark. The receipts are read as the harness wrote them; the
model's claims (`claim.ok`) count only as false successes, never as passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DROCK = HERE.parent.parent
PROMOTIONS = HERE / "PROMOTIONS.md"


def read_receipts(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "receipts.jsonl"
    if not p.is_file():
        raise SystemExit(f"{run_dir}: no receipts.jsonl")
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("task_id"):
                rows.append(d)
    if not rows:
        raise SystemExit(f"{run_dir}: empty receipts.jsonl")
    return rows


def summary(receipts: list[dict]) -> dict:
    """pass / false_success / tasks for one run (run.py's own definitions)."""
    return {"pass": sum(1 for r in receipts if r.get("result") == "PASS"),
            "false_success": sum(1 for r in receipts if (r.get("failure") or {}).get("false_success")),
            "tasks": len(receipts), "task_ids": sorted(r["task_id"] for r in receipts),
            "results": {r["task_id"]: r.get("result") for r in receipts}}


def flips(a: dict, b: dict) -> int:
    """Tasks whose result differs between two runs of the SAME arm: the paired noise floor."""
    return sum(1 for t in a["results"] if a["results"].get(t) != b["results"].get(t))


def decide(before: list[dict], after: list[dict]) -> dict:
    """The adoption rule on run summaries (>= 2 per arm, identical task sets)."""
    if len(before) < 2 or len(after) < 2:
        raise ValueError("two BEFORE and two AFTER runs are required")
    sets = {tuple(s["task_ids"]) for s in before + after}
    if len(sets) != 1:
        raise ValueError("the runs do not cover the same task set; a comparison across different tasks is not paired")
    floor = max(flips(before[i], before[j]) for i in range(len(before)) for j in range(i + 1, len(before)))
    gain = min(s["pass"] for s in after) - max(s["pass"] for s in before)
    fs_ok = max(s["false_success"] for s in after) <= min(s["false_success"] for s in before)
    adopt = gain > floor and fs_ok
    return {"decision": "ADOPT" if adopt else "REJECT", "gain": gain, "flips": floor, "false_success_ok": fs_ok,
            "before_pass": [s["pass"] for s in before], "after_pass": [s["pass"] for s in after],
            "before_false_success": [s["false_success"] for s in before], "after_false_success": [s["false_success"] for s in after],
            "tasks": before[0]["tasks"],
            "why": ("min(after pass) - max(before pass) = %d > %d flips between the BEFORE runs" % (gain, floor)
                    if gain > floor else "gain %d does not clear the %d-task flip floor" % (gain, floor))
                   + ("" if fs_ok else "; false successes rose"),
            }


def paired_decision(before_rs: list[list[dict]], after_rs: list[list[dict]]) -> dict | None:
    """benchmark/paired.py's own verdict when it has a decide(); None when it does not (yet)."""
    sys.path.insert(0, str(DROCK.parent))
    try:
        from drock.benchmark import paired
    except Exception:
        return None
    fn = getattr(paired, "decide", None)
    if fn is None:
        return None
    try:
        out = fn(before_rs, after_rs)
    except TypeError:
        try:
            out = fn(before=before_rs, after=after_rs)
        except Exception as e:
            return {"error": f"paired.decide signature not understood: {type(e).__name__}: {e}"}
    except Exception as e:
        return {"error": f"paired.decide failed: {type(e).__name__}: {e}"}
    if isinstance(out, dict):
        return out
    if isinstance(out, str):
        return {"decision": out.upper()}
    if isinstance(out, bool):
        return {"decision": "ADOPT" if out else "REJECT"}
    return {"error": f"paired.decide returned {type(out).__name__}"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(path: Path) -> dict:
    am = json.loads(path.read_text(encoding="utf-8"))
    sha = (am.get("adapter") or {}).get("sha256")
    if not sha:
        raise SystemExit(f"{path}: no adapter sha256 in the manifest")
    wp = (am.get("adapter") or {}).get("path")
    if wp and Path(wp).is_file() and sha256_file(Path(wp)) != sha:
        raise SystemExit(f"{path}: the adapter file's sha256 does not match the manifest; refusing")
    return am


def apply(models_json: Path, am: dict, manifest_path: Path, model_name: str | None, tier: str = "reflex") -> dict:
    cfg = json.loads(models_json.read_text(encoding="utf-8"))
    entry = cfg.setdefault(tier, {})
    entry["adapter"] = am["adapter"]["sha256"]
    entry["adapter_manifest"] = str(manifest_path)
    entry["adapter_dataset"] = (am.get("dataset") or {}).get("sha256")
    if model_name:
        entry["model"] = model_name
    models_json.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return entry


def record(md: Path, verdict: dict, am: dict, manifest_path: Path, before: list[Path], after: list[Path],
           applied: dict | None, paired_v: dict | None, dry_run: bool) -> None:
    new = not md.exists()
    with open(md, "a", encoding="utf-8", newline="\n") as f:
        if new:
            f.write("# Promotions\n\nEvery decision of tools/train/promote.py, ADOPT or REJECT, with the numbers it was made on. "
                    "The rule: ADOPT only if min(after pass) - max(before pass) > the flip count between the two "
                    "BEFORE runs, and max(after false_success) <= min(before false_success).\n")
        f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M')} -- {verdict['decision']}{' (dry run)' if dry_run else ''}\n\n")
        f.write(f"- adapter: `{am['adapter']['sha256']}` ({am.get('name')}), manifest `{manifest_path}`\n")
        f.write(f"- base: {am.get('base', {}).get('repo')} @ {am.get('base', {}).get('revision')}; dataset `{(am.get('dataset') or {}).get('sha256')}`\n")
        f.write(f"- before: {', '.join(str(p) for p in before)} -> pass {verdict['before_pass']}, false {verdict['before_false_success']}\n")
        f.write(f"- after:  {', '.join(str(p) for p in after)} -> pass {verdict['after_pass']}, false {verdict['after_false_success']}\n")
        f.write(f"- gain {verdict['gain']} vs flips {verdict['flips']} over {verdict['tasks']} tasks; {verdict['why']}\n")
        if paired_v:
            f.write(f"- benchmark/paired.py decide: {json.dumps(paired_v)}\n")
        if applied:
            f.write(f"- models.json reflex: adapter set" + (f", model -> `{applied.get('model')}`" if applied.get("model") else "") + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--before", nargs="+", required=True, help="two or more run dirs of the current reflex tier")
    ap.add_argument("--after", nargs="+", required=True, help="two or more run dirs of the candidate")
    ap.add_argument("--manifest", required=True, help="the adapter_manifest.json of the candidate")
    ap.add_argument("--model", default="", help="Ollama model name to switch the reflex tier to on ADOPT")
    ap.add_argument("--models-json", default="", help="models.json to edit (default: drock's data/models.json)")
    ap.add_argument("--tier", default="reflex")
    ap.add_argument("--promotions", default=str(PROMOTIONS))
    ap.add_argument("--dry-run", action="store_true", help="decide and record, change nothing")
    args = ap.parse_args(argv)

    if len(args.before) < 2 or len(args.after) < 2:
        print("refused: two BEFORE and two AFTER runs are required (the flip count between the BEFORE runs is the floor)", file=sys.stderr)
        return 2
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"refused: no adapter manifest at {manifest_path}", file=sys.stderr)
        return 2
    am = check_manifest(manifest_path)
    before_rs = [read_receipts(Path(p)) for p in args.before]
    after_rs = [read_receipts(Path(p)) for p in args.after]
    try:
        verdict = decide([summary(r) for r in before_rs], [summary(r) for r in after_rs])
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    paired_v = paired_decision(before_rs, after_rs)
    if paired_v and paired_v.get("decision") and paired_v["decision"] != verdict["decision"]:
        print(f"refused: benchmark/paired.py decides {paired_v['decision']} but the local rule decides {verdict['decision']}: "
              f"{json.dumps(paired_v)}", file=sys.stderr)
        record(Path(args.promotions), {**verdict, "decision": "REFUSED (rules disagree)"}, am, manifest_path,
               [Path(p) for p in args.before], [Path(p) for p in args.after], None, paired_v, args.dry_run)
        return 3

    applied = None
    if verdict["decision"] == "ADOPT" and not args.dry_run:
        sys.path.insert(0, str(DROCK.parent))
        if args.models_json:
            mj = Path(args.models_json)
        else:
            from drock import paths
            mj = paths.models_json()
        applied = apply(mj, am, manifest_path, args.model or None, tier=args.tier)
    record(Path(args.promotions), verdict, am, manifest_path, [Path(p) for p in args.before], [Path(p) for p in args.after],
           applied, paired_v, args.dry_run)
    print(json.dumps({"verdict": verdict, "paired": paired_v, "applied": applied}, indent=1))
    return 0 if verdict["decision"] == "ADOPT" else 1


if __name__ == "__main__":
    sys.exit(main())
