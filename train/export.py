"""Build the Phase 5 SFT dataset from verified traces -- standard library only, runs on the core python.

    python tools/train/export.py                      # data/traces.jsonl + benchmark/runs/*/receipts.jsonl
    python tools/train/export.py --out tools/train/data --val-pct 20 --keep-masked

THE LABEL IS THE EXECUTOR'S, NEVER THE MODEL'S. A task enters the dataset only if it is PASS-gated:

  * a benchmark task whose receipt (benchmark/runs/<run>/receipts.jsonl) says `result: PASS` -- the
    independent checker read the artefact; the model's claim played no part -- joined to its full
    row in data/traces.jsonl (same goal, recorded inside the receipt's time window), because the
    receipt's own trace is compacted (arguments cut at 200 chars) and cannot be trained on;
  * a live task (source cli/chat/...) with `ok` true, `unverified` false and `unmet` empty -- the
    completion gate's verdict recomputed at record time (trace.row).

Within a task, Step-RFT masking (JetBrains 2026): a step whose result the executor did not mark
`ok` AND `verified` gets `weight: 0` -- it stays in the CONTEXT of later steps (the model saw the
failure and its error, which is why the next step is right) but is never a TARGET. By default those
rows are dropped from the files; `--keep-masked` keeps them with weight 0 so a trainer can decide.
The live trace stores the same 0/1 plus `ink` (blot, payoff, doodle) from house.py.
This file still admits only PASS-gated tasks, and it does not train anything.

THE INPUT IS THE PROMPT THE MODEL SAW, AS CLOSELY AS THE TRACE ALLOWS. It is rebuilt with agent.py's
own functions (system_prompt, the exclusion rules, _fit, the RESULT feedback line, untrusted's
scrub and note, the gate's steer). What the trace does NOT keep, and what is done about it:

  gap                                      closest form used
  ---------------------------------------  ---------------------------------------------------------
  SYSTEM STATE at the time (free RAM/disk) one snapshot of this machine taken at export, cwd set to
                                           the task's work dir when known (`--state-json` overrides)
  KNOWN FACTS at the time                  "{}" (the benchmark's memory is empty; live facts unknown)
  the policy (legal exclusions)            benchmark: Policy("ask", gui=False) as run.py builds it;
                                           live: the current policy.json
  action descriptions at the time          today's schema.ACTIONS (descriptions are the evolve gene
                                           and have changed since older traces; manifest records the
                                           system-prompt hash so a drift is visible)
  the model's raw JSON text                json.dumps of the parsed action (same keys, same order;
                                           whitespace may differ)
  turns not in the trace (invalid JSON,    absent: the conversation is rebuilt as if the executed
  guard refusals, blocked repeats,         steps were consecutive
  rejected COMPLETEs, recovery restarts)
  the gate's steer after a verified step   "Not done yet: <unmet>" recomputed by expect.missing on
                                           the trace prefix (what _done_check returned when the gate
                                           still wanted evidence); after a failed step "Still
                                           missing: <unmet>" (agent.steer)
  the lookahead menu / knowledge card      the card is re-derived by knowledge.for_goal (Agent's
                                           default is on); the lookahead menu is off by default
  the final COMPLETE turn                  never traced (the loop returns before appending). For a
                                           PASS-gated task whose last step is verified, a COMPLETE
                                           with the recorded final message as summary is added,
                                           flagged `synthetic: true` (`--no-synthetic` drops it)

Live rows are re-judged by TODAY's gate (expect.missing on the trace) before they enter: the gate
only grows stricter, and a row that passed an older gate is not evidence today. `--no-regate` keeps
the record-time verdict. `--models gemma3:4b` keeps only steps a named model emitted (the plan says
"verified 4B traces"; the default keeps every PASS-gated model and the manifest counts per model).

Teacher examples for the reflex tier: every goal that macros.compile() accepts (the rule-based fast
path) yields its first action as a target from the fresh prompt, flagged source "macro"; steps the
fast path actually ran (tier "macro" in a trace) are exported like model steps. Their `reason` is a
fixed sentence, since a rule has none.

Output: <out>/train.jsonl, <out>/val.jsonl (split by a hash of the goal so the same request never
lands on both sides, whichever run it came from), <out>/manifest.json (sha256 of both files, counts
per action class, per source, masked/synthetic counts, git HEAD, system-prompt hash).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DROCK = HERE.parent.parent          # the drock package directory
sys.path.insert(0, str(DROCK.parent))
os.environ.setdefault("DROCK_NO_TRACE", "1")   # nothing here may write the trajectory log

from drock import agent, expect, macros, paths, untrusted  # noqa: E402
from drock.schema import ACTIONS, LOW  # noqa: E402

MACRO_REASON = "The request maps directly to this action."
COMPLETE_REASON = "Every requested step has a verified result; the task is complete."
_NOT_A_COMPLETE = ("(verified memory)", "model did not emit COMPLETE", "not completed:")
LIVE_SOURCES_DEFAULT = ("cli", "chat", "trial", "capbench", "reflexbench", "bench")


# ------------------------------------------------------------------ inputs

def read_jsonl(p: Path) -> list[dict]:
    out = []
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    out.append(d)
    except OSError:
        pass
    return out


def load_receipts(runs_dir: Path) -> dict[str, list[dict]]:
    """run name -> receipts, for every runs/<name>/receipts.jsonl."""
    out = {}
    for p in sorted(runs_dir.glob("*/receipts.jsonl")):
        rows = read_jsonl(p)
        if rows:
            out[p.parent.name] = rows
    return out


def join_receipts(receipts: dict[str, list[dict]], traces: list[dict], slack: float = 120.0) -> tuple[list[tuple[str, dict, dict]], dict]:
    """PASS receipts joined to their full trace row: same goal, recorded within the receipt's window.
    Returns [(run, receipt, trace_row)] and the join statistics."""
    by_goal: dict[str, list[dict]] = {}
    for r in traces:
        if r.get("source") == "benchmark":
            by_goal.setdefault(r.get("goal", ""), []).append(r)
    joined, stats = [], Counter()
    used: set[int] = set()
    for run, rows in receipts.items():
        for rec in rows:
            if rec.get("result") != "PASS":
                stats["receipts_not_pass"] += 1
                continue
            stats["receipts_pass"] += 1
            cands = [t for t in by_goal.get(rec.get("request", ""), [])
                     if id(t) not in used and rec.get("started", 0) - slack <= t.get("at", 0) <= rec.get("completed", 0) + slack]
            if not cands:
                stats["pass_without_full_trace"] += 1
                continue
            t = min(cands, key=lambda t: abs(t.get("at", 0) - rec.get("completed", 0)))
            used.add(id(t))
            joined.append((run, rec, t))
    return joined, dict(stats)


# ------------------------------------------------------------------ the prompt, rebuilt

def _policy_for(source: str, work: str | None):
    from drock.policy import Policy
    if source == "benchmark":
        return Policy("ask", confirm=lambda *a, **k: (False, False), root=work, gui=False)
    return Policy("ask")


def task_message(goal: str, state: dict, policy, knowledge: bool = True) -> str:
    """The user message agent.run builds in fresh(): KNOWN FACTS, exclusions, the card, the state, the TASK."""
    legal = agent.legal_exclusions(policy)
    read_only = agent.read_only_exclusions(goal)
    unavailable = f"\nUNAVAILABLE THIS SESSION (not offered): {', '.join(sorted(legal))}" if legal else ""
    if read_only:
        unavailable += "\nREAD-ONLY GOAL: this task only looks things up; file, app and package changes are not offered"
    if knowledge:
        try:
            from drock import knowledge as kn
            card = kn.for_goal(goal)
            if card:
                unavailable += "\n" + kn.render(card)
        except Exception:
            pass
    return "KNOWN FACTS: {}" + unavailable + f"\nSYSTEM STATE: {json.dumps(state)}" + f"\n\nTASK: {goal}"


def result_message(step: dict) -> str:
    """The RESULT line the loop appends after an executed step (agent.run, execute + verify)."""
    name = step["action"].get("action", "")
    brief = step.get("result") or {}
    shown = untrusted.clean_result(name, brief)[0] if name in untrusted.FOREIGN else brief
    return "RESULT: " + json.dumps(shown, ensure_ascii=False) + (f"\n{untrusted.note(name)}" if name in untrusted.FOREIGN else "")


def _last_verified(step: dict) -> bool:
    r = step.get("result") or {}
    return bool(r.get("ok")) and (bool(r.get("verified")) or agent.risk_of(step["action"]) == LOW)


def steer_after(goal: str, prefix: list[dict], root: str | None) -> str | None:
    """The one-call hint sent with the NEXT model call after the last step of `prefix`."""
    if not prefix:
        return None
    try:
        unmet = expect.missing(goal, prefix, root=root)
    except Exception:
        unmet = []
    if _last_verified(prefix[-1]):
        # _done_check ran: with unmet evidence it returned the joined list, and the loop steered with it
        return ("Not done yet: " + "; ".join(unmet[:3])) if unmet else None
    return ("Still missing: " + "; ".join(unmet[:3])) if unmet else None


def target_text(action: dict) -> str:
    return json.dumps(action, ensure_ascii=False)


def conversation(goal: str, trace: list[dict], upto: int, task_msg: str, tier: str, root: str | None) -> list[dict]:
    """messages the model was sent for the step at index `upto` (exclusive prefix), _fit applied, the steer last."""
    prefix = trace[:upto]
    msgs = [{"role": "system", "content": agent.system_prompt()}, {"role": "user", "content": task_msg}]
    for s in prefix:
        msgs.append({"role": "assistant", "content": target_text(s["action"])})
        msgs.append({"role": "user", "content": result_message(s)})
    msgs = agent._fit(msgs, agent._ctx_tokens(tier), prefix, reserve=300 + agent.max_tokens_for(goal))
    hint = steer_after(goal, prefix, root)
    if hint:
        msgs.append({"role": "user", "content": hint})
    return msgs


# ------------------------------------------------------------------ examples

def task_key(goal: str) -> str:
    return hashlib.sha256(" ".join(goal.lower().split()).encode("utf-8")).hexdigest()[:16]


def step_weight(step: dict) -> int:
    r = step.get("result") or {}
    return 1 if r.get("ok") and r.get("verified") else 0


def task_examples(goal: str, trace: list[dict], *, source: str, tier: str, model: str | None, state: dict,
                  policy, work: str | None, final_message: str | None, ok: bool, synthetic: bool = True,
                  knowledge: bool = True) -> list[dict]:
    key = task_key(goal)
    st = dict(state)
    if work:
        st["cwd"] = work
    task_msg = task_message(goal, st, policy, knowledge=knowledge)
    out = []
    for i, step in enumerate(trace):
        action = step.get("action") or {}
        if not isinstance(action, dict) or action.get("action") not in ACTIONS:
            continue
        tier_i = step.get("tier") or tier
        out.append({
            "id": f"{key}-{i + 1}", "task": key, "goal": goal, "source": source if tier_i != "macro" else f"{source}:macro",
            "step": i + 1, "tier": tier_i, "model": model if tier_i != "macro" else None,
            "messages": conversation(goal, trace, i, task_msg, tier, work),
            "target": target_text(action if tier_i != "macro" else {"reason": MACRO_REASON, **{k: v for k, v in action.items() if k != "reason"}}),
            "action": action["action"], "weight": step_weight(step),
            **({"ink": step["ink"]} if step.get("ink") in ("blot", "payoff", "doodle") else {}),
            "verified": bool((step.get("result") or {}).get("verified")), "synthetic": False,
        })
    # The COMPLETE turn is never traced; a PASS-gated task that ended verified gets one, flagged.
    if synthetic and ok and trace and step_weight(trace[-1]) and final_message \
            and not any(m in final_message for m in _NOT_A_COMPLETE) \
            and any(s.get("tier") != "macro" for s in trace):
        complete = {"reason": COMPLETE_REASON, "action": "COMPLETE", "summary": str(final_message)[:1500]}
        out.append({
            "id": f"{key}-{len(trace) + 1}", "task": key, "goal": goal, "source": source, "step": len(trace) + 1,
            "tier": tier, "model": model, "messages": conversation(goal, trace, len(trace), task_msg, tier, work),
            "target": target_text(complete), "action": "COMPLETE", "weight": 1, "verified": True, "synthetic": True,
        })
    return out


def macro_examples(goals: list[str], state: dict, policy, knowledge: bool = True) -> list[dict]:
    """The fast path's first action for every goal it compiles, from the fresh prompt."""
    out, seen = [], set()
    for goal in goals:
        key = task_key(goal)
        if key in seen:
            continue
        seen.add(key)
        try:
            plan = macros.compile(goal)
        except Exception:
            plan = None
        if not plan:
            continue
        action, _desc = plan[0]
        target = {"reason": MACRO_REASON, **{k: v for k, v in action.items() if k != "reason"}}
        out.append({
            "id": f"{key}-m1", "task": key, "goal": goal, "source": "macro", "step": 1, "tier": "macro", "model": None,
            "messages": [{"role": "system", "content": agent.system_prompt()},
                         {"role": "user", "content": task_message(goal, state, policy, knowledge=knowledge)}],
            "target": target_text(target), "action": action["action"], "weight": 1, "verified": True, "synthetic": True,
            "macro_steps": len(plan),
        })
    return out


# ------------------------------------------------------------------ split + write

def split(examples: list[dict], val_pct: int = 20) -> tuple[list[dict], list[dict]]:
    """Deterministic by task hash: every step of a task, and every run of the same goal, on one side."""
    train, val = [], []
    for ex in examples:
        bucket = int(hashlib.sha256(ex["task"].encode()).hexdigest()[:8], 16) % 100
        (val if bucket < val_pct else train).append(ex)
    return train, val


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root: Path) -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(root), capture_output=True, text=True, timeout=20).stdout.strip())
        return {"head": head or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"head": None, "dirty": None}


def write_split(out_dir: Path, train: list[dict], val: list[dict], stats: dict, opts: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        p = out_dir / name
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for ex in rows:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        files[name] = {"sha256": sha256_file(p), "rows": len(rows), "bytes": p.stat().st_size}

    def per_class(rows):
        return dict(sorted(Counter(ex["action"] for ex in rows if ex["weight"]).items()))

    manifest = {
        "v": 1, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "files": files,
        "dataset_sha256": hashlib.sha256((files["train.jsonl"]["sha256"] + files["val.jsonl"]["sha256"]).encode()).hexdigest(),
        "counts": {
            "train_per_action": per_class(train), "val_per_action": per_class(val),
            "train_per_model": dict(sorted(Counter(str(ex.get("model")) for ex in train).items())),
            "val_per_model": dict(sorted(Counter(str(ex.get("model")) for ex in val).items())),
            "train_per_source": dict(sorted(Counter(ex["source"] for ex in train).items())),
            "val_per_source": dict(sorted(Counter(ex["source"] for ex in val).items())),
            "train_tasks": len({ex["task"] for ex in train}), "val_tasks": len({ex["task"] for ex in val}),
            "masked_weight0": sum(1 for ex in train + val if not ex["weight"]),
            "synthetic": sum(1 for ex in train + val if ex.get("synthetic")),
            "max_prompt_chars": max((sum(len(m["content"]) for m in ex["messages"]) for ex in train + val), default=0),
        },
        "join": stats, "options": opts, "git": git_head(DROCK),
        "system_prompt_sha256": hashlib.sha256(agent.system_prompt().encode("utf-8")).hexdigest(),
        "actions": sorted(ACTIONS),
        "labels": "executor verdicts only: benchmark receipt PASS, or live ok & !unverified & unmet==[]; "
                  "step weight 1 iff result.ok and result.verified (Step-RFT masking)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


# ------------------------------------------------------------------ main

def tier_models() -> dict[str, str]:
    try:
        cfg = json.loads(paths.models_json().read_text(encoding="utf-8"))
        return {t: str(v.get("model")) for t, v in cfg.items() if isinstance(v, dict) and v.get("model")}
    except (OSError, ValueError):
        return {}


def build(traces: list[dict], receipts: dict[str, list[dict]], *, state: dict, live_sources=LIVE_SOURCES_DEFAULT,
          keep_masked: bool = False, synthetic: bool = True, macro_teacher: bool = True, knowledge: bool = True,
          regate: bool = True, models: tuple[str, ...] = (), split: dict | None = None) -> tuple[list[dict], dict]:
    examples: list[dict] = []
    stats: dict = {}
    # heldout.json (tools/train/heldout.py): its contracts and goals are evaluation data, never examples
    held_contracts = set((split or {}).get("contracts") or ())
    held_goals = set((split or {}).get("goals") or ())
    held_requests: set[str] = set()
    dropped = Counter()
    joined, stats["receipts"] = join_receipts(receipts, traces)
    for run, rec, row in joined:
        if rec.get("task_id") in held_contracts or task_key(row["goal"]) in held_goals:
            # a held live goal that is also a training contract's request: held out wins (the stricter side)
            dropped["benchmark"] += 1
            held_requests.add(rec.get("request", ""))
            continue
        work = None
        # the receipt's work dir, if the run still has it (paths in the prompt were under it)
        for s in row.get("trace") or []:
            o = ((s.get("result") or {}).get("output"))
            if isinstance(o, dict) and isinstance(o.get("path"), str) and "\\work\\" in o["path"]:
                work = o["path"].split("\\work\\")[0] + "\\work"
                break
        made = task_examples(row["goal"], row.get("trace") or [], source=f"benchmark:{run}", tier=row.get("tier") or "mid",
                             model=rec.get("model"), state=state, policy=_policy_for("benchmark", work), work=work,
                             final_message=row.get("message"), ok=True, synthetic=synthetic, knowledge=knowledge)
        for ex in made:
            ex["contract"] = rec.get("task_id")
        examples += made
    for rows in receipts.values():   # a held contract's request in any run, PASS or not: no macro example either
        held_requests.update(r.get("request", "") for r in rows if r.get("task_id") in held_contracts)
    live_pass = live_seen = live_regated_out = 0
    live_policy = _policy_for("live", None)
    live_models = tier_models()  # tier -> model name from models.json today (the trace keeps only the tier)
    for row in traces:
        if row.get("source") not in live_sources:
            continue
        live_seen += 1
        if not (row.get("ok") and not row.get("unverified") and not row.get("unmet")):
            continue
        live_pass += 1
        if task_key(row["goal"]) in held_goals or row["goal"] in held_requests:
            dropped["live"] += 1
            continue
        if regate:
            # The gate only grows stricter: a row that passed the gate of its day is re-judged by today's.
            # Without the task's files on disk a content check can only fail, so this is conservative.
            try:
                if expect.missing(row["goal"], row.get("trace") or []):
                    live_regated_out += 1
                    continue
            except Exception:
                live_regated_out += 1
                continue
        examples += task_examples(row["goal"], row.get("trace") or [], source=f"live:{row.get('source')}", tier=row.get("tier") or "mid",
                                  model=live_models.get(row.get("tier") or "mid"), state=state, policy=live_policy, work=None,
                                  final_message=row.get("message"), ok=True, synthetic=synthetic, knowledge=knowledge)
    stats["live"] = {"rows": live_seen, "pass_gated": live_pass, "dropped_by_todays_gate": live_regated_out}
    if models:
        want = {m.lower() for m in models}
        before = len(examples)
        examples = [ex for ex in examples if ex["tier"] == "macro" or (ex.get("model") or "").lower() in want
                    or ex["source"] == "macro"]
        stats["dropped_by_model_filter"] = before - len(examples)
    if macro_teacher:
        goals = [r.get("goal", "") for r in traces if r.get("goal")]
        for rows in receipts.values():
            goals += [r.get("request", "") for r in rows if r.get("request")]
        ms = macro_examples(goals, state, live_policy, knowledge=knowledge)
        stats["macro"] = {"goals_seen": len(set(goals)), "compiled": len(ms)}
        kept = [m for m in ms if m["task"] not in held_goals and m["goal"] not in held_requests]
        dropped["macro"] += len(ms) - len(kept)
        examples += kept
    stats["heldout_dropped"] = {k: dropped[k] for k in ("benchmark", "live", "macro")}
    stats["steps_total"] = len(examples)
    stats["steps_masked"] = sum(1 for ex in examples if not ex["weight"])
    if not keep_masked:
        examples = [ex for ex in examples if ex["weight"]]
    # one row per (task, step): the same benchmark request run in several runs keeps every run's steps
    # (different ids because the goal hash is the same but the run differs) -- dedupe exact duplicates only
    seen, unique = set(), []
    for ex in examples:
        sig = hashlib.sha256((ex["target"] + "\x00" + json.dumps(ex["messages"], ensure_ascii=False)).encode("utf-8")).hexdigest()
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(ex)
    stats["duplicates_dropped"] = len(examples) - len(unique)
    return unique, stats


def machine_state(cwd: str | None = None) -> dict:
    from drock import observe
    try:
        st = observe.snapshot()
    except Exception as e:
        st = {"snapshot_error": f"{type(e).__name__}: {e}"[:200]}
    if cwd:
        st["cwd"] = cwd
    return st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default=str(paths.traces()), help="trajectory log (default: data/traces.jsonl)")
    ap.add_argument("--runs", default=str(DROCK / "benchmark" / "runs"), help="benchmark runs directory")
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--val-pct", type=int, default=20)
    ap.add_argument("--keep-masked", action="store_true", help="keep weight-0 steps in the files")
    ap.add_argument("--no-synthetic", action="store_true", help="no synthesised COMPLETE turns")
    ap.add_argument("--no-macro", action="store_true", help="no macro teacher examples")
    ap.add_argument("--no-knowledge", action="store_true", help="do not re-derive the knowledge card in the prompt")
    ap.add_argument("--live-sources", default=",".join(LIVE_SOURCES_DEFAULT))
    ap.add_argument("--state-json", default="", help="a SYSTEM STATE dict to use instead of a fresh snapshot")
    ap.add_argument("--no-regate", action="store_true", help="do not re-judge live rows with today's gate")
    ap.add_argument("--models", default="", help="comma-separated model names to keep (e.g. gemma3:4b); default all")
    args = ap.parse_args(argv)

    traces = read_jsonl(Path(args.traces))
    for extra in (Path(args.traces + ".1"),):  # the rotated generation, if any
        traces = read_jsonl(extra) + traces
    receipts = load_receipts(Path(args.runs))
    state = json.loads(args.state_json) if args.state_json else machine_state()
    import heldout   # tools/train/heldout.py; exits when no split is frozen yet
    held = heldout.load()
    examples, stats = build(traces, receipts, state=state, live_sources=tuple(s for s in args.live_sources.split(",") if s),
                            keep_masked=args.keep_masked, synthetic=not args.no_synthetic, macro_teacher=not args.no_macro,
                            knowledge=not args.no_knowledge, regate=not args.no_regate,
                            models=tuple(m.strip() for m in args.models.split(",") if m.strip()), split=held)
    stats["heldout_sha256"] = held["sha256"]
    heldout.refuse(examples, held)   # belt and braces: what train.py will check
    train, val = split(examples, args.val_pct)
    manifest = write_split(Path(args.out), train, val, stats, {
        "traces": args.traces, "runs": args.runs, "val_pct": args.val_pct, "keep_masked": args.keep_masked,
        "synthetic": not args.no_synthetic, "macro_teacher": not args.no_macro, "knowledge": not args.no_knowledge,
        "regate": not args.no_regate, "models": args.models, "state": state})
    print(json.dumps({k: manifest[k] for k in ("files", "dataset_sha256", "counts", "join", "git", "system_prompt_sha256")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
