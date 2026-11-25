"""LoRA fine-tune of the reflex base on the exported SFT dataset -- runs in tools/train/.venv, never the core.

    .venv\\Scripts\\python tools\\train\\train.py --name round1                 # 2 epochs, the plan's defaults
    .venv\\Scripts\\python tools\\train\\train.py --name feas --max-steps 20     # feasibility: does it fit, how fast
    .venv\\Scripts\\python tools\\train\\train.py --base Qwen/Qwen3.5-4B --quant nf4 --rank 8 --max-steps 20 --name feas-4b

Defaults are BUILD-PLAN 5.2: base Qwen/Qwen3.5-0.8B (pinned to the revision setup.py downloaded),
seq <= 1024, r=16, gradient checkpointing, bf16 when the XPU supports it else fp32, batch 1 x
grad-accum 8, 2 epochs, restart from base every round (there is no --resume on purpose: ReST-EM).

The input of every example is the exact message list export.py rebuilt (the prompt the agent sends);
the target is the action JSON. Loss is computed on the target tokens only, times the example's
Step-RFT `weight`. An example longer than --seq loses its oldest history exchanges first (the shape
agent._fit gives a trimmed conversation, with the same note), and is skipped -- counted, never
truncated mid-JSON -- when system + task + target alone do not fit.

Unsloth is used when setup.py managed to install its Intel path and --engine is auto/unsloth; the
plain peft path is the fallback and is what this file was written against.

Outputs, under tools/train/adapters/<name>/:
  adapter/                  the peft adapter (adapter_model.safetensors + adapter_config.json)
  adapter_manifest.json     base id + revision, dataset sha256, hyperparameters, seed, drock git HEAD,
                            torch/peft/transformers versions, adapter sha256, token-length stats,
                            loss curve, device and dtype, wall time
  train.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DROCK = HERE.parent.parent
DATA = HERE / "data"
ADAPTERS = HERE / "adapters"
FIT_NOTE = "[{n} earlier messages were dropped to fit the context window. Continue the TASK above.]"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(path: Path, msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


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


def base_default() -> tuple[str, str | None]:
    """(repo, revision) setup.py recorded, else the plan's model unpinned."""
    try:
        rep = json.loads((HERE / "setup-report.json").read_text(encoding="utf-8"))
        bm = rep.get("base_model") or {}
        if bm.get("repo"):
            return bm["repo"], bm.get("revision")
    except (OSError, ValueError):
        pass
    return "Qwen/Qwen3.5-0.8B", None


# ------------------------------------------------------------------ tokenisation

def render_prompt(tok, messages: list[dict]) -> str:
    """The chat template with the generation prompt, thinking off (DRock sends think=False to Ollama)."""
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def fit_messages(tok, messages: list[dict], target: str, seq: int) -> tuple[list[int], list[int], int] | None:
    """(input_ids, labels, dropped) within `seq` tokens, dropping the oldest exchanges after the TASK first.
    None when even system + task + target do not fit."""
    eos = tok.eos_token or ""
    head, tail = messages[:2], list(messages[2:])
    dropped = 0
    # the hint (a trailing user message with no assistant before it) belongs to the current call; keep it last
    while True:
        # note last, as agent._fit places it (2026-11-03: in the middle it cost a re-read of the
        # whole kept history every step), so a trained prompt has the same shape as a live one
        msgs = head + tail + ([{"role": "user", "content": FIT_NOTE.format(n=dropped)}] if dropped else [])
        prompt = render_prompt(tok, msgs)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target + eos, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) <= seq:
            return p_ids + t_ids, [-100] * len(p_ids) + t_ids, dropped
        # drop the oldest assistant/user pair of history
        idx = next((i for i, m in enumerate(tail) if m["role"] == "assistant"), None)
        if idx is None:
            return None
        del tail[idx:idx + 2]
        dropped += 2


def load_examples(path: Path, tok, seq: int, *, no_synthetic: bool, no_macro: bool, log_path: Path) -> tuple[list[dict], dict]:
    rows = read_jsonl(path)
    out, stats = [], {"rows": len(rows), "skipped_weight0": 0, "skipped_synthetic": 0, "skipped_macro": 0,
                      "skipped_too_long": 0, "trimmed": 0, "lengths": []}
    for ex in rows:
        if not ex.get("weight"):
            stats["skipped_weight0"] += 1
            continue
        if no_synthetic and ex.get("synthetic") and ex.get("source") != "macro":
            stats["skipped_synthetic"] += 1
            continue
        if no_macro and (ex.get("source") == "macro" or ex.get("source", "").endswith(":macro")):
            stats["skipped_macro"] += 1
            continue
        fit = fit_messages(tok, ex["messages"], ex["target"], seq)
        if fit is None:
            stats["skipped_too_long"] += 1
            continue
        ids, labels, dropped = fit
        stats["trimmed"] += bool(dropped)
        stats["lengths"].append(len(ids))
        out.append({"input_ids": ids, "labels": labels, "weight": float(ex["weight"]), "action": ex["action"], "id": ex["id"]})
    L = sorted(stats.pop("lengths"))
    stats["kept"] = len(out)
    stats["tokens"] = {"min": L[0] if L else 0, "median": L[len(L) // 2] if L else 0, "p90": L[int(len(L) * 0.9)] if L else 0,
                       "max": L[-1] if L else 0}
    log(log_path, f"{path.name}: {json.dumps(stats)}")
    return out, stats


# ------------------------------------------------------------------ model

def pick_device(want: str):
    import torch
    if want == "auto":
        want = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return want


def pick_dtype(device: str, want: str):
    import torch
    if want == "bf16":
        return torch.bfloat16
    if want == "fp32":
        return torch.float32
    if device == "xpu":
        try:
            if torch.xpu.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float32


def load_model(base: str, revision: str | None, device: str, dtype, quant: str | None, log_path: Path):
    """Tokenizer + causal LM. Qwen3.5 checkpoints are multimodal: AutoModelForCausalLM is tried first
    (the text stack alone), then the image-text class whose language model carries the same modules."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"revision": revision, "dtype": dtype}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": device}
    try:
        model = AutoModelForCausalLM.from_pretrained(base, **kw)
        kind = "AutoModelForCausalLM"
    except (ValueError, KeyError, OSError) as e:
        log(log_path, f"AutoModelForCausalLM failed ({type(e).__name__}: {str(e)[:200]}); trying the image-text class")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(base, **kw)
        kind = "AutoModelForImageTextToText"
    if quant != "nf4":
        model.to(device)
    model.config.use_cache = False
    return tok, model, kind


def add_lora(model, rank: int, alpha: int, dropout: float, quant: str | None):
    from peft import LoraConfig, get_peft_model
    if quant == "nf4":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=TARGET_MODULES)
    return get_peft_model(model, cfg)


def load_unsloth(base: str, revision: str | None, seq: int, dtype, rank: int, alpha: int, quant: str | None):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(model_name=base, revision=revision, max_seq_length=seq, dtype=dtype,
                                                   load_in_4bit=(quant == "nf4"))
    model = FastLanguageModel.get_peft_model(model, r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                                             target_modules=TARGET_MODULES, use_gradient_checkpointing="unsloth")
    return tok, model


# ------------------------------------------------------------------ loop

def batches(examples: list[dict], rng: random.Random, shuffle: bool):
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for i in order:
        yield examples[i]


def step_loss(model, ex: dict, device: str):
    """Mean cross-entropy over the target tokens, times the Step-RFT weight: the same number as
    model(labels=...).loss, but logits are made only at the positions that predict a target token.
    A ~8k-token agent prompt with Qwen3's 151,936-token vocabulary is ~4.9 GB of fp32 logits (and
    as much again for their gradient) for a loss that reads the last ~50 rows; the A380 has 6 GB."""
    import torch
    import torch.nn.functional as F
    labels = ex["labels"]
    pos = [i for i in range(len(labels) - 1) if labels[i + 1] != -100]
    ids = torch.tensor([ex["input_ids"]], device=device)
    logits = model(input_ids=ids, logits_to_keep=torch.tensor(pos, device=device)).logits[0]
    target = torch.tensor([labels[i + 1] for i in pos], device=device)
    return F.cross_entropy(logits.float(), target) * ex["weight"]


def evaluate(model, val: list[dict], device: str, dtype, limit: int | None = None) -> dict:
    import torch
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ex in val[:limit] if limit else val:
            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                total += float(step_loss(model, ex, device))
            n += 1
    model.train()
    return {"loss": round(total / n, 4) if n else None, "examples": n}


def main(argv=None) -> int:
    repo_default, rev_default = base_default()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="adapter name (tools/train/adapters/<name>)")
    ap.add_argument("--base", default=repo_default)
    ap.add_argument("--revision", default=rev_default)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; 0 = run the epochs")
    ap.add_argument("--batch", type=int, default=1, help="sequences per forward (only 1 is implemented: no padding)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "xpu", "cpu"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    ap.add_argument("--quant", default=None, choices=[None, "nf4"], help="nf4 = QLoRA via bitsandbytes (the 4B feasibility test)")
    ap.add_argument("--engine", default="auto", choices=["auto", "peft", "unsloth"])
    ap.add_argument("--no-synthetic", action="store_true", help="drop the synthesised COMPLETE turns")
    ap.add_argument("--no-macro", action="store_true", help="drop the macro teacher examples")
    ap.add_argument("--eval-every", type=int, default=0, help="optimizer steps between val evaluations (0 = per epoch)")
    ap.add_argument("--eval-limit", type=int, default=200)
    args = ap.parse_args(argv)
    if args.batch != 1:
        raise SystemExit("--batch 1 only: sequences are unpadded (grad-accum is the batch)")
    # Before the model loads: nothing in the frozen held-out split may be trained on, however the files were made.
    import heldout
    heldout.refuse(read_jsonl(Path(args.data) / "train.jsonl") + read_jsonl(Path(args.data) / "val.jsonl"), heldout.load())

    out_dir = ADAPTERS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    t_start = time.time()

    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    log(log_path, f"device={device} dtype={dtype} base={args.base}@{args.revision} seq={args.seq} r={args.rank}")

    engine = args.engine
    if engine == "auto":
        try:
            import unsloth  # noqa: F401
            engine = "unsloth"
        except Exception:
            engine = "peft"
    if engine == "unsloth":
        try:
            tok, model = load_unsloth(args.base, args.revision, args.seq, dtype, args.rank, args.alpha, args.quant)
            kind = "unsloth.FastLanguageModel"
        except Exception as e:
            log(log_path, f"unsloth failed ({type(e).__name__}: {str(e)[:300]}); falling back to peft")
            engine = "peft"
    if engine == "peft":
        tok, model, kind = load_model(args.base, args.revision, device, dtype, args.quant, log_path)
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = add_lora(model, args.rank, args.alpha, args.dropout, args.quant)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(log_path, f"engine={engine} class={kind} trainable={trainable:,} / {total:,}")

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    train, tstats = load_examples(data / "train.jsonl", tok, args.seq, no_synthetic=args.no_synthetic, no_macro=args.no_macro, log_path=log_path)
    val, vstats = load_examples(data / "val.jsonl", tok, args.seq, no_synthetic=args.no_synthetic, no_macro=args.no_macro, log_path=log_path)
    if not train:
        raise SystemExit("no training examples fit --seq; raise it or export with fewer history turns")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(train) / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warm = max(1, int(total_steps * args.warmup))

    def lr_at(step: int) -> float:
        if step < warm:
            return args.lr * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    rng = random.Random(args.seed)
    curve, evals = [], []
    step = micro = 0
    running = 0.0
    done = False
    peak = None
    for epoch in range(args.epochs if not args.max_steps else 10 ** 6):
        for ex in batches(train, rng, shuffle=True):
            with torch.autocast(device_type=device, dtype=dtype, enabled=(dtype != torch.float32)):
                loss = step_loss(model, ex, device) / args.grad_accum
            loss.backward()
            running += float(loss)
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            curve.append({"step": step, "loss": round(running, 4), "lr": lr_at(step - 1), "t": round(time.time() - t_start, 1)})
            if step % 5 == 0 or step == 1:
                mem = ""
                if device == "xpu":
                    try:
                        peak = int(torch.xpu.max_memory_allocated())
                        mem = f" peak_alloc={peak / 2**30:.2f}GiB"
                    except Exception:
                        pass
                log(log_path, f"step {step}/{total_steps} loss={running:.4f} lr={lr_at(step - 1):.2e}{mem}")
            running = 0.0
            if args.eval_every and step % args.eval_every == 0 and val:
                evals.append({"step": step, **evaluate(model, val, device, dtype, args.eval_limit)})
                log(log_path, f"eval {evals[-1]}")
            if step >= total_steps:
                done = True
                break
        if val and not args.eval_every:
            evals.append({"step": step, "epoch": epoch + 1, **evaluate(model, val, device, dtype, args.eval_limit)})
            log(log_path, f"eval {evals[-1]}")
        if done:
            break

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    weights = next((p for p in adapter_dir.glob("adapter_model.*") if p.suffix in (".safetensors", ".bin")), None)
    import peft, transformers
    am = {
        "v": 1, "name": args.name, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base": {"repo": args.base, "revision": args.revision, "class": kind},
        "dataset": {"sha256": manifest.get("dataset_sha256"), "manifest": str(data / "manifest.json"),
                    "train_sha256": (manifest.get("files") or {}).get("train.jsonl", {}).get("sha256"),
                    "val_sha256": (manifest.get("files") or {}).get("val.jsonl", {}).get("sha256"),
                    "system_prompt_sha256": manifest.get("system_prompt_sha256"), "train": tstats, "val": vstats},
        "hyperparameters": {"seq": args.seq, "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "lr": args.lr,
                            "epochs": args.epochs, "max_steps": args.max_steps, "batch": args.batch, "grad_accum": args.grad_accum,
                            "warmup": args.warmup, "quant": args.quant, "engine": engine, "target_modules": TARGET_MODULES,
                            "no_synthetic": args.no_synthetic, "no_macro": args.no_macro},
        "seed": args.seed, "device": device, "dtype": str(dtype), "peak_alloc_bytes": peak,
        "git": git_head(DROCK),
        "versions": {"torch": torch.__version__, "peft": peft.__version__, "transformers": transformers.__version__,
                     "python": sys.version.split()[0]},
        "adapter": {"path": str(weights) if weights else None, "sha256": sha256_file(weights) if weights else None,
                    "trainable_params": trainable},
        "steps": step, "loss_curve": curve, "evals": evals, "seconds": round(time.time() - t_start, 1),
    }
    (out_dir / "adapter_manifest.json").write_text(json.dumps(am, indent=1), encoding="utf-8")
    log(log_path, f"adapter sha256={am['adapter']['sha256']} -> {out_dir / 'adapter_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
