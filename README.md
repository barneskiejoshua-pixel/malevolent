# Malevolent

Malevolent is the name for DRock's own fine-tunes of two Qwen3 checkpoints. DRock is a local, offline agent: a
small model proposes one JSON action at a time, a deterministic executor runs and verifies it, and a completion
gate decides when the task is done.

- Malevolent-0.6B would replace the reflex tier (`qwen3:0.6b` today).
- Malevolent-4B would replace the mid and coder tiers (`qwen3:4b-instruct-2507-q4_K_M` today).

Each one is the Qwen3 base plus a LoRA adapter trained only on steps DRock's executor verified. The model learns
one narrow job: given DRock's system prompt, the task and the history so far, emit the next action as JSON.

## Status (2026-11-24)

**No usable weights exist yet, so none are published here.** This repository is the pipeline and the baseline
it will be judged against.

- The only adapter trained so far is a feasibility run: one optimizer step on 2 examples, CPU, Qwen3-0.6B. It
  proves the pipeline runs end to end at 8,192 tokens and says nothing about quality.
- Training on the GPU (Intel Arc A380) is blocked: PyTorch XPU 2.9.1 crashes in `torch.xpu.device_count` on
  driver 31.0.101.4314. Driver 101.9175 fixes it and needs a reboot.
- The baseline is measured: four benchmark runs of the base 4B, in [noise/NOISE.md](noise/NOISE.md).

Weights will be added here when an adapter passes the adoption rule below. A release will carry the adapter,
its `adapter_manifest.json` and the `promote.py` decision it was adopted on.

## Baseline and the adoption rule

The benchmark is DRock's 60 execution contracts. A standard-library checker inspects the files each run left
behind, so a pass never rests on the agent's own claim.

| Run (base qwen3:4b) | Pass / 60 | False successes |
|---|---|---|
| malev-noise-1 | 41 | 3 |
| malev-noise-2 | 41 | 4 |
| malev-noise-3 | 42 | 3 |
| malev-noise-4 | 38 | 4 |

The worst pair of runs disagrees on 10 contracts. `train/promote.py` adopts a candidate only if
`min(after pass) - max(before pass) > flips` and `max(after false successes) <= min(before false successes)`.
With these runs that means at least 53 of 60 on every AFTER run and no more than 3 false successes. A model's
own claim of success only ever counts against it.

## Pipeline

| File | Runs in | What it does |
|---|---|---|
| `train/export.py` | DRock's core Python | Joins benchmark receipts and live traces into step-level examples. Keeps only tasks the executor passed, gives weight 1 only to steps that were ok and verified (other steps stay in context with weight 0), drops held-out ids and duplicates, writes `train.jsonl`, `val.jsonl` and a manifest with hashes. |
| `train/heldout.py`, `train/heldout.json` | core Python | The held-out split, frozen 2026-11-23 before any training (seed 20261123): 19 of the 60 contracts and 220 hashed live-goal keys. `refuse()` stops training on any example it names. |
| `train/train.py` | torch + peft venv | LoRA fine-tune. Loss is cross-entropy on the target tokens only, times the step weight, with logits computed only at target positions (`logits_to_keep`). Writes the adapter and `adapter_manifest.json`. |
| `train/test_train.py` | torch + peft venv | Checks `step_loss` against `model(labels=...).loss` on a tiny random Qwen3, with and without a perturbed LoRA, plus weight scaling and LoRA gradients. |
| `train/export_gguf.py` | venv + llama.cpp converters | Turns an adapter into something Ollama serves: a GGUF adapter beside the base, or a merged and requantised model. |
| `train/promote.py` | core Python | Reads BEFORE and AFTER benchmark runs, applies the adoption rule, and on ADOPT points the tier at the new model. |
| `train/setup.py` | core Python | Builds the training venv and downloads the base weights. |

`export.py`, `promote.py` and `heldout.py`'s command line import DRock modules (`drock.agent`, `drock.expect`,
`drock.benchmark`, ...) that are not in this repository; they run from `drock/tools/train`. `train.py`,
`test_train.py` and `export_gguf.py` run standalone on a dataset in the format below. Some docstrings and
defaults in `train.py` and `export_gguf.py` date from an earlier Qwen3.5 plan; pass `--base` and `--revision`
explicitly.

### Training row format

One JSON object per line in `train.jsonl` / `val.jsonl`:

```json
{"id": "...", "task": "...", "goal": "...", "source": "benchmark", "step": 1, "tier": "mid",
 "model": "qwen3:4b-instruct-2507-q4_K_M", "messages": [{"role": "system", "content": "..."},
 {"role": "user", "content": "..."}], "target": "{\"action\": \"READ_FILE\", ...}",
 "action": "READ_FILE", "weight": 1, "verified": true, "synthetic": false, "contract": "..."}
```

`messages` is the exact prompt DRock sent; `target` is the verified action JSON. Rows with weight 0 are skipped.

### Settings

LoRA r=16, alpha 32, dropout 0.05 on q, k, v, o, gate, up and down projections (1.7% of the 0.6B's
parameters). AdamW, lr 2e-4, 3% warmup, cosine decay, gradient clip 1.0. Batch 1 sequence x 8 accumulation
steps, 2 epochs, every round restarts from the base. 8,192-token sequences; an over-long example loses its
oldest history first and is skipped, never cut mid-JSON, if the system prompt, task and target alone do not fit.
bf16 on the GPU when supported, else fp32; nf4 QLoRA for the 4B.

```
python train/test_train.py
python train/train.py --name round1 --base Qwen/Qwen3-4B-Instruct-2507 --quant nf4 --seq 8192 --data path/to/data
```

## What is not here

The dataset is not published. It is built from DRock's own traces on one person's PC and holds local paths,
file contents and task text. Model weights trained on it may reproduce some of that, which is one reason weights
wait for an adapter worth releasing.

[docs/Malevolent-rundown.pdf](docs/Malevolent-rundown.pdf) is the full write-up: rules, dataset counts, where
the base fails and which failures training can reach. It was written while run 4 was still going; the final
numbers are the ones above.

## License

Apache License 2.0, see [LICENSE](LICENSE). The Qwen3 bases are Apache 2.0 by the Qwen team, Alibaba Cloud; see
[NOTICE](NOTICE).
