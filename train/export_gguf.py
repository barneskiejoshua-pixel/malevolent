"""Turn a trained adapter into something Ollama can serve -- runs in tools/train/.venv.

    .venv\\Scripts\\python tools\\train\\export_gguf.py --adapter tools\\train\\adapters\\round1 [--route adapter|merge|both]
    .venv\\Scripts\\python tools\\train\\export_gguf.py --adapter ... --plan        # print the steps, run nothing

Two routes, both written out so the one that works for this architecture can be chosen after a look
at what llama.cpp and Ollama accept for `qwen3_5` (research F7: "Ollama adapter support for qwen3.5
unverified"):

  A. GGUF LoRA adapter (`--route adapter`): llama.cpp's convert_lora_to_gguf.py turns the peft
     adapter into <out>/adapter.gguf. Ollama Modelfile: `FROM <base gguf>` + `ADAPTER <adapter.gguf>`.
     Cheap (megabytes), the base stays the pulled Ollama model, the adapter hash is the artefact.
     Risk: Ollama applies GGUF adapters through llama.cpp's runtime LoRA, which covers the linear
     projections of an architecture only where the graph builder uses build_lora_mm. Qwen3.5's small
     dense checkpoints are a hybrid (gated DeltaNet + gated attention) with a vision tower; the text
     projections in TARGET_MODULES are ordinary matmuls, but whether Ollama's bundled llama.cpp
     honours an adapter on that arch has to be tried once: `ollama create` then one constrained chat
     whose output differs from the base. If it silently ignores the adapter, route B.

  B. Merge + requantise (`--route merge`): peft merge_and_unload() -> a full HF checkpoint ->
     convert_hf_to_gguf.py --outtype q8_0 (python only, no llama-quantize binary: Smart App Control
     blocks unsigned executables on this PC) -> <out>/merged-q8_0.gguf and a Modelfile with
     `FROM merged-q8_0.gguf`. `ollama create --quantize q4_K_M` can requantise further from an f16
     export if 0.9 GB is too much for the reflex tier. Always works; the artefact is the whole model.

Either way the result is hashed: <out>/gguf_manifest.json carries the adapter manifest's hashes,
the sha256 of every GGUF produced, the converter's git commit, and the Modelfile text. The Ollama
model name is `drock-reflex-<first 12 of the adapter sha256>`, so a promoted model names its
provenance. Nothing here calls Ollama's HTTP API; `ollama create` is printed as the next command
(it loads the model on the GPU -- not while a benchmark runs).

llama.cpp's converters are not on PyPI; `--llama-cpp DIR` names a checkout, else one is shallow-
cloned into tools/train/llama.cpp (network; a git commit is recorded). The venv's `gguf` package is
replaced by the checkout's gguf-py on the import path, as the converters expect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA_CPP = HERE / "llama.cpp"
LLAMA_URL = "https://github.com/ggml-org/llama.cpp.git"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, timeout: int = 7200) -> str:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}):\n{(p.stdout or '')[-2000:]}\n{(p.stderr or '')[-3000:]}")
    return (p.stdout or "") + (p.stderr or "")


def ensure_llama_cpp(path: Path | None) -> tuple[Path, str]:
    """(checkout dir, git commit). Clones shallowly when absent."""
    d = path or LLAMA_CPP
    if not (d / "convert_hf_to_gguf.py").is_file():
        if not shutil.which("git"):
            raise SystemExit("git is not on PATH and no --llama-cpp checkout was given")
        run(["git", "clone", "--depth", "1", LLAMA_URL, str(d)])
    commit = ""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(d), capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return d, commit


def converter_env(llama: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(llama / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def base_snapshot(am: dict) -> Path:
    """The local HF snapshot of the base model the adapter was trained on (from setup-report.json)."""
    try:
        rep = json.loads((HERE / "setup-report.json").read_text(encoding="utf-8"))
        bm = rep.get("base_model") or {}
        if bm.get("path") and bm.get("repo") == am["base"]["repo"]:
            return Path(bm["path"])
    except (OSError, ValueError):
        pass
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(am["base"]["repo"], revision=am["base"].get("revision")))


def route_adapter(adapter_dir: Path, base: Path, out: Path, llama: Path) -> dict:
    gguf = out / "adapter.gguf"
    run([sys.executable, str(llama / "convert_lora_to_gguf.py"), str(adapter_dir), "--base", str(base), "--outfile", str(gguf),
         "--outtype", "f16"], env=converter_env(llama))
    return {"adapter_gguf": str(gguf), "adapter_gguf_sha256": sha256_file(gguf), "size_bytes": gguf.stat().st_size}


def route_merge(adapter_dir: Path, base: Path, out: Path, llama: Path, outtype: str) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    merged_dir = out / "merged-hf"
    try:
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(str(base), dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(str(merged_dir))
    for extra in ("chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"):
        if (base / extra).is_file() and not (merged_dir / extra).exists():
            shutil.copy2(base / extra, merged_dir / extra)
    gguf = out / f"merged-{outtype}.gguf"
    run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged_dir), "--outfile", str(gguf), "--outtype", outtype],
        env=converter_env(llama))
    return {"merged_gguf": str(gguf), "merged_gguf_sha256": sha256_file(gguf), "size_bytes": gguf.stat().st_size,
            "merged_hf": str(merged_dir)}


def modelfile(route: str, res: dict, base_ollama: str, ctx: int) -> str:
    lines = []
    if route == "adapter":
        lines += [f"FROM {base_ollama}", f"ADAPTER {res['adapter_gguf']}"]
    else:
        lines += [f"FROM {res['merged_gguf']}"]
    # The agent sends its own system prompt and a JSON schema every call; the Modelfile sets only what
    # models.json cannot: the context the reflex tier is configured for and greedy-ish sampling.
    lines += [f"PARAMETER num_ctx {ctx}", "PARAMETER temperature 0.1"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", required=True, help="tools/train/adapters/<name> (holds adapter/ and adapter_manifest.json)")
    ap.add_argument("--route", default="both", choices=["adapter", "merge", "both"])
    ap.add_argument("--outtype", default="q8_0", help="convert_hf_to_gguf --outtype for the merge route (f16, q8_0)")
    ap.add_argument("--base-ollama", default="", help="Ollama name of the base GGUF for the ADAPTER route (e.g. qwen3.5:0.8b)")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--llama-cpp", default="", help="an existing llama.cpp checkout")
    ap.add_argument("--plan", action="store_true", help="print what would run and stop")
    args = ap.parse_args(argv)

    adir = Path(args.adapter)
    am_path = adir / "adapter_manifest.json"
    if not am_path.is_file():
        raise SystemExit(f"no adapter_manifest.json in {adir}")
    am = json.loads(am_path.read_text(encoding="utf-8"))
    weights = Path(am["adapter"]["path"]) if am.get("adapter", {}).get("path") else None
    if weights and weights.is_file() and sha256_file(weights) != am["adapter"]["sha256"]:
        raise SystemExit("adapter weights do not match the manifest's sha256; refusing to export a modified adapter")
    out = adir / "gguf"
    tag = (am["adapter"]["sha256"] or "unknown")[:12]
    name = f"drock-reflex-{tag}"
    base_ollama = args.base_ollama or "qwen3.5:0.8b"

    if args.plan:
        print(f"adapter: {adir / 'adapter'}  (sha256 {am['adapter']['sha256']})")
        print(f"base:    {am['base']}")
        print(f"out:     {out}")
        print("A. convert_lora_to_gguf.py <adapter> --base <hf snapshot> --outfile adapter.gguf; Modelfile FROM "
              f"{base_ollama} + ADAPTER adapter.gguf")
        print(f"B. peft merge_and_unload -> merged-hf; convert_hf_to_gguf.py --outtype {args.outtype}; Modelfile FROM merged")
        print(f"then: ollama create {name} -f {out / 'Modelfile.<route>'}   (loads the GPU: not during a benchmark)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    llama, commit = ensure_llama_cpp(Path(args.llama_cpp) if args.llama_cpp else None)
    base = base_snapshot(am)
    res: dict = {"v": 1, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "adapter_manifest": str(am_path),
                 "adapter_sha256": am["adapter"]["sha256"], "dataset_sha256": (am.get("dataset") or {}).get("sha256"),
                 "base": am["base"], "base_snapshot": str(base), "llama_cpp": {"dir": str(llama), "commit": commit},
                 "ollama_model": name, "routes": {}}
    if args.route in ("adapter", "both"):
        try:
            r = route_adapter(adir / "adapter", base, out, llama)
            mf = out / "Modelfile.adapter"
            mf.write_text(modelfile("adapter", r, base_ollama, args.ctx), encoding="utf-8")
            r["modelfile"] = str(mf)
            res["routes"]["adapter"] = r
        except Exception as e:
            res["routes"]["adapter"] = {"error": f"{type(e).__name__}: {str(e)[-1500:]}"}
            print(f"adapter route failed: {res['routes']['adapter']['error']}", file=sys.stderr)
    if args.route in ("merge", "both"):
        try:
            r = route_merge(adir / "adapter", base, out, llama, args.outtype)
            mf = out / "Modelfile.merge"
            mf.write_text(modelfile("merge", r, base_ollama, args.ctx), encoding="utf-8")
            r["modelfile"] = str(mf)
            res["routes"]["merge"] = r
        except Exception as e:
            res["routes"]["merge"] = {"error": f"{type(e).__name__}: {str(e)[-1500:]}"}
            print(f"merge route failed: {res['routes']['merge']['error']}", file=sys.stderr)
    (out / "gguf_manifest.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    ok = [k for k, v in res["routes"].items() if "error" not in v]
    if ok:
        print(f"\nnext (GPU): ollama create {name} -f {res['routes'][ok[0]]['modelfile']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
