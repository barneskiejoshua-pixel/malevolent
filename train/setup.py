"""Create tools/train/.venv for Phase 5 training (BUILD-PLAN 5.2) -- standard library only.

DRock's core is stdlib-only and stays that way; everything PyTorch lives here, out of process, the way
tools/office/.venv holds python-docx (ADAPTERS.md). This script:

  1. creates tools/train/.venv from a CPython that PyTorch XPU ships wheels for (3.12 preferred,
     `py -3.12` on Windows; the core runs on 3.14, which the XPU index also serves but the rest of the
     stack lags behind);
  2. installs PyTorch XPU (https://download.pytorch.org/whl/xpu -- Intel Arc A-series on Windows is
     supported by stock PyTorch since 2.5, no oneAPI toolkit needed for the runtime);
  3. installs transformers, peft, datasets, accelerate, gguf (llama.cpp's converter package);
  4. tries Unsloth's Intel path (https://unsloth.ai/docs/get-started/install/intel, extra
     `intel-gpu-torch290`); on failure the plain peft LoRA path in train.py is used and the reason is
     recorded -- never a silent fallback;
  5. runs a READ-ONLY device query in the venv (torch.xpu.is_available, device name, free memory) --
     no tensor is allocated on the card, so a benchmark running on it is not disturbed;
  6. optionally (--download) fetches the base model weights from Hugging Face into the HF cache and
     pins the revision.

Everything observed is written to tools/train/setup-report.json (versions, venv size, the XPU query,
the base model revision) so README.md can quote measurements rather than intentions.

    python tools/train/setup.py [--download] [--no-unsloth] [--python PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REPORT = HERE / "setup-report.json"
LOG = HERE / "setup.log"
XPU_INDEX = "https://download.pytorch.org/whl/xpu"
# Pinned so a later round trains on the same stack; bump deliberately.
TORCH = "torch==2.9.1"
STACK = ["transformers>=4.57,<5", "peft>=0.17", "datasets>=3.0", "accelerate>=1.0", "safetensors", "sentencepiece",
         "protobuf", "gguf", "huggingface_hub", "numpy"]
UNSLOTH = "unsloth[intel-gpu-torch290]"
# Qwen3.5-0.8B exists on the Hub (model_type qwen3_5); Qwen3-0.6B is the fallback the research notes name.
BASE_CANDIDATES = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], timeout: int = 3600, check: bool = True) -> tuple[int, str]:
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    out = (p.stdout or "") + (p.stderr or "")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(out[-20000:] + "\n")
    if check and p.returncode != 0:
        raise RuntimeError(f"{cmd[:3]} failed ({p.returncode}): {out[-1500:]}")
    return p.returncode, out


def find_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.name == "nt" and shutil.which("py"):
        for ver in ("3.12", "3.13", "3.11", "3.10"):
            rc, out = run(["py", f"-{ver}", "-c", "import sys; print(sys.executable)"], check=False)
            if rc == 0 and out.strip():
                return out.strip().splitlines()[-1]
    for name in ("python3.12", "python3.13", "python3.11"):
        if shutil.which(name):
            return shutil.which(name)
    return sys.executable


def dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def pip(*args: str, timeout: int = 3600, check: bool = True) -> tuple[int, str]:
    return run([str(PY), "-m", "pip", *args], timeout=timeout, check=check)


XPU_QUERY = r"""
import json, torch
r = {"torch": torch.__version__, "xpu_built": hasattr(torch, "xpu")}
try:
    r["xpu_available"] = bool(torch.xpu.is_available())
except Exception as e:
    r["xpu_available"] = False; r["xpu_error"] = f"{type(e).__name__}: {e}"
if r.get("xpu_available"):
    r["device_count"] = torch.xpu.device_count()
    r["device_name"] = torch.xpu.get_device_name(0)
    try:
        p = torch.xpu.get_device_properties(0)
        r["total_memory_bytes"] = int(getattr(p, "total_memory", 0))
        r["driver_version"] = str(getattr(p, "driver_version", ""))
    except Exception as e:
        r["props_error"] = f"{type(e).__name__}: {e}"
    try:  # (free, total) from the runtime; read-only
        free, total = torch.xpu.mem_get_info(0)
        r["free_memory_bytes"], r["total_memory_bytes"] = int(free), int(total)
    except Exception as e:
        r["mem_get_info_error"] = f"{type(e).__name__}: {e}"
    try:
        r["bf16_supported"] = bool(torch.xpu.is_bf16_supported())
    except Exception as e:
        r["bf16_error"] = f"{type(e).__name__}: {e}"
print("XPU_REPORT " + json.dumps(r))
"""

VERSIONS_QUERY = r"""
import importlib, json
out = {}
for m in ("torch", "transformers", "peft", "datasets", "accelerate", "gguf", "huggingface_hub", "unsloth", "unsloth_zoo", "triton"):
    try:
        mod = importlib.import_module(m)
        out[m] = getattr(mod, "__version__", "?")
    except Exception as e:
        out[m] = None
print("VERSIONS " + json.dumps(out))
"""

DOWNLOAD = r"""
import json, sys, huggingface_hub as h
repo = sys.argv[1]
info = h.model_info(repo)
p = h.snapshot_download(repo, revision=info.sha,
                        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "LICENSE*", "NOTICE*", "README.md"])
print("MODEL " + json.dumps({"repo": repo, "revision": info.sha, "path": p,
                             "model_type": (info.config or {}).get("model_type")}))
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", default=None, help="interpreter to build the venv from (default: py -3.12)")
    ap.add_argument("--download", action="store_true", help="also fetch the base model weights into the HF cache")
    ap.add_argument("--no-unsloth", action="store_true", help="skip the Unsloth attempt")
    ap.add_argument("--recreate", action="store_true", help="delete an existing venv first")
    ap.add_argument("--base", action="append", default=[],
                    help="with --download: a repo to fetch (repeatable; every one is fetched). Default: the first of "
                         "BASE_CANDIDATES that downloads")
    args = ap.parse_args(argv)

    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": sys.platform}
    if args.recreate and VENV.exists():
        shutil.rmtree(VENV)
    if not PY.exists():
        base = find_python(args.python)
        log(f"creating {VENV} from {base}")
        run([base, "-m", "venv", str(VENV)])
    report["python"] = run([str(PY), "-c", "import sys; print(sys.version)"])[1].strip()
    pip("install", "--upgrade", "pip", check=False)

    # --- torch XPU ---
    t0 = time.time()
    pip("install", TORCH, "--index-url", XPU_INDEX, timeout=7200)
    report["torch_install_seconds"] = round(time.time() - t0)
    # --- the stack (PyPI); torch already satisfied so the CPU wheel is not pulled in ---
    pip("install", *STACK, timeout=3600)

    # --- Unsloth's Intel path ---
    if args.no_unsloth:
        report["unsloth"] = {"installed": False, "reason": "skipped (--no-unsloth)"}
    else:
        rc, out = pip("install", UNSLOTH, timeout=3600, check=False)
        if rc == 0:
            rc2, out2 = run([str(PY), "-c", "import unsloth; print(unsloth.__version__)"], check=False)
            report["unsloth"] = {"installed": rc2 == 0, "reason": None if rc2 == 0 else "installed but import failed: " + out2[-800:]}
        else:
            report["unsloth"] = {"installed": False, "reason": "pip install failed: " + out[-800:]}
        if not report["unsloth"]["installed"]:
            log("Unsloth not usable here; train.py uses plain peft LoRA (recorded in setup-report.json)")
            # a half-installed unsloth must not shadow torch: make sure torch is still the XPU build
            pip("install", TORCH, "--index-url", XPU_INDEX, timeout=7200, check=False)

    # --- versions + the read-only device query ---
    rc, out = run([str(PY), "-c", VERSIONS_QUERY], check=False)
    for line in out.splitlines():
        if line.startswith("VERSIONS "):
            report["versions"] = json.loads(line[len("VERSIONS "):])
    rc, out = run([str(PY), "-c", XPU_QUERY], check=False)
    report["xpu"] = {"error": out[-1200:]}
    for line in out.splitlines():
        if line.startswith("XPU_REPORT "):
            report["xpu"] = json.loads(line[len("XPU_REPORT "):])
    log(f"xpu: {json.dumps(report['xpu'])}")

    # --- base model ---
    if args.download:
        for repo in args.base or BASE_CANDIDATES:
            rc, out = run([str(PY), "-c", DOWNLOAD, repo], timeout=7200, check=False)
            got = next((json.loads(l[6:]) for l in out.splitlines() if l.startswith("MODEL ")), None)
            if got:
                got["size_bytes"] = dir_size(Path(got["path"]))
                got["license_file"] = (Path(got["path"]) / "LICENSE").exists()
                report.setdefault("base_models", []).append(got)
                report.setdefault("base_model", got)
                if not args.base:
                    break
                continue
            report.setdefault("base_model_attempts", []).append({"repo": repo, "error": out[-500:]})

    report["venv_size_bytes"] = dir_size(VENV)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    log(f"report -> {REPORT}")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
