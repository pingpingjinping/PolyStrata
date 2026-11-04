#!/usr/bin/env python3
r"""Launch experimental Hy-MT2 in PolyStrata's existing web UI, isolated from other models.

Windows from the experimental clone:
    & "$HOME\Desktop\PolyStrata\.venv\Scripts\python.exe" .\tools\hy_v3_web.py

Uses the same native strata-poly.exe and Tencent GGUF as hy_v3_translate_gui.py.
No llama.cpp, new model download, or replacement of existing run configs is needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
DEFAULT_MODEL_DIR = Path.home() / "Desktop" / "PolyStrata"
RUNTIME = ROOT / ".hy-v3-web"


def cuda13_lib_dirs(exe: Path, interpreter: Path | None = None) -> list[str]:
    """Find pip's CUDA 13 DLLs (the user's previous missing-DLL fix)."""
    interpreter = interpreter or Path(sys.executable)
    candidates = [
        interpreter.parents[1] / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
        exe.parent / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
        DEFAULT_MODEL_DIR / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
    ]
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]) / "bin")
    candidates.append(Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"))
    found = []
    for d in candidates:
        if (d / "cublas64_13.dll").is_file() and str(d) not in found:
            found.append(str(d))
    return found


def prepare_tokenizer(model: Path, runtime: Path) -> Path:
    """Extract tokenizer/chat template from the GGUF header; don't read model weights."""
    from gguf_reader import GGUFFile
    from strata_tokenizer import extract

    meta = GGUFFile(model).metadata
    arch = meta.get("general.architecture")
    pre = meta.get("tokenizer.ggml.pre")
    if arch != "hy_v3":
        raise ValueError(f"Expected a hy_v3 GGUF (found {arch!r})")
    if pre != "hunyuan-dense":
        raise ValueError(f"Expected Tencent Hy-MT2 hunyuan-dense tokenizer (found {pre!r})")
    if not meta.get("tokenizer.chat_template"):
        raise ValueError("GGUF has no tokenizer.chat_template; cannot use PolyStrata's normal chat UI safely")

    tok = runtime / "tokenizer-pack" / "tokenizer"
    config = tok / "tokenizer.json"
    try:
        recorded = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        recorded = {}
    if recorded.get("pre") != pre or not all(
        (tok / filename).is_file()
        for filename in ("vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")
    ):
        print("[hy-v3] Extracting the tokenizer and official chat template from GGUF metadata...", flush=True)
        extract(model, runtime / "tokenizer-pack")
    return tok


def build_config(*, exe: Path, model: Path, tokenizer: Path, runtime: Path,
                 port: int, context: int, lib_dirs: list[str], temperature: float,
                 ram_experts_mib: int = -1, prompt_batch: int = 1024,
                 stage_mib: int = 128, stage_read_mib: int = 2048,
                 prompt_cache: int = 6, expert_profile: Path | None = None,
                 threads: int | None = None) -> dict:
    """A private run config understood directly by serve/server.py."""
    config = {
        "exe": str(exe),
        "args": [
            "--model", str(model),
            "--serve",
            "--max-context", str(context),
            "--prompt-batch", str(prompt_batch),
            "--stage-mib", str(stage_mib),
            "--stage-read-mib", str(stage_read_mib),
            "--ram-experts-mib", str(ram_experts_mib),
            "--no-draft",
            "--window", "1",
            "--lookup", "0",
            "--prompt-cache", str(prompt_cache),
        ],
        "cwd": str(ROOT),
        "tokenizer": str(tokenizer),
        "model_name": "hy-mt2-30b-a3b-q4-k-m-experimental",
        "display_name": "Hy-MT2 30B-A3B Q4_K_M (experimental)",
        "arch": "hy_v3",
        "log": str(runtime / "strata-hy-mt2.log"),
        "lib_dirs": lib_dirs,
        "host": "127.0.0.1",
        "port": port,
        # Greedy was tested in the standalone UI; 0.7 is the 30B model card default.
        "sampling": {"temperature": temperature, "top_p": 1.0, "repetition_penalty": 1.0},
        "reasoning_effort": "none",
        "fit_max_tokens": True,
    }
    if expert_profile is not None:
        config["args"] += ["--expert-profile", str(expert_profile)]
    if threads is not None:
        config["args"] += ["--threads", str(threads)]
    return config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, default=DEFAULT_MODEL_DIR / "strata-poly.exe")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR / "Hy-MT2-30B-A3B-Q4_K_M.gguf")
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--prompt-batch", type=int, default=1024,
                    help="prompt tokens per evaluation (default 1024, same as regular strata-poly; "
                         "try 512/256 if RTX 5060 8GB runs out of VRAM)")
    ap.add_argument("--stage-mib", type=int, default=128,
                    help="GPU staging buffer while reading prompts (default 128 MiB)")
    ap.add_argument("--stage-read-mib", type=int, default=2048,
                    help="temporarily widen staging from expert VRAM during long prefill (default 2048; 0 to disable)")
    ap.add_argument("--prompt-cache", type=int, default=6,
                    help="conversation prompt checkpoints (default 6; 0 to disable)")
    ap.add_argument("--learn-experts", action="store_true",
                    help="opt in to a private persistent expert-use profile for translation; "
                         "restart after representative translations to use its cache ranking")
    ap.add_argument("--threads", type=int,
                    help="CPU threads for off-VRAM experts (default: engine auto); "
                         "compare physical cores with auto when CPU expert work dominates")
    ap.add_argument("--ram-experts-mib", type=int, default=-1,
                    help="-1 holds all experts outside VRAM in RAM (~10.7 GiB on this model); "
                         "set 4096 to restore the previous memory-saving mode")
    ap.add_argument("--no-open", action="store_true", help="Don't open the browser automatically")
    args = ap.parse_args(argv)
    exe = args.exe.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not exe.is_file():
        ap.error(f"missing experimental strata-poly.exe: {exe}")
    if not model.is_file():
        ap.error(f"missing GGUF: {model}")
    if args.port < 1024 or args.port > 65535:
        ap.error("--port must be 1024..65535")
    if args.context < 512 or args.context > 8192:
        ap.error("--context must be 512..8192 for the first HyV3 web test")
    if not 0 <= args.temperature <= 2:
        ap.error("--temperature must be 0..2")
    if not 1 <= args.prompt_batch <= 4096:
        ap.error("--prompt-batch must be 1..4096")
    if not 16 <= args.stage_mib <= 2048:
        ap.error("--stage-mib must be 16..2048")
    if not 0 <= args.stage_read_mib <= 4096:
        ap.error("--stage-read-mib must be 0..4096")
    if not 0 <= args.prompt_cache <= 32:
        ap.error("--prompt-cache must be 0..32")
    if args.threads is not None and args.threads < 1:
        ap.error("--threads must be positive")
    if args.ram_experts_mib != -1 and not 1024 <= args.ram_experts_mib <= 16384:
        ap.error("--ram-experts-mib must be -1 or 1024..16384")

    runtime = RUNTIME
    runtime.mkdir(parents=True, exist_ok=True)
    try:
        tokenizer = prepare_tokenizer(model, runtime)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[hy-v3] Tokenizer error: {exc}", file=sys.stderr)
        return 1

    lib_dirs = cuda13_lib_dirs(exe)
    if os.name == "nt" and not lib_dirs and "cublas64_13.dll" not in os.listdir(exe.parent):
        print("[hy-v3] CUDA 13 DLL folder not found. Use the existing PolyStrata .venv Python.",
              file=sys.stderr)
        return 1
    config = build_config(exe=exe, model=model, tokenizer=tokenizer, runtime=runtime,
                          port=args.port, context=args.context, lib_dirs=lib_dirs,
                          temperature=args.temperature, ram_experts_mib=args.ram_experts_mib,
                          prompt_batch=args.prompt_batch, stage_mib=args.stage_mib,
                          stage_read_mib=args.stage_read_mib, prompt_cache=args.prompt_cache,
                          expert_profile=runtime / "hy-mt2-experts.bin" if args.learn_experts else None,
                          threads=args.threads)
    config_path = runtime / "strata-hy-mt2.json"
    temporary = config_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(config_path)
    print(f"[hy-v3] Hy-MT2 Q4_K_M experimental web server: http://127.0.0.1:{args.port}", flush=True)
    print(f"[hy-v3] Config kept separate: {config_path}", flush=True)
    print(f"[hy-v3] Prefill: batch={args.prompt_batch}, stage={args.stage_mib} MiB, "
          f"read stage={args.stage_read_mib} MiB, prompt cache={args.prompt_cache}", flush=True)
    print("[hy-v3] Close the other PolyStrata server first to free VRAM. Ctrl+C stops this server.",
          flush=True)
    print("[hy-v3] Send a translation instruction (e.g., 'Translate into Korean...') with the Japanese text.",
          flush=True)
    from serve import server
    # server.main() supplies the existing Chat page, monitoring, model information,
    # OpenAI-compatible /v1 endpoint and streaming to the native HyV3 engine.
    original_argv = sys.argv
    sys.argv = [str(ROOT / "serve" / "server.py"), "--engine", "strata",
                "--config", str(config_path), "--port", str(args.port)]
    if not args.no_open:
        sys.argv.append("--open")
    try:
        return server.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
