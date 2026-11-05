#!/usr/bin/env python3
"""Small offline Japanese -> Korean test window for experimental native HyV3.

No llama.cpp or PolyStrata server setup needed. Uses the GGUF's own tokenizer
and strata-poly.exe's existing stdin/stdout --serve protocol. This is a
diagnostic UI, NOT a validated translation implementation.

Windows:
  .\.venv\Scripts\python.exe tools\hy_v3_translate_gui.py
  python tools\hy_v3_translate_gui.py --model D:\models\Hy-MT2-30B-A3B-Q4_K_M.gguf

The native engine is kept resident across translation requests.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# Match Tencent's Hy-MT2 30B-A3B chat_template.jinja for a single user message
# with add_generation_prompt=True, reasoning_effort="no_think", and no tools.
BOS = "<｜hy_begin▁of▁sentence｜>"
REASON = "<｜reasoning_mode｜>reasoning_effort:no_think"
USER = "<｜hy_User｜>"
ASSISTANT = "<｜hy_Assistant｜>"
NO_THINK = "<think></think>"
PROMPT_PREFIX = (
    "Translate the following text into Korean. "
    "Note that you should only output the translated result without "
    "any additional explanation:\n\n"
)


def translation_prompt(source: str) -> tuple[str, tuple[int, int]]:
    """Return full model chat prompt and the span that must stay ordinary user text."""
    start = BOS + REASON + USER + PROMPT_PREFIX
    return start + source + ASSISTANT + NO_THINK, (len(start), len(start) + len(source))


def runtime_env(exe: Path) -> dict[str, str]:
    """Make the same CUDA 13 pip-wheel DLL directory visible to the child process."""
    env = os.environ.copy()
    candidates = [
        exe.parent / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
        ROOT / ".venv" / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
        Path(env.get("CUDA_PATH", "C:/missing-cuda-path")) / "bin",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin"),
    ]
    # Also support existing environment-provided CUDA paths.
    valid = [str(d) for d in candidates if (d / "cublas64_13.dll").is_file()]
    if valid:
        env["PATH"] = os.pathsep.join(dict.fromkeys(valid)) + os.pathsep + env.get("PATH", "")
    return env


class TranslateWindow:
    def __init__(self, master: tk.Tk, exe: Path, model: Path):
        self.root, self.exe, self.model = master, exe, model
        self.proc: subprocess.Popen | None = None
        self.tok = None
        self.stop_ids: set[int] = set()
        self.q: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.closed = False
        self.max_new = tk.IntVar(value=192)
        self.temperature = tk.StringVar(value="0")
        master.title("Hy-MT2 · 일본어 → 한국어 (실험용)")
        master.geometry("850x680")
        master.minsize(680, 490)
        frame = ttk.Frame(master, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="일본어 원문").pack(anchor="w")
        self.source = scrolledtext.ScrolledText(frame, height=9, wrap="word", font=("Malgun Gothic", 11))
        self.source.pack(fill="both", expand=True, pady=(4, 12))
        self.source.insert("1.0", "今日はいい天気ですね。\n明日も晴れるといいな。")
        options = ttk.Frame(frame)
        options.pack(fill="x", pady=(0, 10))
        ttk.Label(options, text="최대 생성 토큰").pack(side="left")
        ttk.Spinbox(options, from_=16, to=512, increment=16, textvariable=self.max_new, width=7).pack(
            side="left", padx=(6, 18))
        ttk.Label(options, text="Temperature").pack(side="left")
        ttk.Combobox(options, values=["0", "0.7"], textvariable=self.temperature,
                     state="readonly", width=5).pack(side="left", padx=(6, 18))
        self.translate_btn = ttk.Button(options, text="번역하기", command=self.translate)
        self.translate_btn.pack(side="left")
        self.stop_btn = ttk.Button(options, text="중단", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        ttk.Label(frame, text="한국어 번역 결과").pack(anchor="w")
        self.output = scrolledtext.ScrolledText(frame, height=10, wrap="word", font=("Malgun Gothic", 11))
        self.output.pack(fill="both", expand=True, pady=(4, 9))
        self.output.configure(state="disabled")
        self.status = tk.StringVar(value=f"준비됨 · {model.name}")
        ttk.Label(frame, textvariable=self.status, wraplength=780).pack(anchor="w")
        master.protocol("WM_DELETE_WINDOW", self.close)
        master.after(100, self.flush)

    def emit(self, kind: str, data: object) -> None:
        self.q.put((kind, data))

    def set_output(self, value: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", value)
        self.output.configure(state="disabled")

    def translate(self) -> None:
        if self.busy:
            return
        source = self.source.get("1.0", "end-1c").strip()
        if not source:
            messagebox.showinfo("입력 필요", "일본어 문장을 입력해줘.")
            return
        if not self.exe.is_file() or not self.model.is_file():
            messagebox.showerror("파일 확인", f"실행 파일: {self.exe}\n모델: {self.model}")
            return
        try:
            max_new = int(self.max_new.get())
            if not 1 <= max_new <= 512:
                raise ValueError()
        except (ValueError, tk.TclError):
            messagebox.showerror("설정 오류", "최대 생성 토큰은 1~512로 설정해줘.")
            return
        self.busy = True
        self.translate_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.set_output("")
        self.status.set("토크나이저 / 모델 준비 중...")
        threading.Thread(target=self.worker, args=(source, max_new, self.temperature.get()),
                         daemon=True).start()

    def stderr_reader(self, proc: subprocess.Popen) -> None:
        for line in proc.stderr:
            msg = line.strip()
            if msg.startswith("strata-poly: loaded"):
                self.emit("status", msg)

    def ensure_model(self) -> None:
        if self.tok is None:
            from gguf_reader import GGUFFile
            from strata_tokenizer import Tokenizer
            md = GGUFFile(self.model).metadata
            if md.get("general.architecture") != "hy_v3":
                raise RuntimeError("hy_v3 GGUF 파일이 아닙니다.")
            pre = md.get("tokenizer.ggml.pre")
            if pre not in ("hunyuan", "hunyuan-dense", "qwen2"):
                raise RuntimeError(
                    f"지원 확인이 필요한 토크나이저 pre={pre!r}. 잘못된 토큰으로 "
                    "번역하지 않도록 실행을 중단했습니다.")
            self.tok = Tokenizer.from_gguf(self.model)
            for marker in (BOS, "<｜reasoning_mode｜>", USER, ASSISTANT):
                if len(self.tok.encode(marker, parse_special=True)) != 1:
                    raise RuntimeError(f"모델 특수 토큰을 확인할 수 없습니다: {marker}")
            ids = self.tok.special_ids
            self.stop_ids = {int(ids[k]) for k in
                ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                 "tokenizer.ggml.eom_token_id") if k in ids}
            self.stop_ids.update(i for i, t in enumerate(self.tok.tokens)
                                 if t in ("<eos:6124c78e>", "<｜hy_endofsentence｜>", "<｜hy_EOT｜>"))
            self.emit("status", f"GGUF 토크나이저 준비 · pre={pre} · 정지 토큰={sorted(self.stop_ids)}")

        if self.proc is not None and self.proc.poll() is None:
            return
        args = [str(self.exe), "--model", str(self.model), "--serve",
                "--max-context", "1024", "--prompt-batch", "32",
                "--stage-mib", "64", "--ram-experts-mib", "4096",
                "--no-draft", "--window", "1", "--lookup", "0",
                "--prompt-cache", "0"]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            bufsize=1, env=runtime_env(self.exe), creationflags=flags)
        threading.Thread(target=self.stderr_reader, args=(self.proc,), daemon=True).start()
        for line in self.proc.stdout:
            if line.startswith("READY "):
                self.emit("status", "모델 로딩 완료 · 번역 중...")
                return
            if line.startswith("ERR "):
                raise RuntimeError(line.strip())
        raise RuntimeError(f"엔진이 준비 전에 종료됐습니다 (코드 {self.proc.poll()}). DLL 경로를 확인하세요.")

    def worker(self, source: str, max_new: int, temp: str) -> None:
        try:
            self.ensure_model()
            prompt, plain = translation_prompt(source)
            ids = self.tok.encode(prompt, parse_special=True, plain=(plain,))
            # Check that original Japanese survives tokenization.
            if self.tok.decode(self.tok.encode(source)) != source:
                raise RuntimeError("토크나이저의 일본어 왕복 검사가 실패했습니다.")
            if len(ids) + max_new >= 1024:
                raise RuntimeError(f"입력 {len(ids)} 토큰 + 출력 {max_new} 토큰이 1024 컨텍스트를 초과합니다.")
            cmd = f"GEN {max_new} temperature={temp} " + ",".join(map(str, ids)) + "\n"
            self.proc.stdin.write(cmd)
            self.proc.stdin.flush()
            generated: list[int] = []
            begun = time.monotonic()
            for raw in self.proc.stdout:
                line = raw.strip()
                if line.startswith("T "):
                    tid = int(line[2:])
                    if tid not in self.stop_ids:
                        generated.append(tid)
                    answer = self.tok.decode(generated)
                    self.emit("answer", answer.replace("<think></think>", ""))
                elif line.startswith("DONE "):
                    parts = line.split()
                    self.emit("status", f"완료 · 출력 {len(generated)} 토큰 · {time.monotonic()-begun:.1f}초 · 종료: {parts[5] if len(parts)>5 else 'done'}")
                    break
                elif line.startswith("ERR "):
                    raise RuntimeError(line)
            else:
                raise RuntimeError(f"엔진 연결 종료 (코드 {self.proc.poll()})")
        except Exception as exc:
            self.emit("error", str(exc))
        finally:
            self.emit("idle", None)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write("STOP\n")
                self.proc.stdin.flush()
                self.status.set("중단 요청 중...")
            except (OSError, ValueError):
                pass

    def flush(self) -> None:
        if self.closed:
            return
        while True:
            try:
                kind, value = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "status":
                self.status.set(str(value))
            elif kind == "answer":
                self.set_output(str(value))
            elif kind == "error":
                self.status.set("오류: " + str(value))
                self.set_output("[오류]\n" + str(value))
            elif kind == "idle":
                self.busy = False
                self.translate_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
        self.root.after(100, self.flush)

    def close(self) -> None:
        self.closed = True
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.root.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    desktop = Path.home() / "Desktop" / "PolyStrata"
    ap.add_argument("--exe", type=Path, default=desktop / "strata-poly.exe")
    ap.add_argument("--model", type=Path, default=desktop / "Hy-MT2-30B-A3B-Q4_K_M.gguf")
    ns = ap.parse_args()
    root = tk.Tk()
    TranslateWindow(root, ns.exe.expanduser().resolve(), ns.model.expanduser().resolve())
    root.mainloop()


if __name__ == "__main__":
    main()
