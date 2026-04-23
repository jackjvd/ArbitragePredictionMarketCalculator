from __future__ import annotations

import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

ROOT = Path(".").resolve()

ALLOWED_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("uv", "run", "arb_finder.py"),
    ("uv", "run", "label_pairs.py"),
    ("uv", "run", "top_arb.py"),
    ("uv", "run", "-m", "src.ml.train"),
    ("uv", "run", "-m", "src.ml.score"),
    ("uv", "run", "main.py", "analyze"),
)


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int
    output: str


def _is_allowed(command: Iterable[str]) -> bool:
    cmd = tuple(command)
    return any(cmd[: len(prefix)] == prefix for prefix in ALLOWED_PREFIXES)


def run_command_streaming(command: list[str], cwd: Path | None = None) -> CommandResult:
    if not _is_allowed(command):
        raise ValueError(f"Command not allowed: {' '.join(command)}")

    proc = subprocess.Popen(
        command,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    live = st.empty()
    live.code("Running: " + " ".join(command), language="bash")

    assert proc.stdout is not None
    for line in proc.stdout:
        output_lines.append(line.rstrip("\n"))
        live.code("\n".join(output_lines[-200:]), language="bash")

    proc.wait()
    out = "\n".join(output_lines)
    return CommandResult(command=command, exit_code=proc.returncode, output=out)
