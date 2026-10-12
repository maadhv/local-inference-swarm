"""Locating and interrogating llama.cpp executables. Never runs anything from the CWD implicitly."""
import glob
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
_EXCLUDED_DIRS = {PROJECT_DIR, PROJECT_DIR.parent}


def _norm(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


def _is_excluded(d: Path) -> bool:
    d = _norm(d)
    return d == _norm(Path.cwd()) or d in {_norm(x) for x in _EXCLUDED_DIRS}


def _winget_dirs() -> List[Path]:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return []
    root = Path(base) / "Microsoft" / "WinGet"
    dirs = [root / "Links"]
    pkgs = sorted(glob.glob(str(root / "Packages" / "ggml.llamacpp_*")), key=os.path.getmtime, reverse=True)
    dirs += [Path(p) for p in pkgs]
    return dirs


def find_executable(names: List[str], explicit: Optional[str] = None) -> Tuple[Optional[Path], str]:
    """Return (absolute path | None, how it was found).
    Order: explicit path -> PATH (skipping CWD/project dir) -> WinGet install dirs.
    """
    if explicit:
        p = Path(explicit.strip().strip('"'))
        if p.is_file():
            return p.absolute(), "explicit path"
        return None, f"explicit path not found: {p}"
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or entry == ".":
            continue
        d = Path(entry.strip('"'))
        if _is_excluded(d):
            continue
        for n in names:
            c = d / n
            if c.is_file():
                return c.absolute(), "PATH"
    for d in _winget_dirs():
        for n in names:
            c = d / n
            if c.is_file():
                return c.absolute(), "WinGet"
            hits = sorted(d.rglob(n), key=lambda x: x.stat().st_mtime, reverse=True) if d.is_dir() else []
            if hits:
                return hits[0].absolute(), "WinGet"
    return None, "not found on PATH or in the WinGet install directory"


def run_capture(cmd: List[str], timeout: float = 60.0) -> Tuple[Optional[int], str]:
    """Run a short command, return (returncode | None on timeout, combined output)."""
    try:
        p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or b"").decode("utf-8", "replace")
    except OSError as e:
        return -1, f"Could not execute {cmd[0]}: {e}"


def get_version(exe: Path) -> str:
    _, out = run_capture([str(exe), "--version"], timeout=30)
    for line in out.splitlines():
        if line.lower().startswith("version"):
            return line.strip()
    return (out.strip().splitlines() or ["unknown"])[0]


_help_cache = {}


def get_help(exe: Path) -> str:
    key = str(exe)
    if key not in _help_cache:
        _help_cache[key] = run_capture([key, "--help"], timeout=30)[1]
    return _help_cache[key]


def supports_flag(exe: Path, flag: str) -> bool:
    return re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", get_help(exe)) is not None


@dataclass
class Device:
    id: str        # e.g. Vulkan1
    name: str      # e.g. NVIDIA GeForce GTX 1650
    mem_mib: Optional[int] = None

    @property
    def backend(self) -> str:
        return re.sub(r"\d+$", "", self.id)


_DEV_RE = re.compile(r"^\s+([A-Za-z]+\d+):\s+(.+?)(?:\s+\((\d+)\s*MiB.*\))?\s*$")


def parse_devices(text: str) -> List[Device]:
    devs = []
    for line in text.splitlines():
        m = _DEV_RE.match(line)
        if m:
            devs.append(Device(m.group(1), m.group(2).strip(), int(m.group(3)) if m.group(3) else None))
    return devs


def list_devices(llama_cli: Path, extra_args: Optional[List[str]] = None) -> Tuple[List[Device], str]:
    _, out = run_capture([str(llama_cli)] + (extra_args or []) + ["--list-devices"], timeout=90)
    return parse_devices(out), out
