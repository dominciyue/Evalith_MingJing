from __future__ import annotations

import subprocess
import sys
import tempfile

# Injected at the top of every program. Runs INSIDE the child before user
# code: sets resource limits (thread-safe, unlike preexec_fn) and neuters the
# most dangerous calls (defense-in-depth on top of the subprocess boundary).
_PREAMBLE = """\
import resource as _r
for _name, _lim in [("RLIMIT_AS", {mem}), ("RLIMIT_CPU", {cpu}), ("RLIMIT_FSIZE", {fsize})]:
    try:
        _r.setrlimit(getattr(_r, _name), (_lim, _lim))
    except (ValueError, OSError):
        pass
import os, shutil
for _m, _a in [(os, "system"), (os, "remove"), (os, "unlink"), (os, "rmdir"),
               (os, "removedirs"), (os, "kill"), (os, "killpg"),
               (shutil, "rmtree"), (shutil, "move")]:
    try:
        setattr(_m, _a, None)
    except Exception:
        pass
try:
    import subprocess as _sp
    _sp.Popen = _sp.run = _sp.call = None
except Exception:
    pass
"""


def run_program(source: str, *, timeout: float = 5, memory_mb: int = 256) -> tuple[bool, str]:
    """Run `source` in an isolated subprocess.

    Returns (passed, detail). Exit 0 -> (True, "ok"); timeout / nonzero exit /
    spawn failure -> (False, reason). Never raises.
    """
    preamble = _PREAMBLE.format(
        mem=memory_mb * 1024 * 1024, cpu=int(timeout) + 1, fsize=1024 * 1024
    )
    program = preamble + "\n" + source
    try:
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env={"PATH": "/usr/bin:/bin"},
            )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout:g}s"
    except Exception as e:  # spawn failure, etc. — stay resilient
        return False, f"sandbox error: {e}"
    if proc.returncode == 0:
        return True, "ok"
    lines = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = " ".join(lines[-3:]) if lines else f"exit {proc.returncode}"
    return False, tail[:300]
