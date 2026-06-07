from __future__ import annotations

import re

_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(output: str) -> str | None:
    """Pull code out of a model reply.

    Returns the first fenced block's contents if present, else the whole
    stripped reply. Returns None only when nothing non-blank remains.
    """
    m = _FENCE.search(output)
    if m:
        return m.group(1).strip() or None
    stripped = output.strip()
    return stripped or None
