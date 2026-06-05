"""Idempotent `sys.path` extension — the one place this project does
import-path manipulation.

Why this exists: scripts under `scripts/` need to import each other
(`scripts.corrections_logger`, `scripts.tile_geometry`, etc.) AND
`column_review/` code needs to import `scripts.*` via its `_PROJECT_ROOT`.
Both directions require their parent directory to be on `sys.path`.

Doing `sys.path.insert(...)` unconditionally — especially inside a
function called per-request — leaks one duplicate entry per call. Over
a long-lived `column-review` server lifetime that's thousands of
duplicate entries, slowing every subsequent `import` linearly.

This helper is idempotent: calling `ensure_on_path(p)` twice is a
no-op the second time. `column_review` is registered as a pip-editable
package (see `pyproject.toml`), so this module is importable from
anywhere — including `scripts/*.py` invoked directly with `python3` —
without requiring a separate bootstrap.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def ensure_on_path(*paths: str | Path) -> None:
    """Insert each path at position 0 of `sys.path`, but only if it's
    not already present. Accepts strings or `Path` objects.

    Order: paths inserted in REVERSE so the FIRST argument ends up at
    `sys.path[0]` (highest precedence) — matches the human intuition
    of "first listed, first found".
    """
    for p in reversed(paths):
        s = str(p)
        if s and s not in sys.path:
            sys.path.insert(0, s)
