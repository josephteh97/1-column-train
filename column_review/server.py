"""FastAPI app factory + launch helpers (port picker, browser opener).

`create_app(config)` returns a FastAPI instance with the StaticFiles
mount and a startup hook that ensures the corrections-DB schema is
ready. The shape of `config` is set by `column_review.cli.main` and
read here without further validation — internal contract.
"""
from __future__ import annotations

import socket
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


_PACKAGE_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _PACKAGE_ROOT / "static"


def pick_port(start: int, attempts: int = 20) -> int:
    """Return the first free loopback TCP port in `[start, start+attempts)`.

    Raises `RuntimeError` if every port in the window is busy. Same
    contract as the deleted `correction_app.app.pick_port`.
    """
    for p in range(start, start + attempts):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
            continue
    raise RuntimeError(
        f"No free port in [{start}, {start + attempts}). "
        "Pass --port to retry from a different base."
    )


def open_browser_soon(url: str, delay_seconds: float = 1.5) -> None:
    """Open the browser after `delay_seconds` on a daemon thread.

    Daemon so it doesn't block uvicorn's foreground run. Failures are
    swallowed — opening the browser is a nicety, not a correctness path.
    """
    def _open() -> None:
        time.sleep(delay_seconds)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def create_app(config: dict) -> FastAPI:
    """Build the FastAPI app for one server lifetime.

    Startup hook: ensure the SQLite schema (corrections + sidecar tables
    + retrain_jobs) is in place. The hook runs once per process; the
    DDL is idempotent so repeated launches against the same DB are safe.
    """
    from column_review.db import (
        ensure_retrain_jobs_table,
        ensure_sidecar_tables,
        get_connection,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        db_path = config.get("db_path")
        conn = get_connection(db_path)
        try:
            ensure_sidecar_tables(conn)
            ensure_retrain_jobs_table(conn)
            conn.commit()
        finally:
            conn.close()
        yield

    app = FastAPI(title="column-review", lifespan=lifespan)
    app.state.config = config

    # `html=True` makes `/` serve `static/index.html`; everything else
    # under `/` is served as a static asset.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True),
              name="static")

    return app
