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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


_PACKAGE_ROOT = Path(__file__).resolve().parent
_STATIC_DIR = _PACKAGE_ROOT / "static"


def pick_port(start: int, attempts: int = 20) -> int:
    """Return the first free loopback TCP port in `[start, start+attempts)`.

    Raises `RuntimeError` if every port in the window is busy.
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
        from column_review.retrain_jobs import (
            reap_orphans, start_poller_thread,
        )
        db_path = config.get("db_path")
        conn = get_connection(db_path)
        try:
            ensure_sidecar_tables(conn)
            ensure_retrain_jobs_table(conn)
            conn.commit()
        finally:
            conn.close()
        # Mark any `running` rows whose PID is dead — happens when the
        # server was killed while a retrain was active. Without this,
        # the UI status pill would show "running" forever.
        reap_orphans(db_path)
        # Spawn the daemon thread that polls live Popen objects and
        # flips DB statuses on exit. Once per process.
        start_poller_thread(db_path, config["project_root"])
        # Perf budget contract surfaced at startup — R11 requires
        # loud-fail-at-startup rather than silent degradation.
        print(
            "[perf] budgets: open<=3000ms, mark<=50ms "
            "(set via spec R3/R11)",
            flush=True,
        )
        yield

    app = FastAPI(title="column-review", lifespan=lifespan)
    app.state.config = config

    # Routers are imported lazily so `--help` doesn't pay their import
    # cost (and inference.py's torch+ultralytics chain stays cold until
    # a /api/infer call needs it).
    from column_review.routes import detections as detections_routes
    from column_review.routes import files as files_routes
    from column_review.routes import submit as submit_routes
    from column_review.routes import tiles as tiles_routes
    app.include_router(files_routes.router)
    app.include_router(tiles_routes.router)
    app.include_router(detections_routes.router)
    app.include_router(submit_routes.router)

    # Explicit `/` route so a missing index.html surfaces a 500 with a
    # self-describing body rather than the opaque 404 that
    # `StaticFiles(html=True)` would return after a packaging mishap.
    # Also rewrites `__BUILD__` placeholders to the process build
    # stamp so the browser refetches `/app.js` and `/styles.css` on
    # every server restart (defends against stale-cache failures).
    @app.get("/", include_in_schema=False)
    def index():
        idx = _STATIC_DIR / "index.html"
        if not idx.is_file():
            return HTMLResponse(
                f"<h1>column-review frontend missing</h1>"
                f"<p>Expected at: <code>{idx}</code></p>"
                f"<p>The package data was not installed correctly. "
                f"Re-install with <code>pip install -e .</code> from "
                f"the repo root.</p>",
                status_code=500,
            )
        from column_review import BUILD_STAMP
        html = idx.read_text(encoding="utf-8")
        html = html.replace("__BUILD__", BUILD_STAMP)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, must-revalidate"},
        )

    # Static files mounted LAST so `/api/*`, `/tiles/*`, and `/`
    # take precedence over same-name files under `static/`.
    # `html=False` because the explicit index handler above owns `/`.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=False),
              name="static")

    return app
