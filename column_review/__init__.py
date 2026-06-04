"""column_review — single-command web reviewer for YOLO column detections.

The public CLI is `column-review` (registered as a console_script in
`pyproject.toml`). Programmatic entry: `column_review.cli.main`.
"""
import time

__version__ = "0.1.0"
# Build stamp used as a cache-busting query on `/styles.css`,
# `/app.js`, etc. Re-stamped at every process startup so a server
# restart guarantees the browser refetches static assets — defends
# against the "old cached app.js" failure mode where users hard-
# refresh but stale code keeps running.
BUILD_STAMP = str(int(time.time()))
