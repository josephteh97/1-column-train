"""Library-style modules under `scripts/` consumed by the
`column_review` package and by the ad-hoc CLI scripts that share the
project root. Marker file so setuptools recognises `scripts` as a
top-level package and a non-editable `pip install` ships the modules
that `column_review.db`, `column_review.jobs`, and
`column_review.inference` import (`corrections_logger`,
`ingest_drawings`, `tiled_inference`, `postprocess_pipeline`)."""
