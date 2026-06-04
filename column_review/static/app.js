/* column-review — single-page reviewer UI.
 *
 * Boot path:
 *   DOMContentLoaded
 *     → fetch /api/drawings           (populate picker)
 *     → user picks + clicks Open      (or presses Enter in the picker)
 *     → POST /api/open                (job_id + tile_source URL)
 *     → OpenSeadragon mount           (with the returned tile_source)
 *     → on OSD `open` event           (image is ready)
 *     → fetch /api/detections          (or 412 → show "Run inference")
 *     → installRenderCanary           (R2 regression guard fires here)
 *     → paint overlay + minimap, refresh counts
 *
 * State map: lives on `state` (module-scoped object). Mutation goes
 * through helpers that also trigger paint + count refresh, so we never
 * end up with a state inconsistency the user can see.
 *
 * Mouse model (R5):
 *   left-click on detection    → toggle FP / RESCIND_FP
 *   left-drag in empty space   → draw rubber-band → FN_ADDED on release
 *   middle-drag                → pan (OSD default)
 *   space + left-drag          → pan (we toggle gestureSettingsMouse.dragToPan)
 *   wheel                      → zoom centred on cursor (OSD default)
 *
 * Keyboard model (R4 / D8):
 *   F, X            → toggle FP on active detection
 *   U               → undo
 *   Shift-U, Y      → redo
 *   Enter           → Save & Submit (tranche D wires retrain; today: stub)
 *   0               → 100% zoom
 *   H               → fit-to-window (home)
 *   Z               → zoom-to-active-detection
 *   N, P            → next / previous detection
 *   J               → jump-to-next-unreviewed
 *   Space (held)    → pan modifier on left-drag
 */
"use strict";

/* ──────────────────────────────────────────────────────────────────
 * Global error capture — make any uncaught JS error visible as a
 * fail banner instead of silently breaking the boot path.
 * ────────────────────────────────────────────────────────────────── */

window.addEventListener("error", (e) => {
  const msg = e.message || String(e);
  const where = e.filename ? `${e.filename}:${e.lineno}:${e.colno}` : "?";
  console.error("[uncaught]", msg, where, e.error);
  try {
    document.getElementById("fail-message").textContent =
      `Uncaught JS error: ${msg}\n  at ${where}\n\n` +
      (e.error && e.error.stack ? e.error.stack : "(no stack)");
    document.getElementById("fail-banner").classList.remove("hidden");
  } catch (_) {
    // Fail-banner DOM not built yet — last resort: write to <body>.
    document.body.innerHTML =
      `<pre style="color:#fff;padding:24px;background:#1c1f24">` +
      `Boot error before fail-banner: ${msg} at ${where}\n` +
      `${e.error && e.error.stack ? e.error.stack : ""}</pre>`;
  }
});

window.addEventListener("unhandledrejection", (e) => {
  console.error("[unhandled-promise]", e.reason);
  const fb = document.getElementById("fail-message");
  if (fb) {
    fb.textContent = `Unhandled promise rejection: ${e.reason}`;
    document.getElementById("fail-banner").classList.remove("hidden");
  }
});


/* ──────────────────────────────────────────────────────────────────
 * State
 * ────────────────────────────────────────────────────────────────── */

const state = {
  /** OpenSeadragon viewer instance. Set after first openDrawing(). */
  osd:           null,
  drawingId:     null,
  jobId:         null,
  sessionId:     null,
  reviewerId:    null,

  /** Last detections fetch result. Array of
   *  {element_index, bbox, score, source, state}. */
  detections:    null,

  /** Index of the currently-active detection (white ring) or null. */
  activeIndex:   null,

  /** Image-pixel size from OSD's first source. */
  imageSize:     null,

  /** Track Space-key modifier so left-drag becomes a pan. */
  spaceHeld:     false,

  /** Ongoing FN-drag rubber-band, set by canvas-press / cleared by
   *  canvas-release. */
  pressState:    null,

  /** Tracked rAF id so we can debounce repaints. */
  pendingPaint:  null,

  /** Set after the first /api/detections fetch finishes (any outcome).
   *  Combined with OSD's `open` event to gate the R2 canary. */
  detectionsFetchSettled: false,

  /** Cached palette so we don't getComputedStyle on every box. */
  palette: null,

  /** Memoised detection-count tally. */
  counts:        {unreviewed: 0, fp: 0, fn: 0},

  /** Timestamp (Date.now) of the last successful mark save. The
   *  autosave-pill text refreshes on a setInterval so the human-time
   *  "saved 4s ago" stays current. */
  lastSavedAt:   null,

  /** `performance.now()` snapshot taken at POST /api/open success.
   *  When OSD `open` fires AND /api/detections settles, the elapsed
   *  is POSTed to /api/render-ack so the server can log the open
   *  budget (R3, 3000ms). */
  openStartedAtPerf: null,
  renderAckSent: false,

  /** Retrain status polling — set to a setInterval id while a job
   *  is in a non-terminal state. */
  retrainPollHandle: null,

  /** Last retrain job id for which we surfaced the failure banner.
   *  Used to suppress the banner for already-failed historical jobs
   *  on page load — the pill still shows the failed status, but the
   *  full-viewport banner only fires when a NEW failure transition
   *  is observed in this session. */
  bannerSeenRetrainJobId: null,
};


/* ──────────────────────────────────────────────────────────────────
 * Boot
 * ────────────────────────────────────────────────────────────────── */

window.addEventListener("DOMContentLoaded", boot);

async function boot() {
  try {
    cachePalette();
    await populatePicker();
    await populateLocalImages();
    loadReviewerIdFromStorage();
    wirePicker();
    wireKeyboard();
    wireFailBannerDismiss();
    wireZoomInput();
    wireInferenceButton();
    wireSubmitButton();
    wireSubmitModal();
    wireRetrainFailBanner();
    setInterval(refreshAutosavePill, 1000);
    refreshRetrainPill();
    console.info("[boot] complete");
  } catch (e) {
    showFailBanner(
      `Boot failed: ${e.message}\n\n${e.stack || ""}`);
    throw e;
  }
}


function cachePalette() {
  const css = getComputedStyle(document.documentElement);
  state.palette = {
    unreviewed: css.getPropertyValue("--col-unreviewed").trim(),
    fp:         css.getPropertyValue("--col-fp").trim(),
    fn:         css.getPropertyValue("--col-fn").trim(),
  };
}


async function populatePicker() {
  const select = document.getElementById("drawing-select");
  try {
    const resp = await fetch("/api/drawings");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    select.innerHTML = "";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = data.drawings.length
      ? "— select a drawing —"
      : "— no DZI drawings ingested —";
    select.appendChild(blank);
    for (const id of data.drawings) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = id;
      select.appendChild(opt);
    }
  } catch (e) {
    showFailBanner("Failed to load /api/drawings: " + e.message);
  }
}


async function populateLocalImages() {
  const select = document.getElementById("local-images-select");
  const dirLabel = document.getElementById("local-images-dir");
  try {
    const resp = await fetch("/api/local-images");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    select.innerHTML = "";
    if (!data.exists) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "— no images-dir configured —";
      select.appendChild(opt);
      select.disabled = true;
      dirLabel.textContent = "(use --images-dir <folder>)";
      document.getElementById("local-images-label").classList.add("dim");
      return;
    }
    dirLabel.textContent = "(" + data.images_dir + ")";
    const blank = document.createElement("option");
    blank.value = "";
    blank.textContent = data.images.length
      ? `— pick one of ${data.images.length} files —`
      : "— folder is empty —";
    select.appendChild(blank);
    for (const im of data.images) {
      const opt = document.createElement("option");
      opt.value = im.filename;
      const mb = (im.size_bytes / (1024 * 1024)).toFixed(1);
      opt.textContent = `${im.filename}  (${mb} MB)`;
      select.appendChild(opt);
    }
  } catch (e) {
    console.warn("Failed to load /api/local-images:", e);
    select.innerHTML = "<option value=''>— failed to load —</option>";
    select.disabled = true;
  }
}


function loadReviewerIdFromStorage() {
  const stored = localStorage.getItem("column-review.reviewer_id");
  if (stored) {
    document.getElementById("reviewer-id-input").value = stored;
  }
}


function wirePicker() {
  const openBtn = document.getElementById("open-btn");
  const drawingSelect = document.getElementById("drawing-select");
  const localSelect = document.getElementById("local-images-select");
  const reviewerInput = document.getElementById("reviewer-id-input");

  const trigger = () => {
    const reviewerId = reviewerInput.value.trim();
    if (!reviewerId) { reviewerInput.focus(); return; }
    const localFilename = localSelect ? localSelect.value : "";
    const drawingId = drawingSelect.value;
    if (!localFilename && !drawingId) {
      // Default focus to whichever is non-empty / non-disabled.
      (localSelect && !localSelect.disabled ? localSelect : drawingSelect).focus();
      return;
    }
    localStorage.setItem("column-review.reviewer_id", reviewerId);
    // Local image takes precedence if both are picked — the user
    // explicitly chose a local file last.
    if (localFilename) {
      openLocalImage(localFilename, reviewerId);
    } else {
      openDrawing(drawingId, reviewerId);
    }
  };

  openBtn.addEventListener("click", trigger);
  reviewerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") trigger();
  });
  drawingSelect.addEventListener("keydown", (e) => {
    if (e.key === "Enter") trigger();
  });
  if (localSelect) {
    localSelect.addEventListener("keydown", (e) => {
      if (e.key === "Enter") trigger();
    });
    // Mutual exclusion in the UI — picking one clears the other so
    // it's always obvious which mode you're about to open in.
    localSelect.addEventListener("change", () => {
      if (localSelect.value) drawingSelect.value = "";
    });
    drawingSelect.addEventListener("change", () => {
      if (drawingSelect.value) localSelect.value = "";
    });
  }
}


function wireFailBannerDismiss() {
  document.getElementById("fail-dismiss").addEventListener("click", () => {
    document.getElementById("fail-banner").classList.add("hidden");
  });
}


function wireZoomInput() {
  const input = document.getElementById("zoom-input");
  input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    const pct = parseFloat(input.value);
    if (!isFinite(pct) || pct <= 0) return;
    if (state.osd) {
      // 100 % == imagePixel:devicePixel 1:1
      // OSD: viewport.imageToViewportZoom(1.0) is "100% pixel ratio".
      const targetImageZoom = pct / 100.0;
      state.osd.viewport.zoomTo(
        state.osd.viewport.imageToViewportZoom(targetImageZoom)
      );
    }
  });
}


function wireInferenceButton() {
  const btn = document.getElementById("infer-btn");
  btn.addEventListener("click", async () => {
    if (btn.disabled) return;

    // If model detections already exist, confirm before re-running.
    // FN_ADDED boxes the reviewer drew by hand are preserved through
    // the re-inference. FP marks against the OLD model indices are
    // dropped (they referenced detections that won't survive).
    let force = false;
    if (Array.isArray(state.detections)) {
      const nModel = state.detections.filter(
        (d) => d.source !== "human_added").length;
      const nFn = state.detections.filter(
        (d) => d.source === "human_added").length;
      if (nModel > 0) {
        const ok = confirm(
          `Re-run YOLO with column_detect.pt?\n\n` +
          `This will:\n` +
          `  • drop ${nModel} existing model detections\n` +
          `  • drop FP marks against those (they reference indices ` +
          `that won't survive)\n` +
          `  • preserve ${nFn} hand-drawn FN_ADDED boxes\n\n` +
          `Continue?`);
        if (!ok) return;
        force = true;
      }
    }

    btn.disabled = true;
    btn.querySelector(".spinner").classList.remove("hidden");
    try {
      const resp = await fetch("/api/infer", {
        method:  "POST",
        headers: {"Content-Type": "application/json"},
        body:    JSON.stringify({
          job_id:     state.jobId,
          drawing_id: state.drawingId,
          force:      force,
        }),
      });
      if (!resp.ok) {
        const detail = await safeJson(resp);
        showFailBanner(
          `/api/infer failed (HTTP ${resp.status}): ` +
          (detail?.detail || JSON.stringify(detail) || "no body"));
        return;
      }
      const result = await resp.json();
      console.info("[infer] complete", result);
      await refetchDetections();
    } finally {
      btn.disabled = false;
      btn.querySelector(".spinner").classList.add("hidden");
    }
  });
}


function wireSubmitButton() {
  document.getElementById("submit-btn").addEventListener(
    "click", triggerSubmit);
}


/* ──────────────────────────────────────────────────────────────────
 * Open a drawing → mount OSD → fetch detections → paint.
 * ────────────────────────────────────────────────────────────────── */

async function openDrawing(drawingId, reviewerId) {
  let openResponse;
  try {
    const resp = await fetch("/api/open", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({
        drawing_id: drawingId, reviewer_id: reviewerId,
      }),
    });
    if (!resp.ok) {
      const body = await safeJson(resp);
      const detail = body?.detail;
      // The detail may be a string or a structured object
      // ({error, drawing_id, hint}) — render both shapes.
      const msg = typeof detail === "string"
        ? detail
        : (detail?.hint
            ? `${detail.error || "open failed"}\n${detail.hint}`
            : JSON.stringify(body));
      showFailBanner(`Failed to open ${drawingId}:\n${msg}`);
      return;
    }
    openResponse = await resp.json();
  } catch (e) {
    showFailBanner("POST /api/open failed: " + e.message);
    return;
  }

  state.drawingId   = openResponse.drawing_id;
  state.jobId       = openResponse.job_id;
  state.sessionId   = openResponse.session_id;
  state.reviewerId  = openResponse.reviewer_id;
  // R3 timer starts here. sendRenderAckIfReady() POSTs the elapsed
  // when both OSD `open` and the detections fetch have settled.
  state.openStartedAtPerf = performance.now();
  state.renderAckSent     = false;
  document.getElementById("drawing-id-label").textContent =
    state.drawingId + " · " + state.reviewerId;
  document.getElementById("picker-drawer").classList.add("hidden");
  document.getElementById("submit-btn").classList.remove("hidden");

  // tile_source_type signals OSD's mount mode: "image" → load the raw
  // PNG/JPG directly; default (undefined / "dzi") → tile pyramid.
  mountOsd(openResponse.tile_source, openResponse.tile_source_type);
}


async function openLocalImage(filename, reviewerId) {
  let openResponse;
  try {
    const resp = await fetch("/api/open-local-image", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({
        filename: filename, reviewer_id: reviewerId,
      }),
    });
    if (!resp.ok) {
      const body = await safeJson(resp);
      const detail = typeof body?.detail === "string"
        ? body.detail
        : JSON.stringify(body);
      showFailBanner(`Failed to open ${filename}:\n${detail}`);
      return;
    }
    openResponse = await resp.json();
  } catch (e) {
    showFailBanner("POST /api/open-local-image failed: " + e.message);
    return;
  }

  state.drawingId   = openResponse.drawing_id;
  state.jobId       = openResponse.job_id;
  state.sessionId   = openResponse.session_id;
  state.reviewerId  = openResponse.reviewer_id;
  state.openStartedAtPerf = performance.now();
  state.renderAckSent     = false;
  document.getElementById("drawing-id-label").textContent =
    state.drawingId + " · " + state.reviewerId;
  document.getElementById("picker-drawer").classList.add("hidden");
  document.getElementById("submit-btn").classList.remove("hidden");
  mountOsd(openResponse.tile_source, openResponse.tile_source_type);
}


function mountOsd(tileSourceUrl, tileSourceType) {
  if (typeof OpenSeadragon === "undefined") {
    showFailBanner(
      "OpenSeadragon library failed to load.\n\n" +
      "/vendor/openseadragon.min.js did not register a global " +
      "`OpenSeadragon` function. Check the browser network tab — " +
      "did the vendor script return 200 OK with the right body?");
    return;
  }
  if (state.osd) {
    try { state.osd.destroy(); } catch (_) {}
    state.osd = null;
  }
  const viewerEl = document.getElementById("viewer");
  if (!viewerEl) {
    showFailBanner(
      "DOM mismatch: #viewer element not found. The index.html " +
      "shell did not render correctly.");
    return;
  }
  // OSD's tileSources can be either a string URL (DZI) OR a config
  // object. For type="image" (plain PNG/JPG, no tile pyramid) we pass
  // a config object so OSD loads the whole image in one shot.
  const osdTileSources = (tileSourceType === "image")
    ? { type: "image", url: tileSourceUrl, buildPyramid: false }
    : tileSourceUrl;
  try {
    state.osd = OpenSeadragon({
      element:               viewerEl,
      tileSources:           osdTileSources,
      prefixUrl:             "/vendor/openseadragon/images/", // unused (no controls)
      showNavigationControl: false,
      showFullPageControl:   false,
      showHomeControl:       false,
      showZoomControl:       false,
      showRotationControl:   false,
      // Leave OSD's defaults for visibility/min-zoom alone — earlier
      // explicit values (visibilityRatio: 0.4 + minZoomImageRatio:
      // 0.2 OR 1.0 + 0.8) both produced edge-case home-zoom states
      // where the image rendered offscreen or at near-zero zoom.
      // The default values match what the smoke-test pages used,
      // which rendered the floor plan correctly. `goHome(true)` in
      // the onOsdOpen handler then explicitly forces fit-to-window.
      maxZoomPixelRatio:     8,
      immediateRender:       false,
      preserveImageSizeOnResize: true,
      imageLoaderLimit:      8,
      // R3: bounded tile cache with LRU.
      maxImageCacheCount:    1024,
      gestureSettingsMouse: {
        // R5: left-drag is OURS (FN draw). Pan is middle-drag or Space+left.
        dragToPan:       false,
        clickToZoom:     false,
        dblClickToZoom:  false,
        pinchToZoom:     true,
        scrollToZoom:    true, // wheel zooms on cursor (R5)
        flickEnabled:    false,
      },
    });
  } catch (e) {
    showFailBanner(
      `OpenSeadragon construction threw:\n  ${e.message}\n\n` +
      `tileSources URL: ${tileSourceUrl}\n` +
      (e.stack || ""));
    return;
  }

  // Capture OSD's own internal error events so a tile-load failure
  // surfaces here, not silently.
  state.osd.addHandler("open-failed", (ev) => {
    showFailBanner(
      `OpenSeadragon failed to open the tile source.\n\n` +
      `Source: ${tileSourceUrl}\n` +
      `Message: ${ev.message || "(no message)"}\n` +
      `Source type: ${ev.source && ev.source.type || "?"}`);
  });
  state.osd.addHandler("tile-load-failed", (ev) => {
    console.warn("[osd] tile-load-failed", ev);
  });

  state.osd.addHandler("open", onOsdOpen);
  state.osd.addHandler("update-viewport", schedulePaint);
  state.osd.addHandler("canvas-press",   onCanvasPress);
  state.osd.addHandler("canvas-drag",    onCanvasDrag);
  state.osd.addHandler("canvas-release", onCanvasRelease);
  state.osd.addHandler("canvas-click",   onCanvasClick);

  // ResizeObserver — fires whenever #viewer's box changes (initial
  // layout, window resize, picker-drawer hide, etc.). On every fire
  // we tell OSD's viewport about the new size AND force a fit. This
  // is the safety net against any timing race where OSD captured
  // a 0-size container at construction.
  if (typeof ResizeObserver === "function") {
    const ro = new ResizeObserver((entries) => {
      for (const ent of entries) {
        if (ent.contentRect.width > 0 && ent.contentRect.height > 0) {
          try {
            if (state.osd && state.osd.viewport) {
              state.osd.viewport.resize();
              if (!state._didFirstFit) {
                state.osd.viewport.goHome(true);
                state._didFirstFit = true;
              }
            }
          } catch (e) {
            console.warn("[osd] ResizeObserver fit failed:", e);
          }
        }
      }
    });
    ro.observe(viewerEl);
    state._resizeObserver = ro;
  }

  installRenderCanary();
}


async function onOsdOpen() {
  const item = state.osd.world.getItemAt(0);
  if (item) {
    const size = item.getContentSize();
    state.imageSize = {width: size.x, height: size.y};
  }
  resizeOverlay();
  // Multiple fit-to-window retries so race conditions where layout
  // settles AFTER OSD `open` fires can still recover. Each retry
  // calls `viewport.resize()` first (forces OSD to re-read the
  // container's current bounding rect) then `goHome(true)` (snaps
  // to the now-correct fit position). 0/50/250 ms covers immediate,
  // post-rAF, and well-after-layout timing.
  const refit = () => {
    if (!state.osd || !state.osd.viewport) return;
    try {
      state.osd.viewport.resize();
      state.osd.viewport.goHome(true);
    } catch (e) {
      console.warn("[osd] refit failed:", e);
    }
  };
  refit();
  setTimeout(refit, 50);
  setTimeout(refit, 250);
  await refetchDetections();
}


async function refetchDetections() {
  try {
    const resp = await fetch(
      `/api/detections?job_id=${encodeURIComponent(state.jobId)}`);
    if (resp.status === 412) {
      // Legitimate empty state — no inference yet.
      state.detections = [];
      document.getElementById("infer-btn").classList.remove("hidden");
      state.detectionsFetchSettled = true;
      refreshCounts();
      schedulePaint();
      maybeFireCanary();
      return;
    }
    if (!resp.ok) {
      const body = await safeJson(resp);
      showFailBanner(
        `GET /api/detections failed (HTTP ${resp.status}): ` +
        JSON.stringify(body));
      return;
    }
    const data = await resp.json();
    state.detections = data.detections;
    // Run-YOLO button stays visible whether or not detections exist —
    // the user can re-run inference at any time (the button's click
    // handler asks for confirmation when it would replace existing
    // model detections).
    document.getElementById("infer-btn").classList.remove("hidden");
    state.detectionsFetchSettled = true;
    refreshCounts();
    schedulePaint();
    maybeFireCanary();
  } catch (e) {
    showFailBanner("GET /api/detections failed: " + e.message);
  }
}


/* ──────────────────────────────────────────────────────────────────
 * Painting — overlay canvas + minimap.
 * ────────────────────────────────────────────────────────────────── */

const overlayCanvas = () => document.getElementById("overlay-canvas");
const minimapCanvas = () => document.getElementById("minimap-canvas");


function resizeOverlay() {
  // #viewer is now the direct grid row that holds OSD's canvas and
  // the overlay-canvas as children. Match its bounding rect so the
  // overlay coords line up 1:1 with OSD's world-to-screen transform.
  const wrap = document.getElementById("viewer");
  const canvas = overlayCanvas();
  const dpr = window.devicePixelRatio || 1;
  const rect = wrap.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  canvas.style.width  = rect.width + "px";
  canvas.style.height = rect.height + "px";
}


window.addEventListener("resize", () => {
  if (state.osd) resizeOverlay();
  schedulePaint();
});


function schedulePaint() {
  if (state.pendingPaint) return;
  state.pendingPaint = requestAnimationFrame(() => {
    state.pendingPaint = null;
    paintOverlay();
    paintMinimap();
    updateZoomReadout();
  });
}


function paintOverlay() {
  const canvas = overlayCanvas();
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width / dpr;
  const H = canvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  if (!state.osd || !state.detections) return;

  const viewport = state.osd.viewport;
  const zoom = viewport.getZoom(true);
  const strokeScale = Math.max(1.0, Math.min(4.0, 0.5 / Math.max(zoom, 0.01)));

  for (let i = 0; i < state.detections.length; i++) {
    const det = state.detections[i];
    if (det.state === "REMOVED") continue;
    const bbox = det.bbox;
    if (!bbox || bbox.length < 4) continue;
    const tl = viewport.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(bbox[0], bbox[1]));
    const br = viewport.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(bbox[2], bbox[3]));
    const sx = Math.min(tl.x, br.x);
    const sy = Math.min(tl.y, br.y);
    const sw = Math.abs(br.x - tl.x);
    const sh = Math.abs(br.y - tl.y);
    if (sx + sw < 0 || sy + sh < 0 || sx > W || sy > H) continue;

    let color, dash;
    switch (det.state) {
      case "FP":
        color = state.palette.fp;
        dash = [Math.max(2, 6 * strokeScale), Math.max(2, 4 * strokeScale)];
        break;
      case "FN_ADDED":
        color = state.palette.fn;
        dash = [Math.max(1, 2 * strokeScale), Math.max(2, 3 * strokeScale)];
        break;
      default: // UNREVIEWED
        color = state.palette.unreviewed;
        dash = [];
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5 * strokeScale;
    ctx.setLineDash(dash);
    ctx.strokeRect(sx, sy, sw, sh);

    if (i === state.activeIndex) {
      ctx.strokeStyle = "#ffffff";
      ctx.setLineDash([]);
      ctx.lineWidth = 4 * strokeScale;
      ctx.strokeRect(sx - 2, sy - 2, sw + 4, sh + 4);
    }
  }

  // FN drag rubber-band.
  if (state.pressState && state.pressState.hasMoved) {
    const s = state.pressState.startImage;
    const c = state.pressState.currentImage;
    const tl = viewport.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(Math.min(s.x, c.x), Math.min(s.y, c.y)));
    const br = viewport.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(Math.max(s.x, c.x), Math.max(s.y, c.y)));
    ctx.strokeStyle = state.palette.fn;
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
    ctx.setLineDash([]);
  }
}


function paintMinimap() {
  const canvas = minimapCanvas();
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!state.osd || !state.imageSize) return;

  const imgW = state.imageSize.width;
  const imgH = state.imageSize.height;
  const scale = Math.min(W / imgW, H / imgH);
  const drawW = imgW * scale;
  const drawH = imgH * scale;
  const offX = (W - drawW) / 2;
  const offY = (H - drawH) / 2;

  ctx.fillStyle = "#1c1f24";
  ctx.fillRect(offX, offY, drawW, drawH);

  if (state.detections) {
    for (const det of state.detections) {
      if (det.state === "REMOVED") continue;
      if (!det.bbox || det.bbox.length < 4) continue;
      const cx = (det.bbox[0] + det.bbox[2]) / 2;
      const cy = (det.bbox[1] + det.bbox[3]) / 2;
      const px = offX + cx * scale;
      const py = offY + cy * scale;
      let color;
      switch (det.state) {
        case "FP":       color = state.palette.fp; break;
        case "FN_ADDED": color = state.palette.fn; break;
        default:         color = state.palette.unreviewed;
      }
      ctx.fillStyle = color;
      ctx.fillRect(px - 1, py - 1, 3, 3);
    }
  }

  // Viewport rectangle.
  try {
    const bounds = state.osd.viewport.viewportToImageRectangle(
      state.osd.viewport.getBounds(true));
    const vx = offX + bounds.x * scale;
    const vy = offY + bounds.y * scale;
    const vw = bounds.width * scale;
    const vh = bounds.height * scale;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.strokeRect(vx, vy, vw, vh);
  } catch (_) { /* viewport may not be ready yet on first frame */ }
}


function updateZoomReadout() {
  const input = document.getElementById("zoom-input");
  if (document.activeElement === input) return;
  if (!state.osd) return;
  const pct =
    state.osd.viewport.viewportToImageZoom(
      state.osd.viewport.getZoom(true)) * 100.0;
  input.value = pct.toFixed(0);
}


/* ──────────────────────────────────────────────────────────────────
 * Counts / state book-keeping.
 * ────────────────────────────────────────────────────────────────── */

function refreshCounts() {
  state.counts = {unreviewed: 0, fp: 0, fn: 0};
  if (state.detections) {
    for (const det of state.detections) {
      switch (det.state) {
        case "FP":       state.counts.fp++; break;
        case "FN_ADDED": state.counts.fn++; break;
        case "REMOVED":  break;
        default:         state.counts.unreviewed++;
      }
    }
  }
  document.getElementById("count-unreviewed").textContent = state.counts.unreviewed;
  document.getElementById("count-fp").textContent         = state.counts.fp;
  document.getElementById("count-fn").textContent         = state.counts.fn;
}


function applyStates(stateMap) {
  if (!state.detections || !stateMap) return;
  // If the server returned a state for an index we don't have (e.g.,
  // FN_ADDED extended the columns list), refetch the full list.
  let needRefetch = false;
  for (const k of Object.keys(stateMap)) {
    const idx = Number(k);
    if (idx >= state.detections.length) { needRefetch = true; break; }
    state.detections[idx].state = stateMap[k];
  }
  if (needRefetch) {
    refetchDetections();
  } else {
    refreshCounts();
    schedulePaint();
  }
}


/* ──────────────────────────────────────────────────────────────────
 * Mouse — canvas-press / canvas-drag / canvas-release / canvas-click.
 * ────────────────────────────────────────────────────────────────── */

function onCanvasPress(ev) {
  // Space-held → mark this press as pan-mode. canvas-drag drives
  // panBy directly using `ev.delta`, sidestepping the question of
  // whether mutating `gestureSettingsMouse.dragToPan` after OSD
  // construction takes effect (some OSD versions snapshot the
  // settings; this approach works on any version).
  if (state.spaceHeld) {
    state.pressState = {mode: "pan"};
    return;
  }
  const imgPt = state.osd.viewport.viewerElementToImageCoordinates(ev.position);
  state.pressState = {
    mode:               "fn",
    startImage:         {x: imgPt.x, y: imgPt.y},
    currentImage:       {x: imgPt.x, y: imgPt.y},
    hasMoved:           false,
    startedOnDetection: findHitDetection(imgPt) >= 0,
  };
}


function onCanvasDrag(ev) {
  if (!state.pressState) return;
  if (state.pressState.mode === "pan") {
    // `ev.delta` is the per-event pixel movement in viewer-element
    // space. `deltaPointsFromPixels` converts to viewport-space; the
    // negation moves the WORLD in the opposite direction, i.e. the
    // user's hand drags the image (not the camera).
    const dvp = state.osd.viewport.deltaPointsFromPixels(
      new OpenSeadragon.Point(-ev.delta.x, -ev.delta.y));
    state.osd.viewport.panBy(dvp);
    return;
  }
  if (state.pressState.startedOnDetection) return; // detection-drag is a no-op for us
  const imgPt = state.osd.viewport.viewerElementToImageCoordinates(ev.position);
  state.pressState.currentImage = {x: imgPt.x, y: imgPt.y};
  const dx = imgPt.x - state.pressState.startImage.x;
  const dy = imgPt.y - state.pressState.startImage.y;
  if (Math.hypot(dx, dy) > 4) {
    state.pressState.hasMoved = true;
  }
  schedulePaint();
}


function onCanvasRelease(ev) {
  const p = state.pressState;
  state.pressState = null;
  if (!p) return;
  if (p.mode === "pan") return;
  if (!p.hasMoved) return; // a quick click will be handled by canvas-click

  const s = p.startImage;
  const c = p.currentImage;
  const x1 = Math.min(s.x, c.x);
  const y1 = Math.min(s.y, c.y);
  const x2 = Math.max(s.x, c.x);
  const y2 = Math.max(s.y, c.y);
  if ((x2 - x1) < 4 || (y2 - y1) < 4) { schedulePaint(); return; }
  postMark({action: "FN_ADDED", bbox: [x1, y1, x2, y2]});
}


function onCanvasClick(ev) {
  if (!ev.quick) return; // end-of-drag click — already handled in release
  if (state.spaceHeld) return;
  const imgPt = state.osd.viewport.viewerElementToImageCoordinates(ev.position);
  const idx = findHitDetection(imgPt);
  if (idx < 0) {
    state.activeIndex = null;
    schedulePaint();
    return;
  }
  state.activeIndex = idx;
  const det = state.detections[idx];
  const action = det.state === "FP" ? "RESCIND_FP" : "FP";
  postMark({action, element_index: det.element_index});
}


function findHitDetection(imgPt) {
  if (!state.detections) return -1;
  const zoom = state.osd.viewport.getZoom(true);
  // Tolerance in image-pixel units. At low zoom, expand so tiny boxes
  // remain clickable (R8). Floor 8 image-px ≈ 8/14000 ≈ 0.06% of width.
  const tol = Math.max(8, 8 / Math.max(zoom, 0.01));
  // Topmost first — most recently added FN sits at the end of the list
  // and should beat an underlying model detection on a tie.
  for (let i = state.detections.length - 1; i >= 0; i--) {
    const det = state.detections[i];
    if (det.state === "REMOVED") continue;
    const b = det.bbox;
    if (!b || b.length < 4) continue;
    if (imgPt.x >= b[0] - tol && imgPt.x <= b[2] + tol
        && imgPt.y >= b[1] - tol && imgPt.y <= b[3] + tol) {
      return i;
    }
  }
  return -1;
}


/* ──────────────────────────────────────────────────────────────────
 * Keyboard map.
 * ────────────────────────────────────────────────────────────────── */

function wireKeyboard() {
  window.addEventListener("keydown", onKeyDown, {capture: true});
  window.addEventListener("keyup",   onKeyUp,   {capture: true});
}


function onKeyDown(e) {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT"
            || t.tagName === "TEXTAREA")) {
    return;
  }
  if (e.code === "Space") {
    if (!state.spaceHeld) {
      state.spaceHeld = true;
      // No gestureSettings mutation — onCanvasPress/onCanvasDrag check
      // `state.spaceHeld` directly and drive panBy. Works regardless
      // of whether the vendored OSD snapshots gesture settings.
    }
    e.preventDefault();
    return;
  }
  const k = e.key.toLowerCase();
  switch (k) {
    case "f": case "x":
      if (state.activeIndex != null) {
        toggleActiveFP();
        e.preventDefault();
      }
      break;
    case "u":
      if (e.shiftKey) postRedo(); else postUndo();
      e.preventDefault();
      break;
    case "y":
      postRedo(); e.preventDefault(); break;
    case "enter":
      triggerSubmit(); e.preventDefault(); break;
    case "0":
      if (state.osd) state.osd.viewport.zoomTo(
        state.osd.viewport.imageToViewportZoom(1.0));
      e.preventDefault(); break;
    case "h":
      if (state.osd) state.osd.viewport.goHome();
      e.preventDefault(); break;
    case "z":
      zoomToActive(); e.preventDefault(); break;
    case "n":
      stepDetection(+1); e.preventDefault(); break;
    case "p":
      stepDetection(-1); e.preventDefault(); break;
    case "j":
      jumpToNextUnreviewed(+1); e.preventDefault(); break;
  }
}


function onKeyUp(e) {
  if (e.code === "Space") {
    state.spaceHeld = false;
  }
}


/* ──────────────────────────────────────────────────────────────────
 * Mark/undo/redo POSTs.
 * ────────────────────────────────────────────────────────────────── */

async function postMark(payload) {
  if (!state.jobId || !state.sessionId) return;
  const body = {
    job_id:     state.jobId,
    session_id: state.sessionId,
    ...payload,
  };
  try {
    const resp = await fetch("/api/marks", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify(body),
    });
    if (!resp.ok) {
      const detail = await safeJson(resp);
      console.error("/api/marks failed", resp.status, detail);
      return;
    }
    const data = await resp.json();
    // Server returns `new_detection` only when FN_ADDED grew the
    // columns list — append it locally so paintOverlay can render
    // without a full /api/detections refetch.
    if (data.new_detection) {
      state.detections.push(data.new_detection);
    }
    applyStates(data.states);
    setSaveAck(data.elapsed_ms);
  } catch (e) {
    console.error("/api/marks network error", e);
  }
}


async function postUndo() {
  if (!state.jobId || !state.sessionId) return;
  try {
    const resp = await fetch("/api/undo", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({job_id: state.jobId,
                                session_id: state.sessionId}),
    });
    const data = await resp.json();
    if (!data.ok) return; // empty stack
    // With the DELETE_FN→REMOVED + redo-of-FN→RESTORE_FN fixes, undo
    // and redo NEVER change cols.length — only state. applyStates
    // alone keeps the UI in sync, no full refetch needed.
    applyStates(data.states);
    setSaveAck(0);
  } catch (e) {
    console.error("/api/undo error", e);
  }
}


async function postRedo() {
  if (!state.jobId || !state.sessionId) return;
  try {
    const resp = await fetch("/api/redo", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({job_id: state.jobId,
                                session_id: state.sessionId}),
    });
    const data = await resp.json();
    if (!data.ok) return;
    applyStates(data.states);
    setSaveAck(0);
  } catch (e) {
    console.error("/api/redo error", e);
  }
}


function toggleActiveFP() {
  if (state.activeIndex == null) return;
  const det = state.detections[state.activeIndex];
  const action = det.state === "FP" ? "RESCIND_FP" : "FP";
  postMark({action, element_index: det.element_index});
}


function triggerSubmit() {
  // Tranche D wires the confirm dialog + retrain subprocess. For tranche
  // C this is a stub that surfaces the count and reminds the user that
  // autosave is already done.
  if (!state.detections) return;
  const total = state.counts.fp + state.counts.fn;
  alert(
    `Autosave is already on — your ${total} corrections ` +
    `(${state.counts.fp} FP, ${state.counts.fn} FN) ` +
    `are persisted in data/corrections.db.\n\n` +
    `Retrain trigger is wired in the next tranche.`);
}


/* ──────────────────────────────────────────────────────────────────
 * Navigation helpers.
 * ────────────────────────────────────────────────────────────────── */

function stepDetection(direction) {
  if (!state.detections || state.detections.length === 0) return;
  const N = state.detections.length;
  const start = state.activeIndex == null
    ? (direction > 0 ? -1 : 0)
    : state.activeIndex;
  for (let k = 1; k <= N; k++) {
    const idx = (start + direction * k + N * 100) % N;
    const det = state.detections[idx];
    if (det.state === "REMOVED") continue;
    state.activeIndex = idx;
    panToIndex(idx);
    return;
  }
}


function jumpToNextUnreviewed(direction = +1) {
  if (!state.detections || state.detections.length === 0) return;
  const N = state.detections.length;
  const start = state.activeIndex == null
    ? (direction > 0 ? -1 : 0)
    : state.activeIndex;
  for (let k = 1; k <= N; k++) {
    const idx = (start + direction * k + N * 100) % N;
    const det = state.detections[idx];
    if (det.state !== "UNREVIEWED") continue;
    state.activeIndex = idx;
    panToIndex(idx);
    return;
  }
  // No UNREVIEWED left — leave activeIndex unchanged.
}


function panToIndex(idx) {
  if (!state.osd) return;
  const det = state.detections[idx];
  if (!det || !det.bbox || det.bbox.length < 4) return;
  const cx = (det.bbox[0] + det.bbox[2]) / 2;
  const cy = (det.bbox[1] + det.bbox[3]) / 2;
  const vpPoint = state.osd.viewport.imageToViewportCoordinates(
    new OpenSeadragon.Point(cx, cy));
  state.osd.viewport.panTo(vpPoint, false);
  // Zoom to a comfortable level if currently zoomed out.
  const currentZoom = state.osd.viewport.getZoom(true);
  const tenXImageZoom = state.osd.viewport.imageToViewportZoom(1.5);
  if (currentZoom < tenXImageZoom) {
    state.osd.viewport.zoomTo(tenXImageZoom, vpPoint, false);
  }
  schedulePaint();
}


function zoomToActive() {
  if (state.activeIndex == null) return;
  panToIndex(state.activeIndex);
}


/* ──────────────────────────────────────────────────────────────────
 * R2 regression guard (the canary).
 * ────────────────────────────────────────────────────────────────── */

let canaryOsdOpenFired = false;
let canaryTimeoutHandle = null;

function installRenderCanary() {
  canaryOsdOpenFired = false;
  state.detectionsFetchSettled = false;
  if (canaryTimeoutHandle) clearTimeout(canaryTimeoutHandle);
  state.osd.addHandler("open", () => {
    canaryOsdOpenFired = true;
    maybeFireCanary();
  });
  // Timeout safeguard: if OSD's `open` event never fires (DZI parse
  // failed, wrong MIME, network blip, etc.) the canary would never
  // run and the user would see the silent-blank-canvas mode that
  // motivated this rewrite. After 8s, force-fire the canary so
  // the fail banner shows the diagnostic.
  canaryTimeoutHandle = setTimeout(() => {
    if (!canaryOsdOpenFired) {
      showFailBanner(
        "Renderer state inconsistent: OSD `open` event did not fire " +
        "within 8 seconds.\n\n" +
        "Likely causes:\n" +
        "  • The DZI tile pyramid at /tiles/" + state.drawingId +
        ".dzi was not served as XML.\n" +
        "  • OSD failed to parse the DZI manifest.\n" +
        "  • A network or CORS error blocked tile fetches.\n\n" +
        "Check the browser DevTools network tab and the server " +
        "stdout for tile-fetch errors.");
    }
  }, 8000);
}


function maybeFireCanary() {
  if (!canaryOsdOpenFired || !state.detectionsFetchSettled) return;
  // Both signals settled — fire the perf-budget ack regardless of
  // canary outcome. Even a fail-banner-y open is informative for
  // the perf log.
  sendRenderAckIfReady();
  requestAnimationFrame(() => {
    const hasImage = state.osd && state.osd.world.getItemCount() > 0;
    const hasDetections = Array.isArray(state.detections);
    if (!hasImage) {
      showFailBanner(
        "Renderer state inconsistent: image missing.\n\n" +
        "The DZI tile pyramid did not register with OpenSeadragon. " +
        "Check that /tiles/" + state.drawingId + ".dzi returns 200 " +
        "and that data/raw/drawings/" + state.drawingId +
        "_files/ exists.");
      return;
    }
    if (!hasDetections) {
      showFailBanner(
        "Renderer state inconsistent: detections missing.\n\n" +
        "GET /api/detections did not return an array. " +
        "Check the server stdout for a traceback.");
      return;
    }
    // Legitimate empty state is NOT a failure — the "Run inference"
    // button covers it. Only an inconsistency triggers the banner.
  });
}


/* ──────────────────────────────────────────────────────────────────
 * Utilities.
 * ────────────────────────────────────────────────────────────────── */

async function safeJson(resp) {
  try { return await resp.json(); } catch (_) { return null; }
}


function showFailBanner(message) {
  document.getElementById("fail-message").textContent = message;
  document.getElementById("fail-banner").classList.remove("hidden");
}


/* ──────────────────────────────────────────────────────────────────
 * Autosave pill (R10) — live "saved Ns ago" indicator.
 * ────────────────────────────────────────────────────────────────── */

function setSaveAck(_elapsedMs) {
  state.lastSavedAt = Date.now();
  const pill = document.getElementById("autosave-pill");
  pill.classList.remove("hidden");
  // Brief orange flash on the dot so the eye registers each save.
  pill.classList.add("flash");
  clearTimeout(setSaveAck._flashTimer);
  setSaveAck._flashTimer = setTimeout(
    () => pill.classList.remove("flash"), 200);
  refreshAutosavePill();
}


function refreshAutosavePill() {
  if (state.lastSavedAt == null) return;
  const ago = Math.max(0, Math.round((Date.now() - state.lastSavedAt) / 1000));
  const txt = document.getElementById("autosave-text");
  if (ago < 1)        txt.textContent = "saved just now";
  else if (ago < 60)  txt.textContent = `saved ${ago}s ago`;
  else if (ago < 3600) txt.textContent = `saved ${Math.floor(ago / 60)}m ago`;
  else                 txt.textContent = `saved ${Math.floor(ago / 3600)}h ago`;
}


/* ──────────────────────────────────────────────────────────────────
 * Save & Submit confirm modal (R12, part 1).
 * ────────────────────────────────────────────────────────────────── */

function wireSubmitModal() {
  document.getElementById("submit-cancel").addEventListener(
    "click", hideSubmitModal);
  document.getElementById("submit-confirm").addEventListener(
    "click", confirmSubmit);
}


function hideSubmitModal() {
  document.getElementById("submit-modal").classList.add("hidden");
}


function showSubmitModal(previewText) {
  document.getElementById("submit-preview").textContent = previewText;
  document.getElementById("submit-modal").classList.remove("hidden");
}


async function triggerSubmit() {
  // Replaces the tranche-C stub. Two-step:
  //   step 1: POST /api/submit confirm=false → server returns preview
  //   step 2: user clicks Confirm → POST /api/submit confirm=true →
  //           server spawns retrain subprocess
  if (!state.jobId || !state.sessionId) return;
  try {
    const resp = await fetch("/api/submit", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({
        job_id:     state.jobId,
        session_id: state.sessionId,
        confirm:    false,
      }),
    });
    if (resp.status === 412) {
      const body = await safeJson(resp);
      const d = body?.detail || {};
      const hint = d.hint || `Need ${d.needed} corrections; have ${d.have}.`;
      showFailBanner("Submit refused — " + hint);
      return;
    }
    if (!resp.ok) {
      const detail = await safeJson(resp);
      showFailBanner(`Submit preview failed (HTTP ${resp.status}): ` +
                     JSON.stringify(detail));
      return;
    }
    const data = await resp.json();
    showSubmitModal(
      `Effective corrections:\n` +
      `  ${data.n_fp} FP · ${data.n_fn_added} FN_ADDED ` +
      `(${data.n_total} total)\n\n` +
      `Will spawn:\n  ${data.command}\n\n` +
      `Projected runtime: ${data.projected_runtime_estimate}\n\n` +
      `Autosave is on — your corrections are already in ` +
      `data/corrections.db. Confirm only spawns retraining.`);
  } catch (e) {
    showFailBanner("/api/submit network error: " + e.message);
  }
}


async function confirmSubmit() {
  hideSubmitModal();
  try {
    const resp = await fetch("/api/submit", {
      method:  "POST",
      headers: {"Content-Type": "application/json"},
      body:    JSON.stringify({
        job_id:     state.jobId,
        session_id: state.sessionId,
        confirm:    true,
      }),
    });
    if (!resp.ok) {
      const detail = await safeJson(resp);
      showFailBanner(`Submit failed (HTTP ${resp.status}): ` +
                     JSON.stringify(detail));
      return;
    }
    const data = await resp.json();
    console.info("[retrain] spawned", data.retrain_job);
    // Start polling immediately so the pill flips to queued/running
    // within ~2 s.
    refreshRetrainPill(/*forceShow=*/true);
  } catch (e) {
    showFailBanner("/api/submit (confirm) network error: " + e.message);
  }
}


/* ──────────────────────────────────────────────────────────────────
 * Retrain status pill (R12, part 2).
 * ────────────────────────────────────────────────────────────────── */

async function refreshRetrainPill(forceShow = false) {
  try {
    const resp = await fetch("/api/jobs/latest");
    if (!resp.ok) return;
    const data = await resp.json();
    const job = data.job;
    if (!job) {
      // No retrain ever fired in this DB — keep the pill hidden.
      if (!forceShow) return;
    } else if (state.bannerSeenRetrainJobId === null) {
      // First poll of this session — silently absorb whatever job is
      // currently latest so we don't flash a "Retrain failed" banner
      // for a result the reviewer already saw in a previous session.
      // The pill itself still updates so the historical state is
      // visible, but the full-viewport banner is suppressed.
      state.bannerSeenRetrainJobId = job.id;
    }
    renderRetrainPill(job);
  } catch (e) {
    console.warn("/api/jobs/latest error", e);
  }
}


function renderRetrainPill(job) {
  const pill = document.getElementById("retrain-pill");
  const txt = document.getElementById("retrain-text");
  if (!job) {
    pill.className = "";
    pill.classList.add("hidden");
    return;
  }
  pill.classList.remove("hidden");
  // Reset state class, add the new one.
  pill.className = "";
  pill.classList.add(job.status);
  const elapsed = job.finished_ts
    ? Math.round(job.finished_ts - job.started_ts)
    : Math.round(Date.now() / 1000 - job.started_ts);
  switch (job.status) {
    case "queued":
      txt.textContent = "retrain queued";
      break;
    case "running":
      txt.textContent = `retrain running · ${elapsed}s`;
      break;
    case "completed":
      txt.textContent = `retrain completed · ${elapsed}s`;
      break;
    case "failed":
      txt.textContent = `retrain failed · ${elapsed}s`;
      // Only surface the full-viewport banner when THIS job is new
      // since the page loaded — historical failures (e.g. orphaned
      // jobs from a prior session) are shown via the pill only.
      if (state.bannerSeenRetrainJobId !== job.id) {
        state.bannerSeenRetrainJobId = job.id;
        showRetrainFailBanner(job.stderr_tail || "(no stderr captured)");
      }
      break;
  }
  // Keep polling while non-terminal.
  if (state.retrainPollHandle) {
    clearTimeout(state.retrainPollHandle);
    state.retrainPollHandle = null;
  }
  if (job.status === "queued" || job.status === "running") {
    state.retrainPollHandle = setTimeout(refreshRetrainPill, 2000);
  }
}


function wireRetrainFailBanner() {
  document.getElementById("retrain-fail-dismiss").addEventListener(
    "click", () => {
      document.getElementById("retrain-fail-banner")
        .classList.add("hidden");
    });
}


function showRetrainFailBanner(stderrTail) {
  document.getElementById("retrain-fail-message").textContent =
    stderrTail.length > 8192
      ? "…" + stderrTail.slice(-8192)
      : stderrTail;
  document.getElementById("retrain-fail-banner")
    .classList.remove("hidden");
}


/* ──────────────────────────────────────────────────────────────────
 * R3 perf budget (open-to-first-render) — POST /api/render-ack.
 * ────────────────────────────────────────────────────────────────── */

function sendRenderAckIfReady() {
  if (state.renderAckSent) return;
  if (state.openStartedAtPerf == null) return;
  if (!state.osd || state.osd.world.getItemCount() === 0) return;
  if (!Array.isArray(state.detections)) return;
  const openMs = performance.now() - state.openStartedAtPerf;
  state.renderAckSent = true;
  fetch("/api/render-ack", {
    method:  "POST",
    headers: {"Content-Type": "application/json"},
    body:    JSON.stringify({
      job_id:     state.jobId,
      drawing_id: state.drawingId,
      open_ms:    openMs,
    }),
  }).catch(() => { /* perf-log is non-fatal */ });
}
