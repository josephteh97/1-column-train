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
};


/* ──────────────────────────────────────────────────────────────────
 * Boot
 * ────────────────────────────────────────────────────────────────── */

window.addEventListener("DOMContentLoaded", boot);

async function boot() {
  cachePalette();
  await populatePicker();
  loadReviewerIdFromStorage();
  wirePicker();
  wireKeyboard();
  wireFailBannerDismiss();
  wireZoomInput();
  wireInferenceButton();
  wireSubmitButton();
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
    if (!data.drawings.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "— no drawings ingested —";
      select.appendChild(opt);
      select.disabled = true;
      document.getElementById("open-btn").disabled = true;
    } else {
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "— select a drawing —";
      select.appendChild(blank);
      for (const id of data.drawings) {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = id;
        select.appendChild(opt);
      }
    }
  } catch (e) {
    showFailBanner("Failed to load /api/drawings: " + e.message);
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
  const reviewerInput = document.getElementById("reviewer-id-input");

  const trigger = () => {
    const drawingId = drawingSelect.value;
    const reviewerId = reviewerInput.value.trim();
    if (!drawingId) { drawingSelect.focus(); return; }
    if (!reviewerId) { reviewerInput.focus(); return; }
    localStorage.setItem("column-review.reviewer_id", reviewerId);
    openDrawing(drawingId, reviewerId);
  };

  openBtn.addEventListener("click", trigger);
  reviewerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") trigger();
  });
  drawingSelect.addEventListener("keydown", (e) => {
    if (e.key === "Enter") trigger();
  });
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
    btn.disabled = true;
    btn.querySelector(".spinner").classList.remove("hidden");
    try {
      const resp = await fetch("/api/infer", {
        method:  "POST",
        headers: {"Content-Type": "application/json"},
        body:    JSON.stringify({
          job_id: state.jobId, drawing_id: state.drawingId,
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
  document.getElementById("drawing-id-label").textContent =
    state.drawingId + " · " + state.reviewerId;
  document.getElementById("picker-drawer").classList.add("hidden");
  document.getElementById("submit-btn").classList.remove("hidden");

  mountOsd(openResponse.tile_source);
}


function mountOsd(tileSourceUrl) {
  if (state.osd) {
    state.osd.destroy();
    state.osd = null;
  }
  state.osd = OpenSeadragon({
    element:               document.getElementById("viewer"),
    tileSources:           tileSourceUrl,
    prefixUrl:             "/vendor/openseadragon/images/", // unused (no controls)
    showNavigationControl: false,
    showFullPageControl:   false,
    showHomeControl:       false,
    showZoomControl:       false,
    showRotationControl:   false,
    visibilityRatio:       0.4,
    minZoomImageRatio:     0.2,
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

  state.osd.addHandler("open", onOsdOpen);
  state.osd.addHandler("update-viewport", schedulePaint);
  state.osd.addHandler("canvas-press",   onCanvasPress);
  state.osd.addHandler("canvas-drag",    onCanvasDrag);
  state.osd.addHandler("canvas-release", onCanvasRelease);
  state.osd.addHandler("canvas-click",   onCanvasClick);

  installRenderCanary();
}


async function onOsdOpen() {
  const item = state.osd.world.getItemAt(0);
  if (item) {
    const size = item.getContentSize();
    state.imageSize = {width: size.x, height: size.y};
  }
  resizeOverlay();
  // Fetch detections; the canary fires after BOTH OSD-open and this
  // settle. If detections returns 412 (no inference yet), surface the
  // "Run inference" button instead of a failure banner.
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
    if (data.detections.length > 0) {
      document.getElementById("infer-btn").classList.add("hidden");
    } else {
      document.getElementById("infer-btn").classList.remove("hidden");
    }
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
  const wrap = document.getElementById("viewer-wrap");
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

function installRenderCanary() {
  canaryOsdOpenFired = false;
  state.detectionsFetchSettled = false;
  state.osd.addHandler("open", () => {
    canaryOsdOpenFired = true;
    maybeFireCanary();
  });
}


function maybeFireCanary() {
  if (!canaryOsdOpenFired || !state.detectionsFetchSettled) return;
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
