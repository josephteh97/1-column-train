// column-review — single-file frontend.
//
// Responsibilities (one per OpenSpec spec requirement):
//   - Load drawing + state + config from the backend.
//   - Mount OpenSeadragon over the DZI tile source.
//   - Paint detection bboxes on a single full-viewport canvas overlay
//     pinned to OSD's viewport (NOT per-bbox DOM nodes — 2000 nodes
//     blow the <50 ms budget).
//   - Single-key shortcuts: T/F/D/A/U/Y/N/P/+/-/0/F/Esc/Space.
//   - Zoom-adaptive hit-test.
//   - 100-deep ring-buffer undo/redo.
//   - Single-drag FN add.
//   - Shift+drag rubber-band batch select + mark-FP / delete-FN.
//   - Mini-map with unreviewed-cluster highlights.
//   - Live counters, filter-by-state, jump-to-next-unreviewed.
//   - Autosave: every action posts before showing as saved.
//   - Perf probe at boot; loud-fail banner on miss.

(() => {
  "use strict";

  // ── Constants ───────────────────────────────────────────────────────
  const STATE_COLOURS = {
    UNREVIEWED: { stroke: "#1e90ff", fill: "rgba(30,144,255,0.22)",  dash: []     },
    TP:         { stroke: "#2e8b57", fill: "rgba(46,139,87,0.30)",   dash: []     },
    FP:         { stroke: "#d72631", fill: "rgba(215,38,49,0.26)",   dash: [6, 4] },
    FN_ADDED:   { stroke: "#ff8c00", fill: "rgba(255,140,0,0.30)",   dash: [2, 3] },
  };
  const STATE_ORDER = ["UNREVIEWED", "TP", "FP", "FN_ADDED"];
  const UNDO_DEPTH = 100;
  const PERF_BUDGET_MS = 50;
  const SYNTHETIC_BOXES_FOR_PROBE = 2000;

  // ── DOM handles ─────────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const els = {
    viewer:      $("#viewer"),
    overlay:     $("#overlay-canvas"),
    minimap:     $("#minimap-canvas"),
    drawingId:   $("#drawing-id-label"),
    cntUnrev:    $("#cnt-unreviewed"),
    cntTp:       $("#cnt-tp"),
    cntFp:       $("#cnt-fp"),
    cntFn:       $("#cnt-fn"),
    zoomInput:   $("#zoom-input"),
    failBanner:  $("#fail-banner"),
    failMessage: $("#fail-message"),
    revBar:      $("#reviewer-id-bar"),
    revInput:    $("#reviewer-id-input"),
    revSave:     $("#reviewer-id-save"),
    inferBtn:    $("#infer-btn"),
  };

  // ── Mutable state ───────────────────────────────────────────────────
  const state = {
    drawing:      null,    // /api/drawing response
    cfg:          null,    // /api/config response
    detections:   [],      // columns[] from px_detections.json
    marks:        {},      // index -> "TP" | "FP" | "FN_ADDED" | "UNREVIEWED"
    selected:     new Set(),
    activeIndex:  null,    // last clicked index (focus for keyboard marks)
    addMode:      false,
    filter:       "ALL",
    undoStack:    new Array(UNDO_DEPTH).fill(null),
    undoHead:     0,
    undoLen:      0,
    redoStack:    new Array(UNDO_DEPTH).fill(null),
    redoLen:      0,
    osd:          null,
    rubber:       null,    // {x1,y1,x2,y2} screen-px while shift-dragging
    addDrag:      null,    // {x1,y1,x2,y2} world while A-dragging
  };

  // ── Boot ────────────────────────────────────────────────────────────
  async function boot() {
    try {
      // DZI present?
      const head = await fetch("/api/dzi-exists", { method: "HEAD" });
      if (!head.ok) {
        failHard(
          "DZI tile pyramid missing on disk.\n\n" +
          "Run:\n" +
          "  python3 scripts/hitl.py build-tiles <drawing-id>\n\n" +
          "Reopen the reviewer once tiles are built."
        );
        return;
      }
      // Drawing + state + config in parallel.
      const [drawingRes, stateRes, cfgRes] = await Promise.all([
        fetch("/api/drawing").then((r) => r.json()),
        fetch("/api/state").then((r) => r.json()),
        fetch("/api/config").then((r) => r.json()),
      ]);
      state.drawing    = drawingRes;
      state.cfg        = cfgRes;
      state.detections = (drawingRes.detections.columns || []).slice();

      // Initialise the marks map from the consolidated state response.
      state.marks = {};
      for (let i = 0; i < state.detections.length; i++) {
        state.marks[i] = stateRes[String(i)] || "UNREVIEWED";
      }

      els.drawingId.textContent = drawingRes.drawing_id;

      // Reviewer-id bar (only on first launch).
      if (!cfgRes.reviewer_id) showReviewerIdPrompt();

      // Perf probe BEFORE instantiating OSD — fail fast.
      if (!runPerfProbe()) return;

      mountOsd();
      installKeyboard();
      installMouse();
      installFilterButtons();
      installZoomInput();
      installReviewerIdHandlers();
      installInferButton();
      refreshCounts();
    } catch (e) {
      console.error(e);
      failHard("Boot failed: " + (e.message || e));
    }
  }

  function failHard(message) {
    els.failMessage.textContent = message;
    els.failBanner.classList.remove("hidden");
  }

  function runPerfProbe() {
    // Render SYNTHETIC_BOXES_FOR_PROBE rectangles on a throwaway canvas
    // sized like the LIVE overlay we will actually paint. If a single
    // full pass exceeds the budget, the reviewer's <50 ms interaction-
    // lag SLA cannot be met.
    //
    // Sizing matters: at boot time `els.overlay` hasn't been laid out
    // by OSD yet, so clientWidth/clientHeight can be 0. Fall back to
    // the viewport dimensions and account for devicePixelRatio so the
    // probe matches the real pixel workload on HiDPI / 4K displays —
    // not the 1280×800 hardcoded fallback that under-measured on 4K.
    const dpr = window.devicePixelRatio || 1;
    const cssW = els.overlay.clientWidth
               || document.documentElement.clientWidth
               || window.innerWidth || 1920;
    const cssH = els.overlay.clientHeight
               || document.documentElement.clientHeight
               || window.innerHeight || 1080;
    const w = Math.max(640, Math.floor(cssW * dpr));
    const h = Math.max(480, Math.floor(cssH * dpr));
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const ctx = c.getContext("2d");
    // Use the same per-state stroke/fill patterns the live overlay
    // uses, not a single flat strokeRect — gives a fairer estimate
    // because real frames cycle through stroke + fill + setLineDash.
    const t0 = performance.now();
    const palettes = Object.values(STATE_COLOURS);
    for (let i = 0; i < SYNTHETIC_BOXES_FOR_PROBE; i++) {
      const x = (i * 37) % (w - 12);
      const y = (i * 53) % (h - 12);
      const p = palettes[i & 3];
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = p.stroke;
      ctx.fillStyle = p.fill;
      ctx.setLineDash(p.dash);
      ctx.strokeRect(x, y, 10, 10);
      ctx.fillRect(x, y, 10, 10);
    }
    ctx.setLineDash([]);
    const dt = performance.now() - t0;
    if (dt > PERF_BUDGET_MS) {
      failHard(
        `Performance probe exceeded the ${PERF_BUDGET_MS} ms budget.\n\n` +
        `Synthetic ${SYNTHETIC_BOXES_FOR_PROBE}-box single-pass render took ` +
        `${dt.toFixed(1)} ms on this hardware ` +
        `(canvas ${w}×${h}, DPR ${dpr}).\n\n` +
        `Try a smaller drawing or a faster machine. No silent fallback ` +
        `is offered by design.`,
      );
      return false;
    }
    return true;
  }

  // ── OpenSeadragon mount ─────────────────────────────────────────────
  function mountOsd() {
    state.osd = OpenSeadragon({
      element: els.viewer,
      // prefixUrl is intentionally NOT set — we vendor only osd's JS
      // bundle (no images/ subdir, removed in the dead-code cleanup
      // pass). `showNavigationControl: false` below means OSD never
      // tries to load UI button icons; if either of those flags is
      // ever flipped to true without re-vendoring the OSD images dir,
      // the controls will render as broken-image icons (loud, by
      // design — easier to spot than silent 404s).
      tileSources: state.drawing.dzi_url,
      showNavigator: false,                  // we draw our own mini-map
      showNavigationControl: false,
      animationTime: 0.4,
      blendTime: 0.0,
      springStiffness: 12,
      gestureSettingsMouse: { clickToZoom: false, dblClickToZoom: false },
      // Nullish-coalesce: `--tile-cache-mb 0` from the CLI is a valid
      // explicit override (caps to the Math.max(100,…) floor below);
      // `|| 512` would have silently reverted it.
      maxImageCacheCount: Math.max(100,
        Math.floor((state.cfg.tile_cache_mb ?? 512) * 1024 * 1024 / 65536)),
    });
    state.osd.addHandler("update-viewport", () => { repaintOverlay(); repaintMinimap(); });
    state.osd.addHandler("open", () => {
      resizeOverlay();
      // If reopening a drawing that already has detections (relaunch
      // of a partially-marked session), don't dump the user at home
      // zoom showing the whole drawing — jump to the first unreviewed
      // so they continue where they left off. The pan inside
      // jumpUnreviewed triggers OSD's update-viewport handler which
      // calls repaintOverlay + repaintMinimap, so we elide them here.
      if (state.detections && state.detections.length > 0) {
        jumpUnreviewed(+1);
      } else {
        repaintOverlay();
        repaintMinimap();
      }
    });
    // Surface DZI-load failures loud instead of leaving the user
    // staring at a blank canvas (the "grey blank nothing" UX trap).
    state.osd.addHandler("open-failed", (e) => {
      const msg = (e && (e.message || e.source)) || JSON.stringify(e);
      failHard(
        "OpenSeadragon failed to load the DZI tile pyramid.\n\n" +
        `URL: ${state.drawing.dzi_url}\n` +
        `Detail: ${msg}\n\n` +
        "Open the browser console (F12 → Network) to see which " +
        "/dzi/ request failed. The manifest must return 200 OK with " +
        "valid XML; tile JPEGs must be served from /dzi/<id>_files/."
      );
    });
    state.osd.addHandler("tile-load-failed", (e) => {
      console.error("OSD tile-load-failed:", e);
    });
    window.addEventListener("resize", () => {
      resizeOverlay();
      repaintOverlay();
      repaintMinimap();
    });
    // Track the current zoom factor for the zoom-input indicator.
    state.osd.addHandler("zoom", () => {
      const z = state.osd.viewport.getZoom(true);
      els.zoomInput.value = Math.round(z * 100);
    });
  }

  function resizeOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const w = els.overlay.clientWidth;
    const h = els.overlay.clientHeight;
    els.overlay.width = w * dpr;
    els.overlay.height = h * dpr;
    const ctx = els.overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const mm = els.minimap;
    const mw = mm.clientWidth, mh = mm.clientHeight;
    mm.width = mw * dpr;
    mm.height = mh * dpr;
    mm.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  // ── World ↔ screen mapping ──────────────────────────────────────────
  function worldRectToScreen(bbox) {
    // bbox in image pixel coords [x1, y1, x2, y2].
    const tiled = state.osd.world.getItemAt(0);
    if (!tiled) return null;
    const p1 = tiled.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(bbox[0], bbox[1]));
    const p2 = tiled.imageToViewerElementCoordinates(
      new OpenSeadragon.Point(bbox[2], bbox[3]));
    return {
      x: Math.min(p1.x, p2.x),
      y: Math.min(p1.y, p2.y),
      w: Math.abs(p2.x - p1.x),
      h: Math.abs(p2.y - p1.y),
    };
  }

  function screenToImage(px, py) {
    const tiled = state.osd.world.getItemAt(0);
    if (!tiled) return null;
    const pt = tiled.viewerElementToImageCoordinates(
      new OpenSeadragon.Point(px, py));
    return { x: pt.x, y: pt.y };
  }

  // ── Overlay paint ───────────────────────────────────────────────────
  function repaintOverlay() {
    const ctx = els.overlay.getContext("2d");
    const w = els.overlay.clientWidth;
    const h = els.overlay.clientHeight;
    ctx.clearRect(0, 0, w, h);

    // Zoom-adaptive stroke scaling — at OSD's home/fit-to-image zoom
    // (~0.14 for A0 plans) detection boxes are only a few CSS pixels
    // wide, so the baseline 2.5 px stroke would disappear into the
    // CAD line art. Clamp [1.0, 4.0] so the close-up baseline matches
    // the design intent ("1× at 100% zoom" → full 2.5 px stroke) and
    // extreme zoom-out doesn't draw 20 px strokes over 3 px boxes.
    const _zoom = (state.osd && state.osd.viewport)
      ? state.osd.viewport.getZoom(true) : 1;
    const strokeScale = Math.max(1.0,
      Math.min(4.0, 0.5 / Math.max(_zoom, 0.01)));

    for (let i = 0; i < state.detections.length; i++) {
      const mark = state.marks[i] || "UNREVIEWED";
      // REMOVED is the synthetic state the backend returns for an
      // FN_ADDED the user later deleted. The audit row stays in
      // corrections.db but the entry is hidden from the UI.
      if (mark === "REMOVED") continue;
      if (state.filter !== "ALL" && mark !== state.filter) continue;
      const bbox = state.detections[i].bbox;
      if (!bbox || bbox.length < 4) continue;
      const r = worldRectToScreen(bbox);
      if (!r) continue;
      // Cull anything fully outside the viewport.
      if (r.x + r.w < 0 || r.y + r.h < 0 || r.x > w || r.y > h) continue;

      const palette = STATE_COLOURS[mark];
      const isActive = state.selected.has(i) || state.activeIndex === i;
      ctx.lineWidth   = (isActive ? 4 : 2.5) * strokeScale;
      ctx.strokeStyle = palette.stroke;
      ctx.fillStyle   = palette.fill;
      ctx.setLineDash(palette.dash);
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.fillRect(r.x, r.y, r.w, r.h);
      // Glowing white outer ring on the active detection so the
      // focused box reads unmistakably against the CAD drawing.
      // 2-px outset (not 3) keeps the ring closer to the box, so a
      // box near the viewport edge has its ring clipped on at most
      // 1 px per side instead of 3 — the one-sided-ring visual at
      // edges is much less obvious.
      if (isActive) {
        ctx.save();
        ctx.lineWidth = Math.max(1.5, 2 * strokeScale);
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.setLineDash([]);
        ctx.strokeRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4);
        ctx.restore();
      }
    }
    ctx.setLineDash([]);
    // Live in-flight rubber-band rectangle.
    if (state.rubber) {
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(state.rubber.x1, state.rubber.y1,
                     state.rubber.x2 - state.rubber.x1,
                     state.rubber.y2 - state.rubber.y1);
      ctx.setLineDash([]);
    }
    // Live in-flight add-FN rectangle (world coords transformed).
    if (state.addDrag) {
      const r = worldRectToScreen([
        Math.min(state.addDrag.x1, state.addDrag.x2),
        Math.min(state.addDrag.y1, state.addDrag.y2),
        Math.max(state.addDrag.x1, state.addDrag.x2),
        Math.max(state.addDrag.y1, state.addDrag.y2),
      ]);
      if (r) {
        ctx.strokeStyle = STATE_COLOURS.FN_ADDED.stroke;
        ctx.lineWidth = 2; ctx.setLineDash([2, 3]);
        ctx.strokeRect(r.x, r.y, r.w, r.h);
        ctx.setLineDash([]);
      }
    }
  }

  function repaintMinimap() {
    const ctx = els.minimap.getContext("2d");
    const mw = els.minimap.clientWidth, mh = els.minimap.clientHeight;
    ctx.clearRect(0, 0, mw, mh);
    if (!state.drawing || !state.drawing.raster_size) return;
    const [W, H] = state.drawing.raster_size;
    const scale = Math.min(mw / W, mh / H);
    const offX = (mw - W * scale) / 2;
    const offY = (mh - H * scale) / 2;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(offX, offY, W * scale, H * scale);

    for (let i = 0; i < state.detections.length; i++) {
      const mark = state.marks[i] || "UNREVIEWED";
      if (mark === "REMOVED") continue;
      const bbox = state.detections[i].bbox;
      if (!bbox) continue;
      const cx = offX + ((bbox[0] + bbox[2]) / 2) * scale;
      const cy = offY + ((bbox[1] + bbox[3]) / 2) * scale;
      const palette = STATE_COLOURS[mark];
      if (!palette) continue;
      ctx.fillStyle = palette.stroke;
      // Shape treatment: TP=square, FP=diamond, FN=triangle, UNREVIEWED=circle.
      const r = 2.5;
      ctx.beginPath();
      if (mark === "UNREVIEWED") {
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
      } else if (mark === "TP") {
        ctx.rect(cx - r, cy - r, r * 2, r * 2);
      } else if (mark === "FP") {
        ctx.moveTo(cx, cy - r); ctx.lineTo(cx + r, cy);
        ctx.lineTo(cx, cy + r); ctx.lineTo(cx - r, cy); ctx.closePath();
      } else if (mark === "FN_ADDED") {
        ctx.moveTo(cx, cy - r); ctx.lineTo(cx + r, cy + r);
        ctx.lineTo(cx - r, cy + r); ctx.closePath();
      }
      ctx.fill();
    }

    // Viewport rectangle.
    const vp = state.osd.viewport;
    const tiled = state.osd.world.getItemAt(0);
    if (tiled) {
      const bounds = vp.getBounds();
      const tl = tiled.viewportToImageCoordinates(
        new OpenSeadragon.Point(bounds.x, bounds.y));
      const br = tiled.viewportToImageCoordinates(
        new OpenSeadragon.Point(bounds.x + bounds.width, bounds.y + bounds.height));
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.2;
      ctx.strokeRect(
        offX + tl.x * scale, offY + tl.y * scale,
        (br.x - tl.x) * scale, (br.y - tl.y) * scale,
      );
    }
  }

  // ── Counters ────────────────────────────────────────────────────────
  function refreshCounts() {
    let c = { UNREVIEWED: 0, TP: 0, FP: 0, FN_ADDED: 0 };
    for (let i = 0; i < state.detections.length; i++) {
      const m = state.marks[i] || "UNREVIEWED";
      if (m === "REMOVED") continue;   // hidden — don't count
      if (c[m] === undefined) continue;
      c[m]++;
    }
    els.cntUnrev.textContent = c.UNREVIEWED;
    els.cntTp.textContent    = c.TP;
    els.cntFp.textContent    = c.FP;
    els.cntFn.textContent    = c.FN_ADDED;
  }

  // ── Undo / redo ring buffer ─────────────────────────────────────────
  function pushUndo(op) {
    state.undoStack[state.undoHead] = op;
    state.undoHead = (state.undoHead + 1) % UNDO_DEPTH;
    state.undoLen = Math.min(state.undoLen + 1, UNDO_DEPTH);
    state.redoLen = 0;   // any new action invalidates redo
  }
  function popUndo() {
    if (state.undoLen === 0) return null;
    state.undoHead = (state.undoHead - 1 + UNDO_DEPTH) % UNDO_DEPTH;
    state.undoLen -= 1;
    const op = state.undoStack[state.undoHead];
    state.undoStack[state.undoHead] = null;
    return op;
  }
  function pushRedo(op) {
    state.redoStack[state.redoLen] = op;
    state.redoLen = Math.min(state.redoLen + 1, UNDO_DEPTH);
  }
  function popRedo() {
    if (state.redoLen === 0) return null;
    state.redoLen -= 1;
    return state.redoStack[state.redoLen];
  }

  // ── Mark application (POST then update UI) ──────────────────────────
  // Returns true on server-confirmed success, false on rejection.
  // The undo stack is updated by `mark()` based on this return value,
  // so a 409/500 from the server does NOT corrupt the undo history.
  async function applyMark(idx, kind) {
    const prev = state.marks[idx] || "UNREVIEWED";
    let serverKind = kind;
    if (kind === prev) return true;   // any same-state mark is a no-op
    if (kind === "UNREVIEWED") {
      if (prev === "FP")            serverKind = "RESCIND_FP";
      else if (prev === "TP")       serverKind = "CLEAR_TP";
      else if (prev === "FN_ADDED") serverKind = "DELETE_FN";
      else return true;
    }
    // Undo path for DELETE_FN: redoOnce/undoOnce replays
    // applyMark(idx, "FN_ADDED") on a REMOVED slot. The cols entry
    // still has source=human_added with its original bbox; the server
    // re-inserts the is_delete=0 row, restoring the slot to FN_ADDED.
    // Without this branch the body would be {kind:"FN_ADDED"} with no
    // bbox → server 500 → undo silently dead.
    if (kind === "FN_ADDED" && prev === "REMOVED") {
      serverKind = "RESTORE_FN";
    }
    const body = { kind: serverKind, element_index: idx };

    let res;
    try {
      res = await fetch("/api/marks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (e) {
      console.error("mark POST network error", e);
      flashMarkError("network error — mark not saved");
      return false;
    }

    if (res.ok) {
      const json = await res.json();
      for (let i = 0; i < state.detections.length; i++) {
        state.marks[i] = json.state[String(i)] || "UNREVIEWED";
      }
      refreshCounts();
      repaintOverlay();
      repaintMinimap();
      return true;
    }

    // Server rejected. Common causes the reviewer should see:
    //   409 — reviewer-id not set; the prompt bar above is waiting.
    //   400 — malformed payload (programming bug — log it).
    let detail = "";
    try { detail = (await res.json()).detail || ""; } catch (_) {}
    if (res.status === 409) {
      flashMarkError("set your reviewer id first (top of page)");
      showReviewerIdPrompt();
    } else {
      console.error("mark POST failed", res.status, detail);
      flashMarkError(`mark rejected (${res.status})`);
    }
    return false;
  }

  // One-line flash routed through the existing hint strip — no new DOM.
  // `kind` is "error" (red, ⚠) or "info" (green, ✓); defaults to error
  // because that's the most common caller (mark / inference failures).
  //
  // The "original" hint text is captured ONCE on the first call (not
  // per-flash via closure) so overlapping flashes can't restore a
  // stale in-flight message — every setTimeout returns to the same
  // baseline. Previous implementation captured `prev = hint.textContent`
  // per call, which meant a fast B-after-A sequence would restore the
  // hint to A's flash text instead of the original.
  let _hintOriginal = null;
  function flashHint(msg, kind = "error") {
    const hint = document.querySelector("#progress-strip .hint");
    if (!hint) return;
    if (_hintOriginal === null) _hintOriginal = hint.textContent;
    const isInfo = kind === "info";
    hint.textContent = (isInfo ? "✓ " : "⚠ ") + msg;
    hint.style.color = `var(--col-${isInfo ? "tp" : "fp"})`;
    setTimeout(() => {
      hint.textContent = _hintOriginal;
      hint.style.color = "";
    }, isInfo ? 3500 : 2500);
  }
  // Back-compat alias so existing call sites stay terse.
  const flashMarkError = (msg) => flashHint(msg, "error");

  async function undoOnce() {
    // Peek the top entry without popping — only pop if applyMark
    // succeeds. Otherwise the undo stack would lose the entry on a
    // network failure even though the server-side mark stayed.
    if (state.undoLen === 0) return;
    const peekHead = (state.undoHead - 1 + UNDO_DEPTH) % UNDO_DEPTH;
    const op = state.undoStack[peekHead];
    if (!op) return;
    const ok = await applyMark(op.idx, op.prev);
    if (!ok) return;
    popUndo();
    pushRedo(op);
  }

  async function redoOnce() {
    if (state.redoLen === 0) return;
    const op = state.redoStack[state.redoLen - 1];
    if (!op) return;
    const ok = await applyMark(op.idx, op.next);
    if (!ok) return;
    popRedo();
    state.undoStack[state.undoHead] = op;
    state.undoHead = (state.undoHead + 1) % UNDO_DEPTH;
    state.undoLen = Math.min(state.undoLen + 1, UNDO_DEPTH);
  }

  async function mark(idx, next) {
    // Capture prev BEFORE the server round-trip — applyMark may
    // mutate state.marks via the server response.
    const prev = state.marks[idx] || "UNREVIEWED";
    if (prev === next) return;
    const ok = await applyMark(idx, next);
    if (!ok) return;
    // Server confirmed → push undo. If the call failed, the undo
    // stack stays consistent with what the server actually persisted.
    pushUndo({ idx, prev, next });
  }

  // ── Hit-test (zoom-adaptive) ────────────────────────────────────────
  function hitTest(imgX, imgY) {
    // Nullish-coalesce so an explicit `--hit-tolerance-px 0` is honoured
    // (pixel-perfect mode) instead of silently reverting to 8.
    const baseCss = state.cfg.hit_tolerance_px ?? 8;
    const zoom = state.osd.viewport.getZoom(true);
    // World-space tolerance grows when zoomed out.
    const tol = Math.max(baseCss, baseCss / Math.max(zoom, 0.01));
    let best = null, bestDist = Infinity;
    for (let i = 0; i < state.detections.length; i++) {
      // Skip REMOVED slots: their bbox is still in cols (we kept the
      // JSON entry for audit), but the UI hides them — clicking
      // through onto a hidden box would record TP/FP marks the state
      // map ignores, leaking orphan rows.
      if ((state.marks[i] || "UNREVIEWED") === "REMOVED") continue;
      const bbox = state.detections[i].bbox;
      if (!bbox || bbox.length < 4) continue;
      const inside = (imgX >= bbox[0] - tol && imgX <= bbox[2] + tol
                   && imgY >= bbox[1] - tol && imgY <= bbox[3] + tol);
      if (!inside) continue;
      const cx = (bbox[0] + bbox[2]) / 2;
      const cy = (bbox[1] + bbox[3]) / 2;
      const d = (imgX - cx) ** 2 + (imgY - cy) ** 2;
      if (d < bestDist) { best = i; bestDist = d; }
    }
    return best;
  }

  // ── Keyboard ────────────────────────────────────────────────────────
  function installKeyboard() {
    document.addEventListener("keydown", async (e) => {
      // Don't intercept while typing into the reviewer-id or zoom inputs.
      if (e.target && (e.target.tagName === "INPUT")) return;
      if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        return;   // OSD already handles space-drag panning via mouse
      }
      switch (e.key.toLowerCase()) {
        case "t":
          if (state.activeIndex != null) { e.preventDefault(); await mark(state.activeIndex, "TP"); }
          break;
        case "f":
          if (state.activeIndex != null) { e.preventDefault(); await mark(state.activeIndex, "FP"); }
          else { e.preventDefault(); fitToWindow(); }
          break;
        case "d":
          if (state.activeIndex != null) { e.preventDefault(); await mark(state.activeIndex, "UNREVIEWED"); }
          break;
        case "a": e.preventDefault(); enterAddMode();   break;
        case "u": e.preventDefault(); await undoOnce(); break;
        case "y": e.preventDefault(); await redoOnce(); break;
        case "n": e.preventDefault(); jumpUnreviewed(+1); break;
        case "p": e.preventDefault(); jumpUnreviewed(-1); break;
        case "0": e.preventDefault(); state.osd.viewport.zoomTo(1.0); break;
        case "+": case "=": e.preventDefault(); zoomBy(1.25); break;
        case "-": case "_": e.preventDefault(); zoomBy(0.8); break;
        case "escape": state.selected.clear(); state.activeIndex = null;
                       state.addMode = false; repaintOverlay(); break;
      }
    });
  }

  function zoomBy(factor) {
    const vp = state.osd.viewport;
    // Centre on selection if any, else on cursor (cursor wheel handled
    // natively by OSD — keyboard +/- centres on viewport for now).
    vp.zoomBy(factor);
    vp.applyConstraints();
  }
  function fitToWindow() { state.osd.viewport.fitBounds(state.osd.world.getHomeBounds()); }

  function jumpUnreviewed(direction) {
    if (state.detections.length === 0) return;
    const N = state.detections.length;
    // When activeIndex is null (post-inference auto-pan / relaunch
    // open-handler), anchor `start` so the FIRST iteration (k=1)
    // lands on idx=0 for forward and idx=N-1 for backward. Without
    // this fix, k=1 gives idx=1 (forward) — silently skipping
    // detection 0 on every auto-pan.
    const start = state.activeIndex == null
      ? (direction > 0 ? -1 : 0)
      : state.activeIndex;
    for (let k = 1; k <= N; k++) {
      const idx = (start + direction * k + N * 100) % N;
      const m = state.marks[idx] || "UNREVIEWED";
      if (m === "REMOVED") continue;
      if (m === "UNREVIEWED") {
        state.activeIndex = idx;
        panToIndex(idx);
        repaintOverlay();
        return;
      }
    }
  }

  function panToIndex(idx) {
    const bbox = state.detections[idx].bbox;
    if (!bbox) return;
    const cx = (bbox[0] + bbox[2]) / 2;
    const cy = (bbox[1] + bbox[3]) / 2;
    const tiled = state.osd.world.getItemAt(0);
    if (!tiled) return;
    const target = tiled.imageToViewportCoordinates(new OpenSeadragon.Point(cx, cy));
    state.osd.viewport.panTo(target, true);
    // Zoom in if too far out (box subtends < 80 css px).
    const bboxW = bbox[2] - bbox[0];
    const zoomNeeded = (80 / Math.max(bboxW, 1)) * state.osd.viewport.getContainerSize().x / state.drawing.raster_size[0];
    const currentZoom = state.osd.viewport.getZoom(true);
    if (currentZoom < zoomNeeded) state.osd.viewport.zoomTo(zoomNeeded, target, true);
  }

  function enterAddMode() {
    state.addMode = true;
    state.activeIndex = null;
    document.body.style.cursor = "crosshair";
  }

  // ── Mouse / drag ────────────────────────────────────────────────────
  function installMouse() {
    // We attach to the OSD canvas via the viewer element so OSD's
    // pan handlers and ours can coexist. Shift-drag = rubber band,
    // A-then-drag = add-FN, otherwise OSD handles pan/zoom.
    const tracker = new OpenSeadragon.MouseTracker({
      element: state.osd.canvas,
      pressHandler: (e) => onPress(e),
      dragHandler: (e) => onDrag(e),
      releaseHandler: (e) => onRelease(e),
      clickHandler: (e) => onClick(e),
    });
    tracker.setTracking(true);
  }

  function evToImage(e) {
    const rect = els.viewer.getBoundingClientRect();
    const px = e.position.x;
    const py = e.position.y;
    return screenToImage(px, py);
  }

  function onClick(e) {
    if (state.addMode) return;
    const img = evToImage(e);
    if (!img) return;
    const idx = hitTest(img.x, img.y);
    if (idx == null) {
      state.selected.clear();
      state.activeIndex = null;
    } else {
      state.activeIndex = idx;
      state.selected.clear();
      state.selected.add(idx);
    }
    repaintOverlay();
  }

  function onPress(e) {
    if (state.addMode) {
      const img = evToImage(e);
      if (!img) return;
      state.addDrag = { x1: img.x, y1: img.y, x2: img.x, y2: img.y };
      e.preventDefaultAction = true;
      return;
    }
    if (e.originalEvent && e.originalEvent.shiftKey) {
      state.rubber = { x1: e.position.x, y1: e.position.y,
                       x2: e.position.x, y2: e.position.y };
      e.preventDefaultAction = true;
      return;
    }
  }

  function onDrag(e) {
    if (state.addMode && state.addDrag) {
      const img = evToImage(e);
      if (!img) return;
      state.addDrag.x2 = img.x;
      state.addDrag.y2 = img.y;
      repaintOverlay();
      e.preventDefaultAction = true;
      return;
    }
    if (state.rubber) {
      state.rubber.x2 = e.position.x;
      state.rubber.y2 = e.position.y;
      repaintOverlay();
      e.preventDefaultAction = true;
      return;
    }
  }

  async function onRelease(e) {
    if (state.addMode && state.addDrag) {
      let { x1, y1, x2, y2 } = state.addDrag;
      const grid = state.cfg.snap_grid_px || 0;
      const snap = (v) => grid ? Math.round(v / grid) * grid : v;
      const bbox = [
        snap(Math.min(x1, x2)), snap(Math.min(y1, y2)),
        snap(Math.max(x1, x2)), snap(Math.max(y1, y2)),
      ];
      state.addDrag = null;
      state.addMode = false;
      document.body.style.cursor = "";
      // Don't add a degenerate bbox.
      if (bbox[2] - bbox[0] < 2 || bbox[3] - bbox[1] < 2) {
        repaintOverlay(); return;
      }
      const res = await fetch("/api/marks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: "FN_ADDED", bbox }),
      });
      if (res.ok) {
        const json = await res.json();
        // Re-fetch the drawing to pick up the new detection entry
        // (it was appended to px_detections.json server-side).
        const fresh = await fetch("/api/drawing").then((r) => r.json());
        state.detections = (fresh.detections.columns || []).slice();
        for (let i = 0; i < state.detections.length; i++) {
          state.marks[i] = json.state[String(i)] || "UNREVIEWED";
        }
      }
      refreshCounts();
      repaintOverlay();
      repaintMinimap();
      return;
    }
    if (state.rubber) {
      const r = state.rubber;
      state.rubber = null;
      // Convert screen rect to image rect.
      const tl = screenToImage(Math.min(r.x1, r.x2), Math.min(r.y1, r.y2));
      const br = screenToImage(Math.max(r.x1, r.x2), Math.max(r.y1, r.y2));
      if (!tl || !br) { repaintOverlay(); return; }
      state.selected.clear();
      for (let i = 0; i < state.detections.length; i++) {
        // Skip REMOVED — same reasoning as hitTest: hidden entries
        // must not be picked up by rubber-band.
        if ((state.marks[i] || "UNREVIEWED") === "REMOVED") continue;
        const bb = state.detections[i].bbox;
        if (!bb) continue;
        const cx = (bb[0] + bb[2]) / 2;
        const cy = (bb[1] + bb[3]) / 2;
        if (cx >= tl.x && cx <= br.x && cy >= tl.y && cy <= br.y) {
          state.selected.add(i);
        }
      }
      // Modifier on release: Ctrl+Shift = batch-delete, plain Shift = batch-FP.
      const ev = e.originalEvent || {};
      if (state.selected.size > 0) {
        const kind = ev.ctrlKey ? "DELETE_FN" : "FP";
        const marks = [...state.selected].map((i) => ({ kind, element_index: i }));
        const res = await fetch("/api/marks/batch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ marks }),
        });
        if (res.ok) {
          const json = await res.json();
          for (let i = 0; i < state.detections.length; i++) {
            state.marks[i] = json.state[String(i)] || "UNREVIEWED";
          }
        }
      }
      refreshCounts();
      repaintOverlay();
      repaintMinimap();
    }
  }

  // ── Filter buttons ──────────────────────────────────────────────────
  function installFilterButtons() {
    for (const btn of document.querySelectorAll(".filter-btn")) {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".filter-btn").forEach(
          (b) => b.classList.remove("active"));
        btn.classList.add("active");
        state.filter = btn.dataset.filter;
        repaintOverlay();
      });
    }
  }

  // ── Zoom input ──────────────────────────────────────────────────────
  function installZoomInput() {
    els.zoomInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const pct = parseFloat(els.zoomInput.value);
        if (Number.isFinite(pct) && pct > 0) {
          state.osd.viewport.zoomTo(pct / 100);
        }
        els.zoomInput.blur();
      }
    });
  }

  // ── Reviewer-id prompt ──────────────────────────────────────────────
  function showReviewerIdPrompt() {
    els.revBar.classList.remove("hidden");
    els.revInput.focus();
  }
  function installReviewerIdHandlers() {
    const submit = async () => {
      const id = els.revInput.value.trim();
      if (!id) return;
      const res = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reviewer_id: id }),
      });
      if (res.ok) {
        els.revBar.classList.add("hidden");
      }
    };
    els.revSave.addEventListener("click", submit);
    els.revInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") submit();
    });
  }

  // ── Run-inference button ────────────────────────────────────────────
  // Visible exactly when the drawing has no MODEL detections (entries
  // whose `source` is anything OTHER than "human_added"). Click →
  // POST /api/infer, spinner while we wait (30-90 s on CPU, 2-5 s on
  // GPU), then reload the drawing + state and repaint.
  //
  // The earlier `length === 0` rule trapped the button hidden the
  // moment the user drag-added a single FN, preventing them from
  // EVER running inference on this drawing — even after a server
  // relaunch (the human_added entry survives in px_detections.json).
  // Now any number of FN_ADDED entries is OK; only model detections
  // gate the button.
  function refreshInferBtnVisibility() {
    const hasModel = state.detections.some(
      (c) => c && c.source !== "human_added"
    );
    if (!hasModel) {
      els.inferBtn.classList.remove("hidden");
    } else {
      els.inferBtn.classList.add("hidden");
    }
  }

  function installInferButton() {
    refreshInferBtnVisibility();
    const labelEl   = els.inferBtn.querySelector(".label");
    const spinnerEl = els.inferBtn.querySelector(".spinner");
    const origLabel = labelEl.textContent;
    els.inferBtn.addEventListener("click", async () => {
      els.inferBtn.disabled = true;
      spinnerEl.classList.remove("hidden");
      labelEl.textContent = "Running inference… (~30-90 s on CPU)";
      try {
        let res;
        try {
          res = await fetch("/api/infer", { method: "POST" });
        } catch (e) {
          flashMarkError(`inference network error: ${e.message || e}`);
          return;
        }
        if (!res.ok) {
          let detail = "";
          try { detail = (await res.json()).detail || ""; } catch (_) {}
          flashMarkError(
            `inference failed (${res.status}): ${detail || "see terminal"}`
          );
          return;
        }
        const infer = await res.json();
        // Reload the drawing so state.detections picks up the new entries.
        const fresh = await fetch("/api/drawing").then((r) => r.json());
        state.detections = (fresh.detections.columns || []).slice();
        // Server returned the post-apply state map — use it directly.
        state.marks = {};
        for (let i = 0; i < state.detections.length; i++) {
          state.marks[i] = infer.state[String(i)] || "UNREVIEWED";
        }
        // Inference replaces / extends state.detections, so any in-
        // memory undo / redo entries reference indices that may now
        // point at different bboxes (or have been displaced by the
        // human_added-preservation reshuffle). Cheap to reset; safe
        // because the only marks that could exist pre-inference are
        // drag-adds (which don't push undo) and TP/FP on human_added
        // (rare and recoverable from the server-side state map).
        state.undoStack.fill(null);
        state.undoHead = 0;
        state.undoLen  = 0;
        state.redoStack.fill(null);
        state.redoLen  = 0;
        refreshCounts();
        refreshInferBtnVisibility();
        // Auto-pan + zoom to the first unreviewed detection so the
        // user lands on actionable content — at home zoom the
        // 13480×9536 drawing makes every box a near-invisible dot.
        // `jumpUnreviewed(+1)` reuses the existing helper that pans
        // AND zooms so the target box subtends ≥80 CSS pixels.
        // The pan triggers OSD's `update-viewport` event which calls
        // repaintOverlay + repaintMinimap, so we skip the explicit
        // repaints here to avoid drawing once at the (about-to-be-
        // discarded) home zoom and immediately again at the target.
        if (state.detections.length > 0) {
          state.activeIndex = null;
          jumpUnreviewed(+1);
        } else {
          repaintOverlay();
          repaintMinimap();
        }
        // Visible confirmation of the merge result, including any
        // FN_ADDED entries that survived the inference run AND the
        // empty-detections case (so a zero-detection model run still
        // gives the user feedback — they clicked a button, something
        // should reply).
        const n = infer.n_detections || 0;
        const p = infer.n_preserved  || 0;
        if (p > 0) {
          flashHint(
            `Inference ran — ${n} new detection${n === 1 ? "" : "s"}, `
            + `${p} prior FN_ADDED entr${p === 1 ? "y" : "ies"} preserved`,
            "info",
          );
        } else if (n > 0) {
          flashHint(`Inference ran — ${n} detection${n === 1 ? "" : "s"}`,
                    "info");
        } else {
          flashHint(
            "Inference ran — 0 detections found on this drawing",
            "info",
          );
        }
      } finally {
        els.inferBtn.disabled = false;
        spinnerEl.classList.add("hidden");
        labelEl.textContent = origLabel;
      }
    });
  }

  // ── Go ──────────────────────────────────────────────────────────────
  boot();
})();
