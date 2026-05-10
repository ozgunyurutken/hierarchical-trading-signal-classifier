// V5 Hierarchical Trading Signal Classifier — live terminal frontend.
// Single bundle fetch -> in-memory cache -> scrubber drives all panels.

const $ = (id) => document.getElementById(id);
const fmt = {
  pct:   (v, d=2)  => (v >= 0 ? "+" : "") + (v * 100).toFixed(d) + "%",
  num:   (v, d=2)  => v.toFixed(d),
  price: (v)       => "$" + (v >= 1000 ? v.toLocaleString(undefined, {maximumFractionDigits: 0}) : v.toFixed(2)),
  signed: (v, d=3) => (v >= 0 ? "+" : "") + v.toFixed(d),
};
const COLORS = {
  buy:  "#00ff9f", sell: "#ff3860", hold: "#ffb547",
  bull: "rgba(0, 214, 143, 0.32)", neutral: "rgba(255, 181, 71, 0.20)", bear: "rgba(255, 85, 119, 0.32)",
  bullDot: "#00d68f", neutralDot: "#ffb547", bearDot: "#ff5577",
  txt0: "#e6ecfa", txt1: "#b3bfd6", txt2: "#6b7895", txt3: "#4a5570",
  bg: "#06090f", line: "rgba(120, 144, 184, 0.14)",
};
const LAB_FEATURE_GROUPS = {
  "STAGE 1 · TREND POSTERIOR": [
    {key: "P1_up",    label: "P1_up",    min: 0, max: 1, step: 0.01,
     desc: "Posterior probability the local price trend is UP (Stage 1 Random Forest output)."},
    {key: "P1_range", label: "P1_range", min: 0, max: 1, step: 0.01,
     desc: "Posterior probability the trend is RANGE-bound (sideways)."},
    {key: "P1_down",  label: "P1_down",  min: 0, max: 1, step: 0.01,
     desc: "Posterior probability the trend is DOWN."},
  ],
  "STAGE 1 · SMOOTHED (10d)": [
    {key: "P1_up_smooth10",    label: "P1_up_smooth10",    min: 0, max: 1, step: 0.01,
     desc: "10-day rolling mean of P1_up. Reduces frame-level noise — a sustained up posterior."},
    {key: "P1_range_smooth10", label: "P1_range_smooth10", min: 0, max: 1, step: 0.01,
     desc: "10-day rolling mean of P1_range."},
    {key: "P1_down_smooth10",  label: "P1_down_smooth10",  min: 0, max: 1, step: 0.01,
     desc: "10-day rolling mean of P1_down."},
  ],
  "STAGE 2 · MACRO REGIME": [
    {key: "P2_Bull",    label: "P2_Bull",    min: 0, max: 1, step: 0.01,
     desc: "1.0 if Stage 2 FSM classifies today as Bull (risk-on macro), else 0."},
    {key: "P2_Neutral", label: "P2_Neutral", min: 0, max: 1, step: 0.01,
     desc: "1.0 if FSM regime is Neutral. (One of the three is always 1.0.)"},
    {key: "P2_Bear",    label: "P2_Bear",    min: 0, max: 1, step: 0.01,
     desc: "1.0 if FSM regime is Bear (risk-off — VIX, yield-curve or DXY+M2 stress)."},
    {key: "regime_age_days", label: "regime_age (days)", min: 0, max: 400, step: 1,
     desc: "Days since the last regime transition. Newer regimes (low age) = more uncertain."},
  ],
  "OSCILLATORS": [
    {key: "RSI_14",            label: "RSI (14d)",       min: 0,    max: 100, step: 0.5,
     desc: "Momentum: <30 = OVERSOLD (potential bounce), >70 = OVERBOUGHT (potential pullback)."},
    {key: "MACD_signal_diff",  label: "MACD − signal",   min: -800, max: 800, step: 1,
     desc: "Trend-momentum: positive = bullish (MACD above signal), negative = bearish."},
    {key: "Bollinger_pct_b",   label: "Bollinger %B",    min: -0.5, max: 1.5, step: 0.01,
     desc: "Position within 20d±2σ band: <0 = below lower band (oversold), >1 = above upper band (overbought)."},
    {key: "Stochastic_K_14",   label: "Stochastic %K",   min: 0,    max: 100, step: 0.5,
     desc: "Like RSI: <20 = oversold, >80 = overbought. Faster than RSI."},
    {key: "volume_zscore_20",  label: "Volume z-score",  min: -3,   max: 5,   step: 0.05,
     desc: "Volume relative to 20d mean (z-score). >1 = above-average volume (institutional activity)."},
    {key: "OBV_change_20d",    label: "OBV change 20d",  min: -0.5, max: 0.5, step: 0.005,
     desc: "20-day OBV change: positive = ACCUMULATION (buying), negative = DISTRIBUTION (selling)."},
  ],
};

// ───────────────────────── App state ─────────────────────────
const ST = {
  asset: "BTC",
  arch:  "default",
  model: "default",
  rule:  "default",
  bundle: null,        // current bundle data
  idx:    -1,          // current scrubber index
  playing: false,
  speed: 1,            // multiplier (1x = 10 days/sec, 3x=30, 6x=60)
  playerHandle: null,
  hero: {chart: null, candle: null, volume: null, regime: null, marker: null},
  miniCanvas: null,
  raceStart: -1,         // index from which the portfolio race is computed (set on Play)
  labOpen: false,
  labMode: "actual",   // "actual" | "custom"
  labFeatures: {},     // current slider values (Custom mode)
  labDebounce: null,
  labBaselineProbs: null,  // first /predict_custom result for actual-feature row,
                           // used as comparison reference in Custom mode
  heatmap: null,
};

// ───────────────────────── Bootstrap ─────────────────────────
async function init() {
  // Topbar wiring
  document.querySelectorAll(".asset-btn").forEach(b =>
    b.addEventListener("click", () => switchAsset(b.dataset.asset)));
  for (const id of ["arch", "model", "rule"]) {
    $(id).addEventListener("change", () => loadBundle());
  }

  // Scrubber
  $("scrubber").addEventListener("input", (e) => seek(parseInt(e.target.value, 10)));
  $("playBtn").addEventListener("click", togglePlay);
  $("resetBtn").addEventListener("click", () => {
    // Reset clears the race AND jumps to most-recent date.
    ST.raceStart = -1;
    resetRaceChart();
    seek(ST.bundle ? ST.bundle.n - 1 : 0);
  });
  document.querySelectorAll(".speed-btn").forEach(b =>
    b.addEventListener("click", () => setSpeed(parseInt(b.dataset.speed, 10))));

  // Lab
  $("labToggle").addEventListener("click", () => toggleLab(true));
  $("labClose").addEventListener("click", () => toggleLab(false));

  // Heatmap collapsible — default closed (set in HTML), header click toggles.
  $("heatmapHeader").addEventListener("click", () => {
    $("heatmapPanel").classList.toggle("collapsed");
  });
  document.querySelectorAll(".lab-mode-btn").forEach(b =>
    b.addEventListener("click", () => setLabMode(b.dataset.mode)));
  $("labReset").addEventListener("click", labResetToActual);
  $("labRandom").addEventListener("click", labRandomize);

  buildHeroChart();
  buildRaceChart();
  await loadHeatmap();
  await loadBundle();
}

// ───────────────────────── Bundle / asset ─────────────────────────
async function switchAsset(asset) {
  if (ST.asset === asset) return;
  ST.asset = asset;
  document.querySelectorAll(".asset-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.asset === asset));
  await loadBundle();
}

async function loadBundle() {
  ST.arch  = $("arch").value;
  ST.model = $("model").value;
  ST.rule  = $("rule").value;
  ST.idx   = -1;
  // Reset race when underlying (asset, arch, model, rule) changes — race only
  // makes sense within a single configuration.
  ST.raceStart = -1;
  resetRaceChart();
  setStatus("LOADING…");

  const u = new URL("/bundle", window.location.origin);
  u.searchParams.set("asset", ST.asset);
  u.searchParams.set("arch",  ST.arch);
  u.searchParams.set("model", ST.model);
  u.searchParams.set("rule",  ST.rule);

  const t0 = performance.now();
  const res = await fetch(u.toString());
  if (!res.ok) {
    setStatus("ERROR");
    console.error("bundle fetch failed", res.status);
    return;
  }
  const bundle = await res.json();
  ST.bundle = bundle;

  // Update topbar with resolved config
  $("statusDate").textContent = `${bundle.dates[0]} → ${bundle.dates[bundle.n - 1]}`;
  setStatus("LIVE OOF");
  $("footerMeta").textContent =
    `bundle ${(JSON.stringify(bundle).length / 1024).toFixed(0)} KB · ${(performance.now() - t0).toFixed(0)} ms · ${bundle.label}`;

  // Update topbar ticker price (latest close)
  const last = bundle.n - 1;
  $(`tickerPx${ST.asset}`).textContent = fmt.price(bundle.ohlc.close[last]);

  // Render
  renderHero(bundle);
  renderScrubber(bundle);
  renderHeatmapHighlight(bundle);
  seek(last);  // start at most recent
}

// ───────────────────────── Hero chart ─────────────────────────
function buildHeroChart() {
  const container = $("heroChart");
  const chart = LightweightCharts.createChart(container, {
    layout: { background: { color: COLORS.bg }, textColor: COLORS.txt2,
              fontFamily: "JetBrains Mono, monospace", fontSize: 10 },
    grid:   { vertLines: { color: COLORS.line }, horzLines: { color: COLORS.line } },
    timeScale: { borderColor: "rgba(120,144,184,0.18)", timeVisible: false,
                 secondsVisible: false, rightOffset: 4, barSpacing: 3 },
    rightPriceScale: { borderColor: "rgba(120,144,184,0.18)", scaleMargins: { top: 0.05, bottom: 0.30 } },
    crosshair: { mode: 1 },
    handleScroll: true, handleScale: true,
    width: container.clientWidth, height: container.clientHeight,
  });

  // 1) Candlestick (price)
  const candle = chart.addCandlestickSeries({
    upColor: COLORS.buy, downColor: COLORS.sell,
    wickUpColor: COLORS.buy, wickDownColor: COLORS.sell,
    borderVisible: false,
    priceFormat: { type: "price", precision: 2, minMove: 0.01 },
  });

  // 2) Regime ribbon (histogram, bottom strip)
  const regime = chart.addHistogramSeries({
    priceScaleId: "regime",
    priceFormat: { type: "volume" },
    base: 0,
  });
  chart.priceScale("regime").applyOptions({
    scaleMargins: { top: 0.93, bottom: 0.0 },
    visible: false,
  });

  // 3) Volume (very small histogram)
  const volume = chart.addHistogramSeries({
    priceScaleId: "volume",
    priceFormat: { type: "volume" },
    color: "rgba(120,144,184,0.35)",
  });
  chart.priceScale("volume").applyOptions({
    scaleMargins: { top: 0.78, bottom: 0.07 },
    visible: false,
  });

  // Resize: keep chart sized to its container, and reposition the
  // selected-date vertical line (its x-pixel changes with chart width / zoom).
  new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
    if (ST.bundle && ST.idx >= 0) highlightHeroDate(ST.bundle, ST.idx);
  }).observe(container);

  // When user zooms/pans the time-axis the vertical line must follow.
  chart.timeScale().subscribeVisibleLogicalRangeChange(() => {
    if (ST.bundle && ST.idx >= 0) highlightHeroDate(ST.bundle, ST.idx);
  });

  // Click to seek
  chart.subscribeClick((param) => {
    if (!ST.bundle || !param.time) return;
    const idx = ST.bundle.dates.indexOf(param.time);
    if (idx >= 0) seek(idx);
  });

  ST.hero.chart  = chart;
  ST.hero.candle = candle;
  ST.hero.regime = regime;
  ST.hero.volume = volume;
}

function renderHero(bundle) {
  const candleData = bundle.dates.map((d, i) => ({
    time: d,
    open:  bundle.ohlc.open[i],
    high:  bundle.ohlc.high[i],
    low:   bundle.ohlc.low[i],
    close: bundle.ohlc.close[i],
  }));
  ST.hero.candle.setData(candleData);

  // regime strip — colored bars, value=1 always (height fixed via scaleMargins)
  const regimeData = bundle.dates.map((d, i) => {
    const lbl = bundle.regime.label[i];
    const c = lbl === "Bull" ? "rgba(0,214,143,0.65)" :
              lbl === "Bear" ? "rgba(255,85,119,0.65)" :
                                "rgba(255,181,71,0.45)";
    return { time: d, value: 1.0, color: c };
  });
  ST.hero.regime.setData(regimeData);

  // volume
  const volumeData = bundle.dates.map((d, i) => {
    const c = bundle.ohlc.close[i] >= bundle.ohlc.open[i]
            ? "rgba(0,255,159,0.30)" : "rgba(255,56,96,0.30)";
    return { time: d, value: bundle.ohlc.volume[i], color: c };
  });
  ST.hero.volume.setData(volumeData);

  // Trade markers removed: 179+ markers with text labels overlap and become
  // unreadable. Trade details are shown in the side panels (Outcomes + Trade
  // simulation). Hero chart stays clean — only the selected-date marker is
  // drawn by highlightHeroDate().
  ST.hero.candle.setMarkers([]);

  ST.hero.chart.timeScale().fitContent();
}

function highlightHeroDate(bundle, idx) {
  // Markers shown on the hero candle chart:
  //   - the blue selected-date dot (always)
  //   - if a portfolio race is in progress (raceStart >= 0), Buy ▲ /
  //     Sell ▼ markers for every trade entry/exit between raceStart and
  //     the current scrubber index. Pressing ▶ again later resets these
  //     because raceStart shifts to the new start, so the filter window
  //     naturally clears prior arrows.
  // Plus a DOM-overlay vertical line at the selected date.
  const sel = bundle.dates[idx];

  const markers = [];
  if (ST.raceStart >= 0) {
    // If the race began inside an open trade, show a faint ghost entry at
    // raceStart so the next exit is visually grounded ("we were already
    // in this trade when ▶ was pressed"). Otherwise the chart shows a
    // bare red exit arrow with no preceding green — confusing.
    const startPos = bundle.positions[ST.raceStart] ?? 0;
    if (startPos > 0) {
      markers.push({
        time: bundle.dates[ST.raceStart],
        position: "belowBar",
        color: "rgba(0, 255, 159, 0.45)",   // half-opacity buy color
        shape: "arrowUp",
        text: "",
      });
    }

    for (const t of bundle.trades) {
      // Show entry only if the trade actually opened inside the race window.
      const entryInWindow = t.entry_idx >= ST.raceStart && t.entry_idx <= idx;
      if (entryInWindow) {
        markers.push({
          time: bundle.dates[t.entry_idx],
          position: "belowBar",
          color: COLORS.buy,
          shape: "arrowUp",
          text: "",
        });
      }
      // Show exit only if BOTH entry and exit are inside the race window.
      // Otherwise we'd render an exit with no matching entry — exactly the
      // "first marker is a red exit" bug the user observed.
      if (entryInWindow && t.exit_idx != null && t.exit_idx <= idx && t.exit_date) {
        // Exit always shown in red regardless of trade P&L — color encodes
        // direction (entry = green up, exit = red down), not outcome.
        markers.push({
          time: bundle.dates[t.exit_idx],
          position: "aboveBar",
          color: COLORS.sell,
          shape: "arrowDown",
          text: "",
        });
      }
    }
  }
  // Selected-date dot is always last and on top.
  markers.push({
    time: sel,
    position: "inBar",
    color: "#4d8aff",
    shape: "circle",
    text: "",
  });
  // LightweightCharts requires markers in chronological order.
  markers.sort((a, b) => a.time < b.time ? -1 : a.time > b.time ? 1 : 0);
  ST.hero.candle.setMarkers(markers);

  const vline = $("heroVLine");
  if (!vline || !ST.hero.chart) return;
  const x = ST.hero.chart.timeScale().timeToCoordinate(sel);
  if (x == null) {
    vline.style.display = "none";
    return;
  }
  vline.style.left = x.toFixed(1) + "px";
  vline.style.display = "block";
}

// ───────────────────────── Scrubber ─────────────────────────
function renderScrubber(bundle) {
  const sb = $("scrubber");
  sb.min = 0; sb.max = bundle.n - 1; sb.value = bundle.n - 1;
  $("scrubFirst").textContent = bundle.dates[0];
  $("scrubMid").textContent   = bundle.dates[Math.floor(bundle.n / 2)];
  $("scrubLast").textContent  = bundle.dates[bundle.n - 1];
  // Mini-equity ghost canvas removed (the dedicated PORTFOLIO RACE panel
  // serves the same purpose and is more legible).
}

function seek(idx) {
  if (!ST.bundle) return;
  idx = Math.max(0, Math.min(idx, ST.bundle.n - 1));
  ST.idx = idx;
  $("scrubber").value = idx;
  refreshAll();
}

function setSpeed(speed) {
  ST.speed = speed;
  document.querySelectorAll(".speed-btn").forEach(b =>
    b.classList.toggle("active", parseInt(b.dataset.speed, 10) === speed));
  if (ST.playing) {
    stopPlay();
    startPlay();
  }
}

function togglePlay() {
  if (ST.playing) {
    stopPlay();
  } else {
    // Pressing Play kicks off a fresh portfolio race from the current date.
    // (setSpeed uses stop+start internally and must NOT reset the race —
    //  hence raceStart is set here rather than inside startPlay.)
    if (ST.bundle) {
      ST.raceStart = ST.idx;
      resetRaceChart();
      updateRaceChart(ST.bundle, ST.idx);  // initial point at 0%
    }
    startPlay();
  }
}

function startPlay() {
  if (!ST.bundle) return;
  ST.playing = true;
  $("playBtn").textContent = "⏸";
  $("playBtn").classList.add("playing");
  // 1x = 10 days/sec → 100ms/day. 3x = 30 d/s → 33ms. 6x = 60 d/s → 16ms.
  const periodMs = 1000 / (10 * ST.speed);
  ST.playerHandle = setInterval(() => {
    if (!ST.bundle) return;
    if (ST.idx >= ST.bundle.n - 1) { stopPlay(); return; }
    seek(ST.idx + 1);
  }, periodMs);
}

function stopPlay() {
  ST.playing = false;
  $("playBtn").textContent = "▶";
  $("playBtn").classList.remove("playing");
  if (ST.playerHandle) { clearInterval(ST.playerHandle); ST.playerHandle = null; }
}

// ───────────────────────── Date-driven panel updates ─────────────────────────
function refreshAll() {
  const b = ST.bundle, i = ST.idx;
  if (!b || i < 0) return;

  // Topbar status date
  $("statusDate").textContent = b.dates[i];
  $(`tickerPx${ST.asset}`).textContent = fmt.price(b.ohlc.close[i]);

  // Hero highlight
  highlightHeroDate(b, i);

  // Signal panel
  updateSignal(b, i);
  // Stage 1 gauge
  updateStage1(b, i);
  // Stage 2 orb
  updateStage2(b, i);
  // Votes
  updateVotes(b, i);
  // Reasons
  updateReasons(b, i);
  // Outcomes
  updateOutcomes(b, i);

  // If lab is open and in 'actual' mode, sync sliders with current date
  if (ST.labOpen && ST.labMode === "actual") syncLabFromActual();
  // If lab is open and 'custom', re-run custom inference (date doesn't matter for inference but the from-arrow shows actual)
  if (ST.labOpen) updateLabFromArrow(b, i);

  // Portfolio race chart — only active when a race has been started (Play pressed)
  updateRaceChart(b, i);
}

// ───────────────────────── Portfolio race (vanilla canvas) ─────────────────────────
//
// Replaced Chart.js with a hand-rolled canvas to eliminate ResizeObserver +
// per-tick parsing overhead. Renders only when the race is active.

const RACE = {
  ctx:    null,
  pixelW: 0,
  pixelH: 0,
  cssW:   0,
  cssH:   0,
  pad:    { top: 8, right: 12, bottom: 18, left: 38 },
  rafPending: false,
  pending: null,  // {b, i, stratPct, bhPct, days, lab}
};

function buildRaceChart() {
  const cv = $("raceChart");
  RACE.ctx = cv.getContext("2d");

  // Throttled resize via window resize event only (NOT ResizeObserver).
  // Page scroll never fires window resize, so we get zero scroll jank.
  let resizeTimer = null;
  const onResize = () => {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      sizeRaceCanvas();
      // Re-render with last known state if any.
      if (RACE.pending) drawRaceFromPending();
    }, 150);
  };
  window.addEventListener("resize", onResize, { passive: true });

  sizeRaceCanvas();
  resetRaceChart();
}

function sizeRaceCanvas() {
  const cv  = $("raceChart");
  const dpr = window.devicePixelRatio || 1;
  RACE.cssW   = cv.clientWidth;
  RACE.cssH   = cv.clientHeight;
  RACE.pixelW = Math.round(RACE.cssW * dpr);
  RACE.pixelH = Math.round(RACE.cssH * dpr);
  cv.width  = RACE.pixelW;
  cv.height = RACE.pixelH;
  RACE.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);  // scale once, draw in CSS px
}

function resetRaceChart() {
  RACE.pending = null;
  if (RACE.ctx) {
    RACE.ctx.clearRect(0, 0, RACE.cssW, RACE.cssH);
    drawRaceEmpty("press ▶ to start a race");
  }
  $("raceStartLabel").textContent = "— press ▶ to start a race —";
  $("raceStrategyPill").textContent = "Strategy 0.00%";
  $("raceBHPill").textContent       = "B&H 0.00%";
  setRaceEdgePill(0);
}

function setRaceEdgePill(edge) {
  const el = $("raceEdgePill");
  el.textContent = `Edge ${edge >= 0 ? "+" : ""}${edge.toFixed(2)}pp`;
  el.classList.remove("positive", "negative");
  if (edge >  0.005) el.classList.add("positive");
  if (edge < -0.005) el.classList.add("negative");
}

function updateRaceChart(b, i) {
  if (!RACE.ctx || !b) return;
  const start = ST.raceStart;

  if (start < 0 || i < start) {
    if (RACE.pending) resetRaceChart();
    return;
  }

  const stratBase = b.equity[start];
  const bhBase    = b.bh_equity[start];
  if (!stratBase || !bhBase) return;

  // Read slices once; do NOT allocate intermediate arrays per tick — compute
  // min/max + last value in a single pass.
  const len = i - start + 1;
  let minY = Infinity, maxY = -Infinity;
  let lastS = 0, lastB = 0;
  // Cache last computed series (for redraw on resize). We only store the
  // *normalized* arrays here, which are short-lived and reused.
  const stratArr = new Float32Array(len);
  const bhArr    = new Float32Array(len);
  for (let k = 0; k < len; k++) {
    const idx = start + k;
    const sP = (b.equity[idx]    / stratBase - 1) * 100;
    const bP = (b.bh_equity[idx] / bhBase    - 1) * 100;
    stratArr[k] = sP;
    bhArr[k]    = bP;
    if (sP < minY) minY = sP;
    if (sP > maxY) maxY = sP;
    if (bP < minY) minY = bP;
    if (bP > maxY) maxY = bP;
    lastS = sP; lastB = bP;
  }
  // Pad y-range a tiny bit so lines don't kiss the borders
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const yPad = (maxY - minY) * 0.08;
  minY -= yPad; maxY += yPad;

  RACE.pending = {
    stratArr, bhArr, len,
    minY, maxY,
    startDate: b.dates[start],
    days: i - start,
  };
  scheduleRaceDraw();

  // Pills
  const edge = lastS - lastB;
  $("raceStartLabel").textContent =
    `Race since ${b.dates[start]} · ${i - start} day${(i - start) === 1 ? "" : "s"}`;
  $("raceStrategyPill").textContent = `Strategy ${lastS >= 0 ? "+" : ""}${lastS.toFixed(2)}%`;
  $("raceBHPill").textContent       = `B&H ${lastB >= 0 ? "+" : ""}${lastB.toFixed(2)}%`;
  setRaceEdgePill(edge);
}

function scheduleRaceDraw() {
  if (RACE.rafPending) return;
  RACE.rafPending = true;
  requestAnimationFrame(() => {
    RACE.rafPending = false;
    drawRaceFromPending();
  });
}

function drawRaceFromPending() {
  if (!RACE.pending) return;
  const { stratArr, bhArr, len, minY, maxY, days } = RACE.pending;
  drawRaceFrame(stratArr, bhArr, len, minY, maxY, days);
}

function drawRaceFrame(stratArr, bhArr, len, minY, maxY, days) {
  const ctx = RACE.ctx;
  const W   = RACE.cssW;
  const H   = RACE.cssH;
  const P   = RACE.pad;
  const plotW = W - P.left - P.right;
  const plotH = H - P.top  - P.bottom;

  ctx.clearRect(0, 0, W, H);

  // Y mapping
  const span = (maxY - minY) || 1;
  const yPx  = (v) => P.top + plotH * (1 - (v - minY) / span);
  const xPx  = (k) => P.left + (len > 1 ? plotW * (k / (len - 1)) : plotW / 2);

  // Grid + zero line
  ctx.strokeStyle = "rgba(120,144,184,0.10)";
  ctx.lineWidth   = 1;
  ctx.font        = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle   = "#6b7895";
  ctx.textBaseline = "middle";
  ctx.textAlign   = "right";

  // 3 horizontal grid lines (min, mid, max)
  const ySteps = 3;
  for (let s = 0; s <= ySteps; s++) {
    const v = minY + (maxY - minY) * (s / ySteps);
    const y = yPx(v);
    ctx.beginPath();
    ctx.moveTo(P.left, y);
    ctx.lineTo(W - P.right, y);
    ctx.stroke();
    ctx.fillText((v >= 0 ? "+" : "") + v.toFixed(0) + "%", P.left - 6, y);
  }
  // Highlight zero line if visible
  if (minY < 0 && maxY > 0) {
    const y0 = yPx(0);
    ctx.strokeStyle = "rgba(179,191,214,0.35)";
    ctx.beginPath();
    ctx.moveTo(P.left, y0);
    ctx.lineTo(W - P.right, y0);
    ctx.stroke();
  }

  // X tick: just "Day 0" and "Day N"
  ctx.textAlign    = "left";
  ctx.fillStyle    = "#6b7895";
  ctx.fillText("Day 0", P.left, H - 4);
  ctx.textAlign    = "right";
  ctx.fillText(`Day ${days}`, W - P.right, H - 4);

  // B&H line (gray dashed) — drawn first so Strategy is on top
  ctx.strokeStyle = "rgba(179,191,214,0.85)";
  ctx.lineWidth   = 1.4;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  // Stride if many points so we don't draw 2400 segments per frame
  const stride = Math.max(1, Math.floor(len / 600));
  let started = false;
  for (let k = 0; k < len; k += stride) {
    const x = xPx(k);
    const y = yPx(bhArr[k]);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  // ensure last point drawn
  if ((len - 1) % stride !== 0) {
    ctx.lineTo(xPx(len - 1), yPx(bhArr[len - 1]));
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Strategy line + soft fill
  // Fill region first (Strategy down to baseline)
  ctx.beginPath();
  started = false;
  for (let k = 0; k < len; k += stride) {
    const x = xPx(k);
    const y = yPx(stratArr[k]);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  if ((len - 1) % stride !== 0) {
    ctx.lineTo(xPx(len - 1), yPx(stratArr[len - 1]));
  }
  ctx.lineTo(xPx(len - 1), P.top + plotH);
  ctx.lineTo(xPx(0),       P.top + plotH);
  ctx.closePath();
  ctx.fillStyle = "rgba(0, 255, 159, 0.10)";
  ctx.fill();

  ctx.strokeStyle = COLORS.buy;
  ctx.lineWidth   = 2;
  ctx.beginPath();
  started = false;
  for (let k = 0; k < len; k += stride) {
    const x = xPx(k);
    const y = yPx(stratArr[k]);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else ctx.lineTo(x, y);
  }
  if ((len - 1) % stride !== 0) {
    ctx.lineTo(xPx(len - 1), yPx(stratArr[len - 1]));
  }
  ctx.stroke();
}

function drawRaceEmpty(msg) {
  const ctx = RACE.ctx;
  ctx.clearRect(0, 0, RACE.cssW, RACE.cssH);
  ctx.fillStyle = "#4a5570";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(msg, RACE.cssW / 2, RACE.cssH / 2);
}

function updateSignal(b, i) {
  const sig = b.active_signals[i] || "—";
  const probs = {
    Buy:  b.active_probs.Buy[i]  || 0,
    Hold: b.active_probs.Hold[i] || 0,
    Sell: b.active_probs.Sell[i] || 0,
  };
  const conf = probs[sig] || 0;

  const lbl = $("signalLabel");
  lbl.textContent = sig;
  lbl.className = "signal-label " + sig.toLowerCase();
  $("signalConf").textContent = (conf * 100).toFixed(1) + "%";

  $("pBuyFill").style.width  = (probs.Buy  * 100) + "%";
  $("pHoldFill").style.width = (probs.Hold * 100) + "%";
  $("pSellFill").style.width = (probs.Sell * 100) + "%";
  $("pBuy").textContent  = probs.Buy.toFixed(3);
  $("pHold").textContent = probs.Hold.toFixed(3);
  $("pSell").textContent = probs.Sell.toFixed(3);
}

function updateStage1(b, i) {
  const up = b.stage1.P_up[i]    || 0;
  const ra = b.stage1.P_range[i] || 0;
  const dn = b.stage1.P_down[i]  || 0;
  $("s1Up").textContent    = up.toFixed(3);
  $("s1Range").textContent = ra.toFixed(3);
  $("s1Down").textContent  = dn.toFixed(3);
  const dom = (up >= ra && up >= dn) ? ["UP", "up"]
            : (dn >= ra) ? ["DOWN", "down"]
            : ["RANGE", "range"];
  const cls = $("s1Class");
  cls.textContent = dom[0];
  cls.className = "gauge-class " + dom[1];

  drawGauge($("gaugeS1"), {up, range: ra, down: dn});
}

function drawGauge(svg, p) {
  // Half-circle gauge with three arcs (down=red, range=yellow, up=green).
  // Total angle = 180°, segment lengths proportional to posterior probabilities.
  //
  // Bug fixes vs previous version:
  //  - large-arc-flag is ALWAYS 0 (no single segment can exceed 180°),
  //    not "1 if p > 0.5" (which produced inverted long-way arcs and
  //    rendering glitches at p=1).
  //  - Cap each sweep at 0.998·π so a p=1 segment never hits an exact
  //    180°, where SVG arc semantics are ambiguous and some renderers
  //    drop the arc.
  //  - Solid butt linecap so adjacent arcs join cleanly (no overlap halo).
  //  - Gray background track always drawn so the panel never looks empty
  //    if a class is 0.
  const total = (p.down + p.range + p.up) || 1;
  const r = 50, cx = 60, cy = 60;
  const segs = [
    { p: p.down  / total, color: "#ff3860" },
    { p: p.range / total, color: "#ffb547" },
    { p: p.up    / total, color: "#00ff9f" },
  ];

  // Background track (always full semicircle).
  // sweep-flag = 0 → counter-clockwise from (10,60) to (110,60) which in
  // SVG (y grows downward) traces the UPPER half-circle. sweep-flag = 1
  // would draw the lower half, which is what the previous code did and is
  // why the arcs were overflowing the panel.
  let html = `<path d="M 10 60 A 50 50 0 0 0 110 60"
                stroke="rgba(120,144,184,0.18)" stroke-width="9"
                fill="none" stroke-linecap="butt" />`;

  // We draw the upper half-circle, so y = cy - r·sin(angle), not cy + r·sin.
  let a0 = 0;            // start at the LEFT end of the semicircle (angle 0
                          // measured from +x axis going CCW; we'll flip Y)
  // Conceptually: left end = 180°, right end = 0°. We'll iterate segments
  // each consuming a fraction of π radians, walking from left to right.
  // The path is described in SVG coords where Y is mirrored: we use
  //   x = cx + r·cos(θ),  y = cy − r·sin(θ)
  // and θ goes from π (left) down to 0 (right).
  let theta = Math.PI;
  for (const s of segs) {
    const sweep = Math.min(s.p, 0.998) * Math.PI;
    const theta1 = theta - sweep;
    if (s.p >= 0.005) {
      const x0 = cx + r * Math.cos(theta),  y0 = cy - r * Math.sin(theta);
      const x1 = cx + r * Math.cos(theta1), y1 = cy - r * Math.sin(theta1);
      // sweep-flag = 0 (counter-clockwise) keeps the arc on the upper half.
      html += `<path d="M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 0 0 ${x1.toFixed(2)} ${y1.toFixed(2)}"
                  stroke="${s.color}" stroke-width="9" fill="none" stroke-linecap="butt" />`;
    }
    theta = theta1;
  }
  svg.innerHTML = html;
}

function updateStage2(b, i) {
  const lbl = (b.regime.label[i] || "neutral").toLowerCase();
  const cls = lbl === "bull" ? "bull" : lbl === "bear" ? "bear" : "neutral";
  $("regimeLabel").textContent = lbl.toUpperCase();
  $("regimeLabel").className = "orb-label " + cls;
  document.querySelector("#regimeOrb .orb-ring").className = "orb-ring " + cls;

  const bull = b.regime.P_Bull[i] || 0;
  const neut = b.regime.P_Neutral[i] || 0;
  const bear = b.regime.P_Bear[i] || 0;
  const age  = b.regime.age_days[i];
  $("s2Bull").textContent = bull.toFixed(2);
  $("s2Neut").textContent = neut.toFixed(2);
  $("s2Bear").textContent = bear.toFixed(2);
  $("s2Age").textContent  = (age == null ? "—" : `${age}d`);
}

function updateVotes(b, i) {
  const order = ["xgboost", "lightgbm", "random_forest", "mlp"];
  const labels = { xgboost: "XGB", lightgbm: "LGBM", random_forest: "RF", mlp: "MLP" };
  const body = $("votesBody");
  body.innerHTML = "";

  let counts = { Buy: 0, Hold: 0, Sell: 0 };
  for (const m of order) {
    const v = b.votes[m];
    const row = document.createElement("div");
    row.className = "vote-row" + (m === b.model ? " active" : " dim");

    if (!v) {
      row.innerHTML = `<span class="vote-name">${labels[m]}</span>
                       <span class="vote-bar"></span>
                       <span class="vote-pred">—</span>`;
    } else {
      const pBuy  = v.P_Buy[i]  || 0;
      const pHold = v.P_Hold[i] || 0;
      const pSell = v.P_Sell[i] || 0;
      const pred  = v.pred[i] || "—";
      counts[pred] = (counts[pred] || 0) + 1;
      row.innerHTML = `
        <span class="vote-name">${labels[m]}</span>
        <span class="vote-bar">
          <span class="vote-segment buy"  style="flex-grow: ${pBuy};"></span>
          <span class="vote-segment hold" style="flex-grow: ${pHold};"></span>
          <span class="vote-segment sell" style="flex-grow: ${pSell};"></span>
        </span>
        <span class="vote-pred ${pred.toLowerCase()}">${pred}</span>
      `;
    }
    body.appendChild(row);
  }
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  $("votesAgg").textContent = `${top[1]}/4 → ${top[0]}`;
}

function updateReasons(b, i) {
  const reasons = computeReasons(b, i);
  const ul = $("reasonsList");
  ul.innerHTML = "";
  if (reasons.length === 0) {
    ul.innerHTML = `<li><span class="rank">·</span><span class="desc">No standout features — confidence-driven.</span><span class="strength"></span></li>`;
    return;
  }
  reasons.slice(0, 6).forEach((r, k) => {
    const li = document.createElement("li");
    li.className = r.tone;  // bull / bear / neutral
    li.innerHTML = `
      <span class="rank">${k + 1}</span>
      <span class="desc">${r.text} <small>${r.detail || ""}</small></span>
      <span class="strength">${r.value}</span>`;
    ul.appendChild(li);
  });
}

function computeReasons(b, i) {
  const out = [];
  // Stage 1
  const s1 = { up: b.stage1.P_up[i] || 0, range: b.stage1.P_range[i] || 0, down: b.stage1.P_down[i] || 0 };
  const s1Top = Object.entries(s1).sort((a, b) => b[1] - a[1])[0];
  if (s1Top[1] > 0.45) {
    const tone = s1Top[0] === "up" ? "bull" : s1Top[0] === "down" ? "bear" : "neutral";
    out.push({ text: `Stage 1 trend dominant: <b>${s1Top[0].toUpperCase()}</b>`,
               value: s1Top[1].toFixed(2), tone });
  }
  // Stage 2 regime
  const s2 = { Bull: b.regime.P_Bull[i] || 0, Neutral: b.regime.P_Neutral[i] || 0, Bear: b.regime.P_Bear[i] || 0 };
  const s2Top = Object.entries(s2).sort((a, b) => b[1] - a[1])[0];
  if (s2Top[1] > 0.5) {
    const tone = s2Top[0] === "Bull" ? "bull" : s2Top[0] === "Bear" ? "bear" : "neutral";
    out.push({ text: `Stage 2 macro: <b>${s2Top[0]}</b> regime`,
               value: s2Top[1].toFixed(2), tone });
  }
  // RSI
  const rsi = b.osc.RSI_14[i];
  if (rsi != null) {
    if (rsi < 30)  out.push({ text: "RSI oversold (bullish reversal zone)", detail: `(${rsi.toFixed(1)})`, value: rsi.toFixed(1), tone: "bull" });
    else if (rsi > 70)  out.push({ text: "RSI overbought", detail: `(${rsi.toFixed(1)})`, value: rsi.toFixed(1), tone: "bear" });
  }
  // MACD
  const macd = b.osc.MACD_signal_diff[i];
  if (macd != null) {
    if (macd > 100)  out.push({ text: "Strong bullish MACD momentum", value: macd.toFixed(0), tone: "bull" });
    else if (macd < -100) out.push({ text: "Strong bearish MACD momentum", value: macd.toFixed(0), tone: "bear" });
  }
  // Bollinger
  const bb = b.osc.Bollinger_pct_b[i];
  if (bb != null) {
    if (bb < 0)       out.push({ text: "Below lower Bollinger band (oversold)", value: bb.toFixed(2), tone: "bull" });
    else if (bb > 1)  out.push({ text: "Above upper Bollinger band (overbought)", value: bb.toFixed(2), tone: "bear" });
  }
  // Stochastic
  const stoch = b.osc.Stochastic_K_14[i];
  if (stoch != null) {
    if (stoch < 20)  out.push({ text: "Stochastic oversold", value: stoch.toFixed(0), tone: "bull" });
    else if (stoch > 80) out.push({ text: "Stochastic overbought", value: stoch.toFixed(0), tone: "bear" });
  }
  // Volume z-score
  const volz = b.osc.volume_zscore_20[i];
  if (volz != null && volz > 1.5) out.push({ text: "Above-average volume spike", value: volz.toFixed(1), tone: "neutral" });
  // OBV
  const obv = b.osc.OBV_change_20d[i];
  if (obv != null) {
    if (obv > 0.05)        out.push({ text: "Volume accumulation (OBV up)", value: obv.toFixed(2), tone: "bull" });
    else if (obv < -0.05)  out.push({ text: "Volume distribution (OBV down)", value: obv.toFixed(2), tone: "bear" });
  }
  return out;
}

function updateOutcomes(b, i) {
  const close = b.ohlc.close;
  const set = (id, h) => {
    const j = i + h;
    const cell = $(id);
    if (j >= b.n) {
      cell.className = "outcome-cell";
      cell.innerHTML = `<label>${h}d price</label><b>—</b><small>future</small>`;
      return;
    }
    const r = (close[j] - close[i]) / close[i];
    cell.className = "outcome-cell " + (r >= 0 ? "pos" : "neg");
    cell.innerHTML = `<label>${h}d price</label><b>${fmt.pct(r, 1)}</b><small>${fmt.price(close[j])}</small>`;
  };
  set("oc5", 5); set("oc10", 10); set("oc30", 30);

  // ── Strategy decision (rule-aware): what does our trading rule actually do
  // on this day, given the position trajectory in bundle.positions[]?
  // This is what really matters for backtest P&L; the BUY/HOLD/SELL signal
  // alone does not yet imply a trade.
  const stratEl = $("stratDecText");
  const pos     = b.positions[i] ?? 0;
  const posPrev = i > 0 ? (b.positions[i - 1] ?? 0) : 0;
  const sigNow  = b.active_signals[i];
  let decClass, decText;
  if (pos > 0 && posPrev === 0) {
    decClass = "enter";
    decText  = `▲ ENTERED long position` +
               (b.rule === "prob_weighted" ? ` · size ${pos.toFixed(2)}` : "");
  } else if (pos > 0 && posPrev > 0) {
    decClass = "hold";
    const sizeChange = pos - posPrev;
    if (Math.abs(sizeChange) > 0.005) {
      decText = `→ HOLDING long · size ${posPrev.toFixed(2)} → ${pos.toFixed(2)}`;
    } else {
      decText = `→ HOLDING long position` +
                (b.rule === "prob_weighted" ? ` · size ${pos.toFixed(2)}` : "");
    }
  } else if (pos === 0 && posPrev > 0) {
    decClass = "exit";
    decText  = `▼ EXITED long position`;
  } else {
    decClass = "cash";
    decText  = `· STAY in cash` + (sigNow === "Buy" || sigNow === "Sell" ? ` (rule kept us out despite ${sigNow} signal)` : "");
  }
  stratEl.className = "strat-decision-text " + decClass;
  stratEl.textContent = decText;

  // ── "If we acted on this signal" — finds next opposite signal as the
  // exit point, regardless of the rule. Pedagogical, not the real backtest
  // (which uses positions[] above).
  const sim = $("tradeSim");
  if (sigNow !== "Buy" && sigNow !== "Sell") {
    sim.innerHTML = `<span class="tk">trade sim</span><span class="tv">no entry — Hold day</span>`;
    return;
  }
  const opposite = sigNow === "Buy" ? "Sell" : "Buy";
  let exit = b.n - 1;
  for (let j = i + 1; j < b.n; j++) {
    if (b.active_signals[j] === opposite) { exit = j; break; }
  }
  const p0 = close[i], p1 = close[exit];
  const pnl = sigNow === "Buy" ? (p1 - p0) / p0 : (p0 - p1) / p0;
  const days = exit - i;
  sim.innerHTML = `
    <span class="sim-head">if you acted on the ${sigNow} signal:</span>
    <span><span class="tk">entry</span><span class="tv">${b.dates[i]} @ ${fmt.price(p0)}</span></span>
    <span><span class="tk">exit (next ${opposite})</span><span class="tv">${b.dates[exit]} @ ${fmt.price(p1)}</span></span>
    <span><span class="tk">held</span><span class="tv">${days}d</span></span>
    <span><span class="tk">P&L</span><span class="tv ${pnl >= 0 ? 'pos' : 'neg'}">${fmt.pct(pnl, 2)}</span></span>
  `;
}

function setStatus(label) {
  $("statusLabel").textContent = label;
}

// ───────────────────────── Heatmap ─────────────────────────
async function loadHeatmap() {
  const res = await fetch("/heatmap");
  if (!res.ok) return;
  const data = await res.json();
  ST.heatmap = data;
  renderHeatmap(data);
}

function renderHeatmap(data) {
  const grid = $("heatmap");
  const archs = ["flat", "2stage_trend", "2stage_macro", "3stage_full"];
  const models = ["xgboost", "lightgbm", "random_forest", "mlp"];
  const archLabels = { flat: "Flat", "2stage_trend": "2S Trend", "2stage_macro": "2S Macro", "3stage_full": "3-Stage Full" };
  const modelLabels = { xgboost: "XGB", lightgbm: "LGBM", random_forest: "RF", mlp: "MLP" };

  // Sharpe range for color scale
  const sharpes = data.cells.map(c => c.sharpe).filter(v => v != null);
  const sMin = Math.min(...sharpes), sMax = Math.max(...sharpes);
  const colorFor = (s) => {
    if (s == null) return "#1a2238";
    const t = (s - sMin) / (sMax - sMin || 1);
    // green gradient: low = #182039, high = #00ff9f
    const r = Math.round(24 + (0 - 24) * t);
    const g = Math.round(32 + (255 - 32) * t);
    const b = Math.round(57 + (159 - 57) * t);
    return `rgb(${r},${g},${b})`;
  };

  grid.style.gridTemplateColumns = `90px repeat(${archs.length}, 1fr)`;
  grid.innerHTML = "";

  // Render BTC then ETH
  for (const asset of ["BTC", "ETH"]) {
    // Asset header row
    const ah = document.createElement("div");
    ah.className = "hm-cell hm-label col-asset";
    ah.style.gridColumn = `1 / span ${archs.length + 1}`;
    const bh = data.buy_hold.find(x => x.asset === asset);
    ah.textContent = `${asset}  ·  B&H Sharpe ${bh ? bh.sharpe.toFixed(2) : "—"}  ·  return ${bh ? (bh.total_return * 100).toFixed(0) + "%" : "—"}`;
    grid.appendChild(ah);

    // Header: arch labels
    const corner = document.createElement("div");
    corner.className = "hm-cell hm-label";
    corner.textContent = "model ↓";
    grid.appendChild(corner);
    for (const a of archs) {
      const h = document.createElement("div");
      h.className = "hm-cell hm-label";
      h.textContent = archLabels[a];
      grid.appendChild(h);
    }

    // Each model row
    for (const m of models) {
      const ml = document.createElement("div");
      ml.className = "hm-cell hm-label";
      ml.textContent = modelLabels[m];
      grid.appendChild(ml);

      for (const a of archs) {
        const cell = data.cells.find(c => c.asset === asset && c.arch === a && c.model === m);
        const c = document.createElement("div");
        c.className = "hm-cell";
        c.dataset.asset = asset;
        c.dataset.arch  = a;
        c.dataset.model = m;
        c.dataset.rule  = cell ? cell.rule : "stateful";
        c.style.background = colorFor(cell ? cell.sharpe : null);
        c.style.color = (cell && cell.sharpe > (sMin + sMax) / 2) ? COLORS.bg : COLORS.txt0;
        c.innerHTML = cell
          ? `<b style="font-size:13px">${cell.sharpe.toFixed(2)}</b><br><small style="font-size:8px;opacity:.7">${(cell.total_return * 100).toFixed(0)}%</small>`
          : "—";
        c.addEventListener("click", () => activateHeatmapCell(c));
        grid.appendChild(c);
      }
    }
  }
}

function renderHeatmapHighlight(b) {
  document.querySelectorAll(".heatmap .hm-cell").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".heatmap .hm-cell").forEach(el => {
    if (el.dataset.asset === b.asset && el.dataset.arch === b.arch && el.dataset.model === b.model) {
      el.classList.add("active");
    }
  });
}

function activateHeatmapCell(el) {
  ST.asset = el.dataset.asset;
  ST.arch  = el.dataset.arch;
  ST.model = el.dataset.model;
  ST.rule  = el.dataset.rule;

  // Update topbar
  document.querySelectorAll(".asset-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.asset === ST.asset));
  $("arch").value  = ST.arch;
  $("model").value = ST.model;
  $("rule").value  = ST.rule;
  loadBundle();
}

// ───────────────────────── What-If Lab ─────────────────────────
function toggleLab(open) {
  ST.labOpen = open;
  $("labDrawer").classList.toggle("hidden", !open);
  // Hide the floating WHAT-IF LAB button while the drawer is open so it
  // doesn't overlap the drawer's own controls. It comes back when the
  // drawer is closed (via × in the drawer header).
  $("labToggle").classList.toggle("hidden", open);
  if (open) {
    if (!Object.keys(ST.labFeatures).length) buildLabSliders();
    syncLabFromActual();
  }
}

function setLabMode(mode) {
  ST.labMode = mode;
  document.querySelectorAll(".lab-mode-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.mode === mode));
  if (mode === "actual") {
    syncLabFromActual();
  } else {
    runLabPredict();
  }
}

function buildLabSliders() {
  const body = $("labBody");
  // keep the hint paragraph that was authored in HTML
  const hint = body.querySelector(".lab-hint");
  body.innerHTML = "";
  if (hint) body.appendChild(hint);

  for (const [groupTitle, feats] of Object.entries(LAB_FEATURE_GROUPS)) {
    const g = document.createElement("div");
    g.className = "lab-group";
    g.innerHTML = `<div class="lab-group-title">${groupTitle}</div>`;
    for (const f of feats) {
      const row = document.createElement("div");
      row.className = "lab-slider";
      const desc = f.desc ? `<div class="lab-slider-desc">${f.desc}</div>` : "";
      row.innerHTML = `
        <div class="lab-slider-main">
          <div class="lab-slider-label">${f.label}</div>
          ${desc}
          <input type="range" min="${f.min}" max="${f.max}" step="${f.step}" data-key="${f.key}" />
        </div>
        <div class="lab-slider-value" data-key-val="${f.key}">—</div>
      `;
      g.appendChild(row);
    }
    body.appendChild(g);
  }

  body.querySelectorAll("input[type=range]").forEach(inp => {
    inp.addEventListener("input", (e) => {
      const k = e.target.dataset.key;
      const v = parseFloat(e.target.value);
      ST.labFeatures[k] = v;
      const valEl = body.querySelector(`[data-key-val="${k}"]`);
      if (valEl) {
        valEl.textContent = formatLabValue(k, v);
        valEl.classList.add("changed");
      }
      // switch to custom mode automatically on first change
      if (ST.labMode === "actual") setLabMode("custom");
      else runLabDebounced();
    });
  });
}

function formatLabValue(key, v) {
  if (key === "regime_age_days") return Math.round(v).toString();
  if (Math.abs(v) >= 100) return v.toFixed(0);
  if (Math.abs(v) >= 1)   return v.toFixed(2);
  return v.toFixed(3);
}

function syncLabFromActual() {
  if (!ST.bundle || ST.idx < 0) return;
  const b = ST.bundle, i = ST.idx;
  const features = readActualFeatures(b, i);
  ST.labFeatures = { ...features };
  // Reset baseline so the first /predict_custom result will set a fresh
  // reference for delta comparison in Custom mode.
  ST.labBaselineProbs = null;
  // Update slider DOM
  document.querySelectorAll("#labBody input[type=range]").forEach(inp => {
    const k = inp.dataset.key;
    if (k in features) {
      inp.value = features[k];
      const valEl = document.querySelector(`[data-key-val="${k}"]`);
      if (valEl) {
        valEl.textContent = formatLabValue(k, features[k]);
        valEl.classList.remove("changed");
      }
    }
  });
  updateLabFromArrow(b, i);
  // Single source of truth in Lab Mode: always run the final-fit model so that
  // toggling Actual ↔ Custom never changes the prediction unless features
  // actually change. Walk-forward OOF probs are still shown elsewhere
  // (signal panel, equity, heatmap) — they back the paper-grade numbers.
  runLabPredict();
}

function readActualFeatures(b, i) {
  const out = {
    P1_down:  b.stage1.P_down[i]  || 0,
    P1_range: b.stage1.P_range[i] || 0,
    P1_up:    b.stage1.P_up[i]    || 0,
    P1_down_smooth10:  0,  // not in bundle, use raw as approximation
    P1_range_smooth10: 0,
    P1_up_smooth10:    0,
    P2_Bull:    b.regime.P_Bull[i]    || 0,
    P2_Neutral: b.regime.P_Neutral[i] || 0,
    P2_Bear:    b.regime.P_Bear[i]    || 0,
    regime_age_days: b.regime.age_days[i] || 0,
    RSI_14:           b.osc.RSI_14[i]            ?? 50,
    MACD_signal_diff: b.osc.MACD_signal_diff[i]  ?? 0,
    Bollinger_pct_b:  b.osc.Bollinger_pct_b[i]   ?? 0.5,
    Stochastic_K_14:  b.osc.Stochastic_K_14[i]   ?? 50,
    volume_zscore_20: b.osc.volume_zscore_20[i]  ?? 0,
    OBV_change_20d:   b.osc.OBV_change_20d[i]    ?? 0,
  };
  // smoothed = raw (close-enough proxy when actual smoothed not in bundle)
  out.P1_down_smooth10  = out.P1_down;
  out.P1_range_smooth10 = out.P1_range;
  out.P1_up_smooth10    = out.P1_up;
  return out;
}

function updateLabFromArrow(b, i) {
  const sig = b.active_signals[i] || "—";
  $("labFrom").textContent = sig;
}

function runLabDebounced() {
  if (ST.labDebounce) clearTimeout(ST.labDebounce);
  ST.labDebounce = setTimeout(runLabPredict, 140);
}

async function runLabPredict() {
  if (!ST.bundle) return;
  const arch  = ST.bundle.arch;
  const model = ST.bundle.model;

  const res = await fetch("/predict_custom", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      asset:    ST.bundle.asset,
      arch:     arch,
      model:    model,
      features: ST.labFeatures,
    }),
  });
  if (!res.ok) {
    console.warn("predict_custom failed", res.status);
    return;
  }
  const d = await res.json();
  // First call after syncLabFromActual() establishes the baseline against
  // which subsequent custom edits are compared.
  if (ST.labBaselineProbs == null) {
    ST.labBaselineProbs = { Buy: d.probs.Buy, Hold: d.probs.Hold, Sell: d.probs.Sell, pred: d.pred_label };
  }
  paintLabOutput(d.probs.Buy, d.probs.Hold, d.probs.Sell, d.pred_label);
}

function paintLabOutput(pBuy, pHold, pSell, pred) {
  if (pred == null) {
    pred = (pBuy >= pHold && pBuy >= pSell) ? "Buy" :
           (pSell >= pHold) ? "Sell" : "Hold";
  }
  const to = $("labTo");
  to.textContent = pred;
  to.className = "lab-signal-to " + pred.toLowerCase();
  $("labBuy").style.width  = (pBuy  * 100) + "%";
  $("labHold").style.width = (pHold * 100) + "%";
  $("labSell").style.width = (pSell * 100) + "%";
  $("labBuyVal").textContent  = pBuy.toFixed(3);
  $("labHoldVal").textContent = pHold.toFixed(3);
  $("labSellVal").textContent = pSell.toFixed(3);

  // Delta vs Lab baseline (final-fit on actual features) — only meaningful
  // once a baseline has been established (after the first /predict_custom call).
  // Compares the full probability vector L1 distance, plus headline confidence.
  if (ST.labBaselineProbs) {
    const base = ST.labBaselineProbs;
    const baseConf = base.pred === "Buy" ? base.Buy : base.pred === "Sell" ? base.Sell : base.Hold;
    const newConf  = pred === "Buy" ? pBuy : pred === "Sell" ? pSell : pHold;
    const dBuy  = pBuy  - base.Buy;
    const dHold = pHold - base.Hold;
    const dSell = pSell - base.Sell;
    const l1 = Math.abs(dBuy) + Math.abs(dHold) + Math.abs(dSell);
    if (pred === base.pred && l1 < 1e-6) {
      $("labConfDelta").innerHTML =
        `<span class="muted">unchanged from actual baseline (${base.pred} ${baseConf.toFixed(3)})</span>`;
    } else {
      const sigChanged = pred !== base.pred;
      const cls = sigChanged ? "up" : (newConf > baseConf ? "up" : newConf < baseConf ? "down" : "");
      $("labConfDelta").innerHTML =
        `baseline ${base.pred} ${baseConf.toFixed(3)} → <span class="${cls}">${pred} ${newConf.toFixed(3)}</span> · ` +
        `ΔBuy ${dBuy >= 0 ? "+" : ""}${dBuy.toFixed(3)} ` +
        `ΔHold ${dHold >= 0 ? "+" : ""}${dHold.toFixed(3)} ` +
        `ΔSell ${dSell >= 0 ? "+" : ""}${dSell.toFixed(3)}`;
    }
  } else {
    $("labConfDelta").innerHTML = `<span class="muted">computing baseline…</span>`;
  }
}

function labResetToActual() {
  setLabMode("actual");
}

function labRandomize() {
  setLabMode("custom");
  // Random-but-coherent: random softmax for stages, random oscillators
  const r3 = () => {
    const a = Math.random(), b = Math.random(), c = Math.random();
    const s = a + b + c;
    return [a/s, b/s, c/s];
  };
  const [up, range, down] = r3();
  const [bull, neut, bear] = r3();
  const set = {
    P1_up: up, P1_range: range, P1_down: down,
    P1_up_smooth10: up, P1_range_smooth10: range, P1_down_smooth10: down,
    P2_Bull: bull, P2_Neutral: neut, P2_Bear: bear,
    regime_age_days: Math.floor(Math.random() * 200),
    RSI_14:            10 + Math.random() * 80,
    MACD_signal_diff:  -300 + Math.random() * 600,
    Bollinger_pct_b:   -0.2 + Math.random() * 1.4,
    Stochastic_K_14:   Math.random() * 100,
    volume_zscore_20:  -2 + Math.random() * 5,
    OBV_change_20d:    -0.3 + Math.random() * 0.6,
  };
  ST.labFeatures = set;
  document.querySelectorAll("#labBody input[type=range]").forEach(inp => {
    const k = inp.dataset.key;
    if (k in set) {
      inp.value = set[k];
      const valEl = document.querySelector(`[data-key-val="${k}"]`);
      if (valEl) { valEl.textContent = formatLabValue(k, set[k]); valEl.classList.add("changed"); }
    }
  });
  runLabPredict();
}

// ───────────────────────── Boot ─────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  init().catch(e => { console.error(e); alert("Init failed: " + e.message); });
});
