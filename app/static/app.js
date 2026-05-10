// V5 Hierarchical Trading Signal Classifier — frontend.
// Reads OOF predictions via FastAPI backend (no on-the-fly inference).

const $ = (id) => document.getElementById(id);
let priceChart = null;
let equityChart = null;
let currentTimeline = null;

async function init() {
  const health = await fetch("/health").then(r => r.json());
  $("health").textContent = JSON.stringify(health, null, 2);

  const meta = await fetch("/assets").then(r => r.json());
  const assetSel = $("asset");
  for (const a of meta.assets) {
    const opt = document.createElement("option");
    opt.value = a; opt.textContent = a;
    assetSel.appendChild(opt);
  }

  assetSel.addEventListener("change", onAssetChange);
  $("predictBtn").addEventListener("click", onPredict);

  // Re-render timeline when arch/model/rule changes
  for (const id of ["arch", "model", "rule"]) {
    $(id).addEventListener("change", () => loadTimeline($("asset").value));
  }

  await onAssetChange();
}

async function onAssetChange() {
  const asset = $("asset").value;
  const dates = await fetch(`/test_dates/${asset}`).then(r => r.json());
  const dateSel = $("date");
  dateSel.innerHTML = "";
  for (const d of dates.dates.slice().reverse()) {
    const opt = document.createElement("option");
    opt.value = d; opt.textContent = d;
    dateSel.appendChild(opt);
  }
  dateSel.addEventListener("change", () => updateSelectedDate(dateSel.value));
  await loadTimeline(asset);
}

async function onPredict() {
  const asset = $("asset").value;
  const date  = $("date").value;
  const arch  = $("arch").value;
  const model = $("model").value;
  const rule  = $("rule").value;

  const url = `/predict?asset=${asset}&date=${date}&arch=${arch}&model=${model}&rule=${rule}`;
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(`Predict failed: ${res.status} ${err.detail || ""}`);
    return;
  }
  const data = await res.json();

  $("result").classList.remove("hidden");

  const lbl = $("signalLabel");
  lbl.textContent = data.signal;
  lbl.className = "signal-label " + data.signal.toLowerCase();
  $("signalConfidence").textContent =
    `confidence ${(data.confidence * 100).toFixed(1)}% · arch=${data.arch} · model=${data.model} · rule=${data.rule}`;

  $("pBuy").textContent  = data.probs.Buy.toFixed(3);
  $("pHold").textContent = data.probs.Hold.toFixed(3);
  $("pSell").textContent = data.probs.Sell.toFixed(3);
  $("pBuyFill").style.width  = (data.probs.Buy  * 100) + "%";
  $("pHoldFill").style.width = (data.probs.Hold * 100) + "%";
  $("pSellFill").style.width = (data.probs.Sell * 100) + "%";

  $("price").textContent = "$" + data.price.toFixed(2);
  $("s2regime").textContent = data.stage2_regime || "—";

  if (data.stage1_trend) {
    const t = data.stage1_trend;
    const argmax = Object.entries(t).sort((a,b) => b[1] - a[1])[0];
    $("s1trend").textContent =
      `${argmax[0]} (${(argmax[1]*100).toFixed(1)}%) · D ${t.downtrend.toFixed(2)} R ${t.range.toFixed(2)} U ${t.uptrend.toFixed(2)}`;
  } else {
    $("s1trend").textContent = "—";
  }

  if (data.forward_return_5d !== null && data.forward_return_5d !== undefined) {
    const fwd = data.forward_return_5d;
    const sign = fwd >= 0 ? "+" : "";
    $("fwd").textContent = `${sign}${(fwd*100).toFixed(2)}%  (post-hoc, label-side info)`;
  } else {
    $("fwd").textContent = "—";
  }

  // Highlight selected date on timeline
  updateSelectedDate(date);
}

function updateSelectedDate(dateStr) {
  if (!currentTimeline || !priceChart) return;
  const idx = currentTimeline.dates.indexOf(dateStr);
  if (idx < 0) return;
  // Update vertical line annotation by re-rendering datasets is heavy;
  // instead use Chart.js plugin via custom annotation dataset.
  // We handle this by storing index and re-drawing the marker dataset.
  drawTimeline(currentTimeline, idx);
}

async function loadTimeline(asset) {
  const arch  = $("arch").value;
  const model = $("model").value;
  const rule  = $("rule").value;
  const url = `/timeline?asset=${asset}&arch=${arch}&model=${model}&rule=${rule}`;
  const data = await fetch(url).then(r => r.json());
  currentTimeline = data;
  drawTimeline(data, -1);
  drawStats(data);
}

function drawStats(d) {
  const winColor = d.win_rate >= 0.5 ? "pos" : "neg";
  const retColor = d.final_return >= d.bh_final_return ? "pos" : "neg";
  $("tradeStats").innerHTML = `
    <div class="stat"><label>Config</label><b>${d.label}</b></div>
    <div class="stat"><label>Trades</label><b>${d.n_trades}</b></div>
    <div class="stat ${winColor}"><label>Win Rate</label><b>${(d.win_rate*100).toFixed(1)}%</b></div>
    <div class="stat ${d.final_return >= 0 ? 'pos' : 'neg'}"><label>Strategy Return</label><b>${(d.final_return*100).toFixed(1)}%</b></div>
    <div class="stat ${d.bh_final_return >= 0 ? 'pos' : 'neg'}"><label>B&amp;H Return</label><b>${(d.bh_final_return*100).toFixed(1)}%</b></div>
    <div class="stat ${retColor}"><label>Edge vs B&amp;H</label><b>${((d.final_return - d.bh_final_return)*100).toFixed(1)} pp</b></div>
  `;
  $("equityLabel").textContent = `Stage 3 strategy: ${d.label} (vs Buy & Hold benchmark)`;
}

function drawTimeline(d, selectedIdx) {
  const buyMarkers  = [];
  const sellMarkers = [];
  for (let i = 0; i < d.dates.length; i++) {
    if (d.signals[i] === "Buy")  buyMarkers.push({ x: d.dates[i], y: d.prices[i] });
    if (d.signals[i] === "Sell") sellMarkers.push({ x: d.dates[i], y: d.prices[i] });
  }
  const selected = (selectedIdx >= 0)
    ? [{ x: d.dates[selectedIdx], y: d.prices[selectedIdx] }]
    : [];

  // === Price chart ===
  const ctxP = $("priceChart").getContext("2d");
  if (priceChart) priceChart.destroy();
  priceChart = new Chart(ctxP, {
    type: "line",
    data: {
      labels: d.dates,
      datasets: [
        {
          label: `${d.asset} Close`,
          data: d.prices,
          borderColor: d.asset === "BTC" ? "#f7931a" : "#627eea",
          borderWidth: 1.0,
          pointRadius: 0,
          tension: 0.0,
        },
        {
          label: "Buy entry",
          data: buyMarkers,
          backgroundColor: "rgba(58, 138, 58, 0.85)",
          borderColor: "#1f5a1f",
          borderWidth: 1.0,
          showLine: false,
          pointStyle: "triangle",
          pointRadius: 5,
          pointRotation: 0,
        },
        {
          label: "Sell exit",
          data: sellMarkers,
          backgroundColor: "rgba(204, 68, 68, 0.85)",
          borderColor: "#7a2222",
          borderWidth: 1.0,
          showLine: false,
          pointStyle: "triangle",
          pointRadius: 5,
          pointRotation: 180,
        },
        {
          label: "Selected date",
          data: selected,
          backgroundColor: "rgba(247, 147, 26, 1.0)",
          borderColor: "#a96012",
          borderWidth: 2,
          showLine: false,
          pointStyle: "circle",
          pointRadius: 9,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      scales: {
        x: { type: "category", ticks: { maxTicksLimit: 10, autoSkip: true } },
        y: { type: "logarithmic", title: { display: true, text: `${d.asset} price ($)` } },
      },
      plugins: {
        legend: { position: "top" },
        tooltip: { mode: "index", intersect: false },
      },
    },
  });

  // === Equity chart ===
  const ctxE = $("equityChart").getContext("2d");
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(ctxE, {
    type: "line",
    data: {
      labels: d.dates,
      datasets: [
        {
          label: `Strategy ${d.label}`,
          data: d.equity,
          borderColor: d.asset === "BTC" ? "#f7931a" : "#627eea",
          backgroundColor: d.asset === "BTC" ? "rgba(247,147,26,0.15)" : "rgba(98,126,234,0.15)",
          borderWidth: 1.6,
          pointRadius: 0,
          fill: true,
          tension: 0.0,
        },
        {
          label: "Buy & Hold",
          data: d.bh_equity,
          borderColor: "#666",
          borderDash: [4, 4],
          borderWidth: 1.4,
          pointRadius: 0,
          tension: 0.0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { ticks: { maxTicksLimit: 10, autoSkip: true } },
        y: { type: "logarithmic", title: { display: true, text: "Equity ($1 start)" } },
      },
      plugins: {
        legend: { position: "top" },
        tooltip: { mode: "index", intersect: false },
      },
    },
  });
}

init().catch(e => {
  console.error(e);
  alert("Init failed: " + e.message);
});
