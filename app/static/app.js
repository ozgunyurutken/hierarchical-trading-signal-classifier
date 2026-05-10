// V5 Hierarchical Trading Signal Classifier — minimal frontend.
// Reads OOF predictions via FastAPI backend (no on-the-fly inference).

const $ = (id) => document.getElementById(id);
let equityChart = null;

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
  await onAssetChange();
}

async function onAssetChange() {
  const asset = $("asset").value;
  const dates = await fetch(`/test_dates/${asset}`).then(r => r.json());
  const dateSel = $("date");
  dateSel.innerHTML = "";
  // Show recent dates first; user usually wants the latest
  for (const d of dates.dates.slice().reverse()) {
    const opt = document.createElement("option");
    opt.value = d; opt.textContent = d;
    dateSel.appendChild(opt);
  }
  await loadEquity(asset);
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
}

async function loadEquity(asset) {
  const data = await fetch(`/equity/${asset}`).then(r => r.json());
  if (data.dates.length === 0) {
    $("equityLabel").textContent = "(no equity data)";
    return;
  }
  $("equityLabel").textContent = `Best Stage 3: ${data.best_label} (vs Buy & Hold)`;

  const ctx = $("equityChart").getContext("2d");
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: `Stage 3: ${data.best_label}`,
          data: data.best,
          borderColor: asset === "BTC" ? "#f7931a" : "#627eea",
          borderWidth: 1.6,
          pointRadius: 0,
          tension: 0.0,
        },
        {
          label: "Buy & Hold",
          data: data.buy_hold,
          borderColor: "#666666",
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
        x: { display: true, ticks: { maxTicksLimit: 8 } },
        y: { type: "logarithmic", title: { display: true, text: "Equity (log, $1 start)" } },
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
