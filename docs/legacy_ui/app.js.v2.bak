/**
 * Crypto Signal Classifier — Frontend (MVP)
 */

const API_BASE = "";

async function loadTestDates() {
    const select = document.getElementById("date-select");
    try {
        const response = await fetch(`${API_BASE}/test_dates/BTC`);
        if (!response.ok) throw new Error("Failed to load dates");
        const data = await response.json();
        select.innerHTML = "";
        if (!data.dates.length) {
            select.innerHTML = '<option value="">no test dates available</option>';
            return;
        }
        // Show most recent dates first (better UX for the demo)
        const sorted = [...data.dates].sort((a, b) => b.localeCompare(a));
        for (const d of sorted) {
            const opt = document.createElement("option");
            opt.value = d;
            opt.textContent = d;
            select.appendChild(opt);
        }
    } catch (err) {
        select.innerHTML = `<option value="">error: ${err.message}</option>`;
    }
}

async function getPredictionByDate() {
    const symbol = document.getElementById("symbol-select").value;
    const date = document.getElementById("date-select").value;
    const model = document.getElementById("model-select").value;
    if (!date) {
        showError("Please select a date.");
        return;
    }
    await runPrediction({ symbol, mode: "date", date, model });
}

async function getLivePrediction() {
    const symbol = document.getElementById("symbol-select").value;
    const model = document.getElementById("model-select").value;
    await runPrediction({ symbol, mode: "live", model });
}

async function runPrediction(payload) {
    showLoading(true);
    hideError();
    hideResult();

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(err.detail || "Prediction failed");
        }
        const data = await response.json();
        displayResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        showLoading(false);
    }
}

function displayResult(data) {
    const section = document.getElementById("result-section");
    section.classList.remove("hidden");

    const meta = document.getElementById("result-meta");
    const priceStr = data.price != null ? `· $${data.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : "";
    meta.textContent = `· ${data.symbol || "BTC"} · ${data.date} · ${data.mode} · ${data.model.toUpperCase()} ${priceStr}`;

    const badge = document.getElementById("signal-badge");
    document.getElementById("signal-text").textContent = data.signal;
    badge.className = "signal-badge";
    if (data.signal === "Buy") badge.classList.add("signal-buy");
    else if (data.signal === "Sell") badge.classList.add("signal-sell");
    else badge.classList.add("signal-hold");

    const pct = (data.confidence * 100).toFixed(1);
    document.getElementById("confidence-bar").style.width = `${pct}%`;
    document.getElementById("confidence-value").textContent = `${pct}%`;

    if (data.trend) {
        renderProbBars("trend-probs", data.trend, {
            "Uptrend": "#2ecc71",
            "Downtrend": "#e74c3c",
            "Sideways": "#f1c40f",
        });
    }
    if (data.macro_regime) {
        renderProbBars("regime-probs", data.macro_regime, {
            "Risk-On": "#2ecc71",
            "Risk-Off": "#e74c3c",
            "Neutral": "#f1c40f",
        });
    }
    if (data.signal_probs) {
        renderProbBars("signal-probs", data.signal_probs, {
            "Buy": "#2ecc71",
            "Sell": "#e74c3c",
            "Hold": "#f1c40f",
        });
    }
}

function renderProbBars(containerId, probs, colors) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    for (const [label, prob] of Object.entries(probs)) {
        const pct = (prob * 100).toFixed(1);
        const color = colors[label] || "#5b6eae";
        const item = document.createElement("div");
        item.className = "prob-item";
        item.innerHTML = `
            <span class="label">${label}</span>
            <div class="bar"><div class="bar-fill" style="width: ${pct}%; background: ${color};"></div></div>
            <span class="value">${pct}%</span>
        `;
        container.appendChild(item);
    }
}

function showLoading(show) { document.getElementById("loading").classList.toggle("hidden", !show); }
function showError(msg) {
    document.getElementById("error-message").textContent = msg;
    document.getElementById("error-section").classList.remove("hidden");
}
function hideError() { document.getElementById("error-section").classList.add("hidden"); }
function hideResult() { document.getElementById("result-section").classList.add("hidden"); }

document.addEventListener("DOMContentLoaded", loadTestDates);
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) getPredictionByDate();
});
