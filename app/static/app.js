/**
 * Crypto Signal Classifier - Frontend Application
 */

const API_BASE = "";

// === API Calls ===

async function getPrediction() {
    const symbol = document.getElementById("symbol-select").value;
    showLoading(true);
    hideError();
    hideResult();

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol: symbol }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Prediction failed");
        }

        const data = await response.json();
        displayResult(data);
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

async function predictFromCSV() {
    const fileInput = document.getElementById("csv-file");
    if (!fileInput.files.length) {
        showError("Please select a CSV file first.");
        return;
    }

    showLoading(true);
    hideError();
    hideResult();

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch(`${API_BASE}/predict/csv`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "CSV prediction failed");
        }

        const data = await response.json();
        displayResult(data);
    } catch (error) {
        showError(error.message);
    } finally {
        showLoading(false);
    }
}

// === Display Functions ===

function displayResult(data) {
    const section = document.getElementById("result-section");
    section.classList.remove("hidden");

    // Signal badge
    const badge = document.getElementById("signal-badge");
    const signalText = document.getElementById("signal-text");
    signalText.textContent = data.signal;

    badge.className = "signal-badge";
    if (data.signal === "Buy") badge.classList.add("signal-buy");
    else if (data.signal === "Sell") badge.classList.add("signal-sell");
    else badge.classList.add("signal-hold");

    // Confidence bar
    const confidencePct = (data.confidence * 100).toFixed(1);
    document.getElementById("confidence-bar").style.width = `${confidencePct}%`;
    document.getElementById("confidence-value").textContent = `${confidencePct}%`;

    // Stage probabilities
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
            <div class="bar">
                <div class="bar-fill" style="width: ${pct}%; background: ${color};"></div>
            </div>
            <span class="value">${pct}%</span>
        `;
        container.appendChild(item);
    }
}

// === UI Helpers ===

function showLoading(show) {
    document.getElementById("loading").classList.toggle("hidden", !show);
}

function showError(message) {
    const section = document.getElementById("error-section");
    document.getElementById("error-message").textContent = message;
    section.classList.remove("hidden");
}

function hideError() {
    document.getElementById("error-section").classList.add("hidden");
}

function hideResult() {
    document.getElementById("result-section").classList.add("hidden");
}

// === Keyboard shortcut ===
document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        getPrediction();
    }
});
