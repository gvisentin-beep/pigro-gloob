// static/app.js
let chart = null;

/* -----------------------------
   Format helpers
----------------------------- */
function euro(x) {
  return new Intl.NumberFormat("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(Number(x));
}

function num0(x) {
  return new Intl.NumberFormat("it-IT", { maximumFractionDigits: 0 }).format(Number(x));
}

function pct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (Number(x) * 100).toFixed(1).replace(".", ",") + "%";
}

function years1(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(1).replace(".", ",");
}

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

function parseCapitalInput(str) {
  const s = String(str || "").trim().replace(/\./g, "").replace(",", ".");
  const v = Number(s);
  return Number.isFinite(v) && v > 0 ? v : 10000;
}

/* -----------------------------
   UI helpers
----------------------------- */
function setAskStatus(msg, kind) {
  const el = document.getElementById("ask_status");
  if (!el) return;
  el.textContent = msg || "";
  el.className = "muted" + (kind === "err" ? " err" : "");
}

function updateSliderLabelAndComposition(goldPct) {
  const gold = clamp(goldPct, 0, 50) / 100;
  const ls80 = 1 - gold;

  const goldValEl = document.getElementById("gold_val");
  if (goldValEl) goldValEl.textContent = `${Math.round(gold * 100)}%`;

  const line2 = document.getElementById("metrics_line2");
  if (line2) {
    const az = ls80 * 0.80;
    const ob = ls80 * 0.20;
    line2.textContent =
      `Composizione: Azionario ${Math.round(az * 100)}% | ` +
      `Obbligazionario ${Math.round(ob * 100)}% | ` +
      `Oro ${Math.round(gold * 100)}%`;
  }

  return { w_ls80: ls80, w_gold: gold };
}

function getCapitalValue() {
  const el = document.getElementById("capital_input");
  return parseCapitalInput(el ? el.value : "10000");
}

/* -----------------------------
   Chart
----------------------------- */
function ensureChart() {
  const canvas = document.getElementById("chart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  if (chart) {
    chart.destroy();
    chart = null;
  }

  chart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
          data: [],
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        // ✅ NIENTE DATE / TOOLTIP
        tooltip: { enabled: false },
        legend: { display: true },
      },
      scales: {
        x: {
          // ✅ NASCONDI COMPLETAMENTE ASSE X (DATE)
          display: false,
          type: "time",
          time: {
            unit: "year",
            tooltipFormat: "dd/MM/yyyy",
            displayFormats: { year: "yyyy" },
          },
          ticks: {
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 12,
          },
        },
        y: {
          ticks: {
            callback: (v) => `${num0(v)} €`,
          },
        },
      },
    },
  });
}

/* -----------------------------
   Compute + render
----------------------------- */
async function refreshAll() {
  const slider = document.getElementById("gold_slider");
  const goldPct = Number(slider ? slider.value : 0);

  const capital = getCapitalValue();
  const comp = updateSliderLabelAndComposition(goldPct);

  const url =
    `/api/compute?w_ls80=${encodeURIComponent(comp.w_ls80)}` +
    `&w_gold=${encodeURIComponent(comp.w_gold)}` +
    `&capital=${encodeURIComponent(capital)}` +
    `&t=${Date.now()}`;

  let res;
  let data;

  try {
    res = await fetch(url, { cache: "no-store" });
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    data = ct.includes("application/json") ? await res.json() : { error: await res.text() };
  } catch (e) {
    console.error(e);
    setAskStatus(`Errore rete: ${String(e).slice(0, 240)}`, "err");
    return;
  }

  if (!res.ok || data.error) {
    console.error(data.error || res.statusText);
    setAskStatus(`Errore: ${(data.error || res.statusText).toString().slice(0, 240)}`, "err");
    return;
  }

  const m = data.metrics || {};

  const line1 = document.getElementById("metrics_line1");
  const line3 = document.getElementById("metrics_line3");

  if (line1) {
    line1.textContent =
      `Portafoglio (ETF Azion-Obblig + ETC Oro): ` +
      `Rendimento annualizzato ${pct(m.cagr_portfolio)} | ` +
      `Max Ribasso nel periodo ${pct(m.max_dd_portfolio)}`;
  }
  if (line3) {
    line3.textContent = `Raddoppio del portafoglio in anni: ${years1(m.doubling_years_portfolio)}`;
  }

  const finalValue = document.getElementById("final_value");
  const finalYears = document.getElementById("final_years");
  if (finalValue) finalValue.textContent = euro(m.final_portfolio);
  if (finalYears) finalYears.textContent = years1(m.final_years);

  const dates = Array.isArray(data.dates) ? data.dates : [];
  const values = Array.isArray(data.portfolio) ? data.portfolio : [];

  const points = [];
  for (let i = 0; i < Math.min(dates.length, values.length); i++) {
    points.push({ x: dates[i], y: values[i] });
  }

  ensureChart();
  if (chart) {
    chart.data.datasets[0].data = points;
    chart.update();
  }
}

/* -----------------------------
   PDF + Faxsimile
----------------------------- */
function setupButtons() {
  const btnPdf = document.getElementById("btn_pdf");
  if (btnPdf) btnPdf.addEventListener("click", () => window.print());

  const btnFax = document.getElementById("btn_faxsimile");
  if (btnFax) {
    btnFax.addEventListener("click", () => {
      window.open("/faxsimile_execution_only.pdf", "_blank", "noopener,noreferrer");
    });
  }
}

/* -----------------------------
   Assistant
----------------------------- */
async function askAssistant() {
  const input = document.getElementById("ask_input");
  const btn = document.getElementById("ask_btn");
  const out = document.getElementById("ask_answer");
  const remaining = document.getElementById("ask_remaining");

  const question = (input ? input.value : "").trim();
  if (!question) return;

  try {
    if (btn) btn.disabled = true;
    setAskStatus("Sto pensando…");

    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `Errore (${res.status})`);
    }

    if (out) out.textContent = data.answer || "";
    setAskStatus("");

    if (remaining && typeof data.remaining === "number" && typeof data.limit === "number") {
      remaining.textContent = `Domande rimanenti oggi: ${data.remaining}/${data.limit}`;
    }
  } catch (e) {
    console.error(e);
    setAskStatus(String(e).slice(0, 240), "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

/* -----------------------------
   Init
----------------------------- */
function init() {
  const slider = document.getElementById("gold_slider");
  const cap = document.getElementById("capital_input");
  const btnUpdate = document.getElementById("btn_update");
  const askBtn = document.getElementById("ask_btn");
  const askInput = document.getElementById("ask_input");

  if (slider) {
    updateSliderLabelAndComposition(Number(slider.value));
    slider.addEventListener("input", () => updateSliderLabelAndComposition(Number(slider.value)));
    slider.addEventListener("change", refreshAll);
  }

  if (cap) {
    cap.addEventListener("change", refreshAll);
    cap.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") refreshAll();
    });
  }

  if (btnUpdate) btnUpdate.addEventListener("click", refreshAll);

  setupButtons();

  if (askBtn) askBtn.addEventListener("click", askAssistant);
  if (askInput) {
    askInput.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) askAssistant();
    });
  }

  refreshAll();
}

document.addEventListener("DOMContentLoaded", init);
