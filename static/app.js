// -----------------------------
// Helpers
// -----------------------------
function formatEuro(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  const x = Number(n);
  return x.toLocaleString("it-IT", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
}

function formatPct(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  const x = Number(n);
  return (x * 100).toFixed(1) + "%";
}

// -----------------------------
// State
// -----------------------------
let chart = null;      // Chart.js main chart instance
let chartDD = null;    // Chart.js drawdown chart instance

// -----------------------------
// UI elements
// -----------------------------
const elWG = document.getElementById("w_gold");
const elWGLabel = document.getElementById("w_gold_label");
const elCapital = document.getElementById("capital");

const elFinal = document.getElementById("final_value");
const elYears = document.getElementById("final_years");
const elCagr = document.getElementById("cagr");
const elMaxDD = document.getElementById("maxdd");
const elDbl = document.getElementById("dbl");

const elWEq = document.getElementById("w_equity");
const elWBond = document.getElementById("w_bond");
const elWGold2 = document.getElementById("w_gold2");

const btnUpdate = document.getElementById("btn_update");
const btnPdf = document.getElementById("btn_pdf");

const btnFax = document.getElementById("btn_faxsimile");
const btnCons = document.getElementById("btn_consulente");
const btnLibro = document.getElementById("btn_libro");

const btnAsk = document.getElementById("btn_ask");
const askText = document.getElementById("ask_text");
const askAnswer = document.getElementById("ask_answer");

// -----------------------------
// UI logic
// -----------------------------
function clampGold(v) {
  v = Number(v);
  if (Number.isNaN(v)) v = 20;
  v = Math.max(0, Math.min(50, v));
  // step 5
  v = Math.round(v / 5) * 5;
  return v;
}

function parseCapital() {
  let s = String(elCapital.value || "").trim();
  s = s.replace(/\./g, "").replace(",", ".");
  let v = Number(s);
  if (Number.isNaN(v) || v <= 0) v = 10000;
  return v;
}

function updateSliderLabelAndComposition(wGold01) {
  const pct = Math.round(wGold01 * 100);
  elWGLabel.textContent = pct + "%";

  const w_ls80 = 1 - wGold01;
  const eq = 0.8 * w_ls80;
  const bond = 0.2 * w_ls80;

  elWEq.textContent = Math.round(eq * 100) + "%";
  elWBond.textContent = Math.round(bond * 100) + "%";
  elWGold2.textContent = Math.round(wGold01 * 100) + "%";
}

// -----------------------------
// Data load
// -----------------------------
async function loadData() {
  btnUpdate.disabled = true;
  btnUpdate.textContent = "Aggiorna…";

  try {
    const wGold = clampGold(elWG.value);
    const wGold01 = wGold / 100;
    const capital = parseCapital();

    updateSliderLabelAndComposition(wGold01);

    const url = `/api/compute?w_gold=${encodeURIComponent(wGold01)}&capital=${encodeURIComponent(capital)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Errore API");

    // Summary numbers
    const m = data.metrics || {};
    elFinal.textContent = formatEuro(m.final_portfolio);
    elYears.textContent = (m.final_years ?? "—").toFixed ? Number(m.final_years).toFixed(1) : "—";
    elCagr.textContent = formatPct(m.cagr_portfolio);
    elMaxDD.textContent = formatPct(m.max_dd_portfolio);
    elDbl.textContent = (m.doubling_years_portfolio ?? "—").toFixed ? Number(m.doubling_years_portfolio).toFixed(1) : "—";

    // Serie per chart (Portafoglio + World)
    const dates = Array.isArray(data.dates) ? data.dates : [];
    const valuesP = Array.isArray(data.portfolio) ? data.portfolio : [];
    const valuesW = Array.isArray(data.world) ? data.world : [];

    const pointsP = [];
    const pointsW = [];
    const n = Math.min(dates.length, valuesP.length);
    for (let i = 0; i < n; i++) pointsP.push({ x: dates[i], y: valuesP[i] });

    const nw = Math.min(dates.length, valuesW.length);
    for (let i = 0; i < nw; i++) pointsW.push({ x: dates[i], y: valuesW[i] });

    // Drawdown (%)
    const ddP = Array.isArray(data.drawdown_portfolio_pct) ? data.drawdown_portfolio_pct : [];
    const ddW = Array.isArray(data.drawdown_world_pct) ? data.drawdown_world_pct : [];
    const ddPointsP = [];
    const ddPointsW = [];
    const ndp = Math.min(dates.length, ddP.length);
    for (let i = 0; i < ndp; i++) ddPointsP.push({ x: dates[i], y: ddP[i] });
    const ndw = Math.min(dates.length, ddW.length);
    for (let i = 0; i < ndw; i++) ddPointsW.push({ x: dates[i], y: ddW[i] });

    // (ri)crea chart principale
    const ctx = document.getElementById("chart_main").getContext("2d");
    if (chart) chart.destroy();

    const datasetsMain = [
      {
        label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
        data: pointsP,
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 0,
      },
    ];

    if (pointsW.length > 0) {
      datasetsMain.push({
        label: "MSCI World (SMSWLD) - normalizzato",
        data: pointsW,
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 0,
      });
    }

    chart = new Chart(ctx, {
      type: "line",
      data: { datasets: datasetsMain },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        scales: {
          x: {
            type: "time",
            time: { unit: "year" },
            ticks: { maxRotation: 0, autoSkip: true },
          },
          y: {
            ticks: {
              callback: (v) => formatEuro(v),
            },
          },
        },
        plugins: {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${formatEuro(ctx.parsed.y)}`,
            },
          },
        },
      },
    });

    // (ri)crea chart drawdown
    const ddCanvas = document.getElementById("chart_dd");
    if (ddCanvas) {
      const ctxDD = ddCanvas.getContext("2d");
      if (chartDD) chartDD.destroy();

      const datasetsDD = [
        {
          label: "Drawdown Portafoglio (%)",
          data: ddPointsP,
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        },
      ];
      if (ddPointsW.length > 0) {
        datasetsDD.push({
          label: "Drawdown MSCI World (%)",
          data: ddPointsW,
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        });
      }

      chartDD = new Chart(ctxDD, {
        type: "line",
        data: { datasets: datasetsDD },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          parsing: false,
          scales: {
            x: {
              type: "time",
              time: { unit: "year" },
              ticks: { maxRotation: 0, autoSkip: true },
            },
            y: {
              ticks: { callback: (v) => `${Number(v).toFixed(0)}%` },
              suggestedMin: -60,
              suggestedMax: 0,
            },
          },
          plugins: { legend: { display: true } },
        },
      });

      // Testo riassunto drawdown
      const ddBox = document.getElementById("dd_summary");
      if (ddBox && data.metrics) {
        const m = data.metrics;

        const fmtPct1 = (x) => {
          if (x === null || x === undefined || Number.isNaN(Number(x))) return "n/d";
          return `${(Number(x) * 100).toFixed(1)}%`;
        };

        const fmtEpisode = (ep) => `${ep.start} → ${ep.bottom} → ${ep.end} : ${Number(ep.depth_pct).toFixed(1)}%`;

        const epsP = Array.isArray(m.worst_episodes_portfolio) ? m.worst_episodes_portfolio : [];
        const epsW = Array.isArray(m.worst_episodes_world) ? m.worst_episodes_world : [];

        let html = `<b>Drawdown 2025 ("Dazi Trump")</b>: Portafoglio ${fmtPct1(m.dd_2025_portfolio)} | MSCI World ${fmtPct1(m.dd_2025_world)}<br/>`;

        if (epsP.length) {
          html += `<b>3 discese peggiori (Portafoglio)</b><br/>`;
          html += epsP.map((e) => `• ${fmtEpisode(e)}`).join("<br/>") + "<br/>";
        }
        if (epsW.length) {
          html += `<b>3 discese peggiori (MSCI World)</b><br/>`;
          html += epsW.map((e) => `• ${fmtEpisode(e)}`).join("<br/>");
        }
        ddBox.innerHTML = html;
      }
    }

  } catch (err) {
    alert("Errore: " + (err?.message || err));
  } finally {
    btnUpdate.disabled = false;
    btnUpdate.textContent = "Aggiorna";
  }
}

// -----------------------------
// Buttons
// -----------------------------
btnUpdate.addEventListener("click", loadData);
elWG.addEventListener("input", () => {
  const wGold = clampGold(elWG.value);
  elWG.value = String(wGold);
  updateSliderLabelAndComposition(wGold / 100);
});

btnPdf.addEventListener("click", () => {
  // usa i valori in pagina (se disponibili)
  const cagr = elCagr.textContent || "";
  const maxdd = elMaxDD.textContent || "";
  const finalv = elFinal.textContent || "";
  const years = elYears.textContent || "";
  const url = `/api/pdf?title=${encodeURIComponent("Gloob - Metodo Pigro")}&cagr=${encodeURIComponent(cagr)}&maxdd=${encodeURIComponent(maxdd)}&final=${encodeURIComponent(finalv)}&years=${encodeURIComponent(years)}`;
  window.open(url, "_blank");
});

btnFax?.addEventListener("click", () => {
  // qui puoi puntare al tuo PDF faxsimile quando vuoi (endpoint dedicato)
  // per ora: scarica PDF generico
  btnPdf.click();
});

btnCons?.addEventListener("click", () => {
  // se già hai popup/contatti in altra versione, qui puoi richiamare quella logica
  alert("Funzione da collegare ai contatti consulenti (popup).");
});

btnLibro?.addEventListener("click", () => {
  window.open("https://www.amazon.it/dp/B0GQM925QR/ref=sr", "_blank");
});

// FAQ toggles
document.querySelectorAll(".faqItem").forEach((item) => {
  item.addEventListener("click", () => item.classList.toggle("open"));
});

// Ask (se hai endpoint /api/ask nel tuo app.py completo precedente, qui lo richiami; altrimenti resta inattivo)
btnAsk?.addEventListener("click", async () => {
  const q = (askText?.value || "").trim();
  if (!q) return;

  try {
    btnAsk.disabled = true;
    btnAsk.textContent = "Invio…";

    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });

    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Errore ask");

    askAnswer.style.display = "block";
    askAnswer.textContent = data.answer || "";
  } catch (e) {
    alert("Errore: " + (e?.message || e));
  } finally {
    btnAsk.disabled = false;
    btnAsk.textContent = "Chiedi all’assistente";
  }
});

// Init
updateSliderLabelAndComposition(clampGold(elWG.value) / 100);
loadData();
