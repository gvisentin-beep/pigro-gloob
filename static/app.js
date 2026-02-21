let chart = null;
let _lastCompute = null; // { data, comp, capital }

function euro(x) {
  return new Intl.NumberFormat("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(Number(x));
}

function pct(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return (n * 100).toFixed(1) + "%";
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function parseCapitalEuro() {
  const el = document.getElementById("initial");
  if (!el) return 10000;
  const raw = String(el.value || "").replace(/\./g, "").replace(/,/g, ".");
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 10000;
}

function formatCapitalInput() {
  const el = document.getElementById("initial");
  if (!el) return;
  const n = parseCapitalEuro();
  el.value = new Intl.NumberFormat("it-IT", { maximumFractionDigits: 0 }).format(n);
}

// Oro slider 0–50 step 5; LS80 = 100 - oro; dentro LS80: 80/20
function computeCompositionFromGoldInput() {
  const goldEl = document.getElementById("w_gold");
  const gold = goldEl ? Number(goldEl.value) : 0;
  const g = clamp(Math.round(gold / 5) * 5, 0, 50);

  const ls80 = 100 - g;
  const equity = Math.round(ls80 * 0.8);
  const bonds = 100 - g - equity; // per arrivare a 100

  return { gold: g, ls80, equity, bonds };
}

function updateGoldBadgeAndHideInlineComposition(comp) {
  const badge = document.getElementById("gold_value");
  if (badge) badge.textContent = `${comp.gold}%`;
}

function yearsToDoubleFromCagr(cagr) {
  const r = Number(cagr);
  if (!Number.isFinite(r) || r <= 0) return "—";
  const y = Math.log(2) / Math.log(1 + r);
  if (!Number.isFinite(y)) return "—";
  return y.toFixed(1).replace(".", ",");
}

// labels: mostra solo 1 per anno (e griglia “leggera”, poi la griglia la gestiamo via ticks)
function buildYearTicks(datesISO) {
  const years = new Set();
  const tickIdx = [];
  for (let i = 0; i < datesISO.length; i++) {
    const y = String(datesISO[i]).slice(0, 4);
    if (!years.has(y)) {
      years.add(y);
      tickIdx.push(i);
    }
  }
  const allow = new Set(tickIdx);
  return (value, index) => {
    if (allow.has(index)) return String(datesISO[index]).slice(0, 4);
    return "";
  };
}

async function loadData() {
  const comp = computeCompositionFromGoldInput();
  const capital = parseCapitalEuro();

  updateGoldBadgeAndHideInlineComposition(comp);

  // query verso backend: w_ls80 + w_gold in percento
  const url =
    `/api/compute?w_ls80=${encodeURIComponent(comp.ls80)}` +
    `&w_gold=${encodeURIComponent(comp.gold)}` +
    `&capital=${encodeURIComponent(capital)}` +
    `&t=${Date.now()}`;

  const res = await fetch(url, { cache: "no-store" });
  const data = await res.json();

  if (!res.ok || data.error) {
    console.error(data.error || "Errore compute");
    return;
  }

  _lastCompute = { data, comp, capital };

  // Aggiorna box metriche
  const metrics = data.metrics || {};
  const cagr = metrics.cagr_portfolio;
  const mdd = metrics.max_dd_portfolio;
  const yd = metrics.years_to_double;
  const finalCap = metrics.final_portfolio;

  const line1 =
    `Portafoglio (ETF Azion-Obblig + ETC Oro): ` +
    `Rendimento annualizzato ${pct(cagr)} | ` +
    `Max Ribasso nel periodo ${pct(mdd)}`;

  const line2 =
    `Composizione: Azionario ${comp.equity}% | Obbligazionario ${comp.bonds}% | Oro ${comp.gold}%`;

  const line3 = `Raddoppio del portafoglio in anni: ${yearsToDoubleFromCagr(cagr)}`;

  const box = document.getElementById("metrics_box");
  if (box) {
    box.innerHTML = `
      <div><b>${line1}</b></div>
      <div><b>${line2}</b></div>
      <div>${line3}</div>
    `;
  }

  // Riga “Capitale finale” vicino al bottone (se presente)
  const finalEl = document.getElementById("final_capital");
  const yearsEl = document.getElementById("final_years");
  if (finalEl) finalEl.textContent = euro(finalCap);
  if (yearsEl) {
    const yTot = Number(metrics.years_total);
    yearsEl.textContent = Number.isFinite(yTot) ? yTot.toFixed(1).replace(".", ",") : "—";
  }

  // Chart
  const ctx = document.getElementById("chart").getContext("2d");
  if (chart) chart.destroy();

  const dates = data.dates || [];
  const yearTickCb = buildYearTicks(dates);

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        {
          label: "Portafoglio (ETF+Oro)",
          data: data.portfolio,
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
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${euro(context.parsed.y)}`,
          },
        },
        legend: { display: true },
      },
      scales: {
        x: {
          ticks: {
            callback: yearTickCb,
            maxRotation: 0,
            minRotation: 0,
            autoSkip: false,
          },
          grid: {
            // griglia meno “fitta”
            drawTicks: true,
          },
        },
        y: {
          ticks: { callback: (value) => euro(value) },
        },
      },
    },
  });
}

/* =========================
   ASSISTENTE
   ========================= */
function setAskStatus(text, show = true) {
  const status = document.getElementById("ask_status");
  if (!status) return;
  status.style.display = show ? "inline" : "none";
  status.textContent = text || "";
}

function setAskAnswer(text, isError = false) {
  const out = document.getElementById("ask_answer");
  if (!out) return;
  out.style.display = "block";
  out.style.whiteSpace = "pre-wrap";
  out.style.borderColor = isError ? "#e0a3a3" : "";
  out.style.background = isError ? "#fff6f6" : "";
  out.textContent = text || "";
}

async function askAssistant() {
  const ta = document.getElementById("ask_text");
  const btn = document.getElementById("ask_btn");
  if (!ta || !btn) return;

  const q = String(ta.value || "").trim();
  if (!q) {
    setAskAnswer("Scrivi una domanda prima di inviare.", true);
    return;
  }

  btn.disabled = true;
  setAskStatus("Sto pensando…", true);
  setAskAnswer("");

  const comp = computeCompositionFromGoldInput();
  const capital = parseCapitalEuro();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        question: q,
        context: {
          oro_pct: comp.gold,
          azionario_pct: comp.equity,
          obbligazionario_pct: comp.bonds,
          capitale_eur: new Intl.NumberFormat("it-IT", { maximumFractionDigits: 0 }).format(capital),
        },
      }),
    });

    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      // se il server manda HTML (es. errore), evitiamo crash
      throw new Error("Risposta server non valida (atteso JSON).");
    }

    if (!res.ok || (data && data.error)) {
      const msg = (data && data.error) ? data.error : "Errore assistente.";
      setAskAnswer(msg, true);

      if (data && data.remaining !== undefined && data.limit !== undefined) {
        setAskStatus(`Domande rimanenti oggi: ${data.remaining}/${data.limit}`, true);
      } else {
        setAskStatus("", false);
      }
      return;
    }

    setAskAnswer(data.answer || "(risposta vuota)", false);

    if (data.remaining !== undefined && data.limit !== undefined) {
      setAskStatus(`Domande rimanenti oggi: ${data.remaining}/${data.limit}`, true);
    } else {
      setAskStatus("", false);
    }
  } catch (e) {
    console.error(e);
    setAskAnswer(String(e.message || e), true);
    setAskStatus("", false);
  } finally {
    btn.disabled = false;
  }
}

function setupAssistant() {
  const btn = document.getElementById("ask_btn");
  const ta = document.getElementById("ask_text");
  if (!btn || !ta) return;

  btn.addEventListener("click", askAssistant);
  ta.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) askAssistant();
  });

  // stato iniziale
  setAskStatus(`Domande rimanenti oggi: ${10}/${10}`, true);
}

function init() {
  // format input capitale
  const cap = document.getElementById("initial");
  if (cap) {
    cap.addEventListener("blur", formatCapitalInput);
    formatCapitalInput();
  }

  // aggiorna (primo button della pagina)
  const updateBtn = document.querySelector("button");
  if (updateBtn) {
    updateBtn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  const goldEl = document.getElementById("w_gold");
  if (goldEl) {
    goldEl.addEventListener("input", () => {
      const comp = computeCompositionFromGoldInput();
      updateGoldBadgeAndHideInlineComposition(comp);
    });
    goldEl.addEventListener("change", () => loadData());
  }

  loadData();
  setupAssistant();
}

window.addEventListener("DOMContentLoaded", init);
