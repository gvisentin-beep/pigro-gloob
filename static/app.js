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
  if (x === null || x === undefined || isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

function years1(x) {
  if (x === null || x === undefined || isNaN(x)) return "—";
  return Number(x).toFixed(1).replace(".", ",");
}

/* -----------------------------
   Composition: slider is GOLD %
   LS80 split: 80% equity, 20% bond
----------------------------- */
function computeComposition(goldPct) {
  const w_gold = Number(goldPct);
  const w_ls80 = 100 - w_gold;
  const equity = 0.8 * w_ls80;
  const bond = 0.2 * w_ls80;

  return {
    w_gold,
    w_ls80,
    equity: Math.round(equity),
    bond: Math.round(bond),
  };
}

/* -----------------------------
   UI helpers
----------------------------- */
function setAskStatus(msg, kind = "info") {
  const box = document.getElementById("ask_status");
  if (!box) return;

  box.classList.remove("ok", "err", "info");
  box.classList.add(kind);
  box.textContent = msg || "";
  box.style.display = msg ? "block" : "none";
}

function setRemaining(remaining, limit) {
  const el = document.getElementById("ask_remaining");
  if (!el) return;
  if (remaining === null || remaining === undefined || limit === null || limit === undefined) {
    el.textContent = "";
    return;
  }
  el.textContent = `Domande rimanenti oggi: ${remaining}/${limit}`;
}

/**
 * Normalizza date per Chart.js time scale.
 * Supporta:
 * - "YYYY-MM-DD"
 * - "DD/MM/YYYY"
 * - casi sporchi tipo "21/01/2026,39.37"
 */
function normalizeDateToISO(s) {
  if (!s) return null;
  let x = String(s).trim();

  x = x.split(",")[0].trim();
  x = x.split(";")[0].trim();
  x = x.split(" ")[0].trim();

  if (/^\d{4}-\d{2}-\d{2}$/.test(x)) return x;

  const m = x.match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (m) return `${m[3]}-${m[2]}-${m[1]}`;

  const first10 = x.slice(0, 10);
  if (/^\d{4}-\d{2}-\d{2}$/.test(first10)) return first10;

  const m2 = first10.match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (m2) return `${m2[3]}-${m2[2]}-${m2[1]}`;

  return null;
}

function getCapitalValue() {
  const el = document.getElementById("initial");
  if (!el) return 10000;

  const raw = (el.value || "")
    .toString()
    .replace(/\./g, "")
    .replace(/[^\d]/g, "");
  const n = Number(raw) || 10000;
  return n > 0 ? n : 10000;
}

function updateSliderLabelAndComposition(goldPct) {
  const comp = computeComposition(goldPct);

  const goldVal = document.getElementById("w_gold_val");
  if (goldVal) goldVal.textContent = `${comp.w_gold}%`;

  const line2 = document.getElementById("metrics_line2");
  if (line2) {
    line2.textContent =
      `Composizione: Azionario ${comp.equity}% | Obbligazionario ${comp.bond}% | Oro ${comp.w_gold}%`;
  }

  return comp;
}

/* -----------------------------
   Data load + chart
----------------------------- */
async function loadData() {
  const slider = document.getElementById("w_gold");
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

  // Linea 1 e 3 (linea 2 è già aggiornata in updateSliderLabelAndComposition)
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

  // Finali (sulla riga in alto)
  const finalValue = document.getElementById("final_value");
  const finalYears = document.getElementById("final_years");
  if (finalValue) finalValue.textContent = euro(m.final_portfolio);
  if (finalYears) finalYears.textContent = years1(m.final_years);

  // Serie per chart
  const dates = Array.isArray(data.dates) ? data.dates : [];
  const values = Array.isArray(data.portfolio) ? data.portfolio : [];

  const points = [];
  const n = Math.min(dates.length, values.length);
  for (let i = 0; i < n; i++) {
    const iso = normalizeDateToISO(dates[i]);
    if (!iso) continue;
    points.push({ x: iso, y: Number(values[i]) });
  }

  if (points.length < 2) {
    setAskStatus(
      "Errore dati: non riesco a leggere le date. Controlla i CSV (Date: YYYY-MM-DD o DD/MM/YYYY).",
      "err"
    );
    return;
  }

  // Chart render
  try {
    const canvas = document.getElementById("chart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (chart) chart.destroy();

    chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
            data: points,
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
            type: "time",
            time: {
              unit: "month",
              stepSize: 6,
              tooltipFormat: "dd/MM/yyyy",
              displayFormats: { month: "MM/yyyy" },
            },
            ticks: { maxRotation: 0, autoSkip: true },
          },
          y: {
            ticks: { callback: (value) => euro(value) },
          },
        },
      },
    });

    // se tutto ok, togli eventuali errori a schermo
    setAskStatus("", "info");
  } catch (e) {
    console.error(e);
    setAskStatus(`Errore grafico: ${String(e).slice(0, 240)}`, "err");
  }
}

/* -----------------------------
   Capital field formatting
----------------------------- */
function formatCapitalField() {
  const el = document.getElementById("initial");
  if (!el) return;

  el.addEventListener("blur", () => {
    const raw = (el.value || "").toString().replace(/\./g, "").replace(/[^\d]/g, "");
    const n = Number(raw || "0");
    el.value = n ? num0(n) : "10.000";
  });

  // INVIO nel campo capitale -> aggiorna
  el.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      loadData();
    }
  });
}

/* -----------------------------
   Controls wiring
----------------------------- */
function wireControls() {
  const btn = document.getElementById("btn_update");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  const slider = document.getElementById("w_gold");
  if (slider) {
    let t = null;

    // mentre trascini: aggiorna label + composizione; dopo 250ms ricalcola
    slider.addEventListener("input", () => {
      const goldPct = Number(slider.value || 0);
      updateSliderLabelAndComposition(goldPct);

      if (t) clearTimeout(t);
      t = setTimeout(() => loadData(), 250);
    });

    // quando rilasci: ricalcola subito (così non “rimane indietro”)
    slider.addEventListener("change", () => {
      loadData();
    });
  }
}

/* -----------------------------
   Assistant
----------------------------- */
async function askAssistant() {
  const q = (document.getElementById("ask_question").value || "").trim();
  const btn = document.getElementById("ask_btn");
  const ans = document.getElementById("ask_answer");

  if (!q) {
    setAskStatus("Scrivi una domanda.", "err");
    return;
  }

  if (btn) btn.disabled = true;
  setAskStatus("Sto pensando…", "info");

  try {
    const res = await fetch(`/api/ask?t=${Date.now()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({ question: q }),
    });

    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const data = ct.includes("application/json")
      ? await res.json()
      : { ok: false, error: await res.text() };

    if (!res.ok || !data.ok) {
      const msg = (data.error || `Errore server (${res.status})`).toString();
      setAskStatus(msg.slice(0, 240), "err");
      if (ans) ans.textContent = "";
      if ("remaining" in data && "limit" in data) setRemaining(data.remaining, data.limit);
      return;
    }

    if (ans) ans.textContent = data.answer || "";
    setAskStatus("", "ok");
    setRemaining(data.remaining, data.limit);
  } catch (e) {
    setAskStatus(`Errore rete: ${String(e).slice(0, 240)}`, "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

function wireAssistant() {
  const btn = document.getElementById("ask_btn");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      askAssistant();
    });
  }

  const ta = document.getElementById("ask_question");
  if (ta) {
    ta.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        askAssistant();
      }
    });
  }
}

/* -----------------------------
   Init
----------------------------- */
function init() {
  formatCapitalField();
  wireControls();
  wireAssistant();

  // primo render
  loadData();
}

window.addEventListener("DOMContentLoaded", init);
