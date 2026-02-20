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

function snapToStep(v, step) {
  return Math.round(v / step) * step;
}

function fmtYearsIt(n) {
  const x = Number(n);
  if (!Number.isFinite(x) || x <= 0) return "—";
  return x.toFixed(1).replace(".", ",");
}

// Legge capitale anche se formattato "10.000"
function parseCapitalEuro() {
  const raw = String(document.getElementById("initial")?.value || "");
  const digits = raw.replace(/\D/g, "");
  const n = Number(digits);
  return Number.isFinite(n) && n > 0 ? n : 10000;
}

// Composizione: Oro 0-50%, resto ETF 80/20
function computeCompositionFromGoldInput() {
  let gold = Number(document.getElementById("w_gold")?.value);
  if (!Number.isFinite(gold)) gold = 10;

  gold = clamp(gold, 0, 50);
  gold = snapToStep(gold, 5);

  const goldEl = document.getElementById("w_gold");
  if (goldEl) goldEl.value = gold;

  const ls80 = 100 - gold;
  const equity = ls80 * 0.80;
  const bonds = ls80 * 0.20;

  return { gold, ls80, equity, bonds };
}

function updateGoldBadgeAndHideInlineComposition(comp) {
  const badge = document.getElementById("gold_badge");
  if (badge) badge.innerText = `${comp.gold.toFixed(0)}%`;

  const ge = document.getElementById("w_equity");
  if (ge) {
    const container = ge.closest("div");
    if (container) container.style.display = "none";
  }
}

function findMetricsCard() {
  return Array.from(document.querySelectorAll(".card")).find((el) =>
    el.innerText.includes("Portafoglio")
  );
}

// Marker anni: inizio anno + metà anno (2 linee verticali)
function computeYearMarkers(dateStrings) {
  const byYear = {};
  for (let i = 0; i < dateStrings.length; i++) {
    const y = String(dateStrings[i]).slice(0, 4);
    if (!byYear[y]) byYear[y] = [];
    byYear[y].push(i);
  }

  const yearStart = new Set();
  const yearMarkers = new Set();

  Object.values(byYear).forEach((arr) => {
    if (!arr.length) return;
    yearStart.add(arr[0]);
    yearMarkers.add(arr[0]);
    const mid = arr[Math.floor(arr.length / 2)];
    yearMarkers.add(mid);
  });

  return { yearStart, yearMarkers };
}

function computeYearsBetween(dateStartStr, dateEndStr) {
  const t0 = Date.parse(dateStartStr);
  const t1 = Date.parse(dateEndStr);
  if (!Number.isFinite(t0) || !Number.isFinite(t1) || t1 <= t0) return null;
  return (t1 - t0) / (365.25 * 24 * 3600 * 1000);
}

async function loadData() {
  const comp = computeCompositionFromGoldInput();
  updateGoldBadgeAndHideInlineComposition(comp);

  const w_ls80 = comp.ls80;
  const w_gold = comp.gold;
  const capital = parseCapitalEuro();

  const url =
    `/api/compute` +
    `?w_ls80=${encodeURIComponent(w_ls80)}` +
    `&w_gold=${encodeURIComponent(w_gold)}` +
    `&capital=${encodeURIComponent(capital)}` +
    `&t=${Date.now()}`;

  let data;
  try {
    const res = await fetch(url, { cache: "no-store" });
    data = await res.json();
  } catch (err) {
    console.error(err);
    alert("Errore di rete nel caricamento dati.");
    return;
  }

  if (data?.error) {
    console.error(data.error);
    alert(data.error);
    return;
  }

  // salva ultimo calcolo per PDF
  _lastCompute = { data, comp, capital };

  // Capitale finale + anni (riga accanto al bottone)
  if (Array.isArray(data.dates) && data.portfolio?.length > 1) {
    const finalValue = data.portfolio[data.portfolio.length - 1];
    const years = computeYearsBetween(
      data.dates[0],
      data.dates[data.dates.length - 1]
    );

    const outFinal = document.getElementById("final_capital");
    const outYears = document.getElementById("final_years");

    if (outFinal) outFinal.innerText = euro(finalValue);
    if (outYears) outYears.innerText = fmtYearsIt(years);
  }

  // ===== METRICHE =====
  if (data.metrics) {
    const m = data.metrics;

    const riga1 =
      `Portafoglio (ETF Azion-Obblig + ETC Oro): ` +
      `Rendimento annualizzato ${pct(m.cagr_portfolio)} | ` +
      `Max Ribasso nel periodo ${pct(m.max_dd_portfolio)}`;

    const riga2 =
      `Composizione: Azionario ${comp.equity.toFixed(0)}% | ` +
      `Obbligazionario ${comp.bonds.toFixed(0)}% | ` +
      `Oro ${comp.gold.toFixed(0)}%`;

    let ytd = m.years_to_double;
    if (!(Number.isFinite(Number(ytd)) && Number(ytd) > 0)) {
      const cagr = Number(m.cagr_portfolio);
      if (Number.isFinite(cagr) && cagr > 0) {
        ytd = Math.log(2) / Math.log(1 + cagr);
      } else {
        ytd = null;
      }
    }
    const riga3 = `Raddoppio del portafoglio in anni: ${fmtYearsIt(ytd)}`;

    const card = findMetricsCard();
    if (card) card.innerHTML = `<b>${riga1}</b><br><b>${riga2}</b><br>${riga3}`;
  }

  // ===== GRAFICO =====
  const ctx = document.getElementById("chart").getContext("2d");
  if (chart) chart.destroy();

  const labels = data.dates;
  const { yearStart, yearMarkers } = computeYearMarkers(labels);

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
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
            label: (c) => `Portafoglio: ${euro(c.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            callback: function (_value, index) {
              if (!yearStart.has(index)) return "";
              return String(labels[index]).slice(0, 4);
            },
            autoSkip: false,
            maxRotation: 0,
            minRotation: 0,
          },
          grid: {
            color: (ctx) =>
              yearMarkers.has(ctx.index)
                ? "rgba(0,0,0,0.18)"
                : "rgba(0,0,0,0.03)",
            lineWidth: (ctx) => (yearMarkers.has(ctx.index) ? 1.2 : 0.2),
          },
        },
        y: {
          ticks: { callback: (v) => euro(v) },
        },
      },
    },
  });
}

/* ===== PDF ===== */
async function creaPdf() {
  if (!chart || !_lastCompute || !_lastCompute.data) {
    alert("Prima genera il grafico (premi Aggiorna).");
    return;
  }

  const btn = document.getElementById("pdf_btn");
  if (btn) {
    btn.disabled = true;
    btn.innerText = "Creo PDF…";
  }

  try {
    const { data, comp, capital } = _lastCompute;
    const m = data.metrics || {};

    // immagine del grafico
    const chart_png = chart.toBase64Image("image/png", 1.0);

    // meta per il PDF
    const start = data.dates?.[0] || "";
    const end = data.dates?.[data.dates.length - 1] || "";
    const anni_periodo = computeYearsBetween(start, end);

    const capitale_finale = (data.portfolio && data.portfolio.length)
      ? data.portfolio[data.portfolio.length - 1]
      : null;

    // anni raddoppio (fallback)
    let ytd = m.years_to_double;
    if (!(Number.isFinite(Number(ytd)) && Number(ytd) > 0)) {
      const cagr = Number(m.cagr_portfolio);
      if (Number.isFinite(cagr) && cagr > 0) ytd = Math.log(2) / Math.log(1 + cagr);
      else ytd = null;
    }

    const body = {
      chart_png,
      meta: {
        start,
        end,
        oro_pct: comp.gold.toFixed(0),
        azionario_pct: comp.equity.toFixed(0),
        obbligazionario_pct: comp.bonds.toFixed(0),
        capitale_iniziale: capital,
        capitale_finale: capitale_finale,
        anni_periodo: anni_periodo,
        cagr: m.cagr_portfolio,
        max_dd: m.max_dd_portfolio,
        years_to_double: ytd,
      }
    };

    const res = await fetch("/api/pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || "Errore nella generazione del PDF.");
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "gloob_metodo_pigro.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();

    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (e) {
    console.error(e);
    alert(String(e.message || e));
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = "Crea PDF";
    }
  }
}

/* ===== Assistente (ChatGPT) ===== */
function setupAssistant() {
  const btn = document.getElementById("ask_btn");
  const ta = document.getElementById("ask_text");
  const out = document.getElementById("ask_answer");
  const status = document.getElementById("ask_status");

  if (!btn || !ta || !out) return;

  async function ask() {
    const q = (ta.value || "").trim();
    if (!q) {
      alert("Scrivi prima una domanda.");
      ta.focus();
      return;
    }

    btn.disabled = true;
    if (status) {
      status.style.display = "inline";
      status.innerText = "Sto pensando…";
    }
    out.innerText = "";

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
            oro_pct: comp.gold.toFixed(0),
            azionario_pct: comp.equity.toFixed(0),
            obbligazionario_pct: comp.bonds.toFixed(0),
            capitale_eur: new Intl.NumberFormat("it-IT", { maximumFractionDigits: 0 }).format(capital),
          },
        }),
      });

      const data = await res.json();
      if (!res.ok || data.error) {
        throw new Error(data.error || "Errore assistente.");
      }

      out.innerText = data.answer || "(risposta vuota)";
      if (status) {
        status.innerText = data.remaining !== undefined
          ? `Ok (richieste residue nell’ora: ${data.remaining}).`
          : "Ok.";
        setTimeout(() => (status.style.display = "none"), 2500);
      }
    } catch (e) {
      console.error(e);
      out.innerText = "";
      alert(String(e.message || e));
      if (status) status.style.display = "none";
    } finally {
      btn.disabled = false;
    }
  }

  btn.addEventListener("click", ask);
  ta.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) ask();
  });
}

function init() {
  loadData();

  const btn = document.querySelector("button"); // Aggiorna (primo button)
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  const pdfBtn = document.getElementById("pdf_btn");
  if (pdfBtn) {
    pdfBtn.addEventListener("click", (e) => {
      e.preventDefault();
      creaPdf();
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

  setupAssistant();
}

window.addEventListener("DOMContentLoaded", init);
