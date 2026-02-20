let chart = null;

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

// Calcola composizione (Oro max 50%, resto LS80 80/20)
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

// Badge oro + nasconde composizione duplicata sotto lo slider
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
    el.innerText.includes("Portafoglio (LS80+Oro)")
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
    if (arr.length === 0) return;
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
  const years = (t1 - t0) / (365.25 * 24 * 3600 * 1000);
  return years;
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

  // Periodo (rimane sotto)
  if (Array.isArray(data.dates) && data.dates.length > 1) {
    const p = document.getElementById("period");
    if (p) p.innerText = `${data.dates[0]} → ${data.dates[data.dates.length - 1]}`;
  }

  // ===== NUOVO: capitale finale + anni (sulla riga capitale) =====
  if (Array.isArray(data.dates) && data.dates.length > 1 && Array.isArray(data.portfolio) && data.portfolio.length > 1) {
    const finalValue = data.portfolio[data.portfolio.length - 1];
    const years = computeYearsBetween(data.dates[0], data.dates[data.dates.length - 1]);

    const outFinal = document.getElementById("final_capital");
    const outYears = document.getElementById("final_years");

    if (outFinal) outFinal.innerText = euro(finalValue);
    if (outYears) outYears.innerText = fmtYearsIt(years);
  }

  // ===== METRICHE =====
  if (data.metrics && Array.isArray(data.dates) && data.dates.length > 1) {
    const m = data.metrics;

    const riga1 =
      `Portafoglio (LS80+Oro): Rendimento annualizzato ${pct(m.cagr_portfolio)} | Max Ribasso nel periodo ${pct(m.max_dd_portfolio)}`;

    const riga2 =
      `Solo LS80: Rendimento annualizzato ${pct(m.cagr_solo)} | Max Ribasso nel periodo ${pct(m.max_dd_solo)}`;

    const riga3 =
      `Composizione: Azionario ${comp.equity.toFixed(0)}% | Obbligazionario ${comp.bonds.toFixed(0)}% | Oro ${comp.gold.toFixed(0)}%`;

    let ytd = m.years_to_double;
    if (!(Number.isFinite(Number(ytd)) && Number(ytd) > 0)) {
      const cagr = Number(m.cagr_portfolio);
      if (Number.isFinite(cagr) && cagr > 0) {
        ytd = Math.log(2) / Math.log(1 + cagr);
      } else {
        ytd = null;
      }
    }

    const riga4 = `Raddoppio del portafoglio in anni: ${fmtYearsIt(ytd)}`;

    const card = findMetricsCard();
    if (card) {
      card.innerHTML = `<b>${riga1}</b><br>${riga2}<br><b>${riga3}</b><br>${riga4}`;
    }
  }

  // ===== GRAFICO =====
  const canvas = document.getElementById("chart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (chart) chart.destroy();

  const labels = data.dates;
  const { yearStart, yearMarkers } = computeYearMarkers(labels);

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Portafoglio (LS80+Oro)",
          data: data.portfolio,
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        },
        {
          label: "Solo LS80",
          data: data.solo_ls80,
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        tooltip: {
          callbacks: {
            label: (c) => `${c.dataset.label}: ${euro(c.parsed.y)}`,
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
            color: (context) => (yearMarkers.has(context.index) ? "rgba(0,0,0,0.18)" : "rgba(0,0,0,0.03)"),
            lineWidth: (context) => (yearMarkers.has(context.index) ? 1.2 : 0.2),
          },
        },
        y: {
          ticks: { callback: (v) => euro(v) },
        },
      },
    },
  });
}

function init() {
  loadData();

  const btn = document.querySelector("button");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  const capEl = document.getElementById("initial");
  if (capEl) {
    capEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        loadData();
      }
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
}

window.addEventListener("DOMContentLoaded", init);
