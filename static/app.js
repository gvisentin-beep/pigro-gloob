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
  // 1 decimale, con virgola italiana
  return x.toFixed(1).replace(".", ",");
}

// Calcola composizione a 3 componenti partendo da Oro (0..20 step 5)
// Ritorna: gold, ls80, equity, bonds (tutti in %)
function computeCompositionFromGoldInput() {
  let gold = Number(document.getElementById("w_gold")?.value);
  if (!Number.isFinite(gold)) gold = 10;

  gold = clamp(gold, 0, 20);
  gold = snapToStep(gold, 5);

  // riscrivo il valore (lo slider “scatta” bene)
  const goldEl = document.getElementById("w_gold");
  if (goldEl) goldEl.value = gold;

  const ls80 = 100 - gold;
  const equity = ls80 * 0.80;
  const bonds = ls80 * 0.20;

  return { gold, ls80, equity, bonds };
}

// Nasconde la riga "Composizione risultante ..." sotto i controlli (se presente)
// Mantiene SOLO il badge dello slider Oro
function updateGoldBadgeAndHideInlineComposition(comp) {
  const badge = document.getElementById("gold_badge");
  if (badge) badge.innerText = `${comp.gold.toFixed(0)}%`;

  // Se nel tuo HTML c'è la riga con questi span, nascondila (evita duplicazione)
  const ge = document.getElementById("w_equity");
  if (ge) {
    let container = ge.closest("div");
    if (container) container.style.display = "none";
  }
}

function findMetricsCard() {
  return Array.from(document.querySelectorAll(".card")).find((el) =>
    el.innerText.includes("Portafoglio (LS80+Oro)")
  );
}

// Ritorna:
// - yearStart: Set indici del primo giorno disponibile dell'anno
// - yearMarkers: Set indici per le linee verticali "importanti" (2/anno): inizio + metà
function computeYearMarkers(dateStrings) {
  const byYear = {}; // year -> array indici

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

async function loadData() {
  // 1) composizione vincolata (oro max 20, step 5)
  const comp = computeCompositionFromGoldInput();
  updateGoldBadgeAndHideInlineComposition(comp);

  // 2) parametri per API (strumenti reali: LS80 + Oro)
  const w_ls80 = comp.ls80; // %
  const w_gold = comp.gold; // %
  const capital = Number(document.getElementById("initial")?.value) || 10000;

  // 3) chiamata API con anti-cache
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

  // 4) periodo (resta QUI, sotto, non nel blocco metriche)
  if (Array.isArray(data.dates) && data.dates.length > 1) {
    const p = document.getElementById("period");
    if (p) p.innerText = `${data.dates[0]} → ${data.dates[data.dates.length - 1]}`;
  }

  // 5) metriche + composizione + raddoppio (UNA SOLA VOLTA, qui)
  if (data.metrics && Array.isArray(data.dates) && data.dates.length > 1) {
    const m = data.metrics;

    const riga1 =
      `Portafoglio (LS80+Oro): CAGR ${pct(m.cagr_portfolio)} | Max DD ${pct(m.max_dd_portfolio)}`;

    const riga2 =
      `Solo LS80: CAGR ${pct(m.cagr_solo)} | Max DD ${pct(m.max_dd_solo)}`;

    const riga3 =
      `Composizione: Azionario ${comp.equity.toFixed(0)}% | Obbligazionario ${comp.bonds.toFixed(0)}% | Oro ${comp.gold.toFixed(0)}%`;

    // Raddoppio: uso years_to_double se disponibile, altrimenti lo calcolo dal CAGR
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

  // 6) grafico
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
            callback: function (value, index) {
              if (!yearStart.has(index)) return "";
              return String(labels[index]).slice(0, 4); // YYYY
            },
            autoSkip: false,
            maxRotation: 0,
            minRotation: 0,
          },
          grid: {
            color: function (context) {
              if (yearMarkers.has(context.index)) return "rgba(0,0,0,0.16)";
              return "rgba(0,0,0,0.03)";
            },
            lineWidth: function (context) {
              if (yearMarkers.has(context.index)) return 1.2;
              return 0.2;
            },
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

  // Bottone Aggiorna
  const btn = document.querySelector("button");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  // Enter nel capitale
  const capEl = document.getElementById("initial");
  if (capEl) {
    capEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        loadData();
      }
    });
  }

  // Slider Oro:
  // - input: aggiorna badge (e nasconde composizione duplicata)
  // - change: ricalcola grafico
  const goldEl = document.getElementById("w_gold");
  if (goldEl) {
    goldEl.addEventListener("input", () => {
      const comp = computeCompositionFromGoldInput();
      updateGoldBadgeAndHideInlineComposition(comp);
    });

    goldEl.addEventListener("change", () => {
      loadData();
    });
  }
}

window.addEventListener("DOMContentLoaded", init);
