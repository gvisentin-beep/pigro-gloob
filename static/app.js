/* static/app.js
   Gloob - Metodo Pigro
   - fetch anti-cache
   - update Chart.js
   - aggiornamento metriche (se gli elementi esistono in pagina)
*/

let chart = null;

function euro(x) {
  return new Intl.NumberFormat("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(Number(x));
}

function pct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return (Number(x) * 100).toFixed(1) + "%";
}

function setTextIfExists(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

function getNumberById(id, fallback) {
  const el = document.getElementById(id);
  if (!el) return fallback;
  const v = Number(el.value);
  return Number.isFinite(v) ? v : fallback;
}

function ensureWeightsSumTo100(w1, w2) {
  // Se l’utente mette 60 e 40 ok; se mette 0 e 0, evito divisioni strane lato server
  const s = w1 + w2;
  if (!Number.isFinite(s) || s <= 0) return { w1: 80, w2: 20 };
  return { w1, w2 };
}

async function loadData() {
  const rawWls = getNumberById("w_ls80", 80);
  const rawWg = getNumberById("w_gold", 20);
  const { w1: w_ls80_pct, w2: w_gold_pct } = ensureWeightsSumTo100(rawWls, rawWg);

  const capital = getNumberById("initial", 10000);

  // Cache-busting + no-store (risolve “funziona solo con F12”)
  const url =
    `/api/compute` +
    `?w_ls80=${encodeURIComponent(w_ls80_pct)}` +
    `&w_gold=${encodeURIComponent(w_gold_pct)}` +
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

  if (data.error) {
    console.error(data.error);
    alert(data.error);
    return;
  }

  // Periodo
  if (Array.isArray(data.dates) && data.dates.length >= 2) {
    setTextIfExists("period", `${data.dates[0]} → ${data.dates[data.dates.length - 1]}`);
  }

  // Metriche (se presenti nel JSON e se gli elementi esistono nell’HTML)
  // NB: nel tuo app.py ho previsto data.metrics
  if (data.metrics) {
    // Se nel tuo HTML ci sono questi id, li riempie automaticamente.
    // Altrimenti non succede nulla (nessun errore).
    setTextIfExists("cagr_portfolio", pct(data.metrics.cagr_portfolio));
    setTextIfExists("max_dd_portfolio", pct(data.metrics.max_dd_portfolio));
    setTextIfExists("cagr_solo", pct(data.metrics.cagr_solo));
    setTextIfExists("max_dd_solo", pct(data.metrics.max_dd_solo));

    // Facoltativi
    setTextIfExists("final_portfolio", euro(data.metrics.final_portfolio));
    setTextIfExists("final_solo", euro(data.metrics.final_solo));
  }

  // Grafico
  const canvas = document.getElementById("chart");
  if (!canvas) {
    console.warn("Canvas #chart non trovato.");
    return;
  }
  const ctx = canvas.getContext("2d");

  // Distruggo e ricreo: più semplice e robusto
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Portafoglio (LS80+Oro)",
          data: data.portfolio,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: "Solo LS80",
          data: data.solo_ls80, // attenzione: nel backend è "solo_ls80"
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
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
            label: (ctx) => `${ctx.dataset.label}: ${euro(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        y: {
          ticks: {
            callback: (value) => euro(value),
          },
        },
      },
    },
  });
}

function wireUI() {
  // Bottone Aggiorna: aggancio robusto
  const btn = document.getElementById("updateBtn") || document.querySelector("button");
  if (btn) {
    btn.addEventListener("click", (e) => {
      // evita submit/refresh pagina
      e.preventDefault();
      loadData();
    });
  }

  // Enter dentro ai campi: aggiorna
  ["w_ls80", "w_gold", "initial"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        loadData();
      }
    });
  });
}

// Avvio
window.addEventListener("DOMContentLoaded", () => {
  wireUI();
  loadData();
});
