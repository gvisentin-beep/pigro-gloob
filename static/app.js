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

async function loadData() {
  const w_ls80 = Number(document.getElementById("w_ls80")?.value);
  const w_gold = Number(document.getElementById("w_gold")?.value);
  const capital = Number(document.getElementById("initial")?.value) || 10000;

  // Anti-cache totale (fondamentale su Render)
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

  // Aggiorna periodo (se esiste lo span/id)
  if (Array.isArray(data.dates) && data.dates.length > 1) {
    const periodEl = document.getElementById("period");
    if (periodEl) {
      periodEl.innerText = `${data.dates[0]} → ${data.dates[data.dates.length - 1]}`;
    }
  }

  // ✅ Aggiorna testo CAGR e Max DD in modo ROBUSTO (senza dipendere da <span>)
  if (data.metrics && Array.isArray(data.dates) && data.dates.length > 1) {
    const m = data.metrics;

    const riga1 =
      `Portafoglio (LS80+Oro): CAGR ${pct(m.cagr_portfolio)} | Max DD ${pct(m.max_dd_portfolio)}`;
    const riga2 =
      `Solo LS80: CAGR ${pct(m.cagr_solo)} | Max DD ${pct(m.max_dd_solo)}`;
    const riga3 =
      `Periodo: ${data.dates[0]} → ${data.dates[data.dates.length - 1]}`;

    // trova la card che contiene il testo "Portafoglio (LS80+Oro)"
    const card = Array.from(document.querySelectorAll(".card")).find((el) =>
      el.innerText.includes("Portafoglio (LS80+Oro)")
    );

    if (card) {
      card.innerHTML = `<b>${riga1}</b><br>${riga2}<br>${riga3}`;
    }
  }

  // Grafico
  const canvas = document.getElementById("chart");
  if (!canvas) {
    console.warn("Canvas #chart non trovato.");
    return;
  }
  const ctx = canvas.getContext("2d");

  // Distrugge il grafico precedente (fondamentale)
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
          tension: 0.15,
          pointRadius: 0,
        },
        {
          label: "Solo LS80",
          data: data.solo_ls80, // backend: "solo_ls80"
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
            label: (context) =>
              `${context.dataset.label}: ${euro(context.parsed.y)}`,
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

// Collegamento SOLIDO al bottone Aggiorna (+ Enter sui campi)
function init() {
  // primo caricamento
  loadData();

  // Bottone Aggiorna
  const btn = document.querySelector("button");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  // Enter nei campi
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

window.addEventListener("DOMContentLoaded", init);
