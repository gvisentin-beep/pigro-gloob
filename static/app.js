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

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.innerText = text;
}

async function loadData() {
  const w_ls80 = Number(document.getElementById("w_ls80").value);
  const w_gold = Number(document.getElementById("w_gold").value);
  const capital = Number(document.getElementById("initial").value) || 10000;

  const url =
    `/api/compute?w_ls80=${encodeURIComponent(w_ls80)}` +
    `&w_gold=${encodeURIComponent(w_gold)}` +
    `&capital=${encodeURIComponent(capital)}` +
    `&t=${Date.now()}`;

  const res = await fetch(url, { cache: "no-store" });
  const data = await res.json();

  if (data.error) {
    console.error(data.error);
    alert(data.error);
    return;
  }

  // Periodo
  if (Array.isArray(data.dates) && data.dates.length > 1) {
    setText("period", `${data.dates[0]} → ${data.dates[data.dates.length - 1]}`);
  }

  // Metriche
  if (data.metrics) {
    setText("cagr_portfolio", pct(data.metrics.cagr_portfolio));
    setText("max_dd_portfolio", pct(data.metrics.max_dd_portfolio));
    setText("cagr_solo", pct(data.metrics.cagr_solo));
    setText("max_dd_solo", pct(data.metrics.max_dd_solo));
  }

  // Grafico
  const canvas = document.getElementById("chart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
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
          tension: 0.15,
        },
        {
          label: "Solo LS80",
          data: data.solo_ls80,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          ticks: { callback: (value) => euro(value) },
        },
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: (c) => `${c.dataset.label}: ${euro(c.parsed.y)}`,
          },
        },
      },
    },
  });
}

function wireUI() {
  // Bottone Aggiorna (prende il primo button che trova)
  const btn = document.querySelector("button");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }
}

window.addEventListener("DOMContentLoaded", () => {
  wireUI();
  loadData();
});
