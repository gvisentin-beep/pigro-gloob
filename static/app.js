let chart = null;

function euro(x) {
  return new Intl.NumberFormat("it-IT", { style: "currency", currency: "EUR" }).format(x);
}

async function loadData() {
  const w_ls80_pct = Number(document.getElementById("w_ls80").value);
  const w_gold_pct = Number(document.getElementById("w_gold").value);
  const capital = Number(document.getElementById("initial").value) || 10000;

  // Cache-busting + no-store: evita il problema “funziona solo con F12”
  const url = `/api/compute?w_ls80=${encodeURIComponent(w_ls80_pct)}&w_gold=${encodeURIComponent(w_gold_pct)}&capital=${encodeURIComponent(capital)}&t=${Date.now()}`;

  const res = await fetch(url, { cache: "no-store" });
  const data = await res.json();

  if (data.error) {
    console.error(data.error);
    alert(data.error);
    return;
  }

  document.getElementById("period").innerText =
    data.dates[0] + " → " + data.dates[data.dates.length - 1];

  const ctx = document.getElementById("chart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        { label: "Portafoglio (LS80+Oro)", data: data.portfolio, borderWidth: 2 },
        { label: "Solo LS80", data: data.solo_ls80, borderWidth: 2 },
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
    },
  });
}

// Carica automaticamente all'avvio
window.onload = loadData;
