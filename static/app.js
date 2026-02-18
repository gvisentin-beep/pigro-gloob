let chart;

// % con 1 decimale (per CAGR e Max DD)
function pct1(x) {
  return (x * 100).toFixed(1) + "%";
}

// Euro con decimali (per grafico)
function eur(x) {
  return new Intl.NumberFormat("it-IT", { style: "currency", currency: "EUR" }).format(x);
}

// Euro senza decimali (per valore finale)
function eur0(x) {
  const v = Math.round(Number(x) || 0);
  return new Intl.NumberFormat("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
    minimumFractionDigits: 0,
  }).format(v);
}

function safeSet(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

async function loadData() {
  // Se Chart.js non è caricato correttamente, qui prima si bloccava tutto.
  if (typeof Chart === "undefined") {
    console.error("Chart.js non caricato: Chart è undefined. Usa la build UMD.");
    const box = document.getElementById("chart");
    if (box) {
      box.innerHTML =
        '<div style="padding:12px;font-weight:700;color:#b91c1c">Errore: Chart.js non caricato. (Controlla lo script chart.umd.min.js)</div>';
    }
    return;
  }

  const w_ls80 = Number(document.getElementById("w_ls80").value) / 100;
  const w_gold = Number(document.getElementById("w_gold").value) / 100;
  const capital = Number(document.getElementById("capital").value || 10000);

  const url = `/api/compute?w_ls80=${encodeURIComponent(w_ls80)}&w_gold=${encodeURIComponent(w_gold)}&initial=${encodeURIComponent(capital)}`;

  let data;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
  } catch (e) {
    console.error("Errore fetch /api/compute:", e);
    safeSet("cagr", "—");
    safeSet("mdd", "—");
    safeSet("cagr_ls80", "—");
    safeSet("mdd_ls80", "—");
    safeSet("implied", "—");
    safeSet("endvalues", "—");
    return;
  }

  // STAT
  safeSet("cagr", data.cagr != null ? pct1(data.cagr) : "—");
  safeSet("mdd", data.max_dd != null ? pct1(data.max_dd) : "—");
  safeSet("cagr_ls80", data.cagr_ls80 != null ? pct1(data.cagr_ls80) : "—");
  safeSet("mdd_ls80", data.max_dd_ls80 != null ? pct1(data.max_dd_ls80) : "—");
  safeSet("implied", data.implied != null ? pct1(data.implied) : "—");
  safeSet("endvalues", data.final_value != null ? eur0(data.final_value) : "—");

  // periodo (se presente in pagina)
  if (data.start && data.end) {
    safeSet("period", `${data.start} → ${data.end}`);
  }

  // GRAFICO
  const ctx = document.getElementById("chart_canvas").getContext("2d");

  const labels = data.dates || [];
  const seriesPort = data.portfolio || [];
  const seriesLs80 = data.ls80 || [];

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Portafoglio (LS80+Oro)",
          data: seriesPort,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
        },
        {
          label: "Solo LS80",
          data: seriesLs80,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          ticks: {
            callback: (v) => eur(v),
          },
        },
        x: {
          ticks: {
            maxTicksLimit: 10,
          },
        },
      },
      plugins: {
        legend: { display: true },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${eur(ctx.parsed.y)}`,
          },
        },
      },
    },
  });
}

function setup() {
  const btnUpdate = document.getElementById("btn_update");
  const btnPdf = document.getElementById("btn_pdf");

  if (btnUpdate) btnUpdate.addEventListener("click", loadData);

  if (btnPdf) {
    btnPdf.addEventListener("click", async () => {
      // se hai un endpoint PDF server-side, chiamalo qui.
      // per ora: stampa la pagina (funziona ovunque).
      window.print();
    });
  }

  // Primo caricamento automatico
  loadData();
}

document.addEventListener("DOMContentLoaded", setup);
