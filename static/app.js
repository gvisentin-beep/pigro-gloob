let chart;

// % con 1 decimale (per CAGR e Max DD)
function pct1(x) {
  return (x * 100).toFixed(1) + "%";
}

// % con 0 decimali (se un domani vuoi usarlo)
function pct0(x) {
  return (x * 100).toFixed(0) + "%";
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
    minimumFractionDigits: 0
  }).format(v);
}

function isoToMonthYear(iso) {
  // "YYYY-MM-DD" -> "MM-YYYY"
  if (!iso || iso.length < 10) return iso;
  const [y, m] = iso.slice(0, 7).split("-");
  return `${m}-${y}`;
}

async function loadData() {
  const w_ls80 = Number(document.getElementById("w_ls80").value) / 100;
  const w_gold = Number(document.getElementById("w_gold").value) / 100;
  const initial = Number(document.getElementById("initial").value || 10000);

  const url = `/api/compute?w_ls80=${encodeURIComponent(w_ls80)}&w_gold=${encodeURIComponent(w_gold)}&initial=${encodeURIComponent(initial)}`;
  const r = await fetch(url);
  const data = await r.json();

  if (!r.ok) {
    alert(data.error || "Errore");
    return;
  }

  // Periodo (solo mese/anno)
  document.getElementById("period_top").textContent =
    `${isoToMonthYear(data.stats.start)} → ${isoToMonthYear(data.stats.end)}`;

  // Stats: 1 decimale
  document.getElementById("cagr").textContent = pct1(data.stats.cagr);
  document.getElementById("mdd").textContent = pct1(data.stats.max_drawdown);
  document.getElementById("cagr_ls80").textContent = pct1(data.stats.cagr_ls80);
  document.getElementById("mdd_ls80").textContent = pct1(data.stats.max_drawdown_ls80);

  // Esposizione implicita (qui lascio come prima: 2 decimali in %)
  // Se invece la vuoi a 0 decimali: sostituisci pct1->pct0 o usa una funzione dedicata.
  document.getElementById("implied").textContent =
    `Azioni ${pct1(data.implied.equity)} · Obbligazioni ${pct1(data.implied.bonds)} · Oro ${pct1(data.implied.gold)}`;

  // Valori finali: arrotondati all'euro (senza decimali)
  const last = data.series[data.series.length - 1];
  const lastPort = Math.round(last.eur_port);
  const lastLS80 = Math.round(last.eur_ls80);
  const diff = lastPort - lastLS80;

  document.getElementById("endvalues").textContent =
    `LS80+Oro ${eur0(lastPort)} · Solo LS80 ${eur0(lastLS80)} · Differenza ${eur0(diff)}`;

  // Asse X: mese/anno
  const labelsMY = data.series.map(r => isoToMonthYear(r.date));
  const eurPort = data.series.map(r => r.eur_port);
  const eurLS80 = data.series.map(r => r.eur_ls80);

  const ctx = document.getElementById("chart");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labelsMY,
      datasets: [
        {
          label: "LS80 + Oro (valore in €)",
          data: eurPort,
          fill: false,
          tension: 0.15,
          pointRadius: 0,
          pointHoverRadius: 3,
          borderWidth: 1.6
        },
        {
          label: "Solo LS80 (valore in €)",
          data: eurLS80,
          fill: false,
          tension: 0.15,
          pointRadius: 0,
          pointHoverRadius: 3,
          borderWidth: 1.6
        }
      ]
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { display: true } },
      scales: {
        x: {
          ticks: { maxTicksLimit: 10, autoSkip: true }
        }
      }
    }
  });
}

document.getElementById("btn_update").addEventListener("click", loadData);
document.getElementById("btn_pdf").addEventListener("click", () => window.print());
window.addEventListener("load", loadData);
