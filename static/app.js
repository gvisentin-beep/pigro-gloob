let mainChart = null;
let ddChart = null;

function euro(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0
  });
}

function pct(v, d = 2) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("it-IT", {
    minimumFractionDigits: d,
    maximumFractionDigits: d
  }) + "%";
}

function yearTickCallback(value, index, ticks) {
  const label = this.getLabelForValue(value);
  if (!label) return "";

  const year = String(label).slice(0, 4);

  if (index === 0) return year;

  const prevLabel = this.getLabelForValue(ticks[index - 1].value);
  const prevYear = prevLabel ? String(prevLabel).slice(0, 4) : "";

  return year !== prevYear ? year : "";
}

async function loadCharts() {
  try {
    const capitalEl = document.getElementById("capital");
    const capital = capitalEl ? (capitalEl.value || "10000") : "10000";

    const res = await fetch(`/api/compute?capital=${encodeURIComponent(capital)}`, {
      cache: "no-store"
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Errore backend");

    const labelsRaw = Array.isArray(data.dates) ? data.dates : [];
    const portRaw = Array.isArray(data.portfolio) ? data.portfolio : [];
    const worldRaw = Array.isArray(data.world) ? data.world : [];
    const ddRaw = Array.isArray(data.drawdown_portfolio_pct) ? data.drawdown_portfolio_pct : [];

    const n = Math.min(labelsRaw.length, portRaw.length, worldRaw.length, ddRaw.length);
    if (!n) throw new Error("Nessun dato disponibile");

    const labels = [];
    const portfolio = [];
    const world = [];
    const dd = [];

    for (let i = 0; i < n; i++) {
      const lab = labelsRaw[i];
      const p = Number(portRaw[i]);
      const w = Number(worldRaw[i]);
      const d = Number(ddRaw[i]);

      if (!lab) continue;
      if (!Number.isFinite(p) || !Number.isFinite(w) || !Number.isFinite(d)) continue;

      labels.push(lab);
      portfolio.push(p);
      world.push(w);
      dd.push(d);
    }

    if (!labels.length) throw new Error("Dati non validi per il grafico");

    if (mainChart) mainChart.destroy();
    if (ddChart) ddChart.destroy();

    const ctxMain = document.getElementById("chart_main");
    const ctxDD = document.getElementById("chart_dd");
    if (!ctxMain || !ctxDD) throw new Error("Canvas mancanti");
    if (typeof Chart === "undefined") throw new Error("Chart.js non caricato");

    mainChart = new Chart(ctxMain, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Metodo Pigro 80/15/5",
            data: portfolio,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          },
          {
            label: "MSCI World",
            data: world,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        plugins: {
          legend: {
            display: true
          }
        },
        scales: {
          x: {
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback: yearTickCallback
            }
          },
          y: {
            ticks: {
              callback: function(value) {
                return Number(value).toLocaleString("it-IT");
              }
            }
          }
        }
      }
    });

    ddChart = new Chart(ctxDD, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Drawdown",
            data: dd,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        plugins: {
          legend: {
            display: true
          }
        },
        scales: {
          x: {
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback: yearTickCallback
            }
          },
          y: {
            ticks: {
              callback: function(value) {
                return value;
              }
            }
          }
        }
      }
    });

    const metrics = data.metrics || {};
    const lastPortfolio = portfolio[portfolio.length - 1];
    const lastWorld = world[world.length - 1];

    const finalValueEl = document.getElementById("final_value");
    const finalYearsEl = document.getElementById("final_years");
    const cagrEl = document.getElementById("cagr");
    const maxddEl = document.getElementById("maxdd");
    const dblEl = document.getElementById("dbl");
    const comparePigroEl = document.getElementById("compare_pigro");
    const compareWorldEl = document.getElementById("compare_world");
    const comparePeriodEl = document.getElementById("compare_period");
    const ddSummaryEl = document.getElementById("dd_summary");

    if (finalValueEl) finalValueEl.textContent = euro(lastPortfolio);
    if (finalYearsEl) {
      const y = Number(metrics.final_years);
      finalYearsEl.textContent = Number.isFinite(y) ? y.toFixed(1) : "—";
    }
    if (cagrEl) {
      const c = Number(metrics.cagr_portfolio) * 100;
      cagrEl.textContent = Number.isFinite(c) ? pct(c) : "—";
    }
    if (maxddEl) {
      const m = Number(metrics.max_dd_portfolio) * 100;
      maxddEl.textContent = Number.isFinite(m) ? pct(m) : "—";
    }
    if (dblEl) {
      const dby = Number(metrics.doubling_years_portfolio);
      dblEl.textContent = Number.isFinite(dby) ? dby.toFixed(1) : "—";
    }
    if (comparePigroEl) comparePigroEl.textContent = euro(lastPortfolio);
    if (compareWorldEl) compareWorldEl.textContent = euro(lastWorld);
    if (comparePeriodEl) comparePeriodEl.textContent = `${capital} € investiti all'inizio del periodo`;
    if (ddSummaryEl) {
      const m = Number(metrics.max_dd_portfolio) * 100;
      ddSummaryEl.textContent = Number.isFinite(m)
        ? `Peggior ribasso del portafoglio: ${pct(m)}`
        : "";
    }

    console.log("Grafici caricati:", {
      punti: labels.length,
      primo: labels[0],
      ultimo: labels[labels.length - 1]
    });

  } catch (err) {
    console.error("Errore grafici:", err);
    alert("Errore caricamento grafici: " + err.message);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const btn = document.getElementById("btn_update");
  if (btn) btn.addEventListener("click", loadCharts);
  loadCharts();
});
