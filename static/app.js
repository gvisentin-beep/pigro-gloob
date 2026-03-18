let mainChart = null;
let ddChart = null;

async function loadCharts() {
  try {
    const capitalEl = document.getElementById("capital");
    const capital = capitalEl ? (capitalEl.value || "10000") : "10000";

    const res = await fetch(`/api/compute?capital=${encodeURIComponent(capital)}`, {
      cache: "no-store"
    });

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    console.log("API compute:", data);

    const labelsRaw = Array.isArray(data.dates) ? data.dates : [];
    const portfolioRaw = Array.isArray(data.portfolio) ? data.portfolio : [];
    const worldRaw = Array.isArray(data.world) ? data.world : [];
    const ddRaw = Array.isArray(data.drawdown_portfolio_pct) ? data.drawdown_portfolio_pct : [];

    const n = Math.min(labelsRaw.length, portfolioRaw.length, worldRaw.length, ddRaw.length);
    if (!n) {
      throw new Error("Nessun dato disponibile");
    }

    const labels = [];
    const portfolio = [];
    const world = [];
    const dd = [];

    for (let i = 0; i < n; i++) {
      const p = Number(portfolioRaw[i]);
      const w = Number(worldRaw[i]);
      const d = Number(ddRaw[i]);
      const lab = labelsRaw[i];

      if (!lab) continue;
      if (!Number.isFinite(p) || !Number.isFinite(w) || !Number.isFinite(d)) continue;

      labels.push(lab);
      portfolio.push(p);
      world.push(w);
      dd.push(d);
    }

    if (!labels.length) {
      throw new Error("Tutti i dati risultano non validi");
    }

    const finalValue = portfolio[portfolio.length - 1];
    const finalWorld = world[world.length - 1];

    const metrics = data.metrics || {};
    const cagr = Number(metrics.cagr_portfolio) * 100;
    const maxdd = Number(metrics.max_dd_portfolio) * 100;
    const years = Number(metrics.final_years);
    const dbl = Number(metrics.doubling_years_portfolio);

    const finalValueEl = document.getElementById("final_value");
    const finalYearsEl = document.getElementById("final_years");
    const cagrEl = document.getElementById("cagr");
    const maxddEl = document.getElementById("maxdd");
    const dblEl = document.getElementById("dbl");
    const comparePigroEl = document.getElementById("compare_pigro");
    const compareWorldEl = document.getElementById("compare_world");
    const comparePeriodEl = document.getElementById("compare_period");
    const ddSummaryEl = document.getElementById("dd_summary");

    if (finalValueEl) finalValueEl.textContent = finalValue.toLocaleString("it-IT", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
    if (finalYearsEl) finalYearsEl.textContent = Number.isFinite(years) ? years.toFixed(1) : "—";
    if (cagrEl) cagrEl.textContent = Number.isFinite(cagr) ? `${cagr.toFixed(2)}%` : "—";
    if (maxddEl) maxddEl.textContent = Number.isFinite(maxdd) ? `${maxdd.toFixed(2)}%` : "—";
    if (dblEl) dblEl.textContent = Number.isFinite(dbl) ? dbl.toFixed(1) : "—";
    if (comparePigroEl) comparePigroEl.textContent = finalValue.toLocaleString("it-IT", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
    if (compareWorldEl) compareWorldEl.textContent = finalWorld.toLocaleString("it-IT", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
    if (comparePeriodEl) comparePeriodEl.textContent = `${capital} € investiti all'inizio del periodo`;
    if (ddSummaryEl) ddSummaryEl.textContent = Number.isFinite(maxdd) ? `Peggior ribasso del portafoglio: ${maxdd.toFixed(2)}%` : "";

    const ctx1 = document.getElementById("chart_main");
    const ctx2 = document.getElementById("chart_dd");

    if (!ctx1 || !ctx2) {
      throw new Error("Canvas non trovati");
    }

    if (mainChart) mainChart.destroy();
    if (ddChart) ddChart.destroy();

    mainChart = new Chart(ctx1, {
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
        animation: false
      }
    });

    ddChart = new Chart(ctx2, {
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
        animation: false
      }
    });

  } catch (err) {
    console.error("Errore grafici:", err);
    alert("Errore caricamento grafici: " + err.message);
  }
}

document.addEventListener("DOMContentLoaded", function () {
  const btn = document.getElementById("btn_update");
  if (btn) {
    btn.addEventListener("click", loadCharts);
  }
  loadCharts();
});
