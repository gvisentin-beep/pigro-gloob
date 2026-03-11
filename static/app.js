function formatEuro(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0
  });
}

function formatPctFromDecimal(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return (Number(n) * 100).toFixed(1) + "%";
}

function clampGold(v) {
  v = Number(v);
  if (Number.isNaN(v)) v = 20;
  v = Math.max(0, Math.min(50, v));
  return Math.round(v / 5) * 5;
}

function parseCapital() {
  let s = String(document.getElementById("capital").value || "").trim();
  s = s.replace(/\./g, "").replace(",", ".");
  let v = Number(s);
  if (Number.isNaN(v) || v <= 0) v = 10000;
  return v;
}

function formatDateLabel(dateStr) {
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString("it-IT");
}

function buildYearTickCallback(labels) {
  return function(value, index) {

    const d = new Date(labels[index]);
    if (isNaN(d)) return "";

    const year = d.getFullYear();

    if (index === 0) return year;

    const prev = new Date(labels[index - 1]);

    if (prev.getFullYear() !== year) {
      return year;
    }

    return "";
  };
}

let chartMain = null;
let chartDD = null;

const elWG = document.getElementById("w_gold");
const elWGLabel = document.getElementById("w_gold_label");
const btnUpdate = document.getElementById("btn_update");
const btnPdf = document.getElementById("btn_pdf");

function updateSliderLabelAndComposition(wGold01) {

  elWGLabel.textContent = Math.round(wGold01 * 100) + "%";

  const wLs80 = 1 - wGold01;

  document.getElementById("w_equity").textContent =
    Math.round(0.8 * wLs80 * 100) + "%";

  document.getElementById("w_bond").textContent =
    Math.round(0.2 * wLs80 * 100) + "%";

  document.getElementById("w_gold2").textContent =
    Math.round(wGold01 * 100) + "%";
}

function renderComparison(payload) {

  const period = document.getElementById("compare_period");
  const pigro = document.getElementById("compare_pigro");
  const world = document.getElementById("compare_world");

  const dates = payload.dates || [];
  const portfolio = payload.portfolio || [];
  const worldArr = payload.world || [];

  if (dates.length && period) {
    period.textContent =
      `10.000 € investiti dal ${dates[0]} al ${dates[dates.length - 1]}`;
  }

  if (pigro) {
    pigro.textContent = portfolio.length
      ? formatEuro(portfolio[portfolio.length - 1])
      : "—";
  }

  if (world) {
    world.textContent = worldArr.length
      ? formatEuro(worldArr[worldArr.length - 1])
      : "—";
  }
}

async function loadData() {

  btnUpdate.disabled = true;
  btnUpdate.textContent = "Aggiorna…";

  try {

    const wGold = clampGold(elWG.value) / 100;
    const capital = parseCapital();

    updateSliderLabelAndComposition(wGold);

    const res = await fetch(
      `/api/compute?w_gold=${encodeURIComponent(wGold)}&capital=${encodeURIComponent(capital)}`
    );

    const data = await res.json();

    if (!data.ok) {
      alert("Errore: " + (data.error || "API compute"));
      return;
    }

    const dates = data.dates || [];
    const portfolio = data.portfolio || [];
    const world = data.world || [];
    const ddP = data.drawdown_portfolio_pct || [];
    const ddW = data.drawdown_world_pct || [];

    // -----------------------
    // taglio serie dal 2021
    // -----------------------

    let startIndex = 0;

    for (let i = 0; i < dates.length; i++) {
      if (dates[i] >= "2021-01-01") {
        startIndex = i;
        break;
      }
    }

    const datesCut = dates.slice(startIndex);
    const portfolioCut = portfolio.slice(startIndex);
    const worldCut = world.slice(startIndex);
    const ddPCut = ddP.slice(startIndex);
    const ddWCut = ddW.slice(startIndex);

    const labels = datesCut;

    const xTickCallback = buildYearTickCallback(labels);

    // titolo grafico

    const titleBox = document.getElementById("chart_title");
    if (titleBox) {
      titleBox.textContent = "Andamento negli ultimi anni";
    }

    // -----------------------
    // GRAFICO PRINCIPALE
    // -----------------------

    if (chartMain) chartMain.destroy();

    chartMain = new Chart(
      document.getElementById("chart_main").getContext("2d"),
      {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
              data: portfolioCut,
              borderWidth: 2,
              tension: 0.15,
              pointRadius: 0
            },
            {
              label: "MSCI World (URTH)",
              data: worldCut,
              borderWidth: 2,
              tension: 0.15,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {

            x: {
              type: "category",
              ticks: {
                callback: xTickCallback,
                maxRotation: 0,
                minRotation: 0
              }
            },

            y: {
              ticks: {
                callback: (v) => formatEuro(v)
              }
            }

          },
          plugins: {
            legend: { display: true },
            tooltip: {
              callbacks: {
                title: (items) =>
                  formatDateLabel(labels[items[0].dataIndex]),
                label: (ctx) =>
                  ctx.dataset.label + ": " + formatEuro(ctx.parsed.y)
              }
            }
          }
        }
      }
    );

    // -----------------------
    // GRAFICO DRAWDOWN
    // -----------------------

    if (chartDD) chartDD.destroy();

    chartDD = new Chart(
      document.getElementById("chart_dd").getContext("2d"),
      {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Drawdown Portafoglio (%)",
              data: ddPCut,
              borderWidth: 2,
              tension: 0.15,
              pointRadius: 0
            },
            {
              label: "Drawdown MSCI World (%)",
              data: ddWCut,
              borderWidth: 2,
              tension: 0.15,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {

            x: {
              type: "category",
              ticks: {
                callback: xTickCallback,
                maxRotation: 0,
                minRotation: 0
              }
            },

            y: {
              suggestedMin: -60,
              suggestedMax: 0,
              ticks: {
                callback: (v) => v.toFixed(0) + "%"
              }
            }

          },
          plugins: {
            legend: { display: true }
          }
        }
      }
    );

    renderComparison(data);

  } catch (e) {

    alert("Errore: " + (e?.message || e));

  } finally {

    btnUpdate.disabled = false;
    btnUpdate.textContent = "Aggiorna";

  }
}

elWG.addEventListener("input", () => {
  const wGold = clampGold(elWG.value);
  elWG.value = String(wGold);
  updateSliderLabelAndComposition(wGold / 100);
});

btnUpdate.addEventListener("click", loadData);

btnPdf.addEventListener("click", () => {

  const cagr = document.getElementById("cagr").textContent || "";
  const maxdd = document.getElementById("maxdd").textContent || "";
  const finalv = document.getElementById("final_value").textContent || "";
  const years = document.getElementById("final_years").textContent || "";

  const url =
    `/api/pdf?title=${encodeURIComponent("Gloob - Metodo Pigro")}` +
    `&cagr=${encodeURIComponent(cagr)}` +
    `&maxdd=${encodeURIComponent(maxdd)}` +
    `&final=${encodeURIComponent(finalv)}` +
    `&years=${encodeURIComponent(years)}`;

  window.open(url, "_blank");

});

updateSliderLabelAndComposition(clampGold(elWG.value) / 100);

loadData();
