let currentBenchmark = "world";
let currentMode = "base"; // base | leva_fissa | leva_dinamica

document.querySelectorAll(".benchmarkBtn").forEach(btn => {
  btn.addEventListener("click", () => {

    document.querySelectorAll(".benchmarkBtn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    // gestione benchmark
    if (btn.dataset.benchmark) {
      currentBenchmark = btn.dataset.benchmark;
      currentMode = "base";
    }

    // gestione modalità leva
    if (btn.dataset.mode) {
      currentMode = btn.dataset.mode;
    }

    loadData();
  });
});

async function loadData() {
  try {

    const url = `/api/data?benchmark=${currentBenchmark}&mode=${currentMode}`;
    const res = await fetch(url);

    if (!res.ok) throw new Error("Errore API");

    const data = await res.json();

    buildChart(data);
    buildDrawdownChart(data);
    updateStats(data);

  } catch (err) {
    console.error(err);
    document.getElementById("stats").innerHTML = "Errore caricamento dati";
  }
}

let chart;
let drawChart;

function buildChart(data) {

  const ctx = document.getElementById("chart").getContext("2d");

  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: data.datasets
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top" }
      },
      scales: {
        x: {
          ticks: {
            autoSkip: true,
            maxTicksLimit: 10,
            callback: function(value, index, ticks) {
              const label = this.getLabelForValue(value);
              const year = label.substring(0, 4);
              const prev = index > 0 ? this.getLabelForValue(ticks[index - 1].value).substring(0,4) : null;
              return year !== prev ? year : "";
            }
          }
        }
      }
    }
  });
}

function buildDrawdownChart(data) {

  const ctx = document.getElementById("drawdownChart").getContext("2d");

  if (drawChart) drawChart.destroy();

  drawChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: data.drawdown
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "top" }
      }
    }
  });
}

function updateStats(data) {
  document.getElementById("stats").innerHTML = data.stats_html;
}

// primo caricamento
loadData();
