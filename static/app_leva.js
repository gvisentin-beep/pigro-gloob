document.addEventListener("DOMContentLoaded", async function () {

  function euro(v) {
    return v.toLocaleString("it-IT", { style: "currency", currency: "EUR", maximumFractionDigits: 0 });
  }

  function pct(v) {
    return (v * 100).toFixed(2) + "%";
  }

  function drawChart(labels, pigro, leva) {
    new Chart(document.getElementById("chart"), {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "Pigro", data: pigro, borderWidth: 2 },
          { label: "Pigro Leva", data: leva, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true,
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: true } },
        scales: { x: { ticks: { maxTicksLimit: 10 } } }
      }
    });
  }

  function drawDrawdown(labels, pigro, leva) {
    new Chart(document.getElementById("drawdownChart"), {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          { label: "Drawdown Pigro", data: pigro },
          { label: "Drawdown Leva", data: leva }
        ]
      }
    });
  }

  try {
    const res = await fetch("/api/leva");
    const data = await res.json();

    const labels = data.labels;
    const pigro = data.pigro;
    const leva = data.leva;
    const ddPigro = data.drawdown_pigro;
    const ddLeva = data.drawdown_leva;

    drawChart(labels, pigro, leva);
    drawDrawdown(labels, ddPigro, ddLeva);

    document.getElementById("metrics").innerHTML = `
      <b>Valore finale Pigro:</b> ${euro(pigro[pigro.length-1])}<br>
      <b>Valore finale Leva:</b> ${euro(leva[leva.length-1])}
    `;

  } catch (err) {
    console.error(err);
    alert("Errore caricamento leva");
  }

});
