let chart = null;
let chartDD = null;

async function loadData() {
    const capital = document.getElementById("capital").value || 10000;

    try {
        const res = await fetch(`/api/compute?capital=${capital}`);
        const data = await res.json();

        console.log("DATI RICEVUTI:", data);

        const labels = data.dates;

        // 🔥 conversione FORZATA numerica
        const portfolio = data.portfolio.map(x => Number(x));
        const world = data.world.map(x => Number(x));
        const dd = data.drawdown_portfolio_pct.map(x => Number(x));

        drawChart(labels, portfolio, world);
        drawDD(labels, dd);

    } catch (err) {
        console.error("Errore:", err);
        alert("Errore caricamento dati");
    }
}

function drawChart(labels, portfolio, world) {

    const ctx = document.getElementById("chart_main").getContext("2d");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Pigro",
                    data: portfolio,
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: "MSCI World",
                    data: world,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            animation: false
        }
    });
}

function drawDD(labels, dd) {

    const ctx = document.getElementById("chart_dd").getContext("2d");

    if (chartDD) chartDD.destroy();

    chartDD = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Drawdown",
                    data: dd,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            animation: false
        }
    });
}

document.addEventListener("DOMContentLoaded", () => {
    loadData();

    document.getElementById("btn_update")
        ?.addEventListener("click", loadData);
});
