let chart = null;
let chartDD = null;

async function loadData() {
    const capital = document.getElementById("capital").value || 10000;

    try {
        const res = await fetch(`/api/compute?capital=${capital}`);
        const data = await res.json();

        console.log("DATA:", data);

        // 🔥 conversione FORZATA numerica + sicurezza
        const labels = data.dates || [];

        const portfolio = (data.portfolio || [])
            .map(x => Number(x))
            .filter(x => !isNaN(x));

        const world = (data.world || [])
            .map(x => Number(x))
            .filter(x => !isNaN(x));

        const dd = (data.drawdown_portfolio_pct || [])
            .map(x => Number(x))
            .filter(x => !isNaN(x));

        // 🔴 ALLINEAMENTO LUNGHEZZA (FONDAMENTALE)
        const minLen = Math.min(labels.length, portfolio.length, world.length, dd.length);

        const cleanLabels = labels.slice(-minLen);
        const cleanPortfolio = portfolio.slice(-minLen);
        const cleanWorld = world.slice(-minLen);
        const cleanDD = dd.slice(-minLen);

        drawChart(cleanLabels, cleanPortfolio, cleanWorld);
        drawDD(cleanLabels, cleanDD);

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
