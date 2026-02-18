let chart = null;

function pct(x) {
    return (x * 100).toFixed(1) + "%";
}

function eur(x) {
    return new Intl.NumberFormat("it-IT", {
        style: "currency",
        currency: "EUR",
        maximumFractionDigits: 0
    }).format(x);
}

async function loadData() {
    const w_ls80 = parseFloat(document.getElementById("w_ls80").value) / 100;
    const w_gold = parseFloat(document.getElementById("w_gold").value) / 100;
    const initial = parseFloat(document.getElementById("initial").value);

    const url = `/api/compute?w_ls80=${w_ls80}&w_gold=${w_gold}&initial=${initial}`;

    const res = await fetch(url);
    const data = await res.json();

    if (data.error) {
        alert(data.error);
        return;
    }

    // Serie
    const dates = data.series.map(d => d.date);
    const port = data.series.map(d => d.eur_port);
    const ls80 = data.series.map(d => d.eur_ls80);

    // Stats
    document.getElementById("cagr").textContent = pct(data.stats.cagr);
    document.getElementById("mdd").textContent = pct(data.stats.max_drawdown);
    document.getElementById("cagr_ls").textContent = pct(data.stats.cagr_ls80);
    document.getElementById("mdd_ls").textContent = pct(data.stats.max_drawdown_ls80);
    document.getElementById("period").textContent =
        data.stats.start + " → " + data.stats.end;

    const ctx = document.getElementById("chart").getContext("2d");

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: dates,
            datasets: [
                {
                    label: "Portafoglio (LS80+Oro)",
                    data: port,
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: "Solo LS80",
                    data: ls80,
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: "index",
                intersect: false
            },
            scales: {
                y: {
                    ticks: {
                        callback: value => eur(value)
                    }
                }
            }
        }
    });
}

// Caricamento automatico al primo avvio
window.onload = loadData;
