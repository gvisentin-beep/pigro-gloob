let chart = null;

function euro(x) {
    return new Intl.NumberFormat("it-IT", {
        style: "currency",
        currency: "EUR"
    }).format(x);
}

async function loadData() {
    const w_ls80 = Number(document.getElementById("w_ls80").value) / 100;
    const w_gold = Number(document.getElementById("w_gold").value) / 100;
    const initial = Number(document.getElementById("initial").value) || 10000;

    const url = `/api/compute?w_ls80=${w_ls80}&w_gold=${w_gold}&initial=${initial}`;

    const res = await fetch(url);
    const data = await res.json();

    document.getElementById("period").innerText =
        data.dates[0] + " → " + data.dates[data.dates.length - 1];

    const ctx = document.getElementById("chart").getContext("2d");

    if (chart) {
        chart.destroy();
    }

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: "Portafoglio (LS80+Oro)",
                    data: data.portfolio,
                    borderWidth: 2
                },
                {
                    label: "Solo LS80",
                    data: data.ls80,
                    borderWidth: 2
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
                        callback: function(value) {
                            return euro(value);
                        }
                    }
                }
            }
        }
    });
}

// Carica automaticamente all'avvio
window.onload = loadData;
