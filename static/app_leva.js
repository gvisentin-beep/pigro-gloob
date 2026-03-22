async function loadLeva() {
    try {
        const res = await fetch('/api/compute_leva');
        const data = await res.json();

        if (!data.ok) {
            alert("Errore: " + data.error);
            return;
        }

        const dates = data.dates;
        const pigro = data.pigro;
        const leva = data.leva;
        const dd_pigro = data.dd_pigro;
        const dd_leva = data.dd_leva;

        // ===== GRAFICO VALORE =====
        const ctx = document.getElementById('chartLeva').getContext('2d');

        if (window.chartLeva) window.chartLeva.destroy();

        window.chartLeva = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Pigro',
                        data: pigro,
                        borderWidth: 2,
                        tension: 0.1
                    },
                    {
                        label: 'Pigro Leva',
                        data: leva,
                        borderWidth: 2,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { position: 'top' }
                },
                scales: {
                    x: {
                        ticks: {
                            maxTicksLimit: 10
                        }
                    }
                }
            }
        });

        // ===== GRAFICO DRAWDOWN =====
        const ctx2 = document.getElementById('chartDrawdownLeva').getContext('2d');

        if (window.chartDD) window.chartDD.destroy();

        window.chartDD = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [
                    {
                        label: 'Drawdown Pigro',
                        data: dd_pigro,
                        borderWidth: 1.5
                    },
                    {
                        label: 'Drawdown Leva',
                        data: dd_leva,
                        borderWidth: 1.5
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        ticks: {
                            callback: v => v + '%'
                        }
                    }
                }
            }
        });

        // ===== RISULTATI =====
        const finalPigro = pigro[pigro.length - 1];
        const finalLeva = leva[leva.length - 1];

        document.getElementById("leva_results").innerHTML = `
            <b>Confronto finale:</b><br>
            Pigro: € ${finalPigro.toLocaleString()}<br>
            Leva: € ${finalLeva.toLocaleString()}<br><br>

            CAGR Pigro: ${data.cagr_pigro}%<br>
            CAGR Leva: ${data.cagr_leva}%<br><br>

            Max Ribasso Pigro: ${data.maxdd_pigro}%<br>
            Max Ribasso Leva: ${data.maxdd_leva}%<br>
        `;

    } catch (err) {
        console.error(err);
        alert("Errore caricamento leva");
    }
}

window.onload = loadLeva;
