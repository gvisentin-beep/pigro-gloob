async function loadData() {
    const res = await fetch('/api/compute_leva');
    const data = await res.json();

    const ctx = document.getElementById('chart').getContext('2d');
    const ctx2 = document.getElementById('drawdownChart').getContext('2d');

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Pigro',
                    data: data.pigro,
                    borderWidth: 2
                },
                {
                    label: 'Pigro Leva',
                    data: data.leva,
                    borderWidth: 2
                }
            ]
        }
    });

    new Chart(ctx2, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Drawdown Pigro',
                    data: data.dd_pigro,
                    borderWidth: 2
                },
                {
                    label: 'Drawdown Leva',
                    data: data.dd_leva,
                    borderWidth: 2
                }
            ]
        }
    });

    document.getElementById('metrics').innerHTML = `
        <h3>Metriche</h3>
        <p>CAGR Pigro: ${data.cagr_pigro}%</p>
        <p>CAGR Leva: ${data.cagr_leva}%</p>
        <p>Max Drawdown Pigro: ${data.maxdd_pigro}%</p>
        <p>Max Drawdown Leva: ${data.maxdd_leva}%</p>
    `;
}

loadData();
