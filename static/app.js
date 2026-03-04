(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const elGold = $('w_gold');
  const elGoldVal = $('w_gold_val');
  const elInitial = $('initial');
  const btnUpdate = $('btn_update');

  const elFinalValue = $('final_value');
  const elFinalYears = $('final_years');

  const m1 = $('metrics_line1');
  const m2 = $('metrics_line2');
  const m3 = $('metrics_line3');

  const btnPdf = $('btn_pdf');
  const btnFax = $('btn_faxsimile');

  // Assistente
  const askQ = $('ask_question');
  const askBtn = $('ask_btn');
  const askAns = $('ask_answer');
  const askStatus = $('ask_status');
  const askRemaining = $('ask_remaining');

  let chart;

  function fmtEuro(n) {
    try {
      return new Intl.NumberFormat('it-IT', {
        style: 'currency',
        currency: 'EUR',
        maximumFractionDigits: 0,
      }).format(n);
    } catch {
      return `${Math.round(n)} €`;
    }
  }

  function fmtPct(x) {
    if (x === null || x === undefined || Number.isNaN(x)) return '—';
    return (x * 100).toFixed(1).replace('.', ',') + '%';
  }

  function fmtNum(x, digits = 1) {
    if (x === null || x === undefined || Number.isNaN(x)) return '—';
    return Number(x).toFixed(digits).replace('.', ',');
  }

  function getGoldWeight() {
    // slider 0-50 step 5 in %
    const v = Number(elGold?.value ?? 20);
    return Math.max(0, Math.min(50, v)) / 100;
  }

  function getCapital() {
    const raw = (elInitial?.value || '')
      .toString()
      .replace(/\./g, '')
      .replace(',', '.');
    const v = Number(raw);
    if (!Number.isFinite(v) || v <= 0) return 10000;
    return v;
  }

  function setGoldLabel() {
    if (!elGoldVal || !elGold) return;
    elGoldVal.textContent = `${elGold.value}%`;
  }

  function buildChart() {
    const ctx = $('chart')?.getContext('2d');
    if (!ctx) return;

    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Portafoglio (ETF Azion-Obblig + ETC Oro)',
            data: [],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.15,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        animation: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { display: true },
          // ✅ Nessuna data neanche al passaggio del mouse
          tooltip: { enabled: false },
        },
        scales: {
          // ✅ Nascondo completamente l'asse X (date)
          x: { display: false },
          y: {
            ticks: {
              callback: (val) => {
                try {
                  return new Intl.NumberFormat('it-IT').format(val) + ' €';
                } catch {
                  return val + ' €';
                }
              },
            },
          },
        },
      },
    });
  }

  async function fetchCompute() {
    const wGold = getGoldWeight();
    const capital = getCapital();

    const url = `/api/compute?w_gold=${encodeURIComponent(
      wGold
    )}&capital=${encodeURIComponent(capital)}`;

    const res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) {
      const t = await res.text().catch(() => '');
      throw new Error(`Errore /api/compute (${res.status}): ${t}`);
    }
    return await res.json();
  }

  function updateUI(payload) {
    if (!payload || !payload.ok) return;

    const dates = payload.dates || [];
    const values = payload.portfolio || [];
    const metrics = payload.metrics || {};
    const comp = payload.composition || {};

    // ✅ Grafico: le date restano solo come labels interne, ma non vengono mai mostrate
    if (chart) {
      chart.data.labels = dates;
      chart.data.datasets[0].data = values;
      chart.update('none');
    }

    // Finale
    if (elFinalValue)
      elFinalValue.textContent = fmtEuro(
        metrics.final_portfolio ??
          (values.length ? values[values.length - 1] : 0)
      );
    if (elFinalYears) elFinalYears.textContent = fmtNum(metrics.final_years ?? null, 1);

    // Metriche
    const cagr = metrics.cagr_portfolio;
    const maxdd = metrics.max_dd_portfolio;
    const dbl = metrics.doubling_years_portfolio;

    const az = comp.azionario ?? metrics.weights?.equity;
    const ob = comp.obbligazionario ?? metrics.weights?.bond;
    const oro = comp.oro ?? metrics.weights?.gold;

    if (m1)
      m1.innerHTML = `<b>Portafoglio (ETF Azion-Obblig + ETC Oro)</b>: Rendimento annualizzato <b>${fmtPct(
        cagr
      )}</b> | Max Ribasso nel periodo <b>${fmtPct(maxdd)}</b>`;
    if (m2)
      m2.textContent = `Composizione: Azionario ${(az * 100).toFixed(
        0
      )}% | Obbligazionario ${(ob * 100).toFixed(0)}% | Oro ${(oro * 100).toFixed(0)}%`;
    if (m3)
      m3.textContent = `Raddoppio del portafoglio in anni: ${fmtNum(dbl, 1)}`;
  }

  async function refresh() {
    try {
      btnUpdate && (btnUpdate.disabled = true);
      const payload = await fetchCompute();
      updateUI(payload);
    } catch (e) {
      console.error(e);
      if (m1) m1.textContent = 'Impossibile aggiornare i dati in questo momento.';
    } finally {
      btnUpdate && (btnUpdate.disabled = false);
    }
  }

  async function ask() {
    try {
      const q = (askQ?.value || '').trim();
      if (!q) return;

      askBtn && (askBtn.disabled = true);
      askStatus && (askStatus.textContent = 'Sto pensando…');
      askAns && (askAns.textContent = '');

      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      });

      const payload = await res.json().catch(() => ({}));

      if (!res.ok || !payload.ok) {
        const msg = payload?.error || `Errore (${res.status})`;
        askStatus && (askStatus.textContent = msg);
        return;
      }

      askAns && (askAns.textContent = payload.answer || '');
      askStatus && (askStatus.textContent = '');
      if (askRemaining) {
        if (typeof payload.remaining === 'number' && typeof payload.limit === 'number') {
          askRemaining.textContent = `Domande rimanenti oggi: ${payload.remaining}/${payload.limit}`;
        } else {
          askRemaining.textContent = '';
        }
      }
    } catch (e) {
      console.error(e);
      askStatus && (askStatus.textContent = 'Errore. Riprova.');
    } finally {
      askBtn && (askBtn.disabled = false);
    }
  }

  function bind() {
    if (elGold) {
      elGold.addEventListener('input', () => setGoldLabel());
      elGold.addEventListener('change', refresh);
    }

    if (elInitial) {
      elInitial.addEventListener('change', refresh);
      elInitial.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') refresh();
      });
    }

    btnUpdate && btnUpdate.addEventListener('click', refresh);

    // PDF: stampa browser
    btnPdf && btnPdf.addEventListener('click', () => window.print());

    // Faxsimile: apre il PDF generato dal server
    btnFax &&
      btnFax.addEventListener('click', () => {
        window.open('/faxsimile_execution_only.pdf', '_blank', 'noopener,noreferrer');
      });

    askBtn && askBtn.addEventListener('click', ask);
    askQ &&
      askQ.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) ask();
      });

    // ✅ Se per caso esiste ancora un elemento data_updated, lo nascondiamo
    const du = $('data_updated');
    if (du) du.style.display = 'none';
  }

  document.addEventListener('DOMContentLoaded', () => {
    setGoldLabel();
    buildChart();
    bind();
    refresh();
  });
})();
