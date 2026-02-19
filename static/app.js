let chart = null;

function euro(x) {
  return new Intl.NumberFormat("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(Number(x));
}

function pct(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "—";
  return (n * 100).toFixed(1) + "%";
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function snapToStep(v, step) {
  return Math.round(v / step) * step;
}

// Calcola composizione a 3 componenti partendo da Oro (0..20 step 5)
// Ritorna: gold, ls80, equity, bonds (tutti in %)
function computeCompositionFromGoldInput() {
  let gold = Number(document.getElementById("w_gold")?.value);
  if (!Number.isFinite(gold)) gold = 10;

  gold = clamp(gold, 0, 20);
  gold = snapToStep(gold, 5);

  // riscrivo il valore (così l’utente vede lo snap)
  const goldEl = document.getElementById("w_gold");
  if (goldEl) goldEl.value = gold;

  const ls80 = 100 - gold;
  const equity = ls80 * 0.80;
  const bonds = ls80 * 0.20;

  return { gold, ls80, equity, bonds };
}

function showCompositionInline(comp) {
  const fmt = (x) => `${x.toFixed(0)}%`;

  const ge = document.getElementById("w_equity");
  if (ge) ge.innerText = fmt(comp.equity);

  const gb = document.getElementById("w_bonds");
  if (gb) gb.innerText = fmt(comp.bonds);

  const gg = document.getElementById("w_gold_show");
  if (gg) gg.innerText = fmt(comp.gold);
}

function findMetricsCard() {
  return Array.from(document.querySelectorAll(".card")).find((el) =>
    el.innerText.includes("Portafoglio (LS80+Oro)")
  );
}

async function loadData() {
  // 1) composizione vincolata (oro max 20, step 5)
  const comp = computeCompositionFromGoldInput();
  showCompositionInline(comp);

  // 2) parametri per API (strumenti reali: LS80 + Oro)
  const w_ls80 = comp.ls80; // in %
  const w_gold = comp.gold; // in %
  const capital = Number(document.getElementById("initial")?.value) || 10000;

  // 3) chiamata API con anti-cache
  const url =
    `/api/compute` +
    `?w_ls80=${encodeURIComponent(w_ls80)}` +
    `&w_gold=${encodeURIComponent(w_gold)}` +
    `&capital=${encodeURIComponent(capital)}` +
    `&t=${Date.now()}`;

  let data;
  try {
    const res = await fetch(url, { cache: "no-store" });
    data = await res.json();
  } catch (err) {
    console.error(err);
    alert("Errore di rete nel caricamento dati.");
    return;
  }

  if (data?.error) {
    console.error(data.error);
    alert(data.error);
    return;
  }

  // 4) periodo (se esiste span/id)
  if (Array.isArray(data.dates) && data.dates.length > 1) {
    const p = document.getElementById("period");
    if (p) p.innerText = `${data.dates[0]} → ${data.dates[data.dates.length - 1]}`;
  }

  // 5) metriche + composizione nel testo (didattica)
  if (data.metrics && Array.isArray(data.dates) && data.dates.length > 1) {
    const m = data.metrics;

    const riga1 =
      `Portafoglio (LS80+Oro): CAGR ${pct(m.cagr_portfolio)} | Max DD ${pct(m.max_dd_portfolio)}`;

    const riga2 =
      `Solo LS80: CAGR ${pct(m.cagr_solo)} | Max DD ${pct(m.max_dd_solo)}`;

    const riga3 =
      `Composizione: Azionario ${comp.equity.toFixed(0)}% | Obbligazionario ${comp.bonds.toFixed(0)}% | Oro ${comp.gold.toFixed(0)}%`;

    const riga4 =
      `Periodo: ${data.dates[0]} → ${data.dates[data.dates.length - 1]}`;

    const card = findMetricsCard();
    if (card) {
      card.innerHTML = `<b>${riga1}</b><br>${riga2}<br><b>${riga3}</b><br>${riga4}`;
    }
  }

  // 6) grafico
  const canvas = document.getElementById("chart");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Portafoglio (LS80+Oro)",
          data: data.portfolio,
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        },
        {
          label: "Solo LS80",
          data: data.solo_ls80,
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      plugins: {
        tooltip: {
          callbacks: {
            label: (c) => `${c.dataset.label}: ${euro(c.parsed.y)}`,
          },
        },
      },
      scales: {
        y: {
          ticks: { callback: (v) => euro(v) },
        },
      },
    },
  });
}

function init() {
  // primo caricamento
  loadData();

  // Bottone Aggiorna
  const btn = document.querySelector("button");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  // Enter nei campi
  ["w_gold", "initial"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        loadData();
      }
    });
  });

  // Se l’utente cambia Oro e sposta il focus, aggiorno composizione visiva subito
  const goldEl = document.getElementById("w_gold");
  if (goldEl) {
    goldEl.addEventListener("change", () => {
      const comp = computeCompositionFromGoldInput();
      showCompositionInline(comp);
    });
  }
}

window.addEventListener("DOMContentLoaded", init);
