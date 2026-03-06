function formatEuro(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0
  });
}

function formatPctFromDecimal(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return (Number(n) * 100).toFixed(1) + "%";
}

function clampGold(v) {
  v = Number(v);
  if (Number.isNaN(v)) v = 20;
  v = Math.max(0, Math.min(50, v));
  return Math.round(v / 5) * 5;
}

function parseCapital() {
  let s = String(document.getElementById("capital").value || "").trim();
  s = s.replace(/\./g, "").replace(",", ".");
  let v = Number(s);
  if (Number.isNaN(v) || v <= 0) v = 10000;
  return v;
}

let chartMain = null;
let chartDD = null;

const elWG = document.getElementById("w_gold");
const elWGLabel = document.getElementById("w_gold_label");
const btnUpdate = document.getElementById("btn_update");
const btnPdf = document.getElementById("btn_pdf");

function updateSliderLabelAndComposition(wGold01) {
  elWGLabel.textContent = Math.round(wGold01 * 100) + "%";
  const wLs80 = 1 - wGold01;
  document.getElementById("w_equity").textContent = Math.round(0.8 * wLs80 * 100) + "%";
  document.getElementById("w_bond").textContent = Math.round(0.2 * wLs80 * 100) + "%";
  document.getElementById("w_gold2").textContent = Math.round(wGold01 * 100) + "%";
}

function renderDrawdownSummary(metrics) {
  const box = document.getElementById("dd_summary");
  if (!box) return;

  const fmtEpisode = (ep) => {
    if (!ep) return "";
    return `• ${ep.start} → ${ep.bottom} → ${ep.end} : ${Number(ep.depth_pct).toFixed(1)}%`;
  };

  const p2025 = metrics.dd_2025_portfolio;
  const w2025 = metrics.dd_2025_world;
  const epP = Array.isArray(metrics.worst_episodes_portfolio) ? metrics.worst_episodes_portfolio : [];
  const epW = Array.isArray(metrics.worst_episodes_world) ? metrics.worst_episodes_world : [];

  let html = `<b>Drawdown 2025 ("Dazi Trump")</b>: Portafoglio ${formatPctFromDecimal(p2025)} | MSCI World ${formatPctFromDecimal(w2025)}<br/>`;

  if (epP.length) {
    html += `<b>3 discese peggiori (Portafoglio)</b><br/>`;
    html += epP.map(fmtEpisode).join("<br/>") + "<br/>";
  }

  if (epW.length) {
    html += `<b>3 discese peggiori (MSCI World)</b><br/>`;
    html += epW.map(fmtEpisode).join("<br/>");
  }

  box.innerHTML = html;
}

function renderComparison(payload) {
  const period = document.getElementById("compare_period");
  const pigro = document.getElementById("compare_pigro");
  const world = document.getElementById("compare_world");

  const dates = payload.dates || [];
  const portfolio = payload.portfolio || [];
  const worldArr = payload.world || [];

  if (dates.length && period) {
    period.textContent = `10.000 € investiti dal ${dates[0]} al ${dates[dates.length - 1]}`;
  }

  if (pigro) {
    pigro.textContent = portfolio.length ? formatEuro(portfolio[portfolio.length - 1]) : "—";
  }

  if (world) {
    world.textContent = worldArr.length ? formatEuro(worldArr[worldArr.length - 1]) : "—";
  }
}

async function loadData() {
  btnUpdate.disabled = true;
  btnUpdate.textContent = "Aggiorna…";

  try {
    const wGold = clampGold(elWG.value) / 100;
    const capital = parseCapital();

    updateSliderLabelAndComposition(wGold);

    const res = await fetch(`/api/compute?w_gold=${encodeURIComponent(wGold)}&capital=${encodeURIComponent(capital)}`);
    const data = await res.json();

    if (!data.ok) {
      alert("Errore: " + (data.error || "API compute"));
      return;
    }

    const dates = data.dates || [];
    const portfolio = data.portfolio || [];
    const world = data.world || [];
    const ddP = data.drawdown_portfolio_pct || [];
    const ddW = data.drawdown_world_pct || [];
    const m = data.metrics || {};

    document.getElementById("final_value").textContent = formatEuro(m.final_portfolio);
    document.getElementById("final_years").textContent =
      m.final_years !== null && m.final_years !== undefined ? Number(m.final_years).toFixed(1).replace(".", ",") : "—";

    document.getElementById("cagr").textContent = formatPctFromDecimal(m.cagr_portfolio);
    document.getElementById("maxdd").textContent = formatPctFromDecimal(m.max_dd_portfolio);
    document.getElementById("dbl").textContent =
      m.doubling_years_portfolio !== null && m.doubling_years_portfolio !== undefined
        ? Number(m.doubling_years_portfolio).toFixed(1).replace(".", ",")
        : "—";

    renderComparison(data);
    renderDrawdownSummary(m);

    const mainDatasets = [
      {
        label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
        data: dates.map((d, i) => ({ x: d, y: portfolio[i] })),
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 0,
      }
    ];

    if (world.length) {
      mainDatasets.push({
        label: "MSCI World (SMSWLD) - normalizzato",
        data: dates.map((d, i) => ({ x: d, y: world[i] })),
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 0,
      });
    }

    const ddDatasets = [
      {
        label: "Drawdown Portafoglio (%)",
        data: dates.map((d, i) => ({ x: d, y: ddP[i] })),
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 0,
      }
    ];

    if (ddW.length) {
      ddDatasets.push({
        label: "Drawdown MSCI World (%)",
        data: dates.map((d, i) => ({ x: d, y: ddW[i] })),
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 0,
      });
    }

    if (chartMain) chartMain.destroy();
    chartMain = new Chart(document.getElementById("chart_main").getContext("2d"), {
      type: "line",
      data: { datasets: mainDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        scales: {
          x: { type: "time", time: { unit: "year" }, ticks: { maxRotation: 0, autoSkip: true } },
          y: { ticks: { callback: (v) => formatEuro(v) } }
        },
        plugins: {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${formatEuro(ctx.parsed.y)}`
            }
          }
        }
      }
    });

    if (chartDD) chartDD.destroy();
    chartDD = new Chart(document.getElementById("chart_dd").getContext("2d"), {
      type: "line",
      data: { datasets: ddDatasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        parsing: false,
        scales: {
          x: { type: "time", time: { unit: "year" }, ticks: { maxRotation: 0, autoSkip: true } },
          y: {
            suggestedMin: -60,
            suggestedMax: 0,
            ticks: { callback: (v) => `${Number(v).toFixed(0)}%` }
          }
        },
        plugins: { legend: { display: true } }
      }
    });

  } catch (e) {
    alert("Errore: " + (e?.message || e));
  } finally {
    btnUpdate.disabled = false;
    btnUpdate.textContent = "Aggiorna";
  }
}

document.querySelectorAll(".faqItem").forEach((item) => {
  item.addEventListener("click", () => item.classList.toggle("open"));
});

document.getElementById("btn_faxsimile")?.addEventListener("click", () => {
  window.open("/faxsimile_execution_only.pdf", "_blank");
});

document.getElementById("btn_consulente")?.addEventListener("click", () => {
  alert("Funzione da collegare ai contatti consulenti.");
});

document.getElementById("btn_libro")?.addEventListener("click", () => {
  window.open("https://www.amazon.it/dp/B0GQM925QR/ref=sr", "_blank");
});

document.getElementById("btn_ask")?.addEventListener("click", async () => {
  const q = (document.getElementById("ask_text")?.value || "").trim();
  if (!q) return;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Errore ask");

    const box = document.getElementById("ask_answer");
    box.style.display = "block";
    box.textContent = data.answer || "";
  } catch (e) {
    alert("Errore: " + (e?.message || e));
  }
});

elWG.addEventListener("input", () => {
  const wGold = clampGold(elWG.value);
  elWG.value = String(wGold);
  updateSliderLabelAndComposition(wGold / 100);
});

btnUpdate.addEventListener("click", loadData);

btnPdf.addEventListener("click", () => {
  const cagr = document.getElementById("cagr").textContent || "";
  const maxdd = document.getElementById("maxdd").textContent || "";
  const finalv = document.getElementById("final_value").textContent || "";
  const years = document.getElementById("final_years").textContent || "";
  const url = `/api/pdf?title=${encodeURIComponent("Gloob - Metodo Pigro")}&cagr=${encodeURIComponent(cagr)}&maxdd=${encodeURIComponent(maxdd)}&final=${encodeURIComponent(finalv)}&years=${encodeURIComponent(years)}`;
  window.open(url, "_blank");
});

updateSliderLabelAndComposition(clampGold(elWG.value) / 100);
loadData();
