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

function formatDateLabel(dateStr) {
  const d = new Date(dateStr);
  if (Number.isNaN(d.getTime())) return dateStr;
  return d.toLocaleDateString("it-IT");
}

function buildYearTickCallback(labels) {
  return function(value, index) {
    const s = labels[index];
    if (!s) return "";

    const year = s.slice(0, 4);
    const monthDay = s.slice(5, 10);

    if (index === 0) return year;

    if (monthDay <= "01-10") {
      const prev = labels[index - 1];
      if (prev && prev.slice(0, 4) !== year) {
        return year;
      }
    }

    return "";
  };
}

function yearOnlyVerticalGrid(labels) {
  return {
    drawBorder: false,
    color: function(context) {
      const i = context.index;
      if (i === undefined || i === null) return "transparent";

      const current = labels[i];
      const prev = labels[i - 1];

      if (!current) return "transparent";

      if (!prev) return "#e0e0e0";

      const yearNow = current.slice(0, 4);
      const yearPrev = prev.slice(0, 4);

      return yearNow !== yearPrev ? "#e0e0e0" : "transparent";
    }
  };
}

let chartMain = null;
let chartDD = null;

const elWG = document.getElementById("w_gold");
const elWGLabel = document.getElementById("w_gold_label");
const btnUpdate = document.getElementById("btn_update");
const btnPdf = document.getElementById("btn_pdf");

function updateSliderLabelAndComposition(wGold01) {
  if (elWGLabel) {
    elWGLabel.textContent = Math.round(wGold01 * 100) + "%";
  }

  const wLs80 = 1 - wGold01;

  const eq = document.getElementById("w_equity");
  const bd = document.getElementById("w_bond");
  const gd = document.getElementById("w_gold2");

  if (eq) eq.textContent = Math.round(0.8 * wLs80 * 100) + "%";
  if (bd) bd.textContent = Math.round(0.2 * wLs80 * 100) + "%";
  if (gd) gd.textContent = Math.round(wGold01 * 100) + "%";
}

function renderComparison(datesShown, portfolioShown, worldShown) {
  const period = document.getElementById("compare_period");
  const pigro = document.getElementById("compare_pigro");
  const world = document.getElementById("compare_world");

  if (datesShown.length && period) {
    period.textContent =
      `10.000 € investiti dal ${datesShown[0]} al ${datesShown[datesShown.length - 1]}`;
  }

  if (pigro) {
    pigro.textContent = portfolioShown.length
      ? formatEuro(portfolioShown[portfolioShown.length - 1])
      : "—";
  }

  if (world) {
    world.textContent = worldShown.length
      ? formatEuro(worldShown[worldShown.length - 1])
      : "—";
  }
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

async function loadData() {
  if (btnUpdate) {
    btnUpdate.disabled = true;
    btnUpdate.textContent = "Aggiorna…";
  }

  try {
    const wGold = clampGold(elWG ? elWG.value : 20) / 100;
    const capital = parseCapital();

    updateSliderLabelAndComposition(wGold);

    const res = await fetch(
      `/api/compute?w_gold=${encodeURIComponent(wGold)}&capital=${encodeURIComponent(capital)}`
    );

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

    if (!dates.length || !portfolio.length) {
      alert("Errore: dati grafico non disponibili.");
      return;
    }

    // Taglio serie dal 2021
    let startIndex = 0;
    for (let i = 0; i < dates.length; i++) {
      if (dates[i] >= "2021-01-01") {
        startIndex = i;
        break;
      }
    }

    const datesCut = dates.slice(startIndex);
    const portfolioCut = portfolio.slice(startIndex);
    const worldCut = world.slice(startIndex);
    const ddPCut = ddP.slice(startIndex);
    const ddWCut = ddW.slice(startIndex);

    if (!datesCut.length || !portfolioCut.length) {
      alert("Errore: serie tagliata dal 2021 non disponibile.");
      return;
    }

    const labels = datesCut;
    const xTickCallback = buildYearTickCallback(labels);

    const titleBox = document.getElementById("chart_title");
    if (titleBox) {
      titleBox.textContent = "Andamento negli ultimi anni";
    }

    const finalValueEl = document.getElementById("final_value");
    const finalYearsEl = document.getElementById("final_years");
    const cagrEl = document.getElementById("cagr");
    const maxddEl = document.getElementById("maxdd");
    const dblEl = document.getElementById("dbl");

    if (finalValueEl) {
      finalValueEl.textContent = portfolioCut.length
        ? formatEuro(portfolioCut[portfolioCut.length - 1])
        : "—";
    }

    if (finalYearsEl) {
      const yearsShown =
        labels.length >= 2
          ? ((new Date(labels[labels.length - 1]) - new Date(labels[0])) / (365.25 * 24 * 3600 * 1000))
          : null;

      finalYearsEl.textContent =
        yearsShown !== null && !Number.isNaN(yearsShown)
          ? Number(yearsShown).toFixed(1).replace(".", ",")
          : "—";
    }

    if (cagrEl) cagrEl.textContent = formatPctFromDecimal(m.cagr_portfolio);
    if (maxddEl) maxddEl.textContent = formatPctFromDecimal(m.max_dd_portfolio);
    if (dblEl) {
      dblEl.textContent =
        m.doubling_years_portfolio !== null && m.doubling_years_portfolio !== undefined
          ? Number(m.doubling_years_portfolio).toFixed(1).replace(".", ",")
          : "—";
    }

    renderComparison(labels, portfolioCut, worldCut);
    renderDrawdownSummary(m);

    const mainDatasets = [
      {
        label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
        data: portfolioCut,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 0
      }
    ];

    if (worldCut.length) {
      mainDatasets.push({
        label: "MSCI World (URTH)",
        data: worldCut,
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 0
      });
    }

    const ddDatasets = [
      {
        label: "Drawdown Portafoglio (%)",
        data: ddPCut,
        borderWidth: 2,
        tension: 0.20,
        pointRadius: 0
      }
    ];

    if (ddWCut.length) {
      ddDatasets.push({
        label: "Drawdown MSCI World (%)",
        data: ddWCut,
        borderWidth: 2,
        tension: 0.20,
        pointRadius: 0
      });
    }

    if (chartMain) chartMain.destroy();
    chartMain = new Chart(
      document.getElementById("chart_main").getContext("2d"),
      {
        type: "line",
        data: {
          labels,
          datasets: mainDatasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: "category",
              offset: false,
              ticks: {
                autoSkip: false,
                callback: xTickCallback,
                maxRotation: 0,
                minRotation: 0,
                padding: 6
              },
              grid: yearOnlyVerticalGrid(labels)
            },
            y: {
              ticks: {
                callback: (v) => formatEuro(v)
              },
              grid: {
                color: "#f0f0f0",
                drawBorder: false
              }
            }
          },
          plugins: {
            legend: { display: true },
            tooltip: {
              callbacks: {
                title: (items) => {
                  if (!items || !items.length) return "";
                  return formatDateLabel(labels[items[0].dataIndex]);
                },
                label: (ctx) => `${ctx.dataset.label}: ${formatEuro(ctx.parsed.y)}`
              }
            }
          }
        }
      }
    );

    if (chartDD) chartDD.destroy();
    chartDD = new Chart(
      document.getElementById("chart_dd").getContext("2d"),
      {
        type: "line",
        data: {
          labels,
          datasets: ddDatasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: "category",
              offset: false,
              ticks: {
                autoSkip: false,
                callback: xTickCallback,
                maxRotation: 0,
                minRotation: 0,
                padding: 6
              },
              grid: yearOnlyVerticalGrid(labels)
            },
            y: {
              suggestedMin: -60,
              suggestedMax: 0,
              ticks: {
                callback: (v) => `${Number(v).toFixed(0)}%`
              },
              grid: {
                color: "#f0f0f0",
                drawBorder: false
              }
            }
          },
          plugins: {
            legend: { display: true },
            tooltip: {
              callbacks: {
                title: (items) => {
                  if (!items || !items.length) return "";
                  return formatDateLabel(labels[items[0].dataIndex]);
                },
                label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.parsed.y).toFixed(1)}%`
              }
            }
          }
        }
      }
    );

  } catch (e) {
    alert("Errore: " + (e?.message || e));
  } finally {
    if (btnUpdate) {
      btnUpdate.disabled = false;
      btnUpdate.textContent = "Aggiorna";
    }
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
    if (box) {
      box.style.display = "block";
      box.textContent = data.answer || "";
    }
  } catch (e) {
    alert("Errore: " + (e?.message || e));
  }
});

if (elWG) {
  elWG.addEventListener("input", () => {
    const wGold = clampGold(elWG.value);
    elWG.value = String(wGold);
    updateSliderLabelAndComposition(wGold / 100);
  });
}

if (btnUpdate) {
  btnUpdate.addEventListener("click", loadData);
}

if (btnPdf) {
  btnPdf.addEventListener("click", () => {
    const cagr = document.getElementById("cagr")?.textContent || "";
    const maxdd = document.getElementById("maxdd")?.textContent || "";
    const finalv = document.getElementById("final_value")?.textContent || "";
    const years = document.getElementById("final_years")?.textContent || "";

    const url =
      `/api/pdf?title=${encodeURIComponent("Gloob - Metodo Pigro")}` +
      `&cagr=${encodeURIComponent(cagr)}` +
      `&maxdd=${encodeURIComponent(maxdd)}` +
      `&final=${encodeURIComponent(finalv)}` +
      `&years=${encodeURIComponent(years)}`;

    window.open(url, "_blank");
  });
}

updateSliderLabelAndComposition(clampGold(elWG ? elWG.value : 20) / 100);
loadData();
