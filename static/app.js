(function () {
  let mainChart = null;
  let ddChart = null;
  let currentBenchmark = "world";

  const BENCHMARK_LABELS = {
    world: "MSCI World",
    mib: "iShares FTSE MIB UCITS ETF",
    sp500: "iShares Core S&P 500 UCITS ETF (Acc)"
  };

  function euro(value, digits = 0) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString("it-IT", {
      style: "currency",
      currency: "EUR",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function pct(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString("it-IT", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    }) + "%";
  }

  function plain(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString("it-IT", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function formatIntegerInput(value) {
    const digits = String(value || "").replace(/\D/g, "");
    if (!digits) return "";
    return Number(digits).toLocaleString("it-IT", {
      maximumFractionDigits: 0
    });
  }

  function normalizeCapitalInput() {
    const el = document.getElementById("capital");
    if (!el) return;
    el.value = formatIntegerInput(el.value || "10000");
  }

  function getCapital() {
    const el = document.getElementById("capital");
    if (!el) return 10000;

    const rawDigits = String(el.value || "").replace(/\D/g, "");
    const n = Number(rawDigits);
    return isFinite(n) && n > 0 ? n : 10000;
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function setHtml(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = value;
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  function destroyCharts() {
    if (mainChart) {
      mainChart.destroy();
      mainChart = null;
    }
    if (ddChart) {
      ddChart.destroy();
      ddChart = null;
    }
  }

  function buildYearTicks(labels) {
    if (!Array.isArray(labels) || !labels.length) return labels;
    let lastYear = null;
    return labels.map((label) => {
      const txt = String(label || "");
      const year = txt.slice(0, 4);
      if (/^\d{4}$/.test(year) && year !== lastYear) {
        lastYear = year;
        return year;
      }
      return "";
    });
  }

  function formatDateIt(isoDate) {
    if (!isoDate || typeof isoDate !== "string") return "—";
    const parts = isoDate.split("-");
    if (parts.length !== 3) return isoDate;
    return `${parts[2]}-${parts[1]}-${parts[0]}`;
  }

  function commonChartOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false
      },
      elements: {
        line: {
          tension: 0.14,
          borderWidth: 2
        },
        point: {
          radius: 0,
          hoverRadius: 3
        }
      },
      plugins: {
        legend: {
          display: true,
          labels: {
            usePointStyle: true,
            boxWidth: 10,
            padding: 16,
            font: {
              size: 12,
              weight: "600"
            }
          }
        }
      }
    };
  }

  function renderMain(labels, pigroVals, benchmarkVals, benchmarkLabel) {
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;

    const xTickLabels = buildYearTicks(labels);

    mainChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Metodo Pigro 80/15/5",
            data: pigroVals
          },
          {
            label: benchmarkLabel,
            data: benchmarkVals
          }
        ]
      },
      options: {
        ...commonChartOptions(),
        plugins: {
          ...commonChartOptions().plugins,
          tooltip: {
            callbacks: {
              title: function (items) {
                return items && items.length ? items[0].label : "";
              },
              label: function (ctx) {
                return `${ctx.dataset.label}: ${euro(ctx.parsed.y, 0)}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback: function (value, index) {
                return xTickLabels[index] || "";
              }
            }
          },
          y: {
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: {
              callback: function (value) {
                return euro(value, 0);
              }
            }
          }
        }
      }
    });
  }

  function renderDd(labels, ddPigroVals, ddBenchmarkVals, benchmarkLabel) {
    const canvas = document.getElementById("chart_dd");
    if (!canvas) return;

    const xTickLabels = buildYearTicks(labels);

    ddChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Drawdown Portafoglio Pigro",
            data: ddPigroVals
          },
          {
            label: `Drawdown ${benchmarkLabel}`,
            data: ddBenchmarkVals
          }
        ]
      },
      options: {
        ...commonChartOptions(),
        plugins: {
          ...commonChartOptions().plugins,
          tooltip: {
            callbacks: {
              title: function (items) {
                return items && items.length ? items[0].label : "";
              },
              label: function (ctx) {
                return `${ctx.dataset.label}: ${pct(ctx.parsed.y, 2)}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback: function (value, index) {
                return xTickLabels[index] || "";
              }
            }
          },
          y: {
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: {
              callback: function (value) {
                return pct(value, 0);
              }
            }
          }
        }
      }
    });
  }

  function buildEpisodesComparisonHtml(pigroEpisodes, benchmarkEpisodes, benchmarkLabel) {
    function line(rank, pigro, benchmark) {
      const left = pigro
        ? `Pigro: <b>${pct(pigro.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(pigro.start)} → minimo ${formatDateIt(pigro.bottom)})</span>`
        : `Pigro: —`;

      const right = benchmark
        ? `${benchmarkLabel}: <b>${pct(benchmark.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(benchmark.start)} → minimo ${formatDateIt(benchmark.bottom)})</span>`
        : `${benchmarkLabel}: —`;

      return `<div style="margin-top:4px;"><b>${rank}ª peggiore discesa</b> — ${left} | ${right}</div>`;
    }

    return `
      <div><b>Confronto delle 2 peggiori discese complete</b></div>
      ${line(1, pigroEpisodes[0], benchmarkEpisodes[0])}
      ${line(2, pigroEpisodes[1], benchmarkEpisodes[1])}
    `;
  }

  function setActiveBenchmarkButton() {
    document.querySelectorAll(".benchmarkBtn").forEach((btn) => {
      const key = btn.getAttribute("data-benchmark");
      btn.classList.toggle("active", key === currentBenchmark);
    });
  }

  function renderFaq() {
    document.querySelectorAll(".faqItem").forEach(function (item) {
      item.addEventListener("click", function () {
        item.classList.toggle("open");
      });
    });
  }

  async function askAssistant() {
    const box = document.getElementById("ask_text");
    const out = document.getElementById("ask_answer");
    if (!box || !out) return;

    const question = String(box.value || "").trim();
    if (!question) return;

    out.style.display = "block";
    out.textContent = "Attendi…";

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question: question })
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const payload = await res.json();
      out.textContent =
        payload.answer ||
        payload.response ||
        "Nessuna risposta disponibile.";
    } catch (err) {
      console.error("Errore assistente:", err);
      out.textContent =
        "Non sono riuscito a contattare l’assistente. Riprova tra poco.";
    }
  }

  async function loadCharts() {
    normalizeCapitalInput();
    const capital = getCapital();

    try {
      const payload = await fetchJson(
        `/api/compute?capital=${encodeURIComponent(capital)}&benchmark=${encodeURIComponent(currentBenchmark)}`
      );

      if (!payload || payload.ok !== true) {
        throw new Error(
          payload && payload.error ? payload.error : "Risposta backend non valida"
        );
      }

      const labels = payload.dates || [];
      const pigroVals = payload.portfolio || [];
      const benchmarkVals = payload.benchmark || [];
      const ddPigroVals = payload.drawdown_portfolio_pct || [];
      const ddBenchmarkVals = payload.drawdown_benchmark_pct || [];
      const metrics = payload.metrics || {};
      const benchmarkLabel = payload.benchmark_label || BENCHMARK_LABELS[currentBenchmark];

      if (!labels.length || !pigroVals.length || !benchmarkVals.length) {
        throw new Error("Dataset vuoto");
      }

      destroyCharts();
      renderMain(labels, pigroVals, benchmarkVals, benchmarkLabel);
      renderDd(labels, ddPigroVals, ddBenchmarkVals, benchmarkLabel);

      const last = pigroVals[pigroVals.length - 1];
      const lastBenchmark = benchmarkVals[benchmarkVals.length - 1];
      const firstDate = labels[0] || "inizio periodo";
      const lastDate = labels[labels.length - 1] || "";

      const years = Number(metrics.final_years);
      const cagr = Number(metrics.cagr_portfolio) * 100;
      const maxdd = Number(metrics.max_dd_portfolio) * 100;
      const dbl = Number(metrics.doubling_years_portfolio);

      const cagrBench = Number(metrics.cagr_benchmark) * 100;
      const maxddBench = Number(metrics.max_dd_benchmark) * 100;

      setText("final_value", euro(last, 0));
      setText("final_years", isFinite(years) ? plain(years, 1) : "—");
      setText("cagr", isFinite(cagr) ? pct(cagr, 2) : "—");
      setText("maxdd", isFinite(maxdd) ? pct(maxdd, 2) : "—");
      setText("dbl", isFinite(dbl) ? plain(dbl, 1) : "—");

      setText("chart_title", `Andamento negli ultimi anni — confronto con ${benchmarkLabel}`);
      setText("compare_title_benchmark", benchmarkLabel);

      setText(
        "compare_period",
        `${euro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})`
      );
      setText("compare_pigro", euro(last, 0));
      setText("compare_benchmark", euro(lastBenchmark, 0));

      setHtml(
        "dd_summary",
        buildEpisodesComparisonHtml(
          metrics.worst_episodes_portfolio || [],
          metrics.worst_episodes_benchmark || [],
          benchmarkLabel
        )
      );

      setHtml(
        "benchmark_summary",
        `<b>${benchmarkLabel}</b>: rendimento annualizzato <b>${isFinite(cagrBench) ? pct(cagrBench, 2) : "—"}</b> | max ribasso <b>${isFinite(maxddBench) ? pct(maxddBench, 2) : "—"}</b>`
      );
    } catch (err) {
      console.error("Errore caricamento grafici:", err);
      destroyCharts();

      setText("final_value", "Errore");
      setText("final_years", "—");
      setText("cagr", "—");
      setText("maxdd", "—");
      setText("dbl", "—");
      setText("compare_pigro", "—");
      setText("compare_benchmark", "—");
      setText("compare_title_benchmark", "Benchmark");
      setText("dd_summary", "Impossibile caricare i dati del grafico.");
      setText("benchmark_summary", "Impossibile calcolare il benchmark.");

      alert("Impossibile caricare il grafico. Controlla /api/compute e i CSV.");
    }
  }

  function wireButtons() {
    const btnUpdate = document.getElementById("btn_update");
    if (btnUpdate) btnUpdate.addEventListener("click", loadCharts);

    const btnPdf = document.getElementById("btn_pdf");
    if (btnPdf) {
      btnPdf.addEventListener("click", function () {
        window.print();
      });
    }

    const btnAsk = document.getElementById("btn_ask");
    if (btnAsk) btnAsk.addEventListener("click", askAssistant);

    const btnLibro = document.getElementById("btn_libro");
    if (btnLibro) {
      btnLibro.addEventListener("click", function () {
        window.open(
          "https://www.amazon.it/dp/B0GQM925QR/ref=sr",
          "_blank",
          "noopener"
        );
      });
    }

    const btnFax = document.getElementById("btn_faxsimile");
    if (btnFax) {
      btnFax.addEventListener("click", function () {
        window.location.href = "/static/faxsimile_execution_only.pdf";
      });
    }

    const btnCons = document.getElementById("btn_consulente");
    if (btnCons) {
      btnCons.addEventListener("click", function () {
        alert("Qui puoi collegare la finestra popup o la pagina con i consulenti OCF.");
      });
    }

    document.querySelectorAll(".benchmarkBtn").forEach((btn) => {
      btn.addEventListener("click", function () {
        currentBenchmark = btn.getAttribute("data-benchmark") || "world";
        setActiveBenchmarkButton();
        loadCharts();
      });
    });

    const capital = document.getElementById("capital");
    if (capital) {
      capital.addEventListener("input", function () {
        const cursorAtEnd = capital.selectionStart === capital.value.length;
        capital.value = formatIntegerInput(capital.value);
        if (cursorAtEnd) {
          capital.setSelectionRange(capital.value.length, capital.value.length);
        }
      });

      capital.addEventListener("blur", function () {
        normalizeCapitalInput();
      });

      capital.addEventListener("keydown", function (e) {
        if (e.key === "Enter") loadCharts();
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    normalizeCapitalInput();
    renderFaq();
    wireButtons();
    setActiveBenchmarkButton();
    loadCharts();
  });
})();
