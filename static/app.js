(function () {
  let mainChart = null;
  let ddChart = null;
  let currentBenchmark = "world";
  let currentMode = "normal"; // normal | leva

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
    if (!Array.isArray(labels) || !labels.length) return [];
    const seen = new Set();

    return labels.map((label) => {
      const year = String(label || "").slice(0, 4);

      if (!/^\d{4}$/.test(year)) return "";

      if (!seen.has(year)) {
        seen.add(year);
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

  function renderMain(labels, firstVals, secondVals, secondLabel) {
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
            data: firstVals
          },
          {
            label: secondLabel,
            data: secondVals
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

  function renderDd(labels, ddFirstVals, ddSecondVals, secondLabel) {
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
            data: ddFirstVals
          },
          {
            label: `Drawdown ${secondLabel}`,
            data: ddSecondVals
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

  function buildEpisodesComparisonHtml(pigroEpisodes, secondEpisodes, secondLabel) {
    function line(rank, pigro, second) {
      const left = pigro
        ? `Pigro: <b>${pct(pigro.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(pigro.start)} → minimo ${formatDateIt(pigro.bottom)})</span>`
        : `Pigro: —`;

      const right = second
        ? `${secondLabel}: <b>${pct(second.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(second.start)} → minimo ${formatDateIt(second.bottom)})</span>`
        : `${secondLabel}: —`;

      return `<div style="margin-top:4px;"><b>${rank}ª peggiore discesa</b> — ${left} | ${right}</div>`;
    }

    return `
      <div><b>Confronto delle 2 peggiori discese complete</b></div>
      ${line(1, pigroEpisodes[0], secondEpisodes[0])}
      ${line(2, pigroEpisodes[1], secondEpisodes[1])}
    `;
  }

  function setActiveButtons() {
    document.querySelectorAll(".benchmarkBtn").forEach((btn) => {
      const key = btn.getAttribute("data-benchmark");
      const isLeva = btn.id === "btn_leva";

      if (currentMode === "leva") {
        btn.classList.toggle("active", isLeva);
      } else {
        btn.classList.toggle("active", key === currentBenchmark);
      }
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

  function setLevaSummary(payload, capital) {
    const labels = payload.dates || [];
    const pigroVals = payload.pigro || [];
    const levaVals = payload.leva || [];
    const ddPigroVals = payload.dd_pigro || payload.drawdown_pigro_pct || [];
    const ddLevaVals = payload.dd_leva || payload.drawdown_leva_pct || [];

    const finalPigro = pigroVals[pigroVals.length - 1];
    const finalLeva = levaVals[levaVals.length - 1];
    const diff = finalLeva - finalPigro;

    const cagrPigro = Number(payload.cagr_pigro);
    const cagrLeva = Number(payload.cagr_leva);
    const maxddPigro = Number(payload.maxdd_pigro);
    const maxddLeva = Number(payload.maxdd_leva);

    const startDate = labels[0] || "";
    const endDate = labels[labels.length - 1] || "";
    const years = startDate && endDate
      ? ((new Date(endDate) - new Date(startDate)) / (365.25 * 24 * 3600 * 1000))
      : NaN;

    const dblPigro = isFinite(cagrPigro) && cagrPigro > 0 ? 72 / cagrPigro : NaN;
    const dblLeva = isFinite(cagrLeva) && cagrLeva > 0 ? 72 / cagrLeva : NaN;
    const extraRendimento = isFinite(cagrPigro) && isFinite(cagrLeva) ? (cagrLeva - cagrPigro) : NaN;

    setText("final_value", euro(finalLeva, 0));
    setText("final_years", isFinite(years) ? plain(years, 1) : "—");
    setText("cagr", isFinite(cagrPigro) ? pct(cagrPigro, 2) : "—");
    setText("maxdd", isFinite(maxddPigro) ? pct(maxddPigro, 2) : "—");
    setText("dbl", isFinite(dblPigro) ? plain(dblPigro, 1) : "—");

    setText("chart_title", "Andamento negli ultimi anni — confronto Pigro vs Pigro con leva");
    setText("compare_title_benchmark", "Pigro Leva 20%");

    setText(
      "compare_period",
      `${euro(capital, 0)} investiti all’inizio del periodo (${startDate} → ${endDate})`
    );
    setText("compare_pigro", euro(finalPigro, 0));
    setText("compare_benchmark", euro(finalLeva, 0));

    setHtml(
      "benchmark_summary",
      `<b>Pigro con leva 20%</b>: rendimento annualizzato <b>${isFinite(cagrLeva) ? pct(cagrLeva, 2) : "—"}</b> | max ribasso <b>${isFinite(maxddLeva) ? pct(maxddLeva, 2) : "—"}</b>`
    );

    function computeWorstEpisodes(values, labelsArr, topN = 2) {
      if (!Array.isArray(values) || !values.length) return [];

      let peak = values[0];
      let peakIndex = 0;
      let inDd = false;
      let startIndex = 0;
      let bottomIndex = 0;
      let bottomDepth = 0;
      const episodes = [];

      for (let i = 0; i < values.length; i++) {
        const v = values[i];

        if (v >= peak) {
          if (inDd) {
            episodes.push({
              start: labelsArr[startIndex],
              bottom: labelsArr[bottomIndex],
              end: labelsArr[i],
              depth_pct: bottomDepth
            });
            inDd = false;
          }
          peak = v;
          peakIndex = i;
          continue;
        }

        const dd = ((v / peak) - 1) * 100;

        if (!inDd) {
          inDd = true;
          startIndex = peakIndex;
          bottomIndex = i;
          bottomDepth = dd;
        } else if (dd < bottomDepth) {
          bottomDepth = dd;
          bottomIndex = i;
        }
      }

      if (inDd) {
        episodes.push({
          start: labelsArr[startIndex],
          bottom: labelsArr[bottomIndex],
          end: "in corso",
          depth_pct: bottomDepth
        });
      }

      episodes.sort((a, b) => a.depth_pct - b.depth_pct);
      return episodes.slice(0, topN);
    }

    const worstPigro = computeWorstEpisodes(pigroVals, labels, 2);
    const worstLeva = computeWorstEpisodes(levaVals, labels, 2);

    setHtml(
      "dd_summary",
      `
      <div><b>Confronto delle 2 peggiori discese complete</b></div>
      <div style="margin-top:4px;"><b>1ª peggiore discesa</b> — Pigro: <b>${isFinite(maxddPigro) ? pct(maxddPigro, 2) : "—"}</b> | Leva: <b>${isFinite(maxddLeva) ? pct(maxddLeva, 2) : "—"}</b></div>
      <div style="margin-top:4px;"><b>2ª peggiore discesa</b> — Pigro: ${worstPigro[1] ? `<b>${pct(worstPigro[1].depth_pct, 2)}</b>` : "—"} | Leva: ${worstLeva[1] ? `<b>${pct(worstLeva[1].depth_pct, 2)}</b>` : "—"}</div>
      `
    );

    const compareBox = document.getElementById("compare_box");
    if (compareBox) {
      compareBox.innerHTML = `
        <strong>Confronto immediato</strong><br/>
        ${euro(capital, 0)} investiti all’inizio del periodo (${startDate} → ${endDate})<br/>
        Metodo Pigro → <b>${euro(finalPigro, 0)}</b><br/>
        Pigro Leva 20% → <b>${euro(finalLeva, 0)}</b><br/><br/>

        <b style="color:#1f77b4">CAGR Pigro:</b> ${isFinite(cagrPigro) ? pct(cagrPigro, 2) : "—"}<br/>
        <b style="color:#d94b64">CAGR Pigro Leva:</b> ${isFinite(cagrLeva) ? pct(cagrLeva, 2) : "—"}<br/>
        <b>Extra rendimento leva:</b> ${isFinite(extraRendimento) ? pct(extraRendimento, 2) : "—"}<br/><br/>

        <b>Max Ribasso Pigro:</b> ${isFinite(maxddPigro) ? pct(maxddPigro, 2) : "—"}<br/>
        <b>Max Ribasso Pigro Leva:</b> ${isFinite(maxddLeva) ? pct(maxddLeva, 2) : "—"}<br/><br/>

        <b>Anni teorici per raddoppio Pigro:</b> ${isFinite(dblPigro) ? plain(dblPigro, 1) : "—"}<br/>
        <b>Anni teorici per raddoppio Leva:</b> ${isFinite(dblLeva) ? plain(dblLeva, 1) : "—"}<br/><br/>

        <b>Vantaggio/Svantaggio leva:</b> ${euro(diff, 0)}
      `;
    }

    renderMain(labels, pigroVals, levaVals, "Pigro Leva 20%");
    renderDd(labels, ddPigroVals, ddLevaVals, "Leva");
  }

  function setNormalSummary(payload, capital) {
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

    const compareBox = document.getElementById("compare_box");
    if (compareBox) {
      compareBox.innerHTML = `
        <strong>Confronto immediato</strong><br/>
        <span id="compare_period">${euro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})</span><br/>
        Metodo Pigro → <b id="compare_pigro">${euro(last, 0)}</b><br/>
        <span id="compare_title_benchmark">${benchmarkLabel}</span> → <b id="compare_benchmark">${euro(lastBenchmark, 0)}</b>
      `;
    }

    renderMain(labels, pigroVals, benchmarkVals, benchmarkLabel);
    renderDd(labels, ddPigroVals, ddBenchmarkVals, benchmarkLabel);
  }

  async function loadCharts() {
    normalizeCapitalInput();
    const capital = getCapital();

    try {
      let payload;

      if (currentMode === "leva") {
        payload = await fetchJson(`/api/compute_leva?capital=${encodeURIComponent(capital)}`);
      } else {
        payload = await fetchJson(
          `/api/compute?capital=${encodeURIComponent(capital)}&benchmark=${encodeURIComponent(currentBenchmark)}`
        );
      }

      if (!payload || payload.ok !== true) {
        throw new Error(
          payload && payload.error ? payload.error : "Risposta backend non valida"
        );
      }

      destroyCharts();

      if (currentMode === "leva") {
        setLevaSummary(payload, capital);
      } else {
        setNormalSummary(payload, capital);
      }
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

    document.querySelectorAll(".benchmarkBtn[data-benchmark]").forEach((btn) => {
      btn.addEventListener("click", function () {
        currentMode = "normal";
        currentBenchmark = btn.getAttribute("data-benchmark") || "world";
        setActiveButtons();
        loadCharts();
      });
    });

    const btnLeva = document.getElementById("btn_leva");
    if (btnLeva) {
      btnLeva.addEventListener("click", function () {
        currentMode = "leva";
        setActiveButtons();
        loadCharts();
      });
    }

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
    setActiveButtons();
    loadCharts();
  });
})();
