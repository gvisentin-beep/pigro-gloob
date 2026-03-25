(function () {
  let mainChart = null;
  let ddChart = null;
  let liqChart = null;
  let currentBenchmark = "world";
  let currentMode = "normal"; // normal | leva_fissa | leva_plus

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

  function toggleModeBoxes() {
    const plusBox = document.getElementById("plus_rule_box");
    const liqCard = document.getElementById("liquidity_card");
    if (plusBox) plusBox.classList.toggle("show", currentMode === "leva_plus");
    if (liqCard) liqCard.style.display = currentMode === "leva_plus" ? "block" : "none";
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
    if (liqChart) {
      liqChart.destroy();
      liqChart = null;
    }
  }

  function yearTickIndices(labels) {
    const out = new Set();
    const seen = new Set();

    labels.forEach((label, idx) => {
      const year = String(label || "").slice(0, 4);
      if (/^\d{4}$/.test(year) && !seen.has(year)) {
        seen.add(year);
        out.add(idx);
      }
    });

    return out;
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

  function renderMain(labels, firstVals, secondVals, secondLabel, markerIndices) {
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;

    const keepTicks = yearTickIndices(labels);

    mainChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: (function() {
          const ds = [
            {
              label: "Metodo Pigro 80/15/5",
              data: firstVals
            },
            {
              label: secondLabel,
              data: secondVals
            }
          ];
          if (Array.isArray(markerIndices) && markerIndices.length) {
            ds.push({
              type: "scatter",
              label: "Integrazioni Leva+",
              data: markerIndices.map(function(i) { return ({ x: labels[i], y: secondVals[i] }); }),
              showLine: false,
              pointRadius: 5,
              pointHoverRadius: 6
            });
          }
          return ds;
        })()
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
            afterBuildTicks: function (axis) {
              axis.ticks = axis.ticks.filter(t => keepTicks.has(t.value));
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback: function (value) {
                const lbl = this.getLabelForValue(value);
                return String(lbl || "").slice(0, 4);
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

    const keepTicks = yearTickIndices(labels);

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
            afterBuildTicks: function (axis) {
              axis.ticks = axis.ticks.filter(t => keepTicks.has(t.value));
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback: function (value) {
                const lbl = this.getLabelForValue(value);
                return String(lbl || "").slice(0, 4);
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

  function renderLiquidity(labels, liquidityVals) {
    const canvas = document.getElementById("chart_liq");
    if (!canvas) return;

    const keepTicks = yearTickIndices(labels);

    liqChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Disponibilità Lombard residua",
            data: liquidityVals
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
            afterBuildTicks: function (axis) {
              axis.ticks = axis.ticks.filter(t => keepTicks.has(t.value));
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback: function (value) {
                const lbl = this.getLabelForValue(value);
                return String(lbl || "").slice(0, 4);
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

  function buildEpisodesComparisonHtml(firstEpisodes, secondEpisodes, secondLabel) {
    function line(rank, first, second) {
      const left = first
        ? `Pigro: <b>${pct(first.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(first.start)} → minimo ${formatDateIt(first.bottom)})</span>`
        : `Pigro: —`;

      const right = second
        ? `${secondLabel}: <b>${pct(second.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(second.start)} → minimo ${formatDateIt(second.bottom)})</span>`
        : `${secondLabel}: —`;

      return `<div style="margin-top:4px;"><b>${rank}ª peggiore discesa</b> — ${left} | ${right}</div>`;
    }

    return `
      <div><b>Confronto delle 2 peggiori discese complete</b></div>
      ${line(1, firstEpisodes[0], secondEpisodes[0])}
      ${line(2, firstEpisodes[1], secondEpisodes[1])}
    `;
  }

  function setActiveButtons() {
    document.querySelectorAll(".benchmarkBtn").forEach((btn) => {
      const bench = btn.getAttribute("data-benchmark");
      const mode = btn.getAttribute("data-mode");

      let active = false;

      if (currentMode === "normal" && bench) {
        active = bench === currentBenchmark;
      } else if (currentMode === "leva_fissa" && mode === "leva_fissa") {
        active = true;
      } else if (currentMode === "leva_plus" && mode === "leva_plus") {
        active = true;
      }

      btn.classList.toggle("active", active);
    });

    toggleModeBoxes();
  }

  function renderFaq() {
    document.querySelectorAll(".faqItem").forEach(function (item) {
      item.addEventListener("click", function () {
        item.classList.toggle("open");
      });
    });
  }

  function openAdvisorModal() {
    const modal = document.getElementById("advisor_modal");
    if (!modal) return;
    modal.classList.add("show");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }

  function closeAdvisorModal() {
    const modal = document.getElementById("advisor_modal");
    if (!modal) return;
    modal.classList.remove("show");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
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

  function setStrategySummary(payload, capital) {
    const labels = payload.dates || [];
    const pigroVals = payload.pigro || [];
    const strategyVals = payload.strategy || [];
    const ddPigroVals = payload.dd_pigro || [];
    const ddStrategyVals = payload.dd_strategy || [];

    const strategyLabel = payload.strategy_label || "Strategia";

    const finalPigro = pigroVals[pigroVals.length - 1];
    const finalStrategy = strategyVals[strategyVals.length - 1];
    const diff = finalStrategy - finalPigro;

    const cagrPigro = Number(payload.cagr_pigro);
    const cagrStrategy = Number(payload.cagr_strategy);
    const maxddPigro = Number(payload.maxdd_pigro);
    const maxddStrategy = Number(payload.maxdd_strategy);

    const startDate = labels[0] || "";
    const endDate = labels[labels.length - 1] || "";
    const years = startDate && endDate
      ? ((new Date(endDate) - new Date(startDate)) / (365.25 * 24 * 3600 * 1000))
      : NaN;

    const dblPigro = isFinite(cagrPigro) && cagrPigro > 0 ? 72 / cagrPigro : NaN;
    const dblStrategy = isFinite(cagrStrategy) && cagrStrategy > 0 ? 72 / cagrStrategy : NaN;
    const extraRendimento = isFinite(cagrPigro) && isFinite(cagrStrategy) ? (cagrStrategy - cagrPigro) : NaN;
    const avgLeverage = Number(payload.avg_leverage_pct);
    const maxLeverage = Number(payload.max_leverage_pct);

    setText("final_value", euro(finalStrategy, 0));
    setText("final_years", isFinite(years) ? plain(years, 1) : "—");
    setText("cagr", isFinite(cagrPigro) ? pct(cagrPigro, 2) : "—");
    setText("maxdd", isFinite(maxddPigro) ? pct(maxddPigro, 2) : "—");
    setText("dbl", isFinite(dblPigro) ? plain(dblPigro, 1) : "—");

    setText("chart_title", `Andamento negli ultimi anni — confronto Pigro vs ${strategyLabel}`);
    setText("compare_title_benchmark", strategyLabel);

    setText(
      "compare_period",
      `${euro(capital, 0)} investiti all’inizio del periodo (${startDate} → ${endDate})`
    );
    setText("compare_pigro", euro(finalPigro, 0));
    setText("compare_benchmark", euro(finalStrategy, 0));

    setHtml(
      "benchmark_summary",
      `<b>${strategyLabel}</b>: rendimento annualizzato <b>${isFinite(cagrStrategy) ? pct(cagrStrategy, 2) : "—"}</b> | max ribasso <b>${isFinite(maxddStrategy) ? pct(maxddStrategy, 2) : "—"}</b>`
    );

    if (currentMode === "leva_plus") {
      const triggerValue = Number(payload.trigger_value);
      const incrementAmount = Number(payload.increment_amount);
      const events = payload.trigger_events || [];
      let html = `<div><b>Regola Pigro Leva+</b></div>
        <div style="margin-top:4px;">Si parte con leva iniziale del 20%. La soglia di intervento è il 90% del portafoglio principale dato a garanzia: <b>${euro(triggerValue, 0)}</b>.</div>
        <div style="margin-top:4px;">Ogni nuovo passaggio al ribasso sotto tale soglia attiva un acquisto di <b>${euro(incrementAmount, 0)}</b> solo su LS80, fino a un massimo di 2 integrazioni.</div>
        <div style="margin-top:4px;">Integrazioni effettuate: <b>${events.length}</b>.</div>`;
      if (events.length) {
        html += `<div style="margin-top:8px;"><b>Date integrazione</b></div>`;
        events.forEach(function(ev) {
          html += `<div style="margin-top:4px;">${formatDateIt(ev.date)} — ${euro(ev.amount, 0)} su LS80 | disponibilità residua ${euro(ev.available_after, 0)}</div>`;
        });
      } else {
        html += `<div style="margin-top:4px;">Nessuna integrazione nel periodo.</div>`;
      }
      setHtml("dd_summary", html);
    } else {
      setHtml(
        "dd_summary",
        buildEpisodesComparisonHtml(
          payload.worst_episodes_pigro || [],
          payload.worst_episodes_strategy || [],
          strategyLabel
        )
      );
    }

    const compareBox = document.getElementById("compare_box");
    if (compareBox) {
      compareBox.innerHTML = `
        <strong>Confronto immediato</strong><br/>
        ${euro(capital, 0)} investiti all’inizio del periodo (${startDate} → ${endDate})<br/>
        Metodo Pigro → <b>${euro(finalPigro, 0)}</b><br/>
        ${strategyLabel} → <b>${euro(finalStrategy, 0)}</b><br/><br/>

        <b style="color:#1f77b4">CAGR Pigro:</b> ${isFinite(cagrPigro) ? pct(cagrPigro, 2) : "—"}<br/>
        <b style="color:#d94b64">CAGR ${strategyLabel}:</b> ${isFinite(cagrStrategy) ? pct(cagrStrategy, 2) : "—"}<br/>
        <b>Extra rendimento:</b> ${isFinite(extraRendimento) ? pct(extraRendimento, 2) : "—"}<br/><br/>

        <b>Max Ribasso Pigro:</b> ${isFinite(maxddPigro) ? pct(maxddPigro, 2) : "—"}<br/>
        <b>Max Ribasso ${strategyLabel}:</b> ${isFinite(maxddStrategy) ? pct(maxddStrategy, 2) : "—"}<br/><br/>

        <b>Anni teorici per raddoppio Pigro:</b> ${isFinite(dblPigro) ? plain(dblPigro, 1) : "—"}<br/>
        <b>Anni teorici per raddoppio ${strategyLabel}:</b> ${isFinite(dblStrategy) ? plain(dblStrategy, 1) : "—"}<br/>
        <b>Leva media utilizzata:</b> ${isFinite(avgLeverage) ? pct(avgLeverage, 2) : "—"}${isFinite(maxLeverage) ? `<br/><b>Leva massima utilizzata:</b> ${pct(maxLeverage, 2)}` : ""}${currentMode === "leva_plus" ? `<br/><b>Disponibilità Lombard iniziale:</b> ${euro(payload.initial_available, 0)}` : ""}<br/><br/>

        <b>Vantaggio/Svantaggio:</b> ${euro(diff, 0)}
      `;
    }

    renderMain(labels, pigroVals, strategyVals, strategyLabel, currentMode === "leva_plus" ? (payload.trigger_indices || []) : []);
    renderDd(labels, ddPigroVals, ddStrategyVals, strategyLabel);
    if (currentMode === "leva_plus") {
      renderLiquidity(labels, payload.liquidity_available || []);
      const liqSummary = document.getElementById("liquidity_summary");
      if (liqSummary) {
        const liq = payload.liquidity_available || [];
        const firstLiq = liq.length ? liq[0] : NaN;
        const lastLiq = liq.length ? liq[liq.length - 1] : NaN;
        liqSummary.innerHTML = `Disponibilità iniziale: <b>${euro(firstLiq, 0)}</b> | disponibilità finale: <b>${euro(lastLiq, 0)}</b>. Il ricalcolo avviene a fine anno in base al valore del solo portafoglio principale da ${euro(capital, 0)} dato a garanzia.`;
      }
    } else {
      const liqSummary = document.getElementById("liquidity_summary");
      if (liqSummary) liqSummary.innerHTML = "";
    }
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
        ${euro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})<br/>
        Metodo Pigro → <b>${euro(last, 0)}</b><br/>
        ${benchmarkLabel} → <b>${euro(lastBenchmark, 0)}</b>
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

      if (currentMode === "leva_fissa") {
        payload = await fetchJson(`/api/compute_leva?capital=${encodeURIComponent(capital)}`);
      } else if (currentMode === "leva_plus") {
        payload = await fetchJson(`/api/compute_leva_plus?capital=${encodeURIComponent(capital)}`);
      } else {
        payload = await fetchJson(
          `/api/compute?capital=${encodeURIComponent(capital)}&benchmark=${encodeURIComponent(currentBenchmark)}`
        );
      }

      if (!payload || payload.ok !== true) {
        throw new Error(payload && payload.error ? payload.error : "Risposta backend non valida");
      }

      destroyCharts();

      if (currentMode === "normal") {
        setNormalSummary(payload, capital);
      } else {
        setStrategySummary(payload, capital);
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

      const compareBox = document.getElementById("compare_box");
      if (compareBox) {
        compareBox.innerHTML = `<strong>Confronto immediato</strong><br/>Errore caricamento dati`;
      }
      const liqSummary = document.getElementById("liquidity_summary");
      if (liqSummary) liqSummary.innerHTML = "";
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
        window.open("https://www.amazon.it/dp/B0GQM925QR/ref=sr", "_blank", "noopener");
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
        openAdvisorModal();
      });
    }

    const modalClose = document.getElementById("advisor_modal_close");
    if (modalClose) {
      modalClose.addEventListener("click", closeAdvisorModal);
    }

    const modal = document.getElementById("advisor_modal");
    if (modal) {
      modal.addEventListener("click", function (e) {
        if (e.target === modal) {
          closeAdvisorModal();
        }
      });
    }

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        closeAdvisorModal();
      }
    });

    document.querySelectorAll(".benchmarkBtn[data-benchmark]").forEach((btn) => {
      btn.addEventListener("click", function () {
        currentMode = "normal";
        currentBenchmark = btn.getAttribute("data-benchmark") || "world";
        setActiveButtons();
        loadCharts();
      });
    });

    const btnLevaFissa = document.querySelector('.benchmarkBtn[data-mode="leva_fissa"]');
    if (btnLevaFissa) {
      btnLevaFissa.addEventListener("click", function () {
        currentMode = "leva_fissa";
        setActiveButtons();
        loadCharts();
      });
    }

    const btnLevaPlus = document.querySelector('.benchmarkBtn[data-mode="leva_plus"]');
    if (btnLevaPlus) {
      btnLevaPlus.addEventListener("click", function () {
        currentMode = "leva_plus";
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
    toggleModeBoxes();
    loadCharts();
  });
})();
