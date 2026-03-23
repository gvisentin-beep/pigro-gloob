(function () {
  let mainChart = null;
  let ddChart = null;
  let currentBenchmark = "world";
  let currentMode = "pigro";

  const BENCHMARK_LABELS = {
    world: "MSCI World",
    mib: "iShares FTSE MIB UCITS ETF",
    sp500: "iShares Core S&P 500 UCITS ETF (Acc)"
  };

  const MODE_NOTES = {
    pigro: `<b>Messaggio chiave:</b><br/>
      La differenza non è indovinare il mercato.<br/>
      È avere una struttura semplice e mantenerla nel tempo.`,
    leva20: `<b>Pigro con leva 20%</b><br/>
      Parte con 100.000 € di capitale proprio e 20.000 € di credito Lombard.
      Il portafoglio viene ribilanciato a fine anno e il costo del Lombard è ipotizzato al 2,5% annuo.`,
    levaPlus: `<b>Pigro Leva+</b><br/>
      Parte con 100.000 € + 20.000 € di Lombard. Se il valore lordo del portafoglio scende sotto 108.000 €,
      si investono 13.000 € aggiuntivi solo su LS80. Il segnale può attivarsi al massimo 3 volte,
      solo se la soglia viene nuovamente attraversata al ribasso. Ribilanciamento totale a fine anno.`
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
    el.value = formatIntegerInput(el.value || "100000");
  }

  function getCapital() {
    const el = document.getElementById("capital");
    if (!el) return 100000;
    const rawDigits = String(el.value || "").replace(/\D/g, "");
    const n = Number(rawDigits);
    return isFinite(n) && n > 0 ? n : 100000;
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

  function renderMain(labels, leftVals, rightVals, rightLabel, leftLabel) {
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;

    const xTickLabels = buildYearTicks(labels);

    mainChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: leftLabel,
            data: leftVals
          },
          {
            label: rightLabel,
            data: rightVals
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

  function renderDd(labels, leftVals, rightVals, rightLabel, leftLabel) {
    const canvas = document.getElementById("chart_dd");
    if (!canvas) return;

    const xTickLabels = buildYearTicks(labels);

    ddChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: `Drawdown ${leftLabel}`,
            data: leftVals
          },
          {
            label: `Drawdown ${rightLabel}`,
            data: rightVals
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

  function buildEpisodesComparisonHtml(leftEpisodes, rightEpisodes, rightLabel, leftLabel) {
    function line(rank, left, right) {
      const l = left
        ? `${leftLabel}: <b>${pct(left.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(left.start)} → minimo ${formatDateIt(left.bottom)})</span>`
        : `${leftLabel}: —`;

      const r = right
        ? `${rightLabel}: <b>${pct(right.depth_pct, 2)}</b> <span style="opacity:.9;">(${formatDateIt(right.start)} → minimo ${formatDateIt(right.bottom)})</span>`
        : `${rightLabel}: —`;

      return `<div style="margin-top:4px;"><b>${rank}ª peggiore discesa</b> — ${l} | ${r}</div>`;
    }

    return `
      <div><b>Confronto delle 2 peggiori discese complete</b></div>
      ${line(1, leftEpisodes[0], rightEpisodes[0])}
      ${line(2, leftEpisodes[1], rightEpisodes[1])}
    `;
  }

  function setActiveBenchmarkButton() {
    document.querySelectorAll(".benchmarkBtn").forEach((btn) => {
      const key = btn.getAttribute("data-benchmark");
      btn.classList.toggle("active", key === currentBenchmark);
    });
  }

  function setActiveModeButton() {
    document.querySelectorAll(".modeBtn").forEach((btn) => {
      const key = btn.getAttribute("data-mode");
      btn.classList.toggle("active", key === currentMode);
    });
  }

  function updateModeUi() {
    const benchControls = document.getElementById("benchmark_controls");
    const benchHint = document.getElementById("benchmark_hint");
    const strategyNote = document.getElementById("strategy_note");

    if (strategyNote) strategyNote.innerHTML = MODE_NOTES[currentMode] || "";

    if (currentMode === "pigro") {
      if (benchControls) benchControls.style.display = "flex";
      if (benchHint) benchHint.style.display = "block";
    } else {
      if (benchControls) benchControls.style.display = "none";
      if (benchHint) benchHint.style.display = "none";
    }
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

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const payload = await res.json();
      out.textContent = payload.answer || payload.response || "Nessuna risposta disponibile.";
    } catch (err) {
      console.error("Errore assistente:", err);
      out.textContent = "Non sono riuscito a contattare l’assistente. Riprova tra poco.";
    }
  }

  async function loadPigro(capital) {
    const payload = await fetchJson(
      `/api/compute?capital=${encodeURIComponent(capital)}&benchmark=${encodeURIComponent(currentBenchmark)}`
    );

    if (!payload || payload.ok !== true) {
      throw new Error(payload && payload.error ? payload.error : "Risposta backend non valida");
    }

    const labels = payload.dates || [];
    const leftVals = payload.portfolio || [];
    const rightVals = payload.benchmark || [];
    const ddLeftVals = payload.drawdown_portfolio_pct || [];
    const ddRightVals = payload.drawdown_benchmark_pct || [];
    const metrics = payload.metrics || {};
    const rightLabel = payload.benchmark_label || BENCHMARK_LABELS[currentBenchmark];
    const leftLabel = "Metodo Pigro 80/15/5";

    destroyCharts();
    renderMain(labels, leftVals, rightVals, rightLabel, leftLabel);
    renderDd(labels, ddLeftVals, ddRightVals, rightLabel, leftLabel);

    const last = leftVals[leftVals.length - 1];
    const lastRight = rightVals[rightVals.length - 1];
    const firstDate = labels[0] || "inizio periodo";
    const lastDate = labels[labels.length - 1] || "";

    const years = Number(metrics.final_years);
    const cagr = Number(metrics.cagr_portfolio) * 100;
    const maxdd = Number(metrics.max_dd_portfolio) * 100;
    const dbl = Number(metrics.doubling_years_portfolio);

    const cagrBench = Number(metrics.cagr_benchmark) * 100;
    const maxddBench = Number(metrics.max_dd_benchmark) * 100;

    setText("summary_title", "Portafoglio “Pigro 80/15/5”:");
    setText("final_value", euro(last, 0));
    setText("final_years", isFinite(years) ? plain(years, 1) : "—");
    setText("cagr", isFinite(cagr) ? pct(cagr, 2) : "—");
    setText("maxdd", isFinite(maxdd) ? pct(maxdd, 2) : "—");
    setText("dbl_label", "Raddoppio teorico:");
    setText("dbl", isFinite(dbl) ? `${plain(dbl, 1)} anni` : "—");

    setText("chart_title", `Andamento negli ultimi anni — confronto con ${rightLabel}`);
    setText("compare_title_benchmark", rightLabel);
    setText("compare_period", `${euro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})`);
    setText("compare_pigro", euro(last, 0));
    setText("compare_benchmark", euro(lastRight, 0));

    setHtml(
      "dd_summary",
      buildEpisodesComparisonHtml(
        metrics.worst_episodes_portfolio || [],
        metrics.worst_episodes_benchmark || [],
        rightLabel,
        leftLabel
      )
    );

    setHtml(
      "benchmark_summary",
      `<b>${rightLabel}</b>: rendimento annualizzato <b>${isFinite(cagrBench) ? pct(cagrBench, 2) : "—"}</b> | max ribasso <b>${isFinite(maxddBench) ? pct(maxddBench, 2) : "—"}</b>`
    );
  }

  async function loadLeva20(capital) {
    const payload = await fetchJson(`/api/compute_leva?capital=${encodeURIComponent(capital)}`);

    if (!payload || payload.ok !== true) {
      throw new Error(payload && payload.error ? payload.error : "Risposta backend non valida");
    }

    const labels = payload.dates || [];
    const leftVals = payload.pigro || [];
    const rightVals = payload.leva || [];
    const ddLeftVals = payload.dd_pigro || [];
    const ddRightVals = payload.dd_leva || [];
    const leftLabel = "Pigro";
    const rightLabel = "Pigro con leva 20%";

    destroyCharts();
    renderMain(labels, leftVals, rightVals, rightLabel, leftLabel);
    renderDd(labels, ddLeftVals, ddRightVals, rightLabel, leftLabel);

    const lastRight = rightVals[rightVals.length - 1];
    const years = labels.length >= 2 ? ((new Date(labels[labels.length - 1]) - new Date(labels[0])) / (365.25 * 24 * 3600 * 1000)) : 0;

    setText("summary_title", "Confronto “Pigro” vs “Pigro con leva 20%”:");
    setText("final_value", euro(lastRight, 0));
    setText("final_years", isFinite(years) ? plain(years, 1) : "—");
    setText("cagr", pct(payload.cagr_leva, 2));
    setText("maxdd", pct(payload.maxdd_leva, 2));
    setText("dbl_label", "Leva iniziale:");
    setText("dbl", "20.000 € su 100.000 €");

    setText("chart_title", "Andamento negli ultimi anni — confronto Pigro vs Leva 20%");
    setText("compare_title_benchmark", "Pigro con leva 20%");
    setText("compare_period", `${euro(capital, 0)} capitale proprio iniziale`);
    setText("compare_pigro", euro(leftVals[leftVals.length - 1], 0));
    setText("compare_benchmark", euro(lastRight, 0));

    setHtml(
      "dd_summary",
      `<div><b>Confronto sintetico</b></div>
       <div style="margin-top:4px;">Pigro: CAGR <b>${pct(payload.cagr_pigro, 2)}</b> | Max Drawdown <b>${pct(payload.maxdd_pigro, 2)}</b></div>
       <div style="margin-top:4px;">Leva 20%: CAGR <b>${pct(payload.cagr_leva, 2)}</b> | Max Drawdown <b>${pct(payload.maxdd_leva, 2)}</b></div>`
    );

    setHtml(
      "benchmark_summary",
      `Pigro: rendimento annualizzato <b>${pct(payload.cagr_pigro, 2)}</b> | max ribasso <b>${pct(payload.maxdd_pigro, 2)}</b><br/>
       Leva 20%: rendimento annualizzato <b>${pct(payload.cagr_leva, 2)}</b> | max ribasso <b>${pct(payload.maxdd_leva, 2)}</b>`
    );
  }

  async function loadLevaPlus(capital) {
    const payload = await fetchJson(`/api/compute_leva_plus?capital=${encodeURIComponent(capital)}`);

    if (!payload || payload.ok !== true) {
      throw new Error(payload && payload.error ? payload.error : "Risposta backend non valida");
    }

    const labels = payload.dates || [];
    const leftVals = payload.pigro || [];
    const rightVals = payload.leva_plus || [];
    const ddLeftVals = payload.dd_pigro || [];
    const ddRightVals = payload.dd_leva_plus || [];
    const events = payload.trigger_events || [];
    const leftLabel = "Pigro";
    const rightLabel = "Pigro Leva+";

    destroyCharts();
    renderMain(labels, leftVals, rightVals, rightLabel, leftLabel);
    renderDd(labels, ddLeftVals, ddRightVals, rightLabel, leftLabel);

    const lastRight = rightVals[rightVals.length - 1];
    const years = labels.length >= 2 ? ((new Date(labels[labels.length - 1]) - new Date(labels[0])) / (365.25 * 24 * 3600 * 1000)) : 0;

    let triggerText = `Attivazioni: ${payload.trigger_count || 0} su 3`;
    if (events.length) {
      triggerText += " — " + events.map((e) => `${e.n}ª: ${formatDateIt(e.date)}`).join(" | ");
    }

    let eventsHtml = `<div><b>Regola Leva+</b></div>
      <div style="margin-top:4px;">Soglia fissa: <b>108.000 €</b> di valore lordo del portafoglio.</div>
      <div style="margin-top:4px;">Ogni nuovo attraversamento al ribasso attiva <b>13.000 €</b> investiti solo su LS80.</div>
      <div style="margin-top:4px;">Attivazioni effettive: <b>${payload.trigger_count || 0}</b> su 3.</div>`;

    if (events.length) {
      eventsHtml += `<div style="margin-top:8px;"><b>Date di attivazione</b></div>`;
      events.forEach((e) => {
        eventsHtml += `<div style="margin-top:4px;">${e.n}ª attivazione — <b>${formatDateIt(e.date)}</b> | +${e.amount.toLocaleString("it-IT")} € su LS80</div>`;
      });
    }

    setText("summary_title", "Confronto “Pigro” vs “Pigro Leva+”:");
    setText("final_value", euro(lastRight, 0));
    setText("final_years", isFinite(years) ? plain(years, 1) : "—");
    setText("cagr", pct(payload.cagr_leva_plus, 2));
    setText("maxdd", pct(payload.maxdd_leva_plus, 2));
    setText("dbl_label", "Attivazioni Leva+:");
    setText("dbl", triggerText);

    setText("chart_title", "Andamento negli ultimi anni — confronto Pigro vs Leva+");
    setText("compare_title_benchmark", "Pigro Leva+");
    setText("compare_period", `${euro(capital, 0)} capitale proprio iniziale`);
    setText("compare_pigro", euro(leftVals[leftVals.length - 1], 0));
    setText("compare_benchmark", euro(lastRight, 0));

    setHtml("dd_summary", eventsHtml);

    setHtml(
      "benchmark_summary",
      `Pigro: rendimento annualizzato <b>${pct(payload.cagr_pigro, 2)}</b> | max ribasso <b>${pct(payload.maxdd_pigro, 2)}</b><br/>
       Leva+: rendimento annualizzato <b>${pct(payload.cagr_leva_plus, 2)}</b> | max ribasso <b>${pct(payload.maxdd_leva_plus, 2)}</b>`
    );
  }

  async function loadCharts() {
    normalizeCapitalInput();
    const capital = getCapital();

    try {
      updateModeUi();

      if (currentMode === "leva20") {
        await loadLeva20(capital);
      } else if (currentMode === "levaPlus") {
        await loadLevaPlus(capital);
      } else {
        await loadPigro(capital);
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
      setText("compare_title_benchmark", "Confronto");
      setText("dd_summary", "Impossibile caricare i dati del grafico.");
      setText("benchmark_summary", "Impossibile calcolare i dati.");

      alert("Impossibile caricare il grafico. Controlla app.py, i CSV e gli endpoint /api/compute, /api/compute_leva, /api/compute_leva_plus.");
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
        alert("Qui puoi collegare la finestra popup o la pagina con i consulenti OCF.");
      });
    }

    document.querySelectorAll(".benchmarkBtn").forEach((btn) => {
      btn.addEventListener("click", function () {
        currentBenchmark = btn.getAttribute("data-benchmark") || "world";
        setActiveBenchmarkButton();
        if (currentMode === "pigro") loadCharts();
      });
    });

    document.querySelectorAll(".modeBtn").forEach((btn) => {
      btn.addEventListener("click", function () {
        currentMode = btn.getAttribute("data-mode") || "pigro";
        setActiveModeButton();
        updateModeUi();
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
    setActiveModeButton();
    updateModeUi();
    loadCharts();
  });
})();
