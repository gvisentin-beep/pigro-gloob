(function () {
  let mainChart = null;
  let ddChart = null;

  Chart.defaults.color = "#334155";
  Chart.defaults.font.family = "Arial, sans-serif";

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

  function getCapital() {
    const el = document.getElementById("capital");
    if (!el) return 10000;

    const raw = String(el.value || "")
      .replace(/\./g, "")
      .replace(",", ".")
      .trim();

    const n = Number(raw);
    return isFinite(n) && n > 0 ? n : 10000;
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
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

  function yearTickCallback(value, index, ticks) {
    const label = this.getLabelForValue(value);
    if (!label) return "";

    const year = String(label).slice(0, 4);

    if (index === 0) return year;

    const prevLabel = this.getLabelForValue(ticks[index - 1].value);
    const prevYear = prevLabel ? String(prevLabel).slice(0, 4) : "";

    return year !== prevYear ? year : "";
  }

  function makeGradient(ctx, area, colorTop, colorBottom) {
    const gradient = ctx.createLinearGradient(0, area.top, 0, area.bottom);
    gradient.addColorStop(0, colorTop);
    gradient.addColorStop(1, colorBottom);
    return gradient;
  }

  function commonGridColor() {
    return "rgba(15, 23, 42, 0.06)";
  }

  function commonBorderColor() {
    return "rgba(15, 23, 42, 0.10)";
  }

  function renderMain(labels, pigroVals, worldVals) {
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    mainChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Metodo Pigro 80/15/5",
            data: pigroVals,
            borderColor: "#3b82f6",
            backgroundColor: function (context) {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return "rgba(59,130,246,0.10)";
              return makeGradient(
                ctx,
                chartArea,
                "rgba(59,130,246,0.18)",
                "rgba(59,130,246,0.02)"
              );
            },
            fill: false,
            tension: 0.22,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2.5
          },
          {
            label: "MSCI World",
            data: worldVals,
            borderColor: "#fb7185",
            backgroundColor: function (context) {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return "rgba(251,113,133,0.08)";
              return makeGradient(
                ctx,
                chartArea,
                "rgba(251,113,133,0.12)",
                "rgba(251,113,133,0.02)"
              );
            },
            fill: false,
            tension: 0.22,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        layout: {
          padding: {
            top: 8,
            right: 12,
            bottom: 0,
            left: 6
          }
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: {
              boxWidth: 24,
              boxHeight: 8,
              usePointStyle: false,
              color: "#475569",
              padding: 14
            }
          },
          tooltip: {
            backgroundColor: "rgba(255,255,255,0.96)",
            titleColor: "#0f172a",
            bodyColor: "#334155",
            borderColor: "rgba(15, 23, 42, 0.10)",
            borderWidth: 1,
            padding: 10,
            displayColors: true,
            callbacks: {
              title: function (items) {
                if (!items || !items.length) return "";
                return items[0].label;
              },
              label: function (ctx) {
                return `${ctx.dataset.label}: ${euro(ctx.parsed.y, 0)}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              color: commonGridColor(),
              drawBorder: false
            },
            border: {
              color: commonBorderColor()
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              color: "#64748b",
              callback: yearTickCallback
            }
          },
          y: {
            grid: {
              color: commonGridColor(),
              drawBorder: false
            },
            border: {
              color: commonBorderColor()
            },
            ticks: {
              color: "#64748b",
              callback: function (value) {
                return euro(value, 0);
              }
            }
          }
        }
      }
    });
  }

  function renderDd(labels, ddVals) {
    const canvas = document.getElementById("chart_dd");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    ddChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Drawdown",
            data: ddVals,
            borderColor: "#38bdf8",
            backgroundColor: function (context) {
              const chart = context.chart;
              const { ctx, chartArea } = chart;
              if (!chartArea) return "rgba(56,189,248,0.14)";
              return makeGradient(
                ctx,
                chartArea,
                "rgba(56,189,248,0.18)",
                "rgba(56,189,248,0.03)"
              );
            },
            fill: true,
            tension: 0.18,
            pointRadius: 0,
            pointHoverRadius: 3,
            borderWidth: 2.2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        layout: {
          padding: {
            top: 6,
            right: 12,
            bottom: 0,
            left: 6
          }
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: {
              boxWidth: 24,
              boxHeight: 8,
              color: "#475569",
              padding: 12
            }
          },
          tooltip: {
            backgroundColor: "rgba(255,255,255,0.96)",
            titleColor: "#0f172a",
            bodyColor: "#334155",
            borderColor: "rgba(15, 23, 42, 0.10)",
            borderWidth: 1,
            padding: 10,
            callbacks: {
              title: function (items) {
                if (!items || !items.length) return "";
                return items[0].label;
              },
              label: function (ctx) {
                return `Drawdown: ${pct(ctx.parsed.y, 2)}`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              color: commonGridColor(),
              drawBorder: false
            },
            border: {
              color: commonBorderColor()
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              color: "#64748b",
              callback: yearTickCallback
            }
          },
          y: {
            grid: {
              color: commonGridColor(),
              drawBorder: false
            },
            border: {
              color: commonBorderColor()
            },
            ticks: {
              color: "#64748b",
              callback: function (value) {
                return pct(value, 0);
              }
            }
          }
        }
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

  async function loadCharts() {
    const capital = getCapital();

    try {
      const payload = await fetchJson(
        `/api/compute?capital=${encodeURIComponent(capital)}`
      );

      if (!payload || payload.ok !== true) {
        throw new Error(payload && payload.error ? payload.error : "Risposta backend non valida");
      }

      const labels = Array.isArray(payload.dates) ? payload.dates : [];
      const pigroVals = Array.isArray(payload.portfolio) ? payload.portfolio.map(Number) : [];
      const worldVals = Array.isArray(payload.world) ? payload.world.map(Number) : [];
      const ddVals = Array.isArray(payload.drawdown_portfolio_pct)
        ? payload.drawdown_portfolio_pct.map(Number)
        : [];
      const metrics = payload.metrics || {};

      const n = Math.min(labels.length, pigroVals.length, worldVals.length, ddVals.length);
      if (!n) {
        throw new Error("Dataset vuoto");
      }

      const cleanLabels = [];
      const cleanPigro = [];
      const cleanWorld = [];
      const cleanDd = [];

      for (let i = 0; i < n; i++) {
        const lab = labels[i];
        const p = pigroVals[i];
        const w = worldVals[i];
        const d = ddVals[i];

        if (!lab) continue;
        if (!isFinite(p) || !isFinite(w) || !isFinite(d)) continue;

        cleanLabels.push(lab);
        cleanPigro.push(p);
        cleanWorld.push(w);
        cleanDd.push(d);
      }

      if (!cleanLabels.length) {
        throw new Error("Dati non validi per il grafico");
      }

      destroyCharts();
      renderMain(cleanLabels, cleanPigro, cleanWorld);
      renderDd(cleanLabels, cleanDd);

      const last = cleanPigro[cleanPigro.length - 1];
      const lastWorld = cleanWorld[cleanWorld.length - 1];
      const firstDate = cleanLabels[0] || "inizio periodo";
      const lastDate = cleanLabels[cleanLabels.length - 1] || "";

      const years = Number(metrics.final_years);
      const cagr = Number(metrics.cagr_portfolio) * 100;
      const maxdd = Number(metrics.max_dd_portfolio) * 100;
      const dbl = Number(metrics.doubling_years_portfolio);

      setText("final_value", euro(last, 0));
      setText("final_years", isFinite(years) ? plain(years, 1) : "—");
      setText("cagr", isFinite(cagr) ? pct(cagr, 2) : "—");
      setText("maxdd", isFinite(maxdd) ? pct(maxdd, 2) : "—");
      setText("dbl", isFinite(dbl) ? plain(dbl, 1) : "—");

      setText(
        "compare_period",
        `${euro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})`
      );
      setText("compare_pigro", euro(last, 0));
      setText("compare_world", euro(lastWorld, 0));
      setText(
        "dd_summary",
        `Peggior ribasso del portafoglio nel periodo: ${isFinite(maxdd) ? pct(maxdd, 2) : "—"}`
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
      setText("compare_world", "—");
      setText("dd_summary", "Impossibile caricare i dati del grafico.");

      alert("Impossibile caricare il grafico. Controlla /api/compute.");
    }
  }

  function wireButtons() {
    const btnUpdate = document.getElementById("btn_update");
    if (btnUpdate) {
      btnUpdate.addEventListener("click", loadCharts);
    }

    const btnPdf = document.getElementById("btn_pdf");
    if (btnPdf) {
      btnPdf.addEventListener("click", function () {
        window.print();
      });
    }

    const btnAsk = document.getElementById("btn_ask");
    if (btnAsk) {
      btnAsk.addEventListener("click", askAssistant);
    }

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

    const capital = document.getElementById("capital");
    if (capital) {
      capital.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          loadCharts();
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderFaq();
    wireButtons();
    loadCharts();
  });
})();
