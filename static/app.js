(function () {
  let mainChart = null;
  let ddChart = null;

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

  function computeDrawdownPct(values) {
    if (!Array.isArray(values) || !values.length) return [];

    let peak = Number(values[0]);
    const out = [];

    for (let i = 0; i < values.length; i++) {
      const v = Number(values[i]);

      if (!isFinite(v)) {
        out.push(null);
        continue;
      }

      if (!isFinite(peak) || v > peak) {
        peak = v;
      }

      if (!isFinite(peak) || peak <= 0) {
        out.push(0);
      } else {
        out.push(((v / peak) - 1) * 100);
      }
    }

    return out;
  }

  function minNumber(arr) {
    const nums = (arr || []).filter(v => isFinite(Number(v))).map(Number);
    return nums.length ? Math.min(...nums) : null;
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

  function renderMain(labels, pigroVals, worldVals) {
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
            label: "MSCI World",
            data: worldVals
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
            grid: {
              display: false
            },
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
            grid: {
              color: "rgba(0,0,0,0.06)"
            },
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

  function renderDd(labels, ddPigroVals, ddWorldVals) {
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
            label: "Drawdown MSCI World",
            data: ddWorldVals
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
            grid: {
              display: false
            },
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
            grid: {
              color: "rgba(0,0,0,0.06)"
            },
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
        throw new Error(
          payload && payload.error ? payload.error : "Risposta backend non valida"
        );
      }

      const labels = payload.dates || [];
      const pigroVals = payload.portfolio || [];
      const worldVals = payload.world || [];
      const ddPigroVals =
        payload.drawdown_portfolio_pct && payload.drawdown_portfolio_pct.length
          ? payload.drawdown_portfolio_pct
          : computeDrawdownPct(pigroVals);

      const ddWorldVals =
        payload.drawdown_world_pct && payload.drawdown_world_pct.length
          ? payload.drawdown_world_pct
          : computeDrawdownPct(worldVals);

      const metrics = payload.metrics || {};

      if (!labels.length || !pigroVals.length || !worldVals.length) {
        throw new Error("Dataset vuoto");
      }

      destroyCharts();
      renderMain(labels, pigroVals, worldVals);
      renderDd(labels, ddPigroVals, ddWorldVals);

      const last = pigroVals[pigroVals.length - 1];
      const lastWorld = worldVals[worldVals.length - 1];
      const firstDate = labels[0] || "inizio periodo";
      const lastDate = labels[labels.length - 1] || "";

      const years = Number(metrics.final_years);
      const cagr = Number(metrics.cagr_portfolio) * 100;
      const maxddPigroRaw = Number(metrics.max_dd_portfolio);
      const maxddPigro = isFinite(maxddPigroRaw)
        ? maxddPigroRaw * 100
        : minNumber(ddPigroVals);

      const maxddWorldRaw = Number(metrics.max_dd_world);
      const maxddWorld = isFinite(maxddWorldRaw)
        ? maxddWorldRaw * 100
        : minNumber(ddWorldVals);

      const dbl = Number(metrics.doubling_years_portfolio);

      setText("final_value", euro(last, 0));
      setText("final_years", isFinite(years) ? plain(years, 1) : "—");
      setText("cagr", isFinite(cagr) ? pct(cagr, 2) : "—");
      setText("maxdd", isFinite(maxddPigro) ? pct(maxddPigro, 2) : "—");
      setText("dbl", isFinite(dbl) ? plain(dbl, 1) : "—");

      setText(
        "compare_period",
        `${euro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})`
      );
      setText("compare_pigro", euro(last, 0));
      setText("compare_world", euro(lastWorld, 0));

      const ddTextPigro = isFinite(maxddPigro) ? pct(maxddPigro, 2) : "—";
      const ddTextWorld = isFinite(maxddWorld) ? pct(maxddWorld, 2) : "—";

      setText(
        "dd_summary",
        `Peggior ribasso nel periodo — Portafoglio Pigro: ${ddTextPigro} | MSCI World: ${ddTextWorld}`
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
