(function () {
  let chartMain = null;
  let chartDD = null;

  function fmtEuro(value, digits = 0) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString("it-IT", {
      style: "currency",
      currency: "EUR",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function fmtPct(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString("it-IT", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    }) + "%";
  }

  function fmtNum(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString("it-IT", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
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

  async function fetchJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    return await res.json();
  }

  function destroyCharts() {
    if (chartMain) {
      chartMain.destroy();
      chartMain = null;
    }
    if (chartDD) {
      chartDD.destroy();
      chartDD = null;
    }
  }

  function sanitizeSeries(labels, portfolio, world, dd) {
    const cleanLabels = [];
    const cleanPortfolio = [];
    const cleanWorld = [];
    const cleanDD = [];

    const n = Math.min(
      Array.isArray(labels) ? labels.length : 0,
      Array.isArray(portfolio) ? portfolio.length : 0,
      Array.isArray(world) ? world.length : 0,
      Array.isArray(dd) ? dd.length : 0
    );

    for (let i = 0; i < n; i += 1) {
      const label = labels[i];
      const p = Number(portfolio[i]);
      const w = Number(world[i]);
      const d = Number(dd[i]);

      if (!label) continue;
      if (!isFinite(p) || !isFinite(w) || !isFinite(d)) continue;

      cleanLabels.push(label);
      cleanPortfolio.push(p);
      cleanWorld.push(w);
      cleanDD.push(d);
    }

    return {
      labels: cleanLabels,
      portfolio: cleanPortfolio,
      world: cleanWorld,
      dd: cleanDD
    };
  }

  function thinData(labels, a, b, c, maxPoints = 320) {
    if (labels.length <= maxPoints) {
      return { labels, a, b, c };
    }

    const step = Math.ceil(labels.length / maxPoints);

    return {
      labels: labels.filter((_, i) => i % step === 0 || i === labels.length - 1),
      a: a.filter((_, i) => i % step === 0 || i === labels.length - 1),
      b: b.filter((_, i) => i % step === 0 || i === labels.length - 1),
      c: c.filter((_, i) => i % step === 0 || i === labels.length - 1)
    };
  }

  function renderMain(labels, portfolioVals, worldVals) {
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    chartMain = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Metodo Pigro 80/15/5",
            data: portfolioVals,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          },
          {
            label: "MSCI World",
            data: worldVals,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
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
        plugins: {
          legend: {
            display: true
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return `${ctx.dataset.label}: ${fmtEuro(ctx.parsed.y, 0)}`;
              }
            }
          }
        },
        scales: {
          x: {
            ticks: {
              maxTicksLimit: 10
            }
          },
          y: {
            ticks: {
              callback: function (value) {
                return fmtEuro(value, 0);
              }
            }
          }
        }
      }
    });
  }

  function renderDD(labels, ddVals) {
    const canvas = document.getElementById("chart_dd");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    chartDD = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Drawdown portafoglio",
            data: ddVals,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            display: true
          },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return `Drawdown: ${fmtPct(ctx.parsed.y, 2)}`;
              }
            }
          }
        },
        scales: {
          x: {
            ticks: {
              maxTicksLimit: 10
            }
          },
          y: {
            ticks: {
              callback: function (value) {
                return fmtPct(value, 0);
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
    const input = document.getElementById("ask_text");
    const output = document.getElementById("ask_answer");
    if (!input || !output) return;

    const question = String(input.value || "").trim();
    if (!question) return;

    output.style.display = "block";
    output.textContent = "Attendi…";

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
      output.textContent =
        payload.answer ||
        payload.response ||
        "Nessuna risposta disponibile.";
    } catch (err) {
      console.error("Errore assistente:", err);
      output.textContent =
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

      const rawLabels = Array.isArray(payload.dates) ? payload.dates : [];
      const rawPortfolio = Array.isArray(payload.portfolio) ? payload.portfolio : [];
      const rawWorld = Array.isArray(payload.world) ? payload.world : [];
      const rawDD = Array.isArray(payload.drawdown_portfolio_pct)
        ? payload.drawdown_portfolio_pct
        : [];
      const metrics = payload.metrics || {};

      const cleaned = sanitizeSeries(rawLabels, rawPortfolio, rawWorld, rawDD);

      if (!cleaned.labels.length) {
        throw new Error("Nessun dato valido da disegnare");
      }

      const thinned = thinData(
        cleaned.labels,
        cleaned.portfolio,
        cleaned.world,
        cleaned.dd,
        320
      );

      destroyCharts();
      renderMain(thinned.labels, thinned.a, thinned.b);
      renderDD(thinned.labels, thinned.c);

      const lastPortfolio = cleaned.portfolio[cleaned.portfolio.length - 1];
      const lastWorld = cleaned.world[cleaned.world.length - 1];
      const firstDate = cleaned.labels[0] || "inizio periodo";
      const lastDate = cleaned.labels[cleaned.labels.length - 1] || "";

      const years = Number(metrics.final_years);
      const cagr = Number(metrics.cagr_portfolio) * 100;
      const maxdd = Number(metrics.max_dd_portfolio) * 100;
      const dbl = Number(metrics.doubling_years_portfolio);

      setText("final_value", fmtEuro(lastPortfolio, 0));
      setText("final_years", isFinite(years) ? fmtNum(years, 1) : "—");
      setText("cagr", isFinite(cagr) ? fmtPct(cagr, 2) : "—");
      setText("maxdd", isFinite(maxdd) ? fmtPct(maxdd, 2) : "—");
      setText("dbl", isFinite(dbl) ? fmtNum(dbl, 1) : "—");

      setText(
        "compare_period",
        `${fmtEuro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})`
      );
      setText("compare_pigro", fmtEuro(lastPortfolio, 0));
      setText("compare_world", fmtEuro(lastWorld, 0));
      setText(
        "dd_summary",
        `Peggior ribasso del portafoglio nel periodo: ${isFinite(maxdd) ? fmtPct(maxdd, 2) : "—"}`
      );

      console.log("Grafici caricati:", {
        puntiRicevuti: rawLabels.length,
        puntiValidi: cleaned.labels.length,
        puntiDisegnati: thinned.labels.length
      });
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

      alert("Impossibile caricare il grafico. Controlla la console del browser.");
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
