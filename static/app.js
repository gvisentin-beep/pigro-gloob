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

  function thinSeries(labels, a, b, c, maxPoints = 300) {
    const len = labels.length;
    if (len <= maxPoints) {
      return { labels, a, b, c };
    }

    const step = Math.ceil(len / maxPoints);
    return {
      labels: labels.filter((_, i) => i % step === 0 || i === len - 1),
      a: a.filter((_, i) => i % step === 0 || i === len - 1),
      b: b.filter((_, i) => i % step === 0 || i === len - 1),
      c: c.filter((_, i) => i % step === 0 || i === len - 1)
    };
  }

  function sanitizeSeries(labels, pigroVals, worldVals, ddVals) {
    const cleanLabels = [];
    const cleanPigro = [];
    const cleanWorld = [];
    const cleanDd = [];

    const n = Math.min(labels.length, pigroVals.length, worldVals.length, ddVals.length);

    for (let i = 0; i < n; i += 1) {
      const label = labels[i];
      const p = Number(pigroVals[i]);
      const w = Number(worldVals[i]);
      const d = Number(ddVals[i]);

      if (!label) continue;
      if (!isFinite(p) || !isFinite(w) || !isFinite(d)) continue;

      cleanLabels.push(label);
      cleanPigro.push(p);
      cleanWorld.push(w);
      cleanDd.push(d);
    }

    return {
      labels: cleanLabels,
      pigro: cleanPigro,
      world: cleanWorld,
      dd: cleanDd
    };
  }

  function renderMain(labels, pigroVals, worldVals) {
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;

    mainChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Metodo Pigro 80/15/5",
            data: pigroVals,
            tension: 0.12,
            pointRadius: 0,
            borderWidth: 2
          },
          {
            label: "MSCI World",
            data: worldVals,
            tension: 0.12,
            pointRadius: 0,
            borderWidth: 2
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
                return `${ctx.dataset.label}: ${euro(ctx.parsed.y, 0)}`;
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

    ddChart = new Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Drawdown portafoglio",
            data: ddVals,
            tension: 0.1,
            pointRadius: 0,
            borderWidth: 2
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
                return `Drawdown: ${pct(ctx.parsed.y, 2)}`;
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

      const rawLabels = Array.isArray(payload.dates) ? payload.dates : [];
      const rawPigro = Array.isArray(payload.portfolio) ? payload.portfolio : [];
      const rawWorld = Array.isArray(payload.world) ? payload.world : [];
      const rawDd = Array.isArray(payload.drawdown_portfolio_pct)
        ? payload.drawdown_portfolio_pct
        : [];
      const metrics = payload.metrics || {};

      const cleaned = sanitizeSeries(rawLabels, rawPigro, rawWorld, rawDd);

      if (!cleaned.labels.length || !cleaned.pigro.length || !cleaned.world.length) {
        throw new Error("Dataset vuoto o non numerico");
      }

      const thinned = thinSeries(
        cleaned.labels,
        cleaned.pigro,
        cleaned.world,
        cleaned.dd,
        320
      );

      destroyCharts();
      renderMain(thinned.labels, thinned.a, thinned.b);
      renderDd(thinned.labels, thinned.c);

      const last = cleaned.pigro[cleaned.pigro.length - 1];
      const lastWorld = cleaned.world[cleaned.world.length - 1];
      const firstDate = cleaned.labels[0] || "inizio periodo";
      const lastDate = cleaned.labels[cleaned.labels.length - 1] || "";

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
        `Peggior ribasso del portafoglio nel periodo: ${
          isFinite(maxdd) ? pct(maxdd, 2) : "—"
        }`
      );

      console.log("Grafico caricato:", {
        puntiOriginali: rawLabels.length,
        puntiPuliti: cleaned.labels.length,
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

      alert("Impossibile caricare il grafico. Controlla /api/compute e la console del browser.");
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
