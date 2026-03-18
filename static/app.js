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

  function yearsToDouble(cagrPct) {
    const r = Number(cagrPct) / 100;
    if (!isFinite(r) || r <= 0) return null;
    return Math.log(2) / Math.log(1 + r);
  }

  function getCapital() {
    const el = document.getElementById("capital");
    if (!el) return 10000;
    const raw = String(el.value || "").replace(/\./g, "").replace(",", ".").trim();
    const n = Number(raw);
    return isFinite(n) && n > 0 ? n : 10000;
  }

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function parseSeries(payload) {
    if (!payload) return [];
    if (Array.isArray(payload.series)) return payload.series;
    if (Array.isArray(payload.data)) return payload.data;
    if (Array.isArray(payload.rows)) return payload.rows;
    if (Array.isArray(payload)) return payload;
    return [];
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

  function renderMain(labels, pigroVals, worldVals) {
    const ctx = document.getElementById("chart_main");
    if (!ctx) return;

    mainChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
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
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: true },
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
            ticks: { maxTicksLimit: 10 }
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
    const ctx = document.getElementById("chart_dd");
    if (!ctx) return;

    ddChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
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
        plugins: {
          legend: { display: true },
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
            ticks: { maxTicksLimit: 10 }
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
    document.querySelectorAll(".faqItem").forEach((item) => {
      item.addEventListener("click", () => item.classList.toggle("open"));
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
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const payload = await res.json();
      out.textContent = payload.answer || payload.response || "Nessuna risposta disponibile.";
    } catch (err) {
      out.textContent = "Non sono riuscito a contattare l’assistente. Riprova tra poco.";
      console.error(err);
    }
  }

  async function loadCharts() {
    const capital = getCapital();

    try {
      const payload = await fetchJson(`/api/portfolio?capital=${encodeURIComponent(capital)}`);
      const series = parseSeries(payload);

      if (!series.length) throw new Error("Dataset vuoto");

      const labels = series.map(r => r.date || r.label || "");
      const pigroVals = series.map(r => Number(r.portfolio_value ?? r.portfolio ?? r.pigro_value ?? r.pigro));
      const worldVals = series.map(r => Number(r.world_value ?? r.world ?? r.msci_world_value ?? r.msci_world));
      const ddVals = series.map(r => Number(r.drawdown ?? r.portfolio_drawdown ?? 0));

      const validPigro = pigroVals.filter(v => isFinite(v));
      const validWorld = worldVals.filter(v => isFinite(v));
      if (!validPigro.length || !validWorld.length) throw new Error("Serie dati non valide");

      destroyCharts();
      renderMain(labels, pigroVals, worldVals);
      renderDd(labels, ddVals);

      const last = validPigro[validPigro.length - 1];
      const lastWorld = validWorld[validWorld.length - 1];
      const firstDate = labels[0] || "inizio periodo";
      const lastDate = labels[labels.length - 1] || "";
      const years = payload.years ?? payload.summary?.years ?? null;
      const cagr = payload.cagr ?? payload.summary?.cagr;
      const maxdd = payload.max_drawdown ?? payload.maxdd ?? payload.summary?.max_drawdown ?? Math.min(...ddVals.filter(v => isFinite(v)));
      const dbl = yearsToDouble(cagr);

      setText("final_value", euro(last, 0));
      setText("final_years", isFinite(years) ? plain(years, 1) : "—");
      setText("cagr", pct(cagr, 2));
      setText("maxdd", pct(maxdd, 2));
      setText("dbl", dbl ? plain(dbl, 1) : "—");

      setText("compare_period", `${euro(capital, 0)} investiti all’inizio del periodo (${firstDate} → ${lastDate})`);
      setText("compare_pigro", euro(last, 0));
      setText("compare_world", euro(lastWorld, 0));
      setText("dd_summary", `Peggior ribasso del portafoglio nel periodo: ${pct(maxdd, 2)}`);
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
      alert("Impossibile caricare il grafico. Controlla che /api/portfolio risponda correttamente.");
    }
  }

  function wireButtons() {
    const btnUpdate = document.getElementById("btn_update");
    if (btnUpdate) btnUpdate.addEventListener("click", loadCharts);

    const btnPdf = document.getElementById("btn_pdf");
    if (btnPdf) btnPdf.addEventListener("click", () => window.print());

    const btnAsk = document.getElementById("btn_ask");
    if (btnAsk) btnAsk.addEventListener("click", askAssistant);

    const btnLibro = document.getElementById("btn_libro");
    if (btnLibro) {
      btnLibro.addEventListener("click", () => {
        window.open("https://www.amazon.it/dp/B0GQM925QR/ref=sr", "_blank", "noopener");
      });
    }

    const btnFax = document.getElementById("btn_faxsimile");
    if (btnFax) {
      btnFax.addEventListener("click", () => {
        window.location.href = "/static/faxsimile.pdf";
      });
    }

    const btnCons = document.getElementById("btn_consulente");
    if (btnCons) {
      btnCons.addEventListener("click", () => {
        alert("Qui puoi collegare la finestra popup o la pagina con i consulenti OCF.");
      });
    }

    const capital = document.getElementById("capital");
    if (capital) {
      capital.addEventListener("keydown", (e) => {
        if (e.key === "Enter") loadCharts();
      });
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderFaq();
    wireButtons();
    loadCharts();
  });
})();
