let chart = null;

function euro(x) {
  return new Intl.NumberFormat("it-IT", {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 0,
  }).format(Number(x));
}

function num0(x) {
  return new Intl.NumberFormat("it-IT", {
    maximumFractionDigits: 0,
  }).format(Number(x));
}

function pct(x) {
  if (x === null || x === undefined || isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

function years1(x) {
  if (x === null || x === undefined || isNaN(x)) return "—";
  return Number(x).toFixed(1).replace(".", ",");
}

function clamp(n, a, b) {
  return Math.min(Math.max(n, a), b);
}

function computeComposition(goldPct) {
  const w_gold = goldPct;
  const w_ls80 = 100 - w_gold;

  // LS80 = 80% equity + 20% bond
  const equity = 0.8 * w_ls80;
  const bond = 0.2 * w_ls80;

  return {
    w_gold,
    w_ls80,
    equity: Math.round(equity),
    bond: Math.round(bond),
  };
}

function setAskStatus(msg, kind = "info") {
  const box = document.getElementById("ask_status");
  if (!box) return;

  box.classList.remove("ok", "err", "info");
  box.classList.add(kind);
  box.textContent = msg || "";
  box.style.display = msg ? "block" : "none";
}

function setRemaining(remaining, limit) {
  const el = document.getElementById("ask_remaining");
  if (!el) return;
  if (remaining === null || remaining === undefined || limit === null || limit === undefined) {
    el.textContent = "";
    return;
  }
  el.textContent = `Domande rimanenti oggi: ${remaining}/${limit}`;
}

async function loadData() {
  const goldPct = Number(document.getElementById("w_gold").value || 0);
  const capitalRaw = (document.getElementById("initial").value || "").toString().replace(/\./g, "");
  const capital = Number(capitalRaw) || 10000;

  const comp = computeComposition(goldPct);

  // aggiorna label valore slider
  const goldVal = document.getElementById("w_gold_val");
  if (goldVal) goldVal.textContent = `${comp.w_gold}%`;

  const url =
    `/api/compute?w_ls80=${encodeURIComponent(comp.w_ls80)}` +
    `&w_gold=${encodeURIComponent(comp.w_gold)}` +
    `&capital=${encodeURIComponent(capital)}` +
    `&t=${Date.now()}`;

  const res = await fetch(url, { cache: "no-store" });

  // se il server risponde HTML (errore), evitiamo crash “Unexpected token <”
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  const data = ct.includes("application/json") ? await res.json() : { error: await res.text() };

  if (!res.ok || data.error) {
    console.error(data.error || res.statusText);
    // non facciamo alert: scriviamo nel box dell’assistente (più “premium”)
    setAskStatus(`Errore: ${(data.error || res.statusText).toString().slice(0, 160)}`, "err");
    return;
  }

  // Metrics lines
  const m = data.metrics || {};
  const line1 = document.getElementById("metrics_line1");
  const line2 = document.getElementById("metrics_line2");
  const line3 = document.getElementById("metrics_line3");

  if (line1) {
    line1.textContent =
      `Portafoglio (ETF Azion-Obblig + ETC Oro): ` +
      `Rendimento annualizzato ${pct(m.cagr_portfolio)} | ` +
      `Max Ribasso nel periodo ${pct(m.max_dd_portfolio)}`;
  }

  if (line2) {
    line2.textContent =
      `Composizione: Azionario ${comp.equity}% | Obbligazionario ${comp.bond}% | Oro ${comp.w_gold}%`;
  }

  if (line3) {
    const dbl = m.doubling_years_portfolio;
    line3.textContent = `Raddoppio del portafoglio in anni: ${years1(dbl)}`;
  }

  // Finale a fianco del capitale
  const finalValue = document.getElementById("final_value");
  const finalYears = document.getElementById("final_years");
  if (finalValue) finalValue.textContent = euro(m.final_portfolio);
  if (finalYears) finalYears.textContent = years1(m.final_years);

  // Chart
  const ctx = document.getElementById("chart").getContext("2d");
  if (chart) chart.destroy();

  // time scale: dataset con {x, y}
  const seriesPortfolio = data.dates.map((d, i) => ({ x: d, y: data.portfolio[i] }));

  chart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Portafoglio (ETF Azion-Obblig + ETC Oro)",
          data: seriesPortfolio,
          borderWidth: 2,
          tension: 0.15,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        tooltip: {
          callbacks: {
            label: (context) => `${context.dataset.label}: ${euro(context.parsed.y)}`,
          },
        },
        legend: { display: true },
      },
      scales: {
        x: {
          type: "time",
          time: {
            unit: "month",
            stepSize: 6, // ~ 2 linee/anno
          },
          ticks: {
            maxRotation: 0,
            autoSkip: true,
          },
        },
        y: {
          ticks: { callback: (value) => euro(value) },
        },
      },
    },
  });

  // pulisci eventuale errore “premium” precedente
  setAskStatus("", "info");
}

function formatCapitalField() {
  const el = document.getElementById("initial");
  if (!el) return;

  // lascia digitare, ma formatta on blur
  el.addEventListener("blur", () => {
    const raw = (el.value || "").toString().replace(/\./g, "").replace(/[^\d]/g, "");
    const n = Number(raw || "0");
    if (!n) {
      el.value = "10.000";
      return;
    }
    el.value = num0(n);
  });
}

function wireControls() {
  const btn = document.getElementById("btn_update");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      loadData();
    });
  }

  const slider = document.getElementById("w_gold");
  if (slider) {
    slider.addEventListener("input", () => {
      const goldPct = Number(slider.value || 0);
      const comp = computeComposition(goldPct);
      const goldVal = document.getElementById("w_gold_val");
      if (goldVal) goldVal.textContent = `${comp.w_gold}%`;
    });
  }
}

async function askAssistant() {
  const q = (document.getElementById("ask_question").value || "").trim();
  const btn = document.getElementById("ask_btn");
  const ans = document.getElementById("ask_answer");

  if (!q) {
    setAskStatus("Scrivi una domanda.", "err");
    return;
  }

  if (btn) btn.disabled = true;
  setAskStatus("Sto pensando…", "info");

  try {
    const res = await fetch(`/api/ask?t=${Date.now()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({ question: q }),
    });

    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const data = ct.includes("application/json") ? await res.json() : { ok: false, error: await res.text() };

    if (!res.ok || !data.ok) {
      const msg = (data.error || `Errore server (${res.status})`).toString();
      setAskStatus(msg.slice(0, 240), "err");
      if (ans) ans.textContent = "";
      // se arrivano comunque remaining/limit, li mostriamo
      if ("remaining" in data && "limit" in data) setRemaining(data.remaining, data.limit);
      return;
    }

    if (ans) ans.textContent = data.answer || "";
    setAskStatus("", "ok");
    setRemaining(data.remaining, data.limit);
  } catch (e) {
    setAskStatus(`Errore rete: ${e}`, "err");
  } finally {
    if (btn) btn.disabled = false;
  }
}

function wireAssistant() {
  const btn = document.getElementById("ask_btn");
  if (!btn) return;

  btn.addEventListener("click", (e) => {
    e.preventDefault();
    askAssistant();
  });

  const ta = document.getElementById("ask_question");
  if (ta) {
    ta.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        askAssistant();
      }
    });
  }
}

function init() {
  formatCapitalField();
  wireControls();
  wireAssistant();
  loadData();
}

window.addEventListener("DOMContentLoaded", init);
