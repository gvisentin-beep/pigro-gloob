// =============================
// CONFIG BASE
// =============================

let chartMain = null;
let chartDD = null;

// =============================
// FORMAT UTILI
// =============================

function formatEuro(v) {
  if (!v && v !== 0) return "—";
  return Math.round(v).toLocaleString("it-IT") + " €";
}

function formatPerc(v) {
  if (!v && v !== 0) return "—";
  return v.toFixed(1) + "%";
}

// =============================
// FETCH DATI PORTAFOGLIO
// =============================

async function loadData() {
  try {
    const capital = document.getElementById("capital").value.replace(".", "").replace(",", ".");
    
    const res = await fetch("/api/compute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ capital })
    });

    const data = await res.json();

    updateUI(data);

  } catch (err) {
    console.error("Errore loadData:", err);
  }
}

// =============================
// AGGIORNA UI
// =============================

function updateUI(data) {
  if (!data) return;

  document.getElementById("final_value").innerText = formatEuro(data.final_value);
  document.getElementById("cagr").innerText = formatPerc(data.cagr);
  document.getElementById("maxdd").innerText = formatPerc(data.maxdd);
  document.getElementById("dbl").innerText = data.doubling_years;

  drawChart(data);
  drawDD(data);
}

// =============================
// GRAFICO PRINCIPALE
// =============================

function drawChart(data) {
  const ctx = document.getElementById("chart_main");

  if (chartMain) chartMain.destroy();

  chartMain = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Portafoglio Pigro",
          data: data.values,
          borderWidth: 2,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true }
      },
      scales: {
        x: {
          ticks: {
            callback: function(value, index, ticks) {
              const date = data.dates[index];
              return date.includes("-01-") ? date.substring(0,4) : "";
            }
          }
        }
      }
    }
  });
}

// =============================
// GRAFICO DRAWDOWN
// =============================

function drawDD(data) {
  const ctx = document.getElementById("chart_dd");

  if (chartDD) chartDD.destroy();

  chartDD = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.dates,
      datasets: [
        {
          label: "Drawdown",
          data: data.drawdown,
          borderWidth: 1,
          tension: 0.2
        }
      ]
    }
  });
}

// =============================
// DOMANDE LIBERE (ASSISTENTE)
// =============================

function initAskAssistant() {
  const btn = document.getElementById("btn_ask");
  const textarea = document.getElementById("ask_text");
  const answerBox = document.getElementById("ask_answer");

  if (!btn || !textarea || !answerBox) return;

  const MAX_DAILY = 10;
  const todayKey = new Date().toISOString().slice(0, 10);

  function getUsage() {
    const data = JSON.parse(localStorage.getItem("ask_usage") || "{}");
    return data[todayKey] || 0;
  }

  function setUsage(val) {
    let data = JSON.parse(localStorage.getItem("ask_usage") || "{}");
    data[todayKey] = val;
    localStorage.setItem("ask_usage", JSON.stringify(data));
  }

  async function ask() {
    const question = textarea.value.trim();
    if (!question) return;

    let used = getUsage();

    if (used >= MAX_DAILY) {
      answerBox.style.display = "block";
      answerBox.innerHTML = "Hai raggiunto il limite giornaliero di 10 domande.";
      return;
    }

    btn.disabled = true;
    btn.innerText = "Attendi...";

    answerBox.style.display = "block";
    answerBox.innerHTML = "Sto pensando...";

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ question })
      });

      const data = await res.json();

      if (data.answer) {
        answerBox.innerHTML = data.answer;
        setUsage(used + 1);
      } else {
        answerBox.innerHTML = "Errore nella risposta.";
      }

    } catch (err) {
      answerBox.innerHTML = "Errore di connessione.";
    }

    btn.disabled = false;
    btn.innerText = "Chiedi all’assistente";
  }

  btn.addEventListener("click", ask);

  textarea.addEventListener("keydown", function (e) {
    if (e.ctrlKey && e.key === "Enter") {
      ask();
    }
  });
}

// =============================
// INIT GENERALE
// =============================

document.addEventListener("DOMContentLoaded", function () {

  // grafici
  const btnUpdate = document.getElementById("btn_update");
  if (btnUpdate) {
    btnUpdate.addEventListener("click", loadData);
  }

  // assistente
  initAskAssistant();

  // prima chiamata
  loadData();
});
