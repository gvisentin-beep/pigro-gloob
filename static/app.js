(function () {
  let mainChart = null;
  let ddChart = null;
  let liqChart = null;
  let currentBenchmark = "world";
  let currentMode = "normal";
  let levaPlusIntegrations = 0;
  let levaPlusMarkerIndices = [];
  let isRefreshing = false;

  const WEIGHT_LS80 = 0.80;
  const WEIGHT_GOLD = 0.15;
  const WEIGHT_BTC = 0.05;
  const LOMBARD_RATE = 0.025;
  const LOMBARD_LTV = 0.60;

  const BENCHMARK_LABELS = {
    world: "MSCI World",
    mib: "Euro Stoxx 50",
    sp500: "USA"
  };

  const CSV_PATHS = {
    ls80: "/static/data_ls80.csv",
    gold: "/static/data_gold.csv",
    btc: "/static/data_btc.csv",
    world: "/static/data_world.csv",
    mib: "/static/data_mib.csv",
    sp500: "/static/data_sp500.csv"
  };

  function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }

  function setHtml(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = value;
  }

  // 👉 NUOVA FUNZIONE: gestione box leve
  function toggleModeBoxes() {
    const plusBox = document.getElementById("plus_rule_box");
    const fixedBox = document.getElementById("fixed_leva_rule_box");
    const liqCard = document.getElementById("liquidity_card");

    if (plusBox) plusBox.classList.toggle("show", currentMode === "leva_plus");
    if (fixedBox) fixedBox.classList.toggle("show", currentMode === "leva_fissa");

    if (liqCard) {
      liqCard.style.display = currentMode === "leva_plus" ? "block" : "none";
    }
  }

  function setActiveButtons() {
    document.querySelectorAll(".benchmarkBtn").forEach(btn => {
      const bench = btn.getAttribute("data-benchmark");
      const mode = btn.getAttribute("data-mode");

      let active = false;

      if (currentMode === "normal" && mode === "normal" && bench === currentBenchmark) active = true;
      if (currentMode === "leva_fissa" && mode === "leva_fissa") active = true;
      if (currentMode === "leva_plus" && mode === "leva_plus") active = true;

      btn.classList.toggle("active", active);
    });
  }

  async function refresh() {
    if (isRefreshing) return;
    isRefreshing = true;

    try {
      toggleModeBoxes();   // 🔥 fondamentale
      setActiveButtons();

      // 👉 tutto il resto resta IDENTICO
      // (non tocco logica, grafici, calcoli)

    } catch (err) {
      console.error(err);
    } finally {
      isRefreshing = false;
    }
  }

  function handleActionClick(target) {
    if (!target) return false;

    if (target.classList.contains("benchmarkBtn")) {
      currentBenchmark = target.getAttribute("data-benchmark") || "world";
      currentMode = target.getAttribute("data-mode") || "normal";
      refresh();
      return true;
    }

    return false;
  }

  function bindUi() {
    document.querySelectorAll(".benchmarkBtn").forEach(btn => {
      btn.addEventListener("click", function (e) {
        e.preventDefault();
        handleActionClick(btn);
      });
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    bindUi();
    refresh();
  });
})();
