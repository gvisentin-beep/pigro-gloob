(function () {
  let mainChart = null;
  let ddChart = null;
  let liqChart = null;
  let currentBenchmark = "world";
  let currentMode = "normal"; // normal | leva_fissa | leva_plus
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
    mib: "iShares FTSE MIB UCITS ETF",
    sp500: "iShares Core S&P 500 UCITS ETF (Acc)"
  };

  const CSV_PATHS = {
    ls80: "/static/data_ls80.csv",
    gold: "/static/data_gold.csv",
    btc: "/static/data_btc.csv",
    world: "/static/data_world.csv",
    mib: "/static/data_mib.csv",
    sp500: "/static/data_sp500.csv"
  };

  const COLOR_PIGRO = "#2b6cb0";
  const COLOR_BENCH = "#9aa0a6";
  const COLOR_MARKER = "#d97706";

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

  function normalizeTo100(series) {
    if (!Array.isArray(series) || !series.length) return [];
    const base = Number(series[0]);
    if (!isFinite(base) || base === 0) return series.map(() => null);
    return series.map(v => {
      const n = Number(v);
      return isFinite(n) ? (n / base) * 100 : null;
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

  function parseDateFlexible(value) {
    if (!value) return null;
    const s = String(value).trim();

    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) {
      const d = new Date(s + "T00:00:00");
      return isNaN(d.getTime()) ? null : d;
    }

    if (/^\d{2}\/\d{2}\/\d{4}$/.test(s)) {
      const [dd, mm, yyyy] = s.split("/");
      const d = new Date(`${yyyy}-${mm}-${dd}T00:00:00`);
      return isNaN(d.getTime()) ? null : d;
    }

    const d = new Date(s);
    return isNaN(d.getTime()) ? null : d;
  }

  function toIsoDate(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }

  function formatDateIt(isoDate) {
    if (!isoDate || typeof isoDate !== "string") return "—";
    const parts = isoDate.split("-");
    if (parts.length !== 3) return isoDate;
    return `${parts[2]}-${parts[1]}-${parts[0]}`;
  }

  function parseCsv(text) {
    const lines = String(text || "")
      .replace(/^\uFEFF/, "")
      .split(/\r?\n/)
      .map(x => x.trim())
      .filter(Boolean);

    if (!lines.length) return [];

    let startIndex = 0;
    const first = lines[0].toLowerCase();
    if (first.includes("date") || first.includes("data")) startIndex = 1;

    const rows = [];

    for (let i = startIndex; i < lines.length; i++) {
      const line = lines[i];
      const parts = line.includes(";") ? line.split(";") : line.split(",");
      if (parts.length < 2) continue;

      const d = parseDateFlexible(parts[0]);
      let raw = String(parts[1]).trim().replace(/\s/g, "");

      if (raw.includes(",") && !raw.includes(".")) {
        raw = raw.replace(",", ".");
      } else if (raw.includes(",") && raw.includes(".")) {
        raw = raw.replace(/\./g, "").replace(",", ".");
      }

      const v = Number(raw);
      if (!d || !isFinite(v)) continue;

      rows.push({
        date: toIsoDate(d),
        value: v
      });
    }

    rows.sort((a, b) => a.date.localeCompare(b.date));

    const seen = new Map();
    rows.forEach(r => seen.set(r.date, r.value));

    return Array.from(seen.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([date, value]) => ({ date, value }));
  }

  async function fetchText(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status} su ${url}`);
    return await res.text();
  }

  async function loadAllSeries() {
    const names = Object.keys(CSV_PATHS);
    const entries = await Promise.all(
      names.map(async name => {
        const txt = await fetchText(CSV_PATHS[name] + `?v=${Date.now()}`);
        return [name, parseCsv(txt)];
      })
    );
    return Object.fromEntries(entries);
  }

  function alignSeries(seriesMap) {
    const names = Object.keys(seriesMap);
    const allDates = new Set();

    names.forEach(name => {
      seriesMap[name].forEach(r => allDates.add(r.date));
    });

    const dates = Array.from(allDates).sort();
    const byName = {};

    names.forEach(name => {
      const m = new Map(seriesMap[name].map(r => [r.date, r.value]));
      let last = null;
      byName[name] = dates.map(d => {
        const cur = m.has(d) ? m.get(d) : null;
        if (cur !== null && isFinite(cur)) last = cur;
        return last;
      });
    });

    const validIdx = [];
    for (let i = 0; i < dates.length; i++) {
      if (
        byName.ls80[i] != null &&
        byName.gold[i] != null &&
        byName.btc[i] != null &&
        byName.world[i] != null
      ) {
        validIdx.push(i);
      }
    }

    return {
      dates: validIdx.map(i => dates[i]),
      ls80: validIdx.map(i => byName.ls80[i]),
      gold: validIdx.map(i => byName.gold[i]),
      btc: validIdx.map(i => byName.btc[i]),
      world: validIdx.map(i => byName.world[i]),
      mib: validIdx.map(i => (byName.mib ? byName.mib[i] : null)),
      sp500: validIdx.map(i => (byName.sp500 ? byName.sp500[i] : null))
    };
  }

  function rebalancePortfolio(dates, ls80, gold, btc, capital) {
    let unitsLs80 = (capital * WEIGHT_LS80) / ls80[0];
    let unitsGold = (capital * WEIGHT_GOLD) / gold[0];
    let unitsBtc = (capital * WEIGHT_BTC) / btc[0];

    const values = [];
    const yearsSeen = new Set();

    for (let i = 0; i < dates.length; i++) {
      const v = unitsLs80 * ls80[i] + unitsGold * gold[i] + unitsBtc * btc[i];
      values.push(v);

      const year = dates[i].slice(0, 4);
      if (!yearsSeen.has(year)) {
        yearsSeen.add(year);
        if (i > 0) {
          const total = v;
          unitsLs80 = (total * WEIGHT_LS80) / ls80[i];
          unitsGold = (total * WEIGHT_GOLD) / gold[i];
          unitsBtc = (total * WEIGHT_BTC) / btc[i];
        }
      }
    }

    return values;
  }

  function benchmarkSeries(aligned, benchKey, capital) {
    const arr = aligned[benchKey] || aligned.world;
    const firstValid = arr.find(v => v != null && isFinite(v));
    return arr.map(v =>
      v != null && isFinite(v) && firstValid > 0 ? (capital * v) / firstValid : null
    );
  }

  function computeDrawdownSeriesPct(series) {
    let peak = -Infinity;
    return series.map(v => {
      if (v > peak) peak = v;
      return ((v / peak) - 1) * 100;
    });
  }

  function computeMaxDD(series) {
    return Math.min(...computeDrawdownSeriesPct(series)) / 100;
  }

  function computeCagr(series, dates) {
    if (series.length < 2) return 0;

    const d0 = parseDateFlexible(dates[0]);
    const d1 = parseDateFlexible(dates[dates.length - 1]);
    const years = (d1 - d0) / (365.25 * 24 * 3600 * 1000);

    if (!(years > 0) || !(series[0] > 0) || !(series[series.length - 1] > 0)) return 0;

    return Math.pow(series[series.length - 1] / series[0], 1 / years) - 1;
  }

  function doublingYears(cagr) {
    if (!(cagr > 0)) return null;
    return Math.log(2) / Math.log(1 + cagr);
  }

  function worstDrawdowns(series, dates, n = 2) {
    let peak = series[0];
    const dd = series.map(v => {
      if (v > peak) peak = v;
      return (v / peak) - 1;
    });

    const events = [];
    let inDd = false;
    let startIdx = null;

    for (let i = 1; i < series.length; i++) {
      if (!inDd && dd[i] < 0) {
        startIdx = i - 1;
        inDd = true;
      }

      if (inDd && dd[i] === 0) {
        const sub = dd.slice(startIdx, i);
        if (sub.length > 0) {
          let localMin = 0;
          for (let k = 1; k < sub.length; k++) {
            if (sub[k] < sub[localMin]) localMin = k;
          }
          const bottom = startIdx + localMin;
          events.push({
            start: dates[startIdx],
            bottom: dates[bottom],
            end: dates[i],
            depth_pct: sub[localMin] * 100
          });
        }
        inDd = false;
        startIdx = null;
      }
    }

    if (inDd && startIdx != null) {
      const sub = dd.slice(startIdx);
      let localMin = 0;
      for (let k = 1; k < sub.length; k++) {
        if (sub[k] < sub[localMin]) localMin = k;
      }
      const bottom = startIdx + localMin;
      events.push({
        start: dates[startIdx],
        bottom: dates[bottom],
        end: null,
        depth_pct: sub[localMin] * 100
      });
    }

    return events.sort((a, b) => a.depth_pct - b.depth_pct).slice(0, n);
  }

  function buildDdSummary(firstEpisodes, secondEpisodes, secondLabel) {
    function fmt(rank, first, second) {
      const left = first
        ? `Pigro: <b>${pct(first.depth_pct, 2)}</b> (${formatDateIt(first.start)} → minimo ${formatDateIt(first.bottom)})`
        : `Pigro: —`;

      const right = second
        ? `${secondLabel}: <b>${pct(second.depth_pct, 2)}</b> (${formatDateIt(second.start)} → minimo ${formatDateIt(second.bottom)})`
        : `${secondLabel}: —`;

      return `<div style="margin-top:4px;"><b>${rank}ª peggiore discesa</b> — ${left} | ${right}</div>`;
    }

    return `
      <div><b>Confronto delle 2 peggiori discese complete</b></div>
      ${fmt(1, firstEpisodes[0], secondEpisodes[0])}
      ${fmt(2, firstEpisodes[1], secondEpisodes[1])}
    `;
  }

  function ensureLevaPlusCounter() {
    let counter = document.getElementById("leva_plus_counter");
    if (counter) return counter;

    const plusBox = document.getElementById("plus_rule_box");
    const summary = document.querySelector(".summary");

    counter = document.createElement("div");
    counter.id = "leva_plus_counter";
    counter.className = "dynamicRuleBox";
    counter.style.marginTop = "6px";

    if (plusBox) {
      plusBox.insertAdjacentElement("afterend", counter);
    } else if (summary && summary.parentNode) {
      summary.parentNode.insertBefore(counter, summary);
    }

    return counter;
  }

  function updateLevaPlusCounter() {
    const counter = ensureLevaPlusCounter();
    if (!counter) return;

    if (currentMode === "leva_plus") {
      counter.innerHTML = `<b>Integrazioni effettuate:</b> ${levaPlusIntegrations}`;
      counter.classList.add("show");
    } else {
      counter.innerHTML = "";
      counter.classList.remove("show");
    }
  }

  function updateTextSummary(pigroSeries, secondSeries, labels, secondLabel) {
    const initialCapital = getCapital();

    const cagrBase = computeCagr(pigroSeries, labels);
    const ddBase = computeMaxDD(pigroSeries);
    const dblBase = doublingYears(cagrBase);

    const cagrSecond = computeCagr(secondSeries, labels);
    const ddSecond = computeMaxDD(secondSeries);
    const dblSecond = doublingYears(cagrSecond);

    setText("cagr", pct(cagrBase * 100, 1));
    setText("maxdd", pct(ddBase * 100, 1));
    setText("dbl", dblBase ? plain(dblBase, 1) : "—");
    setText("benchmark_summary", `Benchmark: ${secondLabel}`);

    setText("final_value", euro(pigroSeries[pigroSeries.length - 1], 0));

    const d0 = parseDateFlexible(labels[0]);
    const d1 = parseDateFlexible(labels[labels.length - 1]);
    const years = d0 && d1 ? (d1 - d0) / (365.25 * 24 * 3600 * 1000) : 0;
    setText("final_years", years > 0 ? plain(years, 1) : "—");

    setText("compare_period", `${euro(initialCapital, 0)} investiti all’inizio del periodo`);

    setText("compare_pigro", euro(pigroSeries[pigroSeries.length - 1], 0));
    setText("compare_pigro_cagr", pct(cagrBase * 100, 1));
    setText("compare_pigro_maxdd", pct(ddBase * 100, 1));
    setText("compare_pigro_dbl", dblBase ? plain(dblBase, 1) : "—");

    setText("compare_title_benchmark", secondLabel);
    setText("compare_benchmark", euro(secondSeries[secondSeries.length - 1], 0));
    setText("compare_benchmark_cagr", pct(cagrSecond * 100, 1));
    setText("compare_benchmark_maxdd", pct(ddSecond * 100, 1));
    setText("compare_benchmark_dbl", dblSecond ? plain(dblSecond, 1) : "—");

    const ep1 = worstDrawdowns(pigroSeries, labels, 2);
    const ep2 = worstDrawdowns(secondSeries, labels, 2);
    setHtml("dd_summary", buildDdSummary(ep1, ep2, secondLabel));
  }

  function computeFixedLeverageDetailed(dates, ls80, gold, btc, initialCapital) {
    let borrowed = initialCapital * 0.20;
    let cumCost = 0;
    let prevDate = null;

    let unitsLs80 = (initialCapital * WEIGHT_LS80 * 1.20) / ls80[0];
    let unitsGold = (initialCapital * WEIGHT_GOLD * 1.20) / gold[0];
    let unitsBtc = (initialCapital * WEIGHT_BTC * 1.20) / btc[0];

    const series = [];

    for (let i = 0; i < dates.length; i++) {
      const d = parseDateFlexible(dates[i]);

      let days = 0;
      if (prevDate) {
        days = Math.max(0, Math.round((d - prevDate) / (24 * 3600 * 1000)));
      }
      prevDate = d;

      cumCost += borrowed * LOMBARD_RATE * (days / 365.25);

      const gross =
        unitsLs80 * ls80[i] +
        unitsGold * gold[i] +
        unitsBtc * btc[i];

      const net = gross - borrowed - cumCost;
      series.push(net);

      const nextYear = i < dates.length - 1 ? dates[i + 1].slice(0, 4) : null;
      const currYear = dates[i].slice(0, 4);

      if (nextYear && nextYear !== currYear) {
        const equity = net;
        borrowed = Math.max(0, equity * 0.20);
        const grossTarget = equity + borrowed;

        unitsLs80 = (grossTarget * WEIGHT_LS80) / ls80[i];
        unitsGold = (grossTarget * WEIGHT_GOLD) / gold[i];
        unitsBtc = (grossTarget * WEIGHT_BTC) / btc[i];
      }
    }

    return series;
  }

  function computeLevaPlusDetailed(dates, ls80, gold, btc, baseSeries, initialCapital) {
    let borrowed = initialCapital * 0.20;
    let cumCost = 0;
    let prevDate = null;

    let coreUnitsLs80 = (initialCapital * WEIGHT_LS80 * 1.20) / ls80[0];
    let coreUnitsGold = (initialCapital * WEIGHT_GOLD * 1.20) / gold[0];
    let coreUnitsBtc = (initialCapital * WEIGHT_BTC * 1.20) / btc[0];

    let extraLs80Units = 0;
    let integrations = 0;
    let triggerArmed = true;

    const series = [];
    const liquidity = [];
    const markerIndices = [];

    for (let i = 0; i < dates.length; i++) {
      const d = parseDateFlexible(dates[i]);

      let days = 0;
      if (prevDate) {
        days = Math.max(0, Math.round((d - prevDate) / (24 * 3600 * 1000)));
      }
      prevDate = d;

      cumCost += borrowed * LOMBARD_RATE * (days / 365.25);

      const coreGross =
        coreUnitsLs80 * ls80[i] +
        coreUnitsGold * gold[i] +
        coreUnitsBtc * btc[i];

      const extraGross = extraLs80Units * ls80[i];
      const gross = coreGross + extraGross;
      const net = gross - borrowed - cumCost;

      series.push(net);
      liquidity.push(Math.max(0, baseSeries[i] * LOMBARD_LTV - borrowed));

      const collateralUnderThreshold = baseSeries[i] <= initialCapital * 0.90;

      if (collateralUnderThreshold && triggerArmed && integrations < 2) {
        const extraBorrowed = initialCapital * 0.20;
        borrowed += extraBorrowed;
        extraLs80Units += extraBorrowed / ls80[i];
        integrations += 1;
        markerIndices.push(i);
        triggerArmed = false;
      }

      if (!collateralUnderThreshold) {
        triggerArmed = true;
      }

      const nextYear = i < dates.length - 1 ? dates[i + 1].slice(0, 4) : null;
      const currYear = dates[i].slice(0, 4);

      if (nextYear && nextYear !== currYear) {
        const coreGrossCurrent =
          coreUnitsLs80 * ls80[i] +
          coreUnitsGold * gold[i] +
          coreUnitsBtc * btc[i];

        coreUnitsLs80 = (coreGrossCurrent * WEIGHT_LS80) / ls80[i];
        coreUnitsGold = (coreGrossCurrent * WEIGHT_GOLD) / gold[i];
        coreUnitsBtc = (coreGrossCurrent * WEIGHT_BTC) / btc[i];
      }
    }

    return { series, liquidity, markerIndices, integrations };
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

  function commonChartOptions() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      elements: {
        line: { tension: 0.14, borderWidth: 2 },
        point: { radius: 0, hoverRadius: 3 }
      },
      plugins: {
        legend: {
          display: true,
          labels: {
            usePointStyle: true,
            boxWidth: 10,
            padding: 16,
            font: { size: 12, weight: "600" }
          }
        }
      }
    };
  }

  function buildMarkerDataset(labels, secondVals) {
    if (currentMode !== "leva_plus" || !levaPlusMarkerIndices.length) {
      return null;
    }

    const normalizedSecond = normalizeTo100(secondVals);
    const markerData = labels.map(() => null);

    levaPlusMarkerIndices.forEach(idx => {
      if (idx >= 0 && idx < normalizedSecond.length) {
        markerData[idx] = normalizedSecond[idx];
      }
    });

    return {
      label: "Integrazione Leva+",
      data: markerData,
      type: "line",
      showLine: false,
      borderColor: COLOR_MARKER,
      backgroundColor: COLOR_MARKER,
      pointRadius: 5,
      pointHoverRadius: 7,
      pointStyle: "triangle",
      pointRotation: 180
    };
  }

  function renderMain(labels, firstVals, secondVals, secondLabel) {
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;
    const keepTicks = yearTickIndices(labels);

    if (mainChart) {
      mainChart.destroy();
      mainChart = null;
    }

    const firstPlot = (currentMode === "normal") ? firstVals : normalizeTo100(firstVals);
    const secondPlot = (currentMode === "normal") ? secondVals : normalizeTo100(secondVals);

    const datasets = [
      {
        label: "Metodo Pigro 80/15/5",
        data: firstPlot,
        borderColor: COLOR_PIGRO,
        backgroundColor: COLOR_PIGRO
      },
      {
        label: secondLabel,
        data: secondPlot,
        borderColor: COLOR_BENCH,
        backgroundColor: COLOR_BENCH
      }
    ];

    const markerDataset = buildMarkerDataset(labels, secondVals);
    if (markerDataset) datasets.push(markerDataset);

    mainChart = new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      options: {
        ...commonChartOptions(),
        plugins: {
          ...commonChartOptions().plugins,
          tooltip: {
            callbacks: {
              title: items => (items && items.length ? formatDateIt(items[0].label) : ""),
              label: ctx => {
                if (ctx.dataset.label === "Integrazione Leva+") return "Integrazione Leva+";
                if (currentMode === "normal") return `${ctx.dataset.label}: ${euro(ctx.parsed.y, 0)}`;
                return `${ctx.dataset.label}: ${plain(ctx.parsed.y, 1)} (base 100)`;
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            afterBuildTicks(axis) {
              axis.ticks = axis.ticks.filter(t => keepTicks.has(t.value));
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback(value) {
                const lbl = this.getLabelForValue(value);
                return String(lbl || "").slice(0, 4);
              }
            }
          },
          y: {
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: {
              callback: value => currentMode === "normal" ? euro(value, 0) : plain(value, 0)
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

    if (ddChart) {
      ddChart.destroy();
      ddChart = null;
    }

    ddChart = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Drawdown Portafoglio Pigro",
            data: ddFirstVals,
            borderColor: COLOR_PIGRO,
            backgroundColor: COLOR_PIGRO
          },
          {
            label: `Drawdown ${secondLabel}`,
            data: ddSecondVals,
            borderColor: COLOR_BENCH,
            backgroundColor: COLOR_BENCH
          }
        ]
      },
      options: {
        ...commonChartOptions(),
        plugins: {
          ...commonChartOptions().plugins,
          tooltip: {
            callbacks: {
              title: items => (items && items.length ? formatDateIt(items[0].label) : ""),
              label: ctx => `${ctx.dataset.label}: ${pct(ctx.parsed.y, 2)}`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            afterBuildTicks(axis) {
              axis.ticks = axis.ticks.filter(t => keepTicks.has(t.value));
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback(value) {
                const lbl = this.getLabelForValue(value);
                return String(lbl || "").slice(0, 4);
              }
            }
          },
          y: {
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: {
              callback: value => pct(value, 0)
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

    if (liqChart) {
      liqChart.destroy();
      liqChart = null;
    }

    liqChart = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Disponibilità Lombard residua",
            data: liquidityVals,
            borderColor: COLOR_PIGRO,
            backgroundColor: COLOR_PIGRO
          }
        ]
      },
      options: {
        ...commonChartOptions(),
        plugins: {
          ...commonChartOptions().plugins,
          tooltip: {
            callbacks: {
              title: items => (items && items.length ? formatDateIt(items[0].label) : ""),
              label: ctx => `${ctx.dataset.label}: ${euro(ctx.parsed.y, 0)}`
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            afterBuildTicks(axis) {
              axis.ticks = axis.ticks.filter(t => keepTicks.has(t.value));
            },
            ticks: {
              autoSkip: false,
              maxRotation: 0,
              minRotation: 0,
              callback(value) {
                const lbl = this.getLabelForValue(value);
                return String(lbl || "").slice(0, 4);
              }
            }
          },
          y: {
            grid: { color: "rgba(0,0,0,0.06)" },
            ticks: {
              callback: value => euro(value, 0)
            }
          }
        }
      }
    });
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

  function openAdvisorModal() {
    const modal = document.getElementById("advisor_modal");
    if (!modal) return;
    modal.classList.add("show");
    modal.setAttribute("aria-hidden", "false");
  }

  function closeAdvisorModal() {
    const modal = document.getElementById("advisor_modal");
    if (!modal) return;
    modal.classList.remove("show");
    modal.setAttribute("aria-hidden", "true");
  }

  function toggleModeBoxes() {
    const plusBox = document.getElementById("plus_rule_box");
    const leva20Box = document.getElementById("leva20_rule_box");
    const liqCard = document.getElementById("liquidity_card");

    if (plusBox) plusBox.classList.toggle("show", currentMode === "leva_plus");
    if (leva20Box) leva20Box.classList.toggle("show", currentMode === "leva_fissa");
    if (liqCard) liqCard.style.display = currentMode === "leva_plus" ? "block" : "none";
  }

  async function refresh() {
    if (isRefreshing) return;
    isRefreshing = true;

    try {
      normalizeCapitalInput();
      toggleModeBoxes();
      setActiveButtons();
      destroyCharts();
      levaPlusIntegrations = 0;
      levaPlusMarkerIndices = [];
      updateLevaPlusCounter();

      const capital = getCapital();
      const seriesMap = await loadAllSeries();
      const aligned = alignSeries(seriesMap);
      const labels = aligned.dates;

      if (!labels.length) {
        throw new Error("Nessun dato disponibile nei CSV.");
      }

      const pigroSeries = rebalancePortfolio(labels, aligned.ls80, aligned.gold, aligned.btc, capital);

      let secondSeries = [];
      let secondLabel = BENCHMARK_LABELS[currentBenchmark] || BENCHMARK_LABELS.world;

      if (currentMode === "normal") {
        secondSeries = benchmarkSeries(aligned, currentBenchmark, capital);
      } else if (currentMode === "leva_fissa") {
        secondSeries = computeFixedLeverageDetailed(labels, aligned.ls80, aligned.gold, aligned.btc, capital);
        secondLabel = "Pigro con leva 20%";
      } else {
        const lp = computeLevaPlusDetailed(labels, aligned.ls80, aligned.gold, aligned.btc, pigroSeries, capital);
        secondSeries = lp.series;
        secondLabel = "Pigro Leva+";
        levaPlusIntegrations = lp.integrations || 0;
        levaPlusMarkerIndices = Array.isArray(lp.markerIndices) ? lp.markerIndices : [];

        renderLiquidity(labels, lp.liquidity);
        setHtml(
          "liquidity_summary",
          `Disponibilità Lombard residua finale: <b>${euro(lp.liquidity[lp.liquidity.length - 1], 0)}</b>`
        );
      }

      updateLevaPlusCounter();
      renderMain(labels, pigroSeries, secondSeries, secondLabel);
      renderDd(labels, computeDrawdownSeriesPct(pigroSeries), computeDrawdownSeriesPct(secondSeries), secondLabel);
      updateTextSummary(pigroSeries, secondSeries, labels, secondLabel);

    } catch (err) {
      console.error(err);
      setHtml("compare_box", `<strong>Errore:</strong> ${err.message}`);
    } finally {
      isRefreshing = false;
    }
  }

  function handleActionClick(target) {
    if (!target) return false;

    const id = target.id || "";

    if (id === "btn_update") {
      refresh();
      return true;
    }

    if (id === "btn_pdf") {
      window.print();
      return true;
    }

    if (id === "btn_faxsimile") {
      window.open("/static/faxsimile_execution_only.pdf", "_blank");
      return true;
    }

    if (id === "btn_consulente") {
      openAdvisorModal();
      return true;
    }

    if (id === "advisor_modal_close") {
      closeAdvisorModal();
      return true;
    }

    if (id === "btn_libro") {
      window.open("https://www.amazon.it/dp/B0GQM925QR", "_blank");
      return true;
    }

    if (target.classList && target.classList.contains("benchmarkBtn")) {
      currentBenchmark = target.getAttribute("data-benchmark") || "world";
      currentMode = target.getAttribute("data-mode") || "normal";
      refresh();
      return true;
    }

    return false;
  }

  function bindUi() {
    const capitalInput = document.getElementById("capital");
    if (capitalInput) {
      capitalInput.addEventListener("input", function () {
        this.value = formatIntegerInput(this.value);
      });
    }

    [
      "btn_update",
      "btn_pdf",
      "btn_faxsimile",
      "btn_consulente",
      "btn_libro",
      "advisor_modal_close"
    ].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("click", function (e) {
          e.preventDefault();
          e.stopPropagation();
          handleActionClick(el);
        });
      }
    });

    document.querySelectorAll(".benchmarkBtn").forEach(btn => {
      btn.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        handleActionClick(btn);
      });
    });

    document.addEventListener("click", function (e) {
      const clickable = e.target.closest(
        "#btn_update, #btn_pdf, #btn_faxsimile, #btn_consulente, #btn_libro, #advisor_modal_close, .benchmarkBtn"
      );

      if (clickable) {
        e.preventDefault();
        handleActionClick(clickable);
        return;
      }

      const modal = document.getElementById("advisor_modal");
      if (modal && e.target === modal) {
        closeAdvisorModal();
      }
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") {
        closeAdvisorModal();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    bindUi();
    refresh();
  });
})();
