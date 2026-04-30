(function () {
  let mainChart = null;
  let ddChart = null;
  let liqChart = null;
  let currentBenchmark = "world";
  let currentMode = "normal";
  let levaPlusIntegrations = 0;
  let levaPlusMarkerIndices = [];
  let isRefreshing = false;
  let LANG = localStorage.getItem("lang") || "it";

  const WEIGHT_LS80 = 0.80;
  const WEIGHT_GOLD = 0.15;
  const WEIGHT_BTC = 0.05;
  const LOMBARD_RATE = 0.025;
  const LOMBARD_LTV = 0.60;

  const CSV_PATHS = {
    ls80: "/static/data_ls80.csv",
    gold: "/static/data_gold.csv",
    btc: "/static/data_btc.csv",
    world: "/static/data_world.csv",
    mib: "/static/data_mib.csv",
    sp500: "/static/data_sp500.csv",
    ai_core: "/static/data_ai_core.csv"
  };

  const COLOR_PIGRO = "#2b6cb0";
  const COLOR_BENCH = "#9aa0a6";
  const COLOR_MARKER = "#d97706";
const COLOR_AI = "#16a34a";
const COLOR_COMBO = "#7c3aed";
  
  const TEXT = {
    it: {
      locale: "it-IT",
      title: "Metodo Pigro — Variante 80/15/5",
      subtitle: "Tre strumenti globali. Nessuna previsione. Solo disciplina.",
      bullets: [
        "<b>Pochi strumenti</b>: LifeStrategy 80, Oro, Bitcoin",
        "<b>Controllo del rischio</b>: pesi fissi e struttura semplice",
        "<b>Regole chiare</b>: pesi senza interventi continui",
        "<b>Struttura &gt; Previsioni</b>"
      ],
      allocTitle: "Composizione fissa",
      gold: "Oro",
      capitalLabel: "Capitale iniziale (€)",
      update: "Aggiorna",
      final: "Finale",
      inYears: "in anni",
      choose: "Scegli il confronto da visualizzare.",
      btnEurope: "Confronta con Europa",
      btnUsa: "Confronta con USA S&P 500",
      btnWorld: "Confronta con MSCI World",
      btnLeva20: "Pigro con leva 20%",
      btnLevaPlus: "Pigro Leva+",
      plusRule: "<b>Pigro Leva+:</b> parte con leva iniziale del 20%. Se il portafoglio principale dato a garanzia scende sotto il 90% del capitale iniziale, viene effettuata un’integrazione di 20% del capitale solo su LS80. Il segnale può attivarsi al massimo 2 volte, solo dopo un recupero sopra soglia e una successiva nuova violazione al ribasso. Il costo del Lombard è calcolato in media al 2,5%.",
      portfolio: "Portafoglio",
      pigroName: "Portafoglio Pigro",
      pigroFull: "Portafoglio “Pigro 80/15/5”",
      annualReturn: "Rendimento annualizzato",
      annualReturnShort: "Rendimento annuo",
      maxDrawdown: "Max Ribasso",
      maxDrawdownPeriod: "Max Ribasso nel periodo",
      composition: "Composizione",
      doubleYears: "Raddoppio del portafoglio in anni",
      benchmark: "Benchmark",
      chartTitle: "Andamento Portafogli negli ultimi anni",
      ddTitle: "Andamento Ribassi Portafogli",
      ddPigro: "Drawdown Portafoglio Pigro",
      ddOf: "Drawdown",
      worstTitle: "Confronto delle 2 peggiori discese complete",
      worst1: "1ª peggiore discesa",
      worst2: "2ª peggiore discesa",
      minimum: "minimo",
      immediate: "Confronto immediato",
      investedStart: "investiti all’inizio del periodo",
      capitalDouble: "Raddoppio capitale",
      completeComparison: "Confronto completo portafogli",
      periodCalc: "Periodo di calcolo",
      finalCapital: "Capitale finale",
      msgKey: "<b>Messaggio chiave:</b><br/>La differenza non è indovinare il mercato.<br/>È avere una struttura semplice e mantenerla nel tempo.",
      howTitle: "Come applicarlo concretamente",
      howList: [
        "Investi il capitale in 3 strumenti: 80% LS80, 15% Oro, 5% Bitcoin.",
        "Evita interventi continui.",
        "Confronta il metodo con Europa, USA S&P 500, MSCI World, leva 20% fissa o Pigro Leva+ con riserva Lombard.",
        "Osserva insieme rendimento e drawdown."
      ],
      btnFax: "Facsimile ordine Banca",
      btnAdvisor: "Richiedi Consulente",
      btnBook: "Per saperne di più",
      btnExperts: "Per Consulenti / Esperti",
      btnMission: "Scopri la Mission",
      missionTitle: "MISSION",
      missionHtml: `
        <h2>MISSION</h2>
        <div class="missionBox">
          <p>In Italia una parte enorme del risparmio viene ancora investita in modo inefficiente: costi elevati, prodotti poco trasparenti e scelte spesso guidate più dalla distribuzione che dall’interesse del risparmiatore.</p>
          <p>Gloob nasce da un’idea semplice: aiutare le persone a capire che un portafoglio costruito bene può essere più chiaro, più efficiente e più coerente con obiettivi di lungo periodo.</p>
          <p>Un risparmio investito meglio non aiuta solo il singolo investitore. Può contribuire anche a indirizzare capitali verso strumenti solidi, mercati produttivi e imprese capaci di creare valore.</p>
          <p>Questa pagina, completamente gratuita, mette a disposizione strumenti facili di analisi, confronto e simulazione. L’obiettivo è offrire un punto di riferimento accessibile, dove il risparmiatore possa orientarsi in autonomia oppure, se lo desidera, confrontarsi con consulenti autorizzati, senza conflitti di interesse.</p>
          <p class="missionStrong">Il risparmio non dovrebbe essere eroso da costi nascosti: dovrebbe restare il più possibile nelle mani di chi lo ha costruito.</p>
        </div>
      `,
      advisorsTitle: "Consulenti indipendenti",
      advisorsSub: "Professionisti da contattare direttamente per eventuale consulenza fee-only.",
      advisorNote: `Si consiglia di verificare sempre l’iscrizione all’Albo sul sito ufficiale OCF:
        <a href="https://www.organismocf.it" target="_blank" rel="noopener">www.organismocf.it</a>.
        <br><br>
        Questo sito non ha alcun rapporto di commissione, retrocessione o collaborazione commerciale con i professionisti sopra indicati. L’eventuale contatto avviene in modo diretto e autonomo da parte dell’utente.`,
      integrations: "Integrazioni effettuate",
      residualLiquidity: "Disponibilità residua Lombard",
      residualLiquidityFinal: "Disponibilità Lombard residua finale",
      noData: "Nessun dato disponibile nei CSV.",
      error: "Errore"
    },

    en: {
      locale: "en-US",
      title: "Lazy Portfolio — 80/15/5 Variant",
      subtitle: "Three global assets. No forecasts. Just discipline.",
      bullets: [
        "<b>Few instruments</b>: LifeStrategy 80, Gold, Bitcoin",
        "<b>Risk control</b>: fixed weights and a simple structure",
        "<b>Clear rules</b>: weights without continuous intervention",
        "<b>Structure &gt; Forecasts</b>"
      ],
      allocTitle: "Fixed allocation",
      gold: "Gold",
      capitalLabel: "Initial capital (€)",
      update: "Update",
      final: "Final",
      inYears: "in years",
      choose: "Choose the comparison to display.",
      btnEurope: "Compare with Europe",
      btnUsa: "Compare with USA S&P 500",
      btnWorld: "Compare with MSCI World",
      btnLeva20: "Lazy with 20% leverage",
      btnLevaPlus: "Lazy Leverage+",
      plusRule: "<b>Lazy Leverage+:</b> starts with 20% initial leverage. If the main pledged portfolio falls below 90% of the initial capital, an additional 20% of capital is invested only in LS80. The signal can be triggered at most twice, only after a recovery above the threshold and a subsequent new downside breach. The Lombard financing cost is assumed at an average 2.5%.",
      portfolio: "Portfolio",
      pigroName: "Lazy Portfolio",
      pigroFull: "“Lazy 80/15/5” Portfolio",
      annualReturn: "Annualized return",
      annualReturnShort: "Annual return",
      maxDrawdown: "Max Drawdown",
      maxDrawdownPeriod: "Max Drawdown over the period",
      composition: "Allocation",
      doubleYears: "Portfolio doubling time in years",
      benchmark: "Benchmark",
      chartTitle: "Portfolio Performance in Recent Years",
      ddTitle: "Portfolio Drawdowns",
      ddPigro: "Lazy Portfolio Drawdown",
      ddOf: "Drawdown",
      worstTitle: "Comparison of the 2 worst complete drawdowns",
      worst1: "Worst drawdown",
      worst2: "2nd worst drawdown",
      minimum: "low",
      immediate: "Immediate comparison",
      investedStart: "invested at the beginning of the period",
      capitalDouble: "Capital doubling",
      completeComparison: "Complete portfolio comparison",
      periodCalc: "Calculation period",
      finalCapital: "Final capital",
      msgKey: "<b>Key message:</b><br/>The difference is not predicting the market.<br/>It is having a simple structure and sticking to it over time.",
      howTitle: "How to apply it concretely",
      howList: [
        "Invest the capital in 3 instruments: 80% LS80, 15% Gold, 5% Bitcoin.",
        "Avoid continuous intervention.",
        "Compare the method with Europe, USA S&P 500, MSCI World, fixed 20% leverage or Lazy Leverage+ with Lombard reserve.",
        "Observe return and drawdown together."
      ],
      btnFax: "Bank order template",
      btnAdvisor: "Request an Advisor",
      btnBook: "Learn more",
      btnExperts: "For Advisors / Experts",
      btnMission: "Discover the Mission",
      missionTitle: "MISSION",
      missionHtml: `
        <h2>MISSION</h2>
        <div class="missionBox">
          <p>In Italy, a very large share of savings is still invested inefficiently: high costs, opaque products and choices often driven more by distribution than by the saver’s interest.</p>
          <p>Gloob was born from a simple idea: helping people understand that a well-built portfolio can be clearer, more efficient and more consistent with long-term goals.</p>
          <p>Better-invested savings do not only help the individual investor. They can also help direct capital toward solid instruments, productive markets and companies capable of creating value.</p>
          <p>This page, completely free of charge, provides simple tools for analysis, comparison and simulation. The goal is to offer an accessible reference point where savers can orient themselves independently or, if they wish, speak with authorized advisors, without conflicts of interest.</p>
          <p class="missionStrong">Savings should not be eroded by hidden costs: they should remain as much as possible in the hands of those who built them.</p>
        </div>
      `,
      advisorsTitle: "Independent advisors",
      advisorsSub: "Professionals to contact directly for possible fee-only advice.",
      advisorNote: `Always verify registration on the official OCF website:
        <a href="https://www.organismocf.it" target="_blank" rel="noopener">www.organismocf.it</a>.
        <br><br>
        This website has no commission, rebate or commercial collaboration relationship with the professionals listed above. Any contact takes place directly and independently by the user.`,
      integrations: "Integrations made",
      residualLiquidity: "Residual Lombard availability",
      residualLiquidityFinal: "Final residual Lombard availability",
      noData: "No data available in CSV files.",
      error: "Error"
    }
  };

  function tr(key) {
    return TEXT[LANG][key] || TEXT.it[key] || key;
  }

  function locale() {
    return TEXT[LANG].locale || "it-IT";
  }
function getBenchmarkLabel(key) {
  if (key === "world") return "MSCI World";
  if (key === "mib") return "Euro Stoxx 50";
  if (key === "sp500") return LANG === "en" ? "USA S&P 500" : "USA S&P 500";
  if (key === "ai_core") return "Evoluto (Pigro + AI)";
   return key;
}
  
   function euro(value, digits = 0) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString(locale(), {
      style: "currency",
      currency: "EUR",
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function pct(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString(locale(), {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    }) + "%";
  }

  function plain(value, digits = 2) {
    const n = Number(value);
    if (!isFinite(n)) return "—";
    return n.toLocaleString(locale(), {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });
  }

  function ratingRendimento(cagr) {
    const r = Number(cagr) * 100;
    if (!isFinite(r)) return 0;
    if (r < 0) return 0;
    if (r <= 4) return 1;
    if (r <= 8) return 2;
    if (r <= 12) return 3;
    if (r <= 15) return 4;
    return 5;
  }

  function ratingRibasso(maxdd) {
    const d = Math.abs(Number(maxdd) * 100);
    if (!isFinite(d)) return 0;
    if (d <= 15) return 5;
    if (d <= 20) return 4;
    if (d <= 25) return 3;
    if (d <= 30) return 2;
    if (d <= 35) return 1;
    return 0;
  }

  function stelle(n) {
    const rating = Math.max(0, Math.min(5, Number(n) || 0));
    let out = "";
    for (let i = 0; i < 5; i++) out += i < rating ? "★" : "☆";
    return out;
  }

  function stelleHtml(n) {
    return `<span style="color:#f5b301; font-size:1.05em; letter-spacing:1px; margin-left:6px; white-space:nowrap;">${stelle(n)}</span>`;
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
    return Number(digits).toLocaleString("it-IT", { maximumFractionDigits: 0 });
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

      let parts;
if (line.includes(";")) {
  parts = line.split(";");
} else if (line.includes("\t")) {
  parts = line.split("\t");
} else {
  parts = line.trim().split(/\s+/);
}
      
          if (parts.length < 2) continue;

      const d = parseDateFlexible(parts[0]);
      let raw = String(parts[1]).trim().replace(/\s/g, "");

      if (raw.includes(",") && !raw.includes(".")) raw = raw.replace(",", ".");
      else if (raw.includes(",") && raw.includes(".")) raw = raw.replace(/\./g, "").replace(",", ".");

      const v = Number(raw);
      if (!d || !isFinite(v)) continue;

      rows.push({ date: toIsoDate(d), value: v });
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

  function forwardFillOnBaseDates(baseDates, sourceSeries) {
    const sourceMap = new Map((sourceSeries || []).map(r => [r.date, r.value]));
    const sourceDates = (sourceSeries || []).map(r => r.date);

    let ptr = 0;
    let last = null;
    const out = [];

    for (const d of baseDates) {
      while (ptr < sourceDates.length && sourceDates[ptr] <= d) {
        const curDate = sourceDates[ptr];
        const curValue = sourceMap.get(curDate);
        if (curValue != null && isFinite(curValue)) last = curValue;
        ptr++;
      }
      out.push(last);
    }

    return out;
  }

  function isWeekdayIso(isoDate) {
    const d = parseDateFlexible(isoDate);
    if (!d) return false;
    const day = d.getDay();
    return day >= 1 && day <= 5;
  }

  function alignSeries(seriesMap, benchmarkKey, mode) {
    const ls80Series = Array.isArray(seriesMap.ls80) ? seriesMap.ls80 : [];
    if (!ls80Series.length) {
      return { dates: [], ls80: [], gold: [], btc: [], world: [], mib: [], sp500: [], ai_core: [] };
         }

    let baseDates = ls80Series.map(r => r.date);
    if (mode === "normal") baseDates = baseDates.filter(isWeekdayIso);

    const alignedRaw = {
      ls80: forwardFillOnBaseDates(baseDates, seriesMap.ls80),
      gold: forwardFillOnBaseDates(baseDates, seriesMap.gold),
      btc: forwardFillOnBaseDates(baseDates, seriesMap.btc),
      world: forwardFillOnBaseDates(baseDates, seriesMap.world),
      mib: forwardFillOnBaseDates(baseDates, seriesMap.mib),
      sp500: forwardFillOnBaseDates(baseDates, seriesMap.sp500),
      ai_core: forwardFillOnBaseDates(baseDates, seriesMap.ai_core)
    };

    const needBenchmark = mode === "normal" ? (benchmarkKey || "world") : null;
    const validIdx = [];

    for (let i = 0; i < baseDates.length; i++) {
      const hasCore =
        alignedRaw.ls80[i] != null && isFinite(alignedRaw.ls80[i]) &&
        alignedRaw.gold[i] != null && isFinite(alignedRaw.gold[i]) &&
        alignedRaw.btc[i] != null && isFinite(alignedRaw.btc[i]);

      if (!hasCore) continue;

      if (needBenchmark) {
        const b = alignedRaw[needBenchmark];
        if (!b || b[i] == null || !isFinite(b[i])) continue;
      }

      validIdx.push(i);
    }

    return {
      dates: validIdx.map(i => baseDates[i]),
      ls80: validIdx.map(i => alignedRaw.ls80[i]),
      gold: validIdx.map(i => alignedRaw.gold[i]),
      btc: validIdx.map(i => alignedRaw.btc[i]),
      world: validIdx.map(i => alignedRaw.world[i]),
      mib: validIdx.map(i => alignedRaw.mib[i]),
      sp500: validIdx.map(i => alignedRaw.sp500[i]),
        ai_core: validIdx.map(i => alignedRaw.ai_core[i])
    };
  }

  function rebalancePortfolio(dates, ls80, gold, btc, capital) {
    let unitsLs80 = (capital * WEIGHT_LS80) / ls80[0];
    let unitsGold = (capital * WEIGHT_GOLD) / gold[0];
    let unitsBtc = (capital * WEIGHT_BTC) / btc[0];

    const values = [];
    let currentYear = dates[0] ? dates[0].slice(0, 4) : null;

    for (let i = 0; i < dates.length; i++) {
      const v = unitsLs80 * ls80[i] + unitsGold * gold[i] + unitsBtc * btc[i];
      values.push(v);

      const thisYear = dates[i].slice(0, 4);
      const nextYear = i < dates.length - 1 ? dates[i + 1].slice(0, 4) : null;

      if (thisYear !== currentYear) currentYear = thisYear;

      if (nextYear && nextYear !== thisYear) {
        const total = v;
        unitsLs80 = (total * WEIGHT_LS80) / ls80[i];
        unitsGold = (total * WEIGHT_GOLD) / gold[i];
        unitsBtc = (total * WEIGHT_BTC) / btc[i];
      }
    }

    return values;
  }

  function benchmarkSeries(aligned, benchKey, capital) {
    const arr = aligned[benchKey] || aligned.world;
    const firstValid = arr.find(v => v != null && isFinite(v));
    return arr.map(v => v != null && isFinite(v) && firstValid > 0 ? (capital * v) / firstValid : null);
  }

  function removeIsolatedSpikes(series, thresholdPct = 0.10) {
    if (!Array.isArray(series) || series.length < 3) return series;

    const cleaned = [...series];

    for (let i = 1; i < cleaned.length - 1; i++) {
      const prev = cleaned[i - 1];
      const curr = cleaned[i];
      const next = cleaned[i + 1];

      if (![prev, curr, next].every(v => v != null && isFinite(v) && v > 0)) continue;

      const movePrevCurr = (curr / prev) - 1;
      const moveCurrNext = (next / curr) - 1;
      const movePrevNext = Math.abs((next / prev) - 1);

      const isDownSpike = movePrevCurr < -thresholdPct && moveCurrNext > thresholdPct && movePrevNext < thresholdPct * 0.5;
      const isUpSpike = movePrevCurr > thresholdPct && moveCurrNext < -thresholdPct && movePrevNext < thresholdPct * 0.5;

      if (isDownSpike || isUpSpike) cleaned[i] = (prev + next) / 2;
    }

    return cleaned;
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

  function computePeriodYears(dates) {
    if (!Array.isArray(dates) || dates.length < 2) return 0;
    const d0 = parseDateFlexible(dates[0]);
    const d1 = parseDateFlexible(dates[dates.length - 1]);
    if (!d0 || !d1) return 0;
    return (d1 - d0) / (365.25 * 24 * 3600 * 1000);
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
          for (let k = 1; k < sub.length; k++) if (sub[k] < sub[localMin]) localMin = k;
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
      for (let k = 1; k < sub.length; k++) if (sub[k] < sub[localMin]) localMin = k;
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
    function fmt(rankLabel, first, second) {
      const left = first
        ? `${tr("pigroName")}: <b>${pct(first.depth_pct, 2)}</b> (${formatDateIt(first.start)} → ${tr("minimum")} ${formatDateIt(first.bottom)})`
        : `${tr("pigroName")}: —`;

      const right = second
        ? `${secondLabel}: <b>${pct(second.depth_pct, 2)}</b> (${formatDateIt(second.start)} → ${tr("minimum")} ${formatDateIt(second.bottom)})`
        : `${secondLabel}: —`;

      return `<div style="margin-top:4px;"><b>${rankLabel}</b> — ${left} | ${right}</div>`;
    }

    return `
      <div><b>${tr("worstTitle")}</b></div>
      ${fmt(tr("worst1"), firstEpisodes[0], secondEpisodes[0])}
      ${fmt(tr("worst2"), firstEpisodes[1], secondEpisodes[1])}
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

    if (plusBox) plusBox.insertAdjacentElement("afterend", counter);
    else if (summary && summary.parentNode) summary.parentNode.insertBefore(counter, summary);

    return counter;
  }

  function updateLevaPlusCounter() {
    const counter = ensureLevaPlusCounter();
    if (!counter) return;

    if (currentMode === "leva_plus") {
      counter.innerHTML = `<b>${tr("integrations")}:</b> ${levaPlusIntegrations}`;
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
    setText("benchmark_summary", `${tr("benchmark")}: ${secondLabel}`);
    setText("final_value", euro(pigroSeries[pigroSeries.length - 1], 0));

    const years = computePeriodYears(labels);
    setText("final_years", years > 0 ? plain(years, 1) : "—");

    setHtml("compare_box", `
      <strong>${tr("immediate")}</strong><br/>
      ${euro(initialCapital, 0)} ${tr("investedStart")}<br/>
      ${tr("pigroName")} → <b>${euro(pigroSeries[pigroSeries.length - 1], 0)}</b><br/>
      <span style="font-weight:400;">
        ${tr("annualReturn")}: <b>${pct(cagrBase * 100, 1)}</b> |
        ${tr("maxDrawdown")}: <b>${pct(ddBase * 100, 1)}</b> |
        ${tr("capitalDouble")}: <b>${dblBase ? plain(dblBase, 1) : "—"}</b> ${LANG === "en" ? "years" : "anni"}
      </span><br/>
      ${secondLabel} → <b>${euro(secondSeries[secondSeries.length - 1], 0)}</b><br/>
      <span style="font-weight:400;">
        ${tr("annualReturn")}: <b>${pct(cagrSecond * 100, 1)}</b> |
        ${tr("maxDrawdown")}: <b>${pct(ddSecond * 100, 1)}</b> |
        ${tr("capitalDouble")}: <b>${dblSecond ? plain(dblSecond, 1) : "—"}</b> ${LANG === "en" ? "years" : "anni"}
      </span>
    `);

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
      if (prevDate) days = Math.max(0, Math.round((d - prevDate) / (24 * 3600 * 1000)));
      prevDate = d;

      cumCost += borrowed * LOMBARD_RATE * (days / 365.25);

      const gross = unitsLs80 * ls80[i] + unitsGold * gold[i] + unitsBtc * btc[i];
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
      if (prevDate) days = Math.max(0, Math.round((d - prevDate) / (24 * 3600 * 1000)));
      prevDate = d;

      cumCost += borrowed * LOMBARD_RATE * (days / 365.25);

      const coreGross = coreUnitsLs80 * ls80[i] + coreUnitsGold * gold[i] + coreUnitsBtc * btc[i];
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

      if (!collateralUnderThreshold) triggerArmed = true;

      const nextYear = i < dates.length - 1 ? dates[i + 1].slice(0, 4) : null;
      const currYear = dates[i].slice(0, 4);

      if (nextYear && nextYear !== currYear) {
        const coreGrossCurrent = coreUnitsLs80 * ls80[i] + coreUnitsGold * gold[i] + coreUnitsBtc * btc[i];

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
    if (currentMode !== "leva_plus" || !levaPlusMarkerIndices.length) return null;

    const normalizedSecond = normalizeTo100(secondVals);
    const markerData = labels.map(() => null);

    levaPlusMarkerIndices.forEach(idx => {
      if (idx >= 0 && idx < normalizedSecond.length) markerData[idx] = normalizedSecond[idx];
    });

    return {
      label: LANG === "en" ? "Leverage+ integration" : "Integrazione Leva+",
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
function renderMain(labels, firstVals, secondVals, secondLabel, aiCoreSeries, comboSeries) {
    
    const canvas = document.getElementById("chart_main");
    if (!canvas) return;
    const keepTicks = yearTickIndices(labels);

    if (mainChart) {
      mainChart.destroy();
      mainChart = null;
    }

    const firstPlot = currentMode === "normal" ? firstVals : normalizeTo100(firstVals);
        const secondPlot = currentMode === "normal" ? secondVals : normalizeTo100(secondVals);

    const datasets = [
      {
        label: LANG === "en" ? "Lazy Method 80/15/5" : "Metodo Pigro 80/15/5",
        data: firstPlot,
        borderColor: COLOR_PIGRO,
        backgroundColor: COLOR_PIGRO
      },
      {
        label: secondLabel,
        data: secondPlot,
        borderColor: COLOR_BENCH,
        backgroundColor: COLOR_BENCH
      },

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
                if (ctx.dataset.label.includes("integration") || ctx.dataset.label.includes("Integrazione")) return ctx.dataset.label;
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
              axis.ticks = axis.ticks.filter(tick => keepTicks.has(tick.value));
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
            label: tr("ddPigro"),
            data: ddFirstVals,
            borderColor: COLOR_PIGRO,
            backgroundColor: COLOR_PIGRO
          },
          {
            label: `${tr("ddOf")} ${secondLabel}`,
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
              axis.ticks = axis.ticks.filter(tick => keepTicks.has(tick.value));
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
            ticks: { callback: value => pct(value, 0) }
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
            label: tr("residualLiquidity"),
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
              axis.ticks = axis.ticks.filter(tick => keepTicks.has(tick.value));
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
            ticks: { callback: value => euro(value, 0) }
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
  const liqCard = document.getElementById("liquidity_card");
  const leva20Box = document.getElementById("leva20_rule_box");
  const evolutoBox = document.getElementById("evoluto_rule_box");

  const showPlus = currentMode === "leva_plus";
  const showLeva20 = currentMode === "leva_fissa";
  const showEvoluto = currentMode === "normal" && currentBenchmark === "ai_core";

  if (plusBox) {
    plusBox.classList.toggle("show", showPlus);
    plusBox.style.display = showPlus ? "block" : "none";
  }

  if (leva20Box) {
    leva20Box.classList.toggle("show", showLeva20);
    leva20Box.style.display = showLeva20 ? "block" : "none";
  }

  if (evolutoBox) {
    evolutoBox.classList.toggle("show", showEvoluto);
    evolutoBox.style.display = showEvoluto ? "block" : "none";
  }

  if (liqCard) {
    liqCard.style.display = showPlus ? "block" : "none";
  }
}
   
   function buildComparisonTable(aligned, labels, capital) {
    function compute(series) {
      return {
        cagr: computeCagr(series, labels),
        dd: computeMaxDD(series),
        finalValue: Array.isArray(series) && series.length ? series[series.length - 1] : null
      };
    }

    const pigro = rebalancePortfolio(labels, aligned.ls80, aligned.gold, aligned.btc, capital);

    const world = removeIsolatedSpikes(benchmarkSeries(aligned, "world", capital), 0.10);
    const mib = removeIsolatedSpikes(benchmarkSeries(aligned, "mib", capital), 0.10);
    const sp500 = removeIsolatedSpikes(benchmarkSeries(aligned, "sp500", capital), 0.10);
    const evoluto = removeIsolatedSpikes(benchmarkSeries(aligned, "ai_core", capital), 0.10);
    const leva20 = computeFixedLeverageDetailed(labels, aligned.ls80, aligned.gold, aligned.btc, capital);
    const levaPlusObj = computeLevaPlusDetailed(labels, aligned.ls80, aligned.gold, aligned.btc, pigro, capital);
    const levaPlus = levaPlusObj.series;

const rows = [
  [tr("pigroName"), compute(pigro)],
  ["Euro Stoxx 50", compute(mib)],
  ["USA S&P 500", compute(sp500)],
  ["MSCI World", compute(world)],
  [tr("btnLeva20"), compute(leva20)],
  [tr("btnLevaPlus"), compute(levaPlus)],
  ["Evoluto (Pigro + AI)", compute(evoluto)]   // ← ULTIMO
];
    

    const tbody = document.querySelector("#comparison_table tbody");
    if (!tbody) return;

    const years = computePeriodYears(labels);
    setText("comparison_period_years", years > 0 ? plain(years, 1) : "—");

    tbody.innerHTML = "";

    rows.forEach(([name, stats]) => {
      const trEl = document.createElement("tr");
      trEl.innerHTML = `
        <td><b>${name}</b></td>
        <td>${pct(stats.cagr * 100, 1)} ${stelleHtml(ratingRendimento(stats.cagr))}</td>
        <td>${pct(stats.dd * 100, 1)} ${stelleHtml(ratingRibasso(stats.dd))}</td>
        <td>${euro(stats.finalValue, 0)}</td>
      `;
      tbody.appendChild(trEl);
    });
  }

  function applyStaticTranslations() {
    setHtml("title", tr("title"));
    setHtml("subtitle", tr("subtitle"));

    document.querySelectorAll(".introText ul li").forEach((el, i) => {
      if (TEXT[LANG].bullets[i]) el.innerHTML = TEXT[LANG].bullets[i];
    });

    const allocBox = document.querySelector(".allocBox");
    if (allocBox) {
      allocBox.innerHTML = `
        <b>${tr("allocTitle")}</b><br/>
        LifeStrategy 80: <b>80%</b><br/>
        ${tr("gold")}: <b>15%</b><br/>
        Bitcoin: <b>5%</b>
      `;
    }

    const capitalLabel = document.querySelector(".ctrl.capital label");
    if (capitalLabel) capitalLabel.innerHTML = `<b>${tr("capitalLabel")}</b>`;

    const btnUpdate = document.getElementById("btn_update");
    if (btnUpdate) btnUpdate.textContent = tr("update");

    const finalBox = document.querySelector(".finalBox");
    if (finalBox) {
      finalBox.innerHTML = `${tr("final")}: <span id="final_value">—</span> <span class="sub">(${tr("inYears")} <span id="final_years">—</span>)</span>`;
    }

    const smallTexts = document.querySelectorAll(".smallText");
    if (smallTexts[0]) smallTexts[0].textContent = tr("choose");

    document.querySelectorAll(".benchmarkBtn").forEach(btn => {
      const mode = btn.getAttribute("data-mode");
      const bench = btn.getAttribute("data-benchmark");

      if (mode === "normal" && bench === "mib") btn.textContent = tr("btnEurope");
      if (mode === "normal" && bench === "sp500") btn.textContent = tr("btnUsa");
      if (mode === "normal" && bench === "world") btn.textContent = tr("btnWorld");
      if (mode === "leva_fissa") btn.textContent = tr("btnLeva20");
      if (mode === "leva_plus") btn.textContent = tr("btnLevaPlus");
    });

    const plusRule = document.getElementById("plus_rule_box");
    if (plusRule) plusRule.innerHTML = tr("plusRule");

    const summary = document.querySelector(".summary");
    if (summary) {
      summary.innerHTML = `
        <b>${tr("pigroFull")}:</b>
        ${tr("annualReturn")} <b><span id="cagr">—</span></b> |
        <b>${tr("maxDrawdown")}</b> ${LANG === "en" ? "over the period" : "nel periodo"} <b><span id="maxdd">—</span></b><br/>
        ${tr("composition")}: LS80 <b>80%</b> | ${tr("gold")} <b>15%</b> | Bitcoin <b>5%</b><br/>
        ${tr("doubleYears")}: <span id="dbl">—</span><br/>
        <span id="benchmark_summary">${tr("benchmark")}: —</span>
      `;
    }

    const h2s = document.querySelectorAll("h2");
    if (document.getElementById("chart_title")) document.getElementById("chart_title").textContent = tr("chartTitle");
    if (h2s[1]) h2s[1].textContent = tr("ddTitle");

    const liquidityTitle = document.querySelector(".liquidityTitle");
    if (liquidityTitle) liquidityTitle.textContent = tr("residualLiquidity");

    const comparisonH3 = document.querySelector(".comparisonTableHead h3");
    if (comparisonH3) comparisonH3.textContent = tr("completeComparison");

    const periodBox = document.querySelector(".comparisonTablePeriod");
    if (periodBox) periodBox.innerHTML = `${tr("periodCalc")}: <span id="comparison_period_years">—</span> ${LANG === "en" ? "years" : "anni"}`;

    const ths = document.querySelectorAll("#comparison_table th");
    if (ths[0]) ths[0].textContent = tr("portfolio");
    if (ths[1]) ths[1].textContent = tr("annualReturnShort");
    if (ths[2]) ths[2].textContent = tr("maxDrawdown");
    if (ths[3]) ths[3].textContent = tr("finalCapital");

    if (smallTexts[smallTexts.length - 1]) smallTexts[smallTexts.length - 1].innerHTML = tr("msgKey");

    const howTitle = Array.from(document.querySelectorAll("h2")).find(h => h.textContent.includes("Come") || h.textContent.includes("How"));
    if (howTitle) howTitle.textContent = tr("howTitle");

    const ol = document.querySelector("ol");
    if (ol) {
      ol.innerHTML = TEXT[LANG].howList.map(x => `<li>${x}</li>`).join("");
    }

    const btnFax = document.getElementById("btn_faxsimile");
    const btnAdvisor = document.getElementById("btn_consulente");
    const btnBook = document.getElementById("btn_libro");
    const btnExperts = document.querySelector(".btnRow button:last-child");
    const btnMission = document.querySelector(".btnMission");

    if (btnFax) btnFax.textContent = tr("btnFax");
    if (btnAdvisor) btnAdvisor.textContent = tr("btnAdvisor");
    if (btnBook) btnBook.textContent = tr("btnBook");
    if (btnExperts) btnExperts.textContent = tr("btnExperts");
    if (btnMission) btnMission.textContent = tr("btnMission");

    const mission = document.getElementById("missionSection");
    if (mission) mission.innerHTML = tr("missionHtml");

    const modalTitle = document.getElementById("advisor_modal_title");
    if (modalTitle) modalTitle.textContent = tr("advisorsTitle");

    const modalSub = document.querySelector(".modalSub");
    if (modalSub) modalSub.textContent = tr("advisorsSub");

    const modalNote = document.querySelector(".modalNote");
    if (modalNote) modalNote.innerHTML = tr("advisorNote");

    document.querySelectorAll(".langSwitch span").forEach(el => el.classList.remove("active"));
    const active = document.querySelector(`.langSwitch span[onclick="setLang('${LANG}')"]`);
    if (active) active.classList.add("active");
  }

  async function refresh() {
    if (isRefreshing) return;
    isRefreshing = true;

    try {
      applyStaticTranslations();
      normalizeCapitalInput();
      toggleModeBoxes();
      setActiveButtons();
      destroyCharts();
      levaPlusIntegrations = 0;
      levaPlusMarkerIndices = [];
      updateLevaPlusCounter();

      const capital = getCapital();
      const seriesMap = await loadAllSeries();
      const aligned = alignSeries(seriesMap, currentBenchmark, currentMode);
      const labels = aligned.dates;

      if (!labels.length) throw new Error(tr("noData"));

      const pigroSeries = rebalancePortfolio(labels, aligned.ls80, aligned.gold, aligned.btc, capital);
            
            let secondSeries = [];
      let secondLabel = getBenchmarkLabel(currentBenchmark);

      if (currentMode === "normal") {
        secondSeries = benchmarkSeries(aligned, currentBenchmark, capital);
        secondSeries = removeIsolatedSpikes(secondSeries, 0.10);
      } else if (currentMode === "leva_fissa") {
        secondSeries = computeFixedLeverageDetailed(labels, aligned.ls80, aligned.gold, aligned.btc, capital);
        secondLabel = tr("btnLeva20");
      } else {
        const lp = computeLevaPlusDetailed(labels, aligned.ls80, aligned.gold, aligned.btc, pigroSeries, capital);
        secondSeries = lp.series;
        secondLabel = tr("btnLevaPlus");
        levaPlusIntegrations = lp.integrations || 0;
        levaPlusMarkerIndices = Array.isArray(lp.markerIndices) ? lp.markerIndices : [];

        renderLiquidity(labels, lp.liquidity);
        setHtml("liquidity_summary", `${tr("residualLiquidityFinal")}: <b>${euro(lp.liquidity[lp.liquidity.length - 1], 0)}</b>`);
      }

      updateLevaPlusCounter();

      renderMain(labels, pigroSeries, secondSeries, secondLabel);
          
      renderDd(labels, computeDrawdownSeriesPct(pigroSeries), computeDrawdownSeriesPct(secondSeries), secondLabel);
      updateTextSummary(pigroSeries, secondSeries, labels, secondLabel);
      buildComparisonTable(aligned, labels, capital);

    } catch (err) {
      console.error(err);
      setHtml("compare_box", `<strong>${tr("error")}:</strong> ${err.message}`);
      const tbody = document.querySelector("#comparison_table tbody");
      if (tbody) tbody.innerHTML = "";
      setText("comparison_period_years", "—");
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
  currentMode = target.getAttribute("data-mode") || "normal";
  currentBenchmark = target.getAttribute("data-benchmark") || "world";
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

    ["btn_update", "btn_faxsimile", "btn_consulente", "btn_libro", "advisor_modal_close"].forEach(id => {
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
      const clickable = e.target.closest("#btn_update, #btn_faxsimile, #btn_consulente, #btn_libro, #advisor_modal_close, .benchmarkBtn");

      if (clickable) {
        e.preventDefault();
        handleActionClick(clickable);
        return;
      }

      const modal = document.getElementById("advisor_modal");
      if (modal && e.target === modal) closeAdvisorModal();
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeAdvisorModal();
    });
  }

  window.setLang = function (lang) {
    LANG = lang === "en" ? "en" : "it";
    localStorage.setItem("lang", LANG);
    refresh();
  };

  document.addEventListener("DOMContentLoaded", function () {
    bindUi();
    applyStaticTranslations();
    refresh();
  });
})();
