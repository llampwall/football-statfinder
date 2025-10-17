const els = {
  seasonInput: document.getElementById("season-input"),
  weekInput: document.getElementById("week-input"),
  loadBtn: document.getElementById("load-btn"),
  status: document.getElementById("status"),
  loadedLabel: document.getElementById("loaded-label"),
  tableBody: document.getElementById("games-body"),
  gameViewLink: document.getElementById("game-view-link"),
  weekSummary: document.getElementById("week-summary"),
  latestLink: document.getElementById("latest-link"),
  filtersSection: document.getElementById("filters"),
  teamFilter: document.getElementById("team-filter"),
  exportBtn: document.getElementById("export-btn"),
  filterMeta: document.getElementById("filter-meta"),
  statusLine: document.getElementById("status-line"),
};

const STORAGE_KEY = "week-view:last-selection";
const LAST_GAME_KEY = "week-view:last-game";
const MISSING_VALUE = "\u2014";
let timezoneLogged = false;
let warnedTeamNumber = false;
let warnedGameNumber = false;

const STATE = {
  allRows: [],
  filteredRows: [],
  season: null,
  week: null,
  sourcePath: null,
  lastLoadedAt: null,
  pendingScrollKey: null,
  highlightedGameKey: null,
  gameOrdinals: new Map(),
};

const NFL_TEAM_DEFINITIONS = [
  ["Arizona Cardinals", "ari|arz|arizona cardinals|arizona|cardinals"],
  ["Atlanta Falcons", "atl|atlanta|falcons"],
  ["Baltimore Ravens", "bal|baltimore|ravens"],
  ["Buffalo Bills", "buf|buffalo|bills"],
  ["Carolina Panthers", "car|carolina|panthers"],
  ["Chicago Bears", "chi|chicago|bears"],
  ["Cincinnati Bengals", "cin|cincinnati|bengals"],
  ["Cleveland Browns", "cle|cleveland|browns"],
  ["Dallas Cowboys", "dal|dallas|cowboys"],
  ["Denver Broncos", "den|denver|broncos"],
  ["Detroit Lions", "det|detroit|lions"],
  ["Green Bay Packers", "gb|gnb|green bay|packers"],
  ["Houston Texans", "hou|houston|texans"],
  ["Indianapolis Colts", "ind|indianapolis|colts"],
  ["Jacksonville Jaguars", "jax|jaguars|jacksonville|jacksonville jaguars"],
  ["Kansas City Chiefs", "kc|kan|kcc|kansas city|chiefs|kansas city chiefs"],
  ["Las Vegas Raiders", "lv|lvr|las vegas|raiders|las vegas raiders|oakland raiders|oakland"],
  ["Los Angeles Chargers", "lac|los angeles chargers|la chargers|chargers|san diego chargers|san diego|sdc|sd"],
  ["Los Angeles Rams", "lar|la|los angeles rams|la rams|rams|st louis rams|stl"],
  ["Miami Dolphins", "mia|miami|dolphins"],
  ["Minnesota Vikings", "min|minn|minnesota|vikings|minnesota vikings"],
  ["New England Patriots", "ne|nwe|new england|patriots|new england patriots"],
  ["New Orleans Saints", "no|nor|new orleans|saints|new orleans saints"],
  ["New York Giants", "nyg|new york giants|giants|ny giants"],
  ["New York Jets", "nyj|new york jets|jets|ny jets"],
  ["Philadelphia Eagles", "phi|philadelphia|eagles|philadelphia eagles"],
  ["Pittsburgh Steelers", "pit|pittsburgh|steelers|pittsburgh steelers"],
  ["San Francisco 49ers", "sf|sfo|san francisco|san francisco 49ers|49ers|niners"],
  ["Seattle Seahawks", "sea|seattle|seahawks|seattle seahawks"],
  ["Tampa Bay Buccaneers", "tb|tbb|tampa bay|buccaneers|bucs|tampa bay buccaneers"],
  ["Tennessee Titans", "ten|tennessee|titans|tennessee titans"],
  ["Washington Commanders", "was|wsh|washington|commanders|washington commanders|washington football team|wft"],
];

let TEAM_NUMBER_MAP_CACHE = null;

attachListeners();
bootstrap();

function attachListeners() {
  els.loadBtn.addEventListener("click", () => {
    const season = coerceInt(els.seasonInput.value);
    const week = coerceInt(els.weekInput.value);
    if (!season || !week) {
      setStatus("Provide season and week.");
      return;
    }
    loadAndRender(season, week, { fromControl: true });
  });

  els.teamFilter.addEventListener("input", () => {
    applyFilters();
  });

  els.exportBtn.addEventListener("click", () => {
    exportVisibleCsv();
  });
}

async function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const paramSeason = coerceInt(params.get("season"));
  const paramWeek = coerceInt(params.get("week"));
  const paramGame = params.get("game_key");

  if (paramSeason) els.seasonInput.value = paramSeason;
  if (paramWeek) els.weekInput.value = paramWeek;

  const available = await listAvailableWeeks();
  if (available.length) {
    console.log(
      "Available weeks:",
      available.map((entry) => `${entry.season}w${entry.week}`).join(", ")
    );
  } else {
    console.warn("WARN: Unable to read /out directory listing; falling back to manual detection.");
  }

  let target = null;

  if (paramSeason && paramWeek) {
    target = { season: paramSeason, week: paramWeek };
    console.log("Deep link detected -> direct load.");
  } else if (paramSeason) {
    target = available.find((item) => item.season === paramSeason) ?? null;
    if (!target) {
      console.warn(`WARN: No data found for season ${paramSeason} in /out listing.`);
    }
  } else if (paramWeek) {
    target = available.find((item) => item.week === paramWeek) ?? null;
    if (!target) {
      console.warn(`WARN: No season found containing week ${paramWeek}.`);
    }
  } else if (available.length) {
    target = available[0];
    console.log(
      `AUTO: Using latest available season=${target.season} week=${target.week} from directory listing.`
    );
  }

  const storedSelection = loadStoredSelection();
  const storedLastGame = loadStoredLastGame();

  if (!target && storedSelection) {
    console.log("Fallback to stored selection", storedSelection);
    target = storedSelection;
  }

  if (!target) {
    setStatus("No season/week found. Please enter values.");
    return;
  }

  els.seasonInput.value = target.season;
  els.weekInput.value = target.week;

  preparePendingScroll(target.season, target.week, paramGame, storedSelection, storedLastGame);

  const loaded = await loadAndRender(target.season, target.week);
  if (!loaded) {
    if (
      storedSelection &&
      (storedSelection.season !== target.season || storedSelection.week !== target.week)
    ) {
      console.log("Retrying with stored selection after failed load", storedSelection);
      els.seasonInput.value = storedSelection.season;
      els.weekInput.value = storedSelection.week;
      STATE.pendingScrollKey = storedSelection.last_game_key ?? null;
      STATE.highlightedGameKey = storedSelection.last_game_key ?? null;
      const retry = await loadAndRender(storedSelection.season, storedSelection.week);
      if (!retry) {
        setStatus("Failed to load data; verify season/week values.");
      }
    } else {
      setStatus("Failed to load data; verify season/week values.");
    }
  }
}

async function listAvailableWeeks() {
  try {
    const url = new URL("../out/", window.location.href);
    const res = await fetch(url.toString(), { cache: "no-store" });
    if (!res.ok) {
      console.warn(`WARN: Directory listing fetch failed with status ${res.status}`);
      return [];
    }
    const html = await res.text();
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");
    const links = Array.from(doc.querySelectorAll("a"));
    const entries = [];
    links.forEach((link) => {
      const href = link.getAttribute("href") ?? "";
      const match = href.match(/^(\d{4})_week(\d+)\/?$/i);
      if (match) {
        entries.push({
          season: Number(match[1]),
          week: Number(match[2]),
        });
      }
    });
    return entries.sort((a, b) => {
      if (b.season !== a.season) return b.season - a.season;
      return b.week - a.week;
    });
  } catch (err) {
    console.warn("WARN: Unable to parse /out directory listing.", err);
    return [];
  }
}

async function loadAndRender(season, week, options = {}) {
  const { fromControl = false } = options;
  setStatus("Loading games...");
  const result = await loadGames(season, week);
  if (!result.success) {
    setStatus(result.message);
    console.log(`FAIL: Load season=${season} week=${week} (${result.message})`);
    return false;
  }

  STATE.sourcePath = result.sourcePath ?? STATE.sourcePath;
  STATE.lastLoadedAt = new Date();

  const applied = applyLoadedRows(result.rows, season, week, {
    updateHistory: fromControl,
  });
  if (applied) {
    updateStatusLine();
  }
  return applied;
}

function applyLoadedRows(rows, season, week, { updateHistory = false } = {}) {
  const safeRows = Array.isArray(rows) ? rows.slice() : [];
  STATE.allRows = safeRows;
  STATE.season = season;
  STATE.week = week;

  if (STATE.pendingScrollKey && !STATE.highlightedGameKey) {
    STATE.highlightedGameKey = STATE.pendingScrollKey;
  }

  console.log(`${safeRows.length >= 1 ? "PASS" : "FAIL"}: Games parsed (count=${safeRows.length})`);

  if (safeRows.length > 0) {
    els.filtersSection.classList.remove("hidden");
  } else {
    els.filtersSection.classList.add("hidden");
  }

  applyFilters();
  persistSelection({ season, week, last_game_key: STATE.highlightedGameKey ?? null });

  if (updateHistory) {
    const url = new URL(window.location.href);
    url.searchParams.set("season", season);
    url.searchParams.set("week", week);
    window.history.replaceState(null, "", url.toString());
  }

  return true;
}

async function loadGames(season, week) {
  const relPath = `out/${season}_week${week}/games_week_${season}_${week}.jsonl`;
  const path = `../${relPath}`;
  const url = new URL(path, window.location.href);
  try {
    const res = await fetch(url.toString(), { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const text = await res.text();
    const parsed = parseJsonl(text);
    return {
      success: true,
      rows: parsed.records,
      count: parsed.count,
      message: parsed.count ? "Loaded" : "No games in file",
      sourcePath: relPath,
    };
  } catch (err) {
    console.error(`FAIL: loadGames season=${season} week=${week}`, err);
    return { success: false, message: `Failed to load: ${err.message}`, sourcePath: relPath };
  }
}

function preparePendingScroll(season, week, explicitGameKey, storedSelection, storedLast) {
  let key = explicitGameKey || null;
  if (!key && storedSelection && storedSelection.season === season && storedSelection.week === week) {
    if (storedSelection.last_game_key) {
      key = storedSelection.last_game_key;
    }
  }
  if (
    !key &&
    storedLast &&
    storedLast.season === season &&
    storedLast.week === week &&
    storedLast.game_key
  ) {
    key = storedLast.game_key;
  }
  if (key) { STATE.pendingScrollKey = key; STATE.highlightedGameKey = key; } else { STATE.pendingScrollKey = null; STATE.highlightedGameKey = null; }
}

function applyFilters() {
  if (!Array.isArray(STATE.allRows)) {
    STATE.filteredRows = [];
    renderTable([], { season: STATE.season, week: STATE.week });
    updateFooter(0, STATE.season, STATE.week);
    return;
  }

  const total = STATE.allRows.length;
  const query = (els.teamFilter.value || "").trim().toLowerCase();
  const comparator = (a, b) =>
    (a?.kickoff_iso_utc || "").localeCompare(b?.kickoff_iso_utc || "");
  const sortedAll = STATE.allRows.slice().sort(comparator);
  STATE.gameOrdinals = computeGameOrdinals(sortedAll);

  const filtered = STATE.allRows.filter((row) => matchesTeamFilter(row, query));

  filtered.sort(comparator);

  STATE.filteredRows = filtered;
  renderTable(filtered, { season: STATE.season, week: STATE.week });
  updateFooter(total, STATE.season, STATE.week);
}

function renderTable(rows, { season, week }) {
  const tbody = els.tableBody;
  tbody.innerHTML = "";

  if (!rows || rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 16;
    td.textContent = "No games found.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    updateFilterMeta(0, STATE.allRows.length);
    highlightRow(null);
    return;
  }

  const numericColumns = new Set([6, 8, 9, 10, 11, 12]);
  const ordinals =
    STATE.gameOrdinals && typeof STATE.gameOrdinals.get === "function"
      ? STATE.gameOrdinals
      : new Map();

  let groupIndex = 0;
  rows.forEach((row) => {
    const ord = STATE.gameOrdinals?.get(row.game_key) ?? null;
    const [top, bot] = buildGameGroup(row, ord, groupIndex++);
    els.tableBody.appendChild(top);
    els.tableBody.appendChild(bot);
  });

  // rows.forEach((row) => {
  //   const { kickoff_iso_utc, game_key } = row;
  //   if (!kickoff_iso_utc || !game_key) {
  //     console.warn("WARN: Missing key fields", { kickoff_iso_utc, game_key });
  //   }

  //   const ordinal = ordinals.get(game_key) ?? null;
  //   const gameNumber = formatGameNumber(week, ordinal);

  //   ["away", "home"].forEach((side) => {
  //     const cells = buildTeamRow(row, side, gameNumber);
  //     const tr = document.createElement("tr");
  //     tr.dataset.gameKey = game_key ?? "";
  //     tr.tabIndex = 0;
  //     cells.forEach((value, idx) => {
  //       const td = document.createElement("td");
  //       td.textContent = value;
  //       if (numericColumns.has(idx)) {
  //         td.classList.add("numeric");
  //       }
  //       tr.appendChild(td);
  //     });
  //     tr.addEventListener("click", () => {
  //       openGame(game_key);
  //     });
  //     tr.addEventListener("keydown", (event) => {
  //       if (event.key === "Enter") {
  //         openGame(game_key);
  //       } else if (event.key === "ArrowRight" && event.ctrlKey) {
  //         openGame(game_key, { newTab: true });
  //       }
  //     });
  //     tbody.appendChild(tr);
  //   });
  // });

  const highlightKey = STATE.pendingScrollKey || STATE.highlightedGameKey;
  if (highlightKey) {
    highlightRow(highlightKey, { scroll: Boolean(STATE.pendingScrollKey) });
  } else {
    highlightRow(null);
  }
  STATE.pendingScrollKey = null;

  updateFilterMeta(rows.length, STATE.allRows.length);
}

// ---- Canonicalization for weird team keys/labels (Rams/Jags/Commanders/49ers) ----

// one-shot warning helper
const warnOnce = (() => {
  const seen = new Set();
  return (k, msg) => { if (!seen.has(k)) { console.warn(msg); seen.add(k); } };
})();

// alias map for common non-canonical norms
const TEAM_ALIAS = {
  la: "lar", stl: "lar",
  jac: "jax",
  wsh: "was", wft: "was",
  sfo: "sf",
};

// minimal label-based hints if norm is unknown/ambiguous
function inferFromLabel(label) {
  const t = (label || "").toLowerCase();
  if (t.includes("rams")) return "lar";
  if (t.includes("jaguars")) return "jax";
  if (t.includes("commanders") || t === "washington") return "was";
  if (t.includes("49ers")) return "sf";
  return null;
}

// return a canonical norm we can map 1..32
function coerceTeamNorm(norm, label) {
  if (!norm) return null;
  const k = norm.toLowerCase();
  if (TEAM_ALIAS[k]) return TEAM_ALIAS[k];
  const inferred = inferFromLabel(label);
  if (inferred && inferred !== k) {
    warnOnce(`coerce:${k}->${inferred}`, `Team norm alias: '${k}' → '${inferred}' based on label '${label}'`);
    return inferred;
  }
  return k;
}


let TEAM_NUM_MAP = null;

// if you already have a "formatTeam" or similar resolver by norm, plug it in here.
// this wrapper falls back to using the norm in ALLCAPS if nothing else exists.
function labelFromNorm(norm) {
  try {
    if (typeof resolveTeamLabelFromNorm === "function") return resolveTeamLabelFromNorm(norm);
    if (typeof formatTeam === "function") return formatTeam(norm); // many codebases already have this
  } catch (e) {}
  return (norm || "").toUpperCase();
}

function buildTeamNumberMap() {
  if (TEAM_NUM_MAP) return TEAM_NUM_MAP;

  // Use the normalized keys exactly as they appear in games JSONL.
  // (Washington is "was" in your outputs; keep it that way.)
  const NFL_TEAMS = [
    "ari","atl","bal","buf","car","chi","cin","cle","dal","den","det","gb",
    "hou","ind","jax","kc","lv","lac","lar","mia","min","ne","no","nyg","nyj",
    "phi","pit","sf","sea","tb","ten","was"
  ];

  const pairs = NFL_TEAMS.map(n => [labelFromNorm(n), n]);
  // sort by display label, case-insensitive
  pairs.sort((a, b) => a[0].localeCompare(b[0], undefined, { sensitivity: "base" }));
  TEAM_NUM_MAP = new Map(pairs.map(([, n], i) => [n, i + 1]));
  return TEAM_NUM_MAP;
}


function formatTeamNumber(row, side) {
  const rawNorm = side === "home" ? row.home_team_norm : row.away_team_norm;
  const label = resolveTeamName(row, side); // your existing resolver for the Team column
  const norm = coerceTeamNorm(rawNorm, label);
  if (!norm) return MISSING_VALUE;
  const map = buildTeamNumberMap();
  const num = map.get(norm);
  if (!num) {
    warnOnce(`no-teamnum:${norm}`, `No Team # for norm='${norm}' label='${label}'`);
    return MISSING_VALUE;
  }
  return num;
}

function appendCell(tr, text, { numeric = false, rowspan = 1 } = {}) {
  const td = document.createElement("td");
  if (text !== null && text !== undefined) td.textContent = String(text);
  if (numeric) td.classList.add("numeric");
  if (rowspan > 1) td.rowSpan = rowspan;
  tr.appendChild(td);
  return td;
}

function buildGameGroup(row, ordinal, groupIndex) {
  const iso = row?.kickoff_iso_utc ?? null;
  const gameNumber = formatGameNumber(STATE.week, ordinal);

  // zebra by game group
  const stripe = (groupIndex % 2 === 0) ? "group-even" : "group-odd";

  // create two rows: top = away, bottom = home
  const trTop = document.createElement("tr");
  const trBot = document.createElement("tr");
  [trTop, trBot].forEach(tr => {
    tr.dataset.gameKey = row.game_key ?? "";
    tr.classList.add("group", stripe);
    tr.tabIndex = 0;
    tr.addEventListener("click", () => openGame(row.game_key));
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter") openGame(row.game_key);
      else if (e.key === "ArrowRight" && e.ctrlKey) openGame(row.game_key, { newTab: true });
    });
  });

  // --- shared cells (rowspan = 2): Date, Time, Game #
  appendCell(trTop, fmtDatePT(iso), { rowspan: 2 });
  appendCell(trTop, fmtTimePT(iso), { rowspan: 2 });

  // Team # (away only here; home will be on trBot)
  appendCell(trTop, formatTeamNumber(row, "away"));

  // Game # (shared)
  appendCell(trTop, gameNumber, { rowspan: 2 });

  // Team (away)
  appendCell(trTop, resolveTeamName(row, "away"));

  // Odds (away)
  appendCell(trTop, formatOddsCell(row, "away"));

  // Total (your spec: only on favored row; else blank)
  appendCell(trTop, isFavRow(row, "away") ? formatNumber(row.total, { decimals: 1, signed: true }) : "");

  // W-L-T (away)
  appendCell(trTop, teamRecord(row, "away"));

  // Current PR (away)
  appendCell(trTop, formatNumber(row.away_pr), { numeric: true });

  // Diff (favored only per your buildTeamRow)
  appendCell(trTop, isFavRow(row, "away")
    ? formatNumber(row.rating_diff_favored_team, { decimals: 1, signed: true }) : "");

  // Rating vs Odds (favored only)
  appendCell(trTop, isFavRow(row, "away")
    ? formatNumber(row.rating_vs_odds, { decimals: 1, signed: true }) : "");

  // SoS (away)
  appendCell(trTop, formatNumber(row.away_sos), { numeric: true });

  // SoS diff (only on higher-SoS row; else blank)
  appendCell(trTop, sosDiffForRow(row.home_sos, row.away_sos, "away"), { numeric: true });

  // ===== bottom row (home) =====

  appendCell(trBot, formatTeamNumber(row, "home"));                              // Team #
  appendCell(trBot, resolveTeamName(row, "home"));                               // Team
  appendCell(trBot, formatOddsCell(row, "home"));                                // Odds
  appendCell(trBot, isFavRow(row, "home") ? formatNumber(row.total, {            // Total (fav only)
    decimals: 1, signed: true }) : "");
  appendCell(trBot, teamRecord(row, "home"));                                    // W-L-T
  appendCell(trBot, formatNumber(row.home_pr), { numeric: true });               // Current PR
  appendCell(trBot, isFavRow(row, "home") ?                                      // Diff (fav only)
    formatNumber(row.rating_diff_favored_team, { decimals: 1, signed: true }) : "");
  appendCell(trBot, isFavRow(row, "home") ?                                      // RvO (fav only)
    formatNumber(row.rating_vs_odds, { decimals: 1, signed: true }) : "");
  appendCell(trBot, formatNumber(row.home_sos), { numeric: true });              // SoS
  appendCell(trBot, sosDiffForRow(row.home_sos, row.away_sos, "home"), {         // SoS diff (higher-SoS row only)
    numeric: true });

  return [trTop, trBot];
}


function buildTeamRow(row, side, gameNumber) {
  const iso = row?.kickoff_iso_utc ?? null;
  const favored = isFavRow(row, side);
  const currentPrValue = side === "home" ? row.home_pr : row.away_pr;
  const sosValue = side === "home" ? row.home_sos : row.away_sos;

  const total = favored
    ? formatNumber(row.total, { decimals: 1, signed: true })
    : "";
  const diff = favored
    ? formatNumber(row.rating_diff_favored_team, { decimals: 1, signed: true })
    : "";
  const rvo = favored
    ? formatNumber(row.rating_vs_odds, { decimals: 1, signed: true })
    : "";
  const gameValue = gameNumber ?? placeholderGameNumber();
  const teamNumber = formatTeamNumber(row, side);

  return [
    fmtDatePT(iso),
    fmtTimePT(iso),
    teamNumber,
    gameValue,
    resolveTeamName(row, side),
    formatOddsCell(row, side),
    total,
    teamRecord(row, side),
    formatNumber(currentPrValue),
    diff,
    rvo,
    formatNumber(sosValue),
    sosDiffForRow(row.home_sos, row.away_sos, side),
  ];
}

function resolveTeamName(row, side) {
  const sourceKey = side === "home" ? "sagarin_row_home" : "sagarin_row_away";
  const fromSagarin = row?.raw_sources?.[sourceKey]?.team;
  if (fromSagarin) return fromSagarin;
  const raw = side === "home" ? row.home_team_raw : row.away_team_raw;
  const norm = side === "home" ? row.home_team_norm : row.away_team_norm;
  return formatTeam(norm, raw);
}

function getTeamCode(row, side) {
  const raw = side === "home" ? row.home_team_raw : row.away_team_raw;
  const norm = side === "home" ? row.home_team_norm : row.away_team_norm;
  if (raw) return String(raw).toUpperCase();
  if (norm) return String(norm).toUpperCase();
  return "";
}

function formatOddsCell(row, side) {
  if (!isFavRow(row, side)) return "";
  if (!hasNumeric(row.spread_favored_team)) return MISSING_VALUE;
  const code = getTeamCode(row, side);
  const spread = formatNumber(row.spread_favored_team, { decimals: 1, signed: true });
  return spread;
}

function teamRecord(row, side) {
  const key = side === "home" ? "home_su" : "away_su";
  const record = row?.[key];
  return record ? String(record) : MISSING_VALUE;
}

function placeholderTeamNumber() {
  if (!warnedTeamNumber) {
    console.warn("WARN: Team numbers unavailable; placeholders in use.");
    warnedTeamNumber = true;
  }
  return MISSING_VALUE;
}

function placeholderGameNumber() {
  if (!warnedGameNumber) {
    console.warn("WARN: Game numbers unavailable; placeholders in use.");
    warnedGameNumber = true;
  }
  return MISSING_VALUE;
}

function matchesTeamFilter(row, query) {
  if (!query) return true;
  const teams = [
    resolveTeamName(row, "home"),
    resolveTeamName(row, "away"),
    row.home_team_raw,
    row.away_team_raw,
    row.home_team_norm,
    row.away_team_norm,
  ];
  return teams.some((team) => {
    if (!team) return false;
    return String(team).toLowerCase().includes(query);
  });
}

function computeGameOrdinals(rows) {
  const map = new Map();
  let ordinal = 1;
  rows.forEach((row) => {
    const key = row?.game_key;
    if (!key || map.has(key)) return;
    map.set(key, ordinal++);
  });
  return map;
}

function formatGameNumber(week, ordinal) {
  if (!hasNumeric(week) || !hasNumeric(ordinal)) return null;
  const weekPart = String(Number(week));
  const ordinalPart = String(Number(ordinal)).padStart(2, "0");
  return `${weekPart}${ordinalPart}`;
}

function isFavRow(row, side) {
  if (!row || !row.favored_side) return false;
  if (!hasNumeric(row.spread_favored_team)) return false;
  if (side === "home") return row.favored_side === "HOME";
  if (side === "away") return row.favored_side === "AWAY";
  return false;
}

function sosDiffForRow(homeSos, awaySos, side) {
  if (!hasNumeric(homeSos) || !hasNumeric(awaySos)) return MISSING_VALUE;
  const diff =
    side === "home" ? Number(homeSos) - Number(awaySos) : Number(awaySos) - Number(homeSos);
  if (diff > 0) {
    return formatNumber(diff, { decimals: 1, signed: true });
  }
  return "";
}


function openGame(gameKey, { newTab = false } = {}) {
  if (!STATE.season || !STATE.week) {
    setStatus("Season/week not loaded.");
    return;
  }
  storeLastViewedGame(gameKey);
  highlightRow(gameKey);
  const url = `game_view.html?season=${STATE.season}&week=${STATE.week}&game_key=${encodeURIComponent(
    gameKey
  )}`;
  if (newTab) {
    window.open(url, "_blank", "noopener");
  } else {
    window.location.href = url;
  }
}

function storeLastViewedGame(gameKey) {
  if (!STATE.season || !STATE.week) return;
  STATE.highlightedGameKey = gameKey;
  STATE.pendingScrollKey = gameKey;
  persistSelection({ season: STATE.season, week: STATE.week, last_game_key: gameKey });
  try {
    localStorage.setItem(
      LAST_GAME_KEY,
      JSON.stringify({ season: STATE.season, week: STATE.week, game_key: gameKey })
    );
  } catch {
    // ignore storage failures
  }
}

function updateFilterMeta(visible, total) {
  if (!els.filterMeta) return;
  if (!total) {
    els.filterMeta.textContent = "";
    return;
  }
  const visibleLabel = visible === 1 ? "game" : "games";
  const totalLabel = total === 1 ? "game" : "games";
  if (visible === total) {
    els.filterMeta.textContent = `Showing all ${visible} ${visibleLabel}`;
  } else {
    els.filterMeta.textContent = `Showing ${visible} of ${total} ${totalLabel}`;
  }
}

function highlightRow(gameKey, { scroll = false } = {}) {
  const rows = Array.from(els.tableBody.querySelectorAll("tr[data-game-key]"));
  let found = false;
  rows.forEach((row) => {
    const match = Boolean(gameKey) && row.dataset.gameKey === gameKey;
    row.classList.toggle("active-row", match);
    if (match) {
      found = true;
      if (scroll) {
        row.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }
  });
  if (found) { STATE.highlightedGameKey = gameKey; } else { STATE.highlightedGameKey = null; }
}

function updateFooter(totalCount, season, week) {
  const totalLabel = totalCount === 1 ? "game" : "games";
  const visible = STATE.filteredRows.length;
  const visibleLabel = visible === 1 ? "game" : "games";
  if (visible && visible !== totalCount) {
    els.status.textContent = `Loaded ${totalCount} ${totalLabel} (showing ${visible} ${visibleLabel})`;
  } else {
    els.status.textContent = `Loaded ${totalCount} ${totalLabel}`;
  }
  els.loadedLabel.textContent = `Season ${season}, Week ${week}`;
  els.weekSummary.textContent = `Currently viewing Season ${season}, Week ${week}`;
  els.gameViewLink.href = `game_view.html?season=${season}&week=${week}`;
  els.latestLink.href = "week_view.html";
}

function exportVisibleCsv() {
  if (!STATE.filteredRows || STATE.filteredRows.length === 0) {
    setStatus("No rows available to export.");
    return;
  }
  if (!STATE.season || !STATE.week) {
    setStatus("Season/week not available for export.");
    return;
  }

  const header = [
    "Date (PT)",
    "Time (PT)",
    "Team #",
    "Game #",
    "Team",
    "Odds",
    "O/U",
    "W-L-T",
    "Current PR",
    "Diff",
    "Rating vs Odds",
    "Schedule Strength (SoS)",
    "SoS Diff",
  ];

  const lines = [header.join(",")];
  const ordinals =
    STATE.gameOrdinals && typeof STATE.gameOrdinals.get === "function"
      ? STATE.gameOrdinals
      : new Map();
  STATE.filteredRows.forEach((row) => {
    ["away", "home"].forEach((side) => {
      const ordinal = ordinals.get(row?.game_key) ?? null;
      const gameNumber = formatGameNumber(STATE.week, ordinal);
      const cells = buildTeamRow(row, side, gameNumber).map(csvValue);
      lines.push(cells.join(","));
    });
  });

  const csv = lines.join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const filename = `week_${STATE.season}_${STATE.week}.csv`;
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  const exportedRows = STATE.filteredRows.length * 2;
  setStatus(`Exported ${exportedRows} rows to ${filename}`);
}

function csvValue(value) {
  if (value === null || value === undefined) return "";
  const str = String(value);
  if (str.includes(",") || str.includes("\"") || str.includes("\n")) {
    return '"' + str.replace(/"/g, '""') + '"';
  }
  return str;
}


function updateStatusLine() {
  if (!els.statusLine) return;
  if (!STATE.sourcePath || !STATE.lastLoadedAt) {
    els.statusLine.textContent = "";
    return;
  }
  const time = STATE.lastLoadedAt.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  els.statusLine.textContent = `Source: ${STATE.sourcePath} ┬╖ Loaded ${time}`;
}

function logTimeZone(success) {
  if (timezoneLogged) return;
  console.log(success ? "Time zone: America/Los_Angeles (DST auto via Intl)" : "Time zone: UTC fallback");
  timezoneLogged = true;
}

function getPacificDateParts(iso) {
  if (!iso) return null;
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) throw new Error("Invalid date");
    const tz = "America/Los_Angeles";
    const dateFormatter = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
    const timeFormatter = new Intl.DateTimeFormat("en-US", {
      timeZone: tz,
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
    });
    const [mm, dd, yyyy] = dateFormatter.format(d).split("/");
    logTimeZone(true);
    return {
      date: `${yyyy}-${mm}-${dd}`,
      time: timeFormatter.format(d),
    };
  } catch (err) {
    console.warn("fmtKickoffPT fallback -> UTC", err);
    logTimeZone(false);
    return null;
  }
}

function fmtDatePT(iso) {
  const parts = getPacificDateParts(iso);
  return parts ? parts.date : MISSING_VALUE;
}

function fmtTimePT(iso) {
  const parts = getPacificDateParts(iso);
  return parts ? parts.time : MISSING_VALUE;
}

function fmtKickoffPT(iso) {
  const parts = getPacificDateParts(iso);
  if (!parts) return fmtKickoffUTC(iso);
  return `${parts.date} ${parts.time} PT`;
}

function fmtKickoffUTC(isoString) {
  return formatKickoff(isoString);
}



function formatKickoff(isoString) {
  if (!isoString) return MISSING_VALUE;
  const clean = isoString.replace("Z", "+00:00");
  const match = clean.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
  if (match) {
    return `${match[1]} ${match[2]}`;
  }
  try {
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) throw new Error();
    const year = date.getUTCFullYear();
    const month = String(date.getUTCMonth() + 1).padStart(2, "0");
    const day = String(date.getUTCDate()).padStart(2, "0");
    const hour = String(date.getUTCHours()).padStart(2, "0");
    const minute = String(date.getUTCMinutes()).padStart(2, "0");
    return `${year}-${month}-${day} ${hour}:${minute}`;
  } catch {
    return MISSING_VALUE;
  }
}

function formatTeam(norm, raw) {
  if (raw) return raw.toUpperCase();
  if (norm) return String(norm).toUpperCase();
  return MISSING_VALUE;
}

function formatNumber(value, { decimals = 1, signed = false } = {}) {
  if (!hasNumeric(value)) return MISSING_VALUE;
  const num = Number(value);
  const fixed = num.toFixed(decimals);
  if (!signed) return fixed;
  if (num > 0) return `+${fixed}`;
  if (num < 0) return `\u2212${Math.abs(num).toFixed(decimals)}`;
  return `0.${"0".repeat(decimals)}`;
}

function parseJsonl(text) {
  const records = [];
  if (!text) return { records, count: 0 };
  const lines = text.split(/\r?\n/);
  lines.forEach((rawLine, idx) => {
    let line = rawLine.trim();
    if (!line) return;
    if (idx === 0 && line.charCodeAt(0) === 0xfeff) {
      line = line.slice(1);
    }
    const sanitized = line.replace(/([-+]?Infinity|\bNaN\b)/gi, "null");
    try {
      const parsed = JSON.parse(sanitized);
      records.push(parsed);
    } catch (err) {
      console.warn(`WARN: Failed to parse line ${idx + 1}`, err);
    }
  });
  return { records, count: records.length };
}

function hasNumeric(value) {
  if (value === null || value === undefined) return false;
  const num = Number(value);
  return Number.isFinite(num);
}

function coerceInt(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === "string" && value.trim() === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? Math.trunc(num) : null;
}

function setStatus(message) {
  els.status.textContent = message ?? "";
}

function persistSelection(selection) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
  } catch {
    // ignore storage issues
  }
}

function loadStoredSelection() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function loadStoredLastGame() {
  try {
    const raw = localStorage.getItem(LAST_GAME_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}
