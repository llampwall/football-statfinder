import { sitePath, siteUrl } from "./js/base_path.js";
import { getHfa } from "./js/game_metrics.js";

const els = {
  seasonInput: document.getElementById("season-input"),
  weekInput: document.getElementById("week-input"),
  leagueSelect: document.getElementById("league-select"),
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
const LEAGUE_STORAGE_KEY = "week-view:league";
const DEFAULT_LEAGUE = "nfl";
const VALID_LEAGUES = new Set(["nfl", "cfb"]);
const MISSING_VALUE = "\u2014";
const WEEK_CACHE_PREFIX = "week-view:games:";
const CACHE_VERSION = "v4";
let timezoneLogged = false;
let warnedTeamNumber = false;
let warnedGameNumber = false;
const CFB_BLOCK_MESSAGE =
  "CFB: games_week is missing odds/ratings fields—try re-running refresh or hard-reload.";
const qs = new URLSearchParams(location.search);
const ls = localStorage.getItem("cfb_soft_block");
const CFB_SOFT_BLOCK =
  qs.has("hard") ? false :
  qs.has("soft") ? true :
  (ls == null ? true : ls === "1");

const STATE = {
  allRows: [],
  filteredRows: [],
  season: null,
  week: null,
  league: DEFAULT_LEAGUE,
  sourcePath: null,
  lastLoadedAt: null,
  pendingScrollKey: null,
  highlightedGameKey: null,
  gameOrdinals: new Map(),
  weekPaths: null,
  pathsLogged: false,
  teamNumberMap: null,
  loggedCfbExample: false,
  cfbCoverage: null,
  cfbBlockActive: false,
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

function isCFBLeague() {
  return (STATE.league ?? DEFAULT_LEAGUE) === "cfb";
}

function toDisplayName(primary, fallbackName) {
  const base =
    primary !== undefined && primary !== null && String(primary).trim()
      ? String(primary).trim()
      : fallbackName !== undefined && fallbackName !== null
      ? String(fallbackName).trim()
      : "";
  if (!base) return "";
  const words = base.split(/\s+/).map((part) =>
    part
      .split("-")
      .map((segment) =>
        segment
          .split("'")
          .map((piece) => titleCaseToken(piece))
          .join("'")
      )
      .join("-")
  );
  return words.join(" ");
}

function titleCaseToken(token) {
  if (!token) return "";
  if (token.length === 1) {
    return token.toUpperCase();
  }
  const lower = token.toLowerCase();
  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

function buildCfbTeamNumberMap(rows) {
  const names = new Set();
  (rows || []).forEach((row) => {
    const homeName = resolveTeamName(row, "home");
    const awayName = resolveTeamName(row, "away");
    if (homeName && homeName !== MISSING_VALUE) names.add(homeName);
    if (awayName && awayName !== MISSING_VALUE) names.add(awayName);
  });
  const ordered = Array.from(names).sort((a, b) => a.localeCompare(b));
  const map = new Map();
  ordered.forEach((name, idx) => map.set(name, idx + 1));
  console.info(`[CFB Week] numbered ${ordered.length} teams`);
  return map;
}

function teamNumberFromDisplay(displayName) {
  if (!displayName) return null;
  const map = STATE.teamNumberMap;
  if (!map || typeof map.get !== "function") return null;
  return map.get(displayName) ?? null;
}

function computeCfbCoverageStats(rows) {
  if (!Array.isArray(rows)) {
    return { rows: 0, oddsCovered: 0, rvoCovered: 0 };
  }
  let oddsCovered = 0;
  let rvoCovered = 0;
  rows.forEach((row) => {
    if (row && hasNumeric(row?.spread_favored_team)) {
      oddsCovered += 1;
    }
    if (row && hasNumeric(row?.rating_vs_odds)) {
      rvoCovered += 1;
    }
  });
  return {
    rows: rows.length,
    oddsCovered,
    rvoCovered,
  };
}

function shouldBlockCfbCoverage(stats) {
  if (!stats || !stats.rows) return false;
  const missing = stats.oddsCovered === 0 || stats.rvoCovered === 0;
  return CFB_SOFT_BLOCK ? false : missing;
}

function bannerForCfb(stats) {
  if (!stats || !stats.rows) return null;
  const missing = stats.oddsCovered === 0 || stats.rvoCovered === 0;
  if (!missing) return null;
  const hint =
    stats.oddsCovered === 0 && stats.rvoCovered === 0
      ? "odds and rating fields"
      : stats.oddsCovered === 0
      ? "odds fields"
      : "rating-vs-odds fields";
  return `CFB: games_week loaded (${stats.rows} rows) but ${hint} are blank — run refresh or check backfill logs.`;
}

function syncCfbCoverage(rows) {
  const stats = computeCfbCoverageStats(rows);
  STATE.cfbCoverage = stats;
  STATE.cfbBlockActive = shouldBlockCfbCoverage(stats);
  const logCoverage = () => {
    let negSosDiffs = 0;
    try {
      const cells = document.querySelectorAll('td[data-col="sosDiff"]');
      if (cells && cells.length) {
        negSosDiffs = Array.from(cells).filter((td) =>
          String(td.textContent || "").trim().startsWith("\u2212")
        ).length;
      }
    } catch {
      negSosDiffs = 0;
    }
    console.log(
      `CFB Week rows=${stats.rows}; odds_covered=${stats.oddsCovered}; rvo_covered=${stats.rvoCovered}; sosDiff_negatives=${negSosDiffs}`
    );
  };
  if (typeof requestAnimationFrame === "function") {
    requestAnimationFrame(logCoverage);
  } else {
    setTimeout(logCoverage, 0);
  }
  if (STATE.cfbBlockActive) {
    setStatus(CFB_BLOCK_MESSAGE);
  } else if (els.status && els.status.textContent === CFB_BLOCK_MESSAGE) {
    setStatus("");
  }
}

function computeFavoredMetrics(row) {
  if (!row) return null;
  const favoredRaw = (row.favored_side ?? "").toString().toUpperCase();
  if (favoredRaw !== "HOME" && favoredRaw !== "AWAY") return null;
  const favored = favoredRaw === "HOME" ? "home" : "away";
  const unfavored = favored === "home" ? "away" : "home";
  let pr = hasNumeric(row[`${favored}_pr`]) ? Number(row[`${favored}_pr`]) : null;
  if (favored === "home" && pr !== null) {
    pr += getHfa(row, STATE.league);
  }
  const diff = hasNumeric(row.rating_diff_favored_team) ? Number(row.rating_diff_favored_team) : null;
  const rvo = hasNumeric(row.rating_vs_odds) ? Number(row.rating_vs_odds) : null;
  const favSos = hasNumeric(row[`${favored}_sos`]) ? Number(row[`${favored}_sos`]) : null;
  const oppSos = hasNumeric(row[`${unfavored}_sos`]) ? Number(row[`${unfavored}_sos`]) : null;
  const sosDiff = favSos !== null && oppSos !== null ? favSos - oppSos : null;
  return {
    favoredSide: favored,
    unfavoredSide: unfavored,
    pr,
    diff,
    rvo,
    sos: favSos,
    sosDiff,
  };
}

function rowHasExplicitHfa(row) {
  if (!row) return false;
  if (hasNumeric(row?.hfa) || hasNumeric(row?.hfa_adjust)) return true;
  if (hasNumeric(row?.raw_sources?.sagarin_row_home?.hfa)) return true;
  if (hasNumeric(row?.raw_sources?.sagarin_row_away?.hfa)) return true;
  return false;
}

function formatFavoredMetric(metrics, side, value, options = {}) {
  if (!metrics || !metrics.favoredSide) return MISSING_VALUE;
  if (metrics.favoredSide !== side) return MISSING_VALUE;
  return formatNumber(value, options);
}

function sosDiffCell(row, side) {
  const h = Number(row.home_sos);
  const a = Number(row.away_sos);
  if (!Number.isFinite(h) || !Number.isFinite(a)) return MISSING_VALUE;
  const diff = side === "home" ? h - a : a - h;
  return diff > 0 ? formatNumber(diff, { decimals: 2 }) : "";
}

function logCfbMetrics(row, metrics) {
  if (!metrics) return;
  const awayLabel = resolveTeamName(row, "away");
  const homeLabel = resolveTeamName(row, "home");
  const prText = formatNumber(metrics.pr, { decimals: 2 });
  const diffText = formatNumber(metrics.diff, { decimals: 1, signed: true });
  const rvoText = formatNumber(metrics.rvo, { decimals: 1, signed: true });
  const sosText = formatNumber(metrics.sos, { decimals: 2 });
  const sosDiffText = formatNumber(metrics.sosDiff, { decimals: 2, signed: true });
  if (!STATE.loggedCfbExample) {
    console.log(
      `[CFB Week] ${awayLabel}@${homeLabel} PR=${prText} DIFF=${diffText} RvO=${rvoText} SOS=${sosText} ΔSOS=${sosDiffText}`
    );
    STATE.loggedCfbExample = true;
  }
  if (row.game_key === "20251019_0000_cincinnati_oklahoma_state") {
    console.log(
      `[CFB Week] CIN@OKST PR=${prText} DIFF=${diffText} RvO=${rvoText} SOS=${sosText} ΔSOS=${sosDiffText}`
    );
  }
}

attachListeners();
bootstrap();

function normalizeLeague(raw) {
  if (raw === null || raw === undefined) return DEFAULT_LEAGUE;
  const value = String(raw).trim().toLowerCase();
  return VALID_LEAGUES.has(value) ? value : DEFAULT_LEAGUE;
}

function loadStoredLeague() {
  try {
    const raw = localStorage.getItem(LEAGUE_STORAGE_KEY);
    if (!raw) return null;
    return normalizeLeague(raw);
  } catch {
    return null;
  }
}

function persistLeaguePreference(league) {
  try {
    localStorage.setItem(LEAGUE_STORAGE_KEY, league);
  } catch {
    // ignore storage errors
  }
}

function setActiveLeague(next, { updateSelect = true, persist = true, updateHistory = false } = {}) {
  const league = normalizeLeague(next);
  STATE.league = league;
  if (updateSelect && els.leagueSelect && els.leagueSelect.value !== league) {
    els.leagueSelect.value = league;
  }
  if (persist) {
    persistLeaguePreference(league);
  }
  if (updateHistory) {
    const url = new URL(window.location.href);
    if (league === DEFAULT_LEAGUE) {
      url.searchParams.delete("league");
    } else {
      url.searchParams.set("league", league);
    }
    window.history.replaceState(null, "", url.toString());
  }
}

function buildWeekPaths(league, season, week) {
  const normalized = normalizeLeague(league);
  const baseDir =
    normalized === "cfb"
      ? `out/cfb/${season}_week${week}`
      : `out/${season}_week${week}`;
  return {
    league: normalized,
    season,
    week,
    baseDir: sitePath(baseDir),
    gamesJsonl: sitePath(`${baseDir}/games_week_${season}_${week}.jsonl`),
    sidecarDir: sitePath(`${baseDir}/game_schedules`),
  };
}

function attachListeners() {
  els.loadBtn.addEventListener("click", () => {
    const season = coerceInt(els.seasonInput.value);
    const week = coerceInt(els.weekInput.value);
    if (!season || !week) {
      setStatus("Provide season and week.");
      return;
    }
    const league = normalizeLeague(els.leagueSelect?.value ?? STATE.league);
    setActiveLeague(league, { updateSelect: true, updateHistory: true });
    loadAndRender(season, week, { fromControl: true, league });
  });

  if (els.leagueSelect) {
    els.leagueSelect.addEventListener("change", () => {
      const next = normalizeLeague(els.leagueSelect.value);
      if (next === STATE.league) return;
      setActiveLeague(next, { updateSelect: false, updateHistory: true });
      if (STATE.season && STATE.week) {
        loadAndRender(STATE.season, STATE.week, { fromControl: true, league: next });
      }
    });
  }

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
  const paramLeagueRaw = params.get("league");

  const storedSelection = loadStoredSelection();
  const storedLastGame = loadStoredLastGame();
  const storedLeague = storedSelection ? storedSelection.league : null;
  const fallbackLeague = storedLeague ?? loadStoredLeague();
  const initialLeague = paramLeagueRaw
    ? normalizeLeague(paramLeagueRaw)
    : fallbackLeague ?? DEFAULT_LEAGUE;
  setActiveLeague(initialLeague, {
    updateSelect: true,
    updateHistory: Boolean(paramLeagueRaw) || initialLeague !== DEFAULT_LEAGUE,
  });

  if (paramSeason) els.seasonInput.value = paramSeason;
  if (paramWeek) els.weekInput.value = paramWeek;

  const available = await listAvailableWeeks(STATE.league);
  if (available.length) {
    console.log(
      `Available weeks (league=${STATE.league.toUpperCase()}):`,
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
      console.warn(`WARN: No data found for season ${paramSeason} in directory listing.`);
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

  const selectionMatchesLeague =
    storedSelection &&
    normalizeLeague(storedSelection.league ?? DEFAULT_LEAGUE) === STATE.league;
  const lastMatchesLeague =
    storedLastGame &&
    normalizeLeague(storedLastGame.league ?? DEFAULT_LEAGUE) === STATE.league;

  if (!target && selectionMatchesLeague) {
    console.log("Fallback to stored selection", storedSelection);
    target = { season: storedSelection.season, week: storedSelection.week };
  }

  if (!target) {
    setStatus("No season/week found. Please enter values.");
    return;
  }

  els.seasonInput.value = target.season;
  els.weekInput.value = target.week;

  preparePendingScroll(
    target.season,
    target.week,
    paramGame,
    selectionMatchesLeague ? storedSelection : null,
    lastMatchesLeague ? storedLastGame : null
  );

  const loaded = await loadAndRender(target.season, target.week);
  if (!loaded) {
    if (
      selectionMatchesLeague &&
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

async function listAvailableWeeks(league) {
  try {
    const normalized = normalizeLeague(league);
    const listingPath = normalized === "cfb" ? "out/cfb/" : "out/";
    const url = siteUrl(listingPath);
    const res = await fetch(url, { cache: "no-store" });
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
    console.warn("WARN: Unable to parse directory listing.", err);
    return [];
  }
}

async function loadAndRender(season, week, options = {}) {
  const { fromControl = false, league: leagueOverride = null } = options;
  const league = normalizeLeague(leagueOverride ?? STATE.league ?? DEFAULT_LEAGUE);
  const shouldUpdateLeagueHistory = fromControl || league !== DEFAULT_LEAGUE;
  setActiveLeague(league, {
    updateSelect: true,
    updateHistory: shouldUpdateLeagueHistory,
  });
  const paths = buildWeekPaths(league, season, week);
  STATE.weekPaths = paths;
  STATE.pathsLogged = false;
  setStatus("Loading games...");
  const result = await loadGames(paths);
  if (!result.success) {
    setStatus(result.message);
    console.log(
      `FAIL: Load season=${season} week=${week} league=${league.toUpperCase()} (${result.message})`
    );
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
  if (isCFBLeague()) {
    STATE.teamNumberMap = buildCfbTeamNumberMap(safeRows);
    syncCfbCoverage(safeRows);
  } else {
    STATE.teamNumberMap = null;
    STATE.cfbCoverage = null;
    if (STATE.cfbBlockActive && els.status?.textContent === CFB_BLOCK_MESSAGE) {
      setStatus("");
    }
    STATE.cfbBlockActive = false;
  }
  STATE.loggedCfbExample = false;

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
  persistSelection({
    league: STATE.league,
    season,
    week,
    last_game_key: STATE.highlightedGameKey ?? null,
  });

  if (updateHistory) {
    const url = new URL(window.location.href);
    url.searchParams.set("season", season);
    url.searchParams.set("week", week);
    if (STATE.league === DEFAULT_LEAGUE) {
      url.searchParams.delete("league");
    } else {
      url.searchParams.set("league", STATE.league);
    }
    window.history.replaceState(null, "", url.toString());
  }

  return true;
}

async function loadGames(paths) {
  const resourcePath = paths.gamesJsonl;
  const url = new URL(resourcePath, window.location.origin);
  if (!STATE.pathsLogged) {
    console.log(
      `[league] Week View -> ${paths.league.toUpperCase()} base=${paths.baseDir} games=${resourcePath}`
    );
    STATE.pathsLogged = true;
  }

  const leagueLower = (paths.league || "").toLowerCase();
  const cacheKey = weekCacheKey(leagueLower, paths.season, paths.week);
  let records = readWeekCache(leagueLower, paths.season, paths.week);
  if (records && records.length) {
    const staleSchema = records.some(
      (row) =>
        !row ||
        !Object.prototype.hasOwnProperty.call(row, "home_pr") ||
        !Object.prototype.hasOwnProperty.call(row, "rating_vs_odds")
    );
    if (staleSchema) {
      console.warn("Cache stale (missing odds/ratings fields): refetching from network");
      try {
        localStorage.removeItem(cacheKey);
      } catch {
        // ignore storage errors
      }
      records = null;
    }
  }
  if (records && records.length && leagueLower === "cfb") {
    const sample = records.slice(0, Math.min(25, records.length));
    const covered =
      sample.length === 0
        ? 0
        : sample.reduce(
            (count, row) => count + (hasMetricsCoverageCFB(row) ? 1 : 0),
            0
          );
    const coverage = sample.length === 0 ? 0 : covered / sample.length;
    if (sample.length > 0 && coverage < 0.6) {
      console.warn("Cache stale (missing CFB odds/metrics): refetching from network");
      try {
        localStorage.removeItem(cacheKey);
      } catch {
        // ignore failures
      }
      records = null;
    } else if (records) {
      console.info("Cache OK (v3): using cached games:", records.length);
    }
  }

  if (records && records.length) {
    if (leagueLower !== "cfb") {
      console.info("Cache OK (v3): using cached games:", records.length);
    }
    return {
      success: true,
      rows: records,
      count: records.length,
      message: "Loaded from cache",
      sourcePath: resourcePath,
    };
  }

  try {
    const res = await fetch(url.toString(), { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const text = await res.text();
    const parsed = parseJsonl(text);
    if (parsed.records.length) {
      writeWeekCache(leagueLower, paths.season, paths.week, parsed.records);
    }
    return {
      success: true,
      rows: parsed.records,
      count: parsed.count,
      message: parsed.count ? "Loaded" : "No games in file",
      sourcePath: resourcePath,
    };
  } catch (err) {
    console.error(
      `FAIL: loadGames season=${paths.season} week=${paths.week} league=${paths.league}`,
      err
    );
    return { success: false, message: `Failed to load: ${err.message}`, sourcePath: resourcePath };
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

  const diffHeader = document.querySelector("th.diff-header");
  if (diffHeader) {
    const sample = Array.isArray(rows) ? rows.find((row) => rowHasExplicitHfa(row)) : null;
    if (sample) {
      const hfaValue = getHfa(sample, STATE.league);
      const display = formatNumber(hfaValue, { decimals: 1 });
      diffHeader.innerHTML = `(HFA=${display})<br>Diff`;
    } else {
      diffHeader.textContent = "Diff";
    }
  }

  if (isCFBLeague() && STATE.cfbBlockActive) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 16;
    td.textContent = CFB_BLOCK_MESSAGE;
    tr.appendChild(td);
    tbody.appendChild(tr);
    updateFilterMeta(0, STATE.allRows.length);
    highlightRow(null);
    return;
  }

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
  if (isCFBLeague()) {
    return STATE.teamNumberMap ?? new Map();
  }
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
  if (isCFBLeague()) {
    const label = resolveTeamName(row, side);
    if (!label) return MISSING_VALUE;
    const num = teamNumberFromDisplay(label);
    if (!num) {
      warnOnce(`no-teamnum-cfb:${label}`, `No CFB Team # for label='${label}'`);
      return placeholderTeamNumber();
    }
    return num;
  }
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

function bindGroupHoverAndFocus(rows) {
  const add = () => rows.forEach(r => r.classList.add('hover'));
  const remove = () => rows.forEach(r => r.classList.remove('hover'));
  rows.forEach(r => {
    r.addEventListener('mouseenter', add);
    r.addEventListener('mouseleave', remove);
    r.addEventListener('focusin', add);
    r.addEventListener('focusout', remove);
  });
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
  const cfb = isCFBLeague();
  const metrics = cfb ? computeFavoredMetrics(row) : null;
  if (cfb) {
    logCfbMetrics(row, metrics);
  }

  const stripe = groupIndex % 2 === 0 ? "group-even" : "group-odd";

  const trTop = document.createElement("tr");
  const trBot = document.createElement("tr");
  [trTop, trBot].forEach((tr) => {
    tr.dataset.gameKey = row.game_key ?? "";
    tr.classList.add("group", stripe);
    tr.tabIndex = 0;
    tr.addEventListener("click", () => openGame(row.game_key));
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter") openGame(row.game_key);
      else if (e.key === "ArrowRight" && e.ctrlKey) openGame(row.game_key, { newTab: true });
    });
    bindGroupHoverAndFocus([trTop, trBot]);
  });

  appendCell(trTop, fmtDatePT(iso), { rowspan: 2 });
  appendCell(trTop, fmtTimePT(iso), { rowspan: 2 });
  appendCell(trTop, gameNumber, { rowspan: 2 });

  appendCell(trTop, formatTeamNumber(row, "away"));
  appendCell(trTop, resolveTeamName(row, "away"));
  appendCell(trTop, formatOddsCell(row, "away"));
  appendCell(
    trTop,
    isFavRow(row, "away") ? formatNumber(row.total, { decimals: 1 }) : ""
  );
  appendCell(trTop, teamRecord(row, "away"));
  appendCell(
    trTop,
    cfb ? formatNumber(row.away_pr, { decimals: 2 }) : formatNumber(row.away_pr),
    { numeric: true }
  );
  appendCell(
    trTop,
    cfb
      ? formatFavoredMetric(metrics, "away", metrics?.diff, { decimals: 1, signed: true })
      : isFavRow(row, "away")
      ? formatNumber(row.rating_diff_favored_team, { decimals: 1, signed: true })
      : ""
  );
  appendCell(
    trTop,
    cfb
      ? formatFavoredMetric(metrics, "away", metrics?.rvo, { decimals: 1, signed: true })
      : isFavRow(row, "away")
      ? formatNumber(row.rating_vs_odds, { decimals: 1, signed: true })
      : ""
  );
  appendCell(
    trTop,
    cfb ? formatNumber(row.away_sos, { decimals: 2 }) : formatNumber(row.away_sos),
    { numeric: true }
  );
  const sosDiffAwayCell = appendCell(
    trTop,
    cfb ? sosDiffCell(row, "away") : sosDiffForRow(row.home_sos, row.away_sos, "away"),
    { numeric: true }
  );
  if (cfb && sosDiffAwayCell) {
    sosDiffAwayCell.dataset.col = "sosDiff";
  }

  appendCell(trBot, formatTeamNumber(row, "home"));
  appendCell(trBot, resolveTeamName(row, "home"));
  appendCell(trBot, formatOddsCell(row, "home"));
  appendCell(
    trBot,
    isFavRow(row, "home") ? formatNumber(row.total, { decimals: 1 }) : ""
  );
  appendCell(trBot, teamRecord(row, "home"));
  appendCell(
    trBot,
    cfb ? formatNumber(row.home_pr, { decimals: 2 }) : formatNumber(row.home_pr),
    { numeric: true }
  );
  appendCell(
    trBot,
    cfb
      ? formatFavoredMetric(metrics, "home", metrics?.diff, { decimals: 1, signed: true })
      : isFavRow(row, "home")
      ? formatNumber(row.rating_diff_favored_team, { decimals: 1, signed: true })
      : ""
  );
  appendCell(
    trBot,
    cfb
      ? formatFavoredMetric(metrics, "home", metrics?.rvo, { decimals: 1, signed: true })
      : isFavRow(row, "home")
      ? formatNumber(row.rating_vs_odds, { decimals: 1, signed: true })
      : ""
  );
  appendCell(
    trBot,
    cfb ? formatNumber(row.home_sos, { decimals: 2 }) : formatNumber(row.home_sos),
    { numeric: true }
  );
  const sosDiffHomeCell = appendCell(
    trBot,
    cfb ? sosDiffCell(row, "home") : sosDiffForRow(row.home_sos, row.away_sos, "home"),
    { numeric: true }
  );
  if (cfb && sosDiffHomeCell) {
    sosDiffHomeCell.dataset.col = "sosDiff";
  }

  return [trTop, trBot];
}


function buildTeamRow(row, side, gameNumber) {
  const iso = row?.kickoff_iso_utc ?? null;
  const favored = isFavRow(row, side);
  const cfb = isCFBLeague();
  const metrics = cfb ? computeFavoredMetrics(row) : null;
  const currentPrValue = side === "home" ? row.home_pr : row.away_pr;
  const sosValue = side === "home" ? row.home_sos : row.away_sos;

  const total = favored
    ? formatNumber(row.total, { decimals: 1 })
    : "";
  const diff = cfb
    ? formatFavoredMetric(metrics, side, metrics?.diff, { decimals: 1, signed: true })
    : favored
    ? formatNumber(row.rating_diff_favored_team, { decimals: 1, signed: true })
    : "";
  const rvo = cfb
    ? formatFavoredMetric(metrics, side, metrics?.rvo, { decimals: 1, signed: true })
    : favored
    ? formatNumber(row.rating_vs_odds, { decimals: 1, signed: true })
    : "";
  const gameValue = gameNumber ?? placeholderGameNumber();
  const teamNumber = formatTeamNumber(row, side);

  return [
    fmtDatePT(iso),
    fmtTimePT(iso),
    gameValue,
    teamNumber,
    resolveTeamName(row, side),
    formatOddsCell(row, side),
    total,
    teamRecord(row, side),
    cfb ? formatNumber(currentPrValue, { decimals: 2 }) : formatNumber(currentPrValue),
    diff,
    rvo,
    cfb ? formatNumber(sosValue, { decimals: 2 }) : formatNumber(sosValue),
    cfb ? sosDiffCell(row, side) : sosDiffForRow(row.home_sos, row.away_sos, side),
  ];
}

function resolveTeamName(row, side) {
  const sourceKey = side === "home" ? "sagarin_row_home" : "sagarin_row_away";
  const fromSagarin = row?.raw_sources?.[sourceKey]?.team;
  const raw = side === "home" ? row.home_team_raw : row.away_team_raw;
  const norm = side === "home" ? row.home_team_norm : row.away_team_norm;
  if (isCFBLeague()) {
    const base = fromSagarin ?? raw ?? norm;
    const display = toDisplayName(base, raw ?? norm);
    return display || (raw ? String(raw) : norm ? String(norm) : MISSING_VALUE);
  }
  if (fromSagarin) return fromSagarin;
  return formatTeam(norm, raw);
}

function getTeamCode(row, side) {
  const oddsRow = row?.raw_sources?.odds_row;
  const oddsKey = side === "home" ? "home_team_code" : "away_team_code";
  const oddsValue =
    oddsRow && typeof oddsRow[oddsKey] === "string" ? oddsRow[oddsKey].trim() : "";
  if (oddsValue) {
    return oddsValue.toUpperCase();
  }
  const raw = side === "home" ? row.home_team_raw : row.away_team_raw;
  const norm = side === "home" ? row.home_team_norm : row.away_team_norm;
  const base = raw ?? norm;
  if (!base) return "";
  return deriveTeamCodeFromName(base);
}

function deriveTeamCodeFromName(name) {
  if (!name) return "";
  const text = String(name).trim();
  if (!text) return "";
  const cleaned = text.replace(/[^A-Za-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
  if (!cleaned) return "";
  const upper = cleaned.toUpperCase();
  if (/^[A-Z0-9]{2,5}$/.test(upper)) {
    return upper;
  }
  const words = cleaned.split(" ");
  if (words.length === 1) {
    const token = words[0];
    if (token.length <= 4) return token.toUpperCase();
    return token.slice(0, 4).toUpperCase();
  }
  if (words.length === 2) {
    const first = words[0].slice(0, 2);
    const second = words[1].slice(0, 2);
    return (first + second).toUpperCase();
  }
  let code = "";
  words.forEach((word) => {
    if (!word) return;
    if (word.length <= 2) {
      code += word.toUpperCase();
    } else {
      code += word[0].toUpperCase();
    }
  });
  if (code.length < 3) {
    const fallback = words.join("").slice(0, 4).toUpperCase();
    return fallback;
  }
  return code.slice(0, 4);
}

function formatOddsCell(row, side) {
  if (!isFavRow(row, side)) return "";
  if (!hasNumeric(row.spread_favored_team)) return MISSING_VALUE;
  const spread = formatNumber(row.spread_favored_team, { decimals: 1, signed: true });
  // CFB: show just the signed spread (no team code)
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
  const url = siteUrl("web/game_view.html", {
    league: STATE.league,
    season: STATE.season,
    week: STATE.week,
    game_key: gameKey,
  });
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
  persistSelection({
    league: STATE.league,
    season: STATE.season,
    week: STATE.week,
    last_game_key: gameKey,
  });
  try {
    localStorage.setItem(
      LAST_GAME_KEY,
      JSON.stringify({
        league: STATE.league,
        season: STATE.season,
        week: STATE.week,
        game_key: gameKey,
      })
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
  els.gameViewLink.href = siteUrl("web/game_view.html", {
    league: STATE.league,
    season,
    week,
  });
  els.latestLink.href = siteUrl("web/week_view.html", {
    league: STATE.league && STATE.league !== DEFAULT_LEAGUE ? STATE.league : undefined,
  });
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
    "Game #",
    "Team #",
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
  if (isCFBLeague()) {
    const display = toDisplayName(raw ?? norm, raw ?? norm);
    if (display) return display;
    if (raw) return String(raw);
    if (norm) return String(norm);
    return MISSING_VALUE;
  }
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

function hasMetricsCoverageCFB(row) {
  const homePf = Number(row?.home_pf_pg);
  const awayPf = Number(row?.away_pf_pg);
  return Number.isFinite(homePf) || Number.isFinite(awayPf);
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
    const payload = { ...selection };
    payload.league = normalizeLeague(payload.league ?? STATE.league ?? DEFAULT_LEAGUE);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // ignore storage issues
  }
}

function weekCacheKey(league, season, week) {
  return `${WEEK_CACHE_PREFIX}${CACHE_VERSION}:${league}:${season}:${week}`;
}

function readWeekCache(league, season, week) {
  try {
    const raw = localStorage.getItem(weekCacheKey(league, season, week));
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : null;
    } catch {
      return null;
    }
  } catch {
    return null;
  }
}

function writeWeekCache(league, season, week, records) {
  try {
    localStorage.setItem(
      weekCacheKey(league, season, week),
      JSON.stringify(Array.isArray(records) ? records : [])
    );
  } catch {
    // ignore persistence issues
  }
}

function loadStoredSelection() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      parsed.league = normalizeLeague(parsed.league ?? DEFAULT_LEAGUE);
    }
    return parsed;
  } catch {
    return null;
  }
}

function loadStoredLastGame() {
  try {
    const raw = localStorage.getItem(LAST_GAME_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      parsed.league = normalizeLeague(parsed.league ?? DEFAULT_LEAGUE);
    }
    return parsed;
  } catch {
    return null;
  }
}
