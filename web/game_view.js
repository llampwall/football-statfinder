const els = {
  season: document.getElementById("season-input"),
  week: document.getElementById("week-input"),
  leagueSelect: document.getElementById("league-select"),
  loadGamesBtn: document.getElementById("load-games-btn"),
  loadGameBtn: document.getElementById("load-game-btn"),
  gameSelect: document.getElementById("game-select"),
  status: document.getElementById("status"),
  weekLink: document.getElementById("week-view-link"),
  header: document.getElementById("game-header"),
  teamsBlock: document.getElementById("teams-block"),
  favoriteBlock: document.getElementById("favorite-block"),
  marketBlock: document.getElementById("market-block"),
  teamStatsSection: document.getElementById("team-stats-section"),
  teamStatsBody: document.getElementById("team-stats-body"),
  scheduleCurrent: document.getElementById("schedule-current"),
  schedulePrevious: document.getElementById("schedule-previous"),
  scheduleCurrentTitle: document.getElementById("schedule-current-title"),
  schedulePreviousTitle: document.getElementById("schedule-previous-title"),
  navRow: document.getElementById("nav-row"),
  prevGameBtn: document.getElementById("prev-game-btn"),
  nextGameBtn: document.getElementById("next-game-btn"),
  navSummary: document.getElementById("nav-summary"),
  eoySection: document.getElementById("eoy-stats"),
  tableBodies: {
    home_ytd: document.getElementById("home-ytd-body"),
    away_ytd: document.getElementById("away-ytd-body"),
    home_prev: document.getElementById("home-prev-body"),
    away_prev: document.getElementById("away-prev-body"),
  },
  tableTitles: {
    home_ytd: document.getElementById("home-ytd-title"),
    away_ytd: document.getElementById("away-ytd-title"),
    home_prev: document.getElementById("home-prev-title"),
    away_prev: document.getElementById("away-prev-title"),
  },
  footer: document.getElementById("footer"),
  diagnosticsNote: document.getElementById("diagnostics-note"),
  dataStamp: document.getElementById("data-stamp"),
  statusLine: document.getElementById("status-line"),
};

const STORAGE_KEY = "game-view:last-selection";
const WEEK_CACHE_PREFIX = "week-view:games:";
const CACHE_VERSION = "v4";
const LEAGUE_STORAGE_KEY = "game-view:league";
const DEFAULT_LEAGUE = "nfl";
const VALID_LEAGUES = new Set(["nfl", "cfb"]);
const NUMERIC_KEYS = {
  home: [
    "home_ry_pg",
    "home_py_pg",
    "home_ty_pg",
    "home_rush_rank",
    "home_pass_rank",
    "home_tot_off_rank",
    "home_ry_allowed_pg",
    "home_py_allowed_pg",
    "home_ty_allowed_pg",
    "home_rush_def_rank",
    "home_pass_def_rank",
    "home_tot_def_rank",
    "home_to_margin_pg",
    "home_pf_pg",
    "home_pa_pg",
  ],
  away: [
    "away_ry_pg",
    "away_py_pg",
    "away_ty_pg",
    "away_rush_rank",
    "away_pass_rank",
    "away_tot_off_rank",
    "away_ry_allowed_pg",
    "away_py_allowed_pg",
    "away_ty_allowed_pg",
    "away_rush_def_rank",
    "away_pass_def_rank",
    "away_tot_def_rank",
    "away_to_margin_pg",
    "away_pf_pg",
    "away_pa_pg",
  ],
};

const STATE = {
  games: new Map(),
  season: null,
  week: null,
  league: DEFAULT_LEAGUE,
  deepLinkUsed: false,
  autoFromStorage: false,
  sortedKeys: [],
  currentGameKey: null,
  weekSourcePath: null,
  lastLoadedAt: null,
  eoyCache: new Map(),
  weekPaths: null,
  pathsLogged: false,
};

const REQUIRED_FAVORITE_KEYS = [
  "favored_side",
  "spread_favored_team",
  "rating_diff_favored_team",
  "rating_vs_odds",
];

const MISSING_VALUE = "\u2014";
let timezoneLogged = false;

const TEAM_DATA = [
  { city: "Arizona", nickname: "Cardinals", aliases: ["ari", "arz", "arizona", "cardinals", "arizona cardinals"] },
  { city: "Atlanta", nickname: "Falcons", aliases: ["atl", "atlanta", "falcons", "atlanta falcons"] },
  { city: "Baltimore", nickname: "Ravens", aliases: ["bal", "baltimore", "ravens", "baltimore ravens"] },
  { city: "Buffalo", nickname: "Bills", aliases: ["buf", "buffalo", "bills", "buffalo bills"] },
  { city: "Carolina", nickname: "Panthers", aliases: ["car", "caro", "carolina", "panthers", "carolina panthers"] },
  { city: "Chicago", nickname: "Bears", aliases: ["chi", "chicago", "bears", "chicago bears"] },
  { city: "Cincinnati", nickname: "Bengals", aliases: ["cin", "cincinnati", "bengals", "cincinnati bengals"] },
  { city: "Cleveland", nickname: "Browns", aliases: ["cle", "cleveland", "browns", "cleveland browns"] },
  { city: "Dallas", nickname: "Cowboys", aliases: ["dal", "dallas", "cowboys", "dallas cowboys"] },
  { city: "Denver", nickname: "Broncos", aliases: ["den", "denver", "broncos", "denver broncos"] },
  { city: "Detroit", nickname: "Lions", aliases: ["det", "detroit", "lions", "detroit lions"] },
  { city: "Green Bay", nickname: "Packers", aliases: ["gb", "gnb", "green bay", "packers", "green bay packers"] },
  { city: "Houston", nickname: "Texans", aliases: ["hou", "houston", "texans", "houston texans"] },
  { city: "Indianapolis", nickname: "Colts", aliases: ["ind", "indianapolis", "colts", "indianapolis colts"] },
  { city: "Jacksonville", nickname: "Jaguars", aliases: ["jax", "jac", "jacksonville", "jags", "jaguars", "jacksonville jaguars"] },
  { city: "Kansas City", nickname: "Chiefs", aliases: ["kc", "kan", "kcc", "kansas city", "chiefs", "kansas city chiefs"] },
  { city: "Las Vegas", nickname: "Raiders", aliases: ["lv", "lvr", "las vegas", "raiders", "las vegas raiders", "oakland raiders", "oakland", "oak"] },
  { city: "Los Angeles", nickname: "Chargers", aliases: ["lac", "lax", "los angeles chargers", "la chargers", "chargers", "san diego", "san diego chargers", "sd", "sdc"] },
  { city: "Los Angeles", nickname: "Rams", aliases: ["lar", "la", "los angeles", "los angeles rams", "la rams", "st louis rams", "stl", "rams"] },
  { city: "Miami", nickname: "Dolphins", aliases: ["mia", "miami", "dolphins", "miami dolphins"] },
  { city: "Minnesota", nickname: "Vikings", aliases: ["min", "minn", "minnesota", "vikings", "minnesota vikings"] },
  { city: "New England", nickname: "Patriots", aliases: ["ne", "nwe", "new england", "patriots", "new england patriots"] },
  { city: "New Orleans", nickname: "Saints", aliases: ["no", "nor", "new orleans", "saints", "new orleans saints"] },
  { city: "New York", nickname: "Giants", aliases: ["nyg", "new york giants", "giants", "ny giants"] },
  { city: "New York", nickname: "Jets", aliases: ["nyj", "new york jets", "jets", "ny jets"] },
  { city: "Philadelphia", nickname: "Eagles", aliases: ["phi", "philadelphia", "eagles", "philadelphia eagles"] },
  { city: "Pittsburgh", nickname: "Steelers", aliases: ["pit", "pittsburgh", "steelers", "pittsburgh steelers"] },
  { city: "San Francisco", nickname: "49ers", aliases: ["sf", "sfo", "san francisco", "49ers", "niners", "san francisco 49ers"] },
  { city: "Seattle", nickname: "Seahawks", aliases: ["sea", "seattle", "seahawks", "seattle seahawks"] },
  { city: "Tampa Bay", nickname: "Buccaneers", aliases: ["tb", "tbb", "tampa bay", "buccaneers", "bucs", "tampa bay buccaneers"] },
  { city: "Tennessee", nickname: "Titans", aliases: ["ten", "oti", "tennessee", "titans", "tennessee titans"] },
  {
    city: "Washington",
    nickname: "Commanders",
    aliases: ["was", "wsh", "wft", "washington", "commanders", "washington commanders", "washington football team"],
  },
];

const TEAM_ALIAS_DISPLAY = TEAM_DATA.reduce((acc, entry) => {
  const display = `${entry.nickname}, ${entry.city}`;
  entry.aliases.forEach((alias) => {
    const key = alias.trim().toLowerCase();
    acc[key] = display;
    const compact = key.replace(/\s+/g, "");
    acc[compact] = display;
  });
  return acc;
}, {});

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
    baseDir,
    gamesJsonl: `${baseDir}/games_week_${season}_${week}.jsonl`,
    sidecarPath(gameKey) {
      return `${baseDir}/game_schedules/${gameKey}.json`;
    },
  };
}

function attachListeners() {
  if (els.leagueSelect) {
    els.leagueSelect.addEventListener("change", () => {
      const next = normalizeLeague(els.leagueSelect.value);
      if (next === STATE.league) return;
      const hasInputs = Boolean(coerceInt(els.season.value) && coerceInt(els.week.value));
      setActiveLeague(next, { updateSelect: false, updateHistory: true });
      STATE.games.clear();
      STATE.weekPaths = null;
      STATE.pathsLogged = false;
      syncWeekLink();
      if (hasInputs) {
        loadGames(STATE.currentGameKey);
      }
    });
  }

  els.loadGamesBtn.addEventListener("click", () => {
    loadGames();
  });

  els.loadGameBtn.addEventListener("click", () => {
    const key = els.gameSelect.value;
    if (!key) {
      setStatus("Choose a game first.");
      return;
    }
    loadSingleGame(key);
  });

  ["input", "change"].forEach((evt) => {
    els.season.addEventListener(evt, syncWeekLink);
    els.week.addEventListener(evt, syncWeekLink);
  });

  els.prevGameBtn.addEventListener("click", () => {
    navigateRelative(-1);
  });

  els.nextGameBtn.addEventListener("click", () => {
    navigateRelative(1);
  });

  els.weekLink.addEventListener("click", () => {
    rememberWeekRow(STATE.currentGameKey);
  });

  els.weekLink.addEventListener("click", () => {
    rememberWeekRow(STATE.currentGameKey, null);
  });

  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (event.defaultPrevented) return;
    if (event.key === "ArrowLeft") {
      navigateRelative(-1);
    } else if (event.key === "ArrowRight") {
      navigateRelative(1);
    }
  });
}

function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const stored = safeParseLocalStorage();
  const paramLeagueRaw = params.get("league");
  const storedLeague = stored?.league ?? loadStoredLeague();
  const initialLeague = paramLeagueRaw
    ? normalizeLeague(paramLeagueRaw)
    : storedLeague ?? DEFAULT_LEAGUE;
  setActiveLeague(initialLeague, { updateSelect: true, updateHistory: Boolean(paramLeagueRaw) });

  if (params.has("season") || params.has("week") || params.has("game_key")) {
    STATE.deepLinkUsed = true;
    console.log("Deep link detected -> auto load path.");
  }

  const storedMatchesLeague =
    stored && normalizeLeague(stored.league ?? DEFAULT_LEAGUE) === STATE.league;
  const initialSeason =
    numericFromParam(params.get("season")) ?? (storedMatchesLeague ? stored?.season ?? "" : "");
  const initialWeek =
    numericFromParam(params.get("week")) ?? (storedMatchesLeague ? stored?.week ?? "" : "");
  const storedGameKey = storedMatchesLeague ? stored?.game_key ?? "" : "";
  const initialGameKey = params.get("game_key") ?? storedGameKey;
  STATE.autoFromStorage = !STATE.deepLinkUsed && storedMatchesLeague && Boolean(storedGameKey);

  els.season.value = initialSeason;
  els.week.value = initialWeek;
  syncWeekLink();

  if (initialSeason && initialWeek) {
    loadGames(initialGameKey);
  }
}

async function loadGames(autoGameKey) {
  const season = coerceInt(els.season.value);
  const week = coerceInt(els.week.value);
  if (!season || !week) {
    setStatus("Season and week required.");
    return;
  }

  const previousPath = STATE.weekPaths ? STATE.weekPaths.gamesJsonl : null;
  const paths = buildWeekPaths(STATE.league, season, week);
  STATE.weekPaths = paths;
  if (previousPath !== paths.gamesJsonl) {
    STATE.pathsLogged = false;
  }
  if (!STATE.pathsLogged) {
    console.log(
      `[league] Game View -> ${paths.league.toUpperCase()} base=${paths.baseDir} games=${paths.gamesJsonl}`
    );
    STATE.pathsLogged = true;
  }

  STATE.currentGameKey = null;

  const cached = readWeekCache(paths.league, season, week);
  let records = Array.isArray(cached) ? cached : null;
  let fromCache = false;

  const leagueLower = (paths.league || "").toLowerCase();
  const cacheKey = weekCacheKey(paths.league, season, week);
  if (records && records.length > 0) {
    const staleSchema = records.some(
      (row) =>
        !row ||
        !Object.prototype.hasOwnProperty.call(row, "home_pr") ||
        !Object.prototype.hasOwnProperty.call(row, "rating_vs_odds")
    );
    if (staleSchema) {
      console.warn("Cache stale (missing odds/ratings fields): refetching from network");
      try {
        window.localStorage.removeItem(cacheKey);
      } catch {
        // ignore remove failures
      }
      records = null;
    }
  }
  if (records && records.length > 0 && leagueLower === "cfb") {
    const sample = records.slice(0, Math.min(25, records.length));
    const covered =
      sample.length === 0
        ? 0
        : sample.reduce(
            (count, row) => count + (hasFavoriteCoverageCFB(row) ? 1 : 0),
            0
          );
    const coverage = sample.length === 0 ? 0 : covered / sample.length;
    if (sample.length > 0 && coverage < 0.6) {
      console.warn("Cache stale (missing CFB odds/metrics): refetching from network");
      try {
        window.localStorage.removeItem(cacheKey);
      } catch {
        // ignore remove failures
      }
      records = null;
    } else if (records) {
      console.info("Cache OK (v3): using cached games:", records.length);
    }
  }

  if (!records || records.length === 0) {
    const relPath = paths.gamesJsonl;
    const fetchPath = `../${relPath}`;
    const url = new URL(fetchPath, window.location.href);
    setStatus("Loading games.");
    try {
      const res = await fetch(url.toString());
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const text = await res.text();
      const parsed = parseJsonLines(text);
      records = parsed.records;
      console.log(`${parsed.count >= 1 ? "PASS" : "FAIL"}: Games loaded (count=${parsed.count})`);
      if (parsed.count === 0) {
        setStatus("No games found.");
        return;
      }
      writeWeekCache(paths.league, season, week, records);
      STATE.weekSourcePath = relPath;
    } catch (err) {
      console.log(
        `FAIL: Games loaded (${season} week ${week} league=${paths.league.toUpperCase()})`,
        err
      );
      setStatus(`Failed to load games (${err.message})`);
      return;
    }
  } else {
    fromCache = true;
    if (leagueLower !== "cfb") {
      console.info("Cache OK (v3): using cached games:", records.length);
    }
    setStatus("Loaded games from cache.");
    STATE.weekSourcePath = paths.gamesJsonl;
  }

  if (!Array.isArray(records) || records.length === 0) {
    setStatus("No games found.");
    return;
  }

  STATE.games = new Map(records.map((row) => [row.game_key, row]));
  STATE.season = season;
  STATE.week = week;
  populateGameSelect(records, autoGameKey);
  if (!fromCache) {
    setStatus(`Loaded ${records.length} games.`);
  }

  if ((STATE.deepLinkUsed || STATE.autoFromStorage) && autoGameKey) {
    console.log(
      STATE.deepLinkUsed ? "Deep link auto-loading game:" : "Auto-loading last viewed game:",
      autoGameKey
    );
    loadSingleGame(autoGameKey);
  }
  STATE.deepLinkUsed = false;
  STATE.autoFromStorage = false;
}

function populateGameSelect(records, autoSelectKey) {
  els.gameSelect.innerHTML = "";
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "Select a game";
  els.gameSelect.appendChild(defaultOption);

  const sorted = records
    .slice()
    .sort((a, b) => (a.kickoff_iso_utc ?? "").localeCompare(b.kickoff_iso_utc ?? ""));

  STATE.sortedKeys = sorted.map((game) => game.game_key);

  sorted.forEach((game) => {
    const option = document.createElement("option");
    option.value = game.game_key;
    const home = getTeamDisplayName(game, "home");
    const away = getTeamDisplayName(game, "away");
    option.textContent = `${home} vs ${away} (${game.game_key})`;
    if (autoSelectKey && autoSelectKey === game.game_key) {
      option.selected = true;
    }
    els.gameSelect.append(option);
  });

  updateNavigationControls();
}

async function loadSingleGame(gameKey) {
  const game = STATE.games.get(gameKey);
  console.log(`${game ? "PASS" : "FAIL"}: Game row found (${gameKey})`);
  if (!game) {
    setStatus("Game not in loaded set.");
    return;
  }
  const paths = STATE.weekPaths ?? buildWeekPaths(STATE.league, STATE.season, STATE.week);
  const sidecarRel = paths.sidecarPath(gameKey);
  const sidecarPath = `../${sidecarRel}`;
  const url = new URL(sidecarPath, window.location.href);
  let sidecar;
  let sidecarOk = false;
  try {
    const res = await fetch(url.toString());
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    let raw = await res.text();
    raw = raw.trim();
    if (raw.startsWith("\uFEFF")) {
      raw = raw.slice(1);
    }
    const sanitized = raw.replace(/([-+]?Infinity|\bNaN\b)/gi, "null");
    sidecar = JSON.parse(sanitized);
    console.log(`PASS: Sidecar loaded (${sidecarPath})`);
    sidecarOk = true;
  } catch (err) {
    console.error(`FAIL: Sidecar loaded (${sidecarPath})`, err);
    setStatus(`Sidecar failed (${err.message})`);
    return;
  }

  const teamNames = {
    home: getTeamDisplayName(game, "home"),
    away: getTeamDisplayName(game, "away"),
  };

  STATE.currentGameKey = gameKey;
  if (els.gameSelect.value !== gameKey) {
    els.gameSelect.value = gameKey;
  }

  STATE.lastLoadedAt = new Date();

  const leagueLowerCurrent = (STATE.weekPaths?.league ?? STATE.league ?? DEFAULT_LEAGUE).toLowerCase();
  renderHeader(game, teamNames);
  if (leagueLowerCurrent === "cfb") {
    const hasOdds =
      hasNumeric(game?.spread_favored_team) || hasNumeric(game?.spread_home_relative);
    console.info(`[CFB Game] odds ${hasOdds ? "present" : "missing"} for ${gameKey}`);
    renderFavoriteBox(game);
  }
  renderTeamStats(game, sidecar, teamNames);
  renderTables(game, sidecar, teamNames);
  const eoyStatus = await injectEOYIfAvailable();
  if ((STATE.weekPaths?.league ?? STATE.league) === "cfb") {
    const eoyOk = eoyStatus && eoyStatus.data && (eoyStatus.data.home || eoyStatus.data.away) ? "OK" : "SKIP";
    console.info(
      `[CFB Game] game_key=${gameKey} sidecar=${sidecarOk ? "OK" : "MISS"} eoy=${eoyOk} | season=${STATE.season} week=${STATE.week}`
    );
  }

  els.header.classList.remove("hidden");
  els.teamStatsSection.classList.remove("hidden");
  els.scheduleCurrent.classList.remove("hidden");
  els.schedulePrevious.classList.remove("hidden");
  els.footer.classList.remove("hidden");

  const favCheck = REQUIRED_FAVORITE_KEYS.reduce((acc, key) => {
    acc[key] = Object.prototype.hasOwnProperty.call(game, key);
    return acc;
  }, {});
  const hasAllFavoriteKeys = Object.values(favCheck).every(Boolean);
  console.log(`${hasAllFavoriteKeys ? "PASS" : "FAIL"}: Favorite fields`, favCheck);

  const coverCounts = coverageCounts(game);
  const homePass = coverCounts.home >= 10;
  const awayPass = coverCounts.away >= 10;
  console.log(`${homePass && awayPass ? "PASS" : "FAIL"}: Team stats coverage`, {
    home_present: coverCounts.home,
    away_present: coverCounts.away,
  });
  if ((STATE.weekPaths?.league ?? STATE.league) === "cfb" && (!homePass || !awayPass)) {
    console.warn("CFB snapshot metrics missing fields", coverCounts);
  }

  setStatus("Game loaded.");
  els.diagnosticsNote.textContent = `Diagnostics: favorite fields ${hasAllFavoriteKeys ? "OK" : "missing"}, coverage home ${coverCounts.home}, away ${coverCounts.away}`;
  els.dataStamp.textContent = `Sidecar: ${sidecarRel}`;

  rememberWeekRow(gameKey, sidecarRel);
  persistSelection({
    league: STATE.league,
    season: STATE.season,
    week: STATE.week,
    game_key: gameKey,
  });
  rememberWeekRow(gameKey, sidecarRel);
  updateUrlWithSelection(gameKey);
  updateNavigationControls();
}

function renderHeader(game, teamNames) {
  const homeName = teamNames.home;
  const awayName = teamNames.away;
  const kickoffPT = fmtKickoffPT(game.kickoff_iso_utc);
  const kickoffUTC = formatKickoff(game.kickoff_iso_utc);
  const kickoffTitle =
    kickoffUTC && kickoffUTC !== MISSING_VALUE ? ` title="UTC: ${kickoffUTC}"` : "";

  els.teamsBlock.innerHTML = `
    <h2>${homeName} vs ${awayName}</h2>
    <div class="meta-line">Kickoff (Pacific): <span${kickoffTitle}>${kickoffPT}</span></div>
    <div class="meta-line">Game key: ${fallback(game.game_key)}</div>
  `;

  const league = (STATE.weekPaths?.league ?? STATE.league ?? DEFAULT_LEAGUE).toLowerCase();
  if (league !== "cfb") {
    const favoredSide = (game.favored_side || "").toUpperCase();
    const favTeam =
      favoredSide === "HOME" ? homeName : favoredSide === "AWAY" ? awayName : null;
    let favoredLine = MISSING_VALUE;
    if (favTeam) {
      if (hasNumeric(game.spread_favored_team)) {
        const magnitude = Math.abs(Number(game.spread_favored_team));
        const spread = formatNumber(magnitude, { decimals: 1 });
        favoredLine = spread !== MISSING_VALUE ? `${favTeam} (\u2212${spread})` : favTeam;
      } else {
        favoredLine = favTeam;
      }
    }
    const snapshotIso = typeof game.snapshot_at === "string" ? game.snapshot_at : null;
    const snapshotPT = snapshotIso ? fmtKickoffPT(snapshotIso) : MISSING_VALUE;
    const snapshotUTC = snapshotIso ? formatKickoff(snapshotIso) : null;
    const snapshotTitle =
      snapshotUTC && snapshotUTC !== MISSING_VALUE ? ` title="UTC: ${snapshotUTC}"` : "";
    els.favoriteBlock.innerHTML = `
      <h3>Favorite & Spread</h3>
      <div class="meta-line">Favored: ${favoredLine}</div>
      <div class="meta-line">Odds source: ${fallback(game.odds_source)}</div>
      <div class="meta-line">Snapshot: <span${snapshotTitle}>${snapshotPT}</span></div>
    `;

    els.marketBlock.innerHTML = `
      <h3>Market & Ratings</h3>
      <div class="meta-line">PR Diff (favored): ${formatNumber(game.rating_diff_favored_team, {
        decimals: 1,
        signed: true,
      })}</div>
      <div class="meta-line">Rating vs Odds: ${formatNumber(game.rating_vs_odds, {
        decimals: 1,
        signed: true,
      })}</div>
      <div class="meta-line">Total: ${formatNumber(game.total, { decimals: 1 })}</div>
    `;
  }
}

function ensureFavoriteElements() {
  if (els.favoriteFavored) return;
  els.favoriteBlock.innerHTML = `
    <h3>Favorite & Spread</h3>
    <div class="meta-line">Favored: <span id="favorite-favored">${MISSING_VALUE}</span></div>
    <div class="meta-line">Odds source: <span id="favorite-source">${MISSING_VALUE}</span></div>
    <div class="meta-line">Snapshot: <span id="favorite-snapshot">${MISSING_VALUE}</span></div>
    <div class="meta-line">Total: <span id="favorite-total">${MISSING_VALUE}</span></div>
  `;
  els.favoriteFavored = document.getElementById("favorite-favored");
  els.favoriteSource = document.getElementById("favorite-source");
  els.favoriteSnapshot = document.getElementById("favorite-snapshot");
  els.favoriteTotal = document.getElementById("favorite-total");
}

function ensureMarketElements() {
  if (els.marketPrDiff) return;
  els.marketBlock.innerHTML = `
    <h3>Market & Ratings</h3>
    <div class="meta-line">PR Diff (favored): <span id="market-pr-diff">${MISSING_VALUE}</span></div>
    <div class="meta-line">Rating vs Odds: <span id="market-rvo">${MISSING_VALUE}</span></div>
  `;
  els.marketPrDiff = document.getElementById("market-pr-diff");
  els.marketRvo = document.getElementById("market-rvo");
}

function renderFavoriteBox(game) {
  ensureFavoriteElements();
  ensureMarketElements();

  const favored = (game.favored_side || "").toUpperCase();
  const favCode =
    favored === "HOME"
      ? teamShortCode(game, "home")
      : favored === "AWAY"
      ? teamShortCode(game, "away")
      : null;

  let favoredLine = MISSING_VALUE;
  if (favCode && hasNumeric(game.spread_favored_team)) {
    const magnitude = Math.abs(Number(game.spread_favored_team));
    const spreadTxt = formatNumber(magnitude, { decimals: 1 });
    favoredLine = spreadTxt !== MISSING_VALUE ? `${favCode} \u2212${spreadTxt}` : favCode;
  } else if (favCode) {
    favoredLine = favCode;
  }
  els.favoriteFavored.textContent = favoredLine ?? MISSING_VALUE;

  els.favoriteSource.textContent = fallback(game.odds_source);
  const snapshotIso = typeof game.snapshot_at === "string" ? game.snapshot_at : null;
  els.favoriteSnapshot.textContent = snapshotIso ? fmtKickoffPT(snapshotIso) : MISSING_VALUE;
  els.favoriteTotal.textContent = hasNumeric(game.total)
    ? formatNumber(Number(game.total), { decimals: 1 })
    : MISSING_VALUE;

  els.marketPrDiff.textContent = MISSING_VALUE;
  els.marketRvo.textContent = MISSING_VALUE;
}

function renderTeamStats(game, sidecar, teamNames) {
  const rows = [
    { prefix: "home", label: teamNames.home },
    { prefix: "away", label: teamNames.away },
  ];

  els.teamStatsBody.innerHTML = "";
  const league = STATE.league ?? DEFAULT_LEAGUE;

  rows.forEach(({ prefix, label }) => {
    const tr = document.createElement("tr");
    let atsCell = "";
    if (league === "cfb") {
      atsCell = MISSING_VALUE;
    }
    const cells = [
      label,
      formatNumber(game[`${prefix}_pf_pg`], { decimals: 1 }),
      formatNumber(game[`${prefix}_pa_pg`], { decimals: 1 }),
      fallback(game[`${prefix}_su`]),
      league === "cfb" ? atsCell : "",
      formatSigned(game[`${prefix}_to_margin_pg`], { decimals: 1 }),
      formatNumber(game[`${prefix}_ry_pg`], { decimals: 1 }),
      rankOrDash(game[`${prefix}_rush_rank`]),
      formatNumber(game[`${prefix}_py_pg`], { decimals: 1 }),
      rankOrDash(game[`${prefix}_pass_rank`]),
      formatNumber(game[`${prefix}_ty_pg`], { decimals: 1 }),
      rankOrDash(game[`${prefix}_tot_off_rank`]),
      formatNumber(game[`${prefix}_ry_allowed_pg`], { decimals: 1 }),
      rankOrDash(game[`${prefix}_rush_def_rank`]),
      formatNumber(game[`${prefix}_py_allowed_pg`], { decimals: 1 }),
      rankOrDash(game[`${prefix}_pass_def_rank`]),
      formatNumber(game[`${prefix}_ty_allowed_pg`], { decimals: 1 }),
      rankOrDash(game[`${prefix}_tot_def_rank`]),
    ];

    cells.forEach((value, idx) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (idx === 0) {
        td.style.fontWeight = "600";
      }
      tr.appendChild(td);
    });

    els.teamStatsBody.appendChild(tr);
  });
}

function resolveAtsValue(game, sidecar, prefix) {
  const field = `${prefix}_ats`;
  const directRaw =
    game && game[field] !== undefined && game[field] !== null
      ? String(game[field]).trim()
      : "";
  if (directRaw) {
    return { value: directRaw, source: "row" };
  }
  const bucketKey = `${prefix}_ytd`;
  const bucket = Array.isArray(sidecar?.[bucketKey]) ? sidecar[bucketKey] : [];
  const fallbackValue =
    bucket && bucket.length > 0 ? bucket[0]?.ats ?? null : null;
  const normalized =
    fallbackValue !== null && fallbackValue !== undefined
      ? String(fallbackValue).trim()
      : "";
  if (normalized) {
    return { value: normalized, source: "sidecar" };
  }
  return { value: null, source: null };
}

function renderTables(game, sidecar, teamNames) {
  const currentSeason = game.season;
  const previousSeason = hasNumeric(currentSeason) ? currentSeason - 1 : null;

  els.scheduleCurrentTitle.textContent = hasNumeric(currentSeason)
    ? `Schedule / Scores ${currentSeason}`
    : "Schedule / Scores";
  els.schedulePreviousTitle.textContent = hasNumeric(previousSeason)
    ? `Schedule / Scores ${previousSeason}`
    : "Schedule / Scores (Prior Season)";

  els.tableTitles.home_ytd.textContent = `${teamNames.home} (Home)`;
  els.tableTitles.away_ytd.textContent = `${teamNames.away} (Away)`;
  els.tableTitles.home_prev.textContent = `${teamNames.home} (Home)`;
  els.tableTitles.away_prev.textContent = `${teamNames.away} (Away)`;

  const datasetMeta = [
    { key: "home_ytd", data: sidecar.home_ytd, body: els.tableBodies.home_ytd },
    { key: "away_ytd", data: sidecar.away_ytd, body: els.tableBodies.away_ytd },
    { key: "home_prev", data: sidecar.home_prev, body: els.tableBodies.home_prev },
    { key: "away_prev", data: sidecar.away_prev, body: els.tableBodies.away_prev },
  ];

  const counts = {};

  datasetMeta.forEach(({ key, data, body }) => {
    const rows = Array.isArray(data) ? data.slice() : [];
    rows.sort((a, b) => (numericFromParam(a?.week) ?? 0) - (numericFromParam(b?.week) ?? 0));
    counts[key] = fillScheduleTable(body, rows);
  });

  console.log("Tables row counts", counts);
}

async function injectEOYIfAvailable() {
  const container = document.getElementById("eoy-stats");
  const currentKey = STATE.currentGameKey;
  const game = currentKey ? STATE.games.get(currentKey) : null;
  const season = hasNumeric(game?.season) ? Number(game.season) : null;
  const prior = hasNumeric(season) ? season - 1 : null;
  const homeKey = game?.home_team_norm ?? game?.raw_sources?.sagarin_row_home?.team ?? game?.home_team_raw ?? null;
  const awayKey = game?.away_team_norm ?? game?.raw_sources?.sagarin_row_away?.team ?? game?.away_team_raw ?? null;
  const data = await loadEOY(prior, homeKey, awayKey);
  renderEOY(container, prior, data);
  return { priorSeason: prior, data };
}

async function loadEOY(priorSeason, homeNorm, awayNorm) {
  const league = STATE.league ?? DEFAULT_LEAGUE;
  const seasonNum = hasNumeric(priorSeason) ? Number(priorSeason) : null;
  const path = seasonNum !== null
    ? (league === "cfb"
      ? `out/cfb/final_league_metrics_${seasonNum}.csv`
      : `out/final_league_metrics_${seasonNum}.csv`)
    : (league === "cfb"
      ? `out/cfb/final_league_metrics_${priorSeason ?? "unknown"}.csv`
      : `out/final_league_metrics_${priorSeason ?? "unknown"}.csv`);
  if (seasonNum === null) {
    console.log(`EOY: using ${path} (league=${league}) - FAIL`, "invalid season");
    return null;
  }
  const cacheKey = `${league}:${seasonNum}`;
  let cache = STATE.eoyCache.get(cacheKey);
  const toNum = (value) => {
    const num = Number(String(value ?? "").trim());
    return Number.isFinite(num) ? num : null;
  };
  const gamesFrom = (value) => {
    if (!value) return null;
    const parts = String(value)
      .split("-")
      .map((part) => Number(part));
    const total = parts.reduce((acc, num) => (Number.isFinite(num) ? acc + num : acc), 0);
    return total > 0 ? total : null;
  };
  const buildKeys = (value) => {
    const keys = new Set();
    const push = (val) => {
      if (!val) return;
      const str = String(val).trim();
      if (!str) return;
      keys.add(str.toLowerCase());
      keys.add(str.replace(/\s+/g, "").toLowerCase());
    };
    push(value);
    const display = toDisplayName(value);
    if (display) {
      push(display);
      const entry = TEAM_DATA.find((team) => `${team.nickname}, ${team.city}` === display);
      if (entry) {
        push(`${entry.city} ${entry.nickname}`);
        entry.aliases.forEach(push);
      }
    }
    return keys;
  };
  if (!cache) {
    try {
      const url = new URL(`../${path}`, window.location.href);
      const res = await fetch(url.toString());
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      let text = await res.text();
      if (text.startsWith("\uFEFF")) text = text.slice(1);
      const lines = text.split(/\r?\n/).filter((line) => line.trim());
      if (lines.length === 0) throw new Error("empty csv");
      const headers = lines[0].split(",").map((h) => h.trim());
      const map = new Map();
      lines.slice(1).forEach((line) => {
        const cols = line.split(",");
        if (cols.every((col) => col.trim() === "")) return;
        const raw = {};
        headers.forEach((header, idx) => {
          raw[header] = (cols[idx] ?? "").trim();
        });
        const games = gamesFrom(raw.SU);
        const pf = toNum(raw.PF);
        const pa = toNum(raw.PA);
        const pf_pg = league === "cfb" ? pf : (games && pf !== null ? pf / games : pf);
        const pa_pg = league === "cfb" ? pa : (games && pa !== null ? pa / games : pa);
        const record = {
          team: raw.Team,
          pf_pg,
          pa_pg,
          su: raw.SU || null,
          ats: raw.ATS || null,
          to_margin_pg: toNum(raw.TO),
          ry_pg: toNum(raw["RY(O)"]),
          rush_off_rank: toNum(raw["R(O)_RY"]),
          py_pg: toNum(raw["PY(O)"]),
          pass_off_rank: toNum(raw["R(O)_PY"]),
          ty_pg: toNum(raw["TY(O)"]),
          total_off_rank: toNum(raw["R(O)_TY"]),
          ry_allowed_pg: toNum(raw["RY(D)"]),
          rush_def_rank: toNum(raw["R(D)_RY"]),
          py_allowed_pg: toNum(raw["PY(D)"]),
          pass_def_rank: toNum(raw["R(D)_PY"]),
          ty_allowed_pg: toNum(raw["TY(D)"]),
          total_def_rank: toNum(raw["R(D)_TY"]),
        };
        buildKeys(raw.Team).forEach((key) => {
          if (!map.has(key)) map.set(key, record);
        });
      });
      cache = { map };
      STATE.eoyCache.set(cacheKey, cache);
    } catch (err) {
      console.log(`EOY: using ${path} — FAIL`, err?.message ?? err);
      return null;
    }
  }
  console.log(`EOY: using ${path} — PASS`);
  const pick = (value) => {
    if (!value) return null;
    const tries = [value];
    const aliasDisplay = TEAM_ALIAS_DISPLAY[String(value).trim().toLowerCase()];
    if (aliasDisplay) tries.push(aliasDisplay);
    for (const candidate of tries) {
      for (const key of buildKeys(candidate)) {
        const row = cache.map.get(key);
        if (row) return row;
      }
    }
    return null;
  };
  return { home: pick(homeNorm), away: pick(awayNorm) };
}

function renderEOY(container, priorSeason, data) {
  if (!container) return;
  const heading = hasNumeric(priorSeason) ? `${priorSeason} End of Year Statistics` : "End of Year Statistics";
  const game = STATE.currentGameKey ? STATE.games.get(STATE.currentGameKey) : null;
  const labels = {
    home: game ? getTeamDisplayName(game, "home") : "Home",
    away: game ? getTeamDisplayName(game, "away") : "Away",
  };
  const headers = [
    "Team",
    "PF",
    "PA",
    "SU",
    "ATS",
    "TO Margin",
    "Rush Off",
    "Rush Rank",
    "Pass Off",
    "Pass Rank",
    "Total Off",
    "Total Off Rank",
    "Rush Def",
    "Rush Def Rank",
    "Pass Def",
    "Pass Def Rank",
    "Total Def",
    "Total Def Rank",
  ];
  const rows = data && typeof data === "object" ? data : null;
  const skipForLeague = Boolean(rows?.skipped);
  const matched = skipForLeague
    ? { home: false, away: false }
    : { home: Boolean(rows?.home), away: Boolean(rows?.away) };
  let bodyHtml = "";
  if (skipForLeague) {
    bodyHtml = ["home", "away"]
      .map((side) => {
        const cells = headers.map((_, idx) => (idx === 0 ? labels[side] : MISSING_VALUE));
        return `<tr>${cells
          .map((value, idx) => `<td${idx === 0 ? ' style="font-weight:600"' : ""}>${value}</td>`)
          .join("")}</tr>`;
      })
      .join("");
  } else if (matched.home && matched.away) {
    ["home", "away"].forEach((side) => {
      const row = rows[side];
      const values = [
        labels[side],
        formatNumber(row.pf_pg, { decimals: 1 }),
        formatNumber(row.pa_pg, { decimals: 1 }),
        fallback(row.su),
        fallback(row.ats),
        formatNumber(row.to_margin_pg, { decimals: 1, signed: true }),
        formatNumber(row.ry_pg, { decimals: 1 }),
        rankOrDash(row.rush_off_rank),
        formatNumber(row.py_pg, { decimals: 1 }),
        rankOrDash(row.pass_off_rank),
        formatNumber(row.ty_pg, { decimals: 1 }),
        rankOrDash(row.total_off_rank),
        formatNumber(row.ry_allowed_pg, { decimals: 1 }),
        rankOrDash(row.rush_def_rank),
        formatNumber(row.py_allowed_pg, { decimals: 1 }),
        rankOrDash(row.pass_def_rank),
        formatNumber(row.ty_allowed_pg, { decimals: 1 }),
        rankOrDash(row.total_def_rank),
      ];
      bodyHtml += `<tr>${values
        .map((value, idx) => `<td${idx === 0 ? ' style="font-weight:600"' : ""}>${value}</td>`)
        .join("")}</tr>`;
    });
  } else {
    bodyHtml = '<tr><td colspan="18">No data</td></tr>';
  }
  container.innerHTML = `
    <h3 class="section-title">${heading}</h3>
    <table>
      <thead><tr>${headers.map((label) => `<th>${label}</th>`).join("")}</tr></thead>
      <tbody>${bodyHtml}</tbody>
    </table>
  `;
  if (skipForLeague) {
    console.log("EOY: rendered placeholders (CFB league skip).");
  } else {
    console.log(`EOY: matched rows - home:${matched.home} away:${matched.away}`);
  }
  console.log("EOY: rendered");
}

function fillScheduleTable(tbody, rows) {
  tbody.innerHTML = "";
  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 12;
    td.textContent = "No data";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return 0;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.result === "W") tr.classList.add("win");
    if (row.result === "L") tr.classList.add("loss");
    const cells = [
      fallback(row.week),
      formatOpponent(row.site, row.opp),
      formatScore(row.pf, row.pa),
      fallback(row.result),
      formatNumber(row.pr, { decimals: 1 }),
      rankOrDash(row.pr_rank),
      formatNumber(row.opp_pr, { decimals: 1 }),
      rankOrDash(row.opp_pr_rank),
      formatNumber(row.sos, { decimals: 2 }),
      rankOrDash(row.sos_rank),
      formatNumber(row.opp_sos, { decimals: 2 }),
      rankOrDash(row.opp_sos_rank),
    ];

    cells.forEach((value, idx) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (idx === 1) td.classList.add("opponent");
      if (idx === 2) td.classList.add("score");
      if (idx === 3) td.classList.add("result");
      tr.appendChild(td);
    });

    tbody.appendChild(tr);
  });

  return rows.length;
}

function updateNavigationControls() {
  const keys = Array.isArray(STATE.sortedKeys) ? STATE.sortedKeys : [];
  const total = keys.length;
  if (!STATE.currentGameKey || total === 0 || !STATE.games.has(STATE.currentGameKey)) {
    els.navRow.classList.add("hidden");
    els.navSummary.textContent = "";
    els.prevGameBtn.disabled = true;
    els.nextGameBtn.disabled = true;
    return;
  }

  const index = keys.indexOf(STATE.currentGameKey);
  if (index === -1) {
    els.navRow.classList.add("hidden");
    return;
  }

  const prevEnabled = index > 0;
  const nextEnabled = index < total - 1;
  els.prevGameBtn.disabled = !prevEnabled;
  els.nextGameBtn.disabled = !nextEnabled;
  els.navSummary.textContent = `Game ${index + 1} of ${total}`;
  els.navRow.classList.remove("hidden");
}

function navigateRelative(offset) {
  const keys = Array.isArray(STATE.sortedKeys) ? STATE.sortedKeys : [];
  if (!STATE.currentGameKey || keys.length === 0) return;
  const currentIndex = keys.indexOf(STATE.currentGameKey);
  if (currentIndex === -1) return;
  const nextIndex = currentIndex + offset;
  if (nextIndex < 0 || nextIndex >= keys.length) return;
  const nextKey = keys[nextIndex];
  if (!STATE.games.has(nextKey)) {
    setStatus("Game not available in list.");
    return;
  }
  loadSingleGame(nextKey);
}

function rememberWeekRow(gameKey, sidecarRel) {
  if (!gameKey || !STATE.season || !STATE.week) return;
  try {
    localStorage.setItem(
      "week-view:last-game",
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
  updateStatusLineGame(sidecarRel);
}

function updateStatusLineGame(sidecarRel) {
  if (!els.statusLine) return;
  if (!STATE.weekSourcePath) {
    els.statusLine.textContent = "";
    return;
  }
  const parts = [`Source: ${STATE.weekSourcePath}`];
  if (sidecarRel) {
    parts.push(sidecarRel);
  }
  const timestamp = (STATE.lastLoadedAt || new Date()).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  parts.push(`Loaded ${timestamp}`);
  els.statusLine.textContent = parts.join(" · ");
}

function updateUrlWithSelection(gameKey) {
  if (!STATE.season || !STATE.week) return;
  const url = new URL(window.location.href);
  url.searchParams.set("season", STATE.season);
  url.searchParams.set("week", STATE.week);
  if (STATE.league && STATE.league !== DEFAULT_LEAGUE) {
    url.searchParams.set("league", STATE.league);
  } else {
    url.searchParams.delete("league");
  }
  if (gameKey) {
    url.searchParams.set("game_key", gameKey);
  }
  window.history.replaceState(
    { league: STATE.league, season: STATE.season, week: STATE.week, game_key: gameKey },
    "",
    url.toString()
  );
}

function coverageCounts(game) {
  return {
    home: NUMERIC_KEYS.home.reduce((acc, key) => acc + (hasNumeric(game[key]) ? 1 : 0), 0),
    away: NUMERIC_KEYS.away.reduce((acc, key) => acc + (hasNumeric(game[key]) ? 1 : 0), 0),
  };
}

function parseJsonLines(text) {
  const records = [];
  if (!text) return { records, count: 0 };
  const lines = text.split(/\r?\n/);
  lines.forEach((rawLine, idx) => {
    let line = rawLine.trim();
    if (!line) return;
    if (idx === 0 && line.charCodeAt(0) === 0xfeff) {
      line = line.slice(1);
    }
    // Replace NaN/Infinity tokens so JSON.parse succeeds.
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

function formatNumber(value, { decimals = 1, signed = false } = {}) {
  if (!hasNumeric(value)) return MISSING_VALUE;
  const num = Number(value);
  const fixed = num.toFixed(decimals);
  if (!signed) return fixed;
  if (num > 0) return `+${fixed}`;
  if (num < 0) return `\u2212${Math.abs(num).toFixed(decimals)}`;
  return `0.${"0".repeat(decimals)}`;
}

function fallback(value) {
  if (value === null || value === undefined || value === "" || Number.isNaN(value)) {
    return MISSING_VALUE;
  }
  return value;
}

function logTimeZone(success) {
  if (timezoneLogged) return;
  console.log(success ? "Time zone: America/Los_Angeles (DST auto via Intl)" : "Time zone: UTC fallback");
  timezoneLogged = true;
}

function fmtKickoffPT(iso) {
  if (!iso) {
    logTimeZone(true);
    return MISSING_VALUE;
  }
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) throw new Error("Invalid date");
    const tz = "America/Los_Angeles";
    const dateFormatter = new Intl.DateTimeFormat("en-US", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit" });
    const timeFormatter = new Intl.DateTimeFormat("en-US", { timeZone: tz, hour: "2-digit", minute: "2-digit", hour12: true });
    const [mm, dd, yyyy] = dateFormatter.format(d).split("/");
    const time = timeFormatter.format(d);
    logTimeZone(true);
    return `${yyyy}-${mm}-${dd} ${time} PT`;
  } catch (e) {
    console.warn("fmtKickoffPT fallback \u2192 UTC", e);
    logTimeZone(false);
    return formatKickoff(iso);
  }
}

function formatKickoff(isoString) {
  if (!isoString) return MISSING_VALUE;
  const clean = isoString.replace("Z", "+00:00");
  const match = clean.match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
  if (match) {
    return `${match[1]} ${match[2]} UTC`;
  }
  try {
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) throw new Error();
    const year = date.getUTCFullYear();
    const month = String(date.getUTCMonth() + 1).padStart(2, "0");
    const day = String(date.getUTCDate()).padStart(2, "0");
    const hour = String(date.getUTCHours()).padStart(2, "0");
    const minute = String(date.getUTCMinutes()).padStart(2, "0");
    return `${year}-${month}-${day} ${hour}:${minute} UTC`;
  } catch {
    return fallback(isoString);
  }
}

function getTeamDisplayName(game, side) {
  const prefix = side === "home" ? "home" : "away";
  const sagarin = game?.raw_sources?.[`sagarin_row_${side}`]?.team;
  const fromSagarin = toDisplayName(sagarin);
  if (fromSagarin) return fromSagarin;

  const raw = game[`${prefix}_team_raw`];
  const fromRaw = toDisplayName(raw);
  if (fromRaw) return fromRaw;

  const norm = game[`${prefix}_team_norm`];
  const fromNorm = toDisplayName(norm);
  if (fromNorm) return fromNorm;

  if (sagarin) {
    const fallbackName = nicknameCityFromString(sagarin);
    if (fallbackName) return fallbackName;
  }

  if (raw) {
    const fallbackName = toDisplayName(String(raw).toLowerCase());
    if (fallbackName) return fallbackName;
  }

  if (norm) return String(norm).toUpperCase();
  return side === "home" ? "Home" : "Away";
}

function teamShortCode(game, side) {
  const prefix = side === "home" ? "home" : "away";
  const norm = (game?.[`${prefix}_team_norm`] || "").toUpperCase();
  const raw = (game?.[`${prefix}_team_raw`] || "").toUpperCase();
  const display = (norm || raw).replace(/[^A-Z0-9]/g, "");
  if (!display) return side === "home" ? "HOME" : "AWAY";
  return display.slice(0, 3);
}

function toDisplayName(value) {
  if (value === null || value === undefined) return null;
  const str = String(value).trim();
  if (!str) return null;
  const lower = str.toLowerCase();
  const direct = TEAM_ALIAS_DISPLAY[lower];
  if (direct) return direct;
  const compact = TEAM_ALIAS_DISPLAY[lower.replace(/\s+/g, "")];
  if (compact) return compact;
  const nicknameCity = nicknameCityFromString(str);
  if (nicknameCity) return nicknameCity;
  return null;
}

function nicknameCityFromString(value) {
  if (!value) return null;
  const stripped = String(value).trim();
  if (!stripped) return null;
  if (stripped.includes(",")) return stripped;
  const parts = stripped.split(/\s+/);
  if (parts.length < 2) return null;
  const nickname = parts.pop();
  const city = parts.join(" ");
  return `${nickname}, ${city}`;
}

function rankOrDash(value) {
  return hasNumeric(value) ? String(Number(value)) : MISSING_VALUE;
}

function formatSigned(value, opts = {}) {
  if (!hasNumeric(value)) return MISSING_VALUE;
  const num = Number(value);
  const magnitude = formatNumber(Math.abs(num), { decimals: 1, ...opts });
  if (magnitude === MISSING_VALUE) return MISSING_VALUE;
  return `${num >= 0 ? "+" : "\u2212"}${magnitude}`;
}

function hasFavoriteCoverageCFB(row) {
  if (!row || typeof row !== "object") return false;
  const side = String(row.favored_side || "").trim().toUpperCase();
  const spread = Number(row.spread_favored_team);
  const source = row.odds_source;
  const snap = row.snapshot_at;
  const total = Number(row.total);
  const sideOk = side === "HOME" || side === "AWAY";
  const spreadOk = Number.isFinite(spread);
  const sourceOk = typeof source === "string" && source.trim().length > 0;
  const snapOk = typeof snap === "string" && snap.trim().length > 0;
  const totalOk = Number.isFinite(total);
  return sideOk && spreadOk && sourceOk && snapOk && totalOk;
}

function formatOpponent(site, opponent) {
  const display = toDisplayName(opponent) ?? fallback(opponent);
  if (display === MISSING_VALUE) return display;
  if (site === "A") return `@ ${display}`;
  if (site === "N") return `vs ${display}`;
  return display;
}

function formatScore(pf, pa) {
  if (!hasNumeric(pf) || !hasNumeric(pa)) return MISSING_VALUE;
  return `${Number(pf)}-${Number(pa)}`;
}

function hasNumeric(value) {
  if (value === null || value === undefined) return false;
  const num = Number(value);
  return Number.isFinite(num);
}

function numericFromParam(input) {
  if (input === null || input === undefined || input === "") return null;
  const num = Number(input);
  return Number.isFinite(num) ? num : null;
}

function coerceInt(value) {
  const num = numericFromParam(value);
  return num ? Math.trunc(num) : null;
}

function syncWeekLink() {
  const season = coerceInt(els.season.value);
  const week = coerceInt(els.week.value);
  const params = new URLSearchParams();
  if (STATE.league && STATE.league !== DEFAULT_LEAGUE) {
    params.set("league", STATE.league);
  }
  if (season) params.set("season", season);
  if (week) params.set("week", week);
  els.weekLink.href = params.toString()
    ? `week_view.html?${params.toString()}`
    : "week_view.html";
}

function safeParseLocalStorage() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
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

function persistSelection(payload) {
  try {
    const copy = { ...payload };
    copy.league = normalizeLeague(copy.league ?? STATE.league ?? DEFAULT_LEAGUE);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(copy));
  } catch {
    // ignore storage failures
  }
}

function weekCacheKey(league, season, week) {
  return `${WEEK_CACHE_PREFIX}${CACHE_VERSION}:${league}:${season}:${week}`;
}

function readWeekCache(league, season, week) {
  try {
    const raw = window.localStorage.getItem(weekCacheKey(league, season, week));
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
      if (typeof parsed === "string") {
        const fromText = parseJsonLines(parsed).records;
        return fromText.length ? fromText : null;
      }
    } catch {
      const fallback = parseJsonLines(raw).records;
      return fallback.length ? fallback : null;
    }
  } catch {
    return null;
  }
  return null;
}

function writeWeekCache(league, season, week, records) {
  try {
    window.localStorage.setItem(weekCacheKey(league, season, week), JSON.stringify(records));
  } catch {
    // ignore cache write failures
  }
}

function setStatus(message) {
  els.status.textContent = message ?? "";
}





















