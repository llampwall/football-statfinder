const els = {
  season: document.getElementById("season-input"),
  week: document.getElementById("week-input"),
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
};

const STORAGE_KEY = "game-view:last-selection";
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
  deepLinkUsed: false,
  autoFromStorage: false,
};

const REQUIRED_FAVORITE_KEYS = [
  "favored_side",
  "spread_favored_team",
  "rating_diff_favored_team",
  "rating_vs_odds",
];

const MISSING_VALUE = "\u2014";

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

function attachListeners() {
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
}

function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const stored = safeParseLocalStorage();
  if (params.has("season") || params.has("week") || params.has("game_key")) {
    STATE.deepLinkUsed = true;
    console.log("Deep link detected -> auto load path.");
  }

  const initialSeason = numericFromParam(params.get("season")) ?? stored?.season ?? "";
  const initialWeek = numericFromParam(params.get("week")) ?? stored?.week ?? "";
  const initialGameKey = params.get("game_key") ?? stored?.game_key ?? "";
  STATE.autoFromStorage = !STATE.deepLinkUsed && Boolean(initialGameKey);

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

  const relPath = `out/${season}_week${week}/games_week_${season}_${week}.jsonl`;
  const url = new URL(relPath, window.location.origin);
  setStatus("Loading games…");
  let text;
  try {
    const res = await fetch(url.toString());
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    text = await res.text();
  } catch (err) {
    setStatus(`Failed to load games (${err.message})`);
    console.log(`FAIL: Games loaded (${relPath})`);
    return;
  }

  const { records, count } = parseJsonLines(text);
  console.log(`${count >= 1 ? "PASS" : "FAIL"}: Games loaded (count=${count})`);
  if (count === 0) {
    setStatus("No games found.");
    return;
  }

  STATE.games = new Map(records.map((row) => [row.game_key, row]));
  STATE.season = season;
  STATE.week = week;
  populateGameSelect(records, autoGameKey);
  setStatus(`Loaded ${count} games.`);

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

  records
    .slice()
    .sort((a, b) => a.kickoff_iso_utc.localeCompare(b.kickoff_iso_utc))
    .forEach((game) => {
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
}

async function loadSingleGame(gameKey) {
  const game = STATE.games.get(gameKey);
  console.log(`${game ? "PASS" : "FAIL"}: Game row found (${gameKey})`);
  if (!game) {
    setStatus("Game not in loaded set.");
    return;
  }

  const sidecarPath = `out/${STATE.season}_week${STATE.week}/game_schedules/${gameKey}.json`;
  const url = new URL(sidecarPath, window.location.origin);
  let sidecar;
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
  } catch (err) {
    console.log(`FAIL: Sidecar loaded (${sidecarPath})`, err);
    setStatus(`Sidecar failed (${err.message})`);
    return;
  }

  const teamNames = {
    home: getTeamDisplayName(game, "home"),
    away: getTeamDisplayName(game, "away"),
  };

  renderHeader(game, teamNames);
  renderTeamStats(game, teamNames);
  renderTables(game, sidecar, teamNames);

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

  setStatus("Game loaded.");
  els.diagnosticsNote.textContent = `Diagnostics: favorite fields ${hasAllFavoriteKeys ? "OK" : "missing"}, coverage home ${coverCounts.home}, away ${coverCounts.away}`;
  els.dataStamp.textContent = `Source: ${sidecarPath}`;

  persistSelection({
    season: STATE.season,
    week: STATE.week,
    game_key: gameKey,
  });
}

function renderHeader(game, teamNames) {
  const homeName = teamNames.home;
  const awayName = teamNames.away;
  const kickoff = formatKickoff(game.kickoff_iso_utc);

  els.teamsBlock.innerHTML = `
    <h2>${homeName} vs ${awayName}</h2>
    <div class="meta-line">Kickoff: ${kickoff}</div>
    <div class="meta-line">Game key: ${fallback(game.game_key)}</div>
  `;

  const favTeam =
    game.favored_side === "HOME"
      ? homeName
      : game.favored_side === "AWAY"
      ? awayName
      : null;
  const spread = formatNumber(game.spread_favored_team, { decimals: 1, signed: true });
  els.favoriteBlock.innerHTML = `
    <h3>Favorite & Spread</h3>
    <div class="meta-line">Favored: ${
      favTeam ? `${favTeam}${spread !== MISSING_VALUE ? ` (${spread})` : ""}` : MISSING_VALUE
    }</div>
    <div class="meta-line">Odds source: ${fallback(game.odds_source)}</div>
    <div class="meta-line">Snapshot: ${fallback(game.snapshot_at)}</div>
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

function renderTeamStats(game, teamNames) {
  const rows = [
    { prefix: "home", label: teamNames.home },
    { prefix: "away", label: teamNames.away },
  ];

  els.teamStatsBody.innerHTML = "";

  rows.forEach(({ prefix, label }) => {
    const tr = document.createElement("tr");
    const cells = [
      label,
      formatNumber(game[`${prefix}_pf_pg`], { decimals: 1 }),
      formatNumber(game[`${prefix}_pa_pg`], { decimals: 1 }),
      fallback(game[`${prefix}_su`]),
      fallback(game[`${prefix}_ats`]),
      formatNumber(game[`${prefix}_to_margin_pg`], { decimals: 1, signed: true }),
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
  if (!hasNumeric(value)) return MISSING_VALUE;
  const num = Number(value);
  if (num < 1 || num > 32) return MISSING_VALUE;
  return num.toFixed(0);
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
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function persistSelection(payload) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // ignore storage failures
  }
}

function setStatus(message) {
  els.status.textContent = message ?? "";
}
