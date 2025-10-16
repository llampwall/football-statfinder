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
};

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
  const filtered = STATE.allRows.filter((row) => matchesTeamFilter(row, query));

  filtered.sort((a, b) => (a.kickoff_iso_utc || "").localeCompare(b.kickoff_iso_utc || ""));

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

  rows.forEach((row) => {
    const { kickoff_iso_utc, game_key } = row;
    if (!kickoff_iso_utc || !game_key) {
      console.warn("WARN: Missing key fields", { kickoff_iso_utc, game_key });
    }

    ["away", "home"].forEach((side) => {
      const cells = buildTeamRow(row, side);
      const tr = document.createElement("tr");
      tr.dataset.gameKey = game_key ?? "";
      tr.tabIndex = 0;
      cells.forEach((value, idx) => {
        const td = document.createElement("td");
        td.textContent = value;
        if (numericColumns.has(idx)) {
          td.classList.add("numeric");
        }
        tr.appendChild(td);
      });
      tr.addEventListener("click", () => {
        openGame(game_key);
      });
      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          openGame(game_key);
        } else if (event.key === "ArrowRight" && event.ctrlKey) {
          openGame(game_key, { newTab: true });
        }
      });
      tbody.appendChild(tr);
    });
  });

  const highlightKey = STATE.pendingScrollKey || STATE.highlightedGameKey;
  if (highlightKey) {
    highlightRow(highlightKey, { scroll: Boolean(STATE.pendingScrollKey) });
  } else {
    highlightRow(null);
  }
  STATE.pendingScrollKey = null;

  updateFilterMeta(rows.length, STATE.allRows.length);
}

function buildTeamRow(row, side) {
  const iso = row?.kickoff_iso_utc ?? null;
  const favored = isFavRow(row, side);
  const total = formatNumber(row.total);
  const currentPrValue = side === "home" ? row.home_pr : row.away_pr;
  const sosValue = side === "home" ? row.home_sos : row.away_sos;

  const diff = favored
    ? formatNumber(row.rating_diff_favored_team, { decimals: 1, signed: true })
    : MISSING_VALUE;
  const rvo = favored
    ? formatNumber(row.rating_vs_odds, { decimals: 1, signed: true })
    : MISSING_VALUE;

  return [
    fmtDatePT(iso),
    fmtTimePT(iso),
    placeholderTeamNumber(),
    placeholderGameNumber(),
    resolveTeamName(row, side),
    formatOddsCell(row, side),
    total,
    teamRecord(row, side),
    formatNumber(currentPrValue),
    diff,
    rvo,
    formatNumber(sosValue),
    sosDiffForRow(row.home_sos, row.away_sos, side),
    MISSING_VALUE,
    MISSING_VALUE,
    MISSING_VALUE,
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
  if (!isFavRow(row, side)) return MISSING_VALUE;
  if (!hasNumeric(row.spread_favored_team)) return MISSING_VALUE;
  const code = getTeamCode(row, side);
  const spread = formatNumber(row.spread_favored_team, { decimals: 1, signed: true });
  return code ? `${code} ${spread}` : spread;
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
  return MISSING_VALUE;
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
    "PR-1",
    "PR-2",
    "PR-3",
  ];

  const lines = [header.join(",")];
  STATE.filteredRows.forEach((row) => {
    ["away", "home"].forEach((side) => {
      const cells = buildTeamRow(row, side).map(csvValue);
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
