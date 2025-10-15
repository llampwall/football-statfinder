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
};

const STORAGE_KEY = "week-view:last-selection";
const MISSING_VALUE = "\u2014";

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
}

async function bootstrap() {
  const params = new URLSearchParams(window.location.search);
  const paramSeason = coerceInt(params.get("season"));
  const paramWeek = coerceInt(params.get("week"));

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

  const stored = loadStoredSelection();
  if (!target && stored) {
    console.log("Fallback to stored selection", stored);
    target = stored;
  }

  if (!target) {
    setStatus("No season/week found. Please enter values.");
    return;
  }

  els.seasonInput.value = target.season;
  els.weekInput.value = target.week;
  const loaded = await loadAndRender(target.season, target.week);
  if (!loaded) {
    if (stored && (stored.season !== target.season || stored.week !== target.week)) {
      console.log("Retrying with stored selection after failed load", stored);
      els.seasonInput.value = stored.season;
      els.weekInput.value = stored.week;
      const retry = await loadAndRender(stored.season, stored.week);
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

  applyLoadedRows(result.rows, season, week, {
    updateHistory: fromControl,
  });
  return true;
}

function applyLoadedRows(rows, season, week, { updateHistory = false } = {}) {
  const safeRows = Array.isArray(rows) ? rows : [];
  const count = safeRows.length;
  console.log(`${count >= 1 ? "PASS" : "FAIL"}: Games parsed (count=${count})`);
  renderTable(safeRows, { season, week });
  updateFooter(count, season, week);
  persistSelection({ season, week });

  if (updateHistory) {
    const url = new URL(window.location.href);
    url.searchParams.set("season", season);
    url.searchParams.set("week", week);
    window.history.replaceState(null, "", url.toString());
  }
}

async function loadGames(season, week) {
  const path = `../out/${season}_week${week}/games_week_${season}_${week}.jsonl`;
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
    };
  } catch (err) {
    console.error(`FAIL: loadGames season=${season} week=${week}`, err);
    return { success: false, message: `Failed to load: ${err.message}` };
  }
}

function renderTable(rows, { season, week }) {
  const tbody = els.tableBody;
  tbody.innerHTML = "";
  const sorted = rows.slice().sort((a, b) =>
    (a.kickoff_iso_utc ?? "").localeCompare(b.kickoff_iso_utc ?? "")
  );

  if (sorted.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.textContent = "No games found.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  sorted.forEach((row) => {
    const { kickoff_iso_utc, game_key } = row;
    if (!kickoff_iso_utc || !game_key) {
      console.warn("WARN: Missing key fields", { kickoff_iso_utc, game_key });
    }

    const kickoff = formatKickoff(kickoff_iso_utc);
    const matchup = `${formatTeam(row.away_team_norm, row.away_team_raw)} @ ${formatTeam(
      row.home_team_norm,
      row.home_team_raw
    )}`;
    const favorite = favoriteSpread(row);
    const total = formatNumber(row.total);
    const prDiff = formatNumber(row.rating_diff_favored_team, { signed: true });
    const rvo = formatNumber(row.rating_vs_odds, { signed: true });
    const odds = formatOdds(row);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${kickoff}</td>
      <td class="matchup">${matchup}</td>
      <td>${favorite}</td>
      <td class="numeric">${total}</td>
      <td class="numeric">${prDiff}</td>
      <td class="numeric">${rvo}</td>
      <td>${odds}</td>
    `;
    tr.addEventListener("click", () => {
      const url = `game_view.html?season=${season}&week=${week}&game_key=${encodeURIComponent(
        game_key
      )}`;
      console.log("NAV: Row click ->", url);
      window.location.href = url;
    });
    tbody.appendChild(tr);
  });
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

function favoriteSpread(row) {
  if (!row.favored_side) return MISSING_VALUE;
  const team =
    row.favored_side === "HOME"
      ? formatTeam(row.home_team_norm, row.home_team_raw)
      : row.favored_side === "AWAY"
      ? formatTeam(row.away_team_norm, row.away_team_raw)
      : null;
  const spread = formatNumber(row.spread_favored_team, { signed: true });
  if (!team) return MISSING_VALUE;
  return spread !== MISSING_VALUE ? `${team} ${spread}` : team;
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

function formatOdds(row) {
  const source = row.odds_source ? String(row.odds_source) : MISSING_VALUE;
  if (row.is_closing) {
    if (source === MISSING_VALUE) {
      return `<span class="badge">Closing</span>`;
    }
    return `${source} <span class="badge">Closing</span>`;
  }
  return source;
}

function updateFooter(count, season, week) {
  const label = count === 1 ? "game" : "games";
  els.status.textContent = `Loaded ${count} ${label}`;
  els.loadedLabel.textContent = `Season ${season}, Week ${week}`;
  els.weekSummary.textContent = `Currently viewing Season ${season}, Week ${week}`;
  els.gameViewLink.href = `game_view.html?season=${season}&week=${week}`;
  els.latestLink.href = "week_view.html";
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
