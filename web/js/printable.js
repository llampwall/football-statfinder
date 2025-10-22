/**
 * Printable Game View hydrator.
 *
 * Scope: Populate `web/game_view_printable.html` with week-level data for a single game.
 * Spec anchors: context/merge_summary_global_week.md
 * Invariants: Reads existing JSONL outputs (no schema changes), remains idempotent, UTC inputs converted via browser locale.
 * Side effects: Performs a single fetch to `out/.../games_week_*.jsonl`, writes textContent into known DOM ids.
 * Log contract: Logs to `console.error` when hydration fails; otherwise remains quiet.
 */

const DASH = "\u2014";

const query = new URLSearchParams(window.location.search);
const leagueParam = (query.get("league") || "NFL").toUpperCase();
const seasonParam = toNumber(query.get("season"));
const weekParam = toNumber(query.get("week"));
const gameKey = (query.get("game") || "").trim();

const baseDir =
  Number.isFinite(seasonParam) && Number.isFinite(weekParam)
    ? leagueParam === "CFB"
      ? `out/cfb/${seasonParam}_week${weekParam}`
      : `out/${seasonParam}_week${weekParam}`
    : null;

const weekPath =
  baseDir && Number.isFinite(seasonParam) && Number.isFinite(weekParam)
    ? `${baseDir}/games_week_${seasonParam}_${weekParam}.jsonl`
    : null;

const $ = (id) => document.getElementById(id);
const WARNED = new Set();

function warnOnce(key, payload) {
  if (WARNED.has(key)) return;
  WARNED.add(key);
  console.warn("PRINTABLE:", key, payload ?? "");
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") return NaN;
  const num = Number(value);
  return Number.isFinite(num) ? num : NaN;
}

function trimDecimals(text) {
  return text.includes(".") ? text.replace(/\.0+$|(\.\d*?[1-9])0+$/u, "$1") : text;
}

function formatNumber(value, { decimals = 1, signed = false } = {}) {
  if (value === null || value === undefined || value === "") return DASH;
  const num = Number(value);
  if (!Number.isFinite(num)) return DASH;
  const formatted = trimDecimals(
    num.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
  );
  if (signed && num > 0) return `+${formatted}`;
  if (signed && num === 0) return formatted;
  return signed ? formatted : formatted;
}

function dash(value) {
  return value === null || value === undefined || value === "" ? DASH : value;
}

function setText(id, value) {
  const el = $(id);
  if (el) {
    el.textContent = value;
  }
}

function hasNumeric(value) {
  if (value === null || value === undefined || value === "") return false;
  const num = Number(value);
  return Number.isFinite(num);
}

function formatOpponent(site, opponent) {
  const name = dash(opponent);
  if (name === DASH) return name;
  const loc = (site || "").toUpperCase();
  if (loc === "A") return `@ ${name}`;
  if (loc === "N") return `vs ${name}`;
  return name;
}

function formatScore(pf, pa) {
  if (hasNumeric(pf) && hasNumeric(pa)) {
    const home = trimDecimals(String(Number(pf)));
    const away = trimDecimals(String(Number(pa)));
    return `${home}-${away}`;
  }
  return DASH;
}

async function loadJsonl(path) {
  const url = new URL(`../${path}`, window.location.href);
  const res = await fetch(url.toString(), { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  const text = await res.text();
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

async function loadSidecar(baseDir, gameKey) {
  if (!baseDir || !gameKey) {
    throw new Error("missing sidecar path");
  }
  const rel = `${baseDir}/game_schedules/${gameKey}.json`;
  const url = new URL(`../${rel}`, window.location.href);
  const res = await fetch(url.toString(), { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  let raw = await res.text();
  raw = raw.trim();
  if (!raw) return null;
  if (raw.startsWith("\uFEFF")) raw = raw.slice(1);
  const sanitized = raw.replace(/([-+]?Infinity|\bNaN\b)/gi, "null");
  return JSON.parse(sanitized);
}

function teamName(row, side) {
  const prefix = side === "home" ? "home" : "away";
  return (
    row?.raw_sources?.[`sagarin_row_${side}`]?.team ||
    row?.[`${prefix}_team_name`] ||
    row?.[`${prefix}_team_norm`] ||
    row?.[`${prefix}_team_raw`] ||
    DASH
  );
}

function spreadFor(side, row) {
  const spread = Number(row?.spread_home_relative);
  if (!Number.isFinite(spread)) return DASH;
  const value = side === "home" ? spread : spread * -1;
  return formatNumber(value, { decimals: Math.abs(value * 10) % 10 === 0 ? 0 : 1, signed: true });
}

function ratingDiffFor(side, row) {
  const diff = Number(row?.rating_diff);
  if (!Number.isFinite(diff)) return DASH;
  const value = side === "home" ? diff : diff * -1;
  return formatNumber(value, { decimals: 2, signed: true });
}

function ratingVsOddsFor(side, row) {
  const value = Number(row?.rating_vs_odds);
  if (!Number.isFinite(value)) return DASH;
  const signed = side === "home" ? value : value * -1;
  return formatNumber(signed, { decimals: 2, signed: true });
}

function sosDiffFor(side, row) {
  const home = Number(row?.home_sos);
  const away = Number(row?.away_sos);
  if (!Number.isFinite(home) || !Number.isFinite(away)) return DASH;
  const value = side === "home" ? home - away : away - home;
  return formatNumber(value, { decimals: 2, signed: true });
}

function resolveTeamNumber(row, side) {
  const scheduleRow = row?.raw_sources?.schedule_row;
  if (!scheduleRow) return DASH;
  const field = side === "home" ? "home_team" : "away_team";
  return dash(scheduleRow[field]);
}

function resolveGameNumber(row) {
  const gsis = row?.raw_sources?.schedule_row?.gsis;
  if (gsis === null || gsis === undefined) return DASH;
  const num = Number(gsis);
  return Number.isFinite(num) ? String(Math.trunc(num)) : DASH;
}

function applyHeader(row) {
  const kickoffIso =
    row?.kickoff_iso_utc ||
    row?.kickoff ||
    row?.game_date ||
    row?.start_time ||
    null;
  if (!kickoffIso) {
    setText("hdrDate", DASH);
    setText("hdrTime", DASH);
    return;
  }
  const kickoff = new Date(kickoffIso);
  if (Number.isNaN(kickoff.getTime())) {
    setText("hdrDate", DASH);
    setText("hdrTime", DASH);
    return;
  }
  const dateFmt = new Intl.DateTimeFormat("en-US", {
    weekday: "short",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  const timeFmt = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
  setText("hdrDate", dateFmt.format(kickoff));
  setText("hdrTime", timeFmt.format(kickoff));
}

function fillTeam(prefix, row, side) {
  setText(`${prefix}TeamNo`, resolveTeamNumber(row, side));
  setText(`${prefix}GameNo`, resolveGameNumber(row));
  setText(`${prefix}Name`, dash(teamName(row, side)));
  setText(`${prefix}Odds`, spreadFor(side, row));
  setText(`${prefix}OU`, formatNumber(row?.total, { decimals: 1 }));
  setText(`${prefix}PR`, formatNumber(row?.[`${side}_pr`], { decimals: 2 }));
  setText(`${prefix}Diff`, ratingDiffFor(side, row));
  setText(`${prefix}RVO`, ratingVsOddsFor(side, row));
  setText(`${prefix}SoS`, formatNumber(row?.[`${side}_sos`], { decimals: 2 }));
  setText(`${prefix}SoSDiff`, sosDiffFor(side, row));
  setText(`${prefix}PF`, formatNumber(row?.[`${side}_pf_pg`], { decimals: 1 }));
  setText(`${prefix}PA`, formatNumber(row?.[`${side}_pa_pg`], { decimals: 1 }));
  setText(`${prefix}SU`, dash(row?.[`${side}_su`]));
  setText(`${prefix}ATS`, dash(row?.[`${side}_ats`]));
  setText(
    `${prefix}OffRY`,
    formatNumber(row?.[`${side}_ry_pg`], { decimals: 1 })
  );
  setText(
    `${prefix}OffPY`,
    formatNumber(row?.[`${side}_py_pg`], { decimals: 1 })
  );
  setText(
    `${prefix}OffTY`,
    formatNumber(row?.[`${side}_ty_pg`], { decimals: 1 })
  );
  setText(
    `${prefix}DefRY`,
    formatNumber(row?.[`${side}_ry_allowed_pg`], { decimals: 1 })
  );
  setText(
    `${prefix}DefPY`,
    formatNumber(row?.[`${side}_py_allowed_pg`], { decimals: 1 })
  );
  setText(
    `${prefix}DefTY`,
    formatNumber(row?.[`${side}_ty_allowed_pg`], { decimals: 1 })
  );
  setText(
    `${prefix}DefTO`,
    formatNumber(row?.[`${side}_to_margin_pg`], { decimals: 2, signed: true })
  );
}

function fillSchedule(tbodyId, schedule) {
  const body = $(tbodyId);
  if (!body) return;
  body.innerHTML = "";
  if (!Array.isArray(schedule) || schedule.length === 0) return;

  schedule.forEach((entry, index) => {
    const tr = document.createElement("tr");
    tr.className = "center";
    const cells = [
      dash(entry?.week ?? index + 1),
      formatOpponent(entry?.site, entry?.opp ?? entry?.opponent ?? ""),
      formatNumber(entry?.pr ?? entry?.team_pr, { decimals: 2 }),
      formatNumber(entry?.opp_pr ?? entry?.opponent_pr, { decimals: 2 }),
      formatNumber(entry?.sos ?? entry?.team_sos, { decimals: 2 }),
      formatNumber(entry?.opp_sos ?? entry?.sos_opp, { decimals: 2 }),
      dash(entry?.result ?? entry?.wlt ?? ""),
      formatScore(entry?.pf ?? entry?.points_for, entry?.pa ?? entry?.points_against),
    ];
    cells.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

function fillEoy(prefix, stats) {
  const set = (suffix, value, opts) => {
    const el = $(`${prefix}${suffix}`);
    if (!el) return;
    if (value === null || value === undefined || value === "") {
      el.textContent = DASH;
      return;
    }
    if (opts?.format === "number") {
      el.textContent = formatNumber(value, {
        decimals: opts.decimals ?? 1,
        signed: Boolean(opts.signed),
      });
      return;
    }
    el.textContent = dash(value);
  };

  set("PF", stats?.pf_pg, { format: "number", decimals: 1 });
  set("PA", stats?.pa_pg, { format: "number", decimals: 1 });
  set("SU", stats?.su);
  set("ATS", stats?.ats);
  set("OffRY", stats?.ry_pg, { format: "number", decimals: 1 });
  set("OffPY", stats?.py_pg, { format: "number", decimals: 1 });
  set("OffTY", stats?.ty_pg, { format: "number", decimals: 1 });
  set("DefRY", stats?.ry_allowed_pg, { format: "number", decimals: 1 });
  set("DefPY", stats?.py_allowed_pg, { format: "number", decimals: 1 });
  set("DefTY", stats?.ty_allowed_pg, { format: "number", decimals: 1 });
  set("DefTO", stats?.to_margin_pg, { format: "number", decimals: 2, signed: true });
}

const EOY_CACHE = new Map();

function normalizeKey(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function buildAliasKeys(name) {
  const keys = new Set();
  const str = String(name ?? "").trim();
  if (!str) return keys;
  keys.add(normalizeKey(str));
  if (str.includes(",")) {
    const [nickname, city] = str.split(",").map((part) => part.trim());
    if (nickname && city) {
      keys.add(normalizeKey(`${city} ${nickname}`));
    }
  }
  const swapped = str.replace(",", "");
  if (swapped && swapped !== str) keys.add(normalizeKey(swapped));
  return keys;
}

function parseGames(value) {
  if (!value) return null;
  const parts = String(value)
    .split("-")
    .map((part) => Number(part));
  const total = parts.reduce((acc, num) => (Number.isFinite(num) ? acc + num : acc), 0);
  return total > 0 ? total : null;
}

function toNum(value) {
  const num = Number(String(value ?? "").trim());
  return Number.isFinite(num) ? num : null;
}

async function loadTeamSeasonStats(league, season, teamName) {
  if (!Number.isFinite(season)) return null;
  const cacheKey = `${league}:${season}`;
  let cache = EOY_CACHE.get(cacheKey);
  if (!cache) {
    const path =
      league === "CFB"
        ? `out/cfb/final_league_metrics_${season}.csv`
        : `out/final_league_metrics_${season}.csv`;
    try {
      const url = new URL(`../${path}`, window.location.href);
      const res = await fetch(url.toString(), { cache: "no-cache" });
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
        const games = parseGames(raw.SU);
        const pf = toNum(raw.PF);
        const pa = toNum(raw.PA);
        const record = {
          team: raw.Team,
          pf_pg: league === "CFB" ? pf : games && pf !== null ? pf / games : pf,
          pa_pg: league === "CFB" ? pa : games && pa !== null ? pa / games : pa,
          su: raw.SU || null,
          ats: raw.ATS || null,
          to_margin_pg: toNum(raw.TO),
          ry_pg: toNum(raw["RY(O)"]),
          py_pg: toNum(raw["PY(O)"]),
          ty_pg: toNum(raw["TY(O)"]),
          ry_allowed_pg: toNum(raw["RY(D)"]),
          py_allowed_pg: toNum(raw["PY(D)"]),
          ty_allowed_pg: toNum(raw["TY(D)"]),
        };
        buildAliasKeys(raw.Team).forEach((key) => {
          if (!map.has(key)) {
            map.set(key, record);
          }
        });
      });
      cache = { map };
      EOY_CACHE.set(cacheKey, cache);
    } catch (error) {
      warnOnce("missing prev-season stats", { league, season, error: error?.message ?? error });
      return null;
    }
  }

  const candidates = [
    teamName,
    String(teamName ?? "").replace(/[,]/g, ""),
  ].filter(Boolean);

  for (const candidate of candidates) {
    for (const key of buildAliasKeys(candidate)) {
      const record = cache.map.get(key);
      if (record) return record;
    }
  }

  warnOnce("missing prev-season stats", { league, season, team: teamName });
  return null;
}

function renderError(message) {
  console.error("Printable load failed:", message);
  document.body.innerHTML =
    '<div style="padding:24px;font:14px/1.4 system-ui">Printable data not found. Open from a Game View.</div>';
}

(async function hydrate() {
  try {
    if (!baseDir || !weekPath || !gameKey) {
      throw new Error("missing query params");
    }
    const rows = await loadJsonl(weekPath);
    const row = rows.find((r) => r?.game_key === gameKey);
    if (!row) {
      throw new Error("game not found");
    }

    applyHeader(row);
    const awayDisplay = teamName(row, "away");
    const homeDisplay = teamName(row, "home");

    fillTeam("t1", row, "away");
    fillTeam("t2", row, "home");

    const prevSeasonValue = Number.isFinite(seasonParam) ? seasonParam - 1 : null;

    let sidecar = null;
    try {
      sidecar = await loadSidecar(baseDir, gameKey);
    } catch (err) {
      warnOnce("missing sidecar", {
        league: leagueParam,
        season: seasonParam,
        week: weekParam,
        gameKey,
        error: err?.message ?? err,
      });
    }

    if (sidecar) {
      const schedAwayNow = sidecar.away_ytd || [];
      const schedHomeNow = sidecar.home_ytd || [];
      const schedAwayPrev = sidecar.away_prev || [];
      const schedHomePrev = sidecar.home_prev || [];

      if (!schedAwayNow.length) {
        warnOnce(`missing current schedule:${awayDisplay}`, { team: awayDisplay, season: seasonParam });
      }
      if (!schedHomeNow.length) {
        warnOnce(`missing current schedule:${homeDisplay}`, { team: homeDisplay, season: seasonParam });
      }
      if (!schedAwayPrev.length) {
        warnOnce(`missing prev-season schedule:${awayDisplay}`, { team: awayDisplay, season: prevSeasonValue });
      }
      if (!schedHomePrev.length) {
        warnOnce(`missing prev-season schedule:${homeDisplay}`, { team: homeDisplay, season: prevSeasonValue });
      }

      fillSchedule("sched1Body", schedAwayNow);
      fillSchedule("sched2Body", schedHomeNow);
      fillSchedule("sched1BodyPrev", schedAwayPrev);
      fillSchedule("sched2BodyPrev", schedHomePrev);
    } else {
      warnOnce(`missing sidecar schedules:${gameKey}`, { gameKey });
    }

    if (Number.isFinite(prevSeasonValue)) {
      const [eoyAway, eoyHome] = await Promise.all([
        loadTeamSeasonStats(leagueParam, prevSeasonValue, awayDisplay),
        loadTeamSeasonStats(leagueParam, prevSeasonValue, homeDisplay),
      ]);
      fillEoy("eoy1", eoyAway);
      fillEoy("eoy2", eoyHome);
    }
  } catch (error) {
    renderError(error);
  }
})();

