/**
 * Printable Game View hydrator.
 *
 * Scope: Populate `web/game_view_printable.html` with week-level data for a single game.
 * Spec anchors: context/merge_summary_global_week.md
 * Invariants: Reads existing JSONL outputs (no schema changes), remains idempotent, UTC inputs converted via browser locale.
 * Side effects: Performs a single fetch to `out/.../games_week_*.jsonl`, writes textContent into known DOM ids.
 * Log contract: Logs to `console.error` when hydration fails; otherwise remains quiet.
 */

import { deriveTopMetrics, formatPlus, normalizeTeamName } from "./game_metrics.js";

const DASH = '—';
const $ = (id) => document.getElementById(id);

// Standard DOM text helpers shared across printable hydration.
const setText = (id, v) => {
  const el = $(id);
  if (!el) return;
  const hasValue = v !== null && v !== undefined && v !== "";
  el.textContent = hasValue ? String(v) : DASH;
};

const setPlus = (id, n) => {
  const num = Number(n);
  if (!Number.isFinite(num)) {
    setText(id, null);
    return;
  }
  setText(id, num > 0 ? `+${num}` : String(num));
};

const setRank = (id, n) => {
  const num = Number(n);
  setText(id, Number.isFinite(num) ? String(Math.trunc(num)) : null);
};

const query = new URLSearchParams(window.location.search);
const leagueParam = (query.get("league") || "NFL").toUpperCase();
const seasonParam = toNumber(query.get("season"));
const weekParam = toNumber(query.get("week"));
const gameKey = (query.get("game_key") || query.get("game") || "").trim();

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
  return parseJsonl(text);
}

function parseJsonl(text) {
  if (!text) return [];
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/);
  const out = [];
  for (const raw of lines) {
    const line = raw && raw.trim();
    if (!line) continue;
    const safe = line
      .replace(/:\s*NaN\b/gi, ": null")
      .replace(/:\s*-Infinity\b/gi, ": null")
      .replace(/:\s*Infinity\b/gi, ": null");
    try {
      out.push(JSON.parse(safe));
    } catch (err) {
      console.warn("PRINTABLE: skip bad JSONL line", {
        snippet: line.slice(0, 120),
        err: err?.message ?? err,
      });
    }
  }
  return out;
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

async function loadCsv(path) {
  const url = new URL(`../${path}`, window.location.href);
  const res = await fetch(url.toString(), { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  const text = await res.text();
  const lines = text.trim().split(/\r?\n/);
  if (lines.length === 0) {
    return [];
  }
  const headers = lines[0].split(",").map((token) => token.trim());
  return lines
    .slice(1)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const values = line.split(",");
      const row = {};
      headers.forEach((header, idx) => {
        row[header] = (values[idx] ?? "").trim();
      });
      return row;
    });
}

const toNum = (value) => {
  if (value === null || value === undefined || value === "" || value === DASH) return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

function assignText(id, value) {
  if (value === null || value === undefined || value === "" || value === DASH) return;
  const el = $(id);
  if (el) {
    el.textContent = String(value);
  }
}

function assignNumber(id, raw, { decimals = 1, signed = false } = {}) {
  const value = typeof raw === "number" ? raw : toNum(raw);
  if (value === null) return;
  const el = $(id);
  if (!el) return;
  el.textContent = formatNumber(value, { decimals, signed });
}

function teamCandidates(row, side) {
  const prefix = side === "home" ? "home" : "away";
  const display = teamName(row, side);
  const candidates = [
    display !== DASH ? display : null,
    row?.[`${prefix}_team_name`] || null,
    row?.[`${prefix}_team_norm`] || null,
    row?.[`${prefix}_team_raw`] || null,
  ]
    .filter(Boolean)
    .filter((value, index, array) => array.indexOf(value) === index);
  return candidates.length ? candidates : [display];
}

function findTeamRow(rows, name) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const str = String(name ?? "").trim();
  if (!str || str === DASH) return null;
  const direct = rows.find((row) => (row.Team || "") === str);
  if (direct) return direct;
  const lower = str.toLowerCase();
  const lowerMatch = rows.find(
    (row) => String(row.Team || "").trim().toLowerCase() === lower
  );
  if (lowerMatch) return lowerMatch;
  const key = normalizeKey(str);
  if (!key) return null;
  return rows.find((row) => buildAliasKeys(row.Team).has(key)) || null;
}

function resolveMetricsRow(rows, candidates) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  for (const candidate of candidates) {
    const match = findTeamRow(rows, candidate);
    if (match) return match;
  }
  return null;
}

function teamName(row, side) {
  const prefix = side === "home" ? "home" : "away";
  const candidates = [
    row?.raw_sources?.[`sagarin_row_${side}`]?.team,
    row?.[`${prefix}_team_name`],
    row?.[`${prefix}_team_norm`],
    row?.[`${prefix}_team_raw`],
  ];
  for (const candidate of candidates) {
    const name = normalizeTeamName(candidate);
    if (name) return name;
  }
  return DASH;
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
  const candidate =
    row?.raw_sources?.schedule_row?.game_no ??
    row?.raw_sources?.schedule_row?.rotation ??
    row?.rotation_number ??
    row?.game_no ??
    row?.raw_sources?.schedule_row?.gsis ??
    null;
  if (candidate === null || candidate === undefined || candidate === "") return DASH;
  const num = Number(candidate);
  if (Number.isFinite(num)) {
    return String(Math.trunc(num));
  }
  const str = String(candidate).trim();
  return str ? str : DASH;
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

function fillTeam(prefix, row, side, metrics) {
  const sideKey = side === "home" ? "HOME" : "AWAY";
  setText(`${prefix}TeamNo`, resolveTeamNumber(row, side));
  setText(`${prefix}GameNo`, resolveGameNumber(row));
  setText(`${prefix}Name`, dash(teamName(row, side)));
  setText(`${prefix}Odds`, spreadFor(side, row));
  const totalSource = metrics?.total ?? row?.total;
  setText(`${prefix}OU`, formatNumber(totalSource, { decimals: 1 }));
  setText(`${prefix}PR`, formatNumber(row?.[`${side}_pr`], { decimals: 2 }));
  const diffDerived =
    metrics && metrics.prDiffFavored !== null && metrics.favoredSide
      ? metrics.favoredSide === sideKey
        ? metrics.prDiffFavored
        : metrics.prDiffFavored * -1
      : null;
  if (diffDerived !== null) {
    setText(`${prefix}Diff`, formatPlus(diffDerived, 2));
  } else {
    setText(`${prefix}Diff`, ratingDiffFor(side, row));
  }
  const rvoDerived =
    metrics && metrics.rvo !== null && metrics.favoredSide
      ? metrics.favoredSide === sideKey
        ? metrics.rvo
        : metrics.rvo * -1
      : null;
  if (rvoDerived !== null) {
    setText(`${prefix}RVO`, formatPlus(rvoDerived, 2));
  } else {
    setText(`${prefix}RVO`, ratingVsOddsFor(side, row));
  }
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

function fillOffDefFromMetrics(prefix, metrics) {
  if (!metrics) return;
  assignNumber(`${prefix}OffRY`, metrics["RY(O)"], { decimals: 1 });
  assignNumber(`${prefix}OffRYRank`, metrics["R(O)_RY"], { decimals: 0 });
  assignNumber(`${prefix}OffPY`, metrics["PY(O)"], { decimals: 1 });
  assignNumber(`${prefix}OffPYRank`, metrics["R(O)_PY"], { decimals: 0 });
  assignNumber(`${prefix}OffTY`, metrics["TY(O)"], { decimals: 1 });
  assignNumber(`${prefix}OffTYRank`, metrics["R(O)_TY"], { decimals: 0 });
  assignNumber(`${prefix}DefRY`, metrics["RY(D)"], { decimals: 1 });
  assignNumber(`${prefix}DefRYRank`, metrics["R(D)_RY"], { decimals: 0 });
  assignNumber(`${prefix}DefPY`, metrics["PY(D)"], { decimals: 1 });
  assignNumber(`${prefix}DefPYRank`, metrics["R(D)_PY"], { decimals: 0 });
  assignNumber(`${prefix}DefTY`, metrics["TY(D)"], { decimals: 1 });
  assignNumber(`${prefix}DefTYRank`, metrics["R(D)_TY"], { decimals: 0 });
  assignNumber(`${prefix}DefTO`, metrics["TO"], { decimals: 1, signed: true });
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

function fillEoy(prefix, metrics, teamLabel) {
  setText(`${prefix}Team`, teamLabel || DASH);
  if (!metrics) return;
  assignNumber(`${prefix}PF`, metrics["PF"], { decimals: 0 });
  assignNumber(`${prefix}PA`, metrics["PA"], { decimals: 0 });
  assignText(`${prefix}SU`, metrics["SU"]);
  assignText(`${prefix}ATS`, metrics["ATS"]);
  assignNumber(`${prefix}OffRY`, metrics["RY(O)"], { decimals: 1 });
  assignNumber(`${prefix}OffRYRank`, metrics["R(O)_RY"], { decimals: 0 });
  assignNumber(`${prefix}OffPY`, metrics["PY(O)"], { decimals: 1 });
  assignNumber(`${prefix}OffPYRank`, metrics["R(O)_PY"], { decimals: 0 });
  assignNumber(`${prefix}OffTY`, metrics["TY(O)"], { decimals: 1 });
  assignNumber(`${prefix}OffTYRank`, metrics["R(O)_TY"], { decimals: 0 });
  assignNumber(`${prefix}DefRY`, metrics["RY(D)"], { decimals: 1 });
  assignNumber(`${prefix}DefRYRank`, metrics["R(D)_RY"], { decimals: 0 });
  assignNumber(`${prefix}DefPY`, metrics["PY(D)"], { decimals: 1 });
  assignNumber(`${prefix}DefPYRank`, metrics["R(D)_PY"], { decimals: 0 });
  assignNumber(`${prefix}DefTY`, metrics["TY(D)"], { decimals: 1 });
  assignNumber(`${prefix}DefTYRank`, metrics["R(D)_TY"], { decimals: 0 });
  assignNumber(`${prefix}DefTO`, metrics["TO"], { decimals: 1, signed: true });
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

async function getSeasonMetrics(league, season) {
  const cacheKey = `${league}:${season}`;
  if (EOY_CACHE.has(cacheKey)) return EOY_CACHE.get(cacheKey);
  const path =
    league === "CFB"
      ? `out/cfb/final_league_metrics_${season}.csv`
      : `out/final_league_metrics_${season}.csv`;
  try {
    const rows = await loadCsv(path);
    const entry = { path, rows };
    EOY_CACHE.set(cacheKey, entry);
    return entry;
  } catch (error) {
    warnOnce(`missing EOY metrics:${league}:${season}`, {
      league,
      season,
      path,
      error: error?.message ?? error,
    });
    const entry = { path, rows: [] };
    EOY_CACHE.set(cacheKey, entry);
    return entry;
  }
}

async function loadTeamSeasonStats(league, season, teamName) {
  if (!Number.isFinite(season)) return null;
  const data = await getSeasonMetrics(league, season);
  if (!Array.isArray(data.rows) || data.rows.length === 0) return null;
  return findTeamRow(data.rows, teamName);
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
    const awayCandidates = teamCandidates(row, "away");
    const homeCandidates = teamCandidates(row, "home");
    const awayLabel =
      awayCandidates.find((value) => value && value !== DASH) || DASH;
    const homeLabel =
      homeCandidates.find((value) => value && value !== DASH) || DASH;

    const leagueKey = leagueParam.toLowerCase();
    const metrics = deriveTopMetrics(row, leagueKey);
    const hfaLabel = document.getElementById("top-hfa");
    if (hfaLabel) {
      const hfaText = formatNumber(metrics.hfa, { decimals: 1 });
      hfaLabel.textContent = `HFA=${hfaText}`;
    }

    fillTeam("t1", row, "away", metrics);
    fillTeam("t2", row, "home", metrics);
    setText("stats1Team", awayLabel !== DASH ? awayLabel : row?.away_team_name);
    setText("stats2Team", homeLabel !== DASH ? homeLabel : row?.home_team_name);

    const gameNo = resolveGameNumber(row);
    setText("t1GameNo", gameNo);
    setText("t2GameNo", gameNo);

    let currentMetricsRows = [];
    if (baseDir && Number.isFinite(seasonParam) && Number.isFinite(weekParam)) {
      const metricsPath = `${baseDir}/league_metrics_${seasonParam}_${weekParam}.csv`;
      try {
        currentMetricsRows = await loadCsv(metricsPath);
      } catch (err) {
        warnOnce(
          `missing current league metrics:${leagueParam}:${seasonParam}:${weekParam}`,
          {
            metricsPath,
            error: err?.message ?? err,
          }
        );
      }
    }

    const awayMetrics = resolveMetricsRow(currentMetricsRows, awayCandidates);
    if (!awayMetrics && currentMetricsRows.length) {
      warnOnce(`missing league metrics row:${awayLabel}`, {
        season: seasonParam,
        week: weekParam,
        team: awayLabel,
        candidates: awayCandidates,
      });
    }
    const homeMetrics = resolveMetricsRow(currentMetricsRows, homeCandidates);
    if (!homeMetrics && currentMetricsRows.length) {
      warnOnce(`missing league metrics row:${homeLabel}`, {
        season: seasonParam,
        week: weekParam,
        team: homeLabel,
        candidates: homeCandidates,
      });
    }

    fillOffDefFromMetrics("t1", awayMetrics);
    fillOffDefFromMetrics("t2", homeMetrics);

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
        warnOnce(`missing current schedule:${awayLabel}`, {
          team: awayLabel,
          season: seasonParam,
        });
      }
      if (!schedHomeNow.length) {
        warnOnce(`missing current schedule:${homeLabel}`, {
          team: homeLabel,
          season: seasonParam,
        });
      }
      if (!schedAwayPrev.length) {
        warnOnce(`missing prev-season schedule:${awayLabel}`, {
          team: awayLabel,
          season: prevSeasonValue,
        });
      }
      if (!schedHomePrev.length) {
        warnOnce(`missing prev-season schedule:${homeLabel}`, {
          team: homeLabel,
          season: prevSeasonValue,
        });
      }

      fillSchedule("sched1Body", schedAwayNow);
      fillSchedule("sched2Body", schedHomeNow);
      fillSchedule("sched1BodyPrev", schedAwayPrev);
      fillSchedule("sched2BodyPrev", schedHomePrev);
    } else {
      warnOnce(`missing sidecar schedules:${gameKey}`, { gameKey });
    }

    let eoyAway = null;
    let eoyHome = null;
    if (Number.isFinite(prevSeasonValue)) {
      for (const candidate of awayCandidates) {
        eoyAway = await loadTeamSeasonStats(leagueParam, prevSeasonValue, candidate);
        if (eoyAway) break;
      }
      if (!eoyAway) {
        warnOnce(`missing EOY row:${awayLabel}`, {
          league: leagueParam,
          season: prevSeasonValue,
          team: awayLabel,
          candidates: awayCandidates,
        });
      }

      for (const candidate of homeCandidates) {
        eoyHome = await loadTeamSeasonStats(leagueParam, prevSeasonValue, candidate);
        if (eoyHome) break;
      }
      if (!eoyHome) {
        warnOnce(`missing EOY row:${homeLabel}`, {
          league: leagueParam,
          season: prevSeasonValue,
          team: homeLabel,
          candidates: homeCandidates,
        });
      }
    }

    fillEoy("eoy1", eoyAway, awayLabel);
    fillEoy("eoy2", eoyHome, homeLabel);
  } catch (error) {
    renderError(error);
  }
})();

