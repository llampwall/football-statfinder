const DASH = "—";
const DEFAULT_HFA = { nfl: 2.2, cfb: 2.1 };

const trimTrailingZeros = (str) =>
  str.includes(".") ? str.replace(/\.0+$|(\.\d*?[1-9])0+$/u, "$1") : str;

const toNumber = (value) => {
  if (value === null || value === undefined || value === "") return null;
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const coalesceNumber = (...values) => {
  for (const value of values) {
    const num = toNumber(value);
    if (num !== null) return num;
  }
  return null;
};

const formatFixed = (value, decimals = 1) => {
  const options = {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  };
  return trimTrailingZeros(value.toLocaleString("en-US", options));
};

export function formatNum(value, decimals = 1) {
  const num = toNumber(value);
  if (num === null) return DASH;
  return formatFixed(num, decimals);
}

export function formatPlus(value, decimals = 1) {
  const num = toNumber(value);
  if (num === null) return DASH;
  const formatted = formatFixed(num, decimals);
  return num > 0 ? `+${formatted}` : formatted;
}

export function formatRank(value) {
  const num = toNumber(value);
  if (num === null) return DASH;
  return String(Math.trunc(num));
}

export function normalizeTeamName(name) {
  if (name === null || name === undefined) return null;
  const str = String(name).trim();
  if (!str) return null;
  if (str.includes(",")) {
    const [nickname, city] = str.split(",").map((part) => part.trim());
    if (nickname && city) {
      return `${city} ${nickname}`.replace(/\s+/g, " ").trim();
    }
  }
  return str;
}

function teamNameFromRow(row, side) {
  const prefix = side === "home" ? "home" : "away";
  const sources = [
    row?.raw_sources?.[`sagarin_row_${side}`]?.team,
    row?.[`${prefix}_team_name`],
    row?.[`${prefix}_team_norm`],
    row?.[`${prefix}_team_raw`],
  ];
  for (const source of sources) {
    const normalized = normalizeTeamName(source);
    if (normalized) return normalized;
  }
  return null;
}

export function getHfa(row, league = "nfl") {
  const sources = [
    row?.hfa,
    row?.hfa_adjust,
    row?.raw_sources?.sagarin_row_home?.hfa,
    row?.raw_sources?.sagarin_row_away?.hfa,
  ];
  for (const source of sources) {
    const value = toNumber(source);
    if (value !== null) return value;
  }
  const leagueKey = String(league || "nfl").toLowerCase();
  return DEFAULT_HFA[leagueKey] ?? 0;
}

function resolveFavoredSide(row) {
  let favoredSide = String(row?.favored_side || "").trim().toUpperCase();
  if (favoredSide) return favoredSide;
  const spreadHome = toNumber(row?.spread_home_relative ?? row?.spread);
  if (spreadHome !== null) {
    if (spreadHome < 0) return "HOME";
    if (spreadHome > 0) return "AWAY";
  }
  const diff = toNumber(row?.rating_diff);
  if (diff !== null) return diff >= 0 ? "HOME" : "AWAY";
  return "";
}

export function deriveTopMetrics(row, league = "nfl") {
  if (!row) {
    return {
      total: null,
      prDiffFavored: null,
      rvo: null,
      favoredTeam: null,
      favoredSide: null,
    };
  }

  const leagueKey = String(league || "nfl").toLowerCase();
  const hfa = getHfa(row, leagueKey);
  const homePR = toNumber(row?.home_pr);
  const awayPR = toNumber(row?.away_pr);
  let favoredSide = resolveFavoredSide(row);

  let favoredPr = null;
  let unfavoredPr = null;
  if (favoredSide === "HOME") {
    if (homePR !== null) favoredPr = homePR + hfa;
    if (awayPR !== null) unfavoredPr = awayPR;
  } else if (favoredSide === "AWAY") {
    if (awayPR !== null) favoredPr = awayPR;
    if (homePR !== null) unfavoredPr = homePR + hfa;
  } else if (homePR !== null && awayPR !== null) {
    favoredPr = homePR + hfa;
    unfavoredPr = awayPR;
    favoredSide = "HOME";
  }

  let prDiffFavored = null;
  if (favoredPr !== null && unfavoredPr !== null) {
    prDiffFavored = favoredPr - unfavoredPr;
  } else {
    const diffFavoredField = toNumber(row?.rating_diff_favored_team);
    if (diffFavoredField !== null) {
      prDiffFavored = diffFavoredField;
    } else {
      const diffHome = toNumber(row?.rating_diff);
      if (diffHome !== null) {
        prDiffFavored = favoredSide === "AWAY" ? diffHome * -1 : diffHome;
      }
    }
  }

  const total = coalesceNumber(row?.total, row?.schedule_total);

  let rvo = toNumber(row?.rating_vs_odds ?? row?.rating_vs_odds_favored_team);
  const spreadCandidates = [
    toNumber(row?.spread_favored_team),
    toNumber(row?.spread_home_relative),
    toNumber(row?.spread),
  ];
  const spreadMagnitude = (() => {
    for (const candidate of spreadCandidates) {
      if (candidate !== null) return Math.abs(candidate);
    }
    return null;
  })();
  if (rvo === null && prDiffFavored !== null && spreadMagnitude !== null) {
    rvo = prDiffFavored - spreadMagnitude;
  }

  const favoredTeam =
    favoredSide === "HOME"
      ? teamNameFromRow(row, "home")
      : favoredSide === "AWAY"
      ? teamNameFromRow(row, "away")
      : null;

  return {
    total,
    prDiffFavored,
    rvo,
    favoredTeam,
    favoredSide: favoredSide || null,
    hfa,
  };
}

export const GameMetrics = {
  formatNum,
  formatPlus,
  formatRank,
  deriveTopMetrics,
  normalizeTeamName,
  getHfa,
  DEFAULT_HFA,
};
