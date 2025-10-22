const DASH = "—";

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

export function deriveTopMetrics(row) {
  if (!row) {
    return {
      total: null,
      prDiffFavored: null,
      rvo: null,
      favoredTeam: null,
      favoredSide: null,
    };
  }

  const total = coalesceNumber(row.total, row.schedule_total);

  const favoredSide = String(row?.favored_side || "")
    .trim()
    .toUpperCase();
  const homePR = toNumber(row?.home_pr);
  const awayPR = toNumber(row?.away_pr);
  const hfa = coalesceNumber(row?.hfa, row?.raw_sources?.sagarin_row_home?.hfa);

  let prDiffFavored = null;
  if (homePR !== null && awayPR !== null) {
    const homeEdge = homePR + (hfa ?? 0);
    const diff = homeEdge - awayPR;
    if (favoredSide === "HOME") {
      prDiffFavored = diff;
    } else if (favoredSide === "AWAY") {
      prDiffFavored = diff * -1;
    } else {
      prDiffFavored = diff;
    }
  }

  let rvo = toNumber(row?.rating_vs_odds);
  if (rvo === null) {
    rvo = toNumber(row?.rating_vs_odds_favored_team);
  }

  const spreadFavored = coalesceNumber(row?.spread_favored_team);
  const spreadHome = coalesceNumber(row?.spread_home_relative);
  const spreadAbs =
    spreadFavored !== null
      ? Math.abs(spreadFavored)
      : spreadHome !== null
      ? Math.abs(spreadHome)
      : null;

  if (rvo === null && prDiffFavored !== null && spreadAbs !== null) {
    rvo = prDiffFavored - spreadAbs;
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
  };
}

export const GameMetrics = {
  formatNum,
  formatPlus,
  formatRank,
  deriveTopMetrics,
  normalizeTeamName,
};
