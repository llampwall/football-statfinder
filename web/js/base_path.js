// Detect repo segment when hosted at user.github.io/<repo>/...
// e.g. "/football-statfinder/web/week_view.html" -> "/football-statfinder/"
const pathParts = location.pathname.split('/').filter(Boolean);
const REPO_SEGMENT = pathParts.length > 1 && pathParts[0] !== 'web' ? pathParts[0] : '';
export const BASE_PATH = REPO_SEGMENT ? `/${REPO_SEGMENT}/` : '/';

// Build a site-relative path with the correct base
export function sitePath(p = '') {
  p = String(p);
  // remove leading slash to avoid double slashes
  p = p.replace(/^\/+/, '');
  return BASE_PATH + p;
}

// Build a full URL with optional query params
export function siteUrl(p, params = {}) {
  const u = new URL(sitePath(p), location.origin);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') u.searchParams.set(k, String(v));
  }
  return u.toString();
}
