// Thin typed fetch wrapper over the same-origin /api proxy. Auth is cookie-based (see SECURITY.md).
export type Item = {
  item_id: number; title: string; year: number | null; genres: string[]; tags: string[];
  avg_rating: number | null; n_ratings: number; popularity_percentile: number;
  rank?: number; score?: number; similarity?: number; explanation?: Explanation;
};
export type Reason = { type: string; text: string; evidence: Record<string, unknown>[] };
export type Explanation = { reasons: Reason[]; retrieved_by: string[]; ranker_drivers: { feature: string; text: string; contribution: number }[]; grounding: string };
export type User = { id: number; email: string; display_name: string; role: "viewer" | "analyst" | "admin"; consent_personalization: boolean; consent_analytics: boolean };
export type Profile = { id: number; name: string; color: string; has_imported_history: boolean; n_app_events: number };

export class ApiError extends Error { constructor(public status: number, msg: string) { super(msg); } }

// Session = httpOnly cookies set by the API (access + rotating refresh). JavaScript never sees the
// tokens; it only echoes the readable CSRF cookie in a header on state-changing requests.
function csrfToken(): string | null {
  if (typeof document === "undefined") return null;
  const m = document.cookie.match(/(?:^|; )ntg_csrf=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

export const auth = {
  hasSession: () => !!csrfToken(),
  clear: () => { try { window.localStorage.removeItem("ntg_profile"); } catch {} },
};
export const activeProfile = {
  get: (): number | null => { try { const v = window.localStorage.getItem("ntg_profile"); return v ? Number(v) : null; } catch { return null; } },
  set: (id: number) => { try { window.localStorage.setItem("ntg_profile", String(id)); } catch {} },
};

let refreshing: Promise<boolean> | null = null;
async function refreshSession(): Promise<boolean> {
  // one refresh in flight at a time; concurrent 401s wait for it
  refreshing ??= fetch("/api/v1/auth/refresh", { method: "POST", credentials: "same-origin" })
    .then((r) => r.ok).catch(() => false).finally(() => { setTimeout(() => { refreshing = null; }, 0); });
  return refreshing;
}

export async function api<T = any>(path: string, opts: RequestInit & { json?: unknown } = {}, retried = false): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers as Record<string, string>) };
  const method = (opts.method ?? "GET").toUpperCase();
  if (method !== "GET" && method !== "HEAD") { const c = csrfToken(); if (c) headers["X-CSRF-Token"] = c; }
  let body = opts.body;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(opts.json); }
  const res = await fetch(`/api${path}`, { ...opts, headers, body, cache: "no-store", credentials: "same-origin" });
  if (res.status === 401 && !retried && !path.startsWith("/v1/auth") && (await refreshSession())) {
    return api<T>(path, opts, true);  // access token expired: rotated refresh succeeded, retry once
  }
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401 && typeof window !== "undefined" && !path.startsWith("/v1/auth")) auth.clear();
    const d = (data as any)?.detail;
    throw new ApiError(res.status, typeof d === "string" ? d : Array.isArray(d) ? d.map((x: any) => x.msg).join("; ") : `Request failed (${res.status})`);
  }
  return data as T;
}
