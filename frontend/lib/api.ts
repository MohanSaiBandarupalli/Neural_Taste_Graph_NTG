// Thin typed fetch wrapper. Token lives in localStorage (dev); see SECURITY.md for the httpOnly-cookie production path.
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

const TOKEN = "ntg_token";
export const auth = {
  get: () => (typeof window === "undefined" ? null : window.localStorage.getItem(TOKEN)),
  set: (t: string) => { try { window.localStorage.setItem(TOKEN, t); } catch {} },
  clear: () => { try { window.localStorage.removeItem(TOKEN); window.localStorage.removeItem("ntg_profile"); } catch {} },
};
export const activeProfile = {
  get: (): number | null => { try { const v = window.localStorage.getItem("ntg_profile"); return v ? Number(v) : null; } catch { return null; } },
  set: (id: number) => { try { window.localStorage.setItem("ntg_profile", String(id)); } catch {} },
};

export async function api<T = any>(path: string, opts: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers as Record<string, string>) };
  const t = auth.get();
  if (t) headers["Authorization"] = `Bearer ${t}`;
  let body = opts.body;
  if (opts.json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(opts.json); }
  const res = await fetch(`/api${path}`, { ...opts, headers, body, cache: "no-store" });
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401 && typeof window !== "undefined" && !path.startsWith("/v1/auth")) auth.clear();
    const d = (data as any)?.detail;
    throw new ApiError(res.status, typeof d === "string" ? d : Array.isArray(d) ? d.map((x: any) => x.msg).join("; ") : `Request failed (${res.status})`);
  }
  return data as T;
}
