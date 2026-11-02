"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "./api";

/** Fetch with loading / error state. Errors are shown, never silently swallowed. */
export function useApi<T = any>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const load = useCallback(async () => {
    if (!path) return;
    setLoading(true); setError(null);
    try { setData(await api<T>(path)); } catch (e: any) { setError(e.message || "Request failed"); } finally { setLoading(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  useEffect(() => { load(); }, [load]);
  return { data, error, loading, reload: load, setData };
}

export const fmtPct = (x: number | null | undefined, d = 1) => (x == null ? "—" : `${(100 * x).toFixed(d)}%`);
export const fmtNum = (x: number | null | undefined, d = 3) => (x == null ? "—" : Number(x).toFixed(d));
export const fmtMoney = (x: number) => x.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 0 });
