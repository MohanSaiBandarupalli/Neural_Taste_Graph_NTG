"use client";
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, auth, User } from "./api";

type Ctx = { user: User | null; ready: boolean; login: (e: string, p: string) => Promise<User>; signup: (b: any) => Promise<User>; logout: () => void; refresh: () => Promise<void> };
const AuthCtx = createContext<Ctx | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);
  const refresh = useCallback(async () => {
    if (!auth.get()) { setUser(null); setReady(true); return; }
    try { setUser(await api<User>("/v1/me")); } catch { setUser(null); } finally { setReady(true); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);
  const login = async (email: string, password: string) => {
    const r = await api<{ token: string; user: User }>("/v1/auth/login", { method: "POST", json: { email, password } });
    auth.set(r.token); setUser(r.user); return r.user;
  };
  const signup = async (b: any) => {
    const r = await api<{ token: string; user: User }>("/v1/auth/signup", { method: "POST", json: b });
    auth.set(r.token); setUser(r.user); return r.user;
  };
  const logout = () => { auth.clear(); setUser(null); };
  return <AuthCtx.Provider value={{ user, ready, login, signup, logout, refresh }}>{children}</AuthCtx.Provider>;
}

export function useAuth() {
  const c = useContext(AuthCtx);
  if (!c) throw new Error("useAuth outside provider");
  return c;
}

/** Redirects to /login when signed out; optionally enforces roles. */
export function useRequireAuth(roles?: User["role"][]) {
  const { user, ready } = useAuth();
  const router = useRouter();
  useEffect(() => {
    if (!ready) return;
    if (!user) router.replace("/login");
    else if (roles && !roles.includes(user.role)) router.replace("/browse");
  }, [ready, user, roles, router]);
  return { user, ready: ready && !!user && (!roles || roles.includes(user.role)) };
}
