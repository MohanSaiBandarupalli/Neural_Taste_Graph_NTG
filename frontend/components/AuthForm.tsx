"use client";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Brand } from "./Brand";
import { useAuth } from "@/lib/auth";

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const { login, signup } = useAuth();
  const router = useRouter();
  const sp = useSearchParams();
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [name, setName] = useState("");
  const [consent, setConsent] = useState(true); const [analytics, setAnalytics] = useState(false);
  const [err, setErr] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  useEffect(() => { const d = sp.get("demo"); if (d) { setEmail(`${d}@demo.ntg`); setPassword("demo-password-123"); } }, [sp]);
  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setErr(null); setBusy(true);
    try {
      const u = mode === "login" ? await login(email, password)
        : await signup({ email, password, display_name: name, consent_personalization: consent, consent_analytics: analytics });
      router.push(u.role === "viewer" ? "/profiles" : "/enterprise");
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  return (
    <main className="grid min-h-screen place-items-center bg-[radial-gradient(ellipse_at_top,#2a0a0c,transparent_60%)] px-4">
      <form onSubmit={submit} className="card fade-in w-full max-w-md space-y-4 bg-ink-900/90 p-8">
        <Link href="/"><Brand /></Link>
        <h1 className="font-display text-3xl">{mode === "login" ? "Sign in" : "Create your account"}</h1>
        {mode === "signup" && <label className="block"><span className="label">Name</span><input className="input mt-1" value={name} onChange={(e) => setName(e.target.value)} maxLength={80} /></label>}
        <label className="block"><span className="label">Email</span><input className="input mt-1" type="email" required value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" /></label>
        <label className="block"><span className="label">Password</span><input className="input mt-1" type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} /></label>
        {mode === "signup" && <div className="space-y-2 text-sm text-neutral-300">
          <label className="flex gap-2"><input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />Use my viewing activity to personalise recommendations</label>
          <label className="flex gap-2"><input type="checkbox" checked={analytics} onChange={(e) => setAnalytics(e.target.checked)} />Include my activity in aggregate product analytics</label>
          <p className="text-xs text-neutral-500">You can change these, export or delete your data at any time from Account.</p>
        </div>}
        {err && <p role="alert" className="text-sm text-red-400">{err}</p>}
        <button className="btn-primary w-full !py-3" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
        {mode === "login" ? (
          <div className="space-y-2 text-sm text-neutral-400">
            <p>New here? <Link className="text-white underline" href="/signup">Create an account</Link></p>
            <p className="text-xs">Local demo accounts: <button type="button" className="underline" onClick={() => { setEmail("viewer@demo.ntg"); setPassword("demo-password-123"); }}>viewer</button> · <button type="button" className="underline" onClick={() => { setEmail("analyst@demo.ntg"); setPassword("demo-password-123"); }}>analyst</button> · <button type="button" className="underline" onClick={() => { setEmail("admin@demo.ntg"); setPassword("demo-password-123"); }}>admin</button> (development seed only)</p>
          </div>
        ) : <p className="text-sm text-neutral-400">Already have an account? <Link className="text-white underline" href="/login">Sign in</Link></p>}
      </form>
    </main>
  );
}
