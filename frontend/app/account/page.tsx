"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConsumerShell } from "@/components/ConsumerShell";
import { Card } from "@/components/ui";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";

function Account() {
  const { user, refresh, logout } = useAuth();
  const router = useRouter();
  const [msg, setMsg] = useState<string | null>(null);
  if (!user) return null;
  const toggle = async (k: "consent_personalization" | "consent_analytics") => { await api("/v1/me/consent", { method: "PATCH", json: { [k]: !user[k] } }); await refresh(); setMsg("Preferences saved."); };
  const exportData = async () => {
    const d = await api("/v1/me/export");
    const url = URL.createObjectURL(new Blob([JSON.stringify(d, null, 2)], { type: "application/json" }));
    const a = document.createElement("a"); a.href = url; a.download = "ntg-my-data.json"; a.click(); URL.revokeObjectURL(url);
  };
  const del = async () => { if (!confirm("Permanently delete your account, profiles and history?")) return; await api("/v1/me", { method: "DELETE" }); logout(); router.push("/"); };
  return (
    <main className="mx-auto max-w-3xl space-y-6 px-6 pt-24">
      <h1 className="font-display text-4xl">Account &amp; privacy</h1>
      <Card title="Signed in"><p className="text-sm">{user.display_name} · {user.email} · role <b>{user.role}</b></p></Card>
      <Card title="Consent">
        <label className="flex items-center justify-between gap-4 py-2 text-sm"><span>Personalise recommendations from my activity<br /><span className="text-xs text-neutral-500">Off: you get non-personalised (popularity-based) recommendations and your history is not used.</span></span><input type="checkbox" checked={user.consent_personalization} onChange={() => toggle("consent_personalization")} /></label>
        <label className="flex items-center justify-between gap-4 py-2 text-sm"><span>Include my activity in aggregate analytics</span><input type="checkbox" checked={user.consent_analytics} onChange={() => toggle("consent_analytics")} /></label>
        {msg && <p className="text-sm text-emerald-300">{msg}</p>}
      </Card>
      <Card title="Your data"><div className="flex flex-wrap gap-3"><button className="btn-ghost" onClick={exportData}>Download my data (JSON)</button><button className="btn border border-red-800 text-red-300 hover:bg-red-950" onClick={del}>Delete my account</button></div></Card>
    </main>
  );
}
export default function Page() { return <ConsumerShell><Account /></ConsumerShell>; }
