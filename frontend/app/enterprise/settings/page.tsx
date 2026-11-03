"use client";
import { useState } from "react";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { Card, ErrorBox, Table } from "@/components/ui";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

function Admin() {
  const users = useApi<any[]>("/v1/admin/users"); const keys = useApi<any[]>("/v1/admin/api-keys");
  const audit = useApi<any[]>("/v1/admin/audit?limit=50"); const usage = useApi<any[]>("/v1/admin/usage");
  const [nu, setNu] = useState({ email: "", password: "", role: "analyst" }); const [kn, setKn] = useState(""); const [newKey, setNewKey] = useState<string | null>(null); const [err, setErr] = useState<string | null>(null);
  return <>
    {err && <ErrorBox error={err} />}
    <div className="grid gap-4 xl:grid-cols-2">
      <Card title="Team members">{users.data && <Table rows={users.data} cols={[{ key: "display_name", label: "Name" }, { key: "email", label: "Email" }, { key: "role", label: "Role" }]} />}
        <div className="mt-4 grid grid-cols-4 gap-2"><input className="input col-span-2" placeholder="email" value={nu.email} onChange={(e) => setNu({ ...nu, email: e.target.value })} /><input className="input" type="password" placeholder="password (8+)" value={nu.password} onChange={(e) => setNu({ ...nu, password: e.target.value })} />
          <select className="input" value={nu.role} onChange={(e) => setNu({ ...nu, role: e.target.value })}>{["viewer", "analyst", "admin"].map((r) => <option key={r}>{r}</option>)}</select></div>
        <button className="btn-ghost mt-2" onClick={async () => { try { await api("/v1/admin/users", { method: "POST", json: nu }); setNu({ email: "", password: "", role: "analyst" }); users.reload(); } catch (e: any) { setErr(e.message); } }}>Add member</button></Card>
      <Card title="API keys (B2B recommendation API)">{keys.data && <Table rows={keys.data} cols={[{ key: "name", label: "Name" }, { key: "prefix", label: "Prefix", fmt: (v) => <code>{v}…</code> }, { key: "revoked", label: "Status", fmt: (v) => (v ? "revoked" : "active") },
          { key: "last_used_at", label: "Last used", fmt: (v) => v ?? "never" }, { key: "id", label: "", fmt: (v, r) => !r.revoked && <button className="text-xs text-red-300 hover:underline" onClick={async () => { await api(`/v1/admin/api-keys/${v}`, { method: "DELETE" }); keys.reload(); }}>Revoke</button> }]} />}
        <div className="mt-4 flex gap-2"><input className="input" placeholder="key name" value={kn} onChange={(e) => setKn(e.target.value)} /><button className="btn-ghost" onClick={async () => { try { const k = await api<any>("/v1/admin/api-keys", { method: "POST", json: { name: kn } }); setNewKey(k.key); setKn(""); keys.reload(); } catch (e: any) { setErr(e.message); } }}>Create</button></div>
        {newKey && <p className="mt-2 break-all rounded bg-amber-950/40 p-2 text-xs text-amber-200">Copy now — shown once: <code>{newKey}</code></p>}</Card>
    </div>
    <div className="grid gap-4 xl:grid-cols-2">
      <Card title="Usage">{usage.data && (usage.data.length ? <Table rows={usage.data} cols={[{ key: "day", label: "Day" }, { key: "route", label: "Route" }, { key: "count", label: "Requests", align: "right" }]} /> : <p className="text-sm text-neutral-400">No metered usage yet.</p>)}</Card>
      <Card title="Audit log">{audit.data && <Table rows={audit.data} cols={[{ key: "ts", label: "When", fmt: (v) => v.replace("T", " ").slice(0, 19) }, { key: "action", label: "Action" }, { key: "detail", label: "Detail" }]} />}</Card>
    </div>
    <Card title="Plans (proposals — pricing is a hypothesis, see STARTUP_STRATEGY.md)"><div className="grid gap-3 text-sm md:grid-cols-4">{[["Developer", "Sandbox, synthetic data, rate-limited API"], ["Startup", "Hosted models on your catalogue, TasteTwin + journeys"], ["Business", "Enterprise analytics, simulator, experiments"], ["Enterprise", "Private deployment, SSO, custom models, SLAs"]].map(([n, d]) => <div key={n} className="rounded-lg border border-ink-700 p-3"><div className="font-semibold">{n}</div><div className="mt-1 text-neutral-400">{d}</div></div>)}</div></Card>
  </>;
}

export default function Settings() {
  const { user } = useAuth();
  return <EnterpriseShell title="Settings & administration" subtitle="Organisation members, roles, API keys, metered usage and the audit trail.">
    {user?.role === "admin" ? <Admin /> : <Card><p className="text-sm text-neutral-300">Administration requires the admin role. Sign in as admin@demo.ntg in the local demo.</p></Card>}
  </EnterpriseShell>;
}
