"use client";
import { useState } from "react";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { Card, ErrorBox, Spinner, Table } from "@/components/ui";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";

export default function Experiments() {
  const { data, error, loading, reload } = useApi<any[]>("/v1/enterprise/experiments");
  const [form, setForm] = useState({ key: "", name: "", hypothesis: "", control: "greedy_fresh", treatment: "journey_orchestrator", share: 0.5, metric: "satisfaction_per_journey" });
  const [ss, setSs] = useState({ baseline_mean: 1.0, baseline_std: 1.2, mde_pct: 0.03, alpha: 0.05, power: 0.8 });
  const [ssRes, setSsRes] = useState<any>(null); const [msg, setMsg] = useState<string | null>(null);
  const [lookup, setLookup] = useState({ key: "journey-vs-greedy", user: 42 }); const [arm, setArm] = useState<any>(null);
  const create = async () => {
    try { await api("/v1/enterprise/experiments", { method: "POST", json: { key: form.key, name: form.name, hypothesis: form.hypothesis, primary_metric: form.metric,
      arms: [{ name: "control", share: 1 - form.share, policy: form.control }, { name: "treatment", share: form.share, policy: form.treatment }] } }); setMsg("Experiment created."); reload(); }
    catch (e: any) { setMsg(e.message); }
  };
  return (
    <EnterpriseShell title="Experiment management" subtitle="Define policy experiments, size them, and check deterministic user assignment. Online results require live traffic — none are shown or invented here.">
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <Card title="Experiments"><Table rows={data} cols={[{ key: "key", label: "Key" }, { key: "name", label: "Name" }, { key: "status", label: "Status" }, { key: "primary_metric", label: "Primary metric" },
        { key: "arms", label: "Arms", fmt: (v) => v.map((a: any) => `${a.name}: ${a.policy} (${Math.round(a.share * 100)}%)`).join(" · ") }]} /></Card>}
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="New experiment"><div className="space-y-2">
          <input className="input" placeholder="key (e.g. explore-v2)" value={form.key} onChange={(e) => setForm({ ...form, key: e.target.value })} />
          <input className="input" placeholder="name" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          <textarea className="input" placeholder="hypothesis" value={form.hypothesis} onChange={(e) => setForm({ ...form, hypothesis: e.target.value })} />
          <div className="grid grid-cols-2 gap-2"><label className="text-xs text-neutral-400">Control policy<select className="input mt-1" value={form.control} onChange={(e) => setForm({ ...form, control: e.target.value })}>{["greedy_relevance", "greedy_fresh", "mmr_diversity", "balanced", "familiar"].map((p) => <option key={p}>{p}</option>)}</select></label>
            <label className="text-xs text-neutral-400">Treatment policy<select className="input mt-1" value={form.treatment} onChange={(e) => setForm({ ...form, treatment: e.target.value })}>{["journey_orchestrator", "genre_thompson", "exploratory", "unexpected"].map((p) => <option key={p}>{p}</option>)}</select></label></div>
          <label className="block text-xs text-neutral-400">Treatment share {form.share}<input type="range" className="w-full accent-[#e50914]" min={0.05} max={0.95} step={0.05} value={form.share} onChange={(e) => setForm({ ...form, share: Number(e.target.value) })} /></label>
          <button className="btn-primary w-full" onClick={create}>Create</button>{msg && <p className="text-sm text-neutral-300">{msg}</p>}</div></Card>
        <Card title="Sample-size calculator"><div className="space-y-2">{Object.entries(ss).map(([k, v]) => <label key={k} className="block text-xs text-neutral-400">{k.replace(/_/g, " ")}<input className="input mt-1" type="number" step="0.01" value={v} onChange={(e) => setSs({ ...ss, [k]: Number(e.target.value) })} /></label>)}
          <button className="btn-ghost w-full" onClick={async () => { try { setSsRes(await api("/v1/enterprise/experiments/sample-size", { method: "POST", json: ss })); } catch (e: any) { setSsRes({ error: e.message }); } }}>Compute</button>
          {ssRes && (ssRes.error ? <p className="text-sm text-red-400">{ssRes.error}</p> : <p className="text-sm"><b>{ssRes.n_per_arm.toLocaleString()}</b> users per arm ({ssRes.method}).</p>)}</div></Card>
        <Card title="Assignment lookup"><div className="space-y-2">
          <input className="input" value={lookup.key} onChange={(e) => setLookup({ ...lookup, key: e.target.value })} /><input className="input" type="number" value={lookup.user} onChange={(e) => setLookup({ ...lookup, user: Number(e.target.value) })} />
          <button className="btn-ghost w-full" onClick={async () => { try { setArm(await api(`/v1/enterprise/experiments/${lookup.key}/assignment?user_id=${lookup.user}`)); } catch (e: any) { setArm({ error: e.message }); } }}>Assign</button>
          {arm && (arm.error ? <p className="text-sm text-red-400">{arm.error}</p> : <p className="text-sm">User {arm.user_id} → <b>{arm.arm}</b> (bucket {arm.bucket.toFixed(4)}; {arm.method})</p>)}</div></Card>
      </div>
    </EnterpriseShell>
  );
}
