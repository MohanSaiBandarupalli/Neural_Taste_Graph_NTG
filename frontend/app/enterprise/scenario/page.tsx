"use client";
import { useEffect, useState } from "react";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { Lines } from "@/components/charts";
import { Card, ErrorBox, Evidence, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { fmtMoney, fmtPct } from "@/lib/hooks";

const FIELDS: [string, string, number][] = [["n_subscribers", "Subscribers", 1], ["monthly_price", "Monthly price ($)", 0.01], ["baseline_monthly_churn", "Baseline monthly churn", 0.001],
  ["retention_improvement_rel", "Relative churn reduction from intervention", 0.005], ["intervention_reach", "Share of subscribers reached", 0.01],
  ["intervention_cost_per_user_month", "Intervention cost / reached user / month ($)", 0.01], ["gross_margin", "Gross margin", 0.01], ["horizon_months", "Horizon (months)", 1], ["uncertainty_rel", "± uncertainty on improvement", 0.05]];

export default function Scenario() {
  const [a, setA] = useState<Record<string, number> | null>(null);
  const [r, setR] = useState<any>(null); const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api("/v1/enterprise/scenario/defaults").then(setA).catch((e) => setErr(e.message)); }, []);
  useEffect(() => { if (!a) return; const t = setTimeout(() => api("/v1/enterprise/scenario", { method: "POST", json: a }).then((x) => { setR(x); setErr(null); }).catch((e) => setErr(e.message)), 250); return () => clearTimeout(t); }, [a]);
  return (
    <EnterpriseShell title="Business decision simulator" subtitle="Transparent scenario arithmetic on assumptions you control. Not company financials and not a forecast." actions={<Evidence type="simulated" />}>
      {err && <ErrorBox error={err} />}
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Assumptions (all editable)">{a && <div className="space-y-3">{FIELDS.map(([k, l, step]) => <label key={k} className="block"><span className="text-xs text-neutral-400">{l}</span><input className="input mt-1" type="number" step={step} value={a[k]} onChange={(e) => setA({ ...a, [k]: Number(e.target.value) })} /></label>)}</div>}</Card>
        <div className="space-y-4 xl:col-span-2">{r && <>
          <div className="grid gap-4 md:grid-cols-2">
            <Stat label="Retained subscribers (end)" value={Math.round(r.central.retained_subscribers_end).toLocaleString()} evidence="simulated" />
            <Stat label="Gross modeled revenue" value={fmtMoney(r.central.incremental_revenue)} evidence="simulated" />
            <Stat label="Intervention expense" value={fmtMoney(r.central.intervention_cost)} evidence="simulated" />
            <Stat label="Net modeled impact" value={<span className={r.central.net_impact >= 0 ? "text-emerald-300" : "text-red-300"}>{fmtMoney(r.central.net_impact)}</span>} sub={`range ${fmtMoney(r.sensitivity.low.net_impact)} to ${fmtMoney(r.sensitivity.high.net_impact)}`} evidence="simulated" />
          </div>
          <Card title="Incremental retained subscribers by month"><Lines data={r.central.monthly.map((m: any) => ({ month: m.month, retained: m.retained }))} x="month" series={["retained"]} fmt={(v) => Math.round(v).toLocaleString()} /></Card>
          <Card><p className="text-sm">Break-even relative churn reduction: <b>{r.breakeven_retention_improvement_rel == null ? "not reachable" : fmtPct(r.breakeven_retention_improvement_rel, 2)}</b> — the intervention must reduce churn by at least this much among reached subscribers to pay for itself.</p><p className="mt-2 text-xs text-neutral-500">{r.disclaimer}</p></Card>
        </>}</div>
      </div>
    </EnterpriseShell>
  );
}
