"use client";
import { useState } from "react";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { Lines, VBars } from "@/components/charts";
import { Card, ErrorBox, Evidence, Spinner, Table } from "@/components/ui";
import { api } from "@/lib/api";
import { fmtNum, useApi } from "@/lib/hooks";

const PARAMS = [
  ["fatigue_sensitivity", "Fatigue sensitivity", 0, 2, 0.1], ["repeat_penalty", "Accept prob. for re-shown items (×)", 0, 1, 0.05],
  ["discovery_bonus", "Discovery bonus", 0, 2, 0.1], ["leave_base", "Base leave prob./session", 0, 0.2, 0.01],
  ["leave_fatigue", "Extra leave prob. from fatigue", 0, 1, 0.05], ["steps", "Sessions", 5, 40, 1], ["slate_size", "Slate size", 3, 10, 1],
] as const;

export default function Journey() {
  const rep = useApi<any>("/v1/enterprise/reports/journey");
  const [cfg, setCfg] = useState<Record<string, number>>({ fatigue_sensitivity: 1, repeat_penalty: 0.4, discovery_bonus: 0.5, leave_base: 0.02, leave_fatigue: 0.15, steps: 25, slate_size: 5, n_users: 200 });
  const [res, setRes] = useState<any>(null); const [err, setErr] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  const run = async () => { setBusy(true); setErr(null); try { setRes(await api("/v1/enterprise/journey/simulate", { method: "POST", json: { ...cfg, policies: ["greedy_relevance", "greedy_fresh", "mmr_diversity", "epsilon_greedy", "genre_thompson", "journey_orchestrator"] } })); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } };
  const pol = (res ?? rep.data?.default_world)?.policies;
  const sens = rep.data?.sensitivity_fatigue?.rows;
  return (
    <EnterpriseShell title="Journey simulator" subtitle="Compare recommendation policies over multi-session viewing journeys before deployment. Results are SIMULATED under the assumptions you set — not causal estimates." actions={<Evidence type="simulated" />}>
      {rep.error && <ErrorBox error={rep.error} />}
      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="World assumptions" className="xl:col-span-1">
          <div className="space-y-3">{PARAMS.map(([k, label, lo, hi, step]) => <label key={k} className="block text-sm"><div className="flex justify-between"><span>{label}</span><span className="tabular-nums text-neutral-400">{cfg[k]}</span></div>
            <input className="w-full accent-[#e50914]" type="range" min={lo} max={hi} step={step} value={cfg[k]} onChange={(e) => setCfg({ ...cfg, [k]: Number(e.target.value) })} /></label>)}</div>
          <button className="btn-primary mt-4 w-full" onClick={run} disabled={busy}>{busy ? "Simulating…" : "Run simulation"}</button>
          {err && <p className="mt-2 text-sm text-red-400">{err}</p>}
          <p className="mt-3 text-xs text-neutral-500">World model: {rep.data?.default_world?.world_model?.oracle}. Policies only see past-only models and in-journey feedback.</p>
        </Card>
        <div className="space-y-4 xl:col-span-2">
          {busy && <Spinner label="Simulating" />}
          {pol && <Card title={res ? "Your scenario" : "Default assumptions (from latest build)"}>
            <VBars data={Object.entries(pol).map(([k, v]: any) => ({ p: k.replace(/_/g, " "), sat: v.satisfaction_per_journey, acc: v.accepts_per_journey }))} x="p" series={["sat", "acc"]} names={{ sat: "satisfaction / journey", acc: "accepts / journey" }} fmt={(v) => v.toFixed(1)} height={260} />
            <Table rows={Object.entries(pol).map(([k, v]: any) => ({ k, ...v }))} cols={[
              { key: "k", label: "Policy", fmt: (v) => v.replace(/_/g, " ") },
              { key: "immediate_accept_rate", label: "Session-1 accept", align: "right", fmt: (v) => fmtNum(v, 3) },
              { key: "satisfaction_per_journey", label: "Satisfaction [95% CI]", align: "right", fmt: (v, r) => `${fmtNum(v, 2)} [${fmtNum(r.satisfaction_ci95[0], 2)}–${fmtNum(r.satisfaction_ci95[1], 2)}]` },
              { key: "sessions_before_leaving", label: "Sessions kept", align: "right", fmt: (v) => fmtNum(v, 1) },
              { key: "genres_discovered", label: "New genres", align: "right", fmt: (v) => fmtNum(v, 2) },
              { key: "satisfaction_lift_vs_greedy_fresh", label: "Lift vs fresh greedy", align: "right", fmt: (v, r) => v == null ? "—" : `${v >= 0 ? "+" : ""}${fmtNum(v, 2)} [${fmtNum(r.satisfaction_lift_vs_greedy_fresh_ci95?.[0], 2)}, ${fmtNum(r.satisfaction_lift_vs_greedy_fresh_ci95?.[1], 2)}]` },
            ]} />
            <p className="mt-2 text-xs text-neutral-500">Session-1 accept = immediate engagement; satisfaction and sessions kept = long-term outcomes. Lift CIs are paired bootstrap over the same simulated viewers.</p>
          </Card>}
          {sens && <Card title="Sensitivity: when does the orchestrator help?">
            <Lines data={sens.map((r: any) => ({ f: String(r.fatigue_sensitivity), orch: r.journey_orchestrator.lift_vs_greedy_fresh, mmr: r.mmr_diversity.lift_vs_greedy_fresh }))} x="f" series={["orch", "mmr"]} names={{ orch: "journey orchestrator", mmr: "MMR diversity" }} fmt={(v) => v.toFixed(1)} ref0 />
            <p className="text-xs text-neutral-500">Satisfaction lift over fresh greedy (y) as viewers&apos; fatigue sensitivity (x) increases. Near zero fatigue there is little to gain; the claim is conditional on fatigue being real.</p>
          </Card>}
          {rep.data?.off_policy_evaluation && <Card title="Off-policy evaluation check">
            {(() => { const o = rep.data.off_policy_evaluation; return <p className="text-sm text-neutral-300">{o.protocol}. True value {fmtNum(o.true_target_value, 4)} · IPS {fmtNum(o.ips_estimate, 4)} [{fmtNum(o.ips_ci95[0], 4)}, {fmtNum(o.ips_ci95[1], 4)}] · SNIPS {fmtNum(o.snips_estimate, 4)} [{fmtNum(o.snips_ci95[0], 4)}, {fmtNum(o.snips_ci95[1], 4)}] · naive logged average {fmtNum(o.naive_logged_average, 4)} (biased).</p>; })()}
          </Card>}
        </div>
      </div>
    </EnterpriseShell>
  );
}
