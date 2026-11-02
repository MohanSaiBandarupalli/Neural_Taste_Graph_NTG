"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { HBar } from "@/components/charts";
import { Card, ErrorBox, Evidence, Spinner, Stat } from "@/components/ui";
import { fmtNum, fmtPct, useApi } from "@/lib/hooks";

export default function Fatigue() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/fatigue");
  const ev = data?.evaluation;
  return (
    <EnterpriseShell title="Fatigue intelligence" subtitle="Fatigue = a satisfaction (rating) drop among viewers who stay active. Inactivity alone is never treated as fatigue.">
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <>
        <div className="grid gap-4 md:grid-cols-4">
          <Stat label="Mean fatigue risk" value={fmtPct(data.summary?.mean)} evidence="modeled" />
          <Stat label="Elevated (>30%)" value={fmtPct(data.summary?.share_elevated)} evidence="modeled" />
          <Stat label="Test AUC" value={fmtNum(ev?.test.auc, 3)} sub={`baseline (genre concentration only) ${fmtNum(ev?.test.baseline_recent_hhi_auc, 3)}`} evidence="observed" />
          <Stat label="Test base rate" value={fmtPct(ev?.test.positive_rate)} sub={`${ev?.test.n_users} active users`} evidence="observed" />
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Mean fatigue risk by segment" right={<Evidence type="modeled" />}><HBar data={(data.by_segment ?? []).map((s: any) => ({ s: s.name, v: s.mean_fatigue_risk }))} x="v" y="s" fmt={(v) => `${(v * 100).toFixed(0)}%`} /></Card>
          <Card title="Model coefficients (standardised)" right={<Evidence type="observed" />}><HBar data={Object.entries(ev?.test.coefficients ?? {}).map(([k, v]) => ({ k, v }))} x="v" y="k" fmt={(v) => v.toFixed(2)} colorBy={(d) => (d.v >= 0 ? "var(--series-8)" : "var(--series-3)")} />
            <p className="text-xs text-neutral-500">Honest note: on this dataset the multi-signal model does not beat genre concentration alone; it is kept for its per-signal explanations.</p></Card>
        </div>
      </>}
    </EnterpriseShell>
  );
}
