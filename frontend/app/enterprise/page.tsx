"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { HBar, VBars } from "@/components/charts";
import { Card, ErrorBox, Evidence, Spinner, Stat } from "@/components/ui";
import { fmtNum, fmtPct, useApi } from "@/lib/hooks";

export default function Dashboard() {
  const { data, error, loading, reload } = useApi<any>("/v1/enterprise/overview");
  return (
    <EnterpriseShell title="Executive dashboard" subtitle="Platform health, model quality and audience risk at a glance. Every figure is labelled observed, modeled or simulated.">
      {error && <ErrorBox error={error} onRetry={reload} />}{loading && <Spinner />}
      {data && <>
        {data.dataset.is_synthetic && <div className="rounded-lg border border-fuchsia-800/60 bg-fuchsia-950/30 p-3 text-sm text-fuchsia-200"><Evidence type="synthetic" className="mr-2" />This deployment runs on synthetic MovieLens-schema data. Numbers demonstrate the system; they are not real-viewer results.</div>}
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Stat label="Viewers / titles" value={`${(data.dataset.n_users / 1000).toFixed(1)}k / ${(data.dataset.n_items / 1000).toFixed(1)}k`} sub={`${data.dataset.n_interactions.toLocaleString()} interactions`} evidence="observed" />
          <Stat label="Hybrid NDCG@10" value={fmtNum(data.kpis["hybrid_ndcg@10"], 4)} sub={`vs popularity ${fmtNum(data.kpis["popularity_ndcg@10"], 4)} · ${data.kpis.n_eval_users} test users`} evidence="observed" />
          <Stat label="High engagement risk" value={fmtPct(data.risk_summary?.share_high)} sub={`of ${data.risk_summary?.n_recently_active} recently active viewers (P>50%)`} evidence="modeled" />
          <Stat label="Elevated fatigue" value={fmtPct(data.fatigue_summary?.share_elevated)} sub={`mean fatigue risk ${fmtPct(data.fatigue_summary?.mean)}`} evidence="modeled" />
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Audience segments (TasteTwin clusters)" right={<Evidence type="modeled" />}>
            <HBar data={data.segments.map((s: any) => ({ name: s.name, n: s.n_viewers }))} x="n" y="name" height={240} />
          </Card>
          {data.journey && <Card title="Journey policies — satisfaction per journey" right={<Evidence type="simulated" />}>
            <VBars data={Object.entries(data.journey).map(([k, v]: any) => ({ policy: k.replace(/_/g, " "), sat: v.satisfaction_per_journey }))} x="policy" series={["sat"]} names={{ sat: "satisfaction" }} fmt={(v) => v.toFixed(1)} />
            <p className="text-xs text-neutral-500">Offline simulation under stated assumptions — see Journey simulator for sensitivity and confidence intervals.</p>
          </Card>}
        </div>
        <Card title="Deployed model"><p className="text-sm text-neutral-300">{data.model.version} · git {data.model.git_sha} · channels {data.model.channels.join(", ")} · {data.model.ranker.type}</p><p className="mt-1 text-xs text-neutral-500">{data.evidence_note}</p></Card>
      </>}
    </EnterpriseShell>
  );
}
