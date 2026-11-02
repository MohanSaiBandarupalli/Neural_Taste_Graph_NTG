"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { HBar, VBars } from "@/components/charts";
import { Card, ErrorBox, Evidence, Spinner, Stat, Table } from "@/components/ui";
import { fmtNum, fmtPct, useApi } from "@/lib/hooks";

export default function TTA() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/tastetwin");
  const pop = data?.population, ev = data?.evaluation;
  return (
    <EnterpriseShell title="TasteTwin analytics" subtitle="How viewer taste is distributed and changing — and how well TasteTwin forecasts the next period on held-out data.">
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {pop && <>
        <div className="grid gap-4 md:grid-cols-4">
          <Stat label="Viewers sampled" value={pop.n_sampled} evidence="observed" />
          <Stat label="Mean exploration" value={pop.mean_exploration.toFixed(2)} evidence="modeled" />
          <Stat label="Mean stability" value={pop.mean_stability.toFixed(2)} evidence="modeled" />
          <Stat label="With an emerging interest" value={fmtPct(pop.share_with_emerging_interest)} evidence="modeled" />
        </div>
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Distribution of exploration & stability"><VBars data={pop.exploration_hist.map((v: number, i: number) => ({ bin: `${(i / 10).toFixed(1)}`, exploration: v, stability: pop.stability_hist[i] }))} x="bin" series={["exploration", "stability"]} height={260} /></Card>
          <Card title="Emerging interests (viewers with rising share)"><HBar data={Object.entries(pop.emerging_genre_counts).slice(0, 10).map(([g, n]) => ({ g, n }))} x="n" y="g" height={260} /></Card>
        </div>
      </>}
      {ev && <Card title="Future-preference forecast (temporal holdout)" right={<Evidence type="observed" />}>
        <p className="mb-3 text-sm text-neutral-400">{ev.protocol}. Selected on validation: half-lives {ev.selected_on_val.half_life_long}d / {ev.selected_on_val.half_life_short}d, blend {ev.selected_on_val.blend}. Test users: {ev.n_users_test}.</p>
        <Table rows={Object.entries(ev.test).map(([k, v]: any) => ({ k, ...v }))} cols={[
          { key: "k", label: "Predictor", fmt: (v) => v.replace(/_/g, " ") },
          { key: "js_divergence", label: "JS divergence ↓", align: "right", fmt: (v, r) => `${fmtNum(v, 4)} [${fmtNum(r.js_ci95[0], 4)}, ${fmtNum(r.js_ci95[1], 4)}]` },
          { key: "top3_genre_recall", label: "Top-3 genre recall ↑", align: "right", fmt: (v, r) => `${fmtNum(v, 3)} [${fmtNum(r.top3_ci95[0], 3)}, ${fmtNum(r.top3_ci95[1], 3)}]` },
        ]} />
        <p className="mt-3 text-sm">JS reduction vs static profile: <b>{fmtNum(ev.js_reduction_vs_static, 4)}</b> (95% CI {fmtNum(ev.js_reduction_vs_static_ci95[0], 4)} – {fmtNum(ev.js_reduction_vs_static_ci95[1], 4)}).</p>
        {ev.drift_detection_synthetic && <p className="mt-2 text-sm text-neutral-400">Drift detection vs planted ground truth (synthetic only): AUC {fmtNum(ev.drift_detection_synthetic.auc_instability, 3)} ({ev.drift_detection_synthetic.n_drifting} drifting of {ev.drift_detection_synthetic.n_users}).</p>}
      </Card>}
    </EnterpriseShell>
  );
}
