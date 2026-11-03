"use client";
import { useState } from "react";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { HBar } from "@/components/charts";
import { Card, ErrorBox, Evidence, Segmented, Spinner, Table } from "@/components/ui";
import { fmtNum, fmtPct, useApi } from "@/lib/hooks";

export default function Perf() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/reports/benchmark");
  const [seg, setSeg] = useState<"warm" | "cold">("warm");
  const rows = data ? Object.entries(data.results[seg]).map(([m, r]: any) => ({ m, n: r.n_users, ...r.metrics, ci: r.ci95["ndcg@10"], lat: r.latency_ms_per_user })).sort((a: any, b: any) => (b["ndcg@10"] ?? 0) - (a["ndcg@10"] ?? 0)) : [];
  return (
    <EnterpriseShell title="Recommendation performance" subtitle="Offline evaluation on a global temporal holdout. Hyper-parameters selected on validation; reported on test." actions={<Evidence type="observed" />}>
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <>
        <Card title="Protocol"><ul className="grid gap-1 text-sm text-neutral-300 md:grid-cols-2">{Object.entries(data.protocol).map(([k, v]) => <li key={k}><span className="text-neutral-500">{k.replace(/_/g, " ")}:</span> {String(v)}</li>)}</ul></Card>
        <div className="flex items-center gap-3"><Segmented value={seg} options={["warm", "cold"] as const} onChange={setSeg} labels={{ warm: "Warm users (≥5 events)", cold: "Cold start (1–4)" }} /></div>
        {rows.length > 0 && rows[0].n ? <>
          <Card title={`NDCG@10 — ${seg} users (n=${rows[0].n})`}><HBar data={rows.map((r: any) => ({ m: r.m, v: r["ndcg@10"] }))} x="v" y="m" height={330} fmt={(v) => v.toFixed(3)} colorBy={(d) => (d.m === "hybrid_ltr" ? "var(--series-2)" : "var(--series-1)")} /></Card>
          <Card title="All metrics"><Table rows={rows} cols={[
            { key: "m", label: "Model" }, { key: "ndcg@10", label: "NDCG@10", align: "right", fmt: (v, r) => <span title={`95% CI ${r.ci?.[0]?.toFixed(4)}–${r.ci?.[1]?.toFixed(4)}`}>{fmtNum(v, 4)}</span> },
            { key: "recall@20", label: "Recall@20", align: "right", fmt: (v) => fmtNum(v, 4) }, { key: "hit_rate@10", label: "HR@10", align: "right", fmt: (v) => fmtNum(v, 3) },
            { key: "map@10", label: "MAP@10", align: "right", fmt: (v) => fmtNum(v, 4) }, { key: "coverage@10", label: "Coverage", align: "right", fmt: (v) => fmtPct(v) },
            { key: "ild@10", label: "Diversity", align: "right", fmt: (v) => fmtNum(v, 3) }, { key: "novelty@10", label: "Novelty", align: "right", fmt: (v) => fmtNum(v, 2) },
            { key: "long_tail@10", label: "Long-tail", align: "right", fmt: (v) => fmtPct(v) }, { key: "franchise_dup@10", label: "Franchise dup", align: "right", fmt: (v) => fmtPct(v) },
            { key: "lat", label: "ms/user", align: "right", fmt: (v) => fmtNum(v, 2) },
          ]} /></Card></> : <Card><p className="text-sm text-neutral-400">No users in this segment for this dataset.</p></Card>}
        <Card title="Exploration levels: accuracy vs discovery trade-off" right={<span className="text-xs text-neutral-500">same ranker, different re-ranking</span>}>
          <Table rows={Object.entries(data.rerank_exploration_levels).map(([k, v]: any) => ({ k, ...v }))} cols={[
            { key: "k", label: "Level" }, { key: "ndcg@10", label: "NDCG@10", align: "right", fmt: (v) => fmtNum(v, 4) },
            { key: "ild@10", label: "Diversity", align: "right", fmt: (v) => fmtNum(v, 3) }, { key: "novelty@10", label: "Novelty", align: "right", fmt: (v) => fmtNum(v, 2) },
            { key: "coverage@10", label: "Coverage", align: "right", fmt: (v) => fmtPct(v) }, { key: "franchise_dup@10", label: "Franchise dup", align: "right", fmt: (v) => fmtPct(v) },
          ]} />
        </Card>
      </>}
    </EnterpriseShell>
  );
}
