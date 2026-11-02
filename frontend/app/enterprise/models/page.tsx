"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { HBar } from "@/components/charts";
import { Card, ErrorBox, Spinner, Table } from "@/components/ui";
import { fmtNum, useApi } from "@/lib/hooks";

export default function Models() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/models");
  const imp = data ? Object.entries(data.ranker_feature_importance).slice(0, 15).map(([f, v]) => ({ f, v })) : [];
  return (
    <EnterpriseShell title="Model laboratory" subtitle="Versioned artifacts, hyper-parameter search traces and ranker feature importance.">
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <>
        <Card title="Deployed version"><div className="grid gap-2 text-sm md:grid-cols-2">
          {["version", "created_at", "git_sha", "dataset", "python", "bundle_sha256"].map((k) => <div key={k}><span className="text-neutral-500">{k}: </span><span className="break-all font-mono text-xs">{String(data.manifest[k])}</span></div>)}
        </div><p className="mt-3 text-xs text-neutral-500">Available versions: {data.available_versions.join(", ")}</p></Card>
        <Card title="Ranker feature importance (gain)"><HBar data={imp} x="v" y="f" height={420} fmt={(v) => v.toFixed(0)} /></Card>
        {data.tuning && <Card title={`Hyper-parameter search (${data.tuning.selection_metric})`}>
          <Table rows={Object.entries(data.tuning.trace).flatMap(([m, rs]: any) => rs.map((r: any) => ({ m, p: JSON.stringify(r.params), v: r["ndcg@10"], best: JSON.stringify(data.tuning.selected[m]) === JSON.stringify(r.params) })))} cols={[
            { key: "m", label: "Model" }, { key: "p", label: "Params", fmt: (v) => <code className="text-xs">{v}</code> },
            { key: "v", label: "Val NDCG@10", align: "right", fmt: (v) => fmtNum(v, 4) }, { key: "best", label: "Selected", fmt: (v) => (v ? "✓" : "") },
          ]} /></Card>}
      </>}
    </EnterpriseShell>
  );
}
