"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { Card, ErrorBox, Evidence, Spinner, Stat, Table } from "@/components/ui";
import { fmtNum, fmtPct, useApi } from "@/lib/hooks";

export default function Monitoring() {
  const { data, error, loading, reload } = useApi<any>("/v1/enterprise/monitoring");
  const perf = useApi<any>("/v1/enterprise/reports/performance");
  return (
    <EnterpriseShell title="Model & system monitoring" subtitle="Live API health from this process, data drift, and the benchmark from the last build." actions={<button className="btn-ghost" onClick={reload}>Refresh</button>}>
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <>
        <div className="grid gap-4 md:grid-cols-4">
          <Stat label="Requests (last hour)" value={data.api.requests_last_hour} evidence="observed" />
          <Stat label="API p95 latency" value={data.api.p95_ms ? `${data.api.p95_ms.toFixed(0)} ms` : "—"} sub={`p50 ${data.api.p50_ms?.toFixed(0) ?? "—"} ms`} evidence="observed" />
          <Stat label="Error rate" value={fmtPct(data.api.error_rate)} evidence="observed" />
          <Stat label="Model age" value={`${data.model.age_hours} h`} sub={data.model.version} />
        </div>
        {data.drift && <Card title="Data drift" right={<span className={`text-sm font-semibold ${data.drift.status === "stable" ? "text-emerald-300" : data.drift.status === "moderate" ? "text-amber-300" : "text-red-300"}`}>● {data.drift.status}</span>}>
          <p className="text-sm">Genre-mix PSI {fmtNum(data.drift.genre_mix_psi, 4)} · mean rating {fmtNum(data.drift.mean_rating_recent, 2)} (recent) vs {fmtNum(data.drift.mean_rating_prior, 2)} (prior) · {data.drift.events_recent_90d.toLocaleString()} recent events.</p>
          <p className="mt-1 text-xs text-neutral-500">{data.drift.method}. Thresholds: &lt;0.1 stable, &lt;0.25 moderate.</p></Card>}
        <Card title="API latency by route group"><Table rows={Object.entries(data.api.by_group).map(([k, v]: any) => ({ k, ...v }))} cols={[{ key: "k", label: "Group" }, { key: "n", label: "Requests", align: "right" }, { key: "p95_ms", label: "p95 ms", align: "right", fmt: (v) => v.toFixed(1) }]} /></Card>
        <Card title="Live personalisation"><p className="text-sm">{data.live.viewers_with_live_events} viewers have {data.live.live_events} in-app events applied to TasteTwin without retraining.</p></Card>
      </>}
      {perf.data && <Card title="Last build benchmark" right={<Evidence type="observed" />}><p className="text-sm text-neutral-300">Single request p50 {perf.data.single_request_latency_ms.p50.toFixed(1)} ms / p95 {perf.data.single_request_latency_ms.p95.toFixed(1)} ms · with explanations p50 {perf.data.single_request_with_explanations_ms.p50.toFixed(1)} ms · batch {perf.data.batch_throughput_users_per_s} users/s · bundle {perf.data.bundle_size_mb} MB loads in {perf.data.bundle_load_seconds}s · peak RSS {perf.data.peak_rss_mb} MB.</p><p className="mt-1 text-xs text-neutral-500">{perf.data.note}</p></Card>}
    </EnterpriseShell>
  );
}
