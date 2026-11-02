"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { VBars } from "@/components/charts";
import { Card, ErrorBox, Evidence, Spinner, Table } from "@/components/ui";
import { fmtPct, useApi } from "@/lib/hooks";

export default function Audience() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/audience");
  const risk = Object.fromEntries((data?.risk_by_segment ?? []).map((r: any) => [r.segment_id, r.mean_risk]));
  const fat = Object.fromEntries((data?.fatigue_by_segment ?? []).map((r: any) => [r.segment_id, r.mean_fatigue_risk]));
  return (
    <EnterpriseShell title="Audience intelligence" subtitle="Segments are k-means clusters over TasteTwin long-term genre shares and exploration willingness.">
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <>
        <Card title="Risk and fatigue by segment" right={<Evidence type="modeled" />}>
          <VBars data={data.segments.map((s: any) => ({ seg: s.name.replace(" fans", ""), risk: risk[s.segment_id] ?? 0, fatigue: fat[s.segment_id] ?? 0 }))} x="seg" series={["risk", "fatigue"]} names={{ risk: "engagement risk", fatigue: "fatigue risk" }} fmt={(v) => `${Math.round(v * 100)}%`} height={300} />
        </Card>
        <Card title="Segment profiles"><Table rows={data.segments} cols={[
          { key: "name", label: "Segment" }, { key: "n_viewers", label: "Viewers", align: "right" },
          { key: "top_genres", label: "Top genres", fmt: (v) => v.slice(0, 3).map((g: any) => `${g.genre} ${fmtPct(g.share, 0)}`).join(" · ") },
          { key: "avg_exploration", label: "Exploration", align: "right", fmt: (v) => v.toFixed(2) },
          { key: "avg_stability", label: "Stability", align: "right", fmt: (v) => v.toFixed(2) },
          { key: "avg_events", label: "Avg events", align: "right", fmt: (v) => v.toFixed(0) },
        ]} /></Card>
      </>}
    </EnterpriseShell>
  );
}
