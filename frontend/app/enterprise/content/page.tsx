"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { VBars } from "@/components/charts";
import { Card, ErrorBox, Evidence, Spinner, Stat, Table } from "@/components/ui";
import { fmtNum, fmtPct, useApi } from "@/lib/hooks";

export default function Content() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/content");
  return (
    <EnterpriseShell title="Content intelligence" subtitle="Catalogue coverage, long-tail exposure and discovery gaps from interaction data." actions={<Evidence type="observed" />}>
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <>
        <div className="grid gap-4 md:grid-cols-4">
          <Stat label="Titles" value={data.n_titles.toLocaleString()} sub={`${data.titles_with_any_interaction.toLocaleString()} with any interaction`} />
          <Stat label="Head share of viewing" value={fmtPct(data.head_share_of_consumption)} sub="top 20% of titles" />
          <Stat label="Recommendation coverage" value={fmtPct(data.recommendation_catalog_coverage)} sub="of catalogue, sample of viewers" />
          <Stat label="Long-tail share in recs" value={fmtPct(data.recommendation_long_tail_share)} />
        </div>
        <Card title="Catalogue vs viewing vs recommendation exposure by genre">
          <VBars data={data.genres.map((g: any) => ({ g: g.genre, catalog: g.catalog_share, viewing: g.consumption_share, recs: g.recommendation_exposure_share }))} x="g" series={["catalog", "viewing", "recs"]} names={{ catalog: "catalogue share", viewing: "viewing share", recs: "recommendation share" }} fmt={(v) => `${Math.round(v * 100)}%`} height={320} />
        </Card>
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Under-discovered genres"><Table rows={data.under_discovered_genres} cols={[{ key: "genre", label: "Genre" }, { key: "catalog_share", label: "Catalogue", align: "right", fmt: (v) => fmtPct(v) }, { key: "consumption_share", label: "Viewing", align: "right", fmt: (v) => fmtPct(v) }, { key: "avg_rating", label: "Avg rating", align: "right", fmt: (v) => fmtNum(v, 2) }]} /></Card>
          <Card title="Hidden gems (well rated, outside the popularity head)"><Table rows={data.hidden_gems.slice(0, 12)} cols={[{ key: "title", label: "Title" }, { key: "genres", label: "Genres", fmt: (v) => v.join(", ") }, { key: "avg_rating", label: "Avg", align: "right", fmt: (v) => fmtNum(v, 2) }, { key: "n_ratings", label: "Ratings", align: "right" }]} /></Card>
        </div>
        <p className="text-xs text-neutral-500">{data.limitations}</p>
      </>}
    </EnterpriseShell>
  );
}
