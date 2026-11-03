"use client";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { HBar, Lines } from "@/components/charts";
import { Card, ErrorBox, Evidence, Spinner, Table } from "@/components/ui";
import { fmtNum, fmtPct, useApi } from "@/lib/hooks";

export default function Risk() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/risk");
  const ev = data?.evaluation;
  return (
    <EnterpriseShell title="Engagement-risk analytics" subtitle="A disengagement PROXY (no activity in the next 90 days). The dataset has no subscription data, so this is not subscriber churn.">
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {ev && <>
        <Card title="Out-of-time evaluation" right={<Evidence type="observed" />}>
          <p className="mb-3 text-sm text-neutral-400">Trained at {ev.train_cutoff_iso.slice(0, 10)}, tested at {ev.test_cutoff_iso.slice(0, 10)} · {ev.n_users} recently active viewers · base rate {fmtPct(ev.positive_rate)}.</p>
          <Table rows={["gbm_calibrated", "logistic_regression", "recency_only_baseline"].map((k) => ({ k, ...ev[k] }))} cols={[
            { key: "k", label: "Model", fmt: (v) => v.replace(/_/g, " ") }, { key: "auc", label: "ROC-AUC", align: "right", fmt: (v) => fmtNum(v, 3) },
            { key: "pr_auc", label: "PR-AUC", align: "right", fmt: (v) => fmtNum(v, 3) }, { key: "brier", label: "Brier", align: "right", fmt: (v) => fmtNum(v, 3) },
            { key: "ece", label: "ECE", align: "right", fmt: (v) => fmtNum(v, 3) }, { key: "lift_top_decile", label: "Top-decile lift", align: "right", fmt: (v) => `${fmtNum(v, 2)}×` },
          ]} />
        </Card>
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Calibration (calibrated GBM)"><Lines data={ev.calibration_curve.map((c: any) => ({ bin: c.bin, predicted: c.predicted, observed: c.observed }))} x="bin" series={["predicted", "observed"]} fmt={(v) => `${Math.round(v * 100)}%`} /></Card>
          <Card title="AUC by tenure segment"><HBar data={Object.entries(ev.by_tenure_segment).map(([k, v]: any) => ({ k: `${k} (n=${v.n})`, v: v.auc }))} x="v" y="k" fmt={(v) => v.toFixed(2)} height={200} /></Card>
        </div>
        <Card title="Highest-risk recently active viewers (pseudonymous)" right={<Evidence type="modeled" />}>
          <Table rows={data.at_risk} cols={[{ key: "viewer_ref", label: "Viewer" }, { key: "segment", label: "Segment" }, { key: "risk", label: "Risk", align: "right", fmt: (v) => fmtPct(v) },
            { key: "drivers", label: "Main drivers", fmt: (v) => v.map((d: any) => d.text).join(" · ") }]} />
        </Card>
      </>}
    </EnterpriseShell>
  );
}
