"use client";
import { useState } from "react";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { Card, ErrorBox, Spinner, Table } from "@/components/ui";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";

export default function Data() {
  const { data, error, loading } = useApi<any>("/v1/enterprise/data");
  const [res, setRes] = useState<any>(null); const [err, setErr] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  const upload = async (f: File) => { setBusy(true); setErr(null); const fd = new FormData(); fd.append("file", f);
    try { setRes(await api("/v1/enterprise/data/validate", { method: "POST", body: fd })); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } };
  return (
    <EnterpriseShell title="Data management" subtitle="Dataset lineage, split protocol, contract checks, and validation of customer interaction files.">
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <>
        <Card title="Active dataset"><div className="grid gap-2 text-sm md:grid-cols-3">
          {[["dataset", data.meta.dataset], ["synthetic", String(data.meta.is_synthetic)], ["interactions", data.meta.n_interactions.toLocaleString()], ["users", data.users.toLocaleString()], ["titles", data.items.toLocaleString()],
            ["validation cutoff", data.meta.t_val_iso.slice(0, 10)], ["test cutoff", data.meta.t_test_iso.slice(0, 10)], ["train / val / test", `${data.meta.interactions_by_split.train} / ${data.meta.interactions_by_split.val} / ${data.meta.interactions_by_split.test}`], ["prepare time", `${data.meta.seconds}s`]]
            .map(([k, v]) => <div key={k}><span className="text-neutral-500">{k}: </span>{v}</div>)}</div></Card>
        <Card title="Contract checks"><Table rows={data.checks} cols={[{ key: "check", label: "Check" }, { key: "status", label: "Status", fmt: (v) => <span className={v === "pass" ? "text-emerald-300" : "text-red-300"}>{v}</span> }, { key: "detail", label: "Detail", fmt: (v) => v ?? "" }]} /></Card>
      </>}
      <Card title="Validate an interactions file (CSV: user_id/userId, item_id/movieId, rating, timestamp)">
        <input type="file" accept=".csv,text/csv" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} className="text-sm" />
        {busy && <Spinner label="Validating" />}{err && <ErrorBox error={err} />}
        {res && <div className="mt-4"><p className={`text-sm font-semibold ${res.valid ? "text-emerald-300" : "text-red-300"}`}>{res.valid ? "Valid" : "Invalid"} · {res.rows_checked.toLocaleString()} rows checked</p>
          <Table rows={res.checks} cols={[{ key: "check", label: "Check" }, { key: "status", label: "Status" }, { key: "detail", label: "Detail", fmt: (v) => (v == null ? "" : JSON.stringify(v)) }]} />
          <p className="mt-2 text-xs text-neutral-500">{res.note}</p></div>}
      </Card>
    </EnterpriseShell>
  );
}
