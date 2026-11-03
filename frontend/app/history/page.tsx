"use client";
import Link from "next/link";
import { ConsumerShell } from "@/components/ConsumerShell";
import { useProfileId } from "@/components/Nav";
import { ErrorBox, Spinner, Table } from "@/components/ui";
import { useApi } from "@/lib/hooks";

function History() {
  const pid = useProfileId();
  const { data, error, loading } = useApi<any>(pid ? `/v1/profiles/${pid}/events` : null);
  return (
    <main className="mx-auto max-w-6xl px-6 pt-24">
      <h1 className="font-display text-4xl">Viewing history &amp; feedback</h1>
      <p className="mt-1 text-neutral-400">Everything NTG uses to personalise this profile. Demo personas include an imported (pseudonymous) history.</p>
      {error && <ErrorBox error={error} />}{loading && <Spinner />}
      {data && <div className="card mt-6 p-2"><Table rows={data.events} cols={[
        { key: "title", label: "Title", fmt: (v, r) => <Link className="hover:underline" href={`/title/${r.item_id}`}>{v}</Link> },
        { key: "genres", label: "Genres", fmt: (v) => (v ?? []).join(", ") },
        { key: "kind", label: "Signal" }, { key: "rating", label: "Rating", align: "right", fmt: (v) => (v == null ? "—" : `${v}★`) },
        { key: "ts", label: "When", fmt: (v) => new Date(v * 1000).toLocaleDateString() }, { key: "source", label: "Source" },
      ]} />{data.events.length === 0 && <p className="p-4 text-neutral-400">No activity yet.</p>}</div>}
    </main>
  );
}
export default function Page() { return <ConsumerShell><History /></ConsumerShell>; }
