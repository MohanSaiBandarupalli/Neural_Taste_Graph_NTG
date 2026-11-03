"use client";
import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { ConsumerShell } from "@/components/ConsumerShell";
import { Row } from "@/components/Poster";
import { ErrorBox, Spinner } from "@/components/ui";
import { useApi } from "@/lib/hooks";

function Results() {
  const q = useSearchParams().get("q") ?? "";
  const { data, error, loading } = useApi<any>(q ? `/v1/catalog/search?q=${encodeURIComponent(q)}` : null);
  return (
    <main className="pt-24">
      <h1 className="px-6 font-display text-3xl md:px-12">Results for “{q}”</h1>
      {error && <ErrorBox error={error} />}
      {loading && <Spinner />}
      {data && <div className="mt-6">
        <Row title="Title matches" items={data.title_matches} />
        <Row title="Matches the meaning of your search" subtitle={`Semantic search: ${data.semantic_method}.`} items={data.semantic_matches} />
        {!data.title_matches.length && !data.semantic_matches.length && <p className="px-12 text-neutral-400">No matches.</p>}
      </div>}
    </main>
  );
}
export default function Page() { return <ConsumerShell><Suspense><Results /></Suspense></ConsumerShell>; }
