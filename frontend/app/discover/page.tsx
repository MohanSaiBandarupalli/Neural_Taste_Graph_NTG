"use client";
import { useState } from "react";
import { Sparkles } from "lucide-react";
import { ConsumerShell } from "@/components/ConsumerShell";
import { useProfileId } from "@/components/Nav";
import { Poster } from "@/components/Poster";
import { ExplainList } from "@/components/Explain";
import { ErrorBox, Evidence, Segmented, Spinner } from "@/components/ui";
import { api } from "@/lib/api";

const EXAMPLES = ["I want something exciting but not too long.", "Find a mystery series that is not too dark.",
  "Recommend something completely different from what I usually watch.", "A funny classic from the 80s", "Feel-good family movie, no horror"];
const LEVELS = ["familiar", "balanced", "exploratory", "unexpected"] as const;

function Discover() {
  const pid = useProfileId();
  const [q, setQ] = useState(""); const [level, setLevel] = useState<(typeof LEVELS)[number]>("balanced");
  const [res, setRes] = useState<any>(null); const [err, setErr] = useState<string | null>(null); const [busy, setBusy] = useState(false);
  const run = async (text = q) => {
    if (!pid) return; setBusy(true); setErr(null);
    try { setRes(await api(`/v1/profiles/${pid}/recommendations`, { method: "POST", json: { k: 18, level, intent: text || null } })); }
    catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const it = res?.intent;
  return (
    <main className="mx-auto max-w-6xl px-6 pt-24">
      <h1 className="font-display text-4xl">Tell us what you&apos;re in the mood for</h1>
      <p className="mt-2 text-neutral-400">Plain language works. We show exactly what we understood, what we inferred from your history, and what we can&apos;t do.</p>
      <form onSubmit={(e) => { e.preventDefault(); run(); }} className="mt-6 flex gap-3">
        <input className="input !py-3 text-base" value={q} maxLength={500} onChange={(e) => setQ(e.target.value)} placeholder="e.g. something smart and suspenseful, not horror" />
        <button className="btn-primary !px-6" disabled={busy}><Sparkles className="h-4 w-4" />Discover</button>
      </form>
      <div className="mt-3 flex flex-wrap gap-2">{EXAMPLES.map((e) => <button key={e} onClick={() => { setQ(e); run(e); }} className="rounded-full border border-ink-600 px-3 py-1 text-xs text-neutral-300 hover:border-neutral-400">{e}</button>)}</div>
      <div className="mt-4 flex items-center gap-3"><span className="text-sm text-neutral-400">Exploration</span><Segmented value={level} options={LEVELS} onChange={setLevel} /></div>
      {err && <ErrorBox error={err} />}
      {busy && <Spinner label="Thinking" />}
      {res && !busy && <div className="fade-in mt-8 space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          <div className="card p-4"><div className="label mb-2">You asked for (explicit)</div>
            {it ? <ul className="space-y-1 text-sm">
              {it.boost_genres.length > 0 && <li>Genres: <b>{it.boost_genres.join(", ")}</b></li>}
              {it.avoid_genres.length > 0 && <li>Avoid: <b>{it.avoid_genres.join(", ")}</b></li>}
              {it.tags.length > 0 && <li>Mood: {it.tags.join(", ")}</li>}
              {(it.year_min || it.year_max) && <li>Years: {it.year_min ?? "…"}–{it.year_max ?? "…"}</li>}
              {it.anchor_title && <li>Similar to: “{it.anchor_title}”</li>}
              {it.exploration && <li>Exploration: {it.exploration}</li>}
              {Object.keys(it.explicit).length === 0 && <li className="text-neutral-400">No specific constraints recognised.</li>}
            </ul> : <p className="text-sm text-neutral-400">No request — showing your personal picks.</p>}
            {res.intent_applied?.length > 0 && <p className="mt-2 text-xs text-neutral-500">Applied: {res.intent_applied.join(" · ")}</p>}</div>
          <div className="card p-4"><div className="mb-2 flex items-center gap-2"><span className="label">We inferred</span><Evidence type="modeled" /></div>
            {res.inferred ? <p className="text-sm">Your strongest genres: <b>{res.inferred.top_genres.join(", ")}</b><br /><span className="text-xs text-neutral-500">{res.inferred.source}</span></p> : <p className="text-sm text-neutral-400">No history yet — using popularity.</p>}</div>
          <div className="card p-4"><div className="label mb-2">We can&apos;t do</div>
            {it?.unsupported?.length ? <ul className="space-y-1 text-sm text-amber-200">{it.unsupported.map((u: any) => <li key={u.constraint}>“{u.phrase}”: {u.reason}</li>)}</ul> : <p className="text-sm text-neutral-400">Everything you asked for is supported.</p>}</div>
        </div>
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {res.items.map((c: any) => <div key={c.item_id} className="card flex gap-4 p-3"><Poster item={c} /><div className="min-w-0 py-1"><div className="font-display text-lg leading-tight">{c.title}</div><div className="mb-2 text-xs text-neutral-400">{c.genres.join(" · ")}</div><ExplainList ex={c.explanation} compact /></div></div>)}
        </div>
        {res.items.length === 0 && <p className="text-neutral-400">Nothing in the catalogue satisfies every constraint. Try loosening one.</p>}
      </div>}
    </main>
  );
}
export default function Page() { return <ConsumerShell><Discover /></ConsumerShell>; }
