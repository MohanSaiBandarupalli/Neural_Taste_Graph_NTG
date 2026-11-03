"use client";
import { useEffect, useState } from "react";
import { Ban, ThumbsDown, ThumbsUp, Eye, Star } from "lucide-react";
import { ConsumerShell } from "@/components/ConsumerShell";
import { useProfileId } from "@/components/Nav";
import { artStyle, Row } from "@/components/Poster";
import { ExplainList } from "@/components/Explain";
import { ErrorBox, Spinner } from "@/components/ui";
import { api } from "@/lib/api";
import { useApi } from "@/lib/hooks";

function Title({ id }: { id: string }) {
  const pid = useProfileId();
  const { data, error, loading } = useApi<any>(`/v1/catalog/items/${id}`);
  const [why, setWhy] = useState<any>(null);
  const [msg, setMsg] = useState<string | null>(null); const [rating, setRating] = useState(0);
  useEffect(() => {
    if (!pid || !data) return;
    // grounded "why": ask the engine to score this title for the viewer via an anchor intent
    api(`/v1/profiles/${pid}/recommendations`, { method: "POST", json: { k: 40, level: "familiar" } })
      .then((r: any) => setWhy(r.items.find((c: any) => c.item_id === Number(id)) ?? null)).catch(() => {});
  }, [pid, data, id]);
  const send = async (kind: string, r?: number) => {
    if (!pid) return;
    try { await api(`/v1/profiles/${pid}/events`, { method: "POST", json: { item_id: Number(id), kind, rating: r } });
      setMsg({ watched: "Marked as watched — your TasteTwin has been updated.", like: "Thanks! We'll find more like this.", dislike: "Got it — fewer like this.", not_interested: "Hidden from your recommendations.", rated: `Rated ${r}★ — your TasteTwin has been updated.` }[kind] ?? "Saved");
    } catch (e: any) { setMsg(e.message); }
  };
  if (error) return <div className="pt-24"><ErrorBox error={error} /></div>;
  if (loading || !data) return <div className="pt-24"><Spinner /></div>;
  const it = data.item;
  return (
    <>
      <section className="relative min-h-[60vh] w-full" style={artStyle(it)}>
        <div className="absolute inset-0 bg-gradient-to-r from-black via-black/70 to-black/20" /><div className="absolute inset-0 bg-gradient-to-t from-ink-950 to-transparent" />
        <div className="relative z-10 max-w-3xl px-6 pb-12 pt-32 md:px-12">
          <h1 className="font-display text-5xl">{it.title}</h1>
          <p className="mt-2 text-neutral-300">{it.year} · {it.genres.join(" · ")} · {it.n_ratings.toLocaleString()} ratings{it.avg_rating ? ` · avg ★ ${it.avg_rating}` : ""}</p>
          {it.tags.length > 0 && <div className="mt-3 flex flex-wrap gap-2">{it.tags.map((t: string) => <span key={t} className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs">{t}</span>)}</div>}
          <div className="mt-6 flex flex-wrap gap-2">
            <button className="btn-primary" onClick={() => send("watched")}><Eye className="h-4 w-4" />I watched this</button>
            <button className="btn-ghost" onClick={() => send("like")}><ThumbsUp className="h-4 w-4" />Like</button>
            <button className="btn-ghost" onClick={() => send("dislike")}><ThumbsDown className="h-4 w-4" />Dislike</button>
            <button className="btn-ghost" onClick={() => send("not_interested")}><Ban className="h-4 w-4" />Not interested</button>
          </div>
          <div className="mt-3 flex items-center gap-1" aria-label="Rate">{[1, 2, 3, 4, 5].map((s) => <button key={s} aria-label={`${s} stars`} onClick={() => { setRating(s); send("rated", s); }}><Star className={`h-5 w-5 ${s <= rating ? "fill-amber-400 text-amber-400" : "text-neutral-500"}`} /></button>)}</div>
          {msg && <p role="status" className="mt-3 text-sm text-emerald-300">{msg}</p>}
          {why?.explanation && <div className="card mt-6 bg-black/50 p-4"><div className="label mb-2">Why NTG would recommend this to you</div><ExplainList ex={why.explanation} /></div>}
        </div>
      </section>
      <Row title="Viewers who watched this also watched" subtitle="Item-to-item collaborative similarity (cosine)." items={data.collaborative ?? []} />
      <Row title="Similar in genre, tags and themes" subtitle="Semantic similarity over genres, tags and title words." items={data.semantic ?? []} />
    </>
  );
}
export default function Page({ params }: { params: { id: string } }) { return <ConsumerShell><Title id={params.id} /></ConsumerShell>; }
