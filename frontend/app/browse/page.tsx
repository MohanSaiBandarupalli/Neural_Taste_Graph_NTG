"use client";
import Link from "next/link";
import { useState } from "react";
import { Play, Info } from "lucide-react";
import { ConsumerShell } from "@/components/ConsumerShell";
import { useProfileId } from "@/components/Nav";
import { artStyle, Row } from "@/components/Poster";
import { ExplainList } from "@/components/Explain";
import { ErrorBox, Segmented, Spinner } from "@/components/ui";
import { useApi } from "@/lib/hooks";

const LEVELS = ["familiar", "balanced", "exploratory", "unexpected"] as const;

function Home() {
  const pid = useProfileId();
  const [level, setLevel] = useState<(typeof LEVELS)[number]>("balanced");
  const { data, error, loading, reload } = useApi<any>(pid ? `/v1/profiles/${pid}/home?level=${level}` : null);
  if (error) return <div className="pt-24"><ErrorBox error={error} onRetry={reload} /></div>;
  if (!data || loading) return <div className="pt-24"><Spinner label="Personalising" /></div>;
  const hero = data.rows[0]?.items[0];
  return (
    <>
      {hero && (
        <section className="relative h-[70vh] min-h-[460px] w-full" style={artStyle(hero)}>
          <div className="absolute inset-0 bg-gradient-to-r from-black via-black/60 to-transparent" />
          <div className="absolute inset-0 bg-gradient-to-t from-ink-950 via-transparent to-transparent" />
          <div className="relative z-10 flex h-full max-w-2xl flex-col justify-end px-6 pb-24 md:px-12">
            <p className="label text-accent-glow">{data.cold_start ? "Popular right now" : "Top pick for you"}</p>
            <h1 className="mt-3 font-display text-5xl md:text-6xl">{hero.title}</h1>
            <p className="mt-2 text-sm text-neutral-300">{hero.year} · {hero.genres.join(" · ")}{hero.avg_rating ? ` · ★ ${hero.avg_rating}` : ""}</p>
            <div className="mt-4"><ExplainList ex={hero.explanation} compact /></div>
            <div className="mt-6 flex gap-3"><Link href={`/title/${hero.item_id}`} className="btn-primary !px-6"><Play className="h-4 w-4" />Details</Link>
              <Link href={`/title/${hero.item_id}`} className="btn-ghost !px-6"><Info className="h-4 w-4" />Why this?</Link></div>
          </div>
        </section>)}
      <div className="relative z-10 -mt-16 mb-6 flex flex-wrap items-center gap-3 px-6 md:px-12">
        <span className="text-sm text-neutral-300">Discovery mode</span>
        <Segmented value={level} options={LEVELS} onChange={setLevel} />
        <span className="text-xs text-neutral-500">{{ familiar: "More of what you love", balanced: "Relevance with variety", exploratory: "Wider genres, less mainstream", unexpected: "Far from your usual — still filtered for quality" }[level]}</span>
      </div>
      {data.rows.map((r: any) => <Row key={r.key} title={r.title} subtitle={r.subtitle} items={r.items} />)}
    </>
  );
}
export default function Page() { return <ConsumerShell><Home /></ConsumerShell>; }
