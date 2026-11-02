"use client";
import Link from "next/link";
import type { Item } from "@/lib/api";

// Generated, deterministic cinematic artwork: no third-party poster images are used.
const GENRE_HUE: Record<string, number> = {
  Action: 8, Adventure: 30, Animation: 48, Children: 55, Comedy: 42, Crime: 220, Documentary: 160, Drama: 280,
  Fantasy: 300, "Film-Noir": 230, Horror: 350, IMAX: 195, Musical: 320, Mystery: 250, Romance: 335, "Sci-Fi": 190,
  Thriller: 210, War: 25, Western: 32,
};
function hash(n: number) { let x = n * 2654435761; x ^= x >>> 16; return Math.abs(x); }

export function artStyle(item: Pick<Item, "item_id" | "genres">): React.CSSProperties {
  const h1 = GENRE_HUE[item.genres?.[0]] ?? 210;
  const h2 = GENRE_HUE[item.genres?.[1]] ?? (h1 + 40) % 360;
  const r = hash(item.item_id);
  const ang = r % 360, x = 20 + (r % 60), y = 15 + ((r >> 5) % 50);
  return {
    backgroundImage: `radial-gradient(circle at ${x}% ${y}%, hsla(${h2},80%,60%,.55), transparent 55%),
      linear-gradient(${ang}deg, hsl(${h1},65%,22%) 0%, hsl(${h1},70%,10%) 55%, #050505 100%)`,
  };
}

export function Poster({ item, wide = false, onSelect }: { item: Item; wide?: boolean; onSelect?: (i: Item) => void }) {
  const body = (
    <div className={`poster-hover relative overflow-hidden rounded-md ${wide ? "aspect-video w-72" : "aspect-[2/3] w-40"} shrink-0`} style={artStyle(item)}>
      <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/10 to-transparent" />
      <div className="absolute left-2 top-2 rounded bg-black/50 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-neutral-200">{item.genres?.[0] ?? "Film"}</div>
      <div className="absolute bottom-0 p-2.5">
        <div className="font-display text-[15px] leading-tight text-white drop-shadow">{item.title}</div>
        <div className="mt-1 text-[11px] text-neutral-300">{item.year ?? ""}{item.avg_rating ? ` · ★ ${item.avg_rating.toFixed(1)}` : ""}</div>
      </div>
    </div>
  );
  if (onSelect) return <button onClick={() => onSelect(item)} className="text-left" aria-label={item.title}>{body}</button>;
  return <Link href={`/title/${item.item_id}`} aria-label={item.title}>{body}</Link>;
}

export function Row({ title, subtitle, items, wide }: { title: string; subtitle?: string; items: Item[]; wide?: boolean }) {
  if (!items?.length) return null;
  return (
    <section className="fade-in mb-9">
      <h2 className="mb-1 px-6 text-lg font-semibold text-neutral-100 md:px-12">{title}</h2>
      {subtitle && <p className="mb-2 max-w-3xl px-6 text-sm text-neutral-400 md:px-12">{subtitle}</p>}
      <div className="scroll-row flex gap-3 overflow-x-auto px-6 py-3 md:px-12">
        {items.map((it) => <Poster key={it.item_id} item={it} wide={wide} />)}
      </div>
    </section>
  );
}
