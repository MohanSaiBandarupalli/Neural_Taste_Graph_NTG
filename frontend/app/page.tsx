"use client";
import Link from "next/link";
import { Brand } from "@/components/Brand";
import { artStyle } from "@/components/Poster";
import { useAuth } from "@/lib/auth";

const FEATURES = [
  ["TasteTwin", "A living model of your taste — long-term loves, what's rising lately, and how sure it is."],
  ["Journeys, not just lists", "Recommendations planned across sessions so you don't get the same thing over and over."],
  ["Hidden interests", "Genres you'd likely enjoy but rarely watch — with the evidence for each suggestion."],
  ["Ask in plain words", "“Something exciting but not too dark.” We tell you what we understood — and what we can't do."],
];

export default function Landing() {
  const { user } = useAuth();
  return (
    <main className="relative min-h-screen overflow-hidden">
      <div className="absolute inset-0 grid grid-cols-6 gap-2 opacity-40 [mask-image:linear-gradient(to_bottom,black,transparent_85%)]" aria-hidden>
        {Array.from({ length: 36 }).map((_, i) => <div key={i} className="aspect-[2/3] rounded" style={artStyle({ item_id: i * 37 + 11, genres: [["Drama", "Sci-Fi", "Comedy", "Thriller", "Romance", "Horror"][i % 6]] })} />)}
      </div>
      <div className="absolute inset-0 bg-gradient-to-b from-black/70 via-black/80 to-ink-950" />
      <div className="relative z-10 mx-auto flex max-w-6xl flex-col px-6">
        <header className="flex items-center justify-between py-6"><Brand />
          <div className="flex gap-3">{user ? <Link className="btn-primary" href="/profiles">Open NTG</Link> : <><Link className="btn-ghost" href="/login">Sign in</Link><Link className="btn-primary" href="/signup">Get started</Link></>}</div>
        </header>
        <section className="fade-in mt-20 max-w-3xl">
          <p className="label text-accent-glow">Entertainment journey intelligence</p>
          <h1 className="mt-4 font-display text-5xl leading-[1.05] md:text-7xl">Taste isn&apos;t static.<br />Your recommendations shouldn&apos;t be either.</h1>
          <p className="mt-6 max-w-2xl text-lg text-neutral-300">NTG learns how your interests evolve, plans discovery across sessions instead of repeating itself, and explains every pick with real evidence.</p>
          <div className="mt-8 flex flex-wrap gap-3"><Link className="btn-primary !px-6 !py-3 text-base" href={user ? "/profiles" : "/signup"}>Start discovering</Link><Link className="btn-ghost !px-6 !py-3 text-base" href="/login?demo=analyst">Explore the enterprise demo</Link></div>
        </section>
        <section className="mt-24 grid gap-4 pb-24 md:grid-cols-4">
          {FEATURES.map(([t, d]) => <div key={t} className="card bg-ink-900/80 p-5 backdrop-blur"><h3 className="font-display text-xl">{t}</h3><p className="mt-2 text-sm text-neutral-400">{d}</p></div>)}
        </section>
        <p className="pb-10 text-xs text-neutral-500">Demo runs on a synthetic MovieLens-schema catalogue (generated titles). Real MovieLens data can be loaded locally. NTG is an independent project, not affiliated with any streaming service.</p>
      </div>
    </main>
  );
}
