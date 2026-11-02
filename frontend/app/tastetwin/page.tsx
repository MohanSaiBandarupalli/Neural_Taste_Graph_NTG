"use client";
import { ConsumerShell } from "@/components/ConsumerShell";
import { useProfileId } from "@/components/Nav";
import { Poster } from "@/components/Poster";
import { HBar, StackedShare, SERIES } from "@/components/charts";
import { Bar, Card, ErrorBox, Evidence, Spinner, Stat } from "@/components/ui";
import { fmtPct, useApi } from "@/lib/hooks";

function TT() {
  const pid = useProfileId();
  const tt = useApi<any>(pid ? `/v1/profiles/${pid}/tastetwin` : null);
  const hid = useApi<any>(pid ? `/v1/profiles/${pid}/hidden-interests` : null);
  if (tt.error) return <div className="pt-24"><ErrorBox error={tt.error} onRetry={tt.reload} /></div>;
  if (!tt.data) return <div className="pt-24"><Spinner label="Building your TasteTwin" /></div>;
  const p = tt.data;
  const keys: string[] = [];
  p.evolution.forEach((pt: any) => Object.keys(pt.shares).forEach((k) => { if (!keys.includes(k)) keys.push(k); }));
  const evo = p.evolution.map((pt: any) => ({ date: pt.date, ...Object.fromEntries(keys.map((k) => [k, pt.shares[k] ?? 0])) }));
  const fat = p.fatigue;
  return (
    <main className="mx-auto max-w-7xl px-6 pt-24">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><h1 className="font-display text-4xl">Your TasteTwin</h1><p className="mt-1 text-neutral-400">{p.provenance}. Updated with every rating and like.</p></div>
        <Evidence type="modeled" />
      </div>
      {p.n_events === 0 ? <Card className="mt-6"><p className="text-neutral-300">No viewing history yet. Rate or like a few titles and your TasteTwin appears here instantly.</p></Card> : <>
      <div className="mt-6 grid gap-4 md:grid-cols-4">
        <Stat label="Confidence" value={fmtPct(p.confidence, 0)} sub={`${p.confidence_label} · effective sample ${p.effective_sample_size.toFixed(1)}`} />
        <Stat label="Taste stability" value={fmtPct(p.stability, 0)} sub="similarity of recent vs long-term taste" />
        <Stat label="Exploration willingness" value={fmtPct(p.exploration_willingness, 0)} sub="genre variety + long-tail viewing, recent" />
        <Stat label="Interactions" value={p.n_events} sub={`${p.live_events_applied} from this app`} />
      </div>
      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card title="Long-term taste" right={<span className="text-xs text-neutral-500">± posterior uncertainty</span>}>
          <ul className="space-y-3">{p.long_term.map((g: any, i: number) => <li key={g.genre}><div className="mb-1 flex justify-between text-sm"><span>{g.genre}</span><span className="tabular-nums text-neutral-400">{fmtPct(g.share, 0)} ± {fmtPct(g.uncertainty, 0)}</span></div><Bar value={g.share} max={p.long_term[0].share} color={SERIES[0]} /></li>)}</ul>
        </Card>
        <Card title="Recently">
          <ul className="space-y-3">{p.short_term.map((g: any) => <li key={g.genre}><div className="mb-1 flex justify-between text-sm"><span>{g.genre}</span><span className="tabular-nums text-neutral-400">{fmtPct(g.share, 0)}</span></div><Bar value={g.share} max={p.short_term[0].share} color={SERIES[1]} /></li>)}</ul>
        </Card>
        <Card title="What's changing">
          <div className="space-y-4 text-sm">
            <div><div className="label mb-1">Emerging</div>{p.emerging.length ? p.emerging.map((e: any) => <div key={e.genre} className="flex justify-between"><span>↑ {e.genre}</span><span className="text-emerald-300">+{fmtPct(e.momentum, 0)}</span></div>) : <p className="text-neutral-500">Nothing clearly rising.</p>}</div>
            <div><div className="label mb-1">Declining</div>{p.declining.length ? p.declining.map((e: any) => <div key={e.genre} className="flex justify-between"><span>↓ {e.genre}</span><span className="text-red-300">{fmtPct(e.momentum, 0)}</span></div>) : <p className="text-neutral-500">Nothing clearly fading.</p>}</div>
            {p.themes.length > 0 && <div><div className="label mb-1">Themes (from tags)</div><p>{p.themes.map((t: any) => t.theme).join(" · ")}</p></div>}
            <p className="text-xs text-neutral-500">Not modelled: {Object.values(p.unavailable_dimensions).join("; ")}.</p>
          </div>
        </Card>
      </div>
      {evo.length > 1 && <Card className="mt-6" title="Taste evolution" right={<span className="text-xs text-neutral-500">genre mix over time (top 7)</span>}><StackedShare data={evo} keys={keys} /></Card>}
      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card title="Recommendation fatigue" right={<Evidence type="modeled" />}>
          {fat.available ? <>
            <div className="flex items-baseline gap-3"><span className="font-display text-4xl">{fmtPct(fat.risk, 0)}</span><span className={`text-sm font-semibold ${fat.label === "elevated" ? "text-red-300" : fat.label === "moderate" ? "text-amber-300" : "text-emerald-300"}`}>{fat.label}</span></div>
            <p className="mt-1 text-xs text-neutral-500">{fat.definition}</p>
            <div className="mt-4"><HBar data={fat.drivers.map((d: any) => ({ name: d.text, v: d.contribution }))} x="v" y="name" height={170} fmt={(v) => v.toFixed(2)} colorBy={(d) => (d.v >= 0 ? "var(--series-8)" : "var(--series-3)")} /></div>
            <p className="text-xs text-neutral-500">Bars: contribution of each signal to the score (right = raises risk).</p>
          </> : <p className="text-sm text-neutral-400">{fat.reason}</p>}
        </Card>
        <Card title="Hidden interests" right={<Evidence type="modeled" />}>
          {hid.loading && <Spinner />}
          {hid.data?.hidden_interests?.length === 0 && <p className="text-sm text-neutral-400">No strong hidden interests detected yet.</p>}
          <div className="space-y-5">{hid.data?.hidden_interests?.map((h: any) => <div key={h.genre}>
            <div className="font-semibold">{h.genre}</div><p className="mb-2 text-sm text-neutral-400">{h.explanation}</p>
            <div className="scroll-row flex gap-2 overflow-x-auto">{h.examples.slice(0, 5).map((x: any) => <Poster key={x.item_id} item={x} />)}</div></div>)}</div>
        </Card>
      </div></>}
    </main>
  );
}
export default function Page() { return <ConsumerShell><TT /></ConsumerShell>; }
