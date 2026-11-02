"use client";
import { ConsumerShell } from "@/components/ConsumerShell";
import { useProfileId } from "@/components/Nav";
import { Poster } from "@/components/Poster";
import { ErrorBox, Evidence, Spinner } from "@/components/ui";
import { useApi } from "@/lib/hooks";

function Journey() {
  const pid = useProfileId();
  const { data, error, loading, reload } = useApi<any>(pid ? `/v1/profiles/${pid}/journey?sessions=6` : null);
  return (
    <main className="mx-auto max-w-7xl px-6 pt-24">
      <div className="flex items-end justify-between gap-3"><div>
        <h1 className="font-display text-4xl">Your discovery journey</h1>
        <p className="mt-1 max-w-3xl text-neutral-400">Instead of one list, NTG plans the next few sessions: it rotates themes to avoid saturation, keeps what you love in the mix, and adapts as you react.</p></div>
        <Evidence type="modeled" /></div>
      {error && <ErrorBox error={error} onRetry={reload} />}
      {loading && <Spinner label="Planning" />}
      {data && <>
        <p className="mt-3 text-xs text-neutral-500">{data.assumption}</p>
        <ol className="mt-8 space-y-8 border-l border-ink-600 pl-6">
          {data.sessions.map((s: any) => <li key={s.session} className="fade-in relative">
            <span className="absolute -left-[33px] grid h-5 w-5 place-items-center rounded-full bg-accent text-[10px] font-bold">{s.session}</span>
            <h2 className="text-lg font-semibold">Session {s.session}{s.theme ? <span className="text-neutral-400"> · leading with {s.theme}</span> : null}</h2>
            <div className="scroll-row mt-3 flex gap-3 overflow-x-auto pb-2">{s.slate.map((it: any) => <div key={it.item_id} className={it.item_id === s.assumed_pick_item_id ? "rounded-md ring-2 ring-accent" : ""}><Poster item={it} /></div>)}</div>
          </li>)}
        </ol>
        {data.sessions.length === 0 && <p className="mt-6 text-neutral-400">Watch or like a few titles to get a journey.</p>}
      </>}
    </main>
  );
}
export default function Page() { return <ConsumerShell><Journey /></ConsumerShell>; }
