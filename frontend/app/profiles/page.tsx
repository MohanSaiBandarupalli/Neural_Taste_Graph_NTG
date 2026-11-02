"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Brand } from "@/components/Brand";
import { ErrorBox, Spinner } from "@/components/ui";
import { activeProfile, api, Profile } from "@/lib/api";
import { useRequireAuth } from "@/lib/auth";
import { useApi } from "@/lib/hooks";

const COLORS = ["#e50914", "#f5c518", "#46d369", "#5b8def", "#d55181", "#9085e9"];

export default function Profiles() {
  const { ready } = useRequireAuth();
  const router = useRouter();
  const { data, error, loading, reload } = useApi<Profile[]>(ready ? "/v1/profiles" : null);
  const genres = useApi<{ genres: string[] }>(ready ? "/v1/catalog/genres" : null);
  const [adding, setAdding] = useState(false); const [manage, setManage] = useState(false);
  const [name, setName] = useState(""); const [color, setColor] = useState(COLORS[3]); const [fav, setFav] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { if (data && data.length === 0) setAdding(true); }, [data]);
  if (!ready || loading) return <Spinner />;
  const pick = (p: Profile) => { activeProfile.set(p.id); router.push("/browse"); };
  const create = async () => {
    setErr(null);
    try { const p = await api<Profile>("/v1/profiles", { method: "POST", json: { name, color, favorite_genres: fav } }); setAdding(false); setName(""); setFav([]); await reload(); pick(p); }
    catch (e: any) { setErr(e.message); }
  };
  return (
    <main className="grid min-h-screen place-items-center px-6">
      <div className="fade-in w-full max-w-4xl text-center">
        <div className="mb-10"><Brand /></div>
        {error && <ErrorBox error={error} onRetry={reload} />}
        {!adding ? <>
          <h1 className="font-display text-4xl md:text-5xl">Who&apos;s watching?</h1>
          <div className="mt-10 flex flex-wrap justify-center gap-8">
            {data?.map((p) => (
              <div key={p.id} className="group relative">
                <button onClick={() => pick(p)} className="flex flex-col items-center gap-3">
                  <span className="grid h-28 w-28 place-items-center rounded-lg font-display text-5xl ring-white transition group-hover:ring-4" style={{ background: p.color }}>{p.name[0]}</span>
                  <span className="text-neutral-300 group-hover:text-white">{p.name}</span>
                  <span className="text-[11px] text-neutral-500">{p.has_imported_history ? "demo history" : `${p.n_app_events} in-app events`}</span>
                </button>
                {manage && <button aria-label={`Delete ${p.name}`} className="absolute -right-2 -top-2 rounded-full bg-ink-700 p-1.5 hover:bg-red-700"
                  onClick={async () => { if (confirm(`Delete profile ${p.name} and its history?`)) { await api(`/v1/profiles/${p.id}`, { method: "DELETE" }); reload(); } }}><Trash2 className="h-4 w-4" /></button>}
              </div>))}
            {(data?.length ?? 0) < 5 && <button onClick={() => setAdding(true)} className="flex flex-col items-center gap-3 text-neutral-400 hover:text-white">
              <span className="grid h-28 w-28 place-items-center rounded-lg border-2 border-dashed border-ink-500"><Plus className="h-10 w-10" /></span>Add profile</button>}
          </div>
          <button className="btn-ghost mt-12" onClick={() => setManage(!manage)}>{manage ? "Done" : "Manage profiles"}</button>
        </> : (
          <div className="card mx-auto max-w-xl p-8 text-left">
            <h2 className="font-display text-3xl">New profile</h2>
            <label className="mt-6 block"><span className="label">Name</span><input className="input mt-1" value={name} onChange={(e) => setName(e.target.value)} maxLength={40} /></label>
            <div className="mt-4"><span className="label">Colour</span><div className="mt-2 flex gap-2">{COLORS.map((c) => <button key={c} aria-label={c} onClick={() => setColor(c)} className={`h-8 w-8 rounded ${color === c ? "ring-2 ring-white" : ""}`} style={{ background: c }} />)}</div></div>
            <div className="mt-5"><span className="label">Pick a few genres you enjoy (optional — helps the first recommendations)</span>
              <div className="mt-2 flex flex-wrap gap-2">{genres.data?.genres.map((g) => <button key={g} onClick={() => setFav(fav.includes(g) ? fav.filter((x) => x !== g) : fav.length < 5 ? [...fav, g] : fav)}
                className={`rounded-full border px-3 py-1 text-xs ${fav.includes(g) ? "border-accent bg-accent/20 text-white" : "border-ink-600 text-neutral-300"}`}>{g}</button>)}</div>
              <p className="mt-2 text-xs text-neutral-500">Stated preferences are stored as explicit signals, separately from anything we infer.</p></div>
            {err && <p className="mt-3 text-sm text-red-400">{err}</p>}
            <div className="mt-6 flex gap-3"><button className="btn-primary" disabled={!name.trim()} onClick={create}>Create</button>{(data?.length ?? 0) > 0 && <button className="btn-ghost" onClick={() => setAdding(false)}>Cancel</button>}</div>
          </div>)}
      </div>
    </main>
  );
}
