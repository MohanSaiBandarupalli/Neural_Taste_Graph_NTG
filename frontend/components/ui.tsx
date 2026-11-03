"use client";
import clsx from "clsx";
import { AlertTriangle, Loader2 } from "lucide-react";

export function Spinner({ label = "Loading" }: { label?: string }) {
  return <div className="flex items-center gap-2 p-8 text-sm text-neutral-400"><Loader2 className="h-4 w-4 animate-spin" />{label}…</div>;
}
export function ErrorBox({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="m-4 flex items-start gap-3 rounded-lg border border-red-900/60 bg-red-950/40 p-4 text-sm text-red-200">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" /><div className="flex-1">{error}</div>
      {onRetry && <button className="btn-ghost !py-1" onClick={onRetry}>Retry</button>}
    </div>
  );
}
const EV: Record<string, string> = {
  observed: "border-emerald-700/60 text-emerald-300 bg-emerald-950/40",
  modeled: "border-sky-700/60 text-sky-300 bg-sky-950/40",
  simulated: "border-amber-700/60 text-amber-300 bg-amber-950/40",
  synthetic: "border-fuchsia-700/60 text-fuchsia-300 bg-fuchsia-950/40",
};
/** Every number in the product is labelled by how it was produced. */
export function Evidence({ type, className }: { type: "observed" | "modeled" | "simulated" | "synthetic"; className?: string }) {
  const tip = { observed: "Counted from data or measured on held-out data", modeled: "Output of a model; uncertain",
    simulated: "Computed from stated assumptions; not a measured effect", synthetic: "Generated demo dataset; not real viewers" }[type];
  return <span title={tip} className={clsx("inline-flex rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider", EV[type], className)}>{type}</span>;
}
export function Stat({ label, value, sub, evidence }: { label: string; value: React.ReactNode; sub?: React.ReactNode; evidence?: any }) {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between"><div className="label">{label}</div>{evidence && <Evidence type={evidence} />}</div>
      <div className="mt-2 font-display text-3xl text-neutral-50">{value}</div>
      {sub && <div className="mt-1 text-xs text-neutral-400">{sub}</div>}
    </div>
  );
}
export function Card({ title, right, children, className }: { title?: React.ReactNode; right?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={clsx("card p-5", className)}>
      {(title || right) && <div className="mb-4 flex items-center justify-between gap-3"><h3 className="text-sm font-semibold text-neutral-100">{title}</h3>{right}</div>}
      {children}
    </section>
  );
}
export function Table({ cols, rows }: { cols: { key: string; label: string; fmt?: (v: any, r: any) => React.ReactNode; align?: "right" }[]; rows: any[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead><tr className="border-b border-ink-700 text-left">{cols.map((c) => <th key={c.key} className={clsx("px-3 py-2 text-xs font-medium uppercase tracking-wider text-neutral-400", c.align === "right" && "text-right")}>{c.label}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i} className="border-b border-ink-800 hover:bg-ink-800/60">
            {cols.map((c) => <td key={c.key} className={clsx("px-3 py-2 text-neutral-200", c.align === "right" && "text-right tabular-nums")}>{c.fmt ? c.fmt(r[c.key], r) : String(r[c.key] ?? "—")}</td>)}
          </tr>))}
        </tbody>
      </table>
    </div>
  );
}
export function Segmented<T extends string>({ value, options, onChange, labels }: { value: T; options: readonly T[]; onChange: (v: T) => void; labels?: Record<string, string> }) {
  return (
    <div role="radiogroup" className="inline-flex rounded-lg border border-ink-600 bg-ink-900 p-1">
      {options.map((o) => (
        <button key={o} role="radio" aria-checked={value === o} onClick={() => onChange(o)}
          className={clsx("rounded-md px-3 py-1.5 text-xs font-semibold transition", !labels && "capitalize", value === o ? "bg-accent text-white" : "text-neutral-300 hover:text-white")}>
          {labels?.[o] ?? o}
        </button>))}
    </div>
  );
}
export function Bar({ value, max = 1, color = "var(--series-1)" }: { value: number; max?: number; color?: string }) {
  return <div className="h-2 w-full rounded-full bg-ink-700"><div className="h-2 rounded-full" style={{ width: `${Math.max(2, Math.min(100, (100 * value) / (max || 1)))}%`, background: color }} /></div>;
}
