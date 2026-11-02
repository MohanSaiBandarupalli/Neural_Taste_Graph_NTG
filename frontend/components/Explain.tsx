"use client";
import { Info } from "lucide-react";
import type { Explanation } from "@/lib/api";

export function ExplainList({ ex, compact = false }: { ex?: Explanation; compact?: boolean }) {
  if (!ex) return null;
  return (
    <div className="space-y-2">
      <ul className="space-y-1.5">{ex.reasons.map((r, i) => <li key={i} className="flex gap-2 text-sm text-neutral-200"><span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />{r.text}</li>)}</ul>
      {!compact && <>
        {ex.ranker_drivers.length > 0 && <p className="text-xs text-neutral-400">Ranking drivers: {ex.ranker_drivers.map((d) => d.text).join(" · ")}</p>}
        <p className="text-xs text-neutral-500">Found by: {ex.retrieved_by.join(", ")}</p>
        <p className="flex items-center gap-1 text-[11px] text-neutral-500"><Info className="h-3 w-3" />{ex.grounding}</p>
      </>}
    </div>
  );
}
