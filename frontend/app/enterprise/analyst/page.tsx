"use client";
import { useState } from "react";
import { Bot, Send, ShieldCheck } from "lucide-react";
import { EnterpriseShell } from "@/components/EnterpriseShell";
import { Evidence } from "@/components/ui";
import { api } from "@/lib/api";

const SUGGEST = ["Which viewer segments have the highest recommendation fatigue?", "Which recommendation strategy has the strongest offline results?", "Which content groups are under-discovered?",
  "What happens if exploration increases?", "Which model version is currently deployed?", "What are the latest model evaluation results?", "How accurate is TasteTwin?", "What is our engagement risk?"];

export default function Analyst() {
  const [q, setQ] = useState(""); const [log, setLog] = useState<any[]>([]); const [busy, setBusy] = useState(false);
  const ask = async (text = q) => { if (!text.trim()) return; setBusy(true); setQ("");
    try { const a = await api("/v1/enterprise/analyst", { method: "POST", json: { question: text } }); setLog((l) => [...l, a]); }
    catch (e: any) { setLog((l) => [...l, { question: text, answer: e.message, facts: [], refused: true }]); } finally { setBusy(false); } };
  return (
    <EnterpriseShell title="AI Business Analyst" subtitle="Answers are assembled only from the platform's own evaluation reports, model manifest and live aggregates — every fact carries its source and evidence type.">
      <div className="flex items-center gap-2 text-xs text-neutral-400"><ShieldCheck className="h-4 w-4 text-emerald-400" />Grounded retrieval · no external data · instructions inside questions are not followed · no individual-level data disclosed</div>
      <div className="flex flex-wrap gap-2">{SUGGEST.map((s) => <button key={s} onClick={() => ask(s)} className="rounded-full border border-ink-600 px-3 py-1 text-xs text-neutral-300 hover:border-neutral-400">{s}</button>)}</div>
      <div className="space-y-4">{log.map((a, i) => <div key={i} className="space-y-2">
        <div className="ml-auto max-w-2xl rounded-lg bg-ink-700 px-4 py-2 text-sm">{a.question}</div>
        <div className="card max-w-4xl p-4"><div className="mb-2 flex items-center gap-2 text-xs text-neutral-400"><Bot className="h-4 w-4" />{a.refused ? "Declined" : `Topics: ${a.topics?.join(", ")}`}</div>
          <p className="text-sm leading-relaxed text-neutral-100">{a.answer}</p>
          {a.facts?.length > 0 && <ul className="mt-3 space-y-1 border-t border-ink-700 pt-3">{a.facts.map((f: any, j: number) => <li key={j} className="flex items-center gap-2 text-xs text-neutral-400"><Evidence type={f.evidence_type} /> source: <code>{f.source}</code></li>)}</ul>}
        </div></div>)}</div>
      <form onSubmit={(e) => { e.preventDefault(); ask(); }} className="sticky bottom-4 flex gap-2"><input className="input !py-3" value={q} onChange={(e) => setQ(e.target.value)} maxLength={600} placeholder="Ask about viewers, models, content, experiments…" /><button className="btn-primary" disabled={busy}><Send className="h-4 w-4" /></button></form>
    </EnterpriseShell>
  );
}
