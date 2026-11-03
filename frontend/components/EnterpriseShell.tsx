"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import clsx from "clsx";
import { Activity, BarChart3, Bot, Brain, Database, FlaskConical, Gauge, LayoutDashboard, LineChart, Route, Settings, ShieldAlert, Sparkles, Users, Wallet, Film } from "lucide-react";
import { Brand } from "./Brand";
import { Spinner } from "./ui";
import { useAuth, useRequireAuth } from "@/lib/auth";

const NAV = [
  ["", "Executive dashboard", LayoutDashboard], ["audience", "Audience intelligence", Users], ["tastetwin", "TasteTwin analytics", Brain],
  ["performance", "Recommendation performance", BarChart3], ["models", "Model laboratory", FlaskConical], ["journey", "Journey simulator", Route],
  ["fatigue", "Fatigue intelligence", Sparkles], ["risk", "Engagement risk", ShieldAlert], ["scenario", "Business simulator", Wallet],
  ["content", "Content intelligence", Film], ["experiments", "Experiments", LineChart], ["monitoring", "Monitoring", Activity],
  ["data", "Data management", Database], ["analyst", "AI Business Analyst", Bot], ["settings", "Settings & admin", Settings],
] as const;

export function EnterpriseShell({ title, subtitle, children, actions }: { title: string; subtitle?: string; children: React.ReactNode; actions?: React.ReactNode }) {
  const { ready, user } = useRequireAuth(["analyst", "admin"]);
  const { logout } = useAuth();
  const path = usePathname();
  const router = useRouter();
  if (!ready) return <Spinner />;
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 flex-col border-r border-ink-800 bg-ink-900 p-4 lg:flex">
        <Link href="/enterprise" className="mb-1 px-2"><Brand small /></Link>
        <p className="mb-5 px-2 text-[11px] uppercase tracking-widest text-neutral-500">Enterprise Intelligence</p>
        <nav className="flex-1 space-y-0.5 overflow-y-auto">
          {NAV.map(([slug, label, Icon]) => {
            const href = `/enterprise${slug ? "/" + slug : ""}`;
            const active = slug ? path.startsWith(href) : path === "/enterprise";
            return <Link key={href} href={href} className={clsx("flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm", active ? "bg-ink-700 text-white" : "text-neutral-400 hover:bg-ink-800 hover:text-white")}><Icon className="h-4 w-4" />{label}</Link>;
          })}
        </nav>
        <div className="mt-4 border-t border-ink-800 pt-3 text-xs text-neutral-400">
          <div>{user?.display_name}</div><div className="text-neutral-500">{user?.role}</div>
          <div className="mt-2 flex gap-3"><Link href="/profiles" className="hover:text-white">Consumer app</Link><button onClick={() => { logout(); router.push("/"); }} className="hover:text-white">Sign out</button></div>
        </div>
      </aside>
      <main className="min-w-0 flex-1 px-4 py-6 md:px-8">
        <div className="mb-2 flex gap-2 overflow-x-auto pb-2 lg:hidden">{NAV.map(([slug, label]) => <Link key={slug} href={`/enterprise${slug ? "/" + slug : ""}`} className="whitespace-nowrap rounded-full border border-ink-600 px-3 py-1 text-xs">{label}</Link>)}</div>
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4"><div><h1 className="font-display text-3xl">{title}</h1>{subtitle && <p className="mt-1 max-w-4xl text-sm text-neutral-400">{subtitle}</p>}</div>{actions}</div>
        <div className="fade-in space-y-6">{children}</div>
      </main>
    </div>
  );
}
