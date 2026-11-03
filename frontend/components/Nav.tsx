"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Search, LogOut, LayoutDashboard } from "lucide-react";
import clsx from "clsx";
import { Brand } from "./Brand";
import { useAuth } from "@/lib/auth";
import { activeProfile, api, Profile } from "@/lib/api";

const LINKS = [["/browse", "Home"], ["/discover", "Discover"], ["/journey", "Journeys"], ["/tastetwin", "TasteTwin"], ["/history", "History"]];

export function Nav() {
  const path = usePathname();
  const router = useRouter();
  const { user, logout } = useAuth();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [q, setQ] = useState("");
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 10); window.addEventListener("scroll", on); return () => window.removeEventListener("scroll", on);
  }, []);
  useEffect(() => {
    if (!user) return;
    api<Profile[]>("/v1/profiles").then((ps) => setProfile(ps.find((p) => p.id === activeProfile.get()) ?? null)).catch(() => {});
  }, [user, path]);
  return (
    <header className={clsx("fixed inset-x-0 top-0 z-40 flex h-16 items-center gap-6 px-6 transition md:px-12", scrolled ? "bg-ink-950/95 backdrop-blur" : "bg-gradient-to-b from-black/80 to-transparent")}>
      <Link href="/browse"><Brand small /></Link>
      <nav className="hidden gap-5 text-sm md:flex">
        {LINKS.map(([h, l]) => <Link key={h} href={h} className={clsx("transition hover:text-white", path.startsWith(h) ? "font-semibold text-white" : "text-neutral-300")}>{l}</Link>)}
      </nav>
      <form className="ml-auto hidden items-center sm:flex" onSubmit={(e) => { e.preventDefault(); if (q.trim()) router.push(`/search?q=${encodeURIComponent(q)}`); }}>
        <div className="relative"><Search className="absolute left-2.5 top-2.5 h-4 w-4 text-neutral-400" />
          <input aria-label="Search titles" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Titles, moods, genres" className="input w-56 !bg-black/50 !pl-8" /></div>
      </form>
      {user && user.role !== "viewer" && <Link href="/enterprise" className="hidden items-center gap-1 text-xs text-neutral-300 hover:text-white lg:flex"><LayoutDashboard className="h-4 w-4" />Enterprise</Link>}
      {profile && <Link href="/profiles" title="Switch profile" className="flex items-center gap-2 text-sm"><span className="grid h-8 w-8 place-items-center rounded font-bold" style={{ background: profile.color }}>{profile.name[0]}</span><span className="hidden text-neutral-300 lg:inline">{profile.name}</span></Link>}
      <Link href="/account" className="text-xs text-neutral-400 hover:text-white">Account</Link>
      <button aria-label="Sign out" onClick={() => { logout(); router.push("/"); }} className="text-neutral-400 hover:text-white"><LogOut className="h-4 w-4" /></button>
    </header>
  );
}

export function useProfileId(): number | null {
  const router = useRouter();
  const [pid, setPid] = useState<number | null>(null);
  useEffect(() => { const p = activeProfile.get(); if (!p) router.replace("/profiles"); else setPid(p); }, [router]);
  return pid;
}
