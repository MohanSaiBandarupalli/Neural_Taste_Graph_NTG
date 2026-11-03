"use client";
import { Nav } from "./Nav";
import { useRequireAuth } from "@/lib/auth";
import { Spinner } from "./ui";
import { useApi } from "@/lib/hooks";

export function ConsumerShell({ children }: { children: React.ReactNode }) {
  const { ready } = useRequireAuth();
  const meta = useApi<any>(ready ? "/v1/meta" : null);
  if (!ready) return <Spinner />;
  return (
    <div className="min-h-screen pb-16">
      <Nav />
      {children}
      <footer className="mt-10 border-t border-ink-800 px-6 pt-6 text-xs text-neutral-500 md:px-12">
        {meta.data?.data_notice} · model {meta.data?.model_version} · NTG is an independent project and is not affiliated with any streaming service.
      </footer>
    </div>
  );
}
