import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth";

export const metadata: Metadata = {
  title: "NTG — Neural Taste Graph",
  description: "Entertainment journey intelligence: understand evolving taste, discover beyond the obvious, avoid recommendation fatigue.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ ["--font-display" as any]: "'Iowan Old Style','Palatino Linotype',Palatino,Georgia,serif", ["--font-sans" as any]: "Inter,'Segoe UI',system-ui,-apple-system,sans-serif" }}>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
