import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pitwall",
  description:
    "Live F1 lap-time forecasts per tyre compound, with strategy simulation. Unofficial, personal project.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
