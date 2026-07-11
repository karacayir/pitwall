import { DriverClient } from "./DriverClient";

// Static export: pre-render a shell for every legal F1 race number; all data
// arrives client-side over the WebSocket.
export function generateStaticParams() {
  return Array.from({ length: 99 }, (_, i) => ({ num: String(i + 1) }));
}

export default async function DriverPage({ params }: { params: Promise<{ num: string }> }) {
  const { num } = await params;
  return <DriverClient num={num} />;
}
