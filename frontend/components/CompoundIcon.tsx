import { COMPOUND_COLORS, COMPOUND_SHORT } from "@/lib/types";

export function CompoundIcon({
  compound,
  age,
  size = 20,
}: {
  compound: string | null;
  age?: number | null;
  size?: number;
}) {
  const color = compound ? (COMPOUND_COLORS[compound] ?? "#8B8B98") : "#8B8B98";
  const letter = compound ? (COMPOUND_SHORT[compound] ?? "?") : "?";
  return (
    <span className="inline-flex items-center gap-1">
      <svg width={size} height={size} viewBox="0 0 20 20" aria-label={compound ?? "unknown"}>
        <circle cx="10" cy="10" r="8.5" fill="none" stroke={color} strokeWidth="2.5" />
        <text
          x="10"
          y="13.5"
          textAnchor="middle"
          fontSize="9"
          fontWeight="700"
          fill={color}
          fontFamily="var(--font-mono)"
        >
          {letter}
        </text>
      </svg>
      {age != null && <span className="timing text-xs text-(--muted)">{age}</span>}
    </span>
  );
}
