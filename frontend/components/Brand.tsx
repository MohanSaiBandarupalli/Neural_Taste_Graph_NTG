export function Brand({ small = false }: { small?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2">
      <svg width={small ? 22 : 28} height={small ? 22 : 28} viewBox="0 0 32 32" aria-hidden>
        <circle cx="8" cy="9" r="3.2" fill="#e50914" /><circle cx="24" cy="8" r="2.6" fill="#f2f1ee" /><circle cx="16" cy="23" r="3.6" fill="#e50914" />
        <path d="M8 9 L24 8 L16 23 Z" fill="none" stroke="#e50914" strokeWidth="1.4" opacity=".7" />
      </svg>
      <span className={`font-display tracking-tight ${small ? "text-lg" : "text-2xl"}`}>NTG</span>
    </span>
  );
}
