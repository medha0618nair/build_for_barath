// Dev corpus size (config/corpus.yaml n_cases) — CLAUDE.md: "Dev corpus:
// 10,000 cases." Used only for the "rank N of M" denominator; it is the
// documented corpus size, never an invented candidate-pool count.
export const DEV_CORPUS_SIZE = 10000;

const SENTINELS = {
  __MISSING__: "missing",
  __ABSENT__: "absent",
  __UNKNOWABLE__: "unknowable",
};

export function formatBits(value) {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${Math.abs(value).toFixed(2)} bits`;
}

export function signedBits(value) {
  const sign = value >= 0 ? "+" : "−";
  return `${sign}${Math.abs(value).toFixed(2)}`;
}

export function formatValue(value) {
  if (value === null || value === undefined) return "–";
  if (SENTINELS[value]) return SENTINELS[value];
  if (Array.isArray(value)) return value.length ? value.join(", ") : "none";
  return String(value);
}

export function isSentinel(value) {
  return typeof value === "string" && value in SENTINELS;
}

export function formatFieldName(field) {
  // "tools:screwdriver" -> "tools: screwdriver"
  const [base, tag] = field.split(":");
  const label = base.replace(/_/g, " ");
  return tag ? `${label}: ${tag}` : label;
}

export function formatDate(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-IN", { year: "numeric", month: "short", day: "2-digit" });
}

export function formatDateTime(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-IN", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatCrimeType(crimeType) {
  return crimeType
    .toLowerCase()
    .split("_")
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join(" ");
}

const STATE_PALETTE = [
  "#4fd1a5",
  "#7aa2e0",
  "#e0b97a",
  "#c98ae0",
  "#e07a7a",
  "#7ae0d6",
  "#e0d47a",
  "#a2e07a",
];

const stateColorCache = new Map();
export function colorForState(stateCode) {
  if (!stateColorCache.has(stateCode)) {
    const idx = stateColorCache.size % STATE_PALETTE.length;
    stateColorCache.set(stateCode, STATE_PALETTE[idx]);
  }
  return stateColorCache.get(stateCode);
}

export function driverLabel(field, a, b) {
  const name = formatFieldName(field);
  if (a === b) return `${name}=${a}`;
  return `${name} disagrees`;
}
