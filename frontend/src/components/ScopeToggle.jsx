import { signedBits } from "../lib/format.js";

// CLAUDE.md hard rule #1: retrieval/scoring stay untouched by this control.
// It only swaps which precomputed prior + candidate pool (scope=same vs
// scope=all, TECHNICAL_SPEC.md §4.4) the links panel is showing.
export default function ScopeToggle({ scope, onChange, priorBits }) {
  const isAll = scope === "all";
  return (
    <div className="panel px-3 py-2 flex items-center justify-between">
      <div>
        <div className="text-sm text-base-200">Widen to all property crime</div>
        <div className="text-[11px] text-base-500">
          {isAll ? "Scope: all property-crime types (cross-type prior)" : "Scope: same crime type only"}
        </div>
      </div>
      <div className="flex items-center gap-3">
        {priorBits !== undefined && priorBits !== null && (
          <span className="font-mono text-xs text-base-400">prior {signedBits(priorBits)} bits</span>
        )}
        <button
          role="switch"
          aria-checked={isAll}
          onClick={() => onChange(isAll ? "same" : "all")}
          className={
            "relative inline-flex h-5 w-9 items-center rounded-full transition-colors " +
            (isAll ? "bg-accent-muted" : "bg-base-700")
          }
        >
          <span
            className={
              "inline-block h-3.5 w-3.5 transform rounded-full bg-base-200 transition-transform " +
              (isAll ? "translate-x-[18px]" : "translate-x-1")
            }
          />
        </button>
      </div>
    </div>
  );
}
