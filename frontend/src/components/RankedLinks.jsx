import { useCase } from "../lib/useCase.js";
import PairClassChip from "./PairClassChip.jsx";
import { DEV_CORPUS_SIZE, formatDate, signedBits, colorForState, driverLabel } from "../lib/format.js";

function LinkRow({ link, isSelected, onSelect }) {
  const { data: partner } = useCase(link.partner_id);
  const positive = link.total_bits >= 0;

  return (
    <button
      onClick={() => onSelect(link.partner_id)}
      className={
        "w-full text-left px-3 py-2 border-b border-base-800/70 transition-colors " +
        (isSelected ? "bg-base-800" : "hover:bg-base-850")
      }
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="font-mono text-sm text-base-200 truncate">{link.partner_id}</span>
          {partner && (
            <span
              className="chip border shrink-0"
              style={{ borderColor: colorForState(partner.state_code), color: colorForState(partner.state_code) }}
            >
              {partner.state_code}
            </span>
          )}
        </div>
        <span
          className={"font-mono text-sm shrink-0 " + (positive ? "text-accent" : "text-negative")}
        >
          {signedBits(link.total_bits)} bits
        </span>
      </div>
      <div className="flex items-center gap-2 mt-0.5 text-[11px] text-base-500">
        <span>
          rank {link.rank} of {DEV_CORPUS_SIZE.toLocaleString("en-IN")}
        </span>
        <PairClassChip pairClass={link.pair_class} />
        {partner && <span>{formatDate(partner.occurred_from)}</span>}
      </div>
      <div className="mt-1 text-xs text-base-400 truncate">
        driven by{" "}
        {(link.driven_by || []).map((d) => d).join(" · ") ||
          link.contributions
            .slice(0, 3)
            .map((c) => driverLabel(c.field, c.value_a, c.value_b))
            .join(" · ")}
      </div>
    </button>
  );
}

export default function RankedLinks({ links, caseId, scope, selectedPartnerId, onSelect }) {
  if (!links) {
    return <div className="panel p-4 text-sm text-base-500">No links loaded.</div>;
  }
  return (
    <div className="panel flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-base-700 flex items-center justify-between">
        <div className="field-label">Ranked links {"·"} {scope}</div>
        <div className="text-[11px] text-base-500">{links.length} shown</div>
      </div>
      <div className="overflow-y-auto max-h-[520px]">
        {links.length === 0 && (
          <div className="px-3 py-4 text-sm text-base-500">No candidates cleared the shortlist for this case.</div>
        )}
        {links.map((link) => (
          <LinkRow
            key={link.partner_id}
            link={link}
            isSelected={link.partner_id === selectedPartnerId}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}
