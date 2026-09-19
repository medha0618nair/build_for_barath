import ProvenanceBadge from "./ProvenanceBadge.jsx";
import { formatFieldName, formatValue, signedBits } from "../lib/format.js";

export default function FieldComparisonTable({ rows, caseALabel = "Case A", caseBLabel = "Case B", showBits = true }) {
  if (!rows || rows.length === 0) {
    return <div className="panel p-4 text-sm text-base-500">No comparable fields for this pair.</div>;
  }
  return (
    <div className="panel overflow-hidden">
      <div className="px-3 py-2 border-b border-base-700 field-label">Field comparison</div>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-[11px] uppercase tracking-wide text-base-500 border-b border-base-800">
            <th className="px-3 py-1.5 font-normal">Field</th>
            <th className="px-3 py-1.5 font-normal">{caseALabel}</th>
            <th className="px-3 py-1.5 font-normal">{caseBLabel}</th>
            {showBits && <th className="px-3 py-1.5 font-normal text-right">Bits</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const agree = row.value_a === row.value_b;
            return (
              <tr
                key={row.field}
                className={"border-b border-base-800/60 last:border-0 " + (agree ? "bg-accent-dim/20" : "")}
              >
                <td className="px-3 py-1.5 text-base-300">{formatFieldName(row.field)}</td>
                <td className="px-3 py-1.5 font-mono">
                  <span className="flex items-center gap-1.5">
                    {row.provenance?.a && <ProvenanceBadge provenance={row.provenance.a} />}
                    <span className={agree ? "text-accent" : "text-base-200"}>{formatValue(row.value_a)}</span>
                  </span>
                </td>
                <td className="px-3 py-1.5 font-mono">
                  <span className="flex items-center gap-1.5">
                    {row.provenance?.b && <ProvenanceBadge provenance={row.provenance.b} />}
                    <span className={agree ? "text-accent" : "text-base-200"}>{formatValue(row.value_b)}</span>
                  </span>
                </td>
                {showBits && (
                  <td
                    className={
                      "px-3 py-1.5 text-right font-mono " +
                      (row.bits === undefined || row.bits === null
                        ? "text-base-600"
                        : row.bits >= 0
                        ? "text-accent"
                        : "text-negative")
                    }
                  >
                    {row.bits === undefined || row.bits === null ? "–" : signedBits(row.bits)}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
