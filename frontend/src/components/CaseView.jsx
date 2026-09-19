import ProvenanceBadge from "./ProvenanceBadge.jsx";
import { formatValue, formatFieldName, formatDateTime, formatCrimeType, colorForState } from "../lib/format.js";

function FieldRow({ label, value, provenance }) {
  return (
    <div className="flex items-start justify-between gap-3 py-1 border-b border-base-800/60 last:border-0">
      <div className="flex items-center gap-2 min-w-0">
        <ProvenanceBadge provenance={provenance} />
        <span className="text-sm text-base-300 truncate">{label}</span>
      </div>
      <span className="text-sm font-mono text-base-200 text-right shrink-0">{formatValue(value)}</span>
    </div>
  );
}

export default function CaseView({ caseRecord, title = "Case" }) {
  if (!caseRecord) {
    return (
      <div className="panel p-4 text-sm text-base-500">Select a case to inspect it.</div>
    );
  }

  const { case_id, state_code, district, crime_type, occurred_from, occurred_to, registered_at,
    mo_core, mo_ext, narrative_text, field_provenance } = caseRecord;

  const fields = [...Object.entries(mo_core || {}), ...Object.entries(mo_ext || {})];

  return (
    <div className="panel p-4 flex flex-col gap-4">
      <div>
        <div className="field-label">{title}</div>
        <div className="flex items-baseline gap-2 mt-1">
          <span className="font-mono text-base text-base-200">{case_id}</span>
          <span
            className="chip border"
            style={{ borderColor: colorForState(state_code), color: colorForState(state_code) }}
          >
            {state_code}
          </span>
        </div>
        <div className="text-sm text-base-400 mt-0.5">{district}</div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <div>
          <div className="field-label">Crime type</div>
          <div className="text-base-200">{formatCrimeType(crime_type)}</div>
        </div>
        <div>
          <div className="field-label">Registered</div>
          <div className="text-base-200">{formatDateTime(registered_at)}</div>
        </div>
        <div className="col-span-2">
          <div className="field-label">Occurred</div>
          <div className="text-base-200">
            {formatDateTime(occurred_from)} {"→"} {formatDateTime(occurred_to)}
          </div>
        </div>
      </div>

      <div>
        <div className="field-label mb-1">MO fields</div>
        <div className="flex flex-col">
          {fields.map(([field, value]) => (
            <FieldRow
              key={field}
              label={formatFieldName(field)}
              value={value}
              provenance={field_provenance?.[field] ?? null}
            />
          ))}
        </div>
      </div>

      <div>
        <div className="field-label mb-1">Narrative</div>
        <p className="text-sm text-base-300 leading-relaxed whitespace-pre-wrap">{narrative_text}</p>
      </div>
    </div>
  );
}
