export default function CasePicker({ caseIds, value, onChange, examples }) {
  const exampleIds = new Set(Object.values(examples || {}).flat());
  return (
    <select
      value={value || ""}
      onChange={(e) => onChange(e.target.value)}
      className="bg-base-800 border border-base-700 rounded px-2 py-1 text-sm font-mono text-base-200 focus:outline-none focus:ring-1 focus:ring-accent-muted"
    >
      <option value="" disabled>
        select a case{"…"}
      </option>
      {caseIds.map((id) => (
        <option key={id} value={id}>
          {id}
          {exampleIds.has(id) ? " ★" : ""}
        </option>
      ))}
    </select>
  );
}
