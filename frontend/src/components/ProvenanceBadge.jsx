// CLAUDE.md: "Every MO field carries field_provenance: source or
// llm_extracted. An analyst must be able to tell a recorded value from an
// inferred one." This badge is that tell.
export default function ProvenanceBadge({ provenance }) {
  if (provenance === "llm_extracted") {
    return (
      <span className="chip bg-base-800 border border-base-600 text-base-300" title="Inferred from narrative text by the extraction model, not recorded on the form.">
        llm
      </span>
    );
  }
  if (provenance === "source") {
    return (
      <span className="chip bg-base-800 border border-base-700 text-base-500" title="Recorded directly on the case form.">
        src
      </span>
    );
  }
  return (
    <span className="chip bg-base-900 border border-base-800 text-base-600" title="No value on record.">
      {"–"}
    </span>
  );
}
