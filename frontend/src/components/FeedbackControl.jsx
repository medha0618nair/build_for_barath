import { useState } from "react";
import { postFeedback } from "../lib/dataClient.js";

const STATUSES = [
  { value: "confirmed", label: "Confirm", cls: "border-accent-muted text-accent hover:bg-accent-dim/30" },
  { value: "rejected", label: "Reject", cls: "border-negative-muted text-negative hover:bg-negative-dim/30" },
  { value: "unsure", label: "Unsure", cls: "border-base-600 text-base-300 hover:bg-base-800" },
];

export default function FeedbackControl({ caseIdA, caseIdB }) {
  const [status, setStatus] = useState(null);
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [record, setRecord] = useState(null);
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    if (!status || !reason.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      const result = await postFeedback({
        case_id_a: caseIdA,
        case_id_b: caseIdB,
        status,
        reason: reason.trim(),
      });
      setRecord(result);
    } catch (err) {
      setError(err.message || "submission failed");
    } finally {
      setSubmitting(false);
    }
  }

  if (record) {
    return (
      <div className="panel p-3 text-sm">
        <div className="text-accent">Feedback recorded: {record.status}</div>
        <div className="text-base-500 text-xs mt-0.5">
          {record.pair_id} {"·"} {record.submitted_at}
        </div>
        <button
          className="mt-2 text-xs text-base-400 underline"
          onClick={() => {
            setRecord(null);
            setStatus(null);
            setReason("");
          }}
        >
          submit another
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="panel p-3 flex flex-col gap-2">
      <div className="field-label">Analyst feedback on this link</div>
      <div className="flex gap-2">
        {STATUSES.map((s) => (
          <button
            type="button"
            key={s.value}
            onClick={() => setStatus(s.value)}
            className={
              "chip border px-2 py-1 text-xs " +
              s.cls +
              (status === s.value ? " bg-base-800 ring-1 ring-base-500" : "")
            }
          >
            {s.label}
          </button>
        ))}
      </div>
      <textarea
        required
        value={reason}
        onChange={(e) => setReason(e.target.value)}
        placeholder="Reason (required) — why does this look like a match or not?"
        rows={2}
        className="bg-base-800 border border-base-700 rounded px-2 py-1.5 text-sm text-base-200 placeholder:text-base-600 resize-none focus:outline-none focus:ring-1 focus:ring-accent-muted"
      />
      {error && <div className="text-negative text-xs">{error}</div>}
      <button
        type="submit"
        disabled={!status || !reason.trim() || submitting}
        className="self-start chip border border-base-600 px-3 py-1.5 text-xs text-base-200 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-base-800"
      >
        {submitting ? "submitting…" : "submit feedback"}
      </button>
    </form>
  );
}
