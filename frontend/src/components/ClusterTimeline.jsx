import { useEffect, useState } from "react";
import { ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { getCase } from "../lib/dataClient.js";
import { colorForState, formatDate, signedBits } from "../lib/format.js";

// CLAUDE.md: "Clustering uses a deliberately high edge threshold. Transitive
// chaining creates bogus mega-clusters otherwise." There is no /clusters
// route in api/openapi.yaml (TECHNICAL_SPEC.md's cut order dropped it for
// this phase), so this derives a same-case cluster from the links panel's
// own scored edges instead of a persisted transitive cluster job. Labelled
// as such below rather than presented as the real batch-clustered output.
const EDGE_THRESHOLD_BITS = 5; // evidence bits (total_bits - prior_bits) required to count as an edge

function CustomDot({ cx, cy, payload }) {
  if (cx === undefined || cy === undefined) return null;
  return <circle cx={cx} cy={cy} r={6} fill={colorForState(payload.state_code)} stroke="#111417" strokeWidth={1.5} />;
}

function ClusterTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="panel p-2 text-xs">
      <div className="font-mono text-base-200">{p.case_id}</div>
      <div className="text-base-400">{p.state_code} {"·"} {formatDate(p.occurred_from)}</div>
      {p.total_bits !== undefined && (
        <div className={p.total_bits >= 0 ? "text-accent" : "text-negative"}>{signedBits(p.total_bits)} bits</div>
      )}
    </div>
  );
}

export default function ClusterTimeline({ caseRecord, links }) {
  const [members, setMembers] = useState(null);

  useEffect(() => {
    let cancelled = false;
    if (!caseRecord || !links) {
      setMembers(null);
      return;
    }
    const edges = links.filter((l) => l.total_bits - (l.prior_bits ?? 0) >= EDGE_THRESHOLD_BITS);
    Promise.all(
      edges.map((l) =>
        getCase(l.partner_id)
          .then((c) => ({
            case_id: c.case_id,
            state_code: c.state_code,
            occurred_from: c.occurred_from,
            total_bits: l.total_bits,
          }))
          // Fixtures only carry full case records for the demo set; a
          // partner_id with no case record (404) just drops out of the
          // drawn cluster rather than failing the whole panel.
          .catch(() => null)
      )
    ).then((results) => {
      if (cancelled) return;
      const partners = results.filter(Boolean);
      const self = {
        case_id: caseRecord.case_id,
        state_code: caseRecord.state_code,
        occurred_from: caseRecord.occurred_from,
        total_bits: null,
      };
      setMembers([self, ...partners].sort((a, b) => new Date(a.occurred_from) - new Date(b.occurred_from)));
    });
    return () => {
      cancelled = true;
    };
  }, [caseRecord, links]);

  if (!caseRecord) {
    return <div className="panel p-4 text-sm text-base-500">No case selected.</div>;
  }

  if (members === null) {
    return <div className="panel p-4 text-sm text-base-500">Loading cluster{"…"}</div>;
  }

  if (members.length < 2) {
    return (
      <div className="panel p-4 text-sm text-base-500">
        No other case clears the {EDGE_THRESHOLD_BITS}-bit edge threshold for {caseRecord.case_id} in this scope
        {"—"}no cluster to draw.
      </div>
    );
  }

  const data = members.map((m) => ({ ...m, t: new Date(m.occurred_from).getTime() }));
  const states = [...new Set(data.map((m) => m.state_code))];
  const relocation = states.length > 1;

  return (
    <div className="panel p-4">
      <div className="flex items-center justify-between mb-1">
        <div className="field-label">Cluster timeline</div>
        <div className="text-[11px] text-base-500">
          derived from links {"≥"} {EDGE_THRESHOLD_BITS} bits of evidence, not the persisted batch cluster
        </div>
      </div>
      {relocation && (
        <div className="text-xs text-negative mb-2">
          relocation across state lines: {states.join(" → ")}
        </div>
      )}
      <ResponsiveContainer width="100%" height={120}>
        <ComposedChart data={data} margin={{ top: 20, right: 24, bottom: 4, left: 8 }}>
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(t) => formatDate(new Date(t).toISOString())}
            tick={{ fill: "#8a929a", fontSize: 10 }}
            stroke="#3a4148"
          />
          <YAxis type="number" dataKey={() => 0} hide domain={[-1, 1]} />
          <ReferenceLine y={0} stroke="#2a3037" />
          <Tooltip content={<ClusterTooltip />} />
          <Line dataKey={() => 0} stroke="#3a4148" dot={false} isAnimationActive={false} />
          <Scatter dataKey={() => 0} shape={<CustomDot />} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-2 mt-1">
        {states.map((s) => (
          <span key={s} className="chip border" style={{ borderColor: colorForState(s), color: colorForState(s) }}>
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}
