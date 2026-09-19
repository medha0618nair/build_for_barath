import {
  ComposedChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  ReferenceLine,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { formatFieldName, formatValue, signedBits } from "../lib/format.js";

// Waterfall via the standard stacked-bar trick: an invisible "base" segment
// from 0 to min(before, after), then a visible "delta" segment of length
// |after-before| in the direction of travel. Prior is the starting offset;
// each field row nudges the running total; the final row anchors it.
function buildRows(link) {
  const rows = [];
  let running = link.prior_bits ?? 0;

  rows.push({
    key: "prior",
    label: "prior",
    kind: "prior",
    before: 0,
    after: running,
    bits: running,
  });

  for (const c of link.contributions) {
    const before = running;
    running += c.bits;
    rows.push({
      key: c.field,
      label: formatFieldName(c.field),
      kind: "field",
      field: c.field,
      value_a: c.value_a,
      value_b: c.value_b,
      u: c.u,
      provenance: c.provenance,
      before,
      after: running,
      bits: c.bits,
    });
  }

  rows.push({
    key: "total",
    label: "total",
    kind: "total",
    before: 0,
    after: link.total_bits,
    bits: link.total_bits,
  });

  return rows.map((r) => ({
    ...r,
    base: Math.min(r.before, r.after),
    size: Math.abs(r.after - r.before),
  }));
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  if (row.kind === "prior") {
    return (
      <div className="panel p-2 text-xs">
        <div className="font-mono text-base-200">prior</div>
        <div className="text-base-400">starting offset {signedBits(row.bits)} bits</div>
      </div>
    );
  }
  if (row.kind === "total") {
    return (
      <div className="panel p-2 text-xs">
        <div className="font-mono text-base-200">total</div>
        <div className="text-base-400">prior + evidence = {signedBits(row.bits)} bits</div>
      </div>
    );
  }
  return (
    <div className="panel p-2 text-xs max-w-[220px]">
      <div className="font-mono text-base-200">{row.label}</div>
      <div className="text-base-400 mt-1">
        a: <span className="text-base-300">{formatValue(row.value_a)}</span>
      </div>
      <div className="text-base-400">
        b: <span className="text-base-300">{formatValue(row.value_b)}</span>
      </div>
      <div className="text-base-400">u: {row.u?.toFixed(4)}</div>
      <div className={row.bits >= 0 ? "text-accent" : "text-negative"}>{signedBits(row.bits)} bits</div>
    </div>
  );
}

export default function ContributionPanel({ link }) {
  if (!link) {
    return <div className="panel p-4 text-sm text-base-500">Select a link to see its contribution breakdown.</div>;
  }

  const rows = buildRows(link);
  const allValues = rows.flatMap((r) => [r.before, r.after]);
  const min = Math.min(0, ...allValues);
  const max = Math.max(0, ...allValues);
  const pad = Math.max(0.5, (max - min) * 0.08);

  const height = Math.max(220, rows.length * 32 + 40);

  return (
    <div className="panel p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="field-label">Contribution breakdown</div>
        <div className={"font-mono text-sm " + (link.total_bits >= 0 ? "text-accent" : "text-negative")}>
          {signedBits(link.total_bits)} bits total
        </div>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart
          data={rows}
          layout="vertical"
          margin={{ top: 4, right: 24, bottom: 4, left: 8 }}
          barCategoryGap={6}
        >
          <XAxis
            type="number"
            domain={[min - pad, max + pad]}
            tick={{ fill: "#8a929a", fontSize: 11 }}
            stroke="#3a4148"
          />
          <YAxis
            type="category"
            dataKey="label"
            width={140}
            tick={{ fill: "#b7bec4", fontSize: 11, fontFamily: "ui-monospace, monospace" }}
            stroke="#3a4148"
          />
          <ReferenceLine x={0} stroke="#5b636b" strokeDasharray="3 3" />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          <Bar dataKey="base" stackId="wf" fill="transparent" isAnimationActive={false} />
          <Bar dataKey="size" stackId="wf" isAnimationActive={false} radius={2}>
            {rows.map((row) => (
              <Cell
                key={row.key}
                fill={
                  row.kind === "prior"
                    ? "#5b636b"
                    : row.kind === "total"
                    ? "#b7bec4"
                    : row.bits >= 0
                    ? "#4fd1a5"
                    : "#e07a7a"
                }
              />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
      <div className="text-[11px] text-base-500 mt-1">
        grey bar is the prior (starting offset); green/red bars are field-level evidence; the light bar at the
        bottom is the running total. Dashed line is break-even (0 bits).
      </div>
    </div>
  );
}
