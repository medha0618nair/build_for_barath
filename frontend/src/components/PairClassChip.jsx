export default function PairClassChip({ pairClass }) {
  const isSame = pairClass === "same_type";
  return (
    <span
      className={
        "chip border " +
        (isSame
          ? "border-accent-muted text-accent bg-accent-dim/40"
          : "border-base-600 text-base-300 bg-base-800")
      }
    >
      {isSame ? "same-type" : "cross-type"}
    </span>
  );
}
