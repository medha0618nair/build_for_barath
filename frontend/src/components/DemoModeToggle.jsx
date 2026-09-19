export default function DemoModeToggle({ active, onToggle }) {
  return (
    <button
      onClick={onToggle}
      className={
        "chip border px-2.5 py-1 text-xs font-medium " +
        (active
          ? "border-accent-muted text-accent bg-accent-dim/30"
          : "border-base-600 text-base-300 hover:bg-base-800")
      }
      title="Load the relocated cross-state same-offender pair from fixtures."
    >
      {active ? "Demo mode: on" : "Demo mode"}
    </button>
  );
}
