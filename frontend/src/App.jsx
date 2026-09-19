import { useEffect, useMemo, useState } from "react";
import CaseView from "./components/CaseView.jsx";
import RankedLinks from "./components/RankedLinks.jsx";
import ContributionPanel from "./components/ContributionPanel.jsx";
import FieldComparisonTable from "./components/FieldComparisonTable.jsx";
import ScopeToggle from "./components/ScopeToggle.jsx";
import ClusterTimeline from "./components/ClusterTimeline.jsx";
import FeedbackControl from "./components/FeedbackControl.jsx";
import DemoModeToggle from "./components/DemoModeToggle.jsx";
import CasePicker from "./components/CasePicker.jsx";
import { browsableCaseIds, demoManifest, getCase, isLiveApi } from "./lib/dataClient.js";
import { useCase } from "./lib/useCase.js";
import { useLinks } from "./lib/useLinks.js";

const MO_CORE_FIELDS = [
  "time_band",
  "group_size_est",
  "tools",
  "counter_forensic",
  "target_selection",
  "property_taken",
  "approach_mode",
  "exit_mode",
];

function rawMoCoreComparison(caseA, caseB) {
  return MO_CORE_FIELDS.filter((f) => f in caseA.mo_core && f in caseB.mo_core).map((f) => ({
    field: f,
    value_a: Array.isArray(caseA.mo_core[f]) ? caseA.mo_core[f].join(", ") || "none" : caseA.mo_core[f],
    value_b: Array.isArray(caseB.mo_core[f]) ? caseB.mo_core[f].join(", ") || "none" : caseB.mo_core[f],
    provenance: { a: caseA.field_provenance?.[f], b: caseB.field_provenance?.[f] },
    bits: null,
  }));
}

export default function App() {
  const caseIds = useMemo(() => browsableCaseIds(), []);
  const manifest = useMemo(() => demoManifest(), []);

  const [primaryCaseId, setPrimaryCaseId] = useState(manifest.top_link_case || caseIds[0]);
  const [scope, setScope] = useState("same");
  const [selectedPartnerId, setSelectedPartnerId] = useState(null);
  const [demoMode, setDemoMode] = useState(false);
  const [demoFallback, setDemoFallback] = useState(false);
  const [demoPartnerCase, setDemoPartnerCase] = useState(null);

  const { data: primaryCase } = useCase(primaryCaseId);
  const { data: linksResponse, loading: linksLoading } = useLinks(primaryCaseId, scope);

  // Demo mode: load the relocated cross-state same-offender pair. It's a
  // cross-type pair (VEHICLE_THEFT x ATM_TAMPERING) that doesn't clear the
  // top-50 shortlist in either direction in this fixture corpus — that's
  // the honest limitation this pair is for (TECHNICAL_SPEC.md §7: "before
  // the system can find them with geography removed. That's a finding, not
  // a demo."). We show it as a direct case-to-case comparison instead of
  // pretending it's a scored ranked link.
  useEffect(() => {
    if (!demoMode) return;
    const [a, b] = manifest.relocated_demo_pair || [];
    if (!a || !b) return;
    setPrimaryCaseId(a);
    setScope("all");
    setSelectedPartnerId(b);
  }, [demoMode, manifest]);

  // Default the detail panels to the top-ranked link so they aren't empty
  // on first load of a case.
  useEffect(() => {
    if (demoMode || selectedPartnerId || !linksResponse?.links?.length) return;
    setSelectedPartnerId(linksResponse.links[0].partner_id);
  }, [demoMode, selectedPartnerId, linksResponse]);

  useEffect(() => {
    if (!demoMode || !selectedPartnerId || !linksResponse) return;
    const found = linksResponse.links.find((l) => l.partner_id === selectedPartnerId);
    if (found) {
      setDemoFallback(false);
      setDemoPartnerCase(null);
      return;
    }
    setDemoFallback(true);
    getCase(selectedPartnerId).then(setDemoPartnerCase);
  }, [demoMode, selectedPartnerId, linksResponse]);

  function selectPrimary(id) {
    setDemoMode(false);
    setDemoFallback(false);
    setPrimaryCaseId(id);
    setSelectedPartnerId(null);
  }

  function selectPartner(id) {
    setSelectedPartnerId(id);
    if (!demoMode) setDemoFallback(false);
  }

  function toggleDemo() {
    setDemoMode((v) => !v);
    if (demoMode) {
      setDemoFallback(false);
      setSelectedPartnerId(null);
    }
  }

  const selectedLink =
    !demoFallback && linksResponse?.links.find((l) => l.partner_id === selectedPartnerId);

  const comparisonRows = selectedLink
    ? selectedLink.contributions
    : demoFallback && primaryCase && demoPartnerCase
    ? rawMoCoreComparison(primaryCase, demoPartnerCase)
    : null;

  return (
    <div className="min-h-screen bg-base-950">
      <header className="border-b border-base-800 px-4 py-3 flex items-center justify-between sticky top-0 bg-base-950/95 backdrop-blur z-10">
        <div>
          <h1 className="text-sm font-semibold text-base-200">Cross-jurisdiction crime linkage</h1>
          <p className="text-[11px] text-base-500">
            Ranked shortlist for a human analyst {"·"} synthetic corpus {"·"} never a probability
            {isLiveApi ? " · live API" : " · fixtures (offline)"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <CasePicker caseIds={caseIds} value={primaryCaseId} onChange={selectPrimary} examples={manifest} />
          <DemoModeToggle active={demoMode} onToggle={toggleDemo} />
        </div>
      </header>

      <main className="p-4 grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-4 max-w-[1400px] mx-auto">
        <div className="flex flex-col gap-4">
          <CaseView caseRecord={primaryCase} title="Case A" />
          {demoFallback && demoPartnerCase && <CaseView caseRecord={demoPartnerCase} title="Case B (demo pair)" />}
        </div>

        <div className="flex flex-col gap-4">
          <ScopeToggle scope={scope} onChange={setScope} priorBits={linksResponse?.links?.[0]?.prior_bits} />

          {demoFallback && (
            <div className="panel p-3 text-sm border-negative-muted">
              <div className="text-negative font-medium">Not in the top 50, either direction</div>
              <p className="text-base-400 text-xs mt-1">
                {manifest.relocated_demo_pair?.[0]} and {manifest.relocated_demo_pair?.[1]} are the same offender in
                the synthetic ground truth {"—"} a cross-state, cross-type relocation. MO-only scoring doesn't
                surface this pair in either case's ranked list in this corpus. Shown below as a direct field
                comparison (mo_core only, no bits) instead of a scored link: this is the known-limits case, not a
                success demo.
              </p>
            </div>
          )}

          {linksLoading && <div className="panel p-4 text-sm text-base-500">Loading links{"…"}</div>}

          {linksResponse && (
            <RankedLinks
              links={linksResponse.links}
              caseId={primaryCaseId}
              scope={scope}
              selectedPartnerId={demoFallback ? null : selectedPartnerId}
              onSelect={selectPartner}
            />
          )}

          {selectedLink && <ContributionPanel link={selectedLink} />}
          {demoFallback && (
            <div className="panel p-4 text-sm text-base-500">
              No contribution chart {"—"} this pair was never scored by the linker (outside the shortlist), so
              there are no per-field bits to show.
            </div>
          )}

          {comparisonRows && (
            <FieldComparisonTable
              rows={comparisonRows}
              caseALabel={primaryCaseId}
              caseBLabel={selectedPartnerId}
              showBits={!demoFallback}
            />
          )}

          {primaryCase && <ClusterTimeline caseRecord={primaryCase} links={linksResponse?.links} />}

          {(selectedLink || demoFallback) && selectedPartnerId && (
            <FeedbackControl caseIdA={primaryCaseId} caseIdB={selectedPartnerId} />
          )}
        </div>
      </main>
    </div>
  );
}
