import { useEffect, useState } from "react";
import { getCaseLinks } from "./dataClient.js";

export function useLinks(caseId, scope) {
  const [state, setState] = useState({ loading: Boolean(caseId), data: null, error: null });

  useEffect(() => {
    if (!caseId) {
      setState({ loading: false, data: null, error: null });
      return;
    }
    let cancelled = false;
    setState({ loading: true, data: null, error: null });
    getCaseLinks(caseId, scope)
      .then((data) => {
        if (!cancelled) setState({ loading: false, data, error: null });
      })
      .catch((error) => {
        if (!cancelled) setState({ loading: false, data: null, error });
      });
    return () => {
      cancelled = true;
    };
  }, [caseId, scope]);

  return state;
}
