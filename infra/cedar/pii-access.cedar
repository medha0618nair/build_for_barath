// Requires a stated, non-empty reason before PiiRole may be assumed
// (TECHNICAL_SPEC.md §7: "PiiRole is assumable only after Verified
// Permissions returns ALLOW on a request carrying a stated reason. The
// claim 'the linkage engine cannot see personal data' is an IAM boundary
// plus a separate KMS key — it holds even if the Lambda code has a bug.")
//
// handlers/pii.py calls AVP's IsAuthorized with this action before ever
// touching the `pii` table; a request with an empty or missing reason
// falls through to Cedar's implicit deny (no `permit` matches it).
permit (
    principal,
    action == CrimeLinkage::Action::"AssumePiiRole",
    resource
)
when {
    context has reason &&
    context.reason != ""
};
