# SAM calls `make build-<FunctionLogicalId>` for every function whose
# infra/template.yaml sets Metadata.BuildMethod=makefile — the standard SAM
# pattern for code shared across functions in a monorepo (here: linkage/
# and handlers/, used by all 11 functions). CodeUri is the repo root, so
# SAM copies this whole tree (minus .samignore's exclusions) into a build
# context first and runs `make` from there — `linkage/` and `handlers/`
# are already siblings of this Makefile at build time, no `..` needed.
#
# Deliberately does NOT use the repo root's requirements.txt (pandas,
# scikit-learn, matplotlib, pyarrow — local/training-only). Only
# handlers/requirements.txt (numpy) goes into the Lambda package
# (CLAUDE.md rule 3).

build-IngestFunction: build-common
build-EnrichFunction: build-common
build-BuildIndexFunction: build-common
build-LinkFunction: build-common
build-ListEnrichTargetsFunction: build-common
build-ListLinkChunksFunction: build-common
build-ApiGetCaseFunction: build-common
build-ApiGetCaseLinksFunction: build-common
build-ApiPostFeedbackFunction: build-common
build-PiiRequestFunction: build-common
build-PiiReaderFunction: build-common

build-common:
	mkdir -p "$(ARTIFACTS_DIR)"
	cp -r linkage "$(ARTIFACTS_DIR)/"
	cp -r handlers "$(ARTIFACTS_DIR)/"
	find "$(ARTIFACTS_DIR)" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	python3 -m pip install -r handlers/requirements.txt -t "$(ARTIFACTS_DIR)" --only-binary=:all: --python-version 3.12 --platform manylinux2014_x86_64
