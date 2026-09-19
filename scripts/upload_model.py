"""Deploy step: precomputes frequencies/priors (linkage/frequencies.py,
linkage/prior.py — both pandas-dependent, offline-only) and uploads them
plus weights.json to S3, so handlers/link.py's Lambda can load them with
json.loads instead of recomputing from a DynamoDB scan on every cold start.

    python -m linkage.train                              # writes weights.json
    python -m scripts.upload_model --bucket crime-linkage-dev-artifacts

Also builds and uploads a SEED narrative embedding index (vectors/index.npy,
vectors/case_ids.json) using the same local tfidf_original stub the local
pipeline uses, so /cases/{id}/links answers something immediately after
deploy without waiting on Bedrock. The real BuildIndex Step Functions state
(handlers/build_index.py) overwrites this with Cohere embeddings on the
pipeline's first scheduled/S3-triggered run.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import boto3
import numpy as np

from linkage import config as config_mod
from linkage import frequencies as frequencies_mod
from linkage import prior as prior_mod
from linkage import schema
from linkage.extract import LocalStubExtractor
from linkage.normalise import normalise_all
from linkage.retrieve import HashingTfidfEmbedder, embed_corpus
from linkage.score import fit_cross_type_repeat_rate

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "final"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.upload_model", description=__doc__.splitlines()[0])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--model-prefix", default="model")
    ap.add_argument("--vectors-prefix", default="vectors")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--weights", type=Path, default=Path(__file__).resolve().parent.parent / "weights.json")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    import pandas as pd

    cfg = config_mod.load(args.config_dir)
    print("normalising corpus...")
    cases = normalise_all(cfg, args.data_dir, LocalStubExtractor())
    truth = pd.read_parquet(args.data_dir / "truth.parquet")

    print("computing frequencies and priors...")
    freqs = frequencies_mod.compute(cases)
    priors_raw = prior_mod.compute(truth[truth["ingested"]])
    # prior.compute()'s same_type dict is keyed by crime_type already (JSON-safe);
    # cross_type/all_property are scalars — reshape to match handlers/link.py's expectations.
    priors = {"same_type": priors_raw["same_type"], "cross_type": priors_raw["cross_type"]}

    s3 = boto3.client("s3")
    s3.put_object(Bucket=args.bucket, Key=f"{args.model_prefix}/frequencies.json",
                  Body=json.dumps(freqs).encode("utf-8"), ContentType="application/json")
    s3.put_object(Bucket=args.bucket, Key=f"{args.model_prefix}/priors.json",
                  Body=json.dumps(priors).encode("utf-8"), ContentType="application/json")
    print(f"uploaded s3://{args.bucket}/{args.model_prefix}/frequencies.json")
    print(f"uploaded s3://{args.bucket}/{args.model_prefix}/priors.json")

    if args.weights.exists():
        s3.upload_file(str(args.weights), args.bucket, f"{args.model_prefix}/weights.json")
        print(f"uploaded s3://{args.bucket}/{args.model_prefix}/weights.json")
    else:
        print(f"WARNING: {args.weights} not found — run `python -m linkage.train` first")

    print("embedding narratives for the retrieval index (local tfidf_original stub)...")
    ids = [c["case_id"] for c in cases]
    texts = [c["narrative_text"] for c in cases]
    vectors, _ = embed_corpus(ids, texts, HashingTfidfEmbedder(dim=512))
    buf = io.BytesIO()
    np.save(buf, vectors)
    s3.put_object(Bucket=args.bucket, Key=f"{args.vectors_prefix}/index.npy", Body=buf.getvalue())
    s3.put_object(Bucket=args.bucket, Key=f"{args.vectors_prefix}/case_ids.json",
                  Body=json.dumps(ids).encode("utf-8"), ContentType="application/json")
    print(f"uploaded s3://{args.bucket}/{args.vectors_prefix}/index.npy ({vectors.shape})")
    print(f"uploaded s3://{args.bucket}/{args.vectors_prefix}/case_ids.json ({len(ids)} ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
