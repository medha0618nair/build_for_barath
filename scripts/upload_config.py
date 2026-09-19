"""One-time (and re-run-on-config-change) deploy step: converts
config/*.yaml to JSON and uploads to S3, so Lambda handlers can load config
with json.loads (stdlib) instead of pulling PyYAML into the deployment
package (handlers/config_loader.py; CLAUDE.md rule 3, "Lambda deps are
numpy only").

    python -m scripts.upload_config --bucket crime-linkage-dev-artifacts

Run this, and scripts/upload_model.py, after `sam deploy` and before the
pipeline's first Step Functions execution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3

from linkage import config as config_mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.upload_config", description=__doc__.splitlines()[0])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--prefix", default="config")
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load(args.config_dir)
    readiness = config_mod.check(cfg)
    if readiness.errors:
        print("config has errors — run `python -m linkage.config -v`")
        return 1

    s3 = boto3.client("s3")
    for name, data in cfg.items():
        key = f"{args.prefix}/{name}.json"
        s3.put_object(Bucket=args.bucket, Key=key, Body=json.dumps(data).encode("utf-8"),
                      ContentType="application/json")
        print(f"uploaded s3://{args.bucket}/{key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
