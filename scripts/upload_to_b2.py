#!/usr/bin/env python3
"""
Upload .npz census files to Backblaze B2.

Usage:
    python upload_to_b2.py <local_dir> <remote_prefix>

Example:
    python upload_to_b2.py /workspace/outputs/tmax-2b-census tmax-2b/census/

Env vars:
    B2_APPLICATION_KEY_ID
    B2_APPLICATION_KEY
    B2_BUCKET_NAME (default: atlas-pat)
"""
import os
import sys
from pathlib import Path

from b2sdk.v2 import InMemoryAccountInfo, B2Api


def get_b2_api():
    """Initialize B2 API client from env vars."""
    key_id = os.environ.get("B2_APPLICATION_KEY_ID")
    key = os.environ.get("B2_APPLICATION_KEY")
    bucket_name = os.environ.get("B2_BUCKET_NAME", "atlas-pat")

    if not key_id or not key:
        raise RuntimeError("Set B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY env vars")

    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", key_id, key)
    bucket = b2_api.get_bucket_by_name(bucket_name)

    return b2_api, bucket


def upload_npz_files(local_dir: Path, remote_prefix: str, bucket) -> int:
    """Upload all .npz files from local_dir to B2.

    Returns count of uploaded files.
    """
    npz_files = sorted(local_dir.glob("*.npz"))
    if not npz_files:
        print(f"[warn] no .npz files found in {local_dir}")
        return 0

    uploaded = 0
    for npz in npz_files:
        remote_path = f"{remote_prefix}{npz.name}"
        print(f"[upload] {npz.name} -> b2://{bucket.bucket_name}/{remote_path}")

        file_info = bucket.upload_local_file(str(npz), remote_path)
        uploaded += 1

        # Optional: delete local file after upload to save space
        # npz.unlink()

    return uploaded


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    local_dir = Path(sys.argv[1])
    remote_prefix = sys.argv[2].rstrip("/") + "/"

    if not local_dir.exists():
        print(f"[error] directory not found: {local_dir}")
        sys.exit(1)

    if not local_dir.is_dir():
        print(f"[error] not a directory: {local_dir}")
        sys.exit(1)

    _, bucket = get_b2_api()
    count = upload_npz_files(local_dir, remote_prefix, bucket)

    print(f"\n[done] uploaded {count} files to b2://{bucket.bucket_name}/{remote_prefix}")


if __name__ == "__main__":
    main()
