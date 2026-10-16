"""
Upload release artifacts to Hugging Face Model Hub.
Reads token exclusively from the HF_TOKEN environment variable.
Never commits or logs the token.
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload traffic sign benchmark models to Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="neitpm/traffic-sign-yolo11-benchmark",
        help="Target repository ID on Hugging Face (e.g., neitpm/traffic-sign-yolo11-benchmark)"
    )
    parser.add_argument(
        "--release_dir",
        type=str,
        default="hf_release",
        help="Local release directory containing model artifacts and metadata"
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token or not token.strip():
        print("ERROR: HF_TOKEN environment variable is not set or empty.", file=sys.stderr)
        print("Please set your Hugging Face write token before running:", file=sys.stderr)
        print("  $env:HF_TOKEN = 'hf_...'", file=sys.stderr)
        sys.exit(1)

    release_path = Path(args.release_dir).resolve()
    if not release_path.exists():
        print(f"ERROR: Release directory '{release_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to Hugging Face Hub with repository: {args.repo_id}...")
    api = HfApi(token=token.strip())

    # Create model repository if it doesn't already exist
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            exist_ok=True,
            private=False
        )
        print(f"Repository verified: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"ERROR creating/verifying repository: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading files from '{release_path}' to '{args.repo_id}'...")
    try:
        api.upload_folder(
            folder_path=str(release_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="release: publish traffic sign checkpoints, configs, and inference artifacts"
        )
        print(f"\nSUCCESS! All release artifacts uploaded successfully.")
        print(f"Repository URL: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"ERROR uploading folder to Hugging Face: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
