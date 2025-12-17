#!/usr/bin/env python3
"""
Generate and persist an API key for the HPDB API server.

This updates `keys.ini` (or $API_KEYS_PATH) by:
  - removing the default key "YOU-REALLY-GOTTA-CHANGE-THIS" if present
  - adding a newly generated key as enabled
  - recording a human-readable purpose for the key
"""

from __future__ import annotations

import argparse
import configparser
import os
import secrets
import sys
from pathlib import Path


DEFAULT_KEY = "YOU-REALLY-GOTTA-CHANGE-THIS"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an API key and store it in keys.ini.")
    parser.add_argument(
        "--comment",
        help="Reason/purpose for this key (if omitted, you will be prompted interactively).",
    )
    args = parser.parse_args()

    comment = args.comment
    if not comment:
        if not sys.stdin.isatty():
            raise SystemExit("Missing --comment (stdin is not interactive).")
        comment = input("Purpose for this API key: ").strip()
    comment = (comment or "").strip()
    if not comment:
        raise SystemExit("API key purpose cannot be empty.")

    keys_path = Path(os.environ.get("API_KEYS_PATH", "keys.ini"))
    cp = configparser.ConfigParser()
    cp.optionxform = str  # preserve case for API keys
    if keys_path.exists():
        cp.read(keys_path)

    if "api_keys" not in cp:
        cp["api_keys"] = {}
    if "api_key_comments" not in cp:
        cp["api_key_comments"] = {}

    if DEFAULT_KEY in cp["api_keys"]:
        del cp["api_keys"][DEFAULT_KEY]
    if DEFAULT_KEY.lower() in cp["api_keys"]:
        del cp["api_keys"][DEFAULT_KEY.lower()]
    if DEFAULT_KEY in cp["api_key_comments"]:
        del cp["api_key_comments"][DEFAULT_KEY]
    if DEFAULT_KEY.lower() in cp["api_key_comments"]:
        del cp["api_key_comments"][DEFAULT_KEY.lower()]

    # Human-friendly but high entropy.
    new_key = secrets.token_urlsafe(32)
    cp["api_keys"][new_key] = "enabled"
    cp["api_key_comments"][new_key] = comment

    keys_path.parent.mkdir(parents=True, exist_ok=True)
    with keys_path.open("w", encoding="utf-8") as f:
        cp.write(f)

    print(new_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
