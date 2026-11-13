from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> int:
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            count += 1
    return count
