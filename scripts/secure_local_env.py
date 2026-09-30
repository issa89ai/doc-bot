"""Rotate local app authentication without printing or backing up old secrets."""
from pathlib import Path
import os
import re
import secrets
import tempfile


def secure_env(path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError("Refusing to replace a symlinked environment file.")
    original = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    remove = {
        "API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN", "AWS_SECURITY_TOKEN", "S3_BUCKET",
    }
    kept = []
    for line in original.splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if match and match.group(1) in remove:
            continue
        kept.append(line)
    # No backup: it would retain the exposed credentials.
    content = "\n".join(kept).rstrip() + "\n\nAPI_KEY=" + secrets.token_urlsafe(32) + "\nS3_BUCKET=\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".env-rotate-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    secure_env(Path(__file__).resolve().parents[1] / ".env")
    print("Local API key rotated; AWS key assignments removed; S3 disabled. No secrets displayed.")
