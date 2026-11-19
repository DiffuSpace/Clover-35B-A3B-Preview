"""Check that the distributable package depends only on public libraries/code."""

import ast
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
forbidden = {"megatron", "slime", "sglang", "ray", "_block_diffusion_cpp"}
for path in (root / "clover_sft").glob("*.py"):
    source = path.read_text()
    ast.parse(source, filename=str(path), feature_version=(3, 10))
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [entry.name for entry in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name.split(".")[0] in forbidden:
                raise SystemExit(f"Private/out-of-scope dependency in {path.name}: {name}")
    if re.search(r"/(?:root|home|Users)/", source):
        raise SystemExit(f"Machine-specific source path in {path.name}")
upstream = json.loads((root / "UPSTREAM.json").read_text())
for name, expected in upstream["unmodified_files"].items():
    actual = hashlib.sha256((root / "clover_sft" / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Published inference source was unexpectedly changed: {name}")
for name, patch in upstream.get("patched_files", {}).items():
    actual = hashlib.sha256((root / "clover_sft" / name).read_bytes()).hexdigest()
    if actual != patch["local_sha256"]:
        raise SystemExit(f"Patched inference source differs from its recorded hash: {name}")
print("Public package scope, recorded source hashes and Python 3.10 syntax passed.")
