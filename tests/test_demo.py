import json
import subprocess
import sys
from pathlib import Path


def test_demo_mock(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    cmd = [sys.executable, "main.py", "--demo", "--mock", "--artifacts-dir", str(artifacts)]
    result = subprocess.run(cmd, check=True, capture_output=True)
    assert (artifacts / "thresholds.json").exists()
    data = json.loads((artifacts / "thresholds.json").read_text())
    assert isinstance(data, dict)