"""Relocation check with no sibling checkout or network. Add --infer for CUDA inference."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT))
    from gem_server.models import required_paths
    run_inference = "--infer" in sys.argv
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="standalone-check-", dir=data) as temp:
        moved = Path(temp) / "server"
        moved.mkdir()
        for name in ("gem_server", "vendor"):
            shutil.copytree(ROOT / name, moved / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name in ("config.example.json", "requirements.txt", "start.bat", "start.sh"):
            shutil.copy2(ROOT / name, moved / name)
        for source in required_paths(ROOT / "models"):
            destination = moved / "models" / source.relative_to(ROOT / "models")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Hard links have independent directory entries and avoid copying gigabytes.
            # No junctions/symlinks to the old source or model directory are used.
            os.link(source, destination)
        script = r'''
import sys, pathlib, os, runpy
root = pathlib.Path(sys.argv[1]).resolve()
original = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root))
def guard(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("Standalone validation disallows network access")
    if event in ("open", "os.listdir", "os.scandir") and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = pathlib.Path(os.fsdecode(args[0])).resolve()
        for forbidden in (original.parent / "GEM-X", original / "vendor", original / "models"):
            if p == forbidden or forbidden in p.parents:
                raise RuntimeError("External source access rejected: " + str(p))
sys.addaudithook(guard)
from gem_server.runtime import prepare_source
from gem_server.paths import SOURCE, DINOV3_SOURCE
prepare_source(root / "models")
import gem.gem, soma, sam_3d_body
for module in (gem.gem, soma, sam_3d_body):
    path = pathlib.Path(module.__file__).resolve()
    assert root in path.parents, path
    print("LOCAL_MODULE", path.relative_to(root), flush=True)
import torch
model = torch.hub.load(str(DINOV3_SOURCE), "dinov3_vits16", source="local", pretrained=False)
del model
print("LOCAL_DINO_OK", flush=True)
sys.argv = ["gem_server", "doctor"]
try:
    runpy.run_module("gem_server", run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0, exc
print("RELOCATED_DOCTOR_OK", flush=True)
if (root / "sample.mp4").exists():
    sys.argv = ["gem_server", "infer", "--video", str(root / "sample.mp4"), "--max-frames", "8"]
    runpy.run_module("gem_server", run_name="__main__")
    import json
    results = list((root / "data/manual").glob("*/result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_text("utf-8"))
    assert result["frames"] == 8 and result["joints"] == 78, result
    assert result["metadata"]["options"]["image_features"] is True
    print("RELOCATED_INFERENCE_OK", json.dumps(result, ensure_ascii=False), flush=True)
'''
        if run_inference:
            inputs = list((ROOT / "data/jobs").glob("*/input.mp4"))
            if not inputs:
                raise RuntimeError("No local validation clip found in data/jobs")
            shutil.copy2(inputs[0], moved / "sample.mp4")
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONUTF8"] = "1"
        subprocess.run([sys.executable, "-I", "-u", "-X", "utf8", "-c", script, str(moved), str(ROOT)],
                       cwd=moved, env=env, check=True)
        print("STANDALONE_CHECK_OK", flush=True)


if __name__ == "__main__":
    main()
