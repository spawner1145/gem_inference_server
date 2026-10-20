"""Official upstream assets, stored entirely below the server's models folder."""
from pathlib import Path
import shutil
import zipfile
import urllib.request
import time
import json
import hashlib
from filelock import FileLock, Timeout


GEM_FILES = {
    "gem_soma.ckpt": "inputs/pretrained",
    "vitpose.pth": "inputs/checkpoints/vitpose",
    "sam3d_body.ckpt": "inputs/checkpoints/sam-3d-body-dinov3",
    "model_config.yaml": "inputs/checkpoints/sam-3d-body-dinov3",
    "mhr_model.pt": "inputs/mhr_data",
    "scale_mean.pth": "inputs/soma_data",
    "scale_comps.pth": "inputs/soma_data",
}
SOMA_FILES = [
    "SOMA_neutral.npz", "correctives_model.pt",
    "MHR/mhr_model_lod6.pt", "MHR/base_body_lod6.obj", "MHR/SOMA_wrap_lod1.obj",
    "MHR/base_body_lod1.obj", "README.md",
]
DETECTOR_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
MANIFEST = json.loads(Path(__file__).with_name("model_manifest.json").read_text("utf-8"))


def required_paths(root):
    root = Path(root)
    return ([root / folder / name for name, folder in GEM_FILES.items()]
            + [root / "inputs/soma_assets" / name for name in SOMA_FILES]
            + [root / "inputs/checkpoints/yolox/end2end.onnx"])


def status(root):
    root = Path(root)
    files = []
    for p in required_paths(root):
        ready = p.is_file() and p.stat().st_size > 0
        expected = MANIFEST.get(p.relative_to(root).as_posix())
        if ready and expected:
            ready = p.stat().st_size == expected["size"]
        if ready:
            with p.open("rb") as f:
                ready = not f.read(128).startswith(b"version https://git-lfs.github.com/spec")
        files.append({"path": str(p.relative_to(root)), "ready": ready,
                      "bytes": p.stat().st_size if p.is_file() else 0})
    download_state = {}
    state_path = root / ".download-state.json"
    if state_path.is_file():
        try:
            download_state = json.loads(state_path.read_text("utf-8"))
        except (OSError, ValueError):
            pass
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(root / ".download.lock")
    try:
        with lock.acquire(timeout=0):
            running = False
    except Timeout:
        running = True
    return {"ready": all(f["ready"] for f in files), "files": files,
            "download": {"running": running, **download_state}}


def download(root, report=print):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root / ".download.lock", timeout=0):
        def notify(message):
            report(message)
            (root / ".download-state.json").write_text(json.dumps({"message": message}, ensure_ascii=False), "utf-8")
        _download(root, notify)


def _download(root, report):
    for name, folder in GEM_FILES.items():
        report(f"下载 nvidia/GEM-X/{name}")
        info = MANIFEST[f"{folder}/{name}"]
        fetch(f"https://huggingface.co/nvidia/GEM-X/resolve/{info['revision']}/{name}?download=true", root / folder / name, report, info)
    for name in SOMA_FILES:
        report(f"下载 nvidia/soma-x/{name}")
        info = MANIFEST[f"inputs/soma_assets/{name}"]
        fetch(f"https://huggingface.co/nvidia/soma-x/resolve/{info['revision']}/{name}?download=true", root / "inputs/soma_assets" / name, report, info)
    detector = root / "inputs/checkpoints/yolox/end2end.onnx"
    if not detector.is_file():
        report("下载官方 YOLOX 人物检测器")
        detector.parent.mkdir(parents=True, exist_ok=True)
        archive = detector.parent / "download.zip.part"
        fetch(DETECTOR_URL, archive, report)
        with zipfile.ZipFile(archive) as z:
            entry = next(n for n in z.namelist() if n.endswith("end2end.onnx"))
            with z.open(entry) as src, detector.with_suffix(".part").open("wb") as dst:
                shutil.copyfileobj(src, dst)
        detector.with_suffix(".part").replace(detector)
        archive.unlink(missing_ok=True)
    report("模型下载完成")


def fetch(url, target, report=print, expected=None):
    target = Path(target)
    if target.is_file() and target.stat().st_size > 0 and (not expected or target.stat().st_size == expected["size"]):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(4):
        try:
            if expected and part.exists() and part.stat().st_size == expected["size"]:
                with part.open("rb") as stream:
                    valid = not expected.get("sha256") or hashlib.file_digest(stream, "sha256").hexdigest() == expected["sha256"]
                if valid:
                    part.replace(target)
                    return
                part.unlink()
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "GEM-Blender-Bridge/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = offset > 0 and response.status == 206
                content_bytes = int(response.headers.get("Content-Length", 0))
                if not resume:
                    offset = 0
                total = offset + content_bytes
                done, last = offset, time.monotonic()
                with part.open("ab" if resume else "wb") as out:
                    while chunk := response.read(1024 * 1024):
                        out.write(chunk)
                        done += len(chunk)
                        if time.monotonic() - last > 10:
                            report(f"{target.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
                            last = time.monotonic()
                if total and done != total:
                    raise IOError(f"Incomplete download: {done}/{total}")
            if expected:
                if part.stat().st_size != expected["size"]:
                    raise IOError(f"Size mismatch: {target.name}")
                if expected.get("sha256"):
                    with part.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != expected["sha256"]:
                        part.unlink()
                        raise IOError(f"SHA256 mismatch: {target.name}")
            part.replace(target)
            return
        except Exception as exc:
            if attempt == 3:
                raise
            report(f"{target.name} 下载中断（{exc}），正在断点续传（{attempt + 1}/3）")


if __name__ == "__main__":
    import sys
    download(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "models")
