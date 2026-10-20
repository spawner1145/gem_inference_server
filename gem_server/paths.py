"""All inference source ships with this server, independent of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/gem_x"
DINOV3_SOURCE = ROOT / "vendor/dinov3"


def check_bundled_source():
    required = [SOURCE / "gem/gem.py", SOURCE / "configs/demo_soma.yaml",
                SOURCE / "third_party/soma/soma/soma.py",
                SOURCE / "third_party/sam-3d-body/sam_3d_body/build_models.py",
                DINOV3_SOURCE / "hubconf.py"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("服务端内置源码不完整，请重新解压完整服务端：" + ", ".join(missing))
    return SOURCE
