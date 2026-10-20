import json
from pathlib import Path
from pydantic import BaseModel, Field
from .paths import ROOT


class Options(BaseModel):
    mode: str = Field(default="regression", pattern="^(regression|ddim)$")
    image_features: bool = True
    batch_size: int = Field(default=1, ge=1, le=8)
    max_frames: int = Field(default=300, ge=4, le=1800)
    target_fps: int = Field(default=30, ge=1, le=60)
    max_edge: int = Field(default=720, ge=256, le=1920)
    seed: int = Field(default=42, ge=0, le=2147483647)
    postprocess: bool = True


class Config(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=6700, ge=1024, le=65535)
    max_upload_mb: int = Field(default=512, ge=1, le=4096)
    max_frames: int = Field(default=600, ge=4, le=1800)
    max_edge: int = Field(default=960, ge=256, le=1920)
    defaults: Options = Field(default_factory=Options)

def load_config():
    p = ROOT / "config.json"
    return Config.model_validate_json(p.read_text("utf-8")) if p.exists() else Config()


def save_config(config, root=ROOT):
    p = Path(root) / "config.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(config.model_dump(), ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(p)
