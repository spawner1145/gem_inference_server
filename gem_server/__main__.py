import argparse
import json
from pathlib import Path
from .config import ROOT, load_config, save_config, Options


def main():
    parser = argparse.ArgumentParser(description="GEM-X 视频动作服务")
    parser.add_argument("command", choices=["serve", "download", "doctor", "infer"], nargs="?", default="serve")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--video")
    parser.add_argument("--max-frames", type=int, default=60)
    args = parser.parse_args()
    cfg = load_config()
    if args.command == "download":
        from .models import download
        download(ROOT / "models")
    elif args.command == "doctor":
        from .runtime import prepare_source
        from .models import status
        prepare_source(ROOT / "models")
        import torch
        import importlib
        print("Python/CUDA:", torch.__version__, torch.cuda.is_available())
        failed = []
        for name in ["fastapi", "av", "cv2", "soma", "gem.gem", "sam_3d_body", "gem.utils.vitpose_extractor"]:
            try:
                importlib.import_module(name)
                print("OK", name)
            except Exception as exc:
                failed.append(name)
                print("FAIL", name, repr(exc))
        try:
            from .detector import create_cuda_detector
            detector = create_cuda_detector(ROOT / "models/inputs/checkpoints/yolox/end2end.onnx")
            print("OK YOLOX CUDA:", detector.sess.get_providers())
            del detector
        except Exception as exc:
            failed.append("YOLOX CUDA")
            print("FAIL YOLOX CUDA", repr(exc))
        result = status(ROOT / "models")
        for f in result["files"]:
            print("OK" if f["ready"] else "MISSING", f["path"])
        raise SystemExit(0 if not failed and result["ready"] and torch.cuda.is_available() else 1)
    elif args.command == "infer":
        if not args.video:
            parser.error("infer 需要 --video 路径")
        from .jobs import JobManager
        from .runtime import infer
        import uuid
        folder = ROOT / "data/manual" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        infer({"folder": str(folder), "video": str(Path(args.video).resolve()), "models": str(ROOT / "models"),
               "options": Options(max_frames=args.max_frames).model_dump()})
        print(folder)
    else:
        import uvicorn
        from .app import create_app
        if not (ROOT / "config.json").exists():
            save_config(cfg)
        print(f"GEM WebUI: http://{args.host or cfg.host}:{args.port or cfg.port}/")
        uvicorn.run(create_app(config=cfg), host=args.host or cfg.host, port=args.port or cfg.port, workers=1)


if __name__ == "__main__":
    main()
