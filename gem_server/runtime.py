"""GPU worker. Each job runs in a child process and releases all GPU state on exit."""
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from .paths import SOURCE, check_bundled_source


def prepare_source(models):
    source, models = check_bundled_source(), Path(models).resolve()
    for path in [source, source / "third_party/soma", source / "third_party/sam-3d-body"]:
        sys.path.insert(0, str(path))
    for key, folder in {"HF_HOME": ".hf", "TORCH_HOME": ".torch", "XDG_CACHE_HOME": ".cache"}.items():
        os.environ[key] = str(models / folder)
    # Upstream uses inputs/... relative paths. Only this isolated worker changes cwd.
    os.chdir(models)


def normalize_video(video, output, options):
    import av
    import cv2
    import numpy as np
    frames = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or 30)
        fps = min(float(options["target_fps"]), source_fps)
        next_time = 0.
        for index, frame in enumerate(container.decode(stream)):
            time = float(frame.time) if frame.time is not None else index / source_fps
            if not frames:
                origin = time
            time -= origin
            if time + 0.001 < next_time:
                continue
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            ratio = min(1., options["max_edge"] / max(h, w))
            size = (max(2, int(w * ratio) // 2 * 2), max(2, int(h * ratio) // 2 * 2))
            if (w, h) != size:
                img = cv2.resize(img, size)
            frames.append(img)
            next_time = len(frames) / fps
            if len(frames) >= options["max_frames"]:
                break
    if len(frames) < 4:
        raise ValueError("视频至少需要 4 个有效帧")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("无法写入标准化视频")
    try:
        for frame in frames:
            writer.write(frame[:, :, ::-1])
    finally:
        writer.release()
    return np.stack(frames), fps


def infer(spec):
    folder = Path(spec["folder"])
    options = spec["options"]

    def progress(stage, fraction):
        p = folder / "progress.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"stage": stage, "progress": fraction}, ensure_ascii=False), "utf-8")
        tmp.replace(p)
        print(stage, flush=True)

    prepare_source(spec["models"])
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GEM-X 的官方推理需要 NVIDIA CUDA，请安装匹配显卡的 PyTorch")
    # Only the trusted official checkpoints are loaded in this subprocess.
    original_load = torch.load
    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)
    torch.load = load
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    torch.set_grad_enabled(False)

    def release():
        gc.collect()
        torch.cuda.empty_cache()

    progress("解码视频", .02)
    frames, fps = normalize_video(spec["video"], folder / "input.mp4", options)
    length, height, width = frames.shape[:3]
    from gem.utils.yolox_detector import YOLOXDetector, detect_and_track
    from gem.utils.kp2d_utils import smooth_bbx_xyxy
    from gem.utils.geo_transform import get_bbx_xys_from_xyxy, compute_cam_angvel, compute_cam_tvel
    from gem.utils.cam_utils import estimate_K, get_a_pred_cam
    progress("人物检测与跟踪", .08)
    # CPU detector saves GPU RAM; the three large neural networks run on CUDA.
    detector = YOLOXDetector(onnx_path="inputs/checkpoints/yolox/end2end.onnx", device="cpu")
    boxes, ids = detect_and_track(frames, detector)
    if (ids < 0).all():
        raise ValueError("视频中未检测到人物，请使用清晰、尽量全身可见的单人视频")
    boxes = smooth_bbx_xyxy(torch.from_numpy(boxes).float(), window=5)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, width - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, height - 1)
    bbx = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
    del detector
    release()
    progress("ViTPose：提取 77 个关键点", .2)
    from gem.utils.vitpose_extractor import VitPoseExtractor
    pose_model = VitPoseExtractor(device="cuda:0", pose_type="soma")
    keypoints = pose_model.extract(frames, bbx, batch_size=options["batch_size"], path_type="np")
    del pose_model, frames
    release()
    features = torch.zeros(length, 1024)
    pred_cam = None
    K = estimate_K(width, height).repeat(length, 1, 1)
    if options["image_features"]:
        progress("SAM-3D-Body：提取图像特征", .48)
        from gem.utils.sam3db_extractor import SAM3DBExtractor
        extractor = SAM3DBExtractor(checkpoint_path="inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt",
                                   mhr_path="inputs/mhr_data/mhr_model.pt", device="cuda:0")
        result = extractor.extract_video_features(str(folder / "input.mp4"), bbx, batch_size=options["batch_size"])
        features = result["pose_tokens"]
        pred_cam = get_a_pred_cam(result["transls"], bbx, K)
        del extractor, result
        release()
    rotations = torch.eye(3).repeat(length, 1, 1)
    translations = torch.zeros(length, 3)
    data = {"meta": [{"vid": "uploaded_video"}], "length": torch.tensor(length), "bbx_xys": bbx,
            "kp2d": keypoints[0] if isinstance(keypoints, tuple) else keypoints,
            "K_fullimg": K, "cam_angvel": compute_cam_angvel(rotations), "cam_tvel": compute_cam_tvel(translations),
            "R_w2c": rotations, "T_w2c": torch.eye(4).repeat(length, 1, 1),
            "f_imgseq": features, "noisy_pred_cam": pred_cam, "has_text": torch.tensor([False]),
            "mask": {"valid": torch.ones(length, dtype=torch.bool),
                     "has_img_mask": torch.full((length,), options["image_features"], dtype=torch.bool),
                     "has_2d_mask": torch.ones(length, dtype=torch.bool),
                     "has_cam_mask": torch.ones(length, dtype=torch.bool),
                     "has_audio_mask": torch.zeros(length, dtype=torch.bool),
                     "has_music_mask": torch.zeros(length, dtype=torch.bool)}}
    torch.save(data, folder / "features.pt")
    progress("GEM：还原三维动作", .75)
    import hydra
    from hydra import initialize_config_dir, compose
    with initialize_config_dir(version_base="1.3", config_dir=str(SOURCE / "configs")):
        cfg = compose(config_name="demo_soma", overrides=["task=test", "use_wandb=false"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model("inputs/pretrained/gem_soma.ckpt")
    model.eval().cuda()
    if options["mode"] == "ddim":
        model.pipeline.regression_only = False
    prediction = model.predict(data, static_cam=True, postproc=options["postprocess"])
    params = prediction.get("body_params_global", prediction.get("pred_body_params_global"))
    if not params:
        raise RuntimeError("GEM 未返回全局 SOMA 参数")
    params = {k: v.detach().float().cpu() for k, v in params.items()}
    np.savez_compressed(folder / "soma_params.npz", **{k: v.numpy() for k, v in params.items()}, fps=fps)
    progress("导出 SOMA 动作", .94)
    from .motion import to_motion, write_motion
    metadata = {"model": "nvidia/GEM-X/gem_soma.ckpt", "options": options,
                "camera": "static", "width": width, "height": height,
                "detected_frames": int((ids >= 0).sum())}
    motion = to_motion(params, model.body_model.soma, fps, metadata)
    write_motion(motion, folder)
    (folder / "result.json").write_text(json.dumps({"frames": length, "fps": fps, "joints": len(motion["names"]),
                                                  "metadata": metadata}, ensure_ascii=False, indent=2), "utf-8")
    progress("完成", 1.)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text("utf-8"))
    try:
        infer(spec)
    except Exception:
        error = traceback.format_exc()
        (Path(spec["folder"]) / "error.txt").write_text(error, "utf-8")
        print(error, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
