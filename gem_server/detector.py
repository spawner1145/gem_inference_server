"""CUDA-only person detection; fail clearly instead of silently using CPU."""
from pathlib import Path


def check_cuda_runtime():
    import torch  # Load the installed Torch/cuDNN runtime before ONNX Runtime.
    import onnxruntime as ort

    if not torch.cuda.is_available():
        raise RuntimeError("人物检测需要 NVIDIA CUDA，请先安装 CUDA 版 PyTorch。")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "当前 ONNX Runtime 没有 CUDA 后端。请在服务端使用的 Python 环境中卸载 "
            "onnxruntime，再安装 requirements.txt 中的 onnxruntime-gpu；不要同时安装 CPU/GPU 两个包。"
        )
    # Official pip runtime libraries also support a CUDA-13 Torch environment
    # alongside the CUDA-12 ONNX Runtime wheel without changing Torch itself.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    return ort


def create_cuda_detector(model_path):
    check_cuda_runtime()
    from gem.utils.yolox_detector import YOLOXDetector

    detector = YOLOXDetector(onnx_path=str(Path(model_path)), device="cuda")
    if "CUDAExecutionProvider" not in detector.sess.get_providers():
        raise RuntimeError(
            "ONNX Runtime CUDA 初始化失败，已停止任务以避免回退到缓慢的 CPU 检测。"
            "请检查 worker.log 中的 CUDA/cuDNN DLL 错误并重新安装 requirements.txt。"
        )
    detector.sess.disable_fallback()
    return detector
