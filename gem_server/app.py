from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import ValidationError
from . import __version__
from .config import ROOT, Config, Options, load_config, save_config
from .jobs import ARTIFACTS, JobManager
from . import models
from .paths import check_bundled_source

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".gif"}


def create_app(root=ROOT, config=None):
    root = Path(root)
    current = config or load_config()
    manager = JobManager(root)
    model_state = {"running": False, "message": "", "error": None}
    state_lock = threading.RLock()
    download_thread = None

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="GEM-X 视频动作服务", version=__version__, lifespan=lifespan)
    app.state.manager = manager

    @app.get("/")
    def index():
        return FileResponse(ROOT / "gem_server/web/index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "gem-inference-server", "version": __version__, "api_version": 1}

    @app.get("/v1/config")
    def config_get():
        return {**current.model_dump(), "models_dir": str(root / "models"), "data_dir": str(root / "data")}

    @app.put("/v1/config")
    def config_put(value: Config):
        nonlocal current
        if manager.active() or model_state["running"]:
            raise HTTPException(409, "有任务运行中，完成后再修改配置")
        save_config(value, root)
        current = value
        return {"saved": True, "restart_required": True}

    @app.get("/v1/models")
    def model_status():
        result = models.status(root / "models")
        if model_state["running"] or model_state["error"]:
            result["download"] = dict(model_state)
        return result

    @app.post("/v1/models/download", status_code=202)
    def model_download():
        nonlocal download_thread
        with state_lock:
            if model_state["running"]:
                return dict(model_state)
            if models.status(root / "models").get("download", {}).get("running"):
                raise HTTPException(409, "另一个模型下载任务正在运行")
            if manager.active():
                raise HTTPException(409, "推理任务进行中，请等待任务完成后下载")
            model_state.update(running=True, message="开始下载", error=None)
            def run():
                try:
                    models.download(root / "models", report=lambda message: model_state.update(message=message))
                except Exception as exc:
                    model_state.update(error=str(exc), message="下载失败，可再次点击继续")
                finally:
                    model_state["running"] = False
            download_thread = threading.Thread(target=run, daemon=True)
            download_thread.start()
            return dict(model_state)

    @app.post("/v1/jobs", status_code=202)
    async def submit(video: UploadFile = File(...), options: str = Form("{}")):
        if model_state["running"]:
            raise HTTPException(409, "模型正在下载，请完成后再提交")
        if not models.status(root / "models")["ready"]:
            raise HTTPException(409, "模型尚未准备好，请先在管理页面下载模型")
        try:
            check_bundled_source()
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc))
        extension = Path(video.filename or "").suffix.lower()
        if extension not in VIDEO_EXTENSIONS:
            raise HTTPException(400, "不支持的视频格式")
        try:
            merged = {**current.defaults.model_dump(), **json.loads(options)}
            opts = Options.model_validate(merged)
        except (ValueError, TypeError, ValidationError) as exc:
            raise HTTPException(422, f"推理配置无效：{exc}")
        if opts.max_frames > current.max_frames or opts.max_edge > current.max_edge:
            raise HTTPException(422, f"服务端限制：最多 {current.max_frames} 帧，最长边 {current.max_edge} 像素")
        upload_dir = root / "data/uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp = upload_dir / (uuid.uuid4().hex + extension)
        size = 0
        try:
            with tmp.open("wb") as out:
                while chunk := await video.read(1024 * 1024):
                    size += len(chunk)
                    if size > current.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, f"视频超过 {current.max_upload_mb} MB 上传限制")
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "视频文件为空")
            try:
                with state_lock:
                    if model_state["running"]:
                        raise HTTPException(409, "模型正在下载")
                    return manager.submit(tmp, video.filename, opts)
            except ValueError as exc:
                raise HTTPException(429, str(exc))
        finally:
            tmp.unlink(missing_ok=True)
            await video.close()

    @app.get("/v1/jobs")
    def jobs_list():
        return manager.list()

    def get_folder(job_id):
        try:
            return manager.path(job_id)
        except KeyError:
            raise HTTPException(404, "找不到任务")

    @app.get("/v1/jobs/{job_id}")
    def job_get(job_id: str):
        get_folder(job_id)
        return manager.get(job_id)

    @app.post("/v1/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        get_folder(job_id)
        return manager.cancel(job_id)

    @app.get("/v1/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        log = get_folder(job_id) / "worker.log"
        if not log.exists():
            return "尚未开始"
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 32000))
            return f.read().decode("utf-8", errors="replace")

    @app.get("/v1/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str):
        folder = get_folder(job_id)
        if name not in ARTIFACTS or not (folder / name).is_file():
            raise HTTPException(404, "找不到文件")
        if manager.get(job_id)["status"] != "succeeded":
            raise HTTPException(409, "任务尚未完成")
        return FileResponse(folder / name, media_type=ARTIFACTS[name], filename=name)

    return app
