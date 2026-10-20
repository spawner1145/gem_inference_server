import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .config import ROOT, Options

ARTIFACTS = {"motion.json": "application/json", "motion.bvh": "application/octet-stream",
             "soma_params.npz": "application/octet-stream", "input.mp4": "video/mp4", "result.json": "application/json"}


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


class JobManager:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.folder = self.root / "data/jobs"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gem")
        self.processes = {}
        for p in self.folder.glob("*/job.json"):
            data = json.loads(p.read_text("utf-8"))
            if data["status"] in ("queued", "running"):
                data.update(status="failed", error="服务已重启，任务中断，请重新提交")
                write_json(p, data)

    def path(self, job_id):
        try:
            if uuid.UUID(job_id).hex != job_id:
                raise ValueError()
        except ValueError:
            raise KeyError("Invalid job ID")
        p = self.folder / job_id
        if not (p / "job.json").exists():
            raise KeyError("Job not found")
        return p

    def get(self, job_id):
        with self.lock:
            p = self.path(job_id)
            job = json.loads((p / "job.json").read_text("utf-8"))
            progress = p / "progress.json"
            if progress.exists() and job["status"] == "running":
                try:
                    job.update(json.loads(progress.read_text("utf-8")))
                except (OSError, ValueError):
                    pass
            if job["status"] == "succeeded":
                job["result"] = json.loads((p / "result.json").read_text("utf-8"))
                job["artifacts"] = {name: f"/v1/jobs/{job_id}/files/{name}" for name in ARTIFACTS if (p / name).exists()}
            return job

    def list(self):
        paths = sorted(self.folder.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
        return [self.get(p.parent.name) for p in paths]

    def active(self):
        return [j for j in self.list() if j["status"] in ("queued", "running")]

    def submit(self, video, filename, options):
        with self.lock:
            if len(self.active()) >= 8:
                raise ValueError("任务队列已满（最多 8 个），请等待现有任务完成")
            job_id = uuid.uuid4().hex
            p = self.folder / job_id
            p.mkdir()
            extension = Path(filename).suffix.lower()
            upload = p / ("upload" + extension)
            Path(video).replace(upload)
            job = {"id": job_id, "filename": Path(filename).name, "status": "queued", "stage": "排队中", "progress": 0,
                   "created_at": datetime.now(timezone.utc).isoformat(), "options": options.model_dump(), "error": None}
            write_json(p / "job.json", job)
            spec = {"folder": str(p), "video": str(upload), "models": str(self.root / "models"),
                    "options": options.model_dump()}
            write_json(p / "spec.json", spec)
            self.pool.submit(self._run, job_id)
            return self.get(job_id)

    def update(self, job_id, **fields):
        p = self.path(job_id) / "job.json"
        job = json.loads(p.read_text("utf-8"))
        job.update(fields)
        write_json(p, job)

    def _run(self, job_id):
        p = self.path(job_id)
        try:
            with self.lock:
                if self.get(job_id)["status"] == "cancelled":
                    return
                self.update(job_id, status="running", stage="加载推理环境")
                env = os.environ.copy()
                env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
                env["PYTHONPATH"] = str(self.root) + os.pathsep + env.get("PYTHONPATH", "")
                with (p / "worker.log").open("w", encoding="utf-8") as log:
                    proc = subprocess.Popen([sys.executable, "-m", "gem_server.runtime", str(p / "spec.json")],
                                            cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self.processes[job_id] = proc
            code = proc.wait()
            with self.lock:
                if self.get(job_id)["status"] == "cancelled":
                    return
                if code == 0 and (p / "motion.json").exists() and (p / "result.json").exists():
                    self.update(job_id, status="succeeded", stage="完成", progress=1.)
                else:
                    error_path = p / "error.txt"
                    error = error_path.read_text("utf-8") if error_path.exists() else f"推理进程退出（{code}），查看日志了解原因"
                    self.update(job_id, status="failed", stage="推理失败", error=error[-12000:])
        except Exception as exc:
            with self.lock:
                self.update(job_id, status="failed", error=str(exc))
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def cancel(self, job_id):
        with self.lock:
            job = self.get(job_id)
            if job["status"] in ("queued", "running"):
                self.update(job_id, status="cancelled", stage="已取消")
                proc = self.processes.get(job_id)
                if proc and proc.poll() is None:
                    proc.terminate()
            return self.get(job_id)

    def close(self):
        for job in self.active():
            self.cancel(job["id"])
        self.pool.shutdown(wait=True, cancel_futures=True)
