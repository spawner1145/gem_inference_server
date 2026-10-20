import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from gem_server.app import create_app
from gem_server.config import Config


class APITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.app = create_app(self.root, Config(max_upload_mb=1))
        self.context = TestClient(self.app)
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_health_missing_models_and_bad_job(self):
        self.assertEqual(self.client.get("/health").json()["service"], "gem-inference-server")
        self.assertFalse(self.client.get("/v1/models").json()["ready"])
        self.assertEqual(self.client.post("/v1/jobs", files={"video": ("x.mp4", b"x")}).status_code, 409)
        self.assertEqual(self.client.get("/v1/jobs/not-a-uuid").status_code, 404)

    @patch("gem_server.app.models.status", return_value={"ready": True, "files": []})
    def test_upload_validation_and_cleanup(self, _):
        for name, data, code in [("x.txt", b"x", 400), ("x.mp4", b"", 400), ("x.mp4", b"x" * 1048577, 413)]:
            result = self.client.post("/v1/jobs", files={"video": (name, data)})
            self.assertEqual(result.status_code, code, result.text)
        self.assertEqual(list((self.root / "data/uploads").glob("*")), [])
        result = self.client.post("/v1/jobs", files={"video": ("x.mp4", b"x")}, data={"options": '{"max_frames":700}'})
        self.assertEqual(result.status_code, 422)

    @patch("gem_server.app.models.status", return_value={"ready": True, "files": []})
    def test_queued_cancellation_and_artifact_restriction(self, _):
        with patch.object(self.app.state.manager.pool, "submit"):
            result = self.client.post("/v1/jobs", files={"video": ("中文.mp4", b"video")})
            self.assertEqual(result.status_code, 202, result.text)
            job = result.json()
            self.assertEqual(job["status"], "queued")
            self.assertEqual(job["filename"], "中文.mp4")
            spec = self.app.state.manager.path(job["id"]) / "spec.json"
            self.assertNotIn('"source"', spec.read_text("utf-8"))
            base = "/v1/jobs/" + job["id"]
            self.assertEqual(self.client.get(base + "/files/spec.json").status_code, 404)
            self.assertEqual(self.client.post(base + "/cancel").json()["status"], "cancelled")
            self.assertEqual(self.client.post(base + "/cancel").json()["status"], "cancelled")

    def test_config_persistence(self):
        cfg = self.client.get("/v1/config").json()
        self.assertNotIn("source_dir", cfg)
        html = self.client.get("/").text
        self.assertNotIn("source_dir", html)
        self.assertNotIn("../GEM-X", html)
        cfg["port"] = 6799
        self.assertEqual(self.client.put("/v1/config", json=cfg).status_code, 200)
        self.assertEqual(self.client.get("/v1/config").json()["port"], 6799)
        self.assertTrue((self.root / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
