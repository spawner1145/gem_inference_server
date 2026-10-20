import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from gem_server.models import fetch, GEM_FILES, SOMA_FILES, MANIFEST


class Response(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))}


class DownloadTests(unittest.TestCase):
    def test_inventory_matches_pinned_manifest(self):
        expected = {f"{folder}/{name}" for name, folder in GEM_FILES.items()}
        expected.update("inputs/soma_assets/" + name for name in SOMA_FILES)
        self.assertEqual(expected, set(MANIFEST))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "model.bin"
        self.data = b"test official model contents"
        self.expected = {"size": len(self.data), "sha256": hashlib.sha256(self.data).hexdigest()}

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_download_checks_manifest(self):
        with patch("urllib.request.urlopen", return_value=Response(self.data)):
            fetch("https://example.invalid/model", self.path, expected=self.expected)
        self.assertEqual(self.path.read_bytes(), self.data)
        self.assertFalse(self.path.with_suffix(".bin.part").exists())

    def test_range_resume(self):
        part = self.path.with_suffix(".bin.part")
        part.write_bytes(self.data[:5])
        with patch("urllib.request.urlopen", return_value=Response(self.data[5:], 206)) as request:
            fetch("https://example.invalid/model", self.path, expected=self.expected)
        self.assertEqual(request.call_args.args[0].headers["Range"], "bytes=5-")
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_complete_partial_recovers_without_range_416(self):
        self.path.with_suffix(".bin.part").write_bytes(self.data)
        with patch("urllib.request.urlopen") as request:
            fetch("https://example.invalid/model", self.path, expected=self.expected)
        request.assert_not_called()
        self.assertEqual(self.path.read_bytes(), self.data)


if __name__ == "__main__":
    unittest.main()
