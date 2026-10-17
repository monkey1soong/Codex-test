"""Check the Colab Secrets-to-subprocess bridge without a live Colab session."""
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class ColabNotebookTests(unittest.TestCase):
    def test_preparation_passes_secrets_and_downloads_entry_point(self):
        notebook = Path(__file__).resolve().parents[1] / "SmolLM2-360M-Instruct" / "colab.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        setup = "".join(next(c for c in cells if c["cell_type"] == "code")["source"])
        run_cells = ["".join(c["source"]).strip() for c in cells if c["cell_type"] == "code"]
        self.assertIn("!python model.py", run_cells)
        colab = ModuleType("google.colab")
        colab.userdata = SimpleNamespace(get=lambda name: {"HF_TOKEN": "fake-hf", "GITHUB_TOKEN": "fake-github"}[name])
        google = ModuleType("google")
        google.colab = colab

        def download(request, timeout):
            self.assertEqual(request.get_header("Authorization"), "Bearer fake-github")
            self.assertNotIn("fake-github", request.full_url)
            self.assertIn("/contents/Model_test/SmolLM2-360M-Instruct/model.py?ref=main", request.full_url)
            return io.BytesIO(b"print('downloaded entry point')\n")

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.py"
            with patch.dict("sys.modules", {"google": google, "google.colab": colab}):
                with patch.dict(os.environ, {}, clear=True):
                    with patch("pathlib.Path", return_value=target), patch("os.chdir") as chdir:
                        with patch("urllib.request.urlopen", side_effect=download):
                            exec(compile(setup, str(notebook), "exec"), {})
                        self.assertEqual(os.environ["HF_TOKEN"], "fake-hf")
                        self.assertEqual(os.environ["GITHUB_TOKEN"], "fake-github")
                        chdir.assert_called_once_with("/content")
            self.assertEqual(target.read_text(), "print('downloaded entry point')\n")


if __name__ == "__main__":
    unittest.main()
