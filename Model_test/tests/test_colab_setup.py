"""Credential routing tests; no real credentials or network are used."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

SCRIPT = Path(__file__).resolve().parents[1] / "SmolLM2-360M-Instruct" / "model.py"
spec = importlib.util.spec_from_file_location("colab_model", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ColabSetupTests(unittest.TestCase):
    def test_setup_authenticates_both_services_and_returns_no_credentials(self):
        hub = ModuleType("huggingface_hub")
        hub.login = Mock()
        utils = ModuleType("huggingface_hub.utils")
        utils.logging = SimpleNamespace(set_verbosity_error=Mock())
        args = SimpleNamespace(no_install=False, github_repo="owner/repo", github_branch="main", repo_dir=Path("/tmp/test-checkout"))
        with patch.dict("sys.modules", {"huggingface_hub": hub, "huggingface_hub.utils": utils}):
            with patch.object(module, "read_secret", side_effect=["test-hf", "test-github"]):
                with patch.object(module, "ensure_dependencies") as install:
                    with patch.object(module, "github_json", side_effect=[{"login": "example"}, {"full_name": "owner/repo"}]) as api:
                        with patch.object(module, "checkout_repository", return_value="a" * 40) as checkout:
                            info = module.prepare_runtime(args)
        install.assert_called_once()
        hub.login.assert_called_once_with(token="test-hf", add_to_git_credential=False)
        self.assertEqual(api.call_count, 2)
        checkout.assert_called_once_with("owner/repo", "main", Path("/tmp/test-checkout"), "test-github")
        self.assertEqual(info["user"], "example")
        self.assertNotIn("test-hf", json.dumps(info))
        self.assertNotIn("test-github", json.dumps(info))

    def test_secret_is_read_from_environment_in_a_subprocess(self):
        with patch.dict(os.environ, {"HF_TOKEN": " test-hf-token "}, clear=True):
            self.assertEqual(module.read_secret("HF_TOKEN"), "test-hf-token")

    def test_github_alias_is_supported(self):
        with patch.dict(os.environ, {"GH_TOKEN": "test-github-token"}, clear=True):
            self.assertEqual(module.read_secret("GITHUB_TOKEN", "GH_TOKEN"), "test-github-token")

    def test_missing_secret_has_actionable_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(module.SetupError, "preparation cell"):
                module.read_secret("HF_TOKEN")

    def test_github_token_goes_in_header_not_url(self):
        def respond(request, timeout):
            self.assertEqual(request.full_url, "https://api.github.com/user")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-only-token")
            return io.BytesIO(json.dumps({"login": "example"}).encode())
        with patch.object(module.urllib.request, "urlopen", side_effect=respond):
            self.assertEqual(module.github_json("/user", "test-only-token"), {"login": "example"})

    def test_github_error_does_not_include_token_or_response_body(self):
        error = urllib.error.HTTPError("https://api.github.com/user", 403, "test-only-token", {}, None)
        with patch.object(module.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(module.SetupError) as caught:
                module.github_json("/user", "test-only-token")
        self.assertIn("403", str(caught.exception))
        self.assertNotIn("test-only-token", str(caught.exception))

    def test_clone_uses_temporary_askpass_without_embedded_token(self):
        helper_paths = []
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            self.assertNotIn("test-only-token", " ".join(command))
            self.assertEqual(kwargs["env"]["GITHUB_TOKEN"], "test-only-token")
            helper = Path(kwargs["env"]["GIT_ASKPASS"])
            helper_paths.append(helper)
            self.assertNotIn("test-only-token", helper.read_text())
            self.assertEqual(helper.stat().st_mode & 0o777, 0o700)
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40, stderr="")

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(module.shutil, "which", return_value="/usr/bin/git"):
                with patch.object(module.subprocess, "run", side_effect=run):
                    commit = module.checkout_repository("owner/repo", "main", Path(directory) / "repo", "test-only-token")
        self.assertEqual(commit, "a" * 40)
        self.assertIn("clone", commands[0])
        self.assertTrue(all(not p.exists() for p in helper_paths))

    def test_existing_non_repository_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            keep = Path(directory) / "keep.txt"
            keep.write_text("keep me")
            with patch.object(module.shutil, "which", return_value="/usr/bin/git"):
                with self.assertRaisesRegex(module.SetupError, "not a Git checkout"):
                    module.checkout_repository("owner/repo", "main", Path(directory), "test-only-token")
            self.assertEqual(keep.read_text(), "keep me")

    def test_existing_checkout_is_checked_without_pull_or_reset(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            output = "https://github.com/owner/repo.git" if "get-url" in command else "b" * 40
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".git").mkdir()
            with patch.object(module.shutil, "which", return_value="/usr/bin/git"):
                with patch.object(module.subprocess, "run", side_effect=run):
                    module.checkout_repository("owner/repo", "main", Path(directory), "test-only-token")
        self.assertTrue(any("ls-remote" in command for command in calls))
        self.assertFalse(any(action in command for command in calls for action in ("pull", "reset", "clone")))


if __name__ == "__main__":
    unittest.main()
