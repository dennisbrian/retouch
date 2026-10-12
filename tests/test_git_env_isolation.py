"""The test suite must never write to the repository a git hook is running for.

Inside a hook git exports GIT_DIR / GIT_INDEX_FILE, so a test that runs
``git init`` or ``git commit`` in a temp folder would hit the real repository
instead. These tests point those variables at a decoy repository and check
that it comes through untouched.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import GIT_HOOK_ENV_VARS, strip_git_hook_env

REPO = Path(__file__).resolve().parents[1]

PROBE = '''
import subprocess

def test_probe(tmp_path):
    for cmd in (["git", "init", "-q"], ["git", "add", "."],
                ["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", "commit", "-qm", "base"]):
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(cmd, cwd=tmp_path, check=True)
    top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=tmp_path, text=True).strip()
    assert top.endswith(tmp_path.name)
'''


def _git(args, cwd, env=None):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, env=env).strip()


def _clean_env():
    return {k: v for k, v in os.environ.items() if k not in GIT_HOOK_ENV_VARS}


@pytest.fixture
def decoy(tmp_path):
    """A small repository standing in for the one a push hook runs for."""
    root = tmp_path / "decoy"
    root.mkdir()
    env = _clean_env()
    (root / "keep.txt").write_text("real work\n")
    _git(["init", "-q"], root, env)
    _git(["add", "."], root, env)
    _git(["-c", "user.name=D", "-c", "user.email=d@example.invalid", "commit", "-qm", "decoy"], root, env)
    return root


def _state(root):
    env = _clean_env()
    return (_git(["rev-parse", "HEAD"], root, env), _git(["config", "core.bare"], root, env),
            _git(["status", "--porcelain"], root, env), _git(["ls-files"], root, env))


def _hook_env(decoy):
    env = _clean_env()
    env["GIT_DIR"] = str(decoy / ".git")
    env["GIT_INDEX_FILE"] = str(decoy / ".git" / "index")
    env["GIT_PREFIX"] = ""
    return env


def test_strip_removes_only_git_hook_variables():
    environ = {"GIT_DIR": "/x/.git", "GIT_INDEX_FILE": "/x/.git/index", "GIT_EDITOR": "true", "PATH": "/bin"}
    assert sorted(strip_git_hook_env(environ)) == ["GIT_DIR", "GIT_INDEX_FILE"]
    assert environ == {"GIT_EDITOR": "true", "PATH": "/bin"}


def test_list_covers_what_git_exports_to_hooks():
    exported = set(_git(["rev-parse", "--local-env-vars"], REPO, _clean_env()).split())
    assert exported <= set(GIT_HOOK_ENV_VARS)


def test_this_process_has_no_git_hook_variables():
    assert [name for name in GIT_HOOK_ENV_VARS if name in os.environ] == []


def test_decoy_would_be_hit_without_the_guard(decoy, tmp_path):
    """Shows the hazard is real: the same commands with the variables left in do move the decoy."""
    before = _state(decoy)
    work = tmp_path / "work"
    work.mkdir()
    (work / "f.txt").write_text("x")
    env = _hook_env(decoy)
    subprocess.run(["git", "add", "."], cwd=work, check=True, env=env)
    subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.invalid", "commit", "-qm", "base"],
                   cwd=work, check=True, env=env)
    assert _state(decoy)[0] != before[0]


def test_suite_under_hook_env_leaves_the_repository_alone(decoy):
    """The real offender (tests/test_experiment_lab.py) run by pytest directly, as a hook would see it."""
    before = _state(decoy)
    proc = subprocess.run(
        [sys.executable, "-s", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_experiment_lab.py::test_real_detached_worktree_isolation"],
        cwd=REPO, env=_hook_env(decoy), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert _state(decoy) == before


@pytest.mark.skipif(sys.platform == "win32", reason="bash runner")
def test_test_runner_script_drops_hook_env(decoy, tmp_path):
    """scripts/dev/test clears the variables itself, also for tests outside tests/ (no conftest)."""
    probe = tmp_path / "probe" / "test_probe.py"
    probe.parent.mkdir()
    probe.write_text(PROBE)
    env = _hook_env(decoy)
    env["RETOUCH_PYTHON"] = sys.executable
    before = _state(decoy)
    proc = subprocess.run(["bash", str(REPO / "scripts" / "dev" / "test"), "-q", "--rootdir", str(probe.parent),
                           "-c", os.devnull, str(probe)],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert _state(decoy) == before
