"""The actual merged-PR CLI works after the review router is removed."""
import importlib.util
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def announcement(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    assert not (root / ".github/scripts/review_loop.py").exists()
    spec = importlib.util.spec_from_file_location("merge_announcement", root / ".github/scripts/merge_digest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pull = {"number": 7, "merged": True, "merge_commit_sha": "abc123", "title": "Preserve the spoken line",
            "body": "Every video window now sees the full audio timeline.\nTOKEN=examplecredential12345",
            "html_url": "https://github.com/owner/forge/pull/7", "merged_at": "2026-09-05T19:00:00Z",
            "merged_by": {"login": "owner"},
            "head": {"sha": "def456", "repo": {"full_name": "owner/forge"}},
            "base": {"repo": {"full_name": "owner/forge"}}}
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": pull}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/forge")
    monkeypatch.setenv("GITHUB_TOKEN", "test-github-token")
    monkeypatch.delenv("DIGEST_DRY_RUN", raising=False)
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        if "/pulls/7/commits?" in request.full_url:
            value = [{"commit": {"author": {"name": "Fable"}}}]
        elif request.full_url.endswith("/commits/abc123"):
            value = {"parents": [{"sha": "one"}, {"sha": "two"}]}
        elif request.full_url == "https://slack.com/api/chat.postMessage":
            value = {"ok": False, "error": "channel_not_found"}
        else:
            raise AssertionError(request.full_url)
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    return module, event, pull, calls


def test_merged_event_cli_dry_run_reaches_render_without_router(announcement, capsys):
    module, _, _, calls = announcement
    assert module.main(["announce", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Every video window now sees the full audio timeline." in output
    assert "git revert -m 1 abc123" in output and "examplecredential12345" not in output
    assert all(request.data is None for request in calls)


def test_closed_unmerged_event_performs_no_api_calls(announcement):
    module, event, pull, calls = announcement
    event.write_text(json.dumps({"pull_request": {**pull, "merged": False}}))
    assert module.main(["announce"]) == 0
    assert calls == []


def test_post_failure_is_not_reported_as_delivery(announcement, monkeypatch, capsys):
    module, _, _, calls = announcement
    monkeypatch.setenv("HIVE_BOT_TOKEN", "test-slack-token")
    monkeypatch.setenv("HIVE_CHANNEL", "test-channel")
    assert module.main(["announce"]) == 1
    assert "channel_not_found" in capsys.readouterr().err
    payload = json.loads(calls[-1].data)
    assert payload["channel"] == "test-channel" and "[redacted]" in payload["text"]


@pytest.mark.parametrize("fork", [False, True])
def test_repository_author_alias_does_not_apply_to_forks(announcement, monkeypatch, fork):
    module, _, pull, _ = announcement
    monkeypatch.setenv("REVIEW_AUTHOR_ALIASES", "Human Author=fable")
    if fork:
        pull["head"]["repo"]["full_name"] = "someone/forge"

    def github(path):
        commit = {"commit": {"author": {"name": "Human Author"}}}
        return [commit] if path.startswith("pulls/") else commit

    assert module.machine_author(github, pull) == (None if fork else "Human Author")
