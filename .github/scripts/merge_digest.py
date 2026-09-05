#!/usr/bin/env python3
"""Announce this repository's machine-authored merges without a review router.

The closed-PR event supplies the result and the PR body supplies its substance.
This helper neither requests reviews nor reconstructs a merge-admission policy.
The former cross-repository digest has no workflow in this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Preserve the author filter of this repository's existing announcement job.
SEAT_AUTHORS = {"Fable", "Ariadne", "gnomon", "Talos", "Theoros"}

SUBSTANCE_FLOOR = 180


MAX_SUBSTANCE = 900


HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


BOILERPLATE_PATTERN = re.compile(
    r"(?im)^\s*(?:closes|fixes|resolves)\s+KRA-\d+\s*$"
    r"|^\s*(?:co-authored-by|generated with|🤖).*$"
    r"|^\s*<?https?://\S*claude\.com/claude-code>?\s*$"
)


SECRET_PATTERNS = (
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    # A PEM is the header, the base64 body, and the matching END marker.
    # Replacing only the BEGIN line leaves a usable key in the Slack post.
    re.compile(
        r"-----BEGIN[A-Z ]*PRIVATE KEY-----"
        r"(?:.*?-----END[A-Z ]*PRIVATE KEY-----|.+)",
        re.DOTALL,
    ),
    # An assignment to anything *named* like a credential. The leading
    # identifier run is what makes `HIVE_BOT_TOKEN=…` match: `\b` does not fire
    # inside an underscored name. Over-redaction is the deliberate direction —
    # a mangled sentence costs a reader a click, a leaked token costs a rotation.
    re.compile(
        r"(?i)[A-Za-z0-9_.\-]*(?:token|secret|password|passwd|api[_-]?key)"
        r"\s*[:=]\s*[\"']?([A-Za-z0-9/+_.\-]{12,})"
    ),
)


REDACTED = "[redacted]"


def redact(text: str) -> str:
    """Strip anything shaped like a credential before it reaches Slack.

    PR bodies are human-authored text on the way to a channel; a pasted token
    must not survive the trip.  Redaction is unconditional and applied at the
    single point every message passes through.
    """
    result = text
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            result = pattern.sub(
                lambda match: match.group(0).replace(match.group(1), REDACTED),
                result,
            )
        else:
            result = pattern.sub(REDACTED, result)
    return result


def substance(body: str) -> tuple[str, bool]:
    """The account of what changed, taken from the pull body.

    A digest entry exists to say what is now true of the system that was not
    true before.  Nothing mechanical can write that; the author already did, in
    the body.  What this can do honestly is carry it across, and say so out loud
    when there is too little of it to be cold-readable rather than pass a title
    off as an account.
    """
    text = HTML_COMMENT_PATTERN.sub("", body or "")
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if BOILERPLATE_PATTERN.match(line):
            continue
        heading = re.match(r"^\s*#{1,6}\s+(.*)$", line)
        if heading:
            lines.append(f"*{heading.group(1).strip()}*")
            continue
        lines.append(line)
    prose = "\n".join(lines).strip()
    prose = re.sub(r"\n{3,}", "\n\n", prose)
    measured = len(re.sub(r"\s+", " ", re.sub(r"[*_`>#-]", "", prose)).strip())
    if len(prose) > MAX_SUBSTANCE:
        cut = prose.rfind("\n", 0, MAX_SUBSTANCE)
        if cut < MAX_SUBSTANCE // 2:
            cut = MAX_SUBSTANCE
        prose = f"{prose[:cut].rstrip()}\n…(body continues)"
    return prose, measured < SUBSTANCE_FLOOR

def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is missing")
    return value


def request_json(url, token, payload=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "Accept": "application/json", "User-Agent": "h3forge-merge-announcement"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"merge announcement request failed with HTTP {exc.code}") from exc


def machine_author(github, pull):
    names = []
    page = 1
    while True:
        commits = github(f"pulls/{pull['number']}/commits?per_page=100&page={page}")
        names.extend(commit["commit"]["author"]["name"] for commit in commits)
        if len(commits) < 100:
            break
        page += 1
    burn = os.environ.get("REVIEW_BURN_ACTOR", "").strip().lower() or "talos"

    def choose(seats):
        return (next((name for name in names if name in seats and seats[name] != burn), None)
                or next((name for name in names if name in seats), None))

    author = choose({name: name.lower() for name in SEAT_AUTHORS})
    if author:
        return author
    head_author = github(f"commits/{pull['head']['sha']}")["commit"]["author"]["name"]
    if head_author in SEAT_AUTHORS:
        return head_author
    names.append(head_author)
    head_repo = (pull["head"].get("repo") or {}).get("full_name")
    base_repo = (pull["base"].get("repo") or {}).get("full_name")
    if not head_repo or head_repo != base_repo:
        return None
    aliases = {}
    for entry in re.split(r"[,\n]", os.environ.get("REVIEW_AUTHOR_ALIASES", "")):
        if not entry.strip():
            continue
        name, separator, seat = entry.partition("=")
        seat = seat.strip().lower()
        if not separator or not name.strip() or not re.fullmatch(r"[a-z0-9-]+", seat):
            raise ValueError("REVIEW_AUTHOR_ALIASES must contain Author Name=seat entries")
        aliases[name.strip()] = seat
    return choose(aliases)


def run_announce(*, dry_run=False):
    event = json.loads(Path(required_env("GITHUB_EVENT_PATH")).read_text())
    pull = event["pull_request"]
    if not pull.get("merged"):
        print("Pull request closed without merging; nothing to announce.")
        return
    slug = required_env("GITHUB_REPOSITORY")
    token = required_env("GITHUB_TOKEN")
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")

    def github(path):
        return request_json(f"{api}/repos/{slug}/{path}", token)

    author = machine_author(github, pull)
    if author is None:
        print(f"{slug}#{pull['number']} is not machine-authored; nothing to announce.")
        return
    sha = pull["merge_commit_sha"]
    parents = github(f"commits/{sha}")["parents"]
    revert = f"git revert {'-m 1 ' if len(parents) > 1 else ''}{sha}"
    account, thin = substance(pull.get("body") or "")
    message = (
        f"*Machine merge* — <{pull['html_url']}|{slug}#{pull['number']}> {pull['title']}\n"
        f"{account or '_(empty body)_'}\n"
        f"Merged {pull['merged_at']} by {(pull.get('merged_by') or {}).get('login', '?')}; "
        f"authored by {author}.\nRevert anchor: `{revert}` (reversal cost not measured)."
    )
    if thin:
        message += "\nPR body contains little detail; follow the PR link for context."
    message = redact(message)
    if dry_run:
        print(message)
        return
    result = request_json("https://slack.com/api/chat.postMessage", required_env("HIVE_BOT_TOKEN"),
                          {"channel": required_env("HIVE_CHANNEL"), "text": message})
    if result.get("ok") is not True:
        raise RuntimeError(f"Slack rejected the announcement: {result.get('error', 'unknown error')}")
    print(f"Announced {slug}#{pull['number']}, message {result['ts']}.")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("announce",))
    parser.add_argument("--dry-run", action="store_true", help="Render without posting to Hive.")
    args = parser.parse_args(argv)
    try:
        run_announce(dry_run=args.dry_run or os.environ.get("DIGEST_DRY_RUN", "").lower() == "true")
    except Exception as exc:
        print(f"merge announcement failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
