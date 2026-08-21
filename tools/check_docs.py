from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import SplitResult, unquote, urlsplit

from markdown_it import MarkdownIt

EXCLUDED_DIRECTORIES = {".git", ".uv-cache", ".venv", "node_modules", "target"}
GITHUB_REPOSITORY = ("petrizhang", "parqdb")
GITHUB_DEFAULT_BRANCH = "main"


@dataclass(frozen=True)
class Link:
    source: Path
    target: str


class _HTMLTargets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attribute = "href" if tag == "a" else "src" if tag == "img" else None
        if attribute is None:
            return
        values = dict(attrs)
        target = values.get(attribute)
        if target:
            self.targets.append(target)


def markdown_files(root: Path) -> tuple[Path, ...]:
    documents: list[Path] = []
    for directory, directories, files in os.walk(root):
        directories[:] = sorted(
            name for name in directories if name not in EXCLUDED_DIRECTORIES
        )
        documents.extend(
            Path(directory) / name for name in sorted(files) if name.endswith(".md")
        )
    return tuple(documents)


def github_anchors(parser: MarkdownIt, document: Path) -> set[str]:
    tokens = parser.parse(document.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    anchors: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if token.type != "heading_open" or tokens[index + 1].type != "inline":
            continue
        base = _github_slug(tokens[index + 1].content)
        duplicate = counts[base]
        counts[base] += 1
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
    return anchors


def document_links(parser: MarkdownIt, document: Path) -> tuple[Link, ...]:
    links: list[Link] = []
    for token in parser.parse(document.read_text(encoding="utf-8")):
        for child in token.children or ():
            attribute = (
                "href"
                if child.type == "link_open"
                else "src"
                if child.type == "image"
                else None
            )
            if attribute is not None:
                target = child.attrGet(attribute)
                if target:
                    links.append(Link(document, target))
            if child.type == "html_inline":
                links.extend(
                    Link(document, target) for target in _html_targets(child.content)
                )
        if token.type == "html_block":
            links.extend(
                Link(document, target) for target in _html_targets(token.content)
            )
    return tuple(links)


def validate(root: Path) -> tuple[str, ...]:
    root = root.resolve()
    parser = MarkdownIt()
    documents = markdown_files(root)
    anchors = {
        document.resolve(): github_anchors(parser, document) for document in documents
    }
    failures: list[str] = []
    for document in documents:
        for link in document_links(parser, document):
            parsed = urlsplit(link.target)
            repository_path = _repository_path(parsed)
            if (parsed.scheme or parsed.netloc) and repository_path is None:
                continue
            destination = (
                (root / repository_path).resolve()
                if repository_path is not None
                else (document.parent / unquote(parsed.path)).resolve()
                if parsed.path
                else document.resolve()
            )
            display = document.relative_to(root)
            if not destination.exists():
                failures.append(f"{display}: missing target {link.target}")
                continue
            if (
                parsed.fragment
                and destination.suffix.lower() == ".md"
                and unquote(parsed.fragment).lower()
                not in anchors.get(destination, set())
            ):
                failures.append(f"{display}: missing anchor {link.target}")
    return tuple(failures)


def _github_slug(value: str) -> str:
    normalized = re.sub(r"[^\w\- ]", "", value.strip().lower())
    return re.sub(r"\s+", "-", normalized)


def _html_targets(value: str) -> tuple[str, ...]:
    parser = _HTMLTargets()
    parser.feed(value)
    return tuple(parser.targets)


def _repository_path(parsed: SplitResult) -> Path | None:
    hostname = parsed.hostname
    parts = unquote(parsed.path).strip("/").split("/")
    owner, repository = GITHUB_REPOSITORY
    if (
        hostname == "github.com"
        and parts[:2] == [owner, repository]
        and len(parts) >= 4
        and parts[2] in {"blob", "tree"}
        and parts[3] == GITHUB_DEFAULT_BRANCH
    ):
        return Path(*parts[4:])
    if hostname == "raw.githubusercontent.com" and parts[:3] == [
        owner,
        repository,
        GITHUB_DEFAULT_BRANCH,
    ]:
        return Path(*parts[3:])
    return None


def main() -> None:
    command = argparse.ArgumentParser(description="Check local Markdown links")
    command.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).parents[1],
    )
    args = command.parse_args()
    failures = validate(args.root)
    if failures:
        raise SystemExit("\n".join(failures))
    print("all local Markdown links and anchors resolve")


if __name__ == "__main__":
    main()
