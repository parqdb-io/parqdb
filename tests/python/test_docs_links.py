import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_repository_markdown_links_resolve() -> None:
    result = _check(ROOT)

    assert result.returncode == 0
    assert result.stdout.strip() == "all local Markdown links and anchors resolve"


def test_missing_markdown_target_and_anchor_are_reported(tmp_path: Path) -> None:
    (tmp_path / "target.md").write_text("# Existing\n", encoding="utf-8")
    (tmp_path / "source.md").write_text(
        "[missing file](missing.md)\n[missing anchor](target.md#absent)\n",
        encoding="utf-8",
    )

    result = _check(tmp_path)

    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        "source.md: missing target missing.md",
        "source.md: missing anchor target.md#absent",
    ]


def test_canonical_repository_links_are_checked_locally(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "target.md").write_text("# Existing\n", encoding="utf-8")
    (tmp_path / "asset.svg").write_text("<svg/>\n", encoding="utf-8")
    (tmp_path / "source.md").write_text(
        "[doc](https://github.com/petrizhang/relify/blob/main/docs/target.md#existing)\n"
        "![asset](https://raw.githubusercontent.com/petrizhang/relify/main/asset.svg)\n",
        encoding="utf-8",
    )

    result = _check(tmp_path)

    assert result.returncode == 0


def _check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_docs.py"), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
