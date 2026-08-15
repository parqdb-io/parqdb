from __future__ import annotations

import os
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, quote, unquote, urlsplit, urlunsplit


@dataclass(frozen=True, slots=True)
class _ObjectPrefix:
    scheme: str
    authority: str
    path: str


class SourceUriPolicy:
    """Authorize server-visible Parquet sources against canonical URI prefixes."""

    def __init__(self, prefixes: Sequence[str | Path] = ()) -> None:
        file_roots: list[Path] = []
        object_prefixes: list[_ObjectPrefix] = []
        for value in prefixes:
            reference = os.fspath(value)
            parsed = urlsplit(reference)
            if parsed.scheme in {"", "file"}:
                root = _local_path(parsed, reference).expanduser().resolve()
                if _contains_glob(os.fspath(root)):
                    raise ValueError(
                        "allowed file roots must not contain glob patterns"
                    )
                file_roots.append(root)
            else:
                prefix = _object_prefix(parsed)
                if _contains_glob(prefix.path):
                    raise ValueError(
                        "allowed object-store prefixes must not contain glob patterns"
                    )
                object_prefixes.append(prefix)
        self._file_roots = tuple(file_roots)
        self._object_prefixes = tuple(object_prefixes)

    def authorize(self, source: str | Path) -> str:
        reference = os.fspath(source)
        if not reference:
            raise ValueError("Parquet source must not be empty")
        parsed = urlsplit(reference)
        if parsed.scheme in {"", "file"}:
            path = _local_path(parsed, reference).expanduser().resolve()
            if not any(_is_below(path, root) for root in self._file_roots):
                raise PermissionError(
                    "Parquet source is outside the allowed file roots"
                )
            return os.fspath(path)

        candidate = _object_prefix(parsed)
        if not any(
            _matches_object_prefix(candidate, prefix)
            for prefix in self._object_prefixes
        ):
            raise PermissionError("Parquet source is outside the allowed URI prefixes")
        return urlunsplit(
            SplitResult(
                candidate.scheme,
                candidate.authority,
                quote(candidate.path, safe="/:@-._~!$&'()*+,;=[]"),
                "",
                "",
            )
        )


def _local_path(parsed: SplitResult, reference: str) -> Path:
    if parsed.scheme == "":
        return Path(reference)
    if parsed.query or parsed.fragment:
        raise ValueError("file URIs must not contain a query or fragment")
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError("file URIs must not contain a remote authority")
    path = Path(unquote(parsed.path))
    if not path.is_absolute():
        raise ValueError("file URIs must contain an absolute path")
    return path


def _object_prefix(parsed: SplitResult) -> _ObjectPrefix:
    scheme = parsed.scheme.lower()
    if not scheme or scheme == "file":
        raise ValueError("object-store URI must contain a non-file scheme")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("source URIs must not contain credentials")
    if not parsed.netloc:
        raise ValueError("object-store URI must contain an authority")
    if parsed.query or parsed.fragment:
        raise ValueError("source URIs must not contain a query or fragment")
    path = _normalize_object_path(parsed.path)
    return _ObjectPrefix(scheme, parsed.netloc, path)


def _normalize_object_path(path: str) -> str:
    decoded = unquote(path)
    if any(ord(character) < 32 or ord(character) == 127 for character in decoded):
        raise ValueError("source URI paths must not contain control characters")
    normalized = posixpath.normpath(f"/{decoded.lstrip('/')}")
    return "/" if normalized == "/." else normalized


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _matches_object_prefix(candidate: _ObjectPrefix, prefix: _ObjectPrefix) -> bool:
    if (candidate.scheme, candidate.authority) != (prefix.scheme, prefix.authority):
        return False
    root = prefix.path.rstrip("/") or "/"
    return candidate.path == root or candidate.path.startswith(f"{root.rstrip('/')}/")


def _contains_glob(value: str) -> bool:
    return any(character in value for character in "*?[")
