# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26,<3",
#   "pyarrow>=18,<22",
#   "tantivy>=0.26,<0.27",
# ]
# ///
"""Prototype open Parquet postings for BM25 and phrase search.

This is an isolated feasibility experiment, not a Relify public API or format.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import platform
import re
import shutil
import statistics
import sys
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tantivy

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
BM25_K1 = 1.2
BM25_B = 0.75

PostingLayout = Literal["row", "block"]
Posting = tuple[int, int, list[int]]


@dataclass(frozen=True)
class QueryCase:
    name: str
    text: str
    phrase: bool


@dataclass(frozen=True)
class SearchHit:
    doc_id: int
    score: float


@dataclass(frozen=True)
class IoMetrics:
    files: int
    row_groups: int
    read_operations: int
    bytes_read: int


@dataclass(frozen=True)
class SearchMetrics:
    read_seconds: float
    score_seconds: float
    topk_seconds: float
    source_fetch_seconds: float
    total_seconds: float
    index_io: IoMetrics
    source_io: IoMetrics


@dataclass(frozen=True)
class SearchRun:
    hits: tuple[SearchHit, ...]
    metrics: SearchMetrics


@dataclass(frozen=True)
class CorpusInfo:
    rows: int
    tokens_per_document: int
    vocabulary_size: int
    source_row_group_rows: int
    rare_phrase_documents: int
    common_phrase_documents: int


class CountingFile:
    """Seekable file wrapper that counts reads issued by Arrow."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: BinaryIO = path.open("rb", buffering=0)
        self.bytes_read = 0
        self.read_operations = 0

    def read(self, size: int = -1) -> bytes:
        data = self._file.read(size)
        self.bytes_read += len(data)
        self.read_operations += 1
        return data

    def readinto(self, buffer: bytearray) -> int:
        count = self._file.readinto(buffer)
        if count is None:
            return 0
        self.bytes_read += count
        self.read_operations += 1
        return count

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._file.seek(offset, whence)

    def tell(self) -> int:
        return self._file.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return self._file.closed

    def close(self) -> None:
        self._file.close()

    def reset_counts(self) -> None:
        self.bytes_read = 0
        self.read_operations = 0


class CountingParquetFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = CountingFile(path)
        self.parquet = pq.ParquetFile(self.stream)

    def reset_counts(self) -> None:
        self.stream.reset_counts()

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def term_bucket(term: str, buckets: int) -> int:
    return zlib.crc32(term.encode("utf-8")) % buckets


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[position]


def directory_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def generate_corpus(
    path: Path,
    *,
    rows: int,
    tokens_per_document: int,
    vocabulary_size: int,
    source_row_group_rows: int,
    seed: int,
) -> CorpusInfo:
    vocabulary = np.asarray(
        ["error", "request", "failed", "agent", "tool", "response"]
        + [f"term{value:05d}" for value in range(vocabulary_size - 6)],
        dtype=object,
    )
    ranks = np.arange(1, len(vocabulary) + 1, dtype=np.float64)
    probabilities = np.power(ranks, -1.08)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)

    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("timestamp", pa.int64(), nullable=False),
            pa.field("level", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
        ]
    )
    rare_phrase_documents = 0
    common_phrase_documents = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for start in range(0, rows, source_row_group_rows):
            count = min(source_row_group_rows, rows - start)
            sampled = rng.choice(
                vocabulary,
                size=(count, tokens_per_document),
                replace=True,
                p=probabilities,
            )
            texts: list[str] = []
            levels: list[str] = []
            for offset, tokens in enumerate(sampled):
                doc_id = start + offset
                values = tokens.tolist()
                if doc_id % 997 == 0:
                    values[10:12] = ["needle", "failed"]
                    rare_phrase_documents += 1
                if doc_id % 20 == 0:
                    values[30:32] = ["request", "failed"]
                    common_phrase_documents += 1
                texts.append(" ".join(values))
                levels.append("ERROR" if doc_id % 4 == 0 else "DEFAULT")
            table = pa.Table.from_arrays(
                [
                    pa.array(range(start, start + count), type=pa.int64()),
                    pa.array(
                        range(1_750_000_000 + start, 1_750_000_000 + start + count),
                        type=pa.int64(),
                    ),
                    pa.array(levels, type=pa.string()),
                    pa.array(texts, type=pa.string()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=source_row_group_rows)
    return CorpusInfo(
        rows=rows,
        tokens_per_document=tokens_per_document,
        vocabulary_size=vocabulary_size,
        source_row_group_rows=source_row_group_rows,
        rare_phrase_documents=rare_phrase_documents,
        common_phrase_documents=common_phrase_documents,
    )


def iter_source(path: Path) -> Iterable[tuple[int, str]]:
    source = pq.ParquetFile(path)
    for batch in source.iter_batches(columns=["id", "text"], batch_size=2_048):
        ids = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        yield from zip(ids, texts, strict=True)


def write_row_postings(
    path: Path,
    terms: Sequence[str],
    postings: dict[str, list[Posting]],
    *,
    row_group_rows: int,
) -> int:
    encoded_terms: list[str] = []
    doc_ids: list[int] = []
    frequencies: list[int] = []
    positions: list[list[int]] = []
    for term in terms:
        for doc_id, frequency, term_positions in postings[term]:
            encoded_terms.append(term)
            doc_ids.append(doc_id)
            frequencies.append(frequency)
            positions.append(term_positions)
    pq.write_table(
        pa.table(
            {
                "term": pa.array(encoded_terms, type=pa.string()),
                "doc_id": pa.array(doc_ids, type=pa.int64()),
                "tf": pa.array(frequencies, type=pa.int32()),
                "positions": pa.array(
                    positions,
                    type=pa.list_(pa.field("item", pa.int32(), nullable=False)),
                ),
            }
        ),
        path,
        compression="zstd",
        use_dictionary=["term"],
        write_statistics=True,
        row_group_size=row_group_rows,
    )
    return len(encoded_terms)


def write_block_postings(
    path: Path,
    terms: Sequence[str],
    postings: dict[str, list[Posting]],
    *,
    block_docs: int,
    row_group_rows: int,
) -> int:
    encoded_terms: list[str] = []
    doc_ids: list[list[int]] = []
    frequencies: list[list[int]] = []
    positions: list[list[list[int]]] = []
    for term in terms:
        term_postings = postings[term]
        for start in range(0, len(term_postings), block_docs):
            block = term_postings[start : start + block_docs]
            encoded_terms.append(term)
            doc_ids.append([posting[0] for posting in block])
            frequencies.append([posting[1] for posting in block])
            positions.append([posting[2] for posting in block])
    pq.write_table(
        pa.table(
            {
                "term": pa.array(encoded_terms, type=pa.string()),
                "doc_id": pa.array(
                    doc_ids,
                    type=pa.list_(pa.field("item", pa.int64(), nullable=False)),
                ),
                "tf": pa.array(
                    frequencies,
                    type=pa.list_(pa.field("item", pa.int32(), nullable=False)),
                ),
                "positions": pa.array(
                    positions,
                    type=pa.list_(
                        pa.field(
                            "item",
                            pa.list_(pa.field("item", pa.int32(), nullable=False)),
                            nullable=False,
                        )
                    ),
                ),
            }
        ),
        path,
        compression="zstd",
        use_dictionary=["term"],
        write_statistics=True,
        row_group_size=row_group_rows,
    )
    return len(encoded_terms)


def build_parquet_index(
    source_path: Path,
    index_path: Path,
    *,
    buckets: int,
    posting_layout: PostingLayout,
    posting_block_docs: int,
    posting_row_group_rows: int,
    source_row_group_rows: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    postings: dict[str, list[Posting]] = defaultdict(list)
    document_ids: list[int] = []
    document_lengths: list[int] = []
    source_row_groups: list[int] = []

    for doc_id, text in iter_source(source_path):
        tokens = tokenize(text)
        positions: dict[str, list[int]] = defaultdict(list)
        for position, term in enumerate(tokens):
            positions[term].append(position)
        for term, term_positions in positions.items():
            postings[term].append((doc_id, len(term_positions), term_positions))
        document_ids.append(doc_id)
        document_lengths.append(len(tokens))
        source_row_groups.append(doc_id // source_row_group_rows)

    index_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "document_count": pa.array([len(document_ids)], type=pa.int64()),
                "total_document_length": pa.array(
                    [sum(document_lengths)], type=pa.int64()
                ),
            }
        ),
        index_path / "stats.parquet",
        compression="zstd",
    )
    pq.write_table(
        pa.table(
            {
                "doc_id": pa.array(document_ids, type=pa.int64()),
                "document_length": pa.array(document_lengths, type=pa.int32()),
                "source_row_group": pa.array(source_row_groups, type=pa.int32()),
            }
        ),
        index_path / "documents.parquet",
        compression="zstd",
        row_group_size=posting_row_group_rows,
    )
    pq.write_table(
        pa.table(
            {
                "term": pa.array(sorted(postings), type=pa.string()),
                "document_frequency": pa.array(
                    [len(postings[term]) for term in sorted(postings)],
                    type=pa.int64(),
                ),
            }
        ),
        index_path / "terms.parquet",
        compression="zstd",
    )

    by_bucket: dict[int, list[str]] = defaultdict(list)
    for term in postings:
        by_bucket[term_bucket(term, buckets)].append(term)
    postings_path = index_path / "postings"
    postings_path.mkdir()
    posting_rows = 0
    encoded_rows = 0
    for bucket, terms in sorted(by_bucket.items()):
        sorted_terms = sorted(terms)
        posting_rows += sum(len(postings[term]) for term in sorted_terms)
        bucket_path = postings_path / f"term_bucket={bucket:04d}"
        bucket_path.mkdir()
        if posting_layout == "row":
            encoded_rows += write_row_postings(
                bucket_path / "part-00000.parquet",
                sorted_terms,
                postings,
                row_group_rows=posting_row_group_rows,
            )
        else:
            encoded_rows += write_block_postings(
                bucket_path / "part-00000.parquet",
                sorted_terms,
                postings,
                block_docs=posting_block_docs,
                row_group_rows=posting_row_group_rows,
            )
    elapsed = time.perf_counter() - started
    return {
        "seconds": elapsed,
        "layout": posting_layout,
        "posting_rows": posting_rows,
        "encoded_rows": encoded_rows,
        "terms": len(postings),
        "bytes": directory_bytes(index_path),
    }


def build_tantivy_index(source_path: Path, index_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    index_path.mkdir(parents=True, exist_ok=True)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_integer_field("id", stored=True)
    schema_builder.add_text_field(
        "text",
        stored=False,
        tokenizer_name="default",
        index_option="position",
    )
    schema = schema_builder.build()
    index = tantivy.Index(schema, path=str(index_path), reuse=False)
    writer = index.writer(heap_size=128_000_000, num_threads=1)
    for doc_id, text in iter_source(source_path):
        writer.add_document(
            tantivy.Document.from_dict({"id": doc_id, "text": text}, schema)
        )
    writer.commit()
    return {
        "seconds": time.perf_counter() - started,
        "bytes": directory_bytes(index_path),
    }


class SourceFetcher:
    def __init__(self, path: Path) -> None:
        self.file = CountingParquetFile(path)
        self.id_column = self.file.parquet.schema_arrow.get_field_index("id")
        self.text_column = self.file.parquet.schema_arrow.get_field_index("text")

    def close(self) -> None:
        self.file.close()

    def fetch(
        self, doc_ids: Sequence[int], row_group_rows: int
    ) -> tuple[dict[int, str], IoMetrics]:
        self.file.reset_counts()
        requested = set(doc_ids)
        row_groups = sorted({doc_id // row_group_rows for doc_id in requested})
        table = self.file.parquet.read_row_groups(row_groups, columns=["id", "text"])
        values = {
            doc_id: text
            for doc_id, text in zip(
                table["id"].to_pylist(), table["text"].to_pylist(), strict=True
            )
            if doc_id in requested
        }
        if values.keys() != requested:
            missing = sorted(requested - values.keys())
            raise RuntimeError(f"source documents not found: {missing}")
        return values, IoMetrics(
            files=1,
            row_groups=len(row_groups),
            read_operations=self.file.stream.read_operations,
            bytes_read=self.file.stream.bytes_read,
        )


class ParquetPostingIndex:
    def __init__(
        self, path: Path, *, buckets: int, posting_layout: PostingLayout
    ) -> None:
        self.path = path
        self.buckets = buckets
        self.posting_layout = posting_layout
        stats = pq.read_table(path / "stats.parquet").to_pydict()
        self.document_count = int(stats["document_count"][0])
        self.total_document_length = int(stats["total_document_length"][0])
        documents = pq.read_table(path / "documents.parquet").to_pydict()
        self.document_lengths = {
            int(doc_id): int(length)
            for doc_id, length in zip(
                documents["doc_id"], documents["document_length"], strict=True
            )
        }
        terms = pq.read_table(path / "terms.parquet").to_pydict()
        self.document_frequencies = {
            str(term): int(frequency)
            for term, frequency in zip(
                terms["term"], terms["document_frequency"], strict=True
            )
        }
        self.files: dict[int, CountingParquetFile] = {}
        for file_path in sorted((path / "postings").glob("term_bucket=*/*.parquet")):
            bucket = int(file_path.parent.name.split("=", maxsplit=1)[1])
            self.files[bucket] = CountingParquetFile(file_path)

    def close(self) -> None:
        for file in self.files.values():
            file.close()

    @staticmethod
    def _term_row_groups(file: CountingParquetFile, term: str) -> list[int]:
        term_index = file.parquet.schema_arrow.get_field_index("term")
        selected = []
        for row_group in range(file.parquet.num_row_groups):
            statistics = (
                file.parquet.metadata.row_group(row_group).column(term_index).statistics
            )
            if statistics is None or not statistics.has_min_max:
                selected.append(row_group)
                continue
            minimum = statistics.min
            maximum = statistics.max
            if isinstance(minimum, bytes):
                minimum = minimum.decode("utf-8")
                maximum = maximum.decode("utf-8")
            if minimum <= term <= maximum:
                selected.append(row_group)
        return selected

    def search(self, query: QueryCase, *, k: int) -> SearchRun:
        started = time.perf_counter()
        query_terms = tokenize(query.text)
        requested = set(query_terms)
        selected: dict[tuple[int, int], set[str]] = defaultdict(set)
        selected_files: set[int] = set()
        for term in requested:
            bucket = term_bucket(term, self.buckets)
            file = self.files.get(bucket)
            if file is None:
                continue
            selected_files.add(bucket)
            for row_group in self._term_row_groups(file, term):
                selected[(bucket, row_group)].add(term)

        for bucket in selected_files:
            self.files[bucket].reset_counts()
        posting_rows: dict[str, dict[int, tuple[int, list[int] | None]]] = {
            term: {} for term in requested
        }
        read_started = time.perf_counter()
        columns = ["term", "doc_id", "tf"]
        if query.phrase:
            columns.append("positions")
        for (bucket, row_group), terms in selected.items():
            table = self.files[bucket].parquet.read_row_group(
                row_group, columns=columns
            )
            encoded_terms = table["term"].to_pylist()
            doc_ids = table["doc_id"].to_pylist()
            frequencies = table["tf"].to_pylist()
            positions = (
                table["positions"].to_pylist() if query.phrase else [None] * len(table)
            )
            for term, encoded_doc_ids, encoded_frequencies, encoded_positions in zip(
                encoded_terms, doc_ids, frequencies, positions, strict=True
            ):
                if term not in terms:
                    continue
                if self.posting_layout == "row":
                    posting_rows[term][encoded_doc_ids] = (
                        encoded_frequencies,
                        encoded_positions,
                    )
                    continue
                block_positions = (
                    encoded_positions
                    if encoded_positions is not None
                    else [None] * len(encoded_doc_ids)
                )
                for doc_id, frequency, term_positions in zip(
                    encoded_doc_ids,
                    encoded_frequencies,
                    block_positions,
                    strict=True,
                ):
                    posting_rows[term][doc_id] = (frequency, term_positions)
        read_seconds = time.perf_counter() - read_started

        score_started = time.perf_counter()
        if query.phrase:
            candidates = self._phrase_candidates(query_terms, posting_rows)
        else:
            candidates: set[int] = set()
            for values in posting_rows.values():
                candidates.update(values)
        term_counts = Counter(query_terms)
        scores = {
            doc_id: self._bm25_score(doc_id, term_counts, posting_rows)
            for doc_id in candidates
        }
        score_seconds = time.perf_counter() - score_started

        topk_started = time.perf_counter()
        ordered = heapq.nsmallest(
            k,
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        hits = tuple(SearchHit(doc_id, score) for doc_id, score in ordered)
        topk_seconds = time.perf_counter() - topk_started
        io = IoMetrics(
            files=len(selected_files),
            row_groups=len(selected),
            read_operations=sum(
                self.files[bucket].stream.read_operations for bucket in selected_files
            ),
            bytes_read=sum(
                self.files[bucket].stream.bytes_read for bucket in selected_files
            ),
        )
        total_seconds = time.perf_counter() - started
        return SearchRun(
            hits=hits,
            metrics=SearchMetrics(
                read_seconds=read_seconds,
                score_seconds=score_seconds,
                topk_seconds=topk_seconds,
                source_fetch_seconds=0.0,
                total_seconds=total_seconds,
                index_io=io,
                source_io=IoMetrics(0, 0, 0, 0),
            ),
        )

    @staticmethod
    def _phrase_candidates(
        query_terms: Sequence[str],
        postings: dict[str, dict[int, tuple[int, list[int] | None]]],
    ) -> set[int]:
        if not query_terms:
            return set()
        candidate_sets = [set(postings[term]) for term in set(query_terms)]
        candidates = set.intersection(*candidate_sets) if candidate_sets else set()
        matches = set()
        for doc_id in candidates:
            first_positions = postings[query_terms[0]][doc_id][1]
            if first_positions is None:
                raise RuntimeError("phrase query did not load positions")
            remaining = []
            for term in query_terms[1:]:
                positions = postings[term][doc_id][1]
                if positions is None:
                    raise RuntimeError("phrase query did not load positions")
                remaining.append(set(positions))
            if any(
                all(
                    start + offset in positions
                    for offset, positions in enumerate(remaining, 1)
                )
                for start in first_positions
            ):
                matches.add(doc_id)
        return matches

    def _bm25_score(
        self,
        doc_id: int,
        query_terms: Counter[str],
        postings: dict[str, dict[int, tuple[int, list[int] | None]]],
    ) -> float:
        average_length = self.total_document_length / self.document_count
        document_length = self.document_lengths[doc_id]
        score = 0.0
        for term, query_frequency in query_terms.items():
            posting = postings[term].get(doc_id)
            if posting is None:
                continue
            term_frequency = posting[0]
            document_frequency = self.document_frequencies[term]
            inverse_document_frequency = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + BM25_K1 * (
                1.0 - BM25_B + BM25_B * document_length / average_length
            )
            score += (
                query_frequency
                * inverse_document_frequency
                * term_frequency
                * (BM25_K1 + 1.0)
                / denominator
            )
        return score


class FullScanSearcher:
    def __init__(self, source_path: Path, index: ParquetPostingIndex) -> None:
        self.source = CountingParquetFile(source_path)
        self.index = index

    def close(self) -> None:
        self.source.close()

    def search(self, query: QueryCase, *, k: int) -> SearchRun:
        started = time.perf_counter()
        self.source.reset_counts()
        query_terms = tokenize(query.text)
        query_counts = Counter(query_terms)
        scores: dict[int, float] = {}
        read_seconds = 0.0
        score_seconds = 0.0
        for row_group in range(self.source.parquet.num_row_groups):
            read_started = time.perf_counter()
            table = self.source.parquet.read_row_group(
                row_group, columns=["id", "text"]
            )
            read_seconds += time.perf_counter() - read_started
            score_started = time.perf_counter()
            for doc_id, text in zip(
                table["id"].to_pylist(), table["text"].to_pylist(), strict=True
            ):
                tokens = tokenize(text)
                if query.phrase and not contains_phrase(tokens, query_terms):
                    continue
                frequencies = Counter(tokens)
                if not query.phrase and not any(
                    term in frequencies for term in query_terms
                ):
                    continue
                score = exact_bm25_score(
                    doc_id,
                    frequencies,
                    query_counts,
                    self.index,
                )
                scores[doc_id] = score
            score_seconds += time.perf_counter() - score_started
        topk_started = time.perf_counter()
        ordered = heapq.nsmallest(
            k, scores.items(), key=lambda item: (-item[1], item[0])
        )
        topk_seconds = time.perf_counter() - topk_started
        return SearchRun(
            hits=tuple(SearchHit(doc_id, score) for doc_id, score in ordered),
            metrics=SearchMetrics(
                read_seconds=read_seconds,
                score_seconds=score_seconds,
                topk_seconds=topk_seconds,
                source_fetch_seconds=0.0,
                total_seconds=time.perf_counter() - started,
                index_io=IoMetrics(0, 0, 0, 0),
                source_io=IoMetrics(
                    files=1,
                    row_groups=self.source.parquet.num_row_groups,
                    read_operations=self.source.stream.read_operations,
                    bytes_read=self.source.stream.bytes_read,
                ),
            ),
        )


class TantivySearcher:
    def __init__(self, path: Path) -> None:
        self.index = tantivy.Index.open(str(path))
        self.index.reload()
        self.schema = self.index.schema
        self.searcher = self.index.searcher()

    def search(self, query: QueryCase, *, k: int) -> SearchRun:
        terms = tokenize(query.text)
        clauses = [
            (
                tantivy.Occur.Should,
                tantivy.Query.term_query(
                    self.schema, "text", term, index_option="position"
                ),
            )
            for term in terms
        ]
        if query.phrase:
            phrase = tantivy.Query.phrase_query(self.schema, "text", terms)
            clauses.insert(
                0,
                (
                    tantivy.Occur.Must,
                    tantivy.Query.const_score_query(phrase, 0.0),
                ),
            )
        parsed = tantivy.Query.boolean_query(clauses)
        started = time.perf_counter()
        result = self.searcher.search(parsed, limit=k)
        hits = tuple(
            SearchHit(
                int(self.searcher.doc(address).to_dict()["id"][0]),
                float(score),
            )
            for score, address in result.hits
        )
        total = time.perf_counter() - started
        return SearchRun(
            hits=hits,
            metrics=SearchMetrics(
                read_seconds=total,
                score_seconds=0.0,
                topk_seconds=0.0,
                source_fetch_seconds=0.0,
                total_seconds=total,
                index_io=IoMetrics(0, 0, 0, 0),
                source_io=IoMetrics(0, 0, 0, 0),
            ),
        )


def contains_phrase(tokens: Sequence[str], query_terms: Sequence[str]) -> bool:
    width = len(query_terms)
    return any(
        tokens[start : start + width] == list(query_terms)
        for start in range(len(tokens) - width + 1)
    )


def exact_bm25_score(
    doc_id: int,
    frequencies: Counter[str],
    query_terms: Counter[str],
    index: ParquetPostingIndex,
) -> float:
    average_length = index.total_document_length / index.document_count
    document_length = index.document_lengths[doc_id]
    score = 0.0
    for term, query_frequency in query_terms.items():
        term_frequency = frequencies.get(term, 0)
        if term_frequency == 0:
            continue
        document_frequency = index.document_frequencies.get(term, 0)
        inverse_document_frequency = math.log(
            1.0
            + (index.document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )
        denominator = term_frequency + BM25_K1 * (
            1.0 - BM25_B + BM25_B * document_length / average_length
        )
        score += (
            query_frequency
            * inverse_document_frequency
            * term_frequency
            * (BM25_K1 + 1.0)
            / denominator
        )
    return score


def with_source_fetch(
    run: SearchRun,
    source: SourceFetcher,
    *,
    source_row_group_rows: int,
) -> SearchRun:
    started = time.perf_counter()
    _, io = source.fetch([hit.doc_id for hit in run.hits], source_row_group_rows)
    elapsed = time.perf_counter() - started
    metrics = run.metrics
    return SearchRun(
        hits=run.hits,
        metrics=SearchMetrics(
            read_seconds=metrics.read_seconds,
            score_seconds=metrics.score_seconds,
            topk_seconds=metrics.topk_seconds,
            source_fetch_seconds=elapsed,
            total_seconds=metrics.total_seconds + elapsed,
            index_io=metrics.index_io,
            source_io=io,
        ),
    )


def summarize_runs(runs: Sequence[SearchRun]) -> dict[str, Any]:
    latencies = [run.metrics.total_seconds for run in runs]
    representative = min(
        runs,
        key=lambda run: abs(run.metrics.total_seconds - statistics.median(latencies)),
    )
    metrics = asdict(representative.metrics)
    metrics["median_total_seconds"] = statistics.median(latencies)
    metrics["p95_total_seconds"] = percentile(latencies, 0.95)
    return {
        "hits": [asdict(hit) for hit in representative.hits],
        "metrics": metrics,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.expanduser().resolve()
    if args.rebuild and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    source_path = root / "source.parquet"
    parquet_row_path = root / "parquet-row-index"
    parquet_block_path = root / "parquet-block-index"
    tantivy_path = root / "tantivy-index"

    if not source_path.exists():
        corpus = generate_corpus(
            source_path,
            rows=args.rows,
            tokens_per_document=args.tokens_per_document,
            vocabulary_size=args.vocabulary_size,
            source_row_group_rows=args.source_row_group_rows,
            seed=args.seed,
        )
        (root / "corpus.json").write_text(
            json.dumps(asdict(corpus), indent=2) + "\n", encoding="utf-8"
        )
    else:
        corpus = CorpusInfo(**json.loads((root / "corpus.json").read_text()))

    builds: dict[str, Any] = {}
    if not parquet_row_path.exists():
        builds["parquet_row"] = build_parquet_index(
            source_path,
            parquet_row_path,
            buckets=args.buckets,
            posting_layout="row",
            posting_block_docs=args.posting_block_docs,
            posting_row_group_rows=args.posting_row_group_rows,
            source_row_group_rows=args.source_row_group_rows,
        )
    if not parquet_block_path.exists():
        builds["parquet_block"] = build_parquet_index(
            source_path,
            parquet_block_path,
            buckets=args.buckets,
            posting_layout="block",
            posting_block_docs=args.posting_block_docs,
            posting_row_group_rows=args.posting_block_row_group_rows,
            source_row_group_rows=args.source_row_group_rows,
        )
    if not tantivy_path.exists():
        builds["tantivy"] = build_tantivy_index(source_path, tantivy_path)

    query_cases = (
        QueryCase("rare_term", "needle", False),
        QueryCase("common_term", "error", False),
        QueryCase("rare_phrase", "needle failed", True),
        QueryCase("common_phrase", "request failed", True),
    )
    parquet_row = ParquetPostingIndex(
        parquet_row_path,
        buckets=args.buckets,
        posting_layout="row",
    )
    parquet_block = ParquetPostingIndex(
        parquet_block_path,
        buckets=args.buckets,
        posting_layout="block",
    )
    full_scan = FullScanSearcher(source_path, parquet_row)
    tantivy_searcher = TantivySearcher(tantivy_path)
    source = SourceFetcher(source_path)
    results: dict[str, Any] = {}
    try:
        for query in query_cases:
            reference = full_scan.search(query, k=args.k)
            with_source_fetch(
                parquet_row.search(query, k=args.k),
                source,
                source_row_group_rows=args.source_row_group_rows,
            )
            with_source_fetch(
                parquet_block.search(query, k=args.k),
                source,
                source_row_group_rows=args.source_row_group_rows,
            )
            with_source_fetch(
                tantivy_searcher.search(query, k=args.k),
                source,
                source_row_group_rows=args.source_row_group_rows,
            )
            parquet_row_runs = [
                with_source_fetch(
                    parquet_row.search(query, k=args.k),
                    source,
                    source_row_group_rows=args.source_row_group_rows,
                )
                for _ in range(args.repetitions)
            ]
            parquet_block_runs = [
                with_source_fetch(
                    parquet_block.search(query, k=args.k),
                    source,
                    source_row_group_rows=args.source_row_group_rows,
                )
                for _ in range(args.repetitions)
            ]
            tantivy_runs = [
                with_source_fetch(
                    tantivy_searcher.search(query, k=args.k),
                    source,
                    source_row_group_rows=args.source_row_group_rows,
                )
                for _ in range(args.repetitions)
            ]
            reference_ids = [hit.doc_id for hit in reference.hits]
            parquet_row_ids = [hit.doc_id for hit in parquet_row_runs[0].hits]
            if parquet_row_ids != reference_ids:
                raise RuntimeError(
                    f"row-per-posting result mismatch for {query.name}: "
                    f"expected {reference_ids}, found {parquet_row_ids}"
                )
            parquet_block_ids = [hit.doc_id for hit in parquet_block_runs[0].hits]
            if parquet_block_ids != reference_ids:
                raise RuntimeError(
                    f"posting-block result mismatch for {query.name}: "
                    f"expected {reference_ids}, found {parquet_block_ids}"
                )
            tantivy_ids = [hit.doc_id for hit in tantivy_runs[0].hits]
            results[query.name] = {
                "query": asdict(query),
                "full_scan": summarize_runs([reference]),
                "parquet_row": summarize_runs(parquet_row_runs),
                "parquet_block": summarize_runs(parquet_block_runs),
                "tantivy": summarize_runs(tantivy_runs),
                "tantivy_topk_overlap": len(set(reference_ids) & set(tantivy_ids))
                / len(reference_ids),
            }
    finally:
        source.close()
        full_scan.close()
        parquet_row.close()
        parquet_block.close()

    return {
        "prototype": "parquet-bm25-phrase-v2",
        "configuration": {
            "rows": args.rows,
            "tokens_per_document": args.tokens_per_document,
            "vocabulary_size": args.vocabulary_size,
            "source_row_group_rows": args.source_row_group_rows,
            "posting_row_group_rows": args.posting_row_group_rows,
            "posting_block_docs": args.posting_block_docs,
            "posting_block_row_group_rows": args.posting_block_row_group_rows,
            "buckets": args.buckets,
            "k": args.k,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "tantivy": tantivy.__version__,
        },
        "corpus": asdict(corpus),
        "source_bytes": source_path.stat().st_size,
        "index_bytes": {
            "parquet_row": directory_bytes(parquet_row_path),
            "parquet_block": directory_bytes(parquet_block_path),
            "tantivy": directory_bytes(tantivy_path),
        },
        "builds": builds,
        "queries": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/tmp/relify-text-search-prototype"),
    )
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--tokens-per-document", type=int, default=160)
    parser.add_argument("--vocabulary-size", type=int, default=8_192)
    parser.add_argument("--source-row-group-rows", type=int, default=1_024)
    parser.add_argument("--posting-row-group-rows", type=int, default=4_096)
    parser.add_argument("--posting-block-docs", type=int, default=256)
    parser.add_argument("--posting-block-row-group-rows", type=int, default=64)
    parser.add_argument("--buckets", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    for name in (
        "rows",
        "tokens_per_document",
        "vocabulary_size",
        "source_row_group_rows",
        "posting_row_group_rows",
        "posting_block_docs",
        "posting_block_row_group_rows",
        "buckets",
        "k",
        "repetitions",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.vocabulary_size < 6:
        parser.error("--vocabulary-size must be at least 6")
    return args


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
