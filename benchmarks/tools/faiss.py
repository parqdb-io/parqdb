"""Standalone Faiss IVF benchmark entry point."""

from __future__ import annotations

import argparse
import sys

from benchmarks.tools.ivf import main as run


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = argparse.ArgumentParser(
        prog="python -m benchmarks.tools.faiss",
        description="Build or query the Faiss IVF baseline.",
    )
    command.add_argument("operation", choices=("build", "query"))
    if not arguments or arguments[0] in {"-h", "--help"}:
        command.parse_args(arguments)
        return
    operation = arguments.pop(0)
    if operation not in {"build", "query"}:
        command.error(f"invalid operation: {operation}")
    run(operation, "faiss", arguments)


if __name__ == "__main__":
    main()
