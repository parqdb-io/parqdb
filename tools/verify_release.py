from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from verify_wheel import verify_wheel

SUPPORTED_PYTHONS = ("3.11", "3.12", "3.13", "3.14")
PYPI_INDEX = "https://pypi.org/simple"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("wheel", type=Path)
    command.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).parents[1],
    )
    command.add_argument(
        "--rebuilt-wheel",
        type=Path,
        help="second independently built wheel to compare instead of rebuilding",
    )
    command.add_argument(
        "--python",
        dest="python_versions",
        action="append",
        choices=SUPPORTED_PYTHONS,
        help="Python minor to verify; repeat to override the default full matrix",
    )
    return command


def main() -> None:
    args = parser().parse_args()
    repository = args.repository.resolve()
    wheel = args.wheel.resolve()
    verify_wheel(repository, wheel)
    with TemporaryDirectory(prefix="relify-release-") as directory:
        root = Path(directory)
        if args.rebuilt_wheel is None:
            rebuilt_directory = root / "rebuilt"
            rebuild_target = repository / "target" / "package"
            if rebuild_target.exists():
                raise RuntimeError(
                    f"clean rebuild target already exists: {rebuild_target}"
                )
            rebuild_environment = os.environ.copy()
            rebuild_environment["CARGO_TARGET_DIR"] = str(rebuild_target)
            try:
                subprocess.run(
                    [
                        "maturin",
                        "build",
                        "--release",
                        "--locked",
                        "--compatibility",
                        "pypi",
                        "--out",
                        str(rebuilt_directory),
                    ],
                    cwd=repository,
                    env=rebuild_environment,
                    check=True,
                )
            finally:
                shutil.rmtree(rebuild_target, ignore_errors=True)
            rebuilt = rebuilt_directory / wheel.name
        else:
            rebuilt = args.rebuilt_wheel.resolve()
        if not rebuilt.is_file():
            raise RuntimeError(f"rebuild did not produce {wheel.name}")
        if rebuilt.name != wheel.name:
            raise RuntimeError(
                f"rebuild produced a different wheel: {rebuilt.name} != {wheel.name}"
            )
        verify_wheel(repository, rebuilt)
        original_hash = sha256(wheel)
        rebuilt_hash = sha256(rebuilt)
        if original_hash != rebuilt_hash:
            raise RuntimeError(
                f"wheel rebuild is not reproducible: {original_hash} != {rebuilt_hash}"
            )
        print(f"reproduced {wheel.name}: sha256={original_hash}")

        requirements = root / "requirements.txt"
        uv_environment = os.environ.copy()
        uv_environment["UV_DEFAULT_INDEX"] = PYPI_INDEX
        subprocess.run(
            [
                "uv",
                "export",
                "--locked",
                "--all-extras",
                "--no-default-groups",
                "--group",
                "wheel-test",
                "--no-emit-project",
                "--no-header",
                "--no-annotate",
                "--quiet",
                "--output-file",
                str(requirements),
            ],
            cwd=repository,
            env=uv_environment,
            check=True,
        )

        python_versions = args.python_versions or SUPPORTED_PYTHONS
        for python_version in python_versions:
            with TemporaryDirectory(
                prefix=f"relify-python-{python_version}-"
            ) as environment_directory:
                environment = Path(environment_directory) / "venv"
                subprocess.run(
                    [
                        "uv",
                        "venv",
                        str(environment),
                        "--python",
                        f"cpython-{python_version}",
                    ],
                    cwd=repository,
                    check=True,
                )
                python = environment / (
                    "Scripts/python.exe" if os.name == "nt" else "bin/python"
                )
                requirement = "relify[iceberg,spark,starrocks] @ " + wheel.as_uri()
                subprocess.run(
                    [
                        "uv",
                        "pip",
                        "install",
                        "--python",
                        str(python),
                        requirement,
                        "--requirements",
                        str(requirements),
                    ],
                    cwd=repository,
                    env=uv_environment,
                    check=True,
                )
                subprocess.run(
                    [str(python), str(repository / "tools" / "smoke_wheel.py")],
                    cwd=environment_directory,
                    check=True,
                )
                subprocess.run(
                    [
                        str(python),
                        "-m",
                        "pytest",
                        "-q",
                        str(repository / "tests" / "python"),
                        str(repository / "tests" / "interop"),
                    ],
                    cwd=environment_directory,
                    check=True,
                )


if __name__ == "__main__":
    main()
