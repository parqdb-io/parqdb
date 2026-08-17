from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from verify_wheel import verify_wheel

SUPPORTED_PYTHONS = ("3.11", "3.12", "3.13", "3.14")
PYPI_INDEX = "https://pypi.org/simple"


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument("wheel", type=Path)
    command.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).parents[1],
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
    with TemporaryDirectory(prefix="parqdb-release-") as directory:
        root = Path(directory)
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
            check=True,
        )

        python_versions = args.python_versions or SUPPORTED_PYTHONS
        for python_version in python_versions:
            with TemporaryDirectory(
                prefix=f"parqdb-python-{python_version}-"
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
                requirement = "parqdb[iceberg] @ " + wheel.as_uri()
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
