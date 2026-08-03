from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote


def cargo_purl(name: str, version: str) -> str:
    return f"pkg:cargo/{quote(name, safe='')}@{quote(version, safe='')}"


def lock_checksums(path: Path) -> dict[tuple[str, str, str], str]:
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    return {
        (package["name"], package["version"], package["source"]): package["checksum"]
        for package in lock["package"]
        if "source" in package and "checksum" in package
    }


def cargo_metadata(repository: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "cargo",
            "metadata",
            "--format-version",
            "1",
            "--locked",
            "--manifest-path",
            str(repository / "python" / "Cargo.toml"),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def generate(repository: Path) -> dict[str, Any]:
    project = tomllib.loads(
        (repository / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    distribution_name = project["name"]
    distribution_version = project["version"]
    if not isinstance(distribution_name, str) or not isinstance(
        distribution_version, str
    ):
        raise ValueError("Python project name and version must be strings")

    metadata = cargo_metadata(repository)
    checksums = lock_checksums(repository / "Cargo.lock")
    packages = {package["id"]: package for package in metadata["packages"]}
    references = {
        package_id: cargo_purl(package["name"], package["version"])
        for package_id, package in packages.items()
    }
    if len(set(references.values())) != len(references):
        raise ValueError("Cargo packages do not have unique name/version pairs")

    components = []
    for package_id in sorted(
        packages,
        key=lambda value: (
            packages[value]["name"],
            packages[value]["version"],
        ),
    ):
        package = packages[package_id]
        component: dict[str, Any] = {
            "type": "library",
            "bom-ref": references[package_id],
            "name": package["name"],
            "version": package["version"],
            "scope": "required",
            "licenses": [{"expression": package["license"]}],
            "purl": references[package_id],
        }
        if package["description"]:
            component["description"] = package["description"]
        if package["repository"]:
            component["externalReferences"] = [
                {
                    "type": "vcs",
                    "url": package["repository"],
                }
            ]
        source = package["source"]
        checksum = (
            checksums.get((package["name"], package["version"], source))
            if source is not None
            else None
        )
        if checksum is not None:
            component["hashes"] = [{"alg": "SHA-256", "content": checksum}]
        components.append(component)

    dependency_graph = []
    for node in metadata["resolve"]["nodes"]:
        dependency_graph.append(
            {
                "ref": references[node["id"]],
                "dependsOn": sorted(
                    references[dependency] for dependency in node["dependencies"]
                ),
            }
        )
    dependency_graph.sort(key=lambda dependency: dependency["ref"])

    root_package_id = metadata["resolve"]["root"]
    if root_package_id is None:
        raise ValueError("Cargo metadata did not identify the root package")
    distribution_ref = (
        f"pkg:pypi/{quote(distribution_name, safe='')}@"
        f"{quote(distribution_version, safe='')}"
    )
    dependency_graph.insert(
        0,
        {
            "ref": distribution_ref,
            "dependsOn": [references[root_package_id]],
        },
    )
    lock_digest = hashlib.sha256((repository / "Cargo.lock").read_bytes()).hexdigest()
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "library",
                "bom-ref": distribution_ref,
                "name": distribution_name,
                "version": distribution_version,
                "licenses": [{"expression": "MIT AND Apache-2.0"}],
                "purl": distribution_ref,
            },
            "properties": [
                {
                    "name": "relify:cargo-lock-sha256",
                    "value": lock_digest,
                }
            ],
        },
        "components": components,
        "dependencies": dependency_graph,
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    command.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).parents[1],
    )
    command.add_argument(
        "--output",
        type=Path,
        default=Path("sboms/relify-python.cyclonedx.json"),
    )
    return command


def main() -> None:
    args = parser().parse_args()
    repository = args.repository.resolve()
    output = args.output if args.output.is_absolute() else repository / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(generate(repository), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
