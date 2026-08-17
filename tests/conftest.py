from __future__ import annotations

import pytest
from support.capabilities import (
    CAPABILITY_NAMES,
    CapabilityRegistry,
    CapabilityState,
)
from support.config import TestEnvironmentError, load_test_environment

pytest_plugins = ("support.fixtures",)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("parqdb test environment")
    group.addoption(
        "--test-env",
        dest="test_env",
        metavar="PATH",
        help="load integration capabilities from a TOML file",
    )
    group.addoption(
        "--capabilities",
        action="store_true",
        help="probe configured capabilities, print the result, and exit",
    )
    group.addoption(
        "--require",
        dest="required_capabilities",
        default="",
        metavar="NAME[,NAME...]",
        help="fail immediately unless every named capability is configured",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires(*capabilities): require configured integration capabilities",
    )
    try:
        environment = load_test_environment(config.getoption("test_env"))
        required = _parse_capabilities(config.getoption("required_capabilities"))
        registry = CapabilityRegistry(environment)
        missing = [name for name in required if not registry.configured(name)]
    except (TestEnvironmentError, ValueError) as error:
        raise pytest.UsageError(str(error)) from error
    if missing:
        source = str(environment.path) if environment.path else "no test-env file"
        raise pytest.UsageError(
            f"required capabilities are not configured in {source}: "
            + ", ".join(missing)
        )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> int | None:
    if not config.getoption("capabilities"):
        return None
    try:
        environment = load_test_environment(config.getoption("test_env"))
        registry = CapabilityRegistry(environment)
    except TestEnvironmentError as error:
        print(f"test environment error: {error}")
        return int(pytest.ExitCode.USAGE_ERROR)

    source = str(environment.path) if environment.path else "not configured"
    print(f"test environment: {source}")
    failed = False
    for name in CAPABILITY_NAMES:
        result = registry.inspect(name)
        print(f"{name:10} {result.state.value:9} {result.detail}")
        failed = failed or result.state == CapabilityState.FAILED
    return int(pytest.ExitCode.TESTS_FAILED if failed else pytest.ExitCode.OK)


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    try:
        environment = load_test_environment(config.getoption("test_env"))
    except TestEnvironmentError as error:
        raise pytest.UsageError(str(error)) from error
    for item in items:
        required: set[str] = set()
        for marker in item.iter_markers("requires"):
            required.update(str(name) for name in marker.args)
        try:
            for name in required:
                if name not in CAPABILITY_NAMES:
                    supported = ", ".join(CAPABILITY_NAMES)
                    raise ValueError(
                        f"unknown test capability {name!r}; "
                        f"expected one of: {supported}"
                    )
            missing = sorted(
                name for name in required if not environment.configured(name)
            )
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
        if missing:
            item.add_marker(
                pytest.mark.skip(
                    reason="test capabilities not configured: " + ", ".join(missing)
                )
            )


def _parse_capabilities(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    for name in names:
        if name not in CAPABILITY_NAMES:
            supported = ", ".join(CAPABILITY_NAMES)
            raise ValueError(
                f"unknown test capability {name!r}; expected one of: {supported}"
            )
    return names
