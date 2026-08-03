from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import EntryPoint, PackageNotFoundError, entry_points, version
from typing import Any

from ._builtin import BUILTIN_INFO, BUILTIN_LOADERS
from .v1 import BACKEND_API_VERSION, BackendPlugin

ENTRY_POINT_GROUP = "relify.backends"


class BackendRegistryError(RuntimeError):
    """Base error for backend discovery and loading."""


class BackendNotFoundError(BackendRegistryError):
    """The requested backend is not installed or registered."""


class BackendAlreadyRegisteredError(BackendRegistryError):
    """More than one backend owns the same registry name."""


class IncompatibleBackendError(BackendRegistryError):
    """A backend targets an incompatible extension API version."""


class InvalidBackendPluginError(BackendRegistryError):
    """An entry point did not expose a valid backend plugin."""


class BackendLoadError(BackendRegistryError):
    """An installed backend failed while importing its plugin object."""


@dataclass(frozen=True, slots=True)
class InstalledBackend:
    """One installed backend without importing its plugin module."""

    name: str
    distribution: str
    version: str | None
    built_in: bool


_registered: dict[str, BackendPlugin] = {}
_loaded: dict[str, BackendPlugin] = {}


def installed() -> tuple[InstalledBackend, ...]:
    """List installed backends without loading third-party plugin modules."""
    discovered: dict[str, InstalledBackend] = {}
    relify_version = _distribution_version("relify")
    for name, info in BUILTIN_INFO.items():
        discovered[name] = InstalledBackend(
            name=name,
            distribution=info.distribution,
            version=relify_version,
            built_in=True,
        )
    for name, plugin in _registered.items():
        discovered[name] = InstalledBackend(
            name=name,
            distribution=plugin.info.distribution,
            version=_distribution_version(plugin.info.distribution),
            built_in=name in BUILTIN_INFO,
        )
    for endpoint in _entry_points():
        if endpoint.name in discovered:
            if endpoint.name in _registered:
                continue
            raise BackendAlreadyRegisteredError(
                f"backend name is registered more than once: {endpoint.name}"
            )
        distribution = None
        if endpoint.dist is not None:
            metadata = endpoint.dist.metadata
            distribution = metadata["Name"] if "Name" in metadata else None
        distribution = distribution or endpoint.module.split(".", 1)[0]
        discovered[endpoint.name] = InstalledBackend(
            name=endpoint.name,
            distribution=distribution,
            version=endpoint.dist.version if endpoint.dist is not None else None,
            built_in=False,
        )
    return tuple(discovered[name] for name in sorted(discovered))


def register(plugin: BackendPlugin, *, replace: bool = False) -> None:
    """Register a backend in the current process."""
    _validate_plugin(plugin)
    name = plugin.info.name
    if not replace and (
        name in _registered
        or name in BUILTIN_LOADERS
        or any(endpoint.name == name for endpoint in _entry_points())
    ):
        raise BackendAlreadyRegisteredError(f"backend is already registered: {name}")
    _registered[name] = plugin
    _loaded[name] = plugin


def load(name: str) -> BackendPlugin:
    """Load one built-in, installed, or in-process backend plugin by name."""
    if not isinstance(name, str) or not name:
        raise ValueError("backend name must not be empty")
    cached = _loaded.get(name)
    if cached is not None:
        return cached
    registered = _registered.get(name)
    if registered is not None:
        return registered
    loader = BUILTIN_LOADERS.get(name)
    if loader is not None:
        plugin = loader()
        _validate_plugin(plugin)
        _loaded[name] = plugin
        return plugin
    matching = [endpoint for endpoint in _entry_points() if endpoint.name == name]
    if not matching:
        raise BackendNotFoundError(
            f"backend is not installed: {name}; install its Relify integration package"
        )
    if len(matching) != 1:
        raise BackendAlreadyRegisteredError(
            f"backend name is registered more than once: {name}"
        )
    try:
        plugin = matching[0].load()
    except Exception as error:
        raise BackendLoadError(
            f"failed to load backend {name!r} from entry point {matching[0].value!r}"
        ) from error
    _validate_plugin(plugin)
    if plugin.info.name != name:
        raise InvalidBackendPluginError(
            f"entry point {name!r} exposed backend {plugin.info.name!r}"
        )
    _loaded[name] = plugin
    return plugin


def connect(name: str, *args: Any, **kwargs: Any) -> Any:
    """Connect through a dynamically selected backend."""
    return load(name).connect(*args, **kwargs)


def _entry_points() -> tuple[EntryPoint, ...]:
    return tuple(entry_points(group=ENTRY_POINT_GROUP))


def _validate_plugin(plugin: object) -> None:
    if not isinstance(plugin, BackendPlugin):
        raise InvalidBackendPluginError(
            "backend entry point must expose a relify.backends.v1.BackendPlugin"
        )
    if plugin.info.api_version != BACKEND_API_VERSION:
        raise IncompatibleBackendError(
            f"backend {plugin.info.name!r} uses API version "
            f"{plugin.info.api_version}; Relify requires {BACKEND_API_VERSION}"
        )


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _clear_for_tests() -> None:
    _registered.clear()
    _loaded.clear()
