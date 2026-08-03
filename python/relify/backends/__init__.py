"""Discovery and public extension interfaces for Relify backends."""

from ._registry import (
    ENTRY_POINT_GROUP,
    BackendAlreadyRegisteredError,
    BackendLoadError,
    BackendNotFoundError,
    BackendRegistryError,
    IncompatibleBackendError,
    InstalledBackend,
    InvalidBackendPluginError,
    connect,
    installed,
    load,
    register,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "BackendAlreadyRegisteredError",
    "BackendLoadError",
    "BackendNotFoundError",
    "BackendRegistryError",
    "IncompatibleBackendError",
    "InstalledBackend",
    "InvalidBackendPluginError",
    "connect",
    "installed",
    "load",
    "register",
]
