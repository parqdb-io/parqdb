from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from . import _native
from .errors import ServiceUnavailableError, UnsupportedOperationError


@dataclass(frozen=True, slots=True)
class HttpError:
    status: int
    code: str
    message: str
    request_id: str

    def json(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "request_id": self.request_id,
            }
        }


_SERVER_ERRORS: tuple[tuple[type[BaseException], int, str], ...] = (
    (_native.QueryQueueFullError, 429, "query_queue_full"),
    (_native.QueryQueueTimeoutError, 429, "query_queue_timeout"),
    (_native.IndexNotFoundError, 404, "index_not_found"),
    (_native.AlreadyExistsError, 409, "already_exists"),
    (_native.BuildAlreadyRunningError, 409, "build_already_running"),
    (_native.AmbiguousIndexError, 409, "ambiguous_index"),
    (_native.InvalidSchemaError, 400, "invalid_schema"),
    (_native.InvalidMetadataError, 400, "invalid_metadata"),
    (_native.InvalidArgumentError, 400, "invalid_argument"),
    (UnsupportedOperationError, 501, "unsupported_operation"),
    (TypeError, 400, "invalid_argument"),
    (ValueError, 400, "invalid_argument"),
    (_native.CatalogError, 500, "catalog_error"),
    (_native.StorageError, 500, "storage_error"),
    (_native.BackendError, 500, "backend_error"),
)

_CLIENT_ERRORS: dict[str, type[BaseException]] = {
    "query_queue_full": _native.QueryQueueFullError,
    "query_queue_timeout": _native.QueryQueueTimeoutError,
    "index_not_found": _native.IndexNotFoundError,
    "already_exists": _native.AlreadyExistsError,
    "build_already_running": _native.BuildAlreadyRunningError,
    "ambiguous_index": _native.AmbiguousIndexError,
    "invalid_schema": _native.InvalidSchemaError,
    "invalid_metadata": _native.InvalidMetadataError,
    "invalid_argument": _native.InvalidArgumentError,
    "unsupported_operation": UnsupportedOperationError,
    "catalog_error": _native.CatalogError,
    "storage_error": _native.StorageError,
    "backend_error": _native.BackendError,
}


def classify_server_error(error: BaseException, request_id: str) -> HttpError:
    for error_type, status, code in _SERVER_ERRORS:
        if isinstance(error, error_type):
            return HttpError(status, code, str(error), request_id)
    return HttpError(500, "internal", "internal server error", request_id)


def raise_remote_error(response: httpx.Response, body: object) -> None:
    code = "remote_error"
    message = f"Relify server returned HTTP {response.status_code}"
    request_id = response.headers.get("x-request-id")
    if isinstance(body, dict):
        value = body.get("error")
        if isinstance(value, dict):
            if isinstance(value.get("code"), str):
                code = value["code"]
            if isinstance(value.get("message"), str):
                message = value["message"]
            if isinstance(value.get("request_id"), str):
                request_id = value["request_id"]
    if request_id:
        message = f"{message} (request_id={request_id})"
    error_type = _CLIENT_ERRORS.get(code, _native.BackendError)
    raise error_type(message)


def unavailable(error: httpx.HTTPError) -> ServiceUnavailableError:
    return ServiceUnavailableError(f"Relify server is unavailable: {error}")
