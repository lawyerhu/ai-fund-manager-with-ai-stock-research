"""Capability checks for the official IBKR TWS API Python client."""
from __future__ import annotations

import inspect
import sys
from typing import Any


OFFICIAL_API_ERROR = "OFFICIAL IBKR TWS API NOT INSTALLED OR INCOMPATIBLE"


def _check(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status, "message": message}


def _signature(cls: Any, method_name: str):
    method = getattr(cls, method_name, None) if cls is not None else None
    if method is None:
        return None
    try:
        return inspect.signature(method)
    except (TypeError, ValueError):
        return None


def inspect_api_capabilities(wrapper_cls=None, client_cls=None, order_cancel_cls=None) -> dict[str, Any]:
    """Check the callback and cancellation surface required by this adapter.

    Supplying classes makes the check deterministic for tests. With no classes,
    the function imports the official API installed in the current interpreter.
    """
    checks: list[dict[str, str]] = []
    version: str | None = None
    if wrapper_cls is None and client_cls is None and order_cancel_cls is None:
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
            from ibapi.order_cancel import OrderCancel

            wrapper_cls, client_cls, order_cancel_cls = EWrapper, EClient, OrderCancel
            try:
                from ibapi import version as ibkr_version

                version = str(getattr(ibkr_version, "__version__", getattr(ibkr_version, "VERSION", "unknown")))
            except ImportError:
                version = None
        except ImportError:
            return {
                "ok": False,
                "checks": [_check("official ibapi", "FAIL", OFFICIAL_API_ERROR)],
                "api_version": None,
            }

    error_signature = _signature(wrapper_cls, "error")
    error_names = {parameter.name for parameter in error_signature.parameters.values()} if error_signature else set()
    if "errorTime" in error_names:
        checks.append(_check("EWrapper.error", "PASS", "current callback includes errorTime"))
    else:
        checks.append(_check("EWrapper.error", "FAIL", "EWrapper.error must include errorTime"))

    if getattr(wrapper_cls, "commissionAndFeesReport", None) is not None:
        checks.append(_check("commissionAndFeesReport", "PASS", "current execution cost callback is available"))
    else:
        checks.append(_check("commissionAndFeesReport", "FAIL", "EWrapper.commissionAndFeesReport is required"))

    if order_cancel_cls is not None:
        checks.append(_check("OrderCancel", "PASS", "current cancellation object is available"))
    else:
        checks.append(_check("OrderCancel", "FAIL", "OrderCancel is required by the current IBKR API"))

    cancel_signature = _signature(client_cls, "cancelOrder")
    cancel_names = list(cancel_signature.parameters) if cancel_signature else []
    if cancel_signature and len(cancel_names) >= 3:
        checks.append(_check("EClient.cancelOrder", "PASS", "cancelOrder accepts an OrderCancel argument"))
    else:
        checks.append(_check("EClient.cancelOrder", "FAIL", "EClient.cancelOrder must accept an OrderCancel argument"))

    if version:
        checks.append(_check("IBKR API version", "PASS", version))
    else:
        checks.append(_check("IBKR API version", "WARN", "API version could not be read from the installed client"))

    return {
        "ok": not any(item["status"] == "FAIL" for item in checks),
        "checks": checks,
        "api_version": version,
    }


def main() -> int:
    result = inspect_api_capabilities()
    print("IBKR API diagnostics")
    for item in result["checks"]:
        print(f"{item['status']}: {item['name']}: {item['message']}")
    if not result["ok"]:
        print(OFFICIAL_API_ERROR)
        print("Download the official TWS API that matches TWS/Gateway and install its Python client into this project's .venv.")
        return 1
    print("PASS: required IBKR callback and cancellation capabilities are available")
    print("WARN: socket, account, reconciliation, and market-data permissions require a configured TWS/Gateway session")
    return 0


if __name__ == "__main__":
    sys.exit(main())
