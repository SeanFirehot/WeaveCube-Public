from __future__ import annotations

import sys
import types


def ensure_legacy_compat() -> None:
    """
    Make historical POSIX-telemetry imports safe on Windows.

    This changes telemetry only. Cube semantics are untouched.
    """

    if sys.platform != "win32":
        return

    if "resource" in sys.modules:
        return

    try:
        import resource  # type: ignore
        return
    except ModuleNotFoundError:
        pass

    r = types.ModuleType(
        "resource"
    )

    class _RUsage:
        ru_maxrss = 0
        ru_utime = 0.0
        ru_stime = 0.0

    r.RUSAGE_SELF = 0
    r.RUSAGE_CHILDREN = -1
    r.RLIM_INFINITY = -1

    r.RLIMIT_AS = 0
    r.RLIMIT_DATA = 1
    r.RLIMIT_RSS = 2

    r.getrusage = lambda _who: _RUsage()
    r.getrlimit = lambda _which: (-1, -1)
    r.setrlimit = lambda _which, _limits: None

    def _getattr(name):
        if name.startswith(
            (
                "RLIMIT_",
                "RUSAGE_",
            )
        ):
            return 0

        raise AttributeError(
            name
        )

    r.__getattr__ = _getattr

    sys.modules[
        "resource"
    ] = r
