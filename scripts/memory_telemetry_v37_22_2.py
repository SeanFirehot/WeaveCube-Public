"""CubeLab v37.22.2.1 robust cross-platform memory telemetry.

Priority:
1. psutil when installed.
2. Windows native K32GetProcessMemoryInfo.
3. Windows Psapi.GetProcessMemoryInfo.
4. Linux /proc fallback.

The memory-safe audit must not silently run without RSS telemetry.
"""
from __future__ import annotations

import os
from pathlib import Path

_MB = 1024.0 * 1024.0

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def _windows_process_rss_mb():
    import ctypes
    from ctypes import wintypes

    SIZE_T = ctypes.c_size_t

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", SIZE_T),
            ("WorkingSetSize", SIZE_T),
            ("QuotaPeakPagedPoolUsage", SIZE_T),
            ("QuotaPagedPoolUsage", SIZE_T),
            ("QuotaPeakNonPagedPoolUsage", SIZE_T),
            ("QuotaNonPagedPoolUsage", SIZE_T),
            ("PagefileUsage", SIZE_T),
            ("PeakPagefileUsage", SIZE_T),
        ]

    kernel32 = ctypes.WinDLL(
        "kernel32",
        use_last_error=True,
    )

    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = kernel32.GetCurrentProcess()

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)

    # Windows 7+ / PSAPI v2 path first.
    try:
        fn = kernel32.K32GetProcessMemoryInfo
        fn.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        fn.restype = wintypes.BOOL
        ok = fn(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return float(counters.WorkingSetSize) / _MB
    except (AttributeError, OSError):
        pass

    # Compatible Psapi fallback.
    psapi = ctypes.WinDLL(
        "psapi",
        use_last_error=True,
    )
    fn = psapi.GetProcessMemoryInfo
    fn.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    fn.restype = wintypes.BOOL

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)

    ok = fn(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        err = ctypes.get_last_error()
        raise OSError(
            err,
            "GetProcessMemoryInfo failed",
        )

    return float(counters.WorkingSetSize) / _MB


def current_process_rss_mb():
    if psutil is not None:
        try:
            return (
                float(
                    psutil.Process(
                        os.getpid()
                    ).memory_info().rss
                )
                / _MB
            )
        except Exception:
            pass

    if os.name == "nt":
        try:
            return _windows_process_rss_mb()
        except Exception:
            return None

    # Linux fallback for dev/CI.
    try:
        statm = Path("/proc/self/statm").read_text().split()
        pages = int(statm[1])
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / _MB
    except Exception:
        return None


def system_memory_mb():
    """Return {'total_mb','available_mb'} or None."""
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return {
                "total_mb": float(vm.total) / _MB,
                "available_mb": float(vm.available) / _MB,
            }
        except Exception:
            pass

    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(st)

            kernel32 = ctypes.WinDLL(
                "kernel32",
                use_last_error=True,
            )
            fn = kernel32.GlobalMemoryStatusEx
            fn.argtypes = [
                ctypes.POINTER(MEMORYSTATUSEX)
            ]
            fn.restype = wintypes.BOOL

            ok = fn(ctypes.byref(st))
            if not ok:
                return None

            return {
                "total_mb": float(st.ullTotalPhys) / _MB,
                "available_mb": float(st.ullAvailPhys) / _MB,
            }
        except Exception:
            return None

    try:
        data = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            data[key] = (
                float(value.strip().split()[0])
                / 1024.0
            )
        return {
            "total_mb": data["MemTotal"],
            "available_mb": data.get(
                "MemAvailable",
                data.get("MemFree"),
            ),
        }
    except Exception:
        return None


def format_mb(value, *, decimals=1):
    if value is None:
        return "unknown"
    return f"{float(value):.{int(decimals)}f} MB"


class PeakRSS:
    def __init__(self):
        self.start_mb = current_process_rss_mb()
        self.peak_mb = self.start_mb

    def sample(self):
        value = current_process_rss_mb()
        if value is not None:
            if (
                self.peak_mb is None
                or value > self.peak_mb
            ):
                self.peak_mb = value
        return value

    def snapshot(self):
        return {
            "rss_start_mb": self.start_mb,
            "rss_peak_mb": self.peak_mb,
            "rss_current_mb": current_process_rss_mb(),
        }
