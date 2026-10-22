"""Small dependency-free system and NVIDIA GPU telemetry helpers."""
import os
import shutil
import subprocess


class CpuSampler:
    def __init__(self):
        self._previous = None

    def sample(self):
        current = self._read_counters()
        if current is None:
            return None
        previous, self._previous = self._previous, current
        if previous is None:
            return 0.0
        total = current[0] - previous[0]
        idle = current[1] - previous[1]
        return round(max(0.0, min(100.0, 100.0 * (total - idle) / total)), 1) if total > 0 else 0.0

    @staticmethod
    def _read_counters():
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                class FILETIME(ctypes.Structure):
                    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]

                idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
                if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                    return None

                def number(value):
                    return (value.dwHighDateTime << 32) | value.dwLowDateTime

                k, u, i = number(kernel), number(user), number(idle)
                return k + u, i
            except Exception:
                return None
        try:
            with open("/proc/stat", "r", encoding="ascii") as f:
                values = [int(v) for v in f.readline().split()[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return sum(values), idle
        except (OSError, ValueError, IndexError):
            try:
                load = os.getloadavg()[0]
                return 100, max(0, 100 - min(100, load * 100 / (os.cpu_count() or 1)))
            except (AttributeError, OSError):
                return None


def gpu_metrics():
    """Return NVIDIA utilization and VRAM if nvidia-smi is available."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    query = [exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(query, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            used, total = float(parts[2]), float(parts[3])
            gpus.append({
                "name": parts[0],
                "utilization": float(parts[1]),
                "vram_used_mib": used,
                "vram_total_mib": total,
            })
        except ValueError:
            continue
    return gpus


def system_metrics(cpu_sampler):
    cpu = cpu_sampler.sample()
    return {"cpu_percent": cpu, "ram": ram_metrics(), "gpus": gpu_metrics()}


def ram_metrics():
    """Return system RAM usage in MiB and percent when the platform exposes it."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            total, available = status.ullTotalPhys, status.ullAvailPhys
        elif os.path.exists("/proc/meminfo"):
            values = {}
            with open("/proc/meminfo", "r", encoding="ascii") as f:
                for line in f:
                    key, value = line.split(":", 1)
                    values[key] = int(value.strip().split()[0])
            total = values["MemTotal"] * 1024
            available = values.get("MemAvailable", values.get("MemFree", 0)) * 1024
        else:
            import psutil
            memory = psutil.virtual_memory()
            total, available = memory.total, memory.available
        used = max(0, total - available)
        return {"used_mib": round(used / 1048576, 1), "total_mib": round(total / 1048576, 1),
                "percent": round(used * 100 / total, 1) if total else 0.0}
    except (OSError, ImportError, KeyError, ValueError):
        return None
