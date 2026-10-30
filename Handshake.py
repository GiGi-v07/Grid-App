import os
import sys
import json
import time
import ctypes
import platform


def get_memory_gb():
    """
    Fetches (total_ram_gb, available_ram_gb) using standard library calls
    across Windows, Linux, and macOS without requiring third-party packages.
    """
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_gb = stat.ullAvailPhys / (1024 ** 3)
            return round(total_gb, 2), round(avail_gb, 2)
        elif os.path.exists("/proc/meminfo"):
            mem = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        mem[parts[0].strip()] = int(parts[1].strip().split()[0])
            total_gb = mem.get("MemTotal", 4 * 1024 * 1024) / (1024 ** 2)
            avail_gb = mem.get("MemAvailable", mem.get("MemFree", 2 * 1024 * 1024)) / (1024 ** 2)
            return round(total_gb, 2), round(avail_gb, 2)
    except Exception:
        pass
    return 4.0, 2.0


def measure_cpu_throughput():
    """
    Runs a fast ~0.1s floating-point micro-benchmark to estimate single-core
    CPU execution speed under current system load.
    Returns relative throughput score.
    """
    t0 = time.perf_counter()
    ops = 0
    acc = 1.000001
    while (time.perf_counter() - t0) < 0.1:
        for _ in range(5000):
            acc = (acc * 1.000003) + 0.000001
        ops += 5000
    elapsed = max(0.001, time.perf_counter() - t0)
    # Million operations per second
    mops = (ops / elapsed) / 1e6
    return round(mops, 2)


def collect_system_specs():
    """
    Gathers CPU core count, RAM capacity, OS details, and CPU benchmark score,
    and computes a unified 'spec_score' used to split AI datasets proportionally.
    """
    cores = os.cpu_count() or 1
    total_ram_gb, avail_ram_gb = get_memory_gb()
    cpu_mops = measure_cpu_throughput()

    # Composite hardware capability score combining cores, available RAM, and CPU speed
    spec_score = round((cores * 2.5) + (avail_ram_gb * 1.5) + (cpu_mops * 0.5), 2)
    spec_score = max(1.0, spec_score)

    specs = {
        "status": "ONLINE",
        "cores": cores,
        "total_ram_gb": total_ram_gb,
        "avail_ram_gb": avail_ram_gb,
        "cpu_mops": cpu_mops,
        "spec_score": spec_score,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    print(f"HANDSHAKE_SPECS:{json.dumps(specs)}")


if __name__ == "__main__":
    collect_system_specs()

