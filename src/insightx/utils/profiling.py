"""Resource Profiling Utility for InsightX AI 3.0.

Provides dual measurement:
1. Python heap allocations via tracemalloc.
2. OS Process-level peak RSS via psutil (capturing native C++ buffers in DuckDB/Arrow).
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import time
import tracemalloc
from typing import Generator, Optional
import psutil


@dataclass(frozen=True)
class ProfilingReport:
    """Immutable resource profiling metrics capturing Python heap and OS process RSS."""
    label: str
    duration_seconds: float
    python_heap_current_mb: float
    python_heap_peak_mb: float
    process_current_rss_mb: float
    process_peak_rss_mb: float


def get_process_peak_rss_mb(process: Optional[psutil.Process] = None) -> float:
    """Returns the actual peak RSS (Physical RAM) in megabytes.
    
    On Windows, uses kernel-tracked 'peak_wset' (Peak Working Set).
    On Unix/macOS, queries getrusage ru_maxrss or falls back to rss.
    """
    if process is None:
        process = psutil.Process(os.getpid())
    
    mem = process.memory_info()
    # Windows native peak working set
    if hasattr(mem, "peak_wset"):
        return float(mem.peak_wset) / (1024.0 * 1024.0)
    
    # Unix native peak RSS via resource module
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, ru_maxrss is in Kilobytes; on macOS/BSD, it is in Bytes
        import sys
        if sys.platform == "darwin":
            return float(rusage) / (1024.0 * 1024.0)
        return float(rusage) / 1024.0
    except (ImportError, AttributeError):
        return float(mem.rss) / (1024.0 * 1024.0)


@contextmanager
def profile_execution(label: str = "Execution") -> Generator[dict, None, None]:
    """Context manager measuring both Python heap and OS process-level peak memory.
    
    Yields:
        dict: Populated upon context exit with 'report': ProfilingReport.
    """
    process = psutil.Process(os.getpid())
    
    # Start tracemalloc
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    
    start_time = time.perf_counter()
    metrics_container: dict = {}

    try:
        yield metrics_container
    finally:
        end_time = time.perf_counter()
        current_heap, peak_heap = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()

        current_rss = float(process.memory_info().rss) / (1024.0 * 1024.0)
        peak_rss = get_process_peak_rss_mb(process)
        duration = end_time - start_time

        report = ProfilingReport(
            label=label,
            duration_seconds=round(duration, 4),
            python_heap_current_mb=round(current_heap / (1024.0 * 1024.0), 3),
            python_heap_peak_mb=round(peak_heap / (1024.0 * 1024.0), 3),
            process_current_rss_mb=round(current_rss, 3),
            process_peak_rss_mb=round(peak_rss, 3),
        )
        metrics_container["report"] = report
