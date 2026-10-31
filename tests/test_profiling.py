"""Tests for Resource Profiling Utilities."""

import time
from insightx.utils.profiling import ProfilingReport, get_process_peak_rss_mb, profile_execution


def test_get_process_peak_rss():
    """Verifies that peak RSS helper returns a positive floating point value."""
    peak_rss = get_process_peak_rss_mb()
    assert isinstance(peak_rss, float)
    assert peak_rss > 0.0


def test_profile_execution_dual_measurement():
    """Verifies that profile_execution captures both Python heap and OS peak RSS separately."""
    with profile_execution(label="MemoryAllocationTest") as metrics:
        # Allocate a temporary Python buffer (~5MB)
        large_list = [i for i in range(500000)]
        time.sleep(0.05)
        assert len(large_list) == 500000

    report = metrics["report"]
    assert isinstance(report, ProfilingReport)
    assert report.label == "MemoryAllocationTest"
    assert report.duration_seconds >= 0.04
    assert report.python_heap_peak_mb > 1.0
    assert report.process_peak_rss_mb > 0.0
    assert report.process_peak_rss_mb >= report.python_heap_peak_mb
