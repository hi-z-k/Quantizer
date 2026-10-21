"""Tests for phone/chat.py resource monitoring functions."""

import os
from phone.chat import read_device_memory, read_process_stats, get_terminal_size


def test_get_terminal_size():
    cols, rows = get_terminal_size()
    assert cols > 0
    assert rows > 0


def test_read_device_memory():
    mem = read_device_memory()
    # On Linux, /proc/meminfo exists and returns total_mb > 0
    if os.path.exists("/proc/meminfo"):
        assert mem["total_mb"] > 0
        assert mem["available_mb"] > 0
        assert mem["used_mb"] >= 0


def test_read_process_stats():
    pid = os.getpid()
    stats = read_process_stats(pid)
    if os.path.exists(f"/proc/{pid}/status"):
        assert stats["rss_mb"] > 0
