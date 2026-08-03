"""Profiling-only CUDA capture helpers for standalone Serenity workers."""

from std.ffi import external_call


def profile_capture_start() raises:
    var status = external_call["cuProfilerStart", Int32]()
    if status != 0:
        raise Error(String("cuProfilerStart failed: ") + String(status))


def profile_capture_stop() raises:
    var status = external_call["cuProfilerStop", Int32]()
    if status != 0:
        raise Error(String("cuProfilerStop failed: ") + String(status))
