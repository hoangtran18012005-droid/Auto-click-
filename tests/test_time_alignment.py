import pytest
from tools.time_alignment import input_time


def test_legacy_capture_to_input_includes_capture_but_start_excludes_input_call():
    frame=dict(t=10,capture_ms=15,capture_to_input_ms=40,input_ms=8)
    assert input_time(frame,'frame')==10
    assert input_time(frame,'input_end')==pytest.approx(10.025)
    assert input_time(frame,'start')==pytest.approx(10.017)


def test_explicit_input_timestamps_override_legacy_derived_values():
    frame=dict(t=10,capture_ms=15,capture_to_input_ms=40,input_ms=8,
               input_start_t=10.014,input_sent_t=10.026)
    assert input_time(frame,'input_start')==10.014
    assert input_time(frame,'end')==10.026
