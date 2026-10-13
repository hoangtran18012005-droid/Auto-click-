import threading
import numpy as np
import pytest
from bot.screen_capture import FrameSource, FramePacket


def source_without_desktop():
    source = FrameSource.__new__(FrameSource)
    source.pipeline = True
    source.condition = threading.Condition()
    source.stopped = threading.Event()
    source.packet = None
    source.consumed = 0
    source.error = None
    source.thread = None
    source.sct = None
    return source


def test_consumer_uses_latest_frame_and_never_repeats_it():
    source = source_without_desktop()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    source.packet = FramePacket(frame, 1, 2, 20)
    assert source.next().sequence == 20
    with pytest.raises(TimeoutError):
        source.next(timeout=.001)
    source.packet = FramePacket(frame, 3, 4, 25)
    assert source.next().sequence == 25
    source.close()
    with pytest.raises(TimeoutError):
        source.next(timeout=.001)


def test_capture_failure_reaches_controller_instead_of_reusing_old_frame():
    source = source_without_desktop()
    source.error = OSError('capture failed')
    with pytest.raises(RuntimeError) as error:
        source.next()
    assert error.value.__cause__ is source.error
