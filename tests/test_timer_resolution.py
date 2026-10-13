import pytest
from bot.timer_resolution import TimerResolution


class FakeTimer:
    def __init__(self, result=0):
        self.result = result
        self.calls = []

    def timeBeginPeriod(self, period):
        self.calls.append(('begin', period))
        return self.result

    def timeEndPeriod(self, period):
        self.calls.append(('end', period))
        return 0


def test_timer_restored_after_exception():
    api = FakeTimer()
    timer = TimerResolution(True, api)
    with pytest.raises(ValueError):
        with timer:
            timer.start()
            raise ValueError('failure')
    timer.close()
    assert api.calls == [('begin', 1), ('end', 1)]


def test_disabled_and_failed_timer_do_not_release_unowned_request():
    api = FakeTimer(97)
    with TimerResolution(False, api):
        pass
    assert api.calls == []
    timer = TimerResolution(True, api)
    with pytest.raises(RuntimeError):
        timer.start()
    timer.close()
    assert api.calls == [('begin', 1)]
