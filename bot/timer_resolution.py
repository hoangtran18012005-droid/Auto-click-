"""Độ phân giải timer Windows chỉ được yêu cầu trong thời gian bot chạy."""
import ctypes
import sys


class TimerResolution:
    def __init__(self, enabled=False, api=None):
        self.enabled = enabled
        self.api = api
        self.active = False

    def start(self):
        if self.active or not self.enabled:
            return self
        if self.api is None:
            if sys.platform != 'win32':
                raise RuntimeError('windows_timer_1ms cần Windows.')
            self.api = ctypes.WinDLL('winmm')
        result = self.api.timeBeginPeriod(1)
        if result != 0:
            raise RuntimeError(f'timeBeginPeriod thất bại: {result}')
        self.active = True
        return self

    def close(self):
        if self.active:
            self.api.timeEndPeriod(1)
            self.active = False

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
