import ctypes
import pytest
from bot.input_manager import NativeInput,native_space


def test_native_space_sends_down_and_up_in_order(monkeypatch):
    calls=[]
    def send(count,events,size):
        values=ctypes.cast(events,ctypes.POINTER(NativeInput))
        calls.append([(values[i].type,values[i].data.ki.wVk,values[i].data.ki.wScan,values[i].data.ki.dwFlags) for i in range(count)])
        assert size==ctypes.sizeof(NativeInput)
        return count
    monkeypatch.setattr(ctypes.windll.user32,'SendInput',send)
    native_space()
    assert calls==[[(1,0,0x39,8),(1,0,0x39,10)]]


def test_partial_keyboard_batch_releases_space(monkeypatch):
    calls=[]
    def send(count,events,size):
        values=ctypes.cast(events,ctypes.POINTER(NativeInput))
        calls.append([(values[i].data.ki.wScan,values[i].data.ki.dwFlags) for i in range(count)])
        return 1
    monkeypatch.setattr(ctypes.windll.user32,'SendInput',send)
    with pytest.raises(RuntimeError):native_space()
    assert calls==[[(0x39,8),(0x39,10)],[(0x39,10)]]

@pytest.mark.parametrize('title,bounds,allowed',[
    ('CUỘC ĐUA KỲ THÚ - Edge',(0,0,1920,1080),True),
    ('Vision Auto Player',(0,0,1920,1080),False),
    ('CUỘC ĐUA KỲ THÚ - Edge',(10,0,1920,1080),False),
])
def test_handle_change_requires_game_title_and_identical_geometry(monkeypatch,title,bounds,allowed):
    import bot.input_manager as module
    from threading import Event
    manager=module.InputManager.__new__(module.InputManager)
    manager.cfg=dict(window_title_contains='CUỘC ĐUA',lock_window_geometry=True)
    manager.hwnd=101;manager.bounds=(0,0,1920,1080);manager.stop_reason=None
    monkeypatch.setattr(ctypes.windll.user32,'GetForegroundWindow',lambda:202)
    def get_title(hwnd,buffer,size):buffer.value=title;return len(title)
    monkeypatch.setattr(ctypes.windll.user32,'GetWindowTextW',get_title)
    monkeypatch.setattr(module,'window_rect',lambda hwnd:bounds)
    assert manager.focused()==allowed
    assert manager.hwnd==(202 if allowed else 101)
    assert manager.focus_details['current_title']==title

@pytest.mark.parametrize('own_process,title,allowed',[
    (True,'Cuộc đua Kỳ Thú — Vision Auto Player',False),
    (False,'CUỘC ĐUA KỲ THÚ - Microsoft Edge',True),
    (False,'Visual Studio Code',False),
])
def test_bind_rejects_own_window_even_if_game_title_matches(monkeypatch,own_process,title,allowed):
    import bot.input_manager as module
    manager=module.InputManager.__new__(module.InputManager)
    manager.cfg=dict(window_title_contains='CUỘC ĐUA');manager.hwnd=None;manager.bounds=None
    monkeypatch.setattr(ctypes.windll.user32,'GetForegroundWindow',lambda:123)
    def get_pid(hwnd,pointer):
        ctypes.cast(pointer,ctypes.POINTER(ctypes.wintypes.DWORD)).contents.value=module.os.getpid() if own_process else module.os.getpid()+1
        return 1
    def get_title(hwnd,buffer,size):buffer.value=title;return len(title)
    monkeypatch.setattr(ctypes.windll.user32,'GetWindowThreadProcessId',get_pid)
    monkeypatch.setattr(ctypes.windll.user32,'GetWindowTextW',get_title)
    monkeypatch.setattr(module,'window_rect',lambda hwnd:(0,0,1920,1080))
    if allowed:
        manager.bind_foreground();assert manager.hwnd==123
    else:
        with pytest.raises(ValueError):manager.bind_foreground()
        assert manager.hwnd is None
