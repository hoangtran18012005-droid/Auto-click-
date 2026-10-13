import ctypes
import threading
import time
import os
import pyautogui
from ctypes import wintypes

class MouseInput(ctypes.Structure):
    _fields_=[('dx',wintypes.LONG),('dy',wintypes.LONG),('mouseData',wintypes.DWORD),('dwFlags',wintypes.DWORD),('time',wintypes.DWORD),('dwExtraInfo',ctypes.c_size_t)]

class KeyboardInput(ctypes.Structure):
    _fields_=[('wVk',wintypes.WORD),('wScan',wintypes.WORD),('dwFlags',wintypes.DWORD),('time',wintypes.DWORD),('dwExtraInfo',ctypes.c_size_t)]

class InputUnion(ctypes.Union):
    _fields_=[('mi',MouseInput),('ki',KeyboardInput)]

class NativeInput(ctypes.Structure):
    _fields_=[('type',wintypes.DWORD),('data',InputUnion)]

def native_space():
    """Gửi KEYDOWN/KEYUP Space cùng batch; nhả phím nếu batch bị gửi thiếu."""
    events=(NativeInput*2)(
        NativeInput(1,InputUnion(ki=KeyboardInput(0,0x39,0x0008,0,0))),
        NativeInput(1,InputUnion(ki=KeyboardInput(0,0x39,0x000A,0,0))))
    send=ctypes.windll.user32.SendInput
    send.argtypes=(wintypes.UINT,ctypes.POINTER(NativeInput),ctypes.c_int);send.restype=wintypes.UINT
    if send(2,events,ctypes.sizeof(NativeInput))!=2:
        release=events[1]
        send(1,ctypes.byref(release),ctypes.sizeof(NativeInput))
        raise RuntimeError('SendInput không gửi đủ Space KEYDOWN/KEYUP.')

def native_click(x,y):
    """Một batch MOVE/DOWN/UP; chỉ gọi sau kiểm tra foreground và geometry."""
    user32=ctypes.windll.user32
    left,top,width,height=[user32.GetSystemMetrics(index) for index in (76,77,78,79)]
    if not left<=x<left+width or not top<=y<top+height:raise ValueError('Điểm click nằm ngoài desktop.')
    move=MouseInput(round((x-left)*65535/max(1,width-1)),round((y-top)*65535/max(1,height-1)),0,0x8000|0x4000|0x0001,0,0)
    point=wintypes.POINT()
    needs_move=not user32.GetCursorPos(ctypes.byref(point)) or abs(point.x-x)>1 or abs(point.y-y)>1
    items=[NativeInput(0,InputUnion(mi=move))] if needs_move else []
    items.extend([NativeInput(0,InputUnion(mi=MouseInput(0,0,0,0x0002,0,0))),NativeInput(0,InputUnion(mi=MouseInput(0,0,0,0x0004,0,0)))])
    events=(NativeInput*len(items))(*items)
    send=user32.SendInput
    send.argtypes=(wintypes.UINT,ctypes.POINTER(NativeInput),ctypes.c_int);send.restype=wintypes.UINT
    if send(len(items),events,ctypes.sizeof(NativeInput))!=len(items):
        # Nhả nút khi batch bị gửi thiếu; không tự gửi click lần hai.
        release=NativeInput(0,InputUnion(mi=MouseInput(0,0,0,0x0004,0,0)))
        send(1,ctypes.byref(release),ctypes.sizeof(NativeInput))
        raise RuntimeError('SendInput không gửi đủ sự kiện.')

def window_rect(hwnd):
    rect=wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd,ctypes.byref(rect)):return None
    return (rect.left,rect.top,rect.right,rect.bottom)

class InputManager:
    def __init__(self,cfg=None):
        self.cfg=cfg or {}
        pyautogui.PAUSE=0
        pyautogui.FAILSAFE=True
        self.hwnd=None
        self.bounds=None
        self.stop_reason=None
        self.focus_details={}
        self.lock=threading.Lock()
        self.stopped=threading.Event()
    def bind_foreground(self):
        hwnd=ctypes.windll.user32.GetForegroundWindow()
        pid=wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
        title=ctypes.create_unicode_buffer(1024)
        ctypes.windll.user32.GetWindowTextW(hwnd,title,1024)
        required=self.cfg.get('window_title_contains')
        if pid.value==os.getpid() or required and required.casefold() not in title.value.casefold():
            raise ValueError('Đưa trình duyệt game ra foreground; không chọn cửa sổ bot.')
        self.hwnd=hwnd
        self.bounds=window_rect(hwnd)
    def focused(self):
        required=self.cfg.get('window_title_contains')
        current=ctypes.windll.user32.GetForegroundWindow()
        pid=wintypes.DWORD()
        ctypes.windll.user32.GetWindowThreadProcessId(current,ctypes.byref(pid))
        title=ctypes.create_unicode_buffer(1024)
        ctypes.windll.user32.GetWindowTextW(current,title,1024)
        current_bounds=window_rect(current)
        self.focus_details=dict(expected_hwnd=self.hwnd,current_hwnd=current,current_title=title.value,expected_bounds=self.bounds,current_bounds=current_bounds)
        if pid.value==os.getpid():
            self.stop_reason='foreground is bot window';return False
        if self.hwnd is None:
            self.stop_reason='foreground lost';return False
        if self.hwnd!=current:
            # Chỉ nhận handle mới khi vẫn đúng tên game và cùng hình học.
            if required and required.casefold() in title.value.casefold() and self.bounds is not None and current_bounds==self.bounds:
                self.hwnd=current
            else:
                self.stop_reason='foreground lost';return False
        if required:
            if required.casefold() not in title.value.casefold():
                self.stop_reason='foreground title does not match game';return False
        if self.cfg.get('lock_window_geometry',False):
            if current_bounds!=self.bounds:
                self.stop_reason='window geometry changed';return False
            expected=self.cfg.get('expected_window_rect')
            if expected is not None and tuple(expected)!=self.bounds:
                self.stop_reason='window geometry differs from calibrated ROI';return False
        self.stop_reason=None
        return True
    def click_normalized(self,point):
        with self.lock:
            if self.stopped.is_set() or not self.focused():return False
            from .screen_capture import local_to_screen
            x,y=local_to_screen(*point,self.cfg['roi'],self.cfg['width'])
            roi=self.cfg['roi']
            if not (roi['left']<=x<roi['left']+roi['width'] and roi['top']<=y<roi['top']+roi['height']):
                raise ValueError('Nút điều khiển nằm ngoài ROI.')
            pyautogui.failSafeCheck()
            if self.cfg.get('fast_click',False):native_click(x,y)
            else:pyautogui.click(x,y)
            return True
    def space(self):
        with self.lock:
            if not self.stopped.is_set() and self.focused():
                if self.cfg.get('input_mode')=='click':
                    roi=self.cfg['roi']
                    position=self.cfg.get('jump_click_fraction',[.96,.05])
                    if len(position)!=2 or any(not 0<=value<=1 for value in position):
                        raise ValueError('jump_click_fraction phải gồm hai giá trị trong [0,1].')
                    x=roi['left']+round(roi['width']*position[0]);y=roi['top']+round(roi['height']*position[1])
                    if self.cfg.get('fast_click',False):
                        pyautogui.failSafeCheck()
                        native_click(x,y)
                    else:pyautogui.click(x,y)
                    return True
                if self.cfg.get('fast_space',False):
                    pyautogui.failSafeCheck()
                    native_space()
                    return True
                pyautogui.keyDown('space')
                try:
                    self.stopped.wait(.03)
                finally:
                    pyautogui.keyUp('space')
                return True
        return False
    def stop(self):
        self.stopped.set()
