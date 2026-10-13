import ctypes
import cv2
import mss
import numpy as np
import time,threading
from dataclasses import dataclass

def create_mss():
    factory=getattr(mss,'MSS',None)
    return factory() if factory is not None else mss.mss()

@dataclass
class FramePacket:
    frame: np.ndarray
    started: float
    captured: float
    sequence: int

class FrameSource:
    """MSS trực tiếp hoặc producer một frame mới nhất; không tích hàng đợi ảnh cũ."""
    def __init__(self,cfg):
        self.roi=dict(cfg['roi']);self.pipeline=cfg.get('capture_pipeline',False)
        self.capture_interval=1/cfg.get('capture_hz',90)
        self.condition=threading.Condition();self.stopped=threading.Event()
        self.packet=None;self.sequence=0;self.consumed=0;self.error=None;self.sct=None;self.thread=None
        if self.pipeline:
            self.thread=threading.Thread(target=self._produce,daemon=True);self.thread.start()
        else:self.sct=create_mss()
    def _produce(self):
        try:
            with create_mss() as capture:
                while not self.stopped.is_set():
                    started=time.perf_counter();frame=np.asarray(capture.grab(self.roi))[:,:,:3];captured=time.perf_counter()
                    with self.condition:
                        self.sequence+=1;self.packet=FramePacket(frame,started,captured,self.sequence);self.condition.notify_all()
                    self.stopped.wait(max(0,self.capture_interval-(time.perf_counter()-started)))
        except Exception as error:
            with self.condition:self.error=error;self.condition.notify_all()
    def next(self,timeout=.5):
        if not self.pipeline:
            started=time.perf_counter();frame=np.asarray(self.sct.grab(self.roi))[:,:,:3];captured=time.perf_counter();self.sequence+=1
            return FramePacket(frame,started,captured,self.sequence)
        with self.condition:
            ready=self.condition.wait_for(lambda:self.error is not None or self.packet is not None and self.packet.sequence>self.consumed or self.stopped.is_set(),timeout)
            if self.error is not None:raise RuntimeError('MSS producer thất bại') from self.error
            if not ready or self.stopped.is_set():raise TimeoutError('Không có frame mới từ MSS.')
            packet=self.packet;self.consumed=packet.sequence;return packet
    def close(self):
        self.stopped.set()
        with self.condition:self.condition.notify_all()
        if self.thread:self.thread.join(timeout=1)
        if self.sct:self.sct.close()
    def __enter__(self):return self
    def __exit__(self,*unused):self.close()

def dpi_aware():
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass

def local_to_screen(x, y, roi, normalized_width):
    scale=roi['width']/normalized_width
    return round(roi['left']+x*scale),round(roi['top']+y*scale)

class Capture:
    def __init__(self, roi):
        self.roi=roi
        self.sct=create_mss()
    def grab(self):
        return np.array(self.sct.grab(self.roi))[:,:,:3].copy()
    def close(self):
        self.sct.close()
