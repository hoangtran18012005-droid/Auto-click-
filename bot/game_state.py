import time
from dataclasses import dataclass

@dataclass
class GameState:
    status: str='ĐÃ DỪNG'
    runs: int=0
    best: int | None=None
    score: int | None=None
    fps: float=0.
    duration: float=0.
    started: float=0.
    last_result: int | None=None
    score_status: str='chưa đọc'
    last_log: str=''
    def begin(self):
        self.status='ĐANG CHẠY'
        self.runs+=1
        self.started=time.perf_counter()
        self.score=None
        self.duration=0.
    def observed(self, score):
        if score is not None:
            self.score=score
            self.best=max(self.best or 0,score)
