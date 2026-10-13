import json
from threading import Event
from types import SimpleNamespace
import numpy as np
from bot.vision import Vision,Observation
from bot.score_reader import parse_result
from bot.telemetry import summarize_session
from tools.export_logs import export_recent

def test_ocr_does_not_confuse_result_with_personal_record():
    assert parse_result('Va chạm rồi! 76 điểm Kỷ lục đã đồng bộ: 512 điểm')==76
    assert parse_result('Va chạm rồi! O điểm Kỷ lục 512 điểm')==0
    assert parse_result('Kỷ lục 512 điểm') is None
    assert parse_result('Va chạm rồi! Kỷ lục 512 điểm') is None

def test_modal_action_uses_large_right_button_not_hud(tmp_path):
    import cv2
    frame=np.full((400,600,3),(220,245,255),np.uint8)
    cv2.rectangle(frame,(340,250),(520,285),(65,210,255),-1)
    cv2.rectangle(frame,(18,30),(28,44),(65,210,255),-1)
    assert Vision(dict(pipe_speed=139.6),tmp_path).modal_action(frame)==(430.5,268.)

def test_session_summary_and_export_read_only_existing_files(tmp_path):
    import zipfile
    (tmp_path/'logs').mkdir();(tmp_path/'screenshots').mkdir()
    path=tmp_path/'logs/run-123.jsonl'
    path.write_text('\n'.join(json.dumps(dict(t=t,flap=bool(t),decision='COAST',controller_ms=2,fps_instant=60)) for t in (0.,1.)),encoding='utf-8')
    summary=summarize_session(path,dict(config=dict(input_mode='click')))
    assert summary['frames']==2 and summary['observed_span_s']==1
    assert summary['metrics']['controller_ms']['median']==2
    path.with_suffix('.json').write_text(json.dumps(summary),encoding='utf-8')
    (tmp_path/'screenshots/run-123-stop-999.png').write_bytes(b'fixture')
    out=export_recent(tmp_path)
    with zipfile.ZipFile(out) as archive:assert set(archive.namelist())=={'logs/run-123.json','logs/run-123.jsonl','screenshots/run-123-stop-999.png'}

def test_runtime_clicks_ready_without_real_capture_or_input(tmp_path,monkeypatch):
    import bot.runtime as module
    from bot.screen_capture import FramePacket
    class FastEvent(Event):
        def wait(self,timeout=None):return self.is_set()
    runtime=module.Runtime(dict(roi=dict(left=0,top=0,width=600,height=400),width=600,gravity=710.38,jump=-248.58,pipe_speed=139.6,fps=30,auto_start=True),tmp_path)
    runtime.quit=FastEvent();calls=[]
    def click(point):calls.append(point);runtime.quit.set();return True
    runtime.input=SimpleNamespace(bind_foreground=lambda:None,focused=lambda:True,click_normalized=click,stop=lambda:None)
    class Capture:
        def __init__(self,cfg):pass
        def next(self):return FramePacket(np.zeros((400,600,3),np.uint8),1.,1.,1)
        def close(self):pass
    observation=Observation(None,[],np.zeros((400,600,3),np.uint8),True,None,True,False,(430,268))
    monkeypatch.setattr(module,'FrameSource',Capture)
    monkeypatch.setattr(module,'Vision',lambda *args:SimpleNamespace(detect=lambda *args:observation))
    (tmp_path/'screenshots').mkdir()
    runtime.run()
    assert calls==[(430,268)]

def test_pause_blocks_input_and_old_resume_timer_cannot_rearm(tmp_path,monkeypatch):
    import bot.runtime as module
    runtime=module.Runtime({},tmp_path);stopped=Event();calls=[]
    runtime.input=SimpleNamespace(stop=stopped.set,stopped=stopped,bind_foreground=lambda:calls.append('bind'))
    runtime.thread=SimpleNamespace(is_alive=lambda:True)
    runtime.initialized.set()
    runtime.start()
    assert runtime.pause.is_set() and stopped.is_set()
    class InterruptedEvent:
        def wait(self,timeout):runtime.stop();return False
        def set(self):pass
    class ImmediateThread:
        def __init__(self,target,**kwargs):self.target=target
        def start(self):self.target()
    runtime.quit=InterruptedEvent()
    monkeypatch.setattr(module.threading,'Thread',ImmediateThread)
    runtime.resume()
    assert not calls and stopped.is_set()

def test_f6_during_initialization_does_not_pause_worker(tmp_path):
    from bot.runtime import Runtime
    runtime=Runtime({},tmp_path)
    runtime.thread=SimpleNamespace(is_alive=lambda:True)
    runtime.state.status='ĐANG BIÊN DỊCH NUMBA'
    runtime.start()
    assert not runtime.pause.is_set()
    assert runtime.state.status=='ĐANG BIÊN DỊCH NUMBA'

def test_f8_discards_start_hotkey_waiting_for_ui():
    from threading import Lock
    from pynput import keyboard
    from ui.app import App
    app=App.__new__(App);app.lock=Lock();app.pressed=set();app.pending=[]
    app.start=lambda:None;app.debug=lambda:None;calls=[]
    app.bot=SimpleNamespace(generation=0,stop=lambda:calls.append('stop'))
    app.hotkey(keyboard.Key.f6)
    assert len(app.pending)==1
    app.hotkey(keyboard.Key.f8)
    assert not app.pending and calls==['stop']

def test_runtime_waits_for_ready_modal_transition(tmp_path,monkeypatch):
    import time
    import bot.runtime as module
    from bot.screen_capture import FramePacket
    class FastEvent(Event):
        def wait(self,timeout=None):return self.is_set()
    runtime=module.Runtime(dict(roi=dict(left=0,top=0,width=600,height=400),width=600,gravity=710.38,jump=-248.58,pipe_speed=139.6,fps=30,auto_start=True),tmp_path)
    runtime.quit=FastEvent();calls=[];observations=[]
    runtime.input=SimpleNamespace(bind_foreground=lambda:None,focused=lambda:True,click_normalized=lambda point:calls.append(point) or True,stop=lambda:None)
    frame=np.zeros((400,600,3),np.uint8)
    class Capture:
        def __init__(self,cfg):self.count=0
        def next(self):
            self.count+=1
            if self.count==3:runtime.quit.set()
            observations.append(Observation(None,[],frame,True,None,self.count==1,False,(430,268) if self.count==1 else None))
            now=time.perf_counter();return FramePacket(frame,now,now,self.count)
        def close(self):pass
    monkeypatch.setattr(module,'FrameSource',Capture)
    monkeypatch.setattr(module,'Vision',lambda *args:SimpleNamespace(detect=lambda *args:observations[-1]))
    (tmp_path/'screenshots').mkdir()
    runtime.run()
    assert calls==[(430,268)] and not runtime.pause.is_set()
    summary=json.loads(runtime.log_path.with_suffix('.json').read_text(encoding='utf-8'))
    assert summary['diagnostics']['captured_frames']==3
    assert summary['frames']==0 and summary['diagnostics']['ready']==1

def test_modal_template_tolerates_small_roi_scale_change(tmp_path):
    import cv2
    template=np.random.default_rng(31).integers(0,255,(28,285,3),dtype=np.uint8)
    (tmp_path/'templates').mkdir()
    cv2.imwrite(str(tmp_path/'templates/ready.png'),template)
    frame=np.full((300,500,3),240,np.uint8)
    scaled=cv2.resize(template,None,fx=.98,fy=.98)
    frame[80:80+scaled.shape[0],60:60+scaled.shape[1]]=scaled
    vision=Vision(dict(pipe_speed=139.6),tmp_path)
    assert vision.match(frame,'ready')[0]>.9

def test_modal_chooses_best_title_instead_of_both_ready_and_end(tmp_path,monkeypatch):
    from pathlib import Path
    import yaml
    cfg=yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8'))
    vision=Vision(cfg,tmp_path)
    monkeypatch.setattr(vision,'match',lambda frame,name:({'ready':.96,'end':.85,'pause':.2}.get(name,0),(0,0),(1,1)))
    frame=np.full((400,600,3),(210,240,250),np.uint8)
    obs=vision.detect(frame)
    assert obs.ready and not obs.game_over

def test_dark_invalid_end_template_cannot_override_ready(tmp_path):
    import cv2,yaml
    from pathlib import Path
    cfg=yaml.safe_load(Path('config.yaml').read_text(encoding='utf-8'))
    (tmp_path/'templates').mkdir()
    cv2.imwrite(str(tmp_path/'templates/end.png'),np.zeros((26,144,3),np.uint8))
    vision=Vision(cfg,tmp_path)
    assert vision.invalid_templates==['end']
    assert 'end' not in vision.templates
    assert vision.match(np.zeros((200,300,3),np.uint8),'end')[0]==0
