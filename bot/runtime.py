import json
import hashlib
import threading
import time
from collections import deque
from pathlib import Path
from .screen_capture import FrameSource
from .vision import Vision
from .physics import Physics
from .controller import Controller
from .input_manager import InputManager
from .game_state import GameState
from .timer_resolution import TimerResolution
from .score_reader import read_result
from .telemetry import summarize_session
from ui.debug_overlay import draw

class Runtime:
    def __init__(self,cfg,root):
        self.cfg=cfg
        self.root=Path(root)
        self.state=GameState()
        self.input=InputManager(cfg)
        self.quit=threading.Event()
        self.pause=threading.Event()
        self.thread=None
        self.preview=None
        self.debug=False
        self.model=None
        self.reset_requested=threading.Event()
        self.log_path=None
        self.result_serial=0
        self.generation=0
        self.score_threads=[]
        self.latest_result_id=None
        self.initialized=threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():
            if self.quit.is_set():
                raise ValueError('Bot đang dừng; chờ một giây rồi thử lại.')
            if not self.initialized.is_set():
                return
            if self.pause.is_set():
                self.resume()
            else:
                self.pause.set()
                self.input.stop()
                self.state.status='TẠM DỪNG'
            return
        if not self.cfg.get('roi'):
            raise ValueError('Chọn vùng game trước.')
        self.quit.clear()
        self.pause.clear()
        self.input.stopped.clear()
        self.initialized.clear()
        self.state.status='ĐANG KHỞI TẠO: vui lòng chờ'
        self.generation+=1
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def resume(self):
        self.state.status='CHUYỂN VỀ TAB GAME: chờ 3 giây'
        generation=self.generation
        def arm():
            if self.quit.wait(3) or generation!=self.generation:
                return
            if not self.bind_game_foreground(generation):return
            self.input.stopped.clear()
            self.reset_requested.set()
            self.pause.clear()
            self.state.status='ĐANG CHẠY'
        threading.Thread(target=arm,daemon=True).start()

    def bind_game_foreground(self,generation=None):
        generation=self.generation if generation is None else generation
        while not self.quit.is_set() and generation==self.generation:
            try:
                self.input.bind_foreground()
                return True
            except ValueError:
                self.state.status='CHỜ TRÌNH DUYỆT GAME Ở FOREGROUND'
                self.quit.wait(.1)
        return False

    def stop(self):
        self.generation+=1
        self.input.stop()
        self.quit.set()
        self.state.status='ĐÃ DỪNG'

    def record_result(self,image_path,session_path,serial):
        result_id=(session_path.stem,serial)
        self.latest_result_id=result_id;self.state.score_status='đang đọc'
        def worker():
            error=None;score=None
            try:score=read_result(image_path,self.root)
            except Exception as exception:error=str(exception)
            result=dict(score=score,source='Windows OCR',image=str(image_path),error=error)
            session_path.with_name(f'{session_path.stem}-result-{serial:03d}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            if score is not None:
                self.state.best=max(self.state.best or 0,score)
            if self.latest_result_id==result_id:
                self.state.last_result=score
                self.state.score_status='đã đọc' if score is not None else 'không đọc được'
        thread=threading.Thread(target=worker,daemon=True)
        self.score_threads.append(thread);thread.start()

    def run(self):
        import cv2
        log=None
        cap=None
        controller=None
        metadata=None;log_path=None
        timer=TimerResolution(self.cfg.get('source_high_resolution_wait',True) if self.cfg.get('planner')=='source_euler' else self.cfg.get('windows_timer_1ms',False))
        buffer=[]
        try:
            if self.cfg.get('planner')=='source_euler':
                from .source_model import configure
                configure(self.cfg)
                self.state.status='ĐANG BIÊN DỊCH PLANNER SOURCE'
                from .source_planner import warmup as warmup_source
                warmup_source()
            if self.cfg.get('use_numba',False):
                self.state.status='ĐANG BIÊN DỊCH NUMBA: lần đầu có thể lâu'
                from .compiled_planner import warmup
                warmup()
                if self.cfg.get('planner')=='reachability':
                    from .reachability import warmup as warmup_atlas
                    warmup_atlas()
            self.state.status='CHUYỂN VỀ TAB GAME: chờ 3 giây'
            if self.quit.wait(3):
                return
            if not self.bind_game_foreground():return
            vision=Vision(self.cfg,self.root)
            physics=Physics(self.cfg)
            self.model=physics
            controller=Controller(self.cfg)
            timer.start()
            cap=FrameSource(self.cfg)
            self.state.begin()
            self.state.status='CHỜ FRAME MSS ĐẦU TIÊN'
            self.initialized.set()
            (self.root/'logs').mkdir(exist_ok=True)
            log_path=self.root/'logs'/f'run-{time.time_ns()}.jsonl'
            self.log_path=log_path;self.state.last_log=str(log_path)
            metadata=dict(config=json.loads(json.dumps(self.cfg)),source='MSS/OpenCV/click' if self.cfg.get('input_mode')=='click' else 'MSS/OpenCV/Space',code_sha256={str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [self.root/'main.py',*list((self.root/'bot').glob('*.py')),*list((self.root/'ui').glob('*.py'))] if p.is_file()})
            log_path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            log=open(log_path,'w',encoding='utf-8')
            last=time.perf_counter()
            ring=deque(maxlen=15)
            buffer=[];last_flush=last;last_preview=0.
            missing=0
            last_wait_request=last_wait_actual=0.
            stable=0
            last_ui_click=-1e9;end_saved=False
            game_started=None;game_events=[]
            metadata['game_events']=game_events
            while not self.quit.is_set():
                tick=time.perf_counter()
                if self.pause.is_set():
                    physics.points.clear()
                    stable=0
                    self.quit.wait(.05)
                    continue
                if self.reset_requested.is_set():
                    controller.close();controller=Controller(self.cfg)
                    physics=Physics(self.cfg);self.model=physics
                    self.reset_requested.clear();stable=missing=0
                packet=cap.next();raw=packet.frame
                captured=packet.captured
                handed_off=time.perf_counter()
                obs=vision.detect(raw,captured)
                detected=time.perf_counter()
                now=captured
                dt=now-last
                last=now
                if dt>0:
                    self.state.fps=.9*self.state.fps+.1/dt
                self.state.observed(obs.score)
                focused=self.input.focused()
                buffer.append(json.dumps(dict(kind='diagnostic',t=now-self.state.started,frame_sequence=packet.sequence,focused=focused,focus_reason=getattr(self.input,'stop_reason',None) if not focused else None,ready=obs.ready,game_over=obs.game_over,ended=obs.ended,bird_detected=obs.bird is not None,action_point=obs.action_point,status=self.state.status,capture_ms=1000*(captured-packet.started),detection_ms=1000*(detected-handed_off)),ensure_ascii=False)+'\n')
                if handed_off-last_flush>=1:
                    log.writelines(buffer);buffer.clear();log.flush();last_flush=handed_off
                if not focused:
                    buffer.append(json.dumps(dict(kind='diagnostic',t=now-self.state.started,event='focus_lost',focus_reason=getattr(self.input,'stop_reason',None),focus_details=getattr(self.input,'focus_details',{})),ensure_ascii=False)+'\n')
                    log.writelines(buffer);buffer.clear();log.flush()
                    self.pause.set()
                    self.input.stop()
                    reason=getattr(self.input,'stop_reason',None)
                    detail={'foreground lost':'game không còn foreground','foreground title does not match game':'tên cửa sổ không khớp game','window geometry changed':'vị trí/kích thước cửa sổ đã đổi','window geometry differs from calibrated ROI':'cửa sổ khác cấu hình ROI'}.get(reason,'kiểm tra tab game và vị trí cửa sổ')
                    self.state.status=f'TẠM DỪNG: {detail}'
                    continue
                if obs.ready and self.cfg.get('auto_start',False) and obs.action_point is not None:
                    if now-last_ui_click>.8 and self.input.click_normalized(obs.action_point):
                        last_ui_click=now;end_saved=False;self.reset_requested.set()
                        game_started=time.perf_counter();game_events.append(dict(event='start',t=game_started-self.state.started,source='ready button'))
                        self.state.score=None;self.state.status='ĐÃ CLICK VÀO ĐUA'
                    self.quit.wait(.02);continue
                if obs.game_over and not end_saved:
                    self.result_serial+=1
                    image_path=self.root/'screenshots'/f'{log_path.stem}-end-{self.result_serial:03d}.png'
                    cv2.imwrite(str(image_path),obs.frame)
                    for index,frame in enumerate(ring):
                        cv2.imwrite(str(self.root/'screenshots'/f'{log_path.stem}-end-{self.result_serial:03d}-pre-{index:02d}.jpg'),frame)
                    game_events.append(dict(event='end',t=now-self.state.started,duration_s=now-game_started if game_started is not None else None,image=str(image_path)))
                    if game_started is not None:self.state.duration=now-game_started
                    game_started=None
                    self.record_result(image_path,log_path,self.result_serial)
                    end_saved=True
                if obs.game_over and self.cfg.get('auto_restart',False) and obs.action_point is not None:
                    if now-last_ui_click>.8 and self.input.click_normalized(obs.action_point):
                        last_ui_click=now;self.state.status='ĐÃ CLICK CHƠI LẠI'
                    self.quit.wait(.02);continue
                if obs.ended or obs.bird is None:
                    stable=0
                    missing+=1
                    # Nút VÀO ĐUA có thể biến mất trước khi modal đóng hoàn toàn.
                    if not obs.game_over and now-last_ui_click<.6:
                        self.state.status='CHỜ GAME BẮT ĐẦU'
                        self.quit.wait(.02);continue
                    if obs.ended or missing>=8:
                        self.pause.set()
                        self.state.status=('TẠM DỪNG: nhận diện hộp thoại' if obs.ended else 'TẠM DỪNG: không tìm thấy nhân vật sau 8 frame')
                        failure_image=self.root/'screenshots'/f'{log_path.stem}-stop-{time.time_ns()}.png'
                        cv2.imwrite(str(failure_image),obs.frame)
                        metadata['last_pause_reason']=self.state.status
                        game_events.append(dict(event='pause',t=now-self.state.started,reason=self.state.status,image=str(failure_image)))
                        for i,frame in enumerate(ring):
                            cv2.imwrite(str(failure_image.with_name(f'{failure_image.stem}-pre-{i:02d}.jpg')),frame)
                    self.quit.wait(.02)
                    continue
                missing=0
                end_saved=False
                if game_started is None:
                    game_started=now;game_events.append(dict(event='start',t=now-self.state.started,source='bird detection'))
                stable+=1
                observed=now-(self.cfg.get('capture_age_ms',0)/1000 if self.cfg.get('latency_pipeline',False) else 0)
                physics.update(obs.bird,obs.gates,observed)
                predicted=time.perf_counter()
                self.cfg['reaction_interval']=max(.025,min(.12,dt))
                previous_flap=controller.last_flap
                decision=controller.decide(obs,physics,observed,predicted-observed)
                decided=time.perf_counter()
                pre_flap_v=physics.v
                input_started=time.perf_counter()
                action=decision.flap and stable>=2 and not self.pause.is_set() and not self.quit.is_set() and self.input.space()
                if decision.flap and not action:controller.last_flap=previous_flap
                sent=time.perf_counter()
                if action:
                    physics.flap(input_started if self.cfg.get('input_timestamp_origin')=='start' else sent)
                    if self.cfg.get('align_input_cooldown',False):controller.last_flap=input_started
                if sent-last_preview>=.1:
                    self.preview=draw(obs,decision,physics,self.state.fps) if self.debug else None
                    ring.append(obs.frame.copy());last_preview=sent
                self.state.status=f'ĐANG CHẠY | Khe nhận diện: {len(obs.gates)}'
                self.state.duration=now-game_started
                buffer.append(json.dumps({'t':now-self.state.started,'x':obs.bird.x,'y':obs.bird.y,'w':obs.bird.w,'h':obs.bird.h,'v':physics.v,'pre_flap_v':pre_flap_v,'g':physics.g,'jump':physics.jump,'speed':physics.speed,'gates':[vars(g) for g in obs.gates],'flap':bool(action),'requested_flap':bool(decision.flap),'prediction_residual_px':getattr(physics,'prediction_residual',None),'input_blocked_reason':(getattr(self.input,'stop_reason',None) or ('waiting for stable detection' if stable<2 else 'paused/stopped')) if decision.flap and not action else None,'plan_margin':decision.plan_margin,'decision':decision.reason,'long_plan_margin':decision.long_plan_margin,'prefix_margin':decision.prefix_margin,'coast_margin':decision.coast_margin,'flap_margin':decision.flap_margin,'planner_beam':decision.planner_beam,'input_status':physics.input_status,'input_event_count':len(physics.input_events),'input_start_t':input_started-self.state.started if action else None,'input_sent_t':sent-self.state.started if action else None,'frame_sequence':packet.sequence,'fps_instant':1/max(dt,.0001),'fps':self.state.fps,'score':obs.score,'capture_ms':1000*(captured-packet.started),'handoff_ms':1000*(handed_off-captured),'transfer_ms':0,'detection_ms':1000*(detected-handed_off),'physics_ms':1000*(predicted-detected),'controller_ms':1000*(decided-predicted),'input_ms':1000*(sent-input_started) if action else None,'capture_to_input_ms':1000*(sent-packet.started) if action else None,'total_ms':1000*(sent-packet.started),'previous_wait_requested_ms':1000*last_wait_request,'previous_wait_actual_ms':1000*last_wait_actual})+'\n')
                if sent-last_flush>=1:
                    log.writelines(buffer);buffer.clear();last_flush=sent
                last_wait_request=max(0,1/self.cfg['fps']-(time.perf_counter()-tick))
                wait_started=time.perf_counter();self.quit.wait(last_wait_request)
                last_wait_actual=time.perf_counter()-wait_started
            # Persist only measured estimates with sufficient observations.
            if physics.samples>15:
                self.cfg['gravity']=round(physics.g,2)
                self.cfg['pipe_speed']=round(physics.speed,2)
        except Exception as e:
            self.input.stop()
            self.state.status=f'ERROR: {e}'
        finally:
            self.input.stop()
            if controller is not None:controller.close()
            timer.close()
            if cap:
                cap.close()
            if log:
                log.writelines(buffer)
                log.close()
                if metadata is not None:
                    metadata['reason']=self.state.status
                    summary=summarize_session(log_path,metadata)
                    log_path.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
