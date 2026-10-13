"""Measured MSS/PyAutoGUI runner. Frame ring and JSON telemetry saved after stopping."""
import argparse,json,time,statistics,hashlib
from collections import deque
from pathlib import Path
from threading import Event
import cv2,mss,numpy as np,pyautogui,yaml
from pynput import keyboard
from bot.screen_capture import dpi_aware,FrameSource
from bot.vision import Vision
from bot.physics import Physics
from bot.controller import Controller
from bot.input_manager import InputManager
from bot.timer_resolution import TimerResolution
from bot.collision_geometry import CollisionSamples,body_extents,body_center_offset_y


def main():
    dpi_aware()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds',type=float,default=45)
    ap.add_argument('--start-click',type=int,nargs=2)
    ap.add_argument('--focus-canvas',action='store_true')
    ap.add_argument('--name',default=str(time.time_ns()))
    ap.add_argument('--config',type=Path,help='Cấu hình riêng cho kiểm thử; mặc định vẫn dùng config.yaml.')
    ap.add_argument('--expected-hwnd',type=int,help='Dừng trước input nếu foreground khác cửa sổ đã xác nhận.')
    ap.add_argument('--no-input',action='store_true')
    ap.add_argument('--benchmark',action='store_true',help='profile capture/detection without gameplay input')
    ap.add_argument('--prepare-delay',type=float,default=3,help='Số giây để chuyển về tab game trước khi khóa foreground.')
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    config_path=args.config or root/'config.yaml'
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if (root/'logs'/f'local-{args.name}.json').exists():raise SystemExit('Log đã tồn tại; chọn --name mới để giữ bằng chứng cũ.')
    config_snapshot=dict(cfg)
    code_snapshot={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),*sorted((root/'bot').glob('*.py'))]}
    if cfg.get('use_numba',False):
        from bot.compiled_planner import warmup
        warmup()
        if cfg.get('planner')=='reachability':
            from bot.reachability import warmup as warmup_atlas
            warmup_atlas()
    if not cfg.get('roi'):raise SystemExit('Chọn ROI trong app hoặc config.yaml trước.')
    vision=Vision(cfg,root);physics=Physics(cfg);controller=Controller(cfg)
    if args.prepare_delay>0:
        print(f'Có {args.prepare_delay:g} giây để chuyển về tab game.',flush=True)
        time.sleep(args.prepare_delay)
    inputs=InputManager(cfg);inputs.bind_foreground();stop=Event()
    if args.expected_hwnd is not None and inputs.hwnd!=args.expected_hwnd:
        raise SystemExit('Foreground đã thay đổi; không gửi input.')
    listener=keyboard.Listener(on_press=lambda k:(inputs.stop(),stop.set()) if k==keyboard.Key.f8 else None)
    listener.start()
    ring=deque(maxlen=75);records=[];reason='time limit';missing_since=None
    calibration=CollisionSamples(cfg.get('calibrate_hitbox',False),cfg)
    started=time.perf_counter();previous=started;last_draw=None;last_ring=0;active=False
    try:
        with TimerResolution(cfg.get('windows_timer_1ms',False)), FrameSource(cfg) as source:
            if args.start_click:
                if not inputs.focused():raise RuntimeError('Game must be foreground')
                pyautogui.click(*args.start_click)
                if args.focus_canvas:
                    time.sleep(.08)
                    roi=cfg['roi']
                    pyautogui.click(roi['left']+roi['width']//2,roi['top']+roi['height']//2)
            started=time.perf_counter()
            while not stop.is_set() and time.perf_counter()-started<args.seconds:
                if not inputs.focused():reason=inputs.stop_reason or 'foreground lost';break
                cycle_started=time.perf_counter()
                packet=source.next();tick=packet.started;raw=packet.frame;captured=packet.captured
                handed_off=time.perf_counter();obs=vision.detect(raw,captured);detected=time.perf_counter()
                now=detected
                if args.benchmark:
                    interval=captured-previous;previous=captured
                    records.append(dict(t=captured-started,capture_ms=1000*(captured-tick),detection_ms=1000*(detected-handed_off),handoff_ms=1000*(handed_off-captured),transfer_ms=0,total_ms=1000*(detected-tick),fps=1/max(interval,.0001)))
                    continue
                if obs.bird is None or obs.ended:
                    if missing_since is None:missing_since=now
                    if active and (obs.ended or now-missing_since>.25):
                        reason='modal/end' if obs.ended else 'bird lost';ring.append((now-started,obs.frame.copy()));break
                    if not args.benchmark and not active and now-started>2:
                        reason='no gameplay detected';break
                    if args.benchmark:
                        records.append({'t':now-started,'capture_ms':1000*(captured-tick),'detection_ms':1000*(detected-handed_off),'handoff_ms':1000*(handed_off-captured),'total_ms':1000*(detected-tick)})
                    stop.wait(.005);continue
                active=True;missing_since=None
                observed=captured-(cfg.get('capture_age_ms',0)/1000 if cfg.get('latency_pipeline',False) else 0)
                physics.update(obs.bird,obs.gates,observed)
                predicted=time.perf_counter()
                interval=captured-previous
                cfg['reaction_interval']=max(.025,min(.12,interval))
                previous_flap=controller.last_flap
                decision=controller.decide(obs,physics,observed,predicted-observed)
                decided=time.perf_counter()
                pre_flap_v=physics.v
                action=False;input_ms=None;input_latency=None
                if decision.flap and not args.no_input and not args.benchmark and not stop.is_set():
                    before=time.perf_counter();action=inputs.space();sent=time.perf_counter()
                    input_ms=1000*(sent-before);input_latency=1000*(sent-tick)
                    if action:
                        physics.flap(before if cfg.get('input_timestamp_origin')=='start' else sent)
                        if cfg.get('align_input_cooldown',False):controller.last_flap=before
                if decision.flap and not action:controller.last_flap=previous_flap
                finished=time.perf_counter()
                records.append({'t':captured-started,'x':obs.bird.x,'y':obs.bird.y,'w':obs.bird.w,'h':obs.bird.h,'v':physics.v,'pre_flap_v':pre_flap_v,'g':physics.g,'jump':physics.jump,'speed':physics.speed,'gates':[vars(g) for g in obs.gates],'flap':action,'target':decision.target,'feasible':decision.feasible,'plan_margin':decision.plan_margin,'decision':decision.reason,'capture_ms':1000*(captured-tick),'transfer_ms':0,'detection_ms':1000*(detected-captured),'physics_ms':1000*(predicted-detected),'controller_ms':1000*(decided-predicted),'input_ms':input_ms,'capture_to_input_ms':input_latency,'total_ms':1000*(finished-tick),'fps':1/max(interval,.0001),'score':obs.score})
                records[-1].update(long_plan_margin=decision.long_plan_margin,prefix_margin=decision.prefix_margin,coast_margin=decision.coast_margin,flap_margin=decision.flap_margin,planner_beam=decision.planner_beam,input_status=physics.input_status,input_event_count=len(physics.input_events),input_sent_t=sent-started if action else None,input_start_t=before-started if action else None,bird_confidence=obs.bird.confidence,frame_sequence=packet.sequence,handoff_ms=1000*(handed_off-captured),detection_ms=1000*(detected-handed_off))
                hw,hh=body_extents(cfg,obs.bird)
                records[-1].update(collision_hw=hw,collision_hh=hh,
                                   collision_center_offset_y=body_center_offset_y(cfg))
                calibration.observe(obs,captured-started)
                if captured-last_ring>.05:
                    ring.append((captured-started,obs.frame.copy()));last_ring=captured
                previous=captured
                # No screenshots/log flushing in hot path. Cap rate to configured limit.
                wait_requested=max(0,1/cfg['fps']-(time.perf_counter()-cycle_started))
                wait_started=time.perf_counter();stop.wait(wait_requested);wait_finished=time.perf_counter()
                records[-1].update(wait_requested_ms=1000*wait_requested,wait_actual_ms=1000*(wait_finished-wait_started),cycle_ms=1000*(wait_finished-cycle_started))
            if stop.is_set():reason='F8 stop'
    finally:
        inputs.stop();listener.stop();controller.close()
        (root/'logs').mkdir(exist_ok=True);out=root/'screenshots'/f'local-{args.name}';out.mkdir(parents=True,exist_ok=True)
        for i,(t,f) in enumerate(list(ring)[-cfg.get('collision_frames',15):]):cv2.imwrite(str(out/f'{i:02d}-{t:.3f}s.jpg'),f)
        metrics={}
        for k in ['capture_ms','handoff_ms','transfer_ms','detection_ms','physics_ms','controller_ms','input_ms','capture_to_input_ms','total_ms','wait_requested_ms','wait_actual_ms','cycle_ms','fps']:
            values=[r[k] for r in records if r.get(k) is not None]
            if values:metrics[k]={'median':statistics.median(values),'p95':float(np.percentile(values,95))}
        native_input=cfg.get('fast_click',False) if cfg.get('input_mode')=='click' else cfg.get('fast_space',False)
        input_backend='Win32 SendInput' if native_input else 'PyAutoGUI'
        summary={'name':args.name,'source':'MSS ROI/'+input_backend,'capture_pipeline':cfg.get('capture_pipeline',False),'input_mode':cfg.get('input_mode','space'),'reason':reason,'wall_s':time.perf_counter()-started,'observed_span_s':records[-1]['t'] if records else None,'frames':len(records),'flaps':sum(r.get('flap',False) for r in records),'roi':cfg['roi'],'metrics':metrics,'measured_physics':{'g':physics.g,'jump':physics.jump,'speed':physics.speed}}
        summary.update(config=config_snapshot,config_path=str(config_path.resolve()),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        summary['code_sha256']=code_snapshot
        summary['collision_samples']=calibration.save(out,max((r['t'] for r in records if 'y' in r),default=0))
        summary['input_events']=[{key:(value-started if key in ('sent','observed','effect_estimate') else value) for key,value in event.items()} for event in physics.input_events]
        (root/'logs'/f'local-{args.name}.json').write_text(json.dumps({'summary':summary,'frames':records},indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
