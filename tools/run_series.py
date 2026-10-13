"""Chạy chuỗi live MSS, OCR screenshot điểm; dừng F8 hoặc mất foreground."""
import argparse,ctypes,json,re,subprocess,sys,time
from pathlib import Path
import cv2,mss,numpy as np,pyautogui,yaml
from pynput import keyboard
from threading import Event
ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):sys.path.insert(0,str(ROOT))
from bot.screen_capture import dpi_aware
from bot.input_manager import window_rect
from tools.analyze_failures import crossing_times

def decode_score(raw):
    match=re.match(r'^\s*(\d+|[Oo])(?:\s|$)',raw)
    if not match:return None
    token=match[1]
    return 0 if token in ('O','o') else int(token)

def summarize(episodes):
    values=[e['score'] for e in episodes if e.get('score') is not None]
    estimates=[e['crossing_estimate'] for e in episodes if e.get('crossing_estimate') is not None]
    result={'completed':len(episodes),'officially_read':len(values),'estimated':len(estimates)}
    if values:result.update(median=float(np.median(values)),q25=float(np.percentile(values,25)),q75=float(np.percentile(values,75)),min=min(values),max=max(values))
    if estimates:result.update(estimate_median=float(np.median(estimates)),estimate_max=max(estimates))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--runs',type=int,default=30)
    parser.add_argument('--seconds',type=float,default=180)
    parser.add_argument('--start-click',nargs=2,type=int,default=[960,637])
    parser.add_argument('--restart-click',nargs=2,type=int,default=[960,542])
    parser.add_argument('--name',default=f'series-{time.time_ns()}')
    parser.add_argument('--pilot',action='store_true',help='Cho phép ít hơn 30 lượt để xác minh điều phối, không phải protocol chính.')
    parser.add_argument('--target-score',type=int,help='Dừng chuỗi sau khi modal xác nhận đạt mốc này.')
    parser.add_argument('--resume',action='store_true',help='Tiếp tục chuỗi cùng cấu hình, giữ mọi lượt đã ghi kể cả điểm 0.')
    args=parser.parse_args()
    if args.runs<30 and not args.pilot:parser.error('Protocol chính yêu cầu ≥30 lượt; dùng --pilot cho thử điều phối.')
    dpi_aware();cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'));pyautogui.PAUSE=0
    hwnd=ctypes.windll.user32.GetForegroundWindow();stop=Event();episodes=[];interruptions=[];reason='in_progress'
    bounds=window_rect(hwnd)
    title_buffer=ctypes.create_unicode_buffer(1024)
    ctypes.windll.user32.GetWindowTextW(hwnd,title_buffer,1024)
    if 'CUỘC ĐUA' not in title_buffer.value:parser.error('Đưa cửa sổ game CUỘC ĐUA ra foreground trước khi chạy chuỗi.')
    artifact_path=ROOT/'evaluation'/f'{args.name}.json'
    if args.resume:
        saved=json.loads(artifact_path.read_text(encoding='utf-8'))
        if saved['config']!=cfg:parser.error('Cấu hình khác chuỗi cũ; phải dùng tên chuỗi mới.')
        interruptions=saved.get('interruptions',[])
        episodes=[]
        for episode in saved['episodes']:
            if episode.get('score') is None and episode['runner']['reason']!='modal/end':interruptions.append(episode)
            else:episodes.append(episode)
        for episode in episodes:
            if episode.get('score') is None and episode.get('ocr'):
                episode['score']=decode_score(episode['ocr'])
        if any(e.get('score') is None and e.get('crossing_estimate') is None for e in episodes):
            parser.error('Chuỗi có lượt chưa có điểm OCR hoặc ước tính cột; cần kiểm tra screenshot trước.')
    serial=max((int(e['name'].rsplit('-',1)[1]) for e in episodes+interruptions),default=0)
    def on_press(key):
        if key==keyboard.Key.f8:stop.set()
    listener=keyboard.Listener(on_press=on_press);listener.start()
    def focused():
        return ctypes.windll.user32.GetForegroundWindow()==hwnd and (not cfg.get('lock_window_geometry',False) or window_rect(hwnd)==bounds and (cfg.get('expected_window_rect') is None or list(bounds)==cfg['expected_window_rect']))
    if cfg.get('use_numba'):
        from bot.compiled_planner import warmup
        warmup()
        if cfg.get('planner')=='reachability':
            from bot.reachability import warmup as warmup_atlas
            warmup_atlas()
    try:
        for index in range(len(episodes),args.runs):
            if stop.is_set() or not focused():reason='F8/focus lost';break
            if index:
                pyautogui.click(*args.restart_click);stop.wait(.5)
            if not focused() or stop.is_set():reason='F8/focus lost';break
            serial+=1;name=f'{args.name}-{serial:02d}'
            if (ROOT/'logs'/f'local-{name}.json').exists():parser.error('Tên log đã tồn tại; dùng tên chuỗi mới.')
            command=[sys.executable,'-X','utf8',str(ROOT/'local_runner.py'),'--config',str(args.config.resolve()),'--seconds',str(args.seconds),'--start-click',*map(str,args.start_click),'--name',name,'--expected-hwnd',str(hwnd)]
            runner_output=ROOT/'logs'/f'local-{name}.runner.txt'
            with runner_output.open('w',encoding='utf-8') as output_stream:
                child=subprocess.Popen(command,stdout=output_stream,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    if stop.is_set():
                        child.wait(timeout=5);break
                    if not focused():reason='foreground lost'
                    stop.wait(.1)
            path=ROOT/'logs'/f'local-{name}.json'
            if child.returncode!=0 or not path.exists():
                reason='runner error'
                print(f'{name}: runner exit={child.returncode}; xem {runner_output}',flush=True)
                break
            data=json.loads(path.read_text(encoding='utf-8'));episode={'name':name,'runner':data['summary'],'score':None,
                                                                       'crossing_estimate':len(crossing_times(data['frames']))}
            if data['summary']['reason']!='modal/end' or not focused() or stop.is_set():
                interruptions.append(episode);reason=data['summary']['reason'];break
            stop.wait(.2)
            # Capture chỉ sau khi gameplay kết thúc, không làm chậm vòng điều khiển.
            image_path=ROOT/'screenshots'/f'{name}-result.png'
            with mss.mss() as capture:frame=np.array(capture.grab(cfg['roi']))[:,:,:3]
            cv2.imwrite(str(image_path),frame)
            # Crop riêng số điểm lớn giữa modal, tránh đọc nhầm kỷ lục/bảng xếp hạng.
            crop=frame[125:180,250:540];crop_path=ROOT/'screenshots'/f'{name}-score.png'
            cv2.imwrite(str(crop_path),cv2.resize(crop,None,fx=2,fy=2))
            completed=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'tools/read_score.ps1'),'-ImagePath',str(crop_path)],capture_output=True)
            raw=completed.stdout.decode('utf-8',errors='replace')
            episode.update(ocr=raw,score=decode_score(raw) if completed.returncode==0 else None,image=str(image_path))
            episodes.append(episode)
            print(name,'điểm OCR:',episode['score'],'ước tính cột:',episode['crossing_estimate'],
                  'thời gian:',data['summary']['observed_span_s'],flush=True)
            artifact={'requested':args.runs,'reason':reason,'summary':summarize(episodes),'episodes':episodes,'interruptions':interruptions,'config':cfg}
            (ROOT/'evaluation'/f'{args.name}.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
            if args.target_score is not None and episode['score'] is not None and episode['score']>=args.target_score:
                reason='target reached';break
            if args.target_score is not None and episode['score'] is None and episode['crossing_estimate']>=args.target_score:
                reason='target estimate; confirm modal screenshot';break
    finally:
        listener.stop()
        if reason=='in_progress':reason='completed' if len(episodes)>=args.runs else 'stopped'
        artifact={'requested':args.runs,'reason':reason,'summary':summarize(episodes),'episodes':episodes,'interruptions':interruptions,'config':cfg}
        (ROOT/'evaluation'/f'{args.name}.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'reason':reason,'summary':artifact['summary']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
