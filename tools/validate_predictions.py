"""Replay dự đoán frame kế tiếp, giữ cả thời gian ngay sau input."""
import argparse, json
from pathlib import Path
import numpy as np
import yaml
from bot.physics import Physics
from bot.vision import Bird, Gate
from tools.time_alignment import input_time

ROOT=Path(__file__).resolve().parents[1]


def evaluate(paths,cfg):
    errors=[];post_input=[]
    for path in paths:
        frames=json.loads(path.read_text(encoding='utf-8'))['frames']
        physics=Physics(dict(cfg));previous=None;last_action=-1e9
        for frame in frames:
            if 'y' not in frame:continue
            if previous is not None and 0<frame['t']-previous['t']<.2:
                predicted,_=physics.predict(previous['y'],previous['t'],frame['t']-previous['t'])
                error=predicted-frame['y'];errors.append(error)
                if frame['t']-last_action<.18:post_input.append(error)
            physics.update(Bird(frame['x'],frame['y'],frame['w'],frame['h']),[Gate(**g) for g in frame['gates']],frame['t'])
            if frame['flap']:
                physics.flap(input_time(frame,'start' if cfg.get('input_timestamp_origin')=='start' else 'end'))
                last_action=input_time(frame,'start')
            previous=frame
    def stats(values):
        a=np.array(values)
        return dict(n=len(a),rmse=float(np.sqrt(np.mean(a*a))),bias=float(np.mean(a)),median_abs=float(np.median(abs(a))),p95_abs=float(np.percentile(abs(a),95))) if len(a) else None
    return dict(all=stats(errors),post_input=stats(post_input))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',nargs='+',type=Path,required=True)
    parser.add_argument('--pattern',default='local-goal1000-fast-pilot-*.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--latency-grid-ms',nargs='+',type=float,help='Sweep cùng g; đổi J theo mốc latency để giữ intercept coast đã fit.')
    args=parser.parse_args();paths=sorted((ROOT/'logs').glob(args.pattern))
    if not paths:parser.error('Không có log khớp pattern.')
    results={}
    for path in args.configs:
        cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
        if args.latency_grid_ms:
            for milliseconds in args.latency_grid_ms:
                delay=milliseconds/1000
                case=dict(cfg,input_effect_latency=delay,jump=cfg['jump']+cfg['gravity']*(delay-cfg.get('input_effect_latency',.045)))
                result=evaluate(paths,case);result['config']=case
                results[f'{path}@{milliseconds:g}ms']=result
        else:results[str(path)]=evaluate(paths,cfg)
    artifact=dict(method='Sai số dự đoán frame kế tiếp theo cùng lệnh thật, gồm 180 ms sau input; không đo điểm counterfactual.',logs=[p.name for p in paths],results=results)
    args.output.write_text(json.dumps(artifact,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(results,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
