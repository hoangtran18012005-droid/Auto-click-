"""Fit các đoạn bay chưa flap bằng timestamp thực; không thay cấu hình tự động."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.time_alignment import input_time

ROOT=Path(__file__).resolve().parents[1]

def fit_segment(points,flap_time,latency):
    if len(points)<7:return None
    t=np.array([f['t'] for f in points]);y=np.array([f['y'] for f in points])
    if np.ptp(t)<.20:return None
    x=t-t[0];a,v,y0=np.polyfit(x,y,2);residual=float(np.sqrt(np.mean((y-np.polyval((a,v,y0),x))**2)))
    gravity=2*a
    if not 400<gravity<1000 or residual>2:return None
    jump=v-gravity*(t[0]-flap_time-latency) if flap_time is not None else None
    if jump is not None and not -400<jump<-150:jump=None
    return {'g':float(gravity),'jump':float(jump) if jump is not None else None,'rmse_px':residual,'duration_s':float(np.ptp(t)),'points':len(points)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/physics-fit.json')
    parser.add_argument('--latency-ms',type=float,default=45.)
    parser.add_argument('--patterns',nargs='+',default=['local-*.json'])
    parser.add_argument('--latency-origin',choices=['frame','input_start','input_end'],default='input_start')
    args=parser.parse_args();latency=args.latency_ms/1000;segments=[];speeds=[];collision_bounds=[];runs={}
    paths=sorted({path for pattern in args.patterns for path in (ROOT/'logs').glob(pattern)})
    for path in paths:
        frames=json.loads(path.read_text(encoding='utf-8'))['frames'];points=[];flap_time=None;local=[];previous=None
        for frame in frames:
            if frame['flap']:
                fitted=fit_segment(points,flap_time,latency)
                if fitted:local.append(fitted)
                points=[];flap_time=input_time(frame,args.latency_origin)
            elif flap_time is None or frame['t']-flap_time>latency+.06:points.append(frame)
            if previous:
                dt=frame['t']-previous['t']
                for gate in frame['gates']:
                    matches=[g for g in previous['gates'] if abs(g['top']-gate['top'])<3 and abs(g['right']-gate['right'])<30]
                    if matches and dt>0:
                        old=min(matches,key=lambda g:abs(g['right']-gate['right']));speed=(old['right']-gate['right'])/dt
                        if 50<speed<250:speeds.append(speed)
            previous=frame
        fitted=fit_segment(points,flap_time,latency)
        if fitted:local.append(fitted)
        segments.extend(local);runs[path.stem]=local
        if frames:
            f=frames[-1];overlaps=[g for g in f['gates'] if g['x']<=f['x']+f['w']/2 and g['right']>=f['x']-f['w']/2]
            if overlaps:
                margin=min(min(f['y']-f['h']/2-g['top'],g['bottom']-f['y']-f['h']/2) for g in overlaps)
                collision_bounds.append({'run':path.stem,'last_bbox_clearance_px':margin})
    def stats(values):
        return {'n':len(values),'median':float(np.median(values)),'q25':float(np.percentile(values,25)),'q75':float(np.percentile(values,75))} if values else None
    result={'latency_ms_assumed':args.latency_ms,'latency_origin':args.latency_origin,'input_files':[path.name for path in paths],'gravity':stats([s['g'] for s in segments]),'jump':stats([s['jump'] for s in segments if s['jump'] is not None]),'speed':stats(speeds),'rmse_px':stats([s['rmse_px'] for s in segments]),'runs':runs,'collision_bounds':collision_bounds,
            'hitbox_note':'Clearance bbox frame cuối chỉ là quan sát bị kiểm duyệt theo thời gian/rotation. Chưa đủ xác định hitbox thật; mặc định hitbox_margin_px=0, không tự giảm hitbox.'}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','collision_bounds')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
