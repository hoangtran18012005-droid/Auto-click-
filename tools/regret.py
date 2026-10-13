"""Regret từng frame theo oracle lưới gốc; không thay đổi controller."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from exact_clearance import maxmin_clearance,G,J,SPD,BX
from sim.engine import transition_type
from bot.physics import project
from bot.vision import Bird,Gate
from bot.collision_geometry import body_extents,body_center_offset_y,world_clearance_bounds


def reconstruct_velocity(frame,previous):
    if frame.get('pre_flap_v') is not None:return float(frame['pre_flap_v'])
    if not frame.get('flap'):return float(frame['v'])
    if previous is None:return None
    dt=frame['t']-previous['t']
    if dt<=0:return None
    return float(previous['v']+frame.get('g',G)*dt)


def action_values(frame,velocity,latency=0.,qy=1.,qv=4.,geometry_cfg=None):
    x=float(frame.get('x',BX))
    if geometry_cfg is None:
        hw=float(frame.get('collision_hw',frame['w']/2));hh=float(frame.get('collision_hh',frame['h']/2))
        y_offset=float(frame.get('collision_center_offset_y',0.))
    else:
        bird=Bird(x,float(frame['y']),float(frame['w']),float(frame['h']))
        hw,hh=body_extents(geometry_cfg,bird)
        y_offset=body_center_offset_y(geometry_cfg)
    g=float(frame.get('g',G));jump=float(frame.get('jump',J));speed=float(frame.get('speed',SPD))
    ceiling=float(frame.get('world_ceiling_px',0));floor=float(frame.get('world_floor_px',356))
    if geometry_cfg is not None:
        ceiling=float(geometry_cfg.get('world_ceiling_px',ceiling))
        floor=float(geometry_cfg.get('world_floor_px',floor))
        ceiling,floor=world_clearance_bounds(geometry_cfg,hh,ceiling,floor)
    gates=[dict(gate) for gate in frame.get('gates',[]) if gate['right']>x-hw]
    y=float(frame['y'])+y_offset;prefix=1e3
    if latency:
        from sim.engine import interval_clearance
        prefix=interval_clearance(y,velocity,g,0,latency,ceiling+hh,floor-hh)
        for gate in gates:
            entry=max(0,(gate['x']-x-hw)/speed)
            exit=min(latency,(gate['right']-x+hw)/speed)
            if 0<=entry<=exit:
                prefix=min(prefix,interval_clearance(y,velocity,g,entry,exit,gate['top']+hh,gate['bottom']-hh))
        y,velocity=project(y,velocity,g,latency)
        for gate in gates:gate['x']-=speed*latency;gate['right']-=speed*latency
    options={'g':g,'jump':jump,'spd':speed,'hw':hw,'hh':hh,'x':x,
             'ymin':ceiling,'ymax':floor,'every':2,'qy':qy,'qv':qv}
    if not gates:options['horizon']=1.
    values=[min(prefix,maxmin_clearance(y,velocity,gates,first_action=action,**options)) for action in (False,True)]
    return values


def summarize(rows):
    values=np.array([r['regret_px'] for r in rows if r.get('regret_px') is not None])
    if not len(values):return {'frames':0}
    return {'frames':len(values),'mean':float(values.mean()),'median':float(np.median(values)),
            'p95':float(np.percentile(values,95)),'over_5px':int(np.sum(values>5))}


def evaluate_log(path,output,latency,qy,qv,tail_seconds=None):
    start=time.perf_counter();data=json.loads(Path(path).read_text(encoding='utf-8'))
    frames=data['frames']
    if tail_seconds is not None and frames:
        cutoff=frames[-1]['t']-tail_seconds
        frames=[frame for frame in frames if frame['t']>=cutoff]
    rows=[];previous=None;config=data.get('summary',{}).get('config',{})
    for index,frame in enumerate(frames):
        velocity=reconstruct_velocity(frame,previous)
        x=frame.get('x',BX);hw=frame.get('collision_hw',frame['w']/2)
        active=[Gate(**gate) for gate in frame.get('gates',[]) if gate['right']>x-hw]
        kind=transition_type(active[0] if active else None,active[1] if len(active)>1 else None)
        row={'frame':index,'t':frame['t'],'action':bool(frame.get('flap')),'pre_flap_v':velocity,'transition':kind,
             'm_taken_px':None,'m_best_px':None,'regret_px':None,'status':'ok'}
        if frame.get('input_status')=='pending':row['status']='pending_action_not_modelled'
        elif velocity is None:row['status']='missing_pre_flap_v'
        else:
            values=action_values(frame,velocity,latency,qy,qv,config)
            taken=values[int(row['action'])];best=max(values)
            row.update(m_taken_px=taken,m_best_px=best,regret_px=max(0.,best-taken))
        rows.append(row);previous=frame
    destination=Path(output)/f'{Path(path).stem}.csv'
    with destination.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'run':Path(path).stem,**summarize(rows),'excluded':sum(r['status']!='ok' for r in rows),
             'seconds':time.perf_counter()-start,
             'by_transition':{kind:summarize([r for r in rows if r['transition']==kind]) for kind in ('high_to_low','low_to_high','similar','unknown')}}
    return summary,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',type=Path,default=ROOT/'logs')
    parser.add_argument('--pattern',default='local-*.json',help='Lọc tên log khi tái hiện một tập dữ liệu đã đóng băng.')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/regret')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--latency-ms',type=float,default=0.,help='Mặc định 0 để giữ điều kiện oracle gốc; simulator mặc định 45 ms.')
    parser.add_argument('--qy',type=float,default=1.)
    parser.add_argument('--qv',type=float,default=4.)
    parser.add_argument('--tail-seconds',type=float,help='Chỉ đánh giá đoạn cuối mỗi lượt để chẩn đoán va chạm nhanh.')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.logs.glob(args.pattern))
    if not paths:parser.error(f'Không có log khớp {args.pattern}.')
    summaries=[];all_rows=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(evaluate_log,path,args.output,args.latency_ms/1000,args.qy,args.qv,args.tail_seconds) for path in paths]
        for job in as_completed(jobs):
            summary,rows=job.result();summaries.append(summary);all_rows.extend(rows)
            print(f"{summary['run']}: {summary['frames']} frame, mean={summary['mean']:.3f}, p95={summary['p95']:.3f}, >5 px={summary['over_5px']}",flush=True)
    result={'oracle':'Oracle lưới có gộp trạng thái, Euler, cutoff -40 px; dùng g/jump/speed/x/hitbox từ frame; không phải chứng minh exact liên tục.',
            'latency_ms':args.latency_ms,'qy':args.qy,'qv':args.qv,'cooldown':'Oracle gốc không áp dụng cooldown.',
            'empty_gate_horizon_s':1.,'tail_seconds':args.tail_seconds,'global':summarize(all_rows),
            'by_transition':{kind:summarize([r for r in all_rows if r['transition']==kind]) for kind in ('high_to_low','low_to_high','similar','unknown')},
            'runs':sorted(summaries,key=lambda r:r['run']),'wall_seconds':time.perf_counter()-start}
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Tổng:',result['global'])


if __name__=='__main__':main()
