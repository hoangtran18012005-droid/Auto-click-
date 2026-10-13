"""So sánh cùng oracle JS 120 Hz; không capture/browser/input thật."""
import argparse,json,time
from pathlib import Path
import numpy as np
import yaml
from bot.controller import Controller
from bot.physics import Physics
from bot.source_model import configure
from bot.source_planner import warmup
from sim.source_engine import SourceSimulator,sequence

ROOT=Path(__file__).resolve().parents[1]

def episode(cfg,pipes,seed,noise,pipeline_delay=0):
    sim=SourceSimulator(pipes,535,cfg['latency']+pipeline_delay,seed,noise)
    controller=Controller(dict(cfg));model=Physics(controller.cfg);times=[];actions=0;gates_seen=0
    deadline=(pipes[-1]['x']+108-191)/205+1
    while not sim.dead and sim.tick/120<deadline and sim.score<len(pipes):
        now=sim.tick/120;obs=sim.observe();gates_seen=max(gates_seen,len(obs.gates))
        model.update(obs.bird,obs.gates,now)
        controller.cfg['reaction_interval']=1/30
        start=time.perf_counter();d=controller.decide(obs,model,now,pipeline_delay);times.append(1000*(time.perf_counter()-start))
        if d.flap:sim.flap();model.flap(now+pipeline_delay);actions+=1
        sim.elapsed(1/30)
    controller.close()
    return dict(seed=seed,score=sim.score,complete=sim.score==len(pipes),duration_s=sim.tick/120,flaps=actions,max_visible_gates=gates_seen,
                controller_ms=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),p99=float(np.percentile(times,99)),max=max(times)))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runs',type=int,default=6);parser.add_argument('--gates',type=int,default=12);parser.add_argument('--noise',type=float,default=1);parser.add_argument('--beam',type=int,default=600);parser.add_argument('--pipeline-delay-ms',type=float,default=0);parser.add_argument('--output',type=Path,default=ROOT/'evaluation/source-simulation.json');args=parser.parse_args()
    if args.runs<1 or args.gates<2 or args.beam<2 or args.noise<0 or args.pipeline_delay_ms<0:parser.error('Số lượt/cột/beam hoặc nhiễu/độ trễ không hợp lệ.')
    cfg=configure(yaml.safe_load((ROOT/'profiles/windows-source.yaml').read_text(encoding='utf-8')));cfg.update(source_planner_budget_ms=0,beam_width=args.beam)
    warmup()
    from bot.compiled_planner import warmup as warmup_old
    warmup_old()
    results={}
    for planner in ('maxmin','source_euler'):
        settings=dict(cfg,planner=planner,use_numba=planner=='maxmin',separate_actions=True)
        rows=[]
        for seed in range(args.runs):
            row=episode(settings,sequence(seed,args.gates,535,alternating=seed%2==0),seed,args.noise,args.pipeline_delay_ms/1000);rows.append(row)
            print(planner,seed,row['score'],row['controller_ms'],flush=True)
        results[planner]=dict(completed=sum(r['complete'] for r in rows),episodes=rows)
    result=dict(config=cfg,runs=args.runs,gates=args.gates,noise=args.noise,pipeline_delay_ms=args.pipeline_delay_ms,oracle='JS fixed 120 Hz, native 900x600, groundY535',spawn_range=[90,265],spawn_range_validated=True,limits='Độ trễ synthetic được khai báo; thời gian CPU đo được không tự cộng vào thời gian ảo. Chưa có FPS MSS hoặc gameplay website.',results=results)
    args.output.parent.mkdir(exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(args.output)

if __name__=='__main__':main()
