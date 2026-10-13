"""So sánh hành động planner trên cùng frame/oracle đã lưu, không chơi game."""
import argparse,csv,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from bot.controller import Controller
from bot.vision import Bird,Gate,Observation
from tools.regret import action_values,reconstruct_velocity,summarize

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',type=Path,nargs='+',required=True)
    parser.add_argument('--pattern',default='local-native-*.json')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/planner-replay')
    parser.add_argument('--stride',type=int,default=1)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    variants={path.stem:yaml.safe_load(path.read_text(encoding='utf-8')) for path in args.configs}
    rows={name:[] for name in variants};cache={};frame=np.zeros((495,600,3),np.uint8)
    for path in sorted((ROOT/'logs').glob(args.pattern)):
        source=json.loads(path.read_text(encoding='utf-8'))['frames'];previous=None;last_flap=-1e9
        for index,f in enumerate(source):
            velocity=reconstruct_velocity(f,previous)
            if index%args.stride==0 and velocity is not None:
                obs=Observation(Bird(f['x'],f['y'],f['w'],f['h']),[Gate(**g) for g in f['gates']],frame)
                model=SimpleNamespace(g=f['g'],jump=f['jump'],speed=f['speed'],v=velocity)
                oracle=action_values(f,velocity)
                best=max(oracle)
                for name,cfg in variants.items():
                    controller=Controller(dict(cfg,reaction_interval=1/30));controller.last_flap=last_flap
                    start=time.perf_counter();decision=controller.decide(obs,model,f['t']);milliseconds=1000*(time.perf_counter()-start)
                    candidate=action_values(f,velocity,geometry_cfg=cfg)
                    candidate_best=max(candidate)
                    value=candidate[int(decision.flap)]
                    rows[name].append({'run':path.stem,'frame':index,'t':f['t'],'flap':decision.flap,'plan_margin':decision.plan_margin,'feasible':decision.feasible,'controller_ms':milliseconds,'oracle_margin_px':value,'regret_px':max(0,candidate_best-value)})
            if f['flap']:last_flap=f['t']
            previous=f
        print(f'Đã replay {path.name}',flush=True)
    summary={}
    for name,items in rows.items():
        with (args.output/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
        times=[r['controller_ms'] for r in items]
        summary[name]={**summarize(items),'median_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95))}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
