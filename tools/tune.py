"""Đánh giá độ nhạy trên các seed ghép cặp với nhiễu và sai số vật lý."""
import argparse,json
from pathlib import Path
import numpy as np
import yaml
from bot.controller import Controller
from bot.compiled_planner import warmup
from sim.engine import SimulationConfig,random_sequence,run_episode

ROOT=Path(__file__).resolve().parents[1]

def summary(episodes):
    values=np.array([e['score'] for e in episodes])
    return {'median':float(np.median(values)),'q25':float(np.percentile(values,25)),'q75':float(np.percentile(values,75)),'min':int(values.min()),'max':int(values.max()),'reaching_50_pct':float(100*np.mean(values>=50))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'evaluation/maxmin-numba.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/sensitivity.json')
    parser.add_argument('--runs',type=int,default=30)
    args=parser.parse_args()
    if args.runs<30:parser.error('Cần ≥30 chuỗi ghép cặp.')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'));warmup();result={}
    for scenario,sigma,noise_v,jitter,error in [('nominal',1.5,0,0,0),('moderate',1.5,15,.015,.04),('strong',3,30,.02,.04)]:
        groups={}
        for margin,hitbox in [(0,0),(2,0),(4,0),(0,2),(0,-2)]:
            label=f'margin{margin}_hitbox{hitbox}';episodes=[]
            for index in range(args.runs):
                seed=20261011+index
                settings=dict(cfg,margin=margin,hitbox_margin_px=hitbox,world_ceiling_px=0,world_floor_px=356)
                config=SimulationConfig(noise_y=sigma,noise_v=noise_v,latency_jitter=jitter,parameter_error=error)
                episodes.append(run_episode(Controller(settings),random_sequence(seed),config,seed))
            groups[label]={'summary':summary(episodes),'scores':[e['score'] for e in episodes]}
            print(scenario,label,groups[label]['summary'],flush=True)
        reference=np.array(groups['margin0_hitbox0']['scores'])
        for label,data in groups.items():
            diff=np.array(data['scores'])-reference
            bootstrap=np.random.default_rng(20261011).choice(diff,(5000,len(diff))).mean(axis=1)
            data['paired_mean_difference']=float(diff.mean());data['bootstrap_95_ci']=np.percentile(bootstrap,[2.5,97.5]).tolist()
        result[scenario]={'noise_y':sigma,'noise_v':noise_v,'latency_jitter':jitter,'parameter_error':error,'groups':groups}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
