"""Đối chiếu planner với oracle và thay từng yếu tố; không gửi input."""
import argparse
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from bot.controller import Controller
from bot.vision import Bird,Gate,Observation
from exact_clearance import maxmin_clearance
from tools.regret import reconstruct_velocity

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,default=ROOT/'logs/local-native-17-final-confirm.json')
    parser.add_argument('--config',type=Path,default=ROOT/'evaluation/live-calibration.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/diagnosis')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'));frames=json.loads(args.log.read_text(encoding='utf-8'))['frames']
    variants={'configured':{},'margin0':{'margin':0},'margin0_beam2000':{'margin':0,'beam_width':2000},
              'margin0_finebins':{'margin':0,'bin_y':1,'bin_v':4,'bin_cooldown':1/30},
              'margin0_dt30':{'margin':0,'search_dt':1/30,'max_steps':200},
              'margin0_latency0':{'margin':0,'latency':0},
              'margin0_oraclebounds':{'margin':0,'ceiling':0,'floor':400/495},
              'margin0_cooldown0':{'margin':0,'cooldown':0}}
    rows=[];last_flap=-1e9;previous=None
    for f in frames:
        velocity=reconstruct_velocity(f,previous)
        if 38<=f['t']<=42.2 and velocity is not None:
            obs=Observation(Bird(f['x'],f['y'],f['w'],f['h']),[Gate(**g) for g in f['gates']],np.zeros((495,600,3),np.uint8))
            model=SimpleNamespace(g=f['g'],jump=f['jump'],speed=f['speed'],v=velocity)
            gates=[g for g in f['gates'] if g['right']>145-f['w']/2]
            oracle=maxmin_clearance(f['y'],velocity,gates,every=2,hw=f['w']/2,hh=f['h']/2)
            for name,changes in variants.items():
                settings=dict(cfg,**changes,reaction_interval=1/30)
                controller=Controller(settings);controller.last_flap=last_flap
                start=time.perf_counter();decision=controller.decide(obs,model,f['t'])
                rows.append({'t':f['t'],'variant':name,'oracle_margin':oracle,'feasible':decision.feasible,'flap':decision.flap,'controller_ms':1000*(time.perf_counter()-start)})
            # Đối chứng vận tốc/tham số: cùng margin0, thay từng nguồn.
            for name,values in [('margin0_raw_v',dict(v=f['v'])),('margin0_measured_physics',dict(g=687.4,jump=-264.2,speed=141))]:
                model2=SimpleNamespace(**dict(vars(model),**values));controller=Controller(dict(cfg,margin=0,reaction_interval=1/30));controller.last_flap=last_flap
                start=time.perf_counter();decision=controller.decide(obs,model2,f['t'])
                rows.append({'t':f['t'],'variant':name,'oracle_margin':oracle,'feasible':decision.feasible,'flap':decision.flap,'controller_ms':1000*(time.perf_counter()-start)})
        if f['flap']:last_flap=f['t']
        previous=f
    with (args.output/'frames.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for name in dict.fromkeys(r['variant'] for r in rows):
        selected=[r for r in rows if r['variant']==name];safe=[r for r in selected if r['oracle_margin']>=0]
        failed=[r for r in safe if not r['feasible']]
        summary[name]={'frames':len(selected),'oracle_safe':len(safe),'false_infeasible':len(failed),'first_false_infeasible':failed[0]['t'] if failed else None,
                       'median_ms':float(np.median([r['controller_ms'] for r in selected])),'p95_ms':float(np.percentile([r['controller_ms'] for r in selected],95))}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
