"""Benchmark mô phỏng: không MSS, không trình duyệt, không click bàn phím."""
import argparse,json,time
from pathlib import Path
import numpy as np
import yaml
from bot.controller import Controller
from bot.vision import Gate
from bot.compiled_planner import warmup
from bot.reachability import warmup as warmup_atlas
from sim.engine import SimulationConfig,run_episode

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=int,default=10);parser.add_argument('--gates',type=int,default=12)
    parser.add_argument('--noise',type=float,default=1.5);parser.add_argument('--output',type=Path,default=Path('evaluation/standalone-simulation.json'))
    parser.add_argument('--observed-physics',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'profiles/windows-click.yaml').read_text(encoding='utf-8'))
    cfg.update(safety_shield=False,planner_budget_ms=0,input_call_ms=0,hitbox_half_width=26,hitbox_half_height=24.5,atlas_synchronous=True,simulation_observer=args.observed_physics)
    warmup();warmup_atlas();rows=[];started=time.perf_counter()
    for index in range(args.runs):
        seed=20261012+index;rng=np.random.default_rng(seed)
        gates=[Gate(550+195*i,636+195*i,float(center-60),float(center+60)) for i,center in enumerate(rng.uniform(122,236,args.gates))]
        simulation=SimulationConfig(gravity=cfg['gravity'],jump=cfg['jump'],speed=cfg['pipe_speed'],latency=cfg['input_effect_latency'],noise_y=args.noise)
        result={}
        for mode in ('maxmin','reachability'):
            result[mode]=run_episode(Controller(dict(cfg,planner=mode)),gates,simulation,seed,args.gates)
        rows.append(dict(seed=seed,results=result))
        print(f'{index+1}/{args.runs}: maxmin={result["maxmin"]["score"]}, reachability={result["reachability"]["score"]}',flush=True)
    summary={}
    for mode in ('maxmin','reachability'):
        scores=[r['results'][mode]['score'] for r in rows]
        times=[t for r in rows for t in r['results'][mode]['controller_ms']]
        summary[mode]=dict(median=float(np.median(scores)),min=min(scores),max=max(scores),completed=sum(s==args.gates for s in scores),controller_median_ms=float(np.median(times)),controller_p95_ms=float(np.percentile(times,95)),controller_p99_ms=float(np.percentile(times,99)),controller_max_ms=max(times))
    artifact=dict(method='Mô phỏng liên tục, cùng seed và chuỗi khe 122–236, hitbox thật và model 52×49. DP tính đồng bộ để thời gian ảo không chạy vượt worker. Không đo FPS MSS hay điểm website.',state_estimation='Physics ước lượng từ vị trí nhiễu' if args.observed_physics else 'Vận tốc thật của simulator',runs=args.runs,gates=args.gates,noise=args.noise,wall_s=time.perf_counter()-started,summary=summary,episodes=rows)
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
