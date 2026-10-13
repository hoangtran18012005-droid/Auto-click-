"""Đánh giá baseline trên chuỗi cố định, không mở browser hoặc gửi input."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import yaml
from bot.controller import Controller
from sim.engine import SimulationConfig,source_random_sequence,run_episode
from sim.sequences import rebuild_sequence
from bot.collision_geometry import body_extents,body_center_offset_y
from bot.vision import Bird

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--config',type=Path,default=ROOT/'config.yaml')
    parser.add_argument('--compare-config',type=Path,help='So sánh ghép cặp cùng chuỗi và seed nhiễu; controller chưa thay đổi ở Bước 0.')
    parser.add_argument('--log-sequence',type=Path)
    parser.add_argument('--latency-ms',type=float,default=45.)
    parser.add_argument('--noise-y',type=float,default=1.5)
    parser.add_argument('--observed-physics',action='store_true',help='Cả hai controller chỉ ước lượng v từ ảnh nhiễu, không nhận vận tốc thật simulator.')
    parser.add_argument('--input-drop-probability',type=float,default=0.)
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/baseline.json')
    args=parser.parse_args()
    if args.runs<30:parser.error('Cần ít nhất 30 chuỗi/lượt.')
    raw=args.config.read_bytes();cfg=yaml.safe_load(raw)
    simulated_bird=Bird(145.,0.,52.,49.)
    collision_hw,collision_hh=body_extents(cfg,simulated_bird)
    config=SimulationConfig(gravity=cfg.get('gravity',687.4),jump=cfg.get('jump',-264.2),
        speed=cfg.get('pipe_speed',141.),latency=args.latency_ms/1000,noise_y=args.noise_y,
        input_drop_probability=args.input_drop_probability,bird_x=cfg.get('simulation_bird_x',148.),
        bird_width=cfg.get('simulation_bird_width',52.),bird_height=cfg.get('simulation_bird_height',49.),
        ceiling=cfg.get('world_ceiling_px',0.),floor=cfg.get('world_floor_px',356.),
        cooldown=cfg.get('cooldown',.18),collision_half_width=collision_hw,
        collision_half_height=collision_hh,collision_center_offset_y=body_center_offset_y(cfg))
    rebuilt,metadata=rebuild_sequence(args.log_sequence) if args.log_sequence else (None,None)
    results=[];paired=[]
    compare=yaml.safe_load(args.compare_config.read_bytes()) if args.compare_config else None
    if args.observed_physics:
        cfg=dict(cfg,simulation_observer=True)
        if compare is not None:compare=dict(compare,simulation_observer=True)
    for index in range(args.runs):
        seed=args.seed+index
        gates=rebuilt if rebuilt is not None else source_random_sequence(seed,
            frame_width=cfg.get('width',600),canvas_width=cfg.get('source_canvas_width',900),
            ground_y=cfg.get('source_ground_y',535),gap=cfg.get('source_gap',180),
            pipe_speed=cfg.get('source_pipe_speed',205),
            spawn_interval=cfg.get('source_spawn_interval',1.42),
            pipe_width=cfg.get('source_pipe_width',92),lip_side=cfg.get('source_lip_side',16))
        result=run_episode(Controller(dict(cfg)),gates,config,seed)
        results.append(result)
        if compare is not None:
            paired.append(run_episode(Controller(dict(compare)),gates,config,seed))
        if (index+1)%5==0:print(f'Đã đánh giá {index+1}/{args.runs} lượt.',flush=True)
    scores=np.array([r['score'] for r in results])
    times=[value for r in results for value in r['controller_ms']]
    summary={'runs':args.runs,'median':float(np.median(scores)),'q25':float(np.percentile(scores,25)),
             'q75':float(np.percentile(scores,75)),'min':int(scores.min()),'max':int(scores.max()),
             'reaching_50_pct':float(100*np.mean(scores>=50)) if len(gates)>=50 else None,
             'deaths':dict(Counter(r['death_transition'] for r in results)),
             'controller_median_ms':statistics.median(times),'controller_p95_ms':float(np.percentile(times,95))}
    artifact={'summary':summary,'config':cfg,'config_sha256':hashlib.sha256(raw).hexdigest(),
              'seed':args.seed,'sequence_source':metadata or 'random','episodes':results}
    if paired:
        differences=np.array([other['score']-base['score'] for base,other in zip(results,paired)])
        bootstrap=np.random.default_rng(args.seed+999).choice(differences,(5000,len(differences))).mean(axis=1)
        artifact['paired_comparison']={'mean_score_difference':float(differences.mean()),
            'bootstrap_95_ci':np.percentile(bootstrap,[2.5,97.5]).tolist(),'episodes':paired,'config':compare}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(artifact,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n| Lượt | Trung vị | IQR (Q25–Q75) | Min | Đạt 50 (%) | Controller trung vị/p95 (ms) |')
    print('|---:|---:|---:|---:|---:|---:|')
    reach=f"{summary['reaching_50_pct']:.1f}" if summary['reaching_50_pct'] is not None else 'Không đủ 50 cột'
    print(f"| {args.runs} | {summary['median']:.1f} | {summary['q25']:.1f}–{summary['q75']:.1f} | {summary['min']} | {reach} | {summary['controller_median_ms']:.2f}/{summary['controller_p95_ms']:.2f} |")
    print('Va chạm theo loại chuyển khe:',summary['deaths'])
    if paired:print('So sánh ghép cặp:',{k:v for k,v in artifact['paired_comparison'].items() if k not in ('episodes','config')})


if __name__=='__main__':main()
