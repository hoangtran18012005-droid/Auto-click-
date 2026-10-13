"""Đánh giá greedy trên seed tách train; có đối chiếu MPC tùy chọn."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .env import FlappyEnv,EnvConfig
from .network import Network

ROOT=Path(__file__).resolve().parents[1]

def evaluate(network,seeds,config=None,controller_factory=None):
    episodes=[];times=[]
    for seed in seeds:
        env=FlappyEnv(config);state,info=env.reset(seed);total=0.;controller=controller_factory() if controller_factory else None
        model=None
        if controller:
            from bot.physics import Physics
            model=Physics(controller.cfg)
        while True:
            start=time.perf_counter()
            if controller:
                obs=env.sim.observe();now=env.sim.tick/120
                # Hai policy nhận cùng vị trí/vận tốc oracle (có cùng nhiễu).
                # Đây là đối chiếu thuật toán, không phải pipeline vision Windows.
                obs.bird.y=(float(state[0])*535-.5)*2/3
                model.update(obs.bird,obs.gates,now);model.v=float(state[1])*700*2/3
                controller.cfg['reaction_interval']=env.last_ticks/120
                action=int(controller.decide(obs,model,now).flap)
            else:action=int(np.argmax(np.where(info['action_mask'],network.forward(state[None])[0],-1e9)))
            times.append(1000*(time.perf_counter()-start))
            action_time=env.sim.tick/120
            state,reward,done,truncated,info=env.step(action);total+=reward
            if controller and info['executed_action']:model.flap(action_time)
            if done or truncated:break
        if controller:controller.close()
        episodes.append(dict(seed=seed,score=info['score'],seconds=info['seconds'],return_=total,terminated=done,truncated=truncated,failure=info.get('failure'),last_action=info['executed_action']))
    scores=[r['score'] for r in episodes]
    return dict(episodes=episodes,mean=float(np.mean(scores)),median=float(np.median(scores)),min=min(scores),max=max(scores),
                completion_rate=float(np.mean([r['score']>=env.config.max_score for r in episodes])),
                inference_ms=dict(median=float(np.median(times)),p95=float(np.percentile(times,95)),max=max(times)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=30);p.add_argument('--seed-start',type=int,default=20000)
    p.add_argument('--max-score',type=int,default=100);p.add_argument('--compare-mpc',action='store_true')
    p.add_argument('--max-steps',type=int,default=10000,help='Giới hạn bước mỗi episode; tăng khi thử 1.000 cột.')
    p.add_argument('--noise-y',type=float,default=0);p.add_argument('--latency-jitter',type=float,default=0)
    p.add_argument('--alternating',action='store_true',help='Khe cao/thấp luân phiên cực trị.')
    p.add_argument('--output',type=Path,default=ROOT/'evaluation/dqn-evaluation.json');args=p.parse_args()
    if args.episodes<1 or args.max_score<1 or args.max_steps<1:p.error('Số episode/điểm/bước phải dương.')
    net,meta=Network.load(args.checkpoint);cfg=EnvConfig(max_score=args.max_score,max_steps=args.max_steps,noise_y=args.noise_y,latency_jitter=args.latency_jitter,alternating=args.alternating)
    seeds=list(range(args.seed_start,args.seed_start+args.episodes))
    from dataclasses import asdict
    result=dict(checkpoint=str(args.checkpoint),metadata=meta,environment=asdict(cfg),limits='Hai policy dùng cùng state oracle/noise; chưa có CV/tracker thật. Điểm là số cột vượt, không phải điểm website. Thời gian MPC không cộng vào đồng hồ game ảo.',dqn=evaluate(net,seeds,cfg))
    if args.compare_mpc:
        import yaml
        from bot.controller import Controller
        from bot.source_model import configure
        from bot.source_planner import warmup
        settings=configure(yaml.safe_load((ROOT/'profiles/windows-source.yaml').read_text(encoding='utf-8')));settings['source_planner_budget_ms']=0;warmup()
        result['mpc']=evaluate(None,seeds,cfg,lambda:Controller(dict(settings)))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='episodes'} for k,v in result.items() if k in ('dqn','mpc')},ensure_ascii=False,indent=2));print(args.output)

if __name__=='__main__':main()
