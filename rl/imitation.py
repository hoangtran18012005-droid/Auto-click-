"""Học hành động planner ngoại tuyến, giữ model cũ và kiểm tra độc lập."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
from .env import FlappyEnv,EnvConfig,SCHEMA
from .network import Network
from .evaluate import evaluate

ROOT=Path(__file__).resolve().parents[1]

def greedy(network,state,mask):
    return int(np.argmax(np.where(mask,network.forward(state[None])[0],-1e9)))

def difficult(state):
    """Chỉ dùng các cột nhìn thấy: sắp vào ống hoặc đổi độ cao đáng kể."""
    rows=state[8:20].reshape(3,4);visible=rows[rows[:,3]>.5]
    if not len(visible):return False
    arrival=(float(visible[0,0])*900-25)/205
    if arrival<.9:return True
    if len(visible)>1:
        change=abs(float(visible[1,1]-visible[0,1]))*535
        second_arrival=(float(visible[1,0])*900-25)/205
        return change>=45 and second_arrival<2.5
    return False

class PlannerTeacher:
    def __init__(self):
        import yaml
        from bot.controller import Controller
        from bot.physics import Physics
        from bot.source_model import configure
        from bot.source_planner import warmup
        cfg=configure(yaml.safe_load((ROOT/'profiles/windows-source.yaml').read_text(encoding='utf-8')))
        cfg['source_planner_budget_ms']=0
        self.controller=Controller(cfg);self.physics=Physics(cfg);warmup()
    def label(self,env,state):
        # Không gọi detector/capture; tọa độ và pending input của cùng simulator.
        from bot.vision import Bird,Gate,Observation
        visible=state[8:20].reshape(3,4);gates=[]
        for x,top,bottom,mask in visible:
            if mask>.5:
                left=(float(x)*900+216)*2/3
                gates.append(Gate(left,left+124*2/3,float(top)*535*2/3,float(bottom)*535*2/3))
        bird=Bird(144,(float(state[0])*535-.5)*2/3,48,48)
        obs=Observation(bird,gates,np.empty((400,600,0),np.uint8))
        now=env.sim.tick/120;physics=self.physics
        physics.v=float(state[1])*700*2/3
        physics.pending=[t/120 for t in env.sim.pending if t>env.sim.tick]
        self.controller.last_flap=env.last_click/120
        self.controller.cfg['reaction_interval']=env.last_ticks/120
        decision=self.controller.decide(obs,physics,now)
        action=int(decision.flap)
        margins=(decision.coast_margin,decision.flap_margin)
        accepted=(bool(env.mask()[action]) and decision.feasible and decision.plan_margin is not None
                  and decision.plan_margin>=2 and all(v is not None and np.isfinite(v) for v in margins)
                  and margins[action]-margins[1-action]>=.25)
        return action,bool(accepted),decision
    def close(self):self.controller.close()

def collect(network,seeds,config,query_every=2,teacher=None,critical_only=False,risk_threshold=0.,expert_rollout=False):
    if query_every<1:raise ValueError('query_every phải dương.')
    if expert_rollout and query_every!=1:raise ValueError('Planner phải ra quyết định ở mọi bước khi điều khiển cả lượt.')
    teacher=teacher or PlannerTeacher();states=[];labels=[];anchors=[];timings=[];episodes=[]
    queries=accepted=overrides=0;margins=[]
    try:
        for index,seed in enumerate(seeds):
            cfg=replace(config,alternating=(index%5==0))
            env=FlappyEnv(cfg);state,info=env.reset(seed);steps=0
            while True:
                action=greedy(network,state,info['action_mask'])
                hard=difficult(state)
                if expert_rollout and info['action_mask'][1]:
                    start=time.perf_counter();suggested,valid,decision=teacher.label(env,state)
                    timings.append((time.perf_counter()-start)*1000);queries+=1
                    action_margins=(decision.coast_margin,decision.flap_margin)
                    action=int(suggested) if info['action_mask'][suggested] else 0
                    if valid:
                        states.append(state.copy());labels.append(action);margins.append(action_margins);accepted+=1
                        if action==greedy(network,state,info['action_mask']):anchors.append(state.copy())
                    overrides+=int(action!=greedy(network,state,info['action_mask']))
                elif not expert_rollout and hard and info['action_mask'][1] and steps%query_every==0:
                    start=time.perf_counter();suggested,valid,decision=teacher.label(env,state)
                    timings.append((time.perf_counter()-start)*1000);queries+=1
                    action_margins=(decision.coast_margin,decision.flap_margin)
                    if critical_only:
                        # Không sửa khác biệt sở thích khi action cũ vẫn an toàn.
                        valid=valid and action_margins[action] is not None and action_margins[action]<risk_threshold
                        if not valid and action_margins[action] is not None and action_margins[action]>=max(2.,risk_threshold):
                            anchors.append(state.copy())
                    if valid:
                        states.append(state.copy());labels.append(suggested);margins.append(action_margins);accepted+=1
                        overrides+=int(action!=suggested);action=suggested
                elif not expert_rollout and not hard and steps%query_every==0:
                    anchors.append(state.copy())
                state,_,done,truncated,info=env.step(action);steps+=1
                if done or truncated:break
            episodes.append(dict(seed=seed,alternating=cfg.alternating,score=info['score'],seconds=info['seconds'],failure=info.get('failure')))
            print('Thu thập',index+1,'/',len(seeds),'cột',info['score'],'nhãn',accepted,flush=True)
    finally:teacher.close()
    if not states or not anchors:raise RuntimeError('Chưa đủ nhãn planner và trạng thái neo để học.')
    metadata=dict(schema=SCHEMA,seeds=list(seeds),episodes=episodes,queries=queries,accepted=accepted,overrides=overrides,
                  critical_only=critical_only,risk_threshold=risk_threshold,expert_rollout=expert_rollout,margins=margins,
                  action_counts=np.bincount(labels,minlength=2).tolist(),anchors=len(anchors),query_every=query_every,
                  planner_ms=dict(median=float(np.median(timings)),p95=float(np.percentile(timings,95))))
    return np.asarray(states,np.float32),np.asarray(labels,np.int64),np.asarray(anchors,np.float32),metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'models/dqn-imitation')
    p.add_argument('--episodes',type=int,default=8);p.add_argument('--seed-start',type=int,default=1200000)
    p.add_argument('--max-score',type=int,default=20);p.add_argument('--query-every',type=int,default=2)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch',type=int,default=64)
    p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--anchor-weight',type=float,default=1.)
    p.add_argument('--expand-hidden',type=int,help='Mở rộng MLP mà giữ đầu ra checkpoint cũ lúc khởi tạo.')
    p.add_argument('--margin',type=float,default=.05);p.add_argument('--critical-only',action='store_true')
    p.add_argument('--risk-threshold',type=float,default=0.,help='Dạy sớm khi biên an toàn hành động cũ thấp hơn ngưỡng này (px ảnh chuẩn hóa).')
    p.add_argument('--expert-rollout',action='store_true',help='Planner điều khiển trọn episode; cần --query-every 1.')
    p.add_argument('--dataset',type=Path,help='Tái sử dụng dataset đã thu, không gọi planner lại.')
    args=p.parse_args()
    if min(args.episodes,args.max_score,args.epochs,args.batch,args.query_every)<1 or not 0<args.learning_rate<1 or args.anchor_weight<0 or args.margin<=0 or not 0<=args.risk_threshold<=20:p.error('Thông số không hợp lệ.')
    validation_seeds=list(range(10000,10012))
    if not args.dataset and set(range(args.seed_start,args.seed_start+args.episodes))&set(validation_seeds+[11000]):p.error('Seed học trùng seed validation.')
    if (args.output/'best.npz').exists():p.error('Chọn output mới để bảo toàn checkpoint đã có.')
    args.output.mkdir(parents=True,exist_ok=True)
    model,previous=Network.load(args.resume);reference,_=Network.load(args.resume)
    if args.expand_hidden:
        try:model=model.expanded(args.expand_hidden,seed=13)
        except ValueError as exc:p.error(str(exc))
    # Adam của RL có momentum khác loss imitation; khởi tạo lại optimizer.
    model.m={k:np.zeros_like(v) for k,v in model.params.items()};model.v={k:np.zeros_like(v) for k,v in model.params.items()};model.updates=0
    if args.dataset:
        with np.load(args.dataset,allow_pickle=False) as data:
            states=data['states'];actions=data['actions'];anchors=data['anchors'];dataset_meta=json.loads(str(data['metadata']))
            if dataset_meta['schema']!=SCHEMA:raise ValueError('Dataset khác schema.')
            if states.ndim!=2 or states.shape[1]!=22 or len(states)!=len(actions) or not len(anchors):raise ValueError('Dataset không hợp lệ.')
            if np.any((actions!=0)&(actions!=1)) or not np.all(np.isfinite(states)) or not np.all(np.isfinite(anchors)):raise ValueError('Nhãn/state dataset không hợp lệ.')
    else:
        cfg=EnvConfig(max_score=args.max_score,max_steps=max(3000,int((args.max_score*1.42+6)*30)))
        seeds=list(range(args.seed_start,args.seed_start+args.episodes))
        states,actions,anchors,dataset_meta=collect(reference,seeds,cfg,args.query_every,critical_only=args.critical_only,risk_threshold=args.risk_threshold,expert_rollout=args.expert_rollout)
    np.savez_compressed(args.output/'dataset.npz',states=states,actions=actions,anchors=anchors,metadata=np.array(json.dumps(dataset_meta)))
    (args.output/'dataset-summary.json').write_text(json.dumps(dataset_meta,ensure_ascii=False,indent=2),encoding='utf-8')
    if set(dataset_meta['seeds'])&set(validation_seeds+[11000]):raise ValueError('Seed học trùng seed validation.')
    cfg=EnvConfig(max_score=100,max_steps=5000)
    def validate():
        normal=evaluate(model,validation_seeds,cfg);hard=evaluate(model,[11000],replace(cfg,alternating=True))
        return dict(normal=normal,alternating=hard,objective=.8*normal['mean']+.2*hard['mean'])
    initial=validate();best=initial['objective'];rng=np.random.default_rng(13)
    anchor_values=reference.forward(anchors)
    initial_metadata=dict(previous,selection_objective=best)
    reference.save(args.output/'best.npz',initial_metadata)
    log_path=args.output/'imitation.jsonl'
    with log_path.open('w',encoding='utf-8') as log:
        log.write(json.dumps(dict(event='initial_validation',evaluation=initial))+'\n');log.flush()
        for epoch in range(1,args.epochs+1):
            order=rng.permutation(len(states));losses=[]
            for start in range(0,len(order),args.batch):
                ids=order[start:start+args.batch];anchor_ids=rng.integers(0,len(anchors),size=args.batch)
                losses.append(model.fit_actions(states[ids],actions[ids],np.ones(len(ids),np.float32),args.learning_rate,
                                               margin=args.margin,anchors=anchors[anchor_ids],anchor_values=anchor_values[anchor_ids],anchor_weight=args.anchor_weight))
            if epoch<=10 or epoch%10==0 or epoch==args.epochs:
                result=validate();agreement=float(np.mean(np.argmax(model.forward(states),axis=1)==actions))
                metadata=dict(algorithm='DQN planner imitation',previous=previous,epoch=epoch,dataset_seeds=dataset_meta['seeds'],
                              settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},selection_objective=result['objective'])
                model.save(args.output/'last.npz',metadata);model.save(args.output/f'epoch-{epoch}.npz',metadata)
                if result['objective']>best:best=result['objective'];model.save(args.output/'best.npz',metadata)
                log.write(json.dumps(dict(event='validation',epoch=epoch,loss=float(np.mean(losses)),agreement=agreement,evaluation=result))+'\n');log.flush()
                print('Epoch',epoch,'agreement',round(agreement,3),'ngẫu nhiên',result['normal']['mean'],'cực trị',result['alternating']['mean'],'objective',round(result['objective'],3),flush=True)
    print(args.output/'best.npz',flush=True)

if __name__=='__main__':main()
