"""Huấn luyện Double DQN NumPy trên CPU, không chạy browser/capture/click."""
import os
# Một thread BLAS tránh overhead ma trận nhỏ; đặt trước import NumPy.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,time
from collections import deque
from dataclasses import replace
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .env import FlappyEnv,EnvConfig
from .network import Network,Replay,double_targets,mixed_batch
from .evaluate import evaluate

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--steps',type=int,default=200000);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--output',type=Path,default=ROOT/'models/dqn-offline');p.add_argument('--resume',type=Path)
    p.add_argument('--eval-every',type=int,default=20000);p.add_argument('--eval-episodes',type=int,default=10)
    p.add_argument('--warmup',type=int,default=2000);p.add_argument('--epsilon-decay',type=int,default=100000)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--hidden',type=int,default=64);p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--epsilon-final',type=float,default=.05)
    p.add_argument('--curriculum-probability',type=float,default=0.,help='Tỷ lệ episode khe cao/thấp cực trị.')
    p.add_argument('--hard-replay-fraction',type=float,default=0.,help='Tỷ lệ minibatch lấy từ các bước trước va chạm đầu ống.')
    p.add_argument('--validation-max-score',type=int,default=30)
    args=p.parse_args()
    if min(args.steps,args.eval_every,args.eval_episodes,args.batch,args.hidden,args.epsilon_decay)<1 or args.warmup<0 or args.seed<1 or not 0<args.learning_rate<1:p.error('Thông số huấn luyện không hợp lệ; seed phải >=1.')
    if args.validation_max_score<1 or any(not 0<=v<=1 for v in (args.epsilon_final,args.curriculum_probability,args.hard_replay_fraction)):p.error('Điểm validation phải dương; tỷ lệ phải nằm trong [0,1].')
    args.output.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(args.seed)
    online,previous=Network.load(args.resume) if args.resume else (Network(args.seed,args.hidden),{})
    target=Network(hidden=online.hidden);target.copy_from(online);replay=Replay();config=EnvConfig()
    base_steps=int(previous.get('steps',0))
    base_episode=int(previous.get('episodes',0)) if previous.get('seed')==args.seed else 0
    hard=Replay(20000);recent=deque(maxlen=75)
    def reset_episode(index):
        mixed=replace(config,alternating=bool(rng.random()<args.curriculum_probability))
        fresh=FlappyEnv(mixed);state,info=fresh.reset(args.seed*100000+index)
        return fresh,state,info
    env, state,info=reset_episode(base_episode);episode=base_episode;start=time.perf_counter()
    validation_seeds=list(range(10000,10000+args.eval_episodes))
    validation_config=EnvConfig(max_score=args.validation_max_score,max_steps=max(3000,int((args.validation_max_score*1.42+6)*30)))
    def validate():
        result=evaluate(online,validation_seeds,validation_config)
        if args.curriculum_probability:
            result['alternating']=evaluate(online,[11000],replace(validation_config,alternating=True))
            result['objective']=.8*result['mean']+.2*result['alternating']['mean']
        else:result['objective']=result['mean']
        return result
    initial=validate()
    log_path=args.output/f'train-{time.time_ns()}.jsonl';loss=None;episodes=0;score_sum=0
    best=initial['objective']
    # Bản khởi đầu được giữ lại nếu lượt huấn luyện tiếp làm model kém hơn.
    initial_metadata=dict(previous,steps=base_steps,validation_seeds=validation_seeds,selection_objective=best)
    if not previous:initial_metadata['seed']=args.seed
    online.save(args.output/'best.npz',initial_metadata)
    print('Đánh giá trước học:',initial['mean'],flush=True)
    def metadata(step):return dict(algorithm='Double DQN',seed=args.seed,steps=base_steps+step,episodes=episode,env=asdict(config),train_seed_base=args.seed*100000,validation_seeds=validation_seeds,training_options=dict(epsilon_final=args.epsilon_final,curriculum_probability=args.curriculum_probability,hard_replay_fraction=args.hard_replay_fraction,validation_max_score=args.validation_max_score,learning_rate=args.learning_rate),previous=previous,resume_mode='Weights/Adam restored; replay/target/RNG/environment reset.')
    with log_path.open('w',encoding='utf-8') as log:
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        log.write(json.dumps(dict(event='settings',arguments=settings,base_steps=base_steps))+'\n')
        log.write(json.dumps(dict(event='initial_validation',evaluation=initial))+'\n');log.flush()
        try:
            for step in range(1,args.steps+1):
                epsilon=1.-(1.-args.epsilon_final)*min(1,(base_steps+step)/args.epsilon_decay)
                if rng.random()<epsilon:action=int(rng.choice(np.flatnonzero(info['action_mask'])))
                else:action=int(np.argmax(np.where(info['action_mask'],online.forward(state[None])[0],-1e9)))
                next_state,reward,done,truncated,next_info=env.step(action)
                transition=(state,next_info['executed_action'],reward,next_state,done,next_info['discount'],next_info['action_mask'])
                replay.add(*transition);recent.append(transition)
                if done and next_info.get('failure',{}).get('kind') in ('top_lip','bottom_lip'):
                    for item in recent:hard.add(*item)
                state,info=next_state,next_info
                if step>=args.warmup and replay.size>=args.batch and step%4==0:
                    batch=mixed_batch(replay,hard,args.batch,args.hard_replay_fraction,rng)
                    s,a,r,ns,t,d,m=batch;targets=double_targets(online,target,ns,r,t,d,m)
                    loss=online.fit(s,a,targets,args.learning_rate)
                    if online.updates%1000==0:target.copy_from(online)
                if done or truncated:
                    episodes+=1;score_sum+=info['score'];episode+=1
                    recent.clear();env,state,info=reset_episode(episode)
                if step%5000==0:
                    progress=dict(step=base_steps+step,run_step=step,epsilon=epsilon,loss=loss,episodes=episodes,hard_replay_size=hard.size,train_mean=score_sum/max(1,episodes),steps_per_second=step/(time.perf_counter()-start))
                    log.write(json.dumps(dict(event='progress',**progress))+'\n');log.flush();print(progress,flush=True)
                if step%args.eval_every==0 or step==args.steps:
                    validation=validate()
                    log.write(json.dumps(dict(event='validation',step=base_steps+step,evaluation=validation))+'\n');log.flush()
                    online.save(args.output/'last.npz',metadata(step))
                    # Giữ các ứng viên để không mất model tốt ở một nhóm tình huống.
                    online.save(args.output/f'checkpoint-{base_steps+step}.npz',metadata(step))
                    if validation['objective']>best:
                        best=validation['objective'];online.save(args.output/'best.npz',metadata(step))
                    print('Đánh giá',base_steps+step,validation['mean'],'cao nhất',validation['max'],'objective',validation['objective'],flush=True)
        except KeyboardInterrupt:
            online.save(args.output/'interrupted.npz',metadata(step));print('Đã lưu interrupted.npz; có thể dùng --resume.',flush=True)
    print(log_path,flush=True)

if __name__=='__main__':main()
