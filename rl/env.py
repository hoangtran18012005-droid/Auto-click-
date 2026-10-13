"""Môi trường vector headless dùng oracle source 120 Hz đã có."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from sim.source_engine import SourceSimulator,sequence

STATE_SIZE=22
SCHEMA='flappy-native-v1-22'

@dataclass
class EnvConfig:
    max_score:int=100
    max_steps:int=10000
    latency:float=.032
    latency_jitter:float=0.
    noise_y:float=0.
    noise_v:float=0.
    frame_ticks:int=4
    frame_drop_probability:float=0.
    cooldown:float=.12
    gamma:float=.995
    shaping:float=.2
    alternating:bool=False

class FlappyEnv:
    def __init__(self,config=None):
        self.config=config or EnvConfig()
        c=self.config
        if (c.frame_ticks<1 or c.max_score<1 or c.max_steps<1 or
            not 0<=c.frame_drop_probability<=1 or not 0<c.gamma<=1 or
            any(not math.isfinite(v) or v<0 for v in (c.latency,c.latency_jitter,c.noise_y,c.noise_v,c.cooldown,c.shaping))):
            raise ValueError('Thông số môi trường không hợp lệ.')
    def reset(self,seed=0):
        self.seed=seed;self.rng=np.random.default_rng(seed+900001)
        self.plan=sequence(seed,self.config.max_score+4,535,self.config.alternating)
        self.sim=SourceSimulator(self.plan[:3],535,self.config.latency)
        self.cursor=3;self.score=0;self.steps=0;self.last_click=-100000
        self.last_ticks=self.config.frame_ticks;self.closed=False
        return self.state(),self.info()
    def mask(self):
        return np.array([True,(self.sim.tick-self.last_click)/120>=self.config.cooldown-1e-9 and not self.sim.pending],dtype=bool)
    def info(self):
        info=dict(score=self.score,ticks=self.sim.tick,seconds=self.sim.tick/120,action_mask=self.mask(),seed=self.seed)
        if self.sim.dead:
            info['failure']=self.failure()
        return info
    def failure(self):
        s=self.sim
        data=dict(sprite_y=s.y,velocity=s.v,tick=s.tick,pending_ticks=list(s.pending))
        if s.y+72>=535:return dict(data,kind='ground')
        if s.y<=-10:return dict(data,kind='ceiling')
        for p in s.pipes:
            x=p['x']-205*s.tick/120;top=p['top'];bottom=p['bottom']
            parts=(('top_body',x,0,92,top),('top_lip',x-16,top-28,124,28),
                   ('bottom_body',x,bottom,92,535-bottom),('bottom_lip',x-16,bottom,124,28))
            for name,bx,by,bw,bh in parts:
                if 191<bx+bw and 241>bx and s.y+9<by+bh and s.y+64>by:
                    return dict(data,kind=name,pipe_x=x,top=top,bottom=bottom)
        return dict(data,kind='unknown')
    def visible(self):
        t=self.sim.tick/120
        return [(p,p['x']-16-205*t) for p in self.sim.pipes if p['x']+108-205*t>191 and p['x']-16-205*t<900][:3]
    def state(self):
        s=self.sim;c=self.config
        y=s.y+36.5+self.rng.normal(0,c.noise_y);v=s.v+self.rng.normal(0,c.noise_v)
        waiting=max(0,min(s.pending)-s.tick)/120 if s.pending else 0.
        base=[y/535,v/700,max(0,c.cooldown-(s.tick-self.last_click)/120)/.2,waiting/.2,self.last_ticks/12,c.latency/.2,y/535,(535-y)/535]
        rows=self.visible()
        for i in range(3):
            if i<len(rows):
                p,x=rows[i];base.extend([(x-216)/900,p['top']/535,p['bottom']/535,1.])
            else:base.extend([0.,0.,0.,0.])
        target=(rows[0][0]['top']+rows[0][0]['bottom'])/2 if rows else 267.5
        base.extend([(y-target)/535,v/700])
        return np.asarray(base,dtype=np.float32)
    def potential(self):
        rows=self.visible();target=(rows[0][0]['top']+rows[0][0]['bottom'])/2 if rows else 267.5
        return -abs(self.sim.y+36.5-target)/535-.05*abs(self.sim.v)/700
    def step(self,action):
        if self.closed:raise RuntimeError('Episode đã kết thúc; gọi reset trước.')
        if action not in (0,1):raise ValueError('Hành động phải là COAST=0 hoặc FLAP=1.')
        before=self.potential();previous_score=self.score
        executed=int(action==1 and self.mask()[1])
        if executed:
            self.last_click=self.sim.tick
            delay=max(0.,self.config.latency+self.rng.normal(0,self.config.latency_jitter))
            self.sim.pending.append(self.sim.tick+math.ceil(delay*120))
        ticks=self.config.frame_ticks*(2 if self.rng.random()<self.config.frame_drop_probability else 1)
        self.last_ticks=ticks
        for _ in range(ticks):
            self.sim.update()
            if self.sim.dead:break
            while self.sim.pipes and self.sim.pipes[0]['x']+108-205*self.sim.tick/120<191:
                self.score+=1;self.sim.pipes.pop(0)
                if self.cursor<len(self.plan):self.sim.pipes.append(self.plan[self.cursor]);self.cursor+=1
        self.steps+=1
        terminated=bool(self.sim.dead)
        truncated=not terminated and (self.steps>=self.config.max_steps or self.score>=self.config.max_score)
        discount=self.config.gamma**(ticks/4)
        after=0. if terminated else self.potential()
        reward=float(self.score-previous_score-2*terminated+self.config.shaping*(discount*after-before))
        self.closed=terminated or truncated
        info=self.info();info.update(executed_action=executed,requested_action=action,discount=discount)
        return self.state(),reward,terminated,truncated,info
