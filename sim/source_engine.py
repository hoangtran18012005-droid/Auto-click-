"""Oracle JS độc lập: native 900×600, fixed update 120 Hz, rectHit strict."""
import math
import numpy as np
from bot.vision import Bird,Gate,Observation

class SourceSimulator:
    def __init__(self,pipes,ground_y,latency=.032,seed=0,noise=0.):
        self.pipes=[dict(p) for p in pipes];self.ground_y=ground_y
        self.y=280.;self.v=0.;self.tick=0;self.accumulator=0.;self.paused=False;self.dead=False
        self.pending=[];self.latency=latency;self.rng=np.random.default_rng(seed);self.noise=noise

    def flap(self):
        self.pending.append(self.tick+math.ceil(self.latency*120))

    def update(self):
        if self.dead or self.paused:return
        if any(t<=self.tick for t in self.pending):self.v=-385.
        self.pending=[t for t in self.pending if t>self.tick]
        self.v+=1080/120;self.y+=self.v/120
        self.tick+=1
        if self.y+72>=self.ground_y or self.y<=-10:self.dead=True;return
        a=(191.,self.y+9,50.,55.)
        for p in self.pipes:
            x=p['x']-205*self.tick/120;top=p['top'];bottom=p['bottom']
            for b in ((x,0,92,top),(x-16,top-28,124,28),(x,bottom,92,self.ground_y-bottom),(x-16,bottom,124,28)):
                if a[0]<b[0]+b[2] and a[0]+a[2]>b[0] and a[1]<b[1]+b[3] and a[1]+a[3]>b[1]:
                    self.dead=True;return

    def elapsed(self,dt):
        if dt>.25:self.paused=True;return
        self.accumulator+=max(0,dt)
        while self.accumulator>=1/120-1e-12 and not self.dead and not self.paused:
            self.update();self.accumulator-=1/120

    @property
    def score(self):
        return sum(p['x']+92+16-205*self.tick/120<191 for p in self.pipes)

    def observe(self):
        s=2/3;t=self.tick/120
        gates=[Gate((p['x']-16-205*t)*s,(p['x']+108-205*t)*s,p['top']*s,p['bottom']*s)
               for p in self.pipes if p['x']+108-205*t>170 and p['x']-16-205*t<900]
        return Observation(Bird(216*s,(self.y+36)*s+self.rng.normal(0,self.noise),72*s,72*s),gates,np.zeros((400,600,3),np.uint8))

def sequence(seed,count=12,ground_y=535,alternating=False):
    rng=np.random.default_rng(seed);centers=[]
    for i in range(count):
        centers.append((180 if i%2==0 else ground_y-180-1e-6) if alternating else float(rng.uniform(180,ground_y-180)))
    return [dict(x=920+205*1.42*i,top=c-90,bottom=c+90) for i,c in enumerate(centers)]
