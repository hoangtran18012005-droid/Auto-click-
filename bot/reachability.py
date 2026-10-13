"""Quy hoạch động ngược trên lưới y/v, cache theo các khe đã quan sát.

Nội suy lưới là xấp xỉ; không gọi kết quả là chứng minh tối ưu liên tục.
Kernel nền nhả GIL; vòng điều khiển chỉ tra bảng và kiểm tra đoạn gần.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math,time
import numpy as np
from numba import njit
from .collision_geometry import body_extents,body_center_offset_y,world_clearance_bounds
from .compiled_planner import bounds
from .controller import Decision
from .maxmin_planner import clearance
from .physics import project

@njit(cache=True,nogil=True)
def interpolate(page,y,v,y0,dy,v0,dv):
    fy=(y-y0)/dy;fv=(v-v0)/dv
    if fy<0 or fv<0 or fy>page.shape[0]-1 or fv>page.shape[1]-1:return -1e5
    iy=min(page.shape[0]-2,int(fy));iv=min(page.shape[1]-2,int(fv))
    ay=fy-iy;av=fv-iv
    return (1-ay)*((1-av)*page[iy,iv]+av*page[iy,iv+1])+ay*((1-av)*page[iy+1,iv]+av*page[iy+1,iv+1])

@njit(cache=True,nogil=True)
def segment_margin(y,v,g,duration,gates,x,hw,hh,speed,ceiling,floor,offset):
    low,high=bounds(y,v,g,0.,duration)
    result=min(low-hh-ceiling,floor-high-hh)
    for gate in gates:
        entry=max(0.,(gate[0]-x-hw)/speed-offset)
        exit=min(duration,(gate[1]-x+hw)/speed-offset)
        if entry<=exit:
            low,high=bounds(y,v,g,entry,exit)
            result=min(result,low-hh-gate[2],gate[3]-high-hh)
    return result

@njit(cache=True,nogil=True)
def build_values(g,jump,speed,gates,x,hw,hh,ceiling,floor,dt,steps,cooldown,y0,dy,ny,v0,dv,nv):
    values=np.empty((steps+1,ny,nv),np.float32)
    for iy in range(ny):
        y=y0+iy*dy
        for iv in range(nv):values[steps,iy,iv]=min(y-hh-ceiling,floor-y-hh)
    for step in range(steps-1,-1,-1):
        offset=step*dt
        for iy in range(ny):
            y=y0+iy*dy
            for iv in range(nv):
                v=v0+iv*dv
                cy=y+v*dt+.5*g*dt*dt;cv=v+g*dt
                coast=min(segment_margin(y,v,g,dt,gates,x,hw,hh,speed,ceiling,floor,offset),interpolate(values[step+1],cy,cv,y0,dy,v0,dv))
                best=coast
                # Với g/J cố định, vận tốc cho biết tuổi của lần nhảy gần nhất.
                if v-jump>=g*cooldown-1e-6:
                    fy=y+jump*dt+.5*g*dt*dt;fv=jump+g*dt
                    flap=min(segment_margin(y,jump,g,dt,gates,x,hw,hh,speed,ceiling,floor,offset),interpolate(values[step+1],fy,fv,y0,dy,v0,dv))
                    best=max(best,flap)
                values[step,iy,iv]=best
    return values

def warmup():
    build_values(710.38,-248.58,139.6,np.empty((0,4)),147.,26.,20.,0.,356.,1/30,2,.18,0.,2.,4,-300.,8.,4)
    interpolate(np.zeros((4,4),np.float32),1.,-280.,0.,2.,-300.,8.)
    segment_margin(180.,0.,710.38,1/30,np.empty((0,4)),147.,26.,20.,139.6,0.,356.,0.)

@dataclass
class Atlas:
    key: tuple
    origin: float
    gates: np.ndarray
    x: float
    hw: float
    hh: float
    gravity: float
    jump: float
    speed: float
    ceiling: float
    floor: float
    dt: float
    y0: float
    dy: float
    v0: float
    dv: float
    values: np.ndarray
    build_ms: float

def build_atlas(key,origin,gates,x,hw,hh,g,jump,speed,ceiling,floor,cfg):
    started=time.perf_counter();dt=float(cfg.get('atlas_dt',1/30))
    x,hw,hh,g,jump,speed,ceiling,floor=map(float,(x,hw,hh,g,jump,speed,ceiling,floor))
    horizon=max(.8,max((row[1]-x+hw)/speed for row in gates)) if len(gates) else .8
    steps=max(1,math.ceil(min(horizon,cfg.get('max_horizon',4.5))/dt))
    y0=ceiling;dy=float(cfg.get('atlas_dy',1.));v0=float(cfg.get('atlas_vmin',-300.));dv=float(cfg.get('atlas_dv',4.))
    ny=math.ceil((floor-ceiling)/dy)+1;nv=math.ceil((cfg.get('atlas_vmax',650.)-v0)/dv)+1
    values=build_values(float(g),float(jump),float(speed),gates,float(x),float(hw),float(hh),float(ceiling),float(floor),float(dt),steps,float(cfg['cooldown']),float(y0),float(dy),ny,float(v0),float(dv),nv)
    return Atlas(key,origin,gates,x,hw,hh,g,jump,speed,ceiling,floor,dt,y0,dy,v0,dv,values,1000*(time.perf_counter()-started))

class Reachability:
    def __init__(self):
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='reachability')
        self.future=None;self.cache=None;self.requested=None;self.error=None
    def close(self):self.executor.shutdown(wait=False,cancel_futures=True)
    def request(self,obs,physics,now,cfg):
        if self.future is not None and self.future.done():
            try:self.cache=self.future.result()
            except Exception as error:self.error=f'{type(error).__name__}: {error}'
            self.future=None
        hw,hh=body_extents(cfg,obs.bird)
        gates=[g for g in obs.gates if g.right>obs.bird.x-hw][:cfg.get('horizon_gates',3)]
        model=(round(physics.g,2),round(physics.jump,2),round(physics.speed,2),hw,hh)
        signature=tuple((g.top,g.bottom) for g in gates)
        anchor=round((gates[-1].x+physics.speed*now)/32) if gates else None
        key=(model,signature,anchor)
        if self.future is None and (self.cache is None or self.cache.key!=key):
            array=np.array([[g.x,g.right,g.top,g.bottom] for g in gates],dtype=float).reshape(-1,4)
            ceiling=cfg.get('world_ceiling_px',obs.frame.shape[0]*cfg.get('ceiling',0))
            floor=cfg.get('world_floor_px',obs.frame.shape[0]*cfg['floor'])
            ceiling,floor=world_clearance_bounds(cfg,hh,ceiling,floor)
            self.requested=key
            arguments=(key,now,array,obs.bird.x,hw,hh,physics.g,physics.jump,physics.speed,ceiling,floor,dict(cfg))
            if cfg.get('atlas_synchronous',False):
                self.cache=build_atlas(*arguments)
            else:self.future=self.executor.submit(build_atlas,*arguments)
        return key
    def phase(self,obs,now):
        atlas=self.cache
        if atlas is None:return None
        age=now-atlas.origin;matches=[]
        for gate in obs.gates:
            for saved in atlas.gates:
                if abs(gate.top-saved[2])>1 or abs(gate.bottom-saved[3])>1:continue
                phase=(saved[0]-gate.x+obs.bird.x-atlas.x)/atlas.speed
                if abs(phase-age)<.15:matches.append(phase)
        return float(np.median(matches)) if matches else (age if not len(atlas.gates) and not obs.gates else None)

def _lookup(atlas,phase,y,v):
    step=phase/atlas.dt
    if step<0 or step>atlas.values.shape[0]-1:return -1e5
    lower=min(atlas.values.shape[0]-2,int(step));weight=step-lower
    a=interpolate(atlas.values[lower],y,v,atlas.y0,atlas.dy,atlas.v0,atlas.dv)
    b=interpolate(atlas.values[lower+1],y,v,atlas.y0,atlas.dy,atlas.v0,atlas.dv)
    return float((1-weight)*a+weight*b)

def plan(controller,obs,physics,now,age):
    if not hasattr(controller,'reachability'):controller.reachability=Reachability()
    engine=controller.reachability;key=engine.request(obs,physics,now,controller.cfg)
    atlas=engine.cache;phase=engine.phase(obs,now)
    if atlas is None or phase is None or atlas.key[0]!=key[0] or phase+controller.cfg['latency']+age+atlas.dt>(atlas.values.shape[0]-1)*atlas.dt:
        from .maxmin_planner import plan as fallback
        decision=fallback(controller,obs,physics,now,age)
        decision.reason=('REACHABILITY_ERROR ' if engine.error else 'REACHABILITY_WARMUP ')+decision.reason
        return decision
    cfg=controller.cfg;b=obs.bird
    latency=cfg['latency']+age+(cfg.get('planner_budget_ms',1)+cfg.get('input_call_ms',1))/1000
    gates=[g for g in obs.gates if g.right>b.x-atlas.hw]
    prefix=float('inf')
    initial_y=b.y+body_center_offset_y(cfg)
    for start,end,py,pv in physics.prediction_segments(initial_y,now,latency):
        prefix=min(prefix,float(clearance(np.array([py]),np.array([pv]),physics.g,0,end-start,gates,b.x,atlas.hw,atlas.hh,physics.speed,atlas.ceiling,atlas.floor,start)[0]))
    y,v=physics.predict(initial_y,now,latency)
    duration=max(atlas.dt,cfg.get('reaction_guard_ms',0)/1000)
    margins=[]
    for action in (False,True):
        if action and now-controller.last_flap<cfg['cooldown']:
            margins.append(-1e5);continue
        velocity=physics.jump if action else v
        current=segment_margin(y,velocity,physics.g,duration,atlas.gates,atlas.x,atlas.hw,atlas.hh,physics.speed,atlas.ceiling,atlas.floor,phase+latency)
        ny,nv=project(y,velocity,physics.g,duration)
        margins.append(min(prefix,float(current),_lookup(atlas,phase+latency+duration,ny,nv)))
    coast,flap=margins;action=flap>coast+cfg.get('action_tolerance',.1)
    if action:controller.last_flap=now
    margin=flap if action else coast
    return Decision(action,[],(gates[0].top+gates[0].bottom)/2 if gates else 180,'REACHABILITY '+('CLICK' if action else 'COAST'),margin>=0,margin,coast,flap,None,prefix)
