"""Beam search clearance liên tục; giữ kế hoạch tốt nhất cả khi margin âm."""
import math
import numpy as np
from .physics import project
from .collision_geometry import body_extents,body_center_offset_y,world_clearance_bounds


def extrema(y,v,g,start,end):
    a,_=project(y,v,g,start);b,_=project(y,v,g,end)
    apex=np.clip(-v/g,start,end);c,_=project(y,v,g,apex)
    return np.minimum(np.minimum(a,b),c),np.maximum(np.maximum(a,b),c)


def clearance(y,v,g,start,end,gates,x,hw,hh,speed,ceiling,floor,offset=0.):
    low,high=extrema(y,v,g,start,end)
    result=np.minimum(low-hh-ceiling,floor-high-hh)
    for gate in gates:
        entry=max(start,(gate.x-x-hw)/speed-offset)
        exit=min(end,(gate.right-x+hw)/speed-offset)
        if entry<=exit:
            low,high=extrema(y,v,g,entry,exit)
            result=np.minimum(result,np.minimum(low-hh-gate.top,gate.bottom-high-hh))
    return result


def plan(controller,obs,physics,now,age):
    cases=controller.cfg.get('robust_states')
    if not cases:
        cases=[{'latency_ms':value} for value in (controller.cfg.get('robust_latency_ms') or [])]
    if not cases:return _plan_once(controller,obs,physics,now,age)
    if not controller.cfg.get('use_numba',False):
        raise ValueError('Các nhánh robust yêu cầu use_numba để đánh giá riêng hai hành động.')
    from copy import copy
    from types import SimpleNamespace
    from .controller import Decision
    decisions=[]
    for case in cases:
        milliseconds=case.get('latency_ms',controller.cfg['latency']*1000)
        cfg=dict(controller.cfg,robust_latency_ms=None,separate_actions=True,latency=milliseconds/1000)
        branch=SimpleNamespace(cfg=cfg,last_flap=controller.last_flap)
        model=copy(physics)
        model.v+=case.get('velocity_offset',0)
        model.jump+=case.get('jump_offset',0)
        model.g*=case.get('gravity_scale',1)
        offset=cfg['latency']-controller.cfg.get('input_effect_latency',controller.cfg['latency'])
        model.pending=[event+offset for event in getattr(physics,'pending',[]) if event+offset>now]
        decisions.append(_plan_once(branch,obs,model,now,age+controller.cfg.get('planner_budget_ms',0)/1000))
    coast=min(d.coast_margin for d in decisions);flap=min(d.flap_margin for d in decisions)
    action=flap>coast+controller.cfg.get('action_tolerance',.1)
    margin=flap if action else coast
    if action:controller.last_flap=now
    return Decision(action,[],decisions[0].target,'ROBUST_LATENCY '+('CLICK' if action else 'COAST'),margin>=0,margin,coast,flap,max(d.planner_beam for d in decisions))

def _plan_once(controller,obs,physics,now,age):
    from .controller import Decision
    cfg=controller.cfg;b=obs.bird;height=obs.frame.shape[0]
    hw,hh=body_extents(cfg,b)
    gates=[gate for gate in obs.gates if gate.right>b.x-hw][:cfg.get('horizon_gates',3)]
    ceiling=cfg.get('world_ceiling_px',height*cfg.get('ceiling',0))
    floor=cfg.get('world_floor_px',height*cfg['floor'])
    ceiling,floor=world_clearance_bounds(cfg,hh,ceiling,floor)
    latency=cfg['latency']+min(age,.15)
    if cfg.get('latency_pipeline',False):
        latency+=(cfg.get('input_call_ms',0)+cfg.get('planner_budget_ms',0))/1000
    y,v=b.y+body_center_offset_y(cfg),physics.v
    prefix=clearance(np.array([y]),np.array([v]),physics.g,0,latency,gates,b.x,hw,hh,physics.speed,ceiling,floor)
    if cfg.get('pending_actions',False) and hasattr(physics,'predict'):
        prefix=np.array([float('inf')])
        for start,end,py,pv in physics.prediction_segments(b.y,now,latency):
            prefix=np.minimum(prefix,clearance(np.array([py]),np.array([pv]),physics.g,0,end-start,gates,b.x,hw,hh,physics.speed,ceiling,floor,start))
        y,v=physics.predict(b.y,now,latency)
    else:y,v=project(y,v,physics.g,latency)
    dt=cfg.get('search_dt',1/30)
    first_dt=max(dt,cfg.get('reaction_guard_ms',0)/1000)
    horizon=max(dt,(gates[-1].right-b.x+hw)/physics.speed-latency) if gates else .8
    horizon=min(horizon,cfg.get('max_horizon',4.5))
    steps=math.ceil(horizon/dt) if first_dt==dt else 1+math.ceil(max(0,horizon-first_dt)/dt)
    if cfg.get('use_numba',False):
        from .compiled_planner import kernel
        array=np.array([[gate.x,gate.right,gate.top,gate.bottom] for gate in gates],dtype=float).reshape(-1,4)
        beam=cfg.get('beam_width',400)
        arguments=(float(y),float(v),float(physics.g),float(physics.jump),float(physics.speed),array,float(b.x),float(hw),float(hh),float(ceiling),float(floor),float(prefix[0]),float(latency),float(dt),steps,float(cfg['cooldown']),float(now-controller.last_flap))
        coast_margin=flap_margin=None
        if cfg.get('separate_actions',False):
            def alternatives(width):
                return [float(kernel(*arguments,width,float(cfg.get('bin_y',2)),float(cfg.get('bin_v',8)),initial,cfg.get('rank_suffix',False),first_dt)[1]) for initial in (0,1)]
            coast_margin,flap_margin=alternatives(beam)
            if cfg.get('adaptive_beam',False) and prefix[0]>=0 and max(coast_margin,flap_margin)<cfg.get('rescue_threshold',0):
                beam=cfg.get('rescue_beam',2000)
                coast_margin,flap_margin=alternatives(beam)
            action=flap_margin>coast_margin+cfg.get('action_tolerance',.1)
            margin=flap_margin if action else coast_margin
        else:
            action,margin=kernel(*arguments,beam,float(cfg.get('bin_y',2)),float(cfg.get('bin_v',8)),-1,cfg.get('rank_suffix',False),first_dt)
        action=bool(action);margin=float(margin)
        if action:controller.last_flap=now
        target=(gates[0].top+gates[0].bottom)/2 if gates else height*.44
        return Decision(action,[],target,'MAXMIN_NUMBA '+('CLICK' if action else 'COAST'),margin>=0,margin,coast_margin,flap_margin,beam,float(prefix[0]))
    ys=np.array([y]);vs=np.array([v]);margins=np.array([float('inf')]) if cfg.get('rank_suffix',False) else prefix
    since=np.array([now-controller.last_flap]);first=np.array([False])
    for step in range(steps):
        duration=first_dt if step==0 else dt
        offset=latency if step==0 else latency+first_dt+(step-1)*dt
        parents=np.repeat(np.arange(len(ys)),2);actions=np.tile([False,True],len(ys))
        valid=~actions | (since[parents]>=cfg['cooldown']-1e-9)
        parents=parents[valid];actions=actions[valid]
        velocity=np.where(actions,physics.jump,vs[parents])
        current=clearance(ys[parents],velocity,physics.g,0,duration,gates,b.x,hw,hh,physics.speed,ceiling,floor,offset)
        scores=np.minimum(margins[parents],current)
        ny,nv=project(ys[parents],velocity,physics.g,duration)
        elapsed=np.minimum(cfg['cooldown'],np.where(actions,duration,since[parents]+duration))
        initial=actions if step==0 else first[parents]
        keys=np.column_stack((np.rint(ny/cfg.get('bin_y',2)),np.rint(nv/cfg.get('bin_v',8)),np.rint(elapsed/dt),initial)).astype(np.int64)
        # Với trạng thái tương đương, giữ clearance tốt nhất; sau đó giới hạn beam.
        order=np.lexsort((-scores,keys[:,3],keys[:,2],keys[:,1],keys[:,0]))
        sorted_keys=keys[order];unique=np.r_[True,np.any(sorted_keys[1:]!=sorted_keys[:-1],axis=1)]
        selected=order[unique]
        selected=selected[np.argsort(-scores[selected],kind='stable')[:cfg.get('beam_width',400)]]
        ys,vs,margins,since,first=ny[selected],nv[selected],scores[selected],elapsed[selected],initial[selected]
    index=int(np.argmax(margins));action=bool(first[index]);margin=min(float(prefix[0]),float(margins[index]))
    if action:controller.last_flap=now
    target=(gates[0].top+gates[0].bottom)/2 if gates else height*.44
    return Decision(action,[],target,'MAXMIN '+('CLICK' if action else 'COAST'),margin>=0,margin,prefix_margin=float(prefix[0]))
