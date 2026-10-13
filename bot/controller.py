from dataclasses import dataclass
import math
import time
import numpy as np
from .physics import project

@dataclass
class Decision:
    flap: bool
    path: list
    target: float
    reason: str
    feasible: bool = True
    plan_margin: float | None = None
    coast_margin: float | None = None
    flap_margin: float | None = None
    planner_beam: int | None = None
    prefix_margin: float | None = None
    long_plan_margin: float | None = None

class Controller:
    """Receding-horizon beam search with collision checks at future gate positions."""
    def __init__(self, cfg):
        self.cfg=cfg
        self.last_flap=-1e9

    def close(self):
        if hasattr(self,'reachability'):self.reachability.close()

    def decide(self, obs, physics, now, age=0.):
        b=obs.bird
        if b is None or obs.ended:
            return Decision(False,[],0,'NO INPUT: detection/end')
        if self.cfg.get('planner')=='source_euler':
            from .source_planner import plan
            return plan(self,obs,physics,now,age)
        if self.cfg.get('planner') in ('maxmin','reachability'):
            if self.cfg.get('planner')=='reachability':
                from .reachability import plan
            else:
                from .maxmin_planner import plan
            previous_flap=self.last_flap
            planning_started=time.perf_counter()
            decision=plan(self,obs,physics,now,age)
            if self.cfg.get('planner')=='maxmin' and self.cfg.get('refine_failed_plan',False) and decision.plan_margin is not None and decision.plan_margin<self.cfg.get('refine_trigger_margin',0) and (decision.prefix_margin is None or decision.prefix_margin>=0):
                from types import SimpleNamespace
                original_margin=decision.plan_margin
                widths=self.cfg.get('refine_beams',[200,600])
                budgets=self.cfg.get('refine_budgets_ms',[5,15])
                for stage,width in enumerate(widths):
                    settings=dict(self.cfg,refine_failed_plan=False,near_term_fallback=False,adaptive_beam=False,separate_actions=True,beam_width=width)
                    branch=SimpleNamespace(cfg=settings,last_flap=previous_flap)
                    elapsed=time.perf_counter()-planning_started
                    budget=budgets[min(stage,len(budgets)-1)]/1000
                    refined=plan(branch,obs,physics,now,age+elapsed+budget)
                    refined.reason=f'MAXMIN_REFINED_{width} '+('CLICK' if refined.flap else 'COAST')
                    refined.long_plan_margin=original_margin
                    decision=refined;self.last_flap=branch.last_flap
                    if decision.plan_margin>=self.cfg.get('refine_trigger_margin',0) or decision.prefix_margin is not None and decision.prefix_margin<0:break
            if self.cfg.get('near_term_fallback',False) and decision.plan_margin is not None and decision.plan_margin<0:
                from types import SimpleNamespace
                from .maxmin_planner import plan as plan_short
                settings=dict(self.cfg,planner='maxmin',near_term_fallback=False,robust_states=None,
                              robust_latency_ms=None,horizon_gates=1,separate_actions=True)
                branch=SimpleNamespace(cfg=settings,last_flap=previous_flap)
                elapsed=time.perf_counter()-planning_started
                short=plan_short(branch,obs,physics,now,age+elapsed+self.cfg.get('near_term_budget_ms',2)/1000)
                if short.plan_margin>=self.cfg.get('near_term_min_margin',2) and short.flap!=decision.flap:
                    veto=False
                    if short.flap and (self.cfg.get('near_term_visible_gate_veto',False)
                                       or self.cfg.get('near_term_two_gate_veto',False)):
                        from .collision_geometry import body_extents
                        rescue_hw,_=body_extents(self.cfg,obs.bird)
                        visible=sum(g.right>obs.bird.x-rescue_hw for g in obs.gates)
                        horizon=max(2,min(visible,self.cfg.get('horizon_gates',3)))
                        full_cfg=dict(settings,horizon_gates=horizon,
                                      beam_width=self.cfg.get('near_term_visible_gate_beam',
                                                              self.cfg.get('near_term_two_gate_beam',400)))
                        full_branch=SimpleNamespace(cfg=full_cfg,last_flap=previous_flap)
                        full=plan_short(full_branch,obs,physics,now,
                                       age+time.perf_counter()-planning_started)
                        veto=(full.coast_margin is not None and full.flap_margin is not None
                              and full.coast_margin>=self.cfg.get('near_term_visible_gate_reserve_px',
                                                                  self.cfg.get('near_term_two_gate_reserve_px',-1))
                              and full.coast_margin>full.flap_margin)
                    if not veto:
                        short.reason='NEAR_TERM '+('CLICK' if short.flap else 'COAST')
                        short.long_plan_margin=decision.plan_margin
                        short.feasible=decision.feasible
                        decision=short
                        self.last_flap=now if short.flap else previous_flap
            if self.cfg.get('safety_shield',False):
                from .safety_shield import protect
                shield_age=age+(time.perf_counter()-planning_started if self.cfg.get('refine_failed_plan',False) else 0)
                return protect(self,decision,obs,physics,now,shield_age,previous_flap)
            return decision
        h=obs.frame.shape[0]
        margin=self.cfg['margin']
        radius=b.h/2+margin
        gates=[g for g in obs.gates if g.right>b.x-b.w/2]
        target=(gates[0].top+gates[0].bottom)/2 if gates else h*.44
        latency=self.cfg['latency']+min(age,.15)
        y,v=project(b.y,physics.v,physics.g,latency)
        step_dt=max(self.cfg.get('search_dt',.065),self.cfg.get('reaction_interval',.065))
        # cost, y, velocity, time since jump, first action, trajectory
        costs=np.array([0.]);ys=np.array([y]);vs=np.array([v])
        elapsed=np.array([now-self.last_flap]);first=np.array([False])
        history=[]
        for step in range(min(self.cfg.get('max_steps',64),math.ceil(self.cfg.get('horizon',3.6)/step_dt))):
            dt=step_dt
            t=(step+1)*dt+latency
            parents=np.repeat(np.arange(len(ys)),2)
            flaps=np.tile([False,True],len(ys))
            valid=~flaps | (elapsed[parents]>=self.cfg['cooldown'])
            parents=parents[valid];flaps=flaps[valid]
            velocity=np.where(flaps,physics.jump,vs[parents])
            ny,nv=project(ys[parents],velocity,physics.g,dt)
            since=np.where(flaps,dt,elapsed[parents]+dt)
            actions=flaps if step==0 else first[parents]
            penalty=((ny-radius<h*self.cfg.get('ceiling',0)) | (ny+radius>h*self.cfg['floor'])).astype(float)
            for gate in gates:
                if gate.x-physics.speed*t<=b.x+b.w/2+margin and gate.right-physics.speed*t>=b.x-b.w/2-margin:
                    penalty+=((ny-radius<gate.top) | (ny+radius>gate.bottom))
            future=next((g for g in gates if g.right-physics.speed*t>b.x-b.w/2-margin),None)
            future_target=(future.top+future.bottom)/2 if future else h*.44
            arrival=max(0,future.x-physics.speed*t-b.x)/max(physics.speed,1) if future else 0
            weight=.02 if future and arrival>.5 else .15
            scores=costs[parents]+penalty*1e6/(1+t)**2+weight*((ny-future_target)/h)**2+flaps*.002
            # Keep distinct trajectories. Plain top-N pruning retained many
            # nearly identical paths and lost transitions between high/low gaps.
            selected=[];bins=set()
            keys=list(zip(np.rint(ny/self.cfg.get('bin_y',8)).astype(int),np.rint(nv/self.cfg.get('bin_v',35)).astype(int),np.rint(since/self.cfg.get('bin_cooldown',.1)).astype(int),actions))
            for index in np.argsort(scores,kind='stable'):
                key=keys[index]
                if key in bins:continue
                bins.add(key);selected.append(index)
                if len(selected)>=self.cfg.get('beam_width',200):break
            selected=np.asarray(selected)
            costs=scores[selected];ys=ny[selected];vs=nv[selected]
            elapsed=since[selected];first=actions[selected]
            history.append((parents[selected],ys.copy(),b.x+physics.speed*t))
        index=int(np.argmin(costs));feasible=bool(costs[index]<1000);action=bool(first[index]);path=[]
        for parents,positions,x in reversed(history):
            path.append((x,float(positions[index])));index=int(parents[index])
        path.reverse()
        # Do not trade an imminent trailing-edge collision for a cheaper
        # distant path. Evaluate both inputs over the next measured cycle.
        check_dt=max(.06,min(.12,self.cfg.get('reaction_interval',.065)))
        def imminent(flap):
            for fraction in (.5,1.):
                dt=check_dt*fraction
                cy,_=project(y,physics.jump if flap else v,physics.g,dt)
                if cy-radius<h*self.cfg.get('ceiling',0) or cy+radius>h*self.cfg['floor']:
                    return True
                for gate in gates:
                    if gate.x-physics.speed*(latency+dt)<=b.x+b.w/2+margin and gate.right-physics.speed*(latency+dt)>=b.x-b.w/2-margin:
                        if cy-radius<gate.top or cy+radius>gate.bottom:return True
            return False
        if now-self.last_flap>=self.cfg['cooldown'] and imminent(False) and not imminent(True):
            if not action:path=[]
            action=True
        if action:
            self.last_flap=now
        mode=self.cfg.get('input_mode','space').upper()
        return Decision(action,path,target,(mode if action else 'COAST')+(' | no safe horizon' if not feasible else ''),feasible)
