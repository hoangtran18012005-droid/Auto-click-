import numpy as np
from bot.compiled_planner import warmup
from bot.controller import Controller
from bot.physics import Physics
from bot.vision import Bird,Gate,Observation

def test_refinement_recovers_path_before_near_gate_fallback_changes_phase(monkeypatch):
    warmup()
    # Thời gian tính đã đo được được chừa bằng budget; test giữ clock cố định.
    monkeypatch.setattr('bot.controller.time.perf_counter',lambda:0.)
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,cooldown=.18,latency=.032,input_effect_latency=.032,margin=0,floor=.7,world_ceiling_px=0,world_floor_px=356,planner='maxmin',use_numba=True,pending_actions=True,beam_width=200,bin_y=2,bin_v=8)
    p=Physics(cfg);p.v=107.821644204
    o=Observation(Bird(147.5,161.5,51,47),[Gate(145,231,84,204),Gate(338,424,62,181),Gate(532,600,160,275)],np.zeros((495,600,3),np.uint8))
    t=209.9532773;last=209.4025463
    old=Controller(cfg);old.last_flap=last
    assert old.decide(o,p,t,.0046356).plan_margin<0
    c=Controller(dict(cfg,refine_failed_plan=True,near_term_fallback=True));c.last_flap=last
    d=c.decide(o,p,t,.0046356)
    assert not d.flap and d.feasible and d.plan_margin>0
    assert d.long_plan_margin<0 and d.reason=='MAXMIN_REFINED_200 COAST'
    assert c.last_flap==last
