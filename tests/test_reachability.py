import numpy as np
from bot.reachability import warmup,interpolate,segment_margin
from bot.controller import Controller
from bot.physics import Physics
from bot.vision import Bird,Gate,Observation

def test_interpolation_and_continuous_apex_collision():
    warmup()
    page=np.array([[0,2],[2,4]],np.float32)
    assert interpolate(page,.5,.5,0.,1.,0.,1.)==2
    gates=np.array([[100.,200.,100.,250.]])
    assert segment_margin(150.,-200.,1000.,.4,gates,145.,26.,24.,141.,0.,356.,0.)==6

def test_backward_grid_keeps_route_lost_by_joint_beam():
    warmup()
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,cooldown=.18,latency=.032,input_effect_latency=.032,margin=0,floor=.7,world_ceiling_px=0,world_floor_px=356,planner='reachability',use_numba=True,pending_actions=True,atlas_synchronous=True,atlas_dy=1.,atlas_dv=4.)
    p=Physics(cfg);p.v=107.821644204
    o=Observation(Bird(147.5,161.5,52,47),[Gate(145,231,84,204),Gate(338,424,62,181),Gate(532,600,160,275)],np.zeros((495,600,3),np.uint8))
    c=Controller(cfg);c.last_flap=209.4025463
    try:
        d=c.decide(o,p,209.9532773,.0046356)
        assert d.reason=='REACHABILITY COAST' and not d.flap
        assert d.coast_margin>4 and d.feasible
        assert c.last_flap==209.4025463
    finally:c.close()
