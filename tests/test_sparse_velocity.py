import pytest
from bot.physics import Physics,project
from bot.vision import Bird

def test_gravity_advances_velocity_before_coast_fit_has_enough_time():
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,pending_actions=True,input_effect_latency=.032,freeze_physics=True,advance_sparse_velocity=True)
    physics=Physics(cfg);physics.flap(1.)
    for elapsed in (.013,.030,.046):
        y,v=project(185,cfg['jump'],cfg['gravity'],elapsed)
        physics.update(Bird(148,y,52,49),[],1.032+elapsed)
        assert physics.v==pytest.approx(v)
