import numpy as np
import pytest
from bot.controller import Controller
from bot.physics import Physics,project
from bot.maxmin_planner import clearance
from bot.vision import Bird,Gate,Observation

def settings(**changes):
    return dict(dict(planner='maxmin',gravity=687.4,jump=-264.2,pipe_speed=141.,latency=.045,margin=0,cooldown=.18,floor=.89,ceiling=0,beam_width=200,bin_y=2,bin_v=8,search_dt=1/30),**changes)

def test_continuous_clearance_detects_apex():
    value=clearance(np.array([100.]),np.array([-100.]),100.,0,2,[],145,26,20,141,40,400)
    assert value[0]==pytest.approx(-10)

def test_pending_jump_does_not_set_velocity_early():
    cfg=settings(pending_actions=True);physics=Physics(cfg);physics.v=100
    physics.flap(1)
    assert physics.v==100
    y,v=physics.predict(200,1,.1)
    expected=project(project(200,100,687.4,.045)[0],-264.2,687.4,.055)
    assert (y,v)==pytest.approx(expected)

def test_frozen_physics_keeps_parameters():
    physics=Physics(settings(freeze_physics=True));original=physics.g,physics.jump,physics.speed
    for index in range(30):
        physics.update(Bird(145,200+index,52,49),[Gate(200-index*5,286-index*5,100,220)],index/30)
    assert (physics.g,physics.jump,physics.speed)==original

def test_maxmin_reports_negative_clearance_instead_of_center_fallback():
    cfg=settings();obs=Observation(Bird(145,80,52,49),[Gate(140,160,150,270)],np.zeros((400,600,3),np.uint8))
    decision=Controller(cfg).decide(obs,Physics(cfg),1)
    assert decision.plan_margin<0
    assert not decision.feasible
    assert decision.reason.startswith('MAXMIN')

def test_compiled_and_numpy_agree_without_pruning():
    pytest.importorskip('numba')
    cfg=settings(beam_width=20000,latency=0)
    obs=Observation(Bird(145,200,52,49),[Gate(140,145,100,270)],np.zeros((400,600,3),np.uint8))
    a=Controller(cfg).decide(obs,Physics(cfg),1)
    cfg=dict(cfg,use_numba=True)
    b=Controller(cfg).decide(obs,Physics(cfg),1)
    assert b.plan_margin==pytest.approx(a.plan_margin)
    assert b.flap==a.flap

def test_simulation_parameter_noise_is_json_serializable():
    import json
    from sim.engine import SimulationConfig,run_episode,random_sequence
    result=run_episode(Controller(settings()),random_sequence(42,count=2),SimulationConfig(parameter_error=.04),42)
    assert isinstance(result['score'],int)
    json.dumps(result)

def test_score_ocr_handles_zero_without_reading_record_score():
    from tools.run_series import decode_score
    assert decode_score('O diém')==0
    assert decode_score('37 điểm')==37
    assert decode_score('Kỷ lục 39 điểm') is None
