import ctypes
import numpy as np
import pytest
from bot.physics import Physics,project
from bot.vision import Bird,Gate,Observation
from bot.controller import Controller
from bot.input_manager import InputManager

def settings(**changes):
    return dict(dict(gravity=687.4,jump=-264.2,pipe_speed=141,freeze_physics=True,pending_actions=True,confirm_input=True),**changes)

def test_ignored_click_does_not_reset_velocity():
    physics=Physics(settings())
    for t in np.arange(0,.10,.02):
        y,v=project(170,150,687.4,t);physics.update(Bird(145,y,52,49),[],t)
    physics.flap(.09)
    for t in np.arange(.10,.30,.02):
        y,v=project(170,150,687.4,t);physics.update(Bird(145,y,52,49),[],t)
        assert physics.v>0
    assert physics.input_events[-1]['status']=='unconfirmed'
    assert physics.last_flap<0

def test_observed_jump_is_confirmed_with_latency_estimate():
    physics=Physics(settings())
    for t in np.arange(0,.10,.02):
        y,v=project(170,150,687.4,t);physics.update(Bird(145,y,52,49),[],t)
    physics.flap(.09);effect=.142
    for t in np.arange(.10,.30,.02):
        y=project(170,150,687.4,t)[0] if t<effect else project(project(170,150,687.4,effect)[0],-264.2,687.4,t-effect)[0]
        physics.update(Bird(145,y,52,49),[],t)
    event=physics.input_events[-1]
    assert event['status']=='confirmed'
    assert event['effect_estimate']==pytest.approx(effect,abs=.006)
    assert physics.v==pytest.approx(-264.2+687.4*(.28-effect),abs=5)

def test_geometry_change_blocks_input(monkeypatch):
    import bot.input_manager as module
    monkeypatch.setattr(ctypes.windll.user32,'GetForegroundWindow',lambda:42)
    monkeypatch.setattr(module,'window_rect',lambda hwnd:(0,0,1920,1080))
    manager=InputManager({'lock_window_geometry':True});manager.bind_foreground()
    assert manager.focused()
    monkeypatch.setattr(module,'window_rect',lambda hwnd:(10,0,1930,1080))
    assert not manager.focused()
    assert manager.stop_reason=='window geometry changed'

def test_forced_actions_keep_both_counterfactuals():
    cfg=dict(settings(),planner='maxmin',use_numba=True,separate_actions=True,latency=.045,margin=0,cooldown=.18,ceiling=0,floor=.89,beam_width=100,search_dt=1/30)
    model=Physics(cfg);obs=Observation(Bird(145,220,52,49),[Gate(150,200,100,260)],np.zeros((400,600,3),np.uint8))
    decision=Controller(cfg).decide(obs,model,1)
    assert decision.coast_margin is not None and decision.flap_margin is not None
    assert decision.plan_margin==pytest.approx(decision.flap_margin if decision.flap else decision.coast_margin)


def test_robust_states_preserve_observer_and_cooldown_state():
    cfg=dict(settings(),planner='maxmin',use_numba=True,latency=.032,margin=0,
             cooldown=.18,ceiling=0,floor=.89,beam_width=100,search_dt=1/30,
             robust_states=[dict(latency_ms=24,velocity_offset=-10,jump_offset=-8),
                            dict(latency_ms=48,velocity_offset=10,jump_offset=8)])
    model=Physics(cfg);model.v=130;model.pending=[1.04]
    controller=Controller(cfg);controller.last_flap=.9
    obs=Observation(Bird(145,180,52,49),[Gate(150,236,100,260)],np.zeros((400,600,3),np.uint8))
    original=(model.v,model.g,model.jump,list(model.pending))
    decision=controller.decide(obs,model,1)
    assert (model.v,model.g,model.jump,model.pending)==original
    assert not decision.flap
    assert controller.last_flap==.9


def test_image_impulse_does_not_reset_velocity_for_ignored_command():
    physics=Physics(dict(settings(),confirm_input=False,image_impulse_reset=True))
    for t in np.arange(0,.10,.02):
        y,_=project(170,150,687.4,t);physics.update(Bird(145,y,52,49),[],t)
    physics.flap(.09)
    for t in np.arange(.10,.30,.02):
        y,_=project(170,150,687.4,t);physics.update(Bird(145,y,52,49),[],t)
        assert physics.v>0
    assert not physics.input_events
