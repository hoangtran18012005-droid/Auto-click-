from types import SimpleNamespace
import numpy as np
import pytest
import pytest
from bot.controller import Controller, Decision
from bot.physics import Physics
from bot.safety_shield import protect
from bot.vision import Bird, Gate, Observation


def setup_state(y=167.5, velocity=181.4):
    cfg = dict(gravity=687.4, jump=-264.2, pipe_speed=141, latency=.045,
               cooldown=.18, margin=0, world_ceiling_px=0, world_floor_px=356, floor=.7)
    controller = Controller(cfg)
    physics = Physics(cfg); physics.v = velocity
    obs = Observation(Bird(148,y,52,46), [Gate(62,148,93,212)], np.zeros((495,600,3),np.uint8))
    decision = Decision(False, [], 152.5, 'NO SAFE HORIZON', False, -20)
    return controller, physics, obs, decision


def test_avoids_immediate_bottom_collision_despite_failed_distant_plan():
    controller, physics, obs, decision = setup_state()
    protected = protect(controller, decision, obs, physics, 1, .01)
    assert protected.flap and protected.reason == 'SAFETY_SHIELD CLICK'
    assert protected.flap_margin > 0
    assert not protected.feasible  # Không tuyên bố đường dài đã an toàn.


def test_refuses_jump_that_would_hit_upper_pipe():
    controller, physics, obs, decision = setup_state()
    obs.gates[0].top = 130
    assert protect(controller, decision, obs, physics, 1, .01) is decision


def test_never_duplicates_pending_click_or_breaks_cooldown():
    controller, physics, obs, decision = setup_state()
    physics.pending = [1.02]
    assert protect(controller, decision, obs, physics, 1, .01) is decision
    physics.pending = []; controller.last_flap = .9
    assert protect(controller, decision, obs, physics, 1, .01) is decision


def test_vetoes_upper_collision_and_restores_cooldown_for_unsent_click():
    controller,physics,obs,decision=setup_state(y=160.5,velocity=-135.547)
    obs.gates=[Gate(128,214,100,219)]
    decision.flap=True;controller.last_flap=1
    protected=protect(controller,decision,obs,physics,1,.00423,previous_flap=.7)
    assert not protected.flap and protected.reason=='SAFETY_SHIELD VETO CLICK'
    assert protected.coast_margin>0 and protected.flap_margin<0
    assert controller.last_flap==.7


def test_veto_only_keeps_original_coast_for_isolated_live_comparison():
    controller,physics,obs,decision=setup_state()
    controller.cfg['shield_veto_only']=True
    assert protect(controller,decision,obs,physics,1,.01) is decision


def test_short_coast_veto_handles_448_state_without_claiming_long_path_safe():
    controller,physics,obs,decision=setup_state(y=148,velocity=-83.225236)
    physics.g=710.38;physics.jump=-248.58494
    controller.cfg.update(latency=.032,veto_coast_horizon=.08)
    obs.bird=Bird(144.5,148,53,50);obs.gates=[Gate(112,198,93,213)]
    decision.flap=True;controller.last_flap=1
    protected=protect(controller,decision,obs,physics,1,.0044,previous_flap=.7)
    assert not protected.flap and protected.coast_margin>0 and protected.flap_margin<0
    assert not protected.feasible
    assert controller.last_flap==.7


def logged_reachability_state(y,velocity,gates,enabled):
    controller,physics,obs,decision=setup_state(y=y,velocity=velocity)
    controller.cfg.update(planner='reachability',latency=.032,pipe_speed=139.6,
                          gravity=710.38,jump=-248.58494,hitbox_half_width=26,
                          hitbox_half_height=20,shield_veto_only=True,
                          shield_current_top_veto=enabled)
    physics.g=710.38;physics.jump=-248.58494;physics.speed=139.6
    obs.bird=Bird(142.5,y,51,49)
    obs.gates=[Gate(*gate) for gate in gates]
    decision.flap=True;controller.last_flap=1.
    return controller,physics,obs,decision


@pytest.mark.parametrize('y,velocity,gate',[
    (131.5,32.8,(92,178,77,198)),
    (152.,71.1,(90,176,95,216)),
    (125.,-74.,(108,194,68,190)),
    (148.,-11.8,(99,185,97,217)),
    (161.,95.2,(96,182,103,224)),
    (127.5,126.,(100,186,69,190)),
])
def test_current_top_guard_blocks_six_logged_upper_collisions(y,velocity,gate):
    controller,physics,obs,decision=logged_reachability_state(y,velocity,[gate],True)
    protected=protect(controller,decision,obs,physics,1.,.01,previous_flap=.5)
    assert protected.reason=='SAFETY_SHIELD CURRENT_TOP VETO CLICK'
    assert not protected.flap and protected.coast_margin>3>protected.flap_margin
    assert controller.last_flap==.5


@pytest.mark.parametrize('y,velocity,gates',[
    (211.,108.,[(39,125,169,288),(192,278,68,188)]),
    (231.5,202.7,[(20,106,169,270),(192,278,68,188)]),
    (226.,191.6,[(28,114,166,286),(192,278,68,188)]),
])
def test_current_top_guard_does_not_veto_for_distant_pipe_or_trailing_edge(y,velocity,gates):
    controller,physics,obs,decision=logged_reachability_state(y,velocity,gates,True)
    protected=protect(controller,decision,obs,physics,1.,.01,previous_flap=.5)
    assert protected.reason!='SAFETY_SHIELD CURRENT_TOP VETO CLICK'


def test_current_top_guard_opt_in_keeps_stable_profile_unchanged():
    controller,physics,obs,decision=logged_reachability_state(131.5,32.8,[(92,178,77,198)],False)
    protected=protect(controller,decision,obs,physics,1.,.01,previous_flap=.5)
    assert protected.reason!='SAFETY_SHIELD CURRENT_TOP VETO CLICK'


@pytest.mark.parametrize('y,velocity,left,right,top,bottom',[
    (131.5, 32.8, 92, 178, 77, 198),
    (152.0, 71.1, 90, 176, 95, 216),
    (125.0, -74.0, 108, 194, 68, 190),
])
def test_short_veto_replays_three_logged_upper_lip_failures(y,velocity,left,right,top,bottom):
    # Ba cú click cuối từ log Windows; planner chọn CLICK khi margin dài hạn âm.
    controller, physics, obs, decision = setup_state(y=y, velocity=velocity)
    controller.cfg.update(latency=.032,pipe_speed=139.6,gravity=710.38,
                          jump=-248.58494,hitbox_half_width=26,
                          hitbox_half_height=20,shield_veto_only=True,
                          veto_coast_horizon=.2)
    physics.g=710.38;physics.jump=-248.58494;physics.speed=139.6
    obs.bird=Bird(142.5,y,51,49)
    obs.gates=[Gate(left,right,top,bottom)]
    decision.flap=True;controller.last_flap=1.
    protected=protect(controller,decision,obs,physics,1.,.01,previous_flap=.5)
    assert protected.reason=='SAFETY_SHIELD VETO CLICK'
    assert not protected.flap and protected.coast_margin>0>protected.flap_margin
    assert controller.last_flap==.5
