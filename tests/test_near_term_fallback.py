import numpy as np
from bot.physics import Physics
from bot.controller import Controller
from bot.vision import Bird,Gate,Observation
from bot.compiled_planner import warmup
from bot.controller import Decision

def test_distant_negative_plan_does_not_force_unsafe_jump_in_current_gate():
    warmup()
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,latency=.032,input_effect_latency=.032,cooldown=.18,margin=0,world_ceiling_px=0,world_floor_px=356,floor=.7,planner='maxmin',use_numba=True,pending_actions=True,beam_width=400,bin_y=1,bin_v=4,minimum_body_height=50,minimum_body_width=54,horizon_gates=2,planner_budget_ms=8,robust_states=[{'latency_ms':32},{'latency_ms':24,'jump_offset':-6,'velocity_offset':-6},{'latency_ms':48,'jump_offset':11,'velocity_offset':6}])
    physics=Physics(cfg);physics.v=59.061800949
    obs=Observation(Bird(146.5,116,51,48),[Gate(90.77224879,176.77224879,63,182),Gate(285,371,121,241),Gate(478,564,108,227)],np.zeros((495,600,3),np.uint8))
    now=42.7542296;last=42.2869276
    old=Controller(cfg);old.last_flap=last
    bad=old.decide(obs,physics,now,.006088)
    assert bad.flap and bad.plan_margin<0
    fixed=Controller(dict(cfg,near_term_fallback=True));fixed.last_flap=last
    good=fixed.decide(obs,physics,now,.006088)
    assert not good.flap and good.reason=='NEAR_TERM COAST'
    assert good.plan_margin>2 and good.long_plan_margin<0
    assert not good.feasible and fixed.last_flap==last


def test_two_gate_check_vetoes_local_rescue_before_lower_following_gap(monkeypatch):
    # Trạng thái ngay trước va chạm lượt 382: cứu cột trước đưa chim vào cột sau quá cao.
    warmup()
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,latency=.065,
             input_effect_latency=.065,cooldown=.18,margin=0,hitbox_half_width=26,
             hitbox_half_height=20,world_ceiling_px=0,world_floor_px=356,floor=.7,
             planner='reachability',use_numba=True,pending_actions=True,beam_width=200,
             bin_y=2,bin_v=8,search_dt=1/30,reaction_guard_ms=60,
             near_term_fallback=True,near_term_min_margin=2,near_term_budget_ms=2,
             near_term_two_gate_veto=True,near_term_two_gate_reserve_px=-1,
             near_term_two_gate_beam=400,safety_shield=False)
    obs=Observation(Bird(148,158,51,45),
                    [Gate(57,143,99,218),Gate(249,335,167,285)],
                    np.zeros((400,600,3),np.uint8))
    physics=Physics(cfg);physics.v=214.14
    monkeypatch.setattr('bot.reachability.plan',lambda *args: Decision(False,[],0,'REACHABILITY COAST',False,-.34,-.34,-48))
    old=Controller(dict(cfg,near_term_two_gate_veto=False));old.last_flap=545.02
    assert old.decide(obs,physics,545.736,.02).reason=='NEAR_TERM CLICK'
    controller=Controller(cfg);controller.last_flap=545.02
    choice=controller.decide(obs,physics,545.736,.02)
    assert not choice.flap and choice.reason=='REACHABILITY COAST'


def test_two_gate_check_keeps_rescue_when_both_long_paths_are_bad(monkeypatch):
    warmup()
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,latency=.065,
             input_effect_latency=.065,cooldown=.18,margin=0,hitbox_half_width=26,
             hitbox_half_height=20,world_ceiling_px=0,world_floor_px=356,floor=.7,
             planner='reachability',use_numba=True,pending_actions=True,beam_width=200,
             bin_y=2,bin_v=8,search_dt=1/30,reaction_guard_ms=60,
             near_term_fallback=True,near_term_min_margin=2,near_term_budget_ms=2,
             near_term_two_gate_veto=True,near_term_two_gate_reserve_px=-1,
             near_term_two_gate_beam=400,safety_shield=False)
    obs=Observation(Bird(148.5,175.5,51,45),
                    [Gate(80,166,113,233),Gate(272,358,170,289),Gate(466,552,140,259)],
                    np.zeros((400,600,3),np.uint8))
    physics=Physics(cfg);physics.v=144.18
    monkeypatch.setattr('bot.reachability.plan',lambda *args: Decision(False,[],0,'REACHABILITY COAST',False,-7.32,-7.32,-18))
    controller=Controller(cfg);controller.last_flap=72.08
    choice=controller.decide(obs,physics,72.703,.02)
    assert choice.flap and choice.reason=='NEAR_TERM CLICK'


def test_visible_third_gate_blocks_rescue_that_spoils_lower_gap(monkeypatch):
    warmup()
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,latency=.065,
             input_effect_latency=.065,cooldown=.18,margin=0,hitbox_half_width=26,
             hitbox_half_height=20,world_ceiling_px=0,world_floor_px=356,floor=.7,
             planner='reachability',use_numba=True,pending_actions=True,beam_width=200,
             bin_y=2,bin_v=8,search_dt=1/30,reaction_guard_ms=60,
             near_term_fallback=True,near_term_min_margin=2,near_term_budget_ms=2,
             near_term_visible_gate_veto=True,near_term_visible_gate_reserve_px=-1,
             near_term_visible_gate_beam=400,safety_shield=False)
    obs=Observation(Bird(148,169.5,51,45),
                    [Gate(68,154,114,233),Gate(260,346,110,230),Gate(454,540,176,294)],
                    np.zeros((400,600,3),np.uint8))
    physics=Physics(cfg);physics.v=168.42
    monkeypatch.setattr('bot.reachability.plan',lambda *args: Decision(False,[],0,'REACHABILITY COAST',False,-1.11,-1.11,-5))
    old=Controller(dict(cfg,near_term_visible_gate_veto=False));old.last_flap=5.40
    assert old.decide(obs,physics,6.053,.02).reason=='NEAR_TERM CLICK'
    controller=Controller(cfg);controller.last_flap=5.40
    choice=controller.decide(obs,physics,6.053,.02)
    assert not choice.flap and choice.reason=='REACHABILITY COAST'


def test_source_scaled_hitbox_removes_false_rescue_at_recorded_382_state():
    warmup()
    cfg=dict(gravity=710.38,jump=-248.58494,pipe_speed=139.6,latency=.065,
             input_effect_latency=.065,cooldown=.18,margin=0,world_ceiling_px=0,
             world_floor_px=356,floor=.7,planner='maxmin',use_numba=True,
             pending_actions=True,beam_width=200,bin_y=2,bin_v=8,
             search_dt=1/30,reaction_guard_ms=60,horizon_gates=3,
             near_term_fallback=True,near_term_min_margin=2,near_term_budget_ms=2,
             separate_actions=True,safety_shield=False,
             roi=dict(width=816,height=544),width=600)
    obs=Observation(Bird(148,158,52,46),
                    [Gate(57,143,99,218),Gate(249,335,167,285)],
                    np.zeros((400,600,3),np.uint8))
    old=Controller(dict(cfg,hitbox_half_width=26,hitbox_half_height=20));old.last_flap=545.078
    old_physics=Physics(old.cfg);old_physics.v=214.139914724
    old_choice=old.decide(obs,old_physics,545.7356484,.04)
    assert old_choice.flap and old_choice.reason=='NEAR_TERM CLICK'

    exact=dict(cfg,source_collision_geometry=True,source_canvas_width=900,
               source_canvas_height=600,source_hitbox_width=50,
               source_hitbox_height=55,source_hitbox_offset_y=9,
               source_player_height=72,source_ground_y=535,
               source_player_top_limit=-10)
    corrected=Controller(exact);corrected.last_flap=545.078
    corrected_physics=Physics(exact);corrected_physics.v=214.139914724
    corrected_choice=corrected.decide(obs,corrected_physics,545.7356484,.04)
    assert not corrected_choice.flap
    assert corrected_choice.plan_margin>18
