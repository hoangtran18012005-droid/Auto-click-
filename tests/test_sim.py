import json
import numpy as np
import pytest
from bot.physics import project
from bot.vision import Gate
from exact_clearance import maxmin_clearance
from sim.engine import (SimulationConfig,Simulator,interval_clearance,random_sequence,
                        source_random_sequence,transition_type)
from sim.sequences import rebuild_sequence
from tools.regret import reconstruct_velocity,summarize


def test_free_fall_matches_closed_form():
    c=SimulationConfig(noise_y=0)
    s=Simulator([],c,y=100,velocity=20)
    for dt in [.016,.033,.05,.019,.022]:s.advance(dt)
    assert s.y==pytest.approx(project(100,20,c.gravity,.14)[0])
    assert s.velocity==pytest.approx(20+c.gravity*.14)


def test_delayed_jump_sets_velocity():
    c=SimulationConfig(noise_y=0)
    s=Simulator([],c,y=100,velocity=100)
    assert s.command(True)
    s.advance(.2)
    before=project(100,100,c.gravity,c.latency)[0]
    expected=project(before,c.jump,c.gravity,.2-c.latency)
    assert (s.y,s.velocity)==pytest.approx(expected)


def test_pipe_entry_checks_whole_interval():
    c=SimulationConfig(noise_y=0)
    # Va chạm bắt đầu đúng lúc mép trái tới mép phải nhân vật.
    gate=Gate(c.bird_x+c.bird_width/2+14.1,300,200,320)
    s=Simulator([gate],c,y=100,velocity=0)
    s.advance(.099)
    assert not s.dead
    s.advance(.05)
    assert s.dead
    assert s.time==pytest.approx(.1,abs=1e-8)


def test_pipe_after_trailing_edge_does_not_collide():
    c=SimulationConfig(noise_y=0)
    gate=Gate(0,c.bird_x-c.bird_width/2-.1,200,320)
    s=Simulator([gate],c,y=100)
    s.advance(.1)
    assert not s.dead
    assert s.score==1


def test_simulation_separates_sprite_bounds_from_pipe_hitbox():
    gate=Gate(100,200,125,300)
    legacy=Simulator([gate],SimulationConfig(gravity=1,noise_y=0),y=145)
    exact=Simulator([gate],SimulationConfig(gravity=1,noise_y=0,
        collision_half_width=50/3,collision_half_height=55/3,
        collision_center_offset_y=1/3),y=145)
    legacy.advance(.01)
    exact.advance(.01)
    assert legacy.dead  # sprite-sized proxy overlaps the upper pipe
    assert not exact.dead  # source hitbox clears the same visible edge
    assert exact.y<exact.config.floor-exact.config.bird_height/2


def test_source_sequence_matches_native_spawn_height_gap_and_spacing():
    gates=source_random_sequence(17,4)
    assert all(60<=gate.top<265*2/3 for gate in gates)
    assert all(gate.bottom-gate.top==pytest.approx(120) for gate in gates)
    assert gates[0].x==pytest.approx((920-16)*2/3)
    assert gates[0].right-gates[0].x==pytest.approx(124*2/3)
    expected_spacing=205*1.42*2/3
    assert gates[1].x-gates[0].x==pytest.approx(expected_spacing)


def test_apex_between_observations_is_checked():
    assert interval_clearance(30,-100,1000,0,.2,26,100)==pytest.approx(-1.)


def test_cooldown_and_pending_input():
    s=Simulator([],SimulationConfig(noise_y=0),y=150)
    assert s.command(True)
    assert not s.command(True)
    s.advance(.18)
    assert s.command(True)
    assert len(s.pending)==1


def test_random_sequences_are_paired_and_bounded():
    a=random_sequence(42);b=random_sequence(42)
    assert a==b
    assert all(g.bottom-g.top==pytest.approx(120) for g in a)
    assert all(a[i+1].x-a[i].x==195 for i in range(len(a)-1))
    centers=np.array([(g.top+g.bottom)/2 for g in a])
    assert np.max(np.abs(np.diff(centers)))<=230


def test_rebuild_known_log_sequence(tmp_path):
    frames=[]
    for t in [0.,.1,.2,.3]:
        frames.append({'t':t,'gates':[{'x':300-141*t,'right':386-141*t,'top':100,'bottom':220},
                                    {'x':495-141*t,'right':581-141*t,'top':170,'bottom':290}]})
    path=tmp_path/'log.json';path.write_text(json.dumps({'frames':frames}))
    gates,meta=rebuild_sequence(path)
    assert len(gates)==2
    assert gates[0].x==pytest.approx(300)
    assert gates[1].top==170
    assert meta['offset_residual_max_px']==pytest.approx(0)


def test_reconstruct_pre_flap_v_uses_irregular_timestamp():
    previous={'t':1.,'v':100}
    frame={'t':1.047,'flap':True,'v':-264.2,'g':687.4}
    assert reconstruct_velocity(frame,previous)==pytest.approx(100+687.4*.047)
    assert reconstruct_velocity(frame,None) is None
    frame['flap']=False
    assert reconstruct_velocity(frame,previous)==-264.2


def test_oracle_forced_action_matches_short_enumeration():
    y=200.;v=100.;dt=1/60
    gate={'x':140,'right':220,'top':100,'bottom':280}
    values=[]
    for action in [False,True]:
        vy=-264.2 if action else v
        margins=[]
        for _ in range(2):
            vy+=687.4*dt;y_step=(y if not margins else position)+vy*dt
            position=y_step
            margins.append(min(position-24.5-100,280-position-24.5))
        expected=min(margins)
        actual=maxmin_clearance(y,v,[gate],first_action=action,every=2,horizon=2*dt)
        assert actual==pytest.approx(expected)
        values.append(actual)
    assert max(values)==pytest.approx(maxmin_clearance(y,v,[gate],every=2,horizon=2*dt))


def test_transition_and_regret_summary():
    high=Gate(0,86,65,185);low=Gate(195,281,151,271)
    assert transition_type(high,low)=='high_to_low'
    assert transition_type(low,high)=='low_to_high'
    result=summarize([{'regret_px':0},{'regret_px':6},{'regret_px':None}])
    assert result['frames']==2
    assert result['mean']==3
    assert result['over_5px']==1
