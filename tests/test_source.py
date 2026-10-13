import numpy as np
import pytest
from bot.source_model import configure,step,predict,rectangles,hitbox_center
from bot.source_planner import frame_margin
from bot.vision import Bird,Gate
from sim.source_engine import SourceSimulator

def config():
    return configure(dict(width=600,source_canvas_width=900,source_canvas_height=600,source_ground_y=535,source_frame_dt=1/120,margin=0))

@pytest.mark.parametrize('ticks',[1,4,40,120])
def test_semi_implicit_matches_repeated_js_update(ticks):
    y=280.;v=-385.
    for _ in range(ticks):v+=1080/120;y+=v/120
    assert predict(280.,-385.,1080.,ticks/120,1/120)==pytest.approx((y,v))

def test_coordinate_hitbox_and_sprite_ground_bounds():
    cfg=config();assert cfg['gravity']==720 and cfg['jump']==pytest.approx(-385*2/3)
    assert cfg['pipe_speed']==pytest.approx(205*2/3)
    assert cfg['hitbox_half_height']==pytest.approx(55/3)
    assert hitbox_center(Bird(999,200,48,48),cfg)==pytest.approx((144,200+1/3))
    assert cfg['world_floor_px']==pytest.approx((535-8)*2/3,abs=1e-5)

def test_roi_cannot_be_cropped_middle_of_canvas():
    cfg=dict(config(),roi=dict(width=1000,height=600))
    with pytest.raises(ValueError,match='ROI'):configure(cfg)

def test_pipe_rectangles_include_lips_without_shrinking_gap():
    cfg=config();rs=rectangles([Gate(100,100+124*2/3,100,220)],cfg,0,360)
    assert rs.shape==(4,4)
    assert rs[0,0]==pytest.approx(100+16*2/3)
    assert rs[1,2:]==pytest.approx([100-28*2/3,100])
    assert rs[2,2:]==pytest.approx([220,220+28*2/3])

def test_render_elapsed_does_not_change_physics_tick():
    a=SourceSimulator([],20000);b=SourceSimulator([],20000)
    for _ in range(30):a.elapsed(1/30)
    for _ in range(120):b.elapsed(1/120)
    assert a.tick==b.tick==120 and a.y==pytest.approx(b.y)
    b.elapsed(.251);assert b.paused and b.tick==120

def test_strict_pipe_touch_and_sprite_ground():
    rs=np.array([[100.,192.,0.,100.]])
    assert frame_margin(75.,75.,25.,27.5,rs,205.,0.,-1.,532.)>0
    assert frame_margin(75.001,75.,25.,27.5,rs,205.,0.,-1.,532.)<0

def test_collision_model_matches_independent_native_rect_oracle():
    cfg=config();s=2/3;gate=Gate(184*s,308*s,180*s,360*s)
    rs=rectangles([gate],cfg,cfg['world_ceiling_px'],cfg['world_floor_px'])
    rng=np.random.default_rng(42)
    for _ in range(200):
        y=float(rng.uniform(-20,535));hb=(191,y+9,50,55)
        blocks=((200,0,92,180),(184,152,124,28),(200,360,92,175),(184,360,124,28))
        dead=y<=-10 or y+72>=535 or any(hb[0]<x+w and hb[0]+hb[2]>x and hb[1]<top+h and hb[1]+hb[3]>top for x,top,w,h in blocks)
        margin=frame_margin(216*s,(y+36.5)*s,25*s,27.5*s,rs,205*s,0.,cfg['world_ceiling_px'],cfg['world_floor_px'])
        assert (margin<0)==dead

def test_source_observer_recovers_end_of_tick_velocity():
    from bot.physics import Physics
    cfg=dict(config(),planner='source_euler',cooldown=.12)
    physics=Physics(cfg);sim=SourceSimulator([],535)
    for _ in range(4):
        obs=sim.observe();physics.update(obs.bird,obs.gates,sim.tick/120);sim.elapsed(1/30)
    assert physics.v==pytest.approx(1080*.1*2/3,abs=1e-6)

def test_closed_form_compensation_matches_120hz_endpoints():
    cfg=config();g=cfg['gravity'];v=cfg['jump'];h=1/120
    for ticks in (4,40,120):
        t=ticks*h;analytic=200+(v+g*h/2)*t+.5*g*t*t
        assert analytic==pytest.approx(predict(200,v,g,t,h)[0])

def test_mobile_not_silently_treated_as_desktop():
    with pytest.raises(ValueError,match='mobile'):configure(dict(config(),source_canvas_height=900))

def test_known_impulse_prior_reduces_noisy_velocity_error():
    from bot.physics import Physics
    noise=np.random.default_rng(13).normal(0,1,12)
    errors=[]
    for weight in (0.,.7):
        cfg=dict(config(),planner='source_euler',source_velocity_prior_weight=weight)
        model=Physics(cfg);model.last_flap=1.;values=[]
        for i in range(12):
            dt=i/30;y,v=predict(220,cfg['jump'],cfg['gravity'],dt,1/120)
            model.update(Bird(144,y+noise[i],48,48),[],1+dt)
            if i>=3:values.append(abs(model.v-v))
        errors.append(np.mean(values))
    assert errors[1]<errors[0]

def test_impulse_prior_does_not_hide_inconsistent_motion():
    from bot.physics import Physics
    cfg=dict(config(),planner='source_euler');model=Physics(cfg);model.last_flap=1.
    for i in range(5):
        y,v=predict(150,100,cfg['gravity'],i/30,1/120)
        model.update(Bird(144,y,48,48),[],1+i/30)
    assert model.input_status=='source_velocity_mismatch'
    assert model.v==pytest.approx(v,abs=1e-5)

def test_generated_pipes_match_source_spawn_range():
    from sim.source_engine import sequence
    for alternating in (False,True):
        for p in sequence(42,100,535,alternating):
            assert 90<=p['top']<265
            assert p['bottom']-p['top']==pytest.approx(180)


def test_source_tracking_speed_is_not_reestimated_from_noisy_images(tmp_path):
    import yaml
    from pathlib import Path
    from bot.vision import Vision
    cfg=yaml.safe_load(Path('profiles/windows-source.yaml').read_text(encoding='utf-8'));configure(cfg)
    v=Vision(cfg,tmp_path);v.gate_speed=86.
    v.detect(np.zeros((400,600,3),np.uint8),1.)
    assert v.gate_speed==pytest.approx(205*2/3)


def test_source_57_point_failure_replay_removes_false_early_jump():
    import yaml
    from pathlib import Path
    from bot.controller import Controller
    from bot.physics import Physics
    from bot.vision import Observation
    cfg=yaml.safe_load(Path('profiles/windows-source.yaml').read_text(encoding='utf-8'));cfg['reaction_interval']=.05
    results=[]
    for corrected in (False,True):
        model=Physics(dict(cfg));model.v=266.2266130605186
        controller=Controller(model.cfg);controller.last_flap=251.2646261
        gates=[Gate(79.76912947402751,162.43579614069418,64,183,.5,'tracked_occlusion'),Gate(233.33333333333331,316.,117,237),Gate(427.3333333333333,510.,129,249)]
        if corrected:
            gates[0].x=gates[1].x-205*1.42*2/3;gates[0].right=gates[0].x+124*2/3
        obs=Observation(Bird(147,149,52,48),gates,np.zeros((400,600,3),np.uint8))
        decision=controller.decide(obs,model,252.06468630000018,.025)
        results.append(decision)
    assert results[0].flap and results[0].plan_margin<0
    assert not results[1].flap and results[1].coast_margin>10
