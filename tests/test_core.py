import numpy as np
import pytest
from bot.physics import project,Physics
from bot.controller import Controller
from bot.vision import Bird,Gate,Observation,Vision
from bot.screen_capture import local_to_screen

@pytest.fixture
def cfg():
    import yaml
    from pathlib import Path
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'config.yaml').read_text())

def test_project():
    y,v=project(100,-200,1000,.2)
    assert y==pytest.approx(80)
    assert v==pytest.approx(0)

def test_coordinates_negative_monitor():
    assert local_to_screen(300,200,{'left':-1920,'top':100,'width':1200},600)==(-1320,500)

def test_velocity_fit(cfg):
    p=Physics(cfg)
    p.g=1000
    for t in np.linspace(0,.3,10):
        y=200-100*t+500*t*t
        p.update(Bird(140,y,35,35),[],float(t))
    assert p.v==pytest.approx(200,abs=3)
    assert p.g==pytest.approx(1000,abs=10)

def test_post_input_latency(cfg):
    p=Physics(cfg)
    p.flap(1.)
    p.update(Bird(140,200,35,35),[],1.2)
    assert p.v==pytest.approx(p.jump+.2*p.g)

def test_missing_no_input(cfg):
    assert not Controller(cfg).decide(Observation(None,[],np.zeros((600,600,3),np.uint8)),Physics(cfg),1).flap

def test_flap_before_floor(cfg):
    p=Physics(cfg);p.v=300
    o=Observation(Bird(140,500,35,35),[],np.zeros((600,600,3),np.uint8))
    assert Controller(cfg).decide(o,p,1).flap

def test_ceiling_and_cooldown(cfg):
    p=Physics(cfg);p.v=-200
    o=Observation(Bird(140,60,35,35),[Gate(300,365,180,320)],np.zeros((600,600,3),np.uint8))
    c=Controller(cfg)
    assert not c.decide(o,p,1).flap
    c.last_flap=1
    assert not c.decide(o,p,1.05).flap

def test_pipe_tree_discrimination(cfg,tmp_path):
    import cv2
    frame=np.full((600,600,3),(220,220,120),np.uint8)
    cv2.rectangle(frame,(350,0),(410,180),(60,150,60),-1)
    cv2.rectangle(frame,(350,330),(410,570),(60,150,60),-1)
    cv2.rectangle(frame,(50,350),(75,410),(60,150,60),-1)
    o=Vision(cfg,tmp_path).detect(frame)
    assert len(o.gates)==1
    assert o.gates[0].top==pytest.approx(181,abs=2)
    assert o.gates[0].bottom==pytest.approx(330,abs=2)

def test_real_terrain_does_not_shrink_gap(cfg,tmp_path):
    import cv2
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    image=cv2.imread(str(root/'screenshots/local-native-6-controller/13-4.488s.jpg'))
    assert image is not None
    calibration=dict(cfg)
    # Ngưỡng đã kiểm chứng trên ảnh này; tách khỏi config người dùng có thể đổi.
    calibration['pipe_hsv_high']=[85,255,190]
    gate=Vision(calibration,tmp_path).detect(image).gates[0]
    # Visually verified pipe lips: terrain starts at 256, lower lip at 270.
    assert gate.top==pytest.approx(151,abs=2)
    assert gate.bottom==pytest.approx(270,abs=2)

def test_trailing_pipe_is_kept_when_hud_occludes_it(cfg,tmp_path):
    v=Vision(cfg,tmp_path)
    v.previous_gates=[Gate(60,146,100,220)]
    v.previous_time=1.;v.gate_seen={0:1.}
    blank=np.full((400,600,3),(230,240,197),np.uint8)
    o=v.detect(blank,1.05)
    assert len(o.gates)==1
    assert o.gates[0].right==pytest.approx(146-cfg['pipe_speed']*.05)

def test_gate_speed_uses_moving_right_edge(cfg):
    p=Physics(cfg)
    p.update(Bird(147,200,50,45),[Gate(110,180,100,220)],1.)
    p.update(Bird(147,200,50,45),[Gate(110,166,100,220)],1.1)
    assert p.speed==pytest.approx(.97*cfg['pipe_speed']+.03*140)


def test_tracking_rejects_new_clipped_pipe_until_width_is_reliable(cfg,tmp_path):
    import cv2
    cfg=dict(cfg,track_gates=True,pipe_hsv_high=[85,255,190])
    vision=Vision(cfg,tmp_path)
    partial=np.full((400,600,3),(220,220,120),np.uint8)
    cv2.rectangle(partial,(565,0),(599,120),(60,150,60),-1)
    cv2.rectangle(partial,(565,240),(599,390),(60,150,60),-1)
    assert not vision.detect(partial,1).gates
    full=np.full_like(partial,(220,220,120))
    cv2.rectangle(full,(520,0),(579,120),(60,150,60),-1)
    cv2.rectangle(full,(520,240),(579,390),(60,150,60),-1)
    gates=vision.detect(full,1.1).gates
    assert len(gates)==1 and gates[0].right-gates[0].x>=86*.7


def test_tracked_occlusion_exposes_lower_confidence(cfg,tmp_path):
    vision=Vision(dict(cfg,track_gates=True),tmp_path)
    vision.previous_gates=[Gate(60,146,100,220)]
    vision.previous_time=1;vision.gate_seen={0:1}
    obs=vision.detect(np.full((400,600,3),(230,240,197),np.uint8),1.05)
    assert len(obs.gates)==1
    assert obs.gates[0].source=='tracked_occlusion' and obs.gates[0].confidence<1


def test_entering_gate_recovers_hidden_width_from_calibration(cfg,tmp_path):
    import cv2
    settings=dict(cfg,reconstruct_entering_gate=True,gate_width_px=86,pipe_hsv_high=[85,255,190])
    vision=Vision(settings,tmp_path)
    partial=np.full((400,600,3),(220,220,120),np.uint8)
    cv2.rectangle(partial,(565,0),(599,120),(60,150,60),-1)
    cv2.rectangle(partial,(565,240),(599,390),(60,150,60),-1)
    gates=vision.detect(partial,1).gates
    assert len(gates)==1
    assert gates[0].right>600
    assert gates[0].right-gates[0].x==86
    assert gates[0].source=='reconstructed_entering'
    assert gates[0].confidence<1
