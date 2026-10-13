import numpy as np
import pytest
from bot.vision import Bird,Gate,Observation
from bot.collision_geometry import (body_extents,body_center_offset_y,
                                    canvas_to_frame_scale,world_clearance_bounds,
                                    CollisionSamples)

def test_calibrated_hitbox_keeps_detection_bbox_and_default_geometry():
    b=Bird(145,125,52,50);cfg=dict(margin=0)
    assert body_extents(cfg,b)==(26,25)
    assert body_extents(dict(cfg,hitbox_half_height=20),b)==(26,20)
    assert b.h==50

def test_atlas_and_shield_use_same_geometry_while_sprite_rotates():
    cfg=dict(margin=0,planner='reachability')
    assert body_extents(cfg,Bird(145,125,50,44))==(27,25)
    assert body_extents(cfg,Bird(145,125,55,52))==(27,25)

def test_source_hitbox_scales_from_canvas_and_preserves_center_offset():
    cfg=dict(width=600,source_canvas_width=900,source_canvas_height=600,
             source_hitbox_width=50,source_hitbox_height=55,source_hitbox_offset_y=9,
             source_player_height=72,source_ground_y=535,source_player_top_limit=-10,
             source_collision_geometry=True,margin=0,
             roi=dict(width=816,height=544))
    assert canvas_to_frame_scale(cfg)==pytest.approx(2/3)
    assert body_extents(cfg,Bird(145,125,52,50))==pytest.approx((50/3,55/3))
    assert body_center_offset_y(cfg)==pytest.approx(1/3)
    assert world_clearance_bounds(cfg,55/3,0,356)==pytest.approx((-2/3,527*2/3))
    cfg['width']=300
    assert body_extents(cfg,Bird(72.5,62.5,26,25))==pytest.approx((25/3,55/6))
    assert body_center_offset_y(cfg)==pytest.approx(1/6)

def test_source_hitbox_rejects_roi_with_wrong_canvas_aspect_ratio():
    cfg=dict(width=600,source_canvas_width=900,source_canvas_height=600,
             source_hitbox_width=50,source_hitbox_height=55,source_hitbox_offset_y=9,
             source_player_height=72,source_ground_y=535,source_player_top_limit=-10,
             source_collision_geometry=True,margin=0,
             roi=dict(width=816,height=500))
    with pytest.raises(ValueError,match='ROI'):
        body_extents(cfg,Bird(145,125,52,50))

def test_collision_sample_requires_later_alive_observations(tmp_path):
    collector=CollisionSamples(True)
    obs=Observation(Bird(145,135,52,50),[Gate(100,186,100,220)],np.zeros((495,600,3),np.uint8))
    collector.observe(obs,1)
    obs.bird.y=120;collector.observe(obs,1.2)
    records=collector.save(tmp_path,1.65)
    assert len(records)==1 and records[0]['t']==1
    assert records[0]['alive_after_s']>=.5
    assert (tmp_path/'calibration-1.000s.jpg').is_file()
