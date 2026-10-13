"""Mô hình source; mọi độ dài native được đổi cùng một tỉ lệ canvas → ROI."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SourcePhysics:
    gravity: float=1080.
    jump: float=-385.
    speed: float=205.
    pipe_width: float=92.
    gap: float=180.
    spawn_interval: float=1.42
    player_width: float=72.
    player_height: float=72.
    hitbox_x: float=11.
    hitbox_y: float=9.
    hitbox_width: float=50.
    hitbox_height: float=55.
    lip_side: float=16.
    lip_height: float=28.

SOURCE=SourcePhysics()

def configure(cfg):
    width=cfg.get('source_canvas_width');height=cfg.get('source_canvas_height')
    if not width or not height:raise ValueError('Điền source_canvas_width/source_canvas_height từ JavaScript trong hồ sơ source.')
    if width<=0 or height<=0:raise ValueError('Kích thước canvas phải dương.')
    if width!=900 or height!=600:raise ValueError('Hồ sơ source này dành cho desktop 900×600; mobile có offset chưa được mô hình hóa.')
    if abs(cfg.get('source_frame_dt',1/120)-1/120)>1e-9:raise ValueError('Source đã xác nhận fixed timestep 120 Hz; source_frame_dt phải là 1/120.')
    scale=cfg['width']/width
    cfg.setdefault('source_high_resolution_wait',True)
    ground=cfg.get('source_ground_y')
    if ground is None:raise ValueError('Điền source_ground_y bằng groundY trong JavaScript.')
    roi=cfg.get('roi')
    if roi and abs(roi['height']/roi['width']-height/width)>height/width*.05:
        raise ValueError('ROI sai tỉ lệ Canvas 900×600: chọn lại toàn bộ canvas, không cắt riêng phần giữa.')
    cfg.update(gravity=SOURCE.gravity*scale,jump=SOURCE.jump*scale,pipe_speed=SOURCE.speed*scale,
               hitbox_half_width=SOURCE.hitbox_width/2*scale,hitbox_half_height=SOURCE.hitbox_height/2*scale,
               world_floor_px=(ground-8)*scale-1e-6,world_ceiling_px=-scale+1e-6,source_scale=scale,
               gate_width_px=(SOURCE.pipe_width+2*SOURCE.lip_side)*scale,
               pipe_lip_side_px=SOURCE.lip_side*scale,freeze_physics=True)
    return cfg

def step(y,v,g,dt):
    """Đúng thứ tự JS: tăng vy trước, sau đó cộng vy mới vào y."""
    v=v+g*dt
    return y+v*dt,v

def predict(y,v,g,duration,frame_dt):
    if duration<0 or frame_dt<=0:raise ValueError('dt không hợp lệ.')
    whole=int(duration/frame_dt+1e-9);remainder=max(0.,duration-whole*frame_dt)
    y=y+v*whole*frame_dt+.5*g*frame_dt**2*whole*(whole+1)
    v=v+g*whole*frame_dt
    return step(y,v,g,remainder) if remainder>1e-9 else (y,v)

def rectangles(gates,cfg,ceiling,floor):
    """Gate.x/right là mép đầu ống nhìn thấy, không phải mép thân.

    Các thân và đầu ống đúng bốn rectHit trong source người dùng cung cấp.
    """
    s=cfg['source_scale'];side=SOURCE.lip_side*s;lip=SOURCE.lip_height*s
    rows=[]
    for gate in gates:
        left=gate.x;right=gate.right
        rows.extend([(left+side,right-side,0.,gate.top),
                     (left,right,gate.top-lip,gate.top),
                     (left,right,gate.bottom,gate.bottom+lip),
                     (left+side,right-side,gate.bottom,cfg['source_ground_y']*s)])
    return np.asarray([r for r in rows if r[3]>r[2]],dtype=np.float64).reshape(-1,4)

def hitbox_center(bird,cfg):
    s=cfg['source_scale']
    x=(cfg.get('source_player_x',180)+SOURCE.hitbox_x+SOURCE.hitbox_width/2)*s
    return x, bird.y+(SOURCE.hitbox_y+SOURCE.hitbox_height/2-SOURCE.player_height/2)*s
