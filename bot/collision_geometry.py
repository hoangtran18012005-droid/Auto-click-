"""Tách kích thước hình vẽ khỏi mô hình va chạm đã hiệu chỉnh."""

def canvas_to_frame_scale(cfg):
    """Scale theo chiều ngang sau resize; ROI phải chứa trọn canvas đúng tỉ lệ."""
    canvas_width=cfg.get('source_canvas_width')
    canvas_height=cfg.get('source_canvas_height')
    if not canvas_width or not canvas_height or canvas_width<=0 or canvas_height<=0:
        raise ValueError('Thiếu kích thước Canvas nguồn dương.')
    roi=cfg.get('roi')
    if roi:
        expected=canvas_width/canvas_height
        actual=roi['width']/roi['height']
        if abs(actual/expected-1)>cfg.get('source_roi_aspect_tolerance',.01):
            raise ValueError('ROI không cùng tỉ lệ Canvas nguồn; không thể đổi hitbox chính xác.')
    return cfg['width']/canvas_width


def body_center_offset_y(cfg):
    """Offset từ tâm sprite sang tâm hitbox, theo đơn vị frame đã resize."""
    if cfg.get('source_collision_geometry',False):
        scale=canvas_to_frame_scale(cfg)
        return (cfg['source_hitbox_offset_y']+
                cfg['source_hitbox_height']/2-cfg['source_player_height']/2)*scale
    return cfg.get('hitbox_center_offset_y',0.)


def world_clearance_bounds(cfg,half_height,ceiling,floor):
    """Translate sprite top/ground stops to the planner's hitbox-center bounds."""
    if not cfg.get('source_collision_geometry',False):return ceiling,floor
    scale=canvas_to_frame_scale(cfg)
    required=('source_hitbox_offset_y','source_hitbox_height',
              'source_player_height','source_ground_y')
    if any(key not in cfg for key in required):
        raise ValueError('Thiếu thông số source để đổi giới hạn trần/sàn.')
    hitbox_center=cfg['source_hitbox_offset_y']+cfg['source_hitbox_height']/2
    minimum_center=(cfg.get('source_player_top_limit',-10)+hitbox_center)*scale
    maximum_center=(cfg['source_ground_y']-cfg['source_player_height']+hitbox_center)*scale
    return minimum_center-half_height,maximum_center+half_height


def body_extents(cfg,bird):
    padding=cfg['margin']+cfg.get('hitbox_margin_px',0)
    width=cfg.get('hitbox_half_width')
    height=cfg.get('hitbox_half_height')
    if cfg.get('source_collision_geometry',False):
        scale=canvas_to_frame_scale(cfg)
        width=cfg['source_hitbox_width']*scale/2
        height=cfg['source_hitbox_height']*scale/2
    if cfg.get('planner')=='reachability':
        if width is None:width=max(cfg.get('atlas_half_width',27),cfg.get('minimum_body_width',0)/2)
        if height is None:height=max(cfg.get('atlas_half_height',25),cfg.get('minimum_body_height',0)/2)
    if width is None:width=max(bird.w,cfg.get('minimum_body_width',0))/2
    if height is None:height=max(bird.h,cfg.get('minimum_body_height',0))/2
    if width<=0 or height<=0:raise ValueError('Kích thước hitbox phải dương.')
    return max(1,width+padding),max(1,height+padding)

class CollisionSamples:
    """Giữ vài ảnh gần mép; chỉ ghi file sau khi gameplay dừng."""
    def __init__(self,enabled=False,cfg=None):
        self.enabled=enabled;self.cfg=cfg or {'margin':0};self.best=30.;self.samples=[]
    def observe(self,obs,t):
        if not self.enabled or obs.bird is None or len(self.samples)>=20:return
        b=obs.bird
        hw,hh=body_extents(self.cfg,b)
        center_y=b.y+body_center_offset_y(self.cfg)
        for gate in obs.gates:
            if gate.source!='image' or not (0<gate.x<gate.right<600) or abs(gate.right-gate.x-86)>2:continue
            if any(abs(value-round(value))>.01 for value in (gate.x,gate.right,gate.top,gate.bottom)):continue
            if not gate.x+hw<b.x<gate.right-hw:continue
            top_clearance=center_y-hh-gate.top
            bottom_clearance=gate.bottom-center_y-hh
            distance=min(top_clearance,bottom_clearance)
            if 0<distance<=self.best-.5:
                self.best=distance
                self.samples.append((t,obs.frame.copy(),dict(t=t,x=b.x,y=b.y,w=b.w,h=b.h,
                    collision_center_y=center_y,collision_hw=hw,collision_hh=hh,
                    top_clearance=top_clearance,bottom_clearance=bottom_clearance,
                    distance=distance,gate=vars(gate).copy())))
    def save(self,folder,last_observed):
        import cv2
        records=[]
        for t,frame,record in self.samples:
            if last_observed-t<.5:continue
            path=folder/f'calibration-{t:.3f}s.jpg'
            cv2.imwrite(str(path),frame)
            records.append(dict(record,image=str(path),alive_after_s=last_observed-t))
        return records
