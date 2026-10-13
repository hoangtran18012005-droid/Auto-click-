"""Kiểm tra quỹ đạo gần trước khi chấp nhận COAST của planner đường dài."""
import numpy as np
from .maxmin_planner import clearance, extrema
from .collision_geometry import body_extents,body_center_offset_y,world_clearance_bounds


def protect(controller, decision, obs, physics, now, age, previous_flap=None):
    cfg = controller.cfg
    if cfg.get('shield_veto_only',False) and not decision.flap:
        return decision
    previous_flap=controller.last_flap if previous_flap is None else previous_flap
    if now-previous_flap < cfg['cooldown']:
        return decision
    if any(event > now for event in getattr(physics, 'pending', [])):
        return decision
    bird = obs.bird
    hw,hh=body_extents(cfg,bird)
    ceiling = cfg.get('world_ceiling_px', obs.frame.shape[0]*cfg.get('ceiling', 0))
    floor = cfg.get('world_floor_px', obs.frame.shape[0]*cfg['floor'])
    ceiling,floor=world_clearance_bounds(cfg,hh,ceiling,floor)
    delay = cfg['latency'] + min(age, .15)
    reaction = max(.025, min(.12, cfg.get('reaction_interval', 1/30)))
    gates = [gate for gate in obs.gates if gate.right >= bird.x-hw]
    args = (gates, bird.x, hw, hh, physics.speed, ceiling, floor)
    y = np.array([bird.y+body_center_offset_y(cfg)]); v = np.array([physics.v])
    coast = float(clearance(y, v, physics.g, 0, delay+reaction, *args)[0])
    if not decision.flap and coast > cfg.get('shield_reserve_px', 4):
        return decision
    prefix = float(clearance(y, v, physics.g, 0, delay, *args)[0])
    future_y, _ = physics.predict(float(y[0]), now, delay)
    jump = float(clearance(np.array([future_y]), np.array([physics.jump]), physics.g,
                           0, cfg.get('shield_horizon', .45), *args, offset=delay)[0])
    safe = min(prefix, jump)
    from .controller import Decision
    if decision.flap:
        if cfg.get('shield_current_top_veto',False):
            # Chỉ chặn cú nhảy đụng đỉnh ống ĐANG chồng ngang với nhân vật.
            # Không dùng ống kế tiếp để veto: ở log thực tế điều đó khiến bot
            # bỏ nhảy liên tiếp rồi rơi vào đáy khe hiện tại.
            reserve=cfg.get('top_guard_margin_px',3.)
            coast_horizon=cfg.get('top_guard_coast_horizon',.12)
            for gate in gates:
                if not (gate.x < bird.x+hw and gate.right > bird.x-hw):continue
                exit_in=(gate.right-bird.x+hw)/physics.speed-delay
                if exit_in<=0:continue
                low,_=extrema(np.array([future_y]),np.array([physics.jump]),physics.g,0,exit_in)
                top_margin=float(low[0]-hh-gate.top)
                if top_margin>=reserve:continue
                short_coast=float(clearance(y,v,physics.g,0,delay+coast_horizon,*args)[0])
                if short_coast>reserve:
                    controller.last_flap=previous_flap
                    return Decision(False,[],decision.target,'SAFETY_SHIELD CURRENT_TOP VETO CLICK',
                                    decision.feasible,decision.plan_margin,short_coast,top_margin,
                                    decision.planner_beam,decision.prefix_margin,decision.long_plan_margin)
        long_coast=float(clearance(y,v,physics.g,0,delay+cfg.get('veto_coast_horizon',cfg.get('shield_horizon',.45)),*args)[0])
        if safe < cfg.get('shield_min_margin_px',0) and long_coast > cfg.get('shield_min_margin_px',0):
            controller.last_flap=previous_flap
            return Decision(False,[],decision.target,'SAFETY_SHIELD VETO CLICK',
                            decision.feasible,decision.plan_margin,long_coast,safe,decision.planner_beam,decision.prefix_margin,decision.long_plan_margin)
        return decision
    if safe <= cfg.get('shield_min_margin_px', 0) or safe <= coast:
        return decision
    controller.last_flap = now
    return Decision(True, [], decision.target, 'SAFETY_SHIELD CLICK',
                    decision.feasible, decision.plan_margin, coast, safe, decision.planner_beam,decision.prefix_margin,decision.long_plan_margin)
