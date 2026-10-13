"""Beam y/v/cooldown qua nhiều khe, bước Euler bán ẩn đúng thứ tự JS."""
import math,time
import numpy as np
from numba import njit
from .controller import Decision
from .source_model import rectangles,hitbox_center,predict

@njit(cache=True,nogil=True)
def frame_margin(x,y,hw,hh,rects,speed,t,ceiling,floor):
    margin=min(y-hh-ceiling,floor-y-hh)
    for r in rects:
        left=r[0]-speed*t;right=r[1]-speed*t
        if x+hw>left and x-hw<right:
            margin=min(margin,max(r[2]-y-hh,y-hh-r[3]))
    return margin

@njit(cache=True,nogil=True)
def search(y,v,g,jump,x,hw,hh,rects,speed,ceiling,floor,dt,steps,action_every,cooldown,since,beam,bin_y,bin_v,reserve):
    ys=np.array([y]);vs=np.array([v]);ages=np.array([since]);scores=np.array([reserve]);first=np.array([-1])
    for index in range(steps):
        count=0;n=len(ys)
        ay=np.empty(n*2);av=np.empty(n*2);aa=np.empty(n*2);am=np.empty(n*2);af=np.empty(n*2,np.int64);keys=np.empty(n*2,np.int64)
        for p in range(n):
            for action in range(2):
                if action and (index%action_every!=0 or ages[p]<cooldown-1e-9):continue
                velocity=(jump if action else vs[p])+g*dt
                position=ys[p]+velocity*dt
                margin=min(scores[p],frame_margin(x,position,hw,hh,rects,speed,(index+1)*dt,ceiling,floor))
                # Giữ đường ít xấu nhất khi nhiễu khiến mọi nhánh âm;
                # không mặc định COAST chỉ vì không còn nhánh hợp lệ.
                age=min(cooldown,dt if action else ages[p]+dt)
                initial=action if index==0 else first[p]
                ay[count]=position;av[count]=velocity;aa[count]=age;am[count]=margin;af[count]=initial
                ky=int(np.rint(position/bin_y));kv=int(np.rint(velocity/bin_v))+10000;ka=int(np.rint(age/dt))
                keys[count]=((ky*30000+kv)*100+ka)*2+initial;count+=1
        if count==0:return False,-1e5,-1e5
        if index%action_every!=0:
            ys=ay[:count];vs=av[:count];ages=aa[:count];scores=am[:count];first=af[:count]
            continue
        order=np.argsort(keys[:count]);chosen=np.empty(count,np.int64);size=0
        for k in order:
            if size==0 or keys[k]!=keys[chosen[size-1]]:chosen[size]=k;size+=1
            elif am[k]>am[chosen[size-1]]:chosen[size-1]=k
        ranked=chosen[:size][np.argsort(-am[chosen[:size]])]
        selected=np.empty(min(size,beam),np.int64);used=0;counts=np.zeros(2,np.int64)
        # Giữ ngân sách riêng cho COAST và FLAP, tránh mất cả nhánh đầu.
        for k in ranked:
            a=af[k]
            if counts[a]>=max(1,beam//2):continue
            selected[used]=k;used+=1;counts[a]+=1
            if used==len(selected):break
        selected=selected[:used]
        ys=ay[selected];vs=av[selected];ages=aa[selected];scores=am[selected];first=af[selected]
    best=np.full(2,-1e5)
    for k in range(len(scores)):
        # Giữa các đường an toàn, ưu tiên trạng thái ra khe còn khả năng đổi độ cao.
        value=scores[k]-.002*abs(vs[k])
        best[first[k]]=max(best[first[k]],value)
    return best[1]>best[0]+.05,best[0],best[1]

def warmup():
    search(200.,0.,720.,-256.,150.,16.,18.,np.empty((0,4)),136.,0.,400.,1/60,2,2,.1,1.,100,1.,4.,30.)
    from .compiled_planner import kernel
    for action in (0,1):
        kernel(200.,0.,720.,-253.,136.,np.empty((0,4)),144.,16.,18.,0.,351.,1000.,0.,1/30,2,.12,1.,600,1.,4.,action,True,1/30,True)

def plan(controller,obs,physics,now,age):
    cfg=controller.cfg;start=time.perf_counter();b=obs.bird
    x,y=hitbox_center(b,cfg);hw=cfg['hitbox_half_width'];hh=cfg['hitbox_half_height']
    gates=[gate for gate in obs.gates if gate.right>x-hw][:cfg.get('horizon_gates',3)]
    ceiling=cfg['world_ceiling_px']
    rects=rectangles(gates,cfg,ceiling,cfg['world_floor_px'])
    frame_dt=cfg.get('source_frame_dt',1/120);latency=cfg['latency']+age+cfg.get('source_planner_budget_ms',0)/1000
    action_every=max(cfg.get('source_action_every',4),math.ceil(cfg.get('reaction_interval',1/30)/frame_dt-1e-9))
    # Quantification au prochain tick supposé; phase réelle inconnue.
    delay=math.ceil(latency/frame_dt)*frame_dt
    prefix=float('inf')
    for tick in range(math.ceil(delay/frame_dt)+1):
        elapsed=min(delay,tick*frame_dt)
        py,_=physics.predict(y,now,elapsed)
        prefix=min(prefix,float(frame_margin(float(x),float(py),float(hw),float(hh),rects,float(physics.speed),float(elapsed),float(ceiling),float(cfg['world_floor_px']))))
    y,v=physics.predict(y,now,delay)
    rects=rects.copy();rects[:,:2]-=physics.speed*delay
    horizon=max(.8,(gates[-1].right-x+hw)/physics.speed-delay) if gates else .8
    steps=max(1,math.ceil(min(cfg.get('max_horizon',6),horizon)/frame_dt))
    worst=np.full(2,np.inf)
    if cfg.get('source_search','compensated_maxmin')=='compensated_maxmin':
        from .compiled_planner import kernel
        conservative=cfg.get('source_conservative_envelope',True)
        collision=rects if not conservative else np.asarray([[gate.x-physics.speed*delay,gate.right-physics.speed*delay,gate.top,gate.bottom] for gate in gates],dtype=float).reshape(-1,4)
        # y(nh)=y0+(v0+g*h/2)*nh+g*(nh)^2/2: khớp Euler ở mọi tick.
        # Xét cả đoạn liên tục giữa tick là bao an toàn bảo thủ.
        correction=physics.g*frame_dt/2
        action_dt=action_every*frame_dt
        steps=max(1,math.ceil(min(cfg.get('max_horizon',6),horizon)/action_dt))
        for offset in cfg.get('source_velocity_offsets',[0.]):
            margins=[]
            for action in (0,1):
                _,margin=kernel(float(y),float(v+offset+correction),float(physics.g),float(physics.jump+correction),float(physics.speed),collision,float(x),float(hw),float(hh),float(ceiling),float(cfg['world_floor_px']),1000.,0.,float(action_dt),steps,float(cfg['cooldown']),float(now-controller.last_flap),int(cfg.get('beam_width',600)),float(cfg.get('bin_y',1)),float(cfg.get('bin_v',4)),action,True,float(action_dt),not conservative)
                margins.append(float(margin))
            worst=np.minimum(worst,margins)
        action=bool(worst[1]>worst[0]+.05)
        if action:controller.last_flap=now
        margin=min(prefix,float(worst[int(action)]))
        return Decision(action,[],(gates[0].top+gates[0].bottom)/2 if gates else 200,'SOURCE_COMPENSATED '+('CLICK' if action else 'COAST'),margin>=0,margin,float(worst[0]),float(worst[1]),cfg.get('beam_width',600),prefix)
    for offset in cfg.get('source_velocity_offsets',[0.]):
        _,coast,flap=search(float(y),float(v+offset),float(physics.g),float(physics.jump),float(x),float(hw),float(hh),rects,float(physics.speed),float(ceiling),float(cfg['world_floor_px']),float(frame_dt),steps,int(action_every),float(cfg['cooldown']),float(now-controller.last_flap),int(cfg.get('beam_width',600)),float(cfg.get('bin_y',1)),float(cfg.get('bin_v',4)),float(cfg.get('source_reserve_px',35)))
        worst=np.minimum(worst,[coast,flap])
    action=bool(worst[1]>worst[0]+.05)
    if action:controller.last_flap=now
    return Decision(action,[],(gates[0].top+gates[0].bottom)/2 if gates else obs.frame.shape[0]/2,
                    'SOURCE_EULER '+('CLICK' if action else 'COAST'),min(prefix,max(worst))>=0,min(prefix,float(worst[int(action)])),float(worst[0]),float(worst[1]),cfg.get('beam_width',600),prefix)
