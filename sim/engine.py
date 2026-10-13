"""Vật lý liên tục, kiểm tra va chạm trong khoảng thời gian giữa hai quan sát."""
from dataclasses import dataclass, asdict, replace
import heapq
import numpy as np
from bot.physics import project
from bot.vision import Bird, Gate, Observation


@dataclass(frozen=True)
class SimulationConfig:
    gravity: float = 687.4
    jump: float = -264.2
    speed: float = 141.
    latency: float = .045
    decision_hz: float = 30.
    noise_y: float = 1.5
    noise_v: float = 0.
    latency_jitter: float = 0.
    parameter_error: float = 0.
    input_drop_probability: float = 0.
    bird_x: float = 145.
    bird_width: float = 52.
    bird_height: float = 49.
    collision_half_width: float | None = None
    collision_half_height: float | None = None
    collision_center_offset_y: float = 0.
    ceiling: float = 0.
    floor: float = 356.
    cooldown: float = .18


def transition_type(previous, following, tolerance=15.):
    if previous is None or following is None:
        return 'unknown'
    change=(following.top+following.bottom-previous.top-previous.bottom)/2
    return 'high_to_low' if change>tolerance else 'low_to_high' if change < -tolerance else 'similar'


def interval_clearance(y, velocity, gravity, start, end, low, high):
    """Khoảng cách nhỏ nhất tới hai biên, gồm cả đỉnh parabol."""
    times=[start,end]
    apex=-velocity/gravity
    if start<apex<end:times.append(apex)
    values=[project(y,velocity,gravity,t)[0] for t in times]
    return min(min(values)-low,high-max(values))


class Simulator:
    def __init__(self, gates, config=None, seed=0, y=200., velocity=0.):
        self.config=config or SimulationConfig()
        self.gates=list(gates)
        self.rng=np.random.default_rng(seed)
        self.latency_rng=np.random.default_rng(seed+104729)
        self.dropout_rng=np.random.default_rng(seed+32452843)
        self.time=0.;self.y=y;self.velocity=velocity
        self.pending=[];self.last_command=-1e9;self.serial=0
        self.dead=False;self.death_gate=None;self.death_transition='unknown'
        self.minimum_clearance=float('inf')
        self.frame=np.zeros((400,600,3),np.uint8)

    def observe(self):
        c=self.config
        gates=[Gate(g.x-c.speed*self.time,g.right-c.speed*self.time,g.top,g.bottom)
               for g in self.gates if g.right-c.speed*self.time>c.bird_x-c.bird_width/2-10
               and g.x-c.speed*self.time<600]
        return Observation(Bird(c.bird_x,self.y+self.rng.normal(0,c.noise_y),c.bird_width,c.bird_height),gates,self.frame)

    def command(self, flap):
        if not flap or self.dead or self.time-self.last_command<self.config.cooldown-1e-9:
            return False
        self.last_command=self.time;self.serial+=1
        if self.config.input_drop_probability and self.dropout_rng.random()<self.config.input_drop_probability:return True
        delay=max(0,self.config.latency+self.latency_rng.normal(0,self.config.latency_jitter))
        heapq.heappush(self.pending,(self.time+delay,self.serial))
        return True

    def clearance(self, duration):
        c=self.config;hh=c.bird_height/2;hw=c.bird_width/2
        result=interval_clearance(self.y,self.velocity,c.gravity,0,duration,c.ceiling+hh,c.floor-hh)
        collision_hw=c.collision_half_width if c.collision_half_width is not None else hw
        collision_hh=c.collision_half_height if c.collision_half_height is not None else hh
        collision_y=self.y+c.collision_center_offset_y
        culprit=None
        for index,g in enumerate(self.gates):
            left=g.x-c.speed*self.time;right=g.right-c.speed*self.time
            entry=max(0.,(left-c.bird_x-collision_hw)/c.speed)
            exit=min(duration,(right-c.bird_x+collision_hw)/c.speed)
            if entry>exit+1e-10 or exit<0:continue
            margin=interval_clearance(collision_y,self.velocity,c.gravity,entry,exit,
                                     g.top+collision_hh,g.bottom-collision_hh)
            if margin<result:result=margin;culprit=index
        return result,culprit

    def _advance_segment(self, duration):
        if duration<=1e-12:return
        clearance,culprit=self.clearance(duration)
        if clearance<0:
            lo=0.;hi=duration
            for _ in range(35):
                mid=(lo+hi)/2
                if self.clearance(mid)[0]<0:hi=mid
                else:lo=mid
            duration=hi
            clearance,culprit=self.clearance(duration)
            self.dead=True;self.death_gate=culprit
            self.death_transition=transition_type(self.gates[culprit-1] if culprit is not None and culprit>0 else None,
                                                 self.gates[culprit] if culprit is not None else None)
        self.minimum_clearance=min(self.minimum_clearance,clearance)
        self.y,self.velocity=project(self.y,self.velocity,self.config.gravity,duration)
        self.time+=duration

    def advance(self, duration):
        end=self.time+duration
        while self.pending and self.pending[0][0]<=end+1e-12 and not self.dead:
            when,_=heapq.heappop(self.pending)
            self._advance_segment(max(0,when-self.time))
            if not self.dead:self.velocity=self.config.jump  # SET, không cộng lực.
        if not self.dead:self._advance_segment(end-self.time)

    @property
    def score(self):
        c=self.config
        return int(sum(g.right-c.speed*self.time<c.bird_x-c.bird_width/2 for g in self.gates))


def random_sequence(seed, count=60, first_x=550., jump_limit=230., floor=356.):
    rng=np.random.default_rng(seed);center=200.;gates=[]
    for index in range(count):
        center=float(np.clip(center+rng.uniform(-jump_limit,jump_limit),60,floor-60))
        left=first_x+195*index
        gates.append(Gate(left,left+86,center-60,center+60))
    return gates


def source_random_sequence(seed,count=60,frame_width=600,canvas_width=900,
                          ground_y=535,gap=180,pipe_speed=205,
                          spawn_interval=1.42,pipe_width=92,lip_side=16):
    """Generate desktop gates from the supplied game's native spawn rules."""
    rng=np.random.default_rng(seed);scale=frame_width/canvas_width
    top_max=ground_y-gap-90
    if top_max<=90:raise ValueError('Dải sinh khe source rỗng.')
    gates=[]
    for index in range(count):
        top=float(rng.uniform(90,top_max))
        native_x=canvas_width+20+pipe_speed*spawn_interval*index
        left=(native_x-lip_side)*scale
        right=(native_x+pipe_width+lip_side)*scale
        gates.append(Gate(left,right,top*scale,(top+gap)*scale))
    return gates


def run_episode(controller, gates, config=None, seed=0, max_points=50):
    import time
    from bot.physics import Physics
    c=config or SimulationConfig()
    errors=np.random.default_rng(seed+99991).uniform(-c.parameter_error,c.parameter_error,3)
    actual=replace(c,gravity=c.gravity*(1+errors[0]),jump=c.jump*(1+errors[1]),speed=c.speed*(1+errors[2]))
    sim=Simulator(gates,actual,seed)
    velocity_rng=np.random.default_rng(seed+70001)
    model=Physics(dict(controller.cfg,**{'gravity':controller.cfg.get('gravity',c.gravity) if controller.cfg.get('freeze_physics') else c.gravity,
                   'jump':controller.cfg.get('jump',c.jump) if controller.cfg.get('freeze_physics') else c.jump,
                   'pipe_speed':controller.cfg.get('pipe_speed',c.speed) if controller.cfg.get('freeze_physics') else c.speed}))
    times=[];flaps=0;expected_pending=[]
    limit=(gates[-1].right-c.bird_x+c.bird_width/2)/c.speed+1
    while not sim.dead and sim.time<limit and sim.score<max_points:
        observation=sim.observe()
        if controller.cfg.get('simulation_observer',False):
            model.update(observation.bird,observation.gates,sim.time)
        else:model.v=sim.velocity
        model.v+=velocity_rng.normal(0,c.noise_v)
        if controller.cfg.get('pending_actions',False) and not controller.cfg.get('simulation_observer',False):
            expected_pending=[event for event in expected_pending if event>sim.time]
            model.pending=expected_pending[:]
        controller.cfg['reaction_interval']=1/c.decision_hz
        start=time.perf_counter()
        decision=controller.decide(observation,model,sim.time)
        times.append(1000*(time.perf_counter()-start))
        issued=sim.command(decision.flap);flaps+=issued
        if issued:
            if controller.cfg.get('simulation_observer',False):model.flap(sim.time)
            else:expected_pending.append(sim.time+controller.cfg.get('input_effect_latency',c.latency))
        sim.advance(1/c.decision_hz)
    if hasattr(controller,'close'):controller.close()
    return {'score':sim.score,'duration_s':sim.time,'death_transition':sim.death_transition if sim.dead else 'survived',
            'flaps':flaps,'controller_ms':times,'model':asdict(c),'seed':seed}
