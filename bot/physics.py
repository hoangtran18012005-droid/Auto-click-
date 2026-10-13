from collections import deque
import numpy as np

def project(y, v, g, dt):
    return y + v*dt + .5*g*dt*dt, v+g*dt

class Physics:
    def __init__(self, cfg):
        if cfg.get('planner')=='source_euler':
            from .source_model import configure
            configure(cfg)
        self.cfg=cfg
        self.pending=[]
        self.g = cfg['gravity']
        self.jump = cfg['jump']
        self.speed = cfg['pipe_speed']
        self.initial_g=self.g;self.initial_jump=self.jump
        self.v = 0.
        self.points = deque(maxlen=40)
        self.last_gate = None
        self.last_flap = -1e9
        self.samples = 0
        self.commands=[]
        self.input_events=[]
        self.input_status='idle'
        self.observer_state=None
        self.last_command_effect=None
        self.observed_command_effect=None
        self.prediction_residual=None

    def flap(self, now):
        self.last_command_effect=now+self.cfg.get('input_effect_latency',.045)
        if self.cfg.get('confirm_input',False) and self.points:
            observed,y=self.points[-1]
            self.commands.append({'sent':now,'expected':now+self.cfg.get('input_effect_latency',.045),'t0':observed,'y0':y,'v0':self.v,'samples':[]})
            self.input_status='pending'
        if self.cfg.get('pending_actions',False):
            self.pending.append(now+self.cfg.get('input_effect_latency',.045))
            return
        self.last_flap = now
        self.v = self.jump
        self.points.clear()

    def update(self, bird, gates, now):
        if self.points:
            previous_time,previous_y=self.points[-1]
            if 0<now-previous_time<.25:
                expected,_=self.predict(previous_y,previous_time,now-previous_time)
                self.prediction_residual=bird.y-expected
        previous_velocity=self.v
        confirmed=False
        if self.cfg.get('confirm_input',False):
            remaining=[]
            for command in self.commands:
                if now<=command['sent']:
                    remaining.append(command);continue
                command['samples'].append((now,bird.y))
                observations=command['samples'][-5:]
                times=np.array([t for t,y in observations]);ys=np.array([y for t,y in observations])
                coast=project(command['y0'],command['v0'],self.g,times-command['t0'])[0]
                coast_error=float(np.sqrt(np.mean((ys-coast)**2)))
                best_error=float('inf');best_time=None
                for delay in np.arange(self.cfg.get('input_latency_min',.016),self.cfg.get('input_latency_max',.10)+.001,.004):
                    event=command['sent']+delay
                    if event>=now-.006:continue
                    before=project(command['y0'],command['v0'],self.g,event-command['t0'])[0]
                    trajectory=np.where(times<event,coast,project(before,self.jump,self.g,np.maximum(0,times-event))[0])
                    error=float(np.sqrt(np.mean((ys-trajectory)**2)))
                    if error<best_error:best_error=error;best_time=event
                if len(observations)>=2 and best_time is not None and coast_error-best_error>=self.cfg.get('input_confirmation_px',2.5) and best_error<=self.cfg.get('input_confirmation_rmse',4.):
                    self.last_flap=best_time;self.points.clear()
                    self.v=self.jump+self.g*(now-best_time);confirmed=True
                    self.input_status='confirmed'
                    self.input_events.append({'sent':command['sent'],'observed':now,'effect_estimate':best_time,'status':'confirmed','coast_rmse':coast_error,'jump_rmse':best_error})
                elif now-command['sent']>=self.cfg.get('input_confirmation_timeout',.16):
                    self.input_status='unconfirmed'
                    self.input_events.append({'sent':command['sent'],'observed':now,'status':'unconfirmed','coast_rmse':coast_error,'jump_rmse':best_error if np.isfinite(best_error) else None})
                else:remaining.append(command)
            self.commands=remaining
        if self.cfg.get('pending_actions',False):
            due=[event for event in self.pending if event<=now]
            self.pending=[event for event in self.pending if event>now]
            if due and not self.cfg.get('confirm_input',False) and not self.cfg.get('image_impulse_reset',False):
                self.last_flap=max(due);self.points.clear()
        if self.cfg.get('image_impulse_reset',False) and self.points and self.last_command_effect is not None and self.observed_command_effect!=self.last_command_effect:
            then,position=self.points[-1];dt=now-then
            if .008<dt<.15 and -.08<now-self.last_command_effect<.20:
                measured=(bird.y-position)/dt+.5*self.g*dt
                if measured < self.v-self.cfg.get('impulse_velocity_drop',120):
                    self.last_flap=min(now,self.last_command_effect)
                    self.observed_command_effect=self.last_command_effect
                    self.points.clear()
                    self.input_status='image_impulse'
                    self.input_events.append({'observed':now,'effect_estimate':self.last_flap,'status':'image_impulse'})
        self.points.append((now, bird.y))
        points = list(self.points)
        # Trọng lực vẫn tác động trong các frame chưa đủ 40 ms để fit coast.
        if (len(points)==1 or self.cfg.get('advance_sparse_velocity',False)) and 0<=now-self.last_flap<1:
            self.v=self.jump+self.g*(now-self.last_flap)
        if len(points) >= 2 and points[-1][0]-points[0][0]>.04:
            times = np.array([p[0]-now for p in points])
            ys = np.array([p[1] for p in points])
            # Fixed-g fit estimates velocity without amplifying pixel noise.
            v, _ = np.linalg.lstsq(np.column_stack((times,np.ones(len(times)))),ys-.5*self.g*times**2,rcond=None)[0]
            self.v = float(np.clip(v, -700, 1000))
            if self.cfg.get('planner')=='source_euler':
                self.v-=.5*self.g*self.cfg.get('source_frame_dt',1/120)
                if 0<=now-self.last_flap<.8:
                    prior=self.jump+self.g*(now-self.last_flap)
                    if abs(self.v-prior)<self.cfg.get('source_velocity_innovation_limit',80):
                        weight=self.cfg.get('source_velocity_prior_weight',.7)
                        self.v=(1-weight)*self.v+weight*prior
                    else:self.input_status='source_velocity_mismatch'
            if not self.cfg.get('freeze_physics',False) and 0<now-self.last_flap<.7:
                estimate=self.v-self.g*(now-self.last_flap)
                if self.initial_jump*1.15<estimate<self.initial_jump*.85:
                    self.jump=.98*self.jump+.02*estimate
            if not self.cfg.get('freeze_physics',False) and len(points)>=10 and times[-1]-times[0]>.4:
                fitted=np.polyfit(times,ys,2);a=fitted[0]
                residual=float(np.std(ys-np.polyval(fitted,times)))
                if self.initial_g*.85<2*a<self.initial_g*1.15 and residual<2:
                    self.g = .98*self.g+.02*float(2*a)
                    self.samples += 1
        if self.cfg.get('state_observer',False):
            if self.observer_state is None or confirmed:
                self.observer_state=(now,bird.y,self.v)
            else:
                then,position,velocity=self.observer_state
                dt=now-then
                if 0<dt<.2:
                    predicted,predicted_v=project(position,velocity,self.g,dt)
                    residual=float(np.clip(bird.y-predicted,-12,12))
                    position=predicted+self.cfg.get('observer_alpha',.75)*residual
                    velocity=float(np.clip(predicted_v+self.cfg.get('observer_beta',.08)*residual/dt,-700,1000))
                    self.observer_state=(now,position,velocity)
                    self.v=velocity
                else:self.observer_state=(now,bird.y,previous_velocity)
        if gates:
            current = gates[0]
            if self.last_gate:
                prev, then = self.last_gate
                dt=now-then
                dx=prev-current.right
                if not self.cfg.get('freeze_physics',False) and .005<dt<.5 and 0<dx<150 and 30<dx/dt<500:
                    self.speed=.97*self.speed+.03*dx/dt
            self.last_gate=(current.right,now)
        else:
            self.last_gate=None

    def prediction_segments(self,y,now,duration):
        start=0.;velocity=self.v;segments=[]
        for event in sorted(self.pending):
            offset=event-now
            if not 0<offset<duration:continue
            segments.append((start,offset,y,velocity))
            y,_=self.project(y,velocity,offset-start)
            velocity=self.jump;start=offset
        segments.append((start,duration,y,velocity))
        return segments

    def predict(self,y,now,duration):
        start,end,y,velocity=self.prediction_segments(y,now,duration)[-1]
        return self.project(y,velocity,end-start)

    def project(self,y,v,duration):
        if self.cfg.get('planner')=='source_euler':
            from .source_model import predict
            return predict(y,v,self.g,duration,self.cfg.get('source_frame_dt',1/120))
        return project(y,v,self.g,duration)
