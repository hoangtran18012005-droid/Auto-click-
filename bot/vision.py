"""Image-only detector; coordinates normalized to configured width."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import time
from collections import deque

@dataclass
class Bird:
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0

@dataclass
class Gate:
    x: float
    right: float
    top: float
    bottom: float
    confidence: float = 1.0
    source: str = 'image'

@dataclass
class Observation:
    bird: Bird | None
    gates: list[Gate]
    frame: np.ndarray
    ended: bool = False
    score: int | None = None
    ready: bool = False
    game_over: bool = False
    action_point: tuple | None = None

class Vision:
    def __init__(self, cfg, root):
        cv2.setNumThreads(1)
        self.cfg = cfg
        self.templates = {}
        self.invalid_templates=[]
        for name in ('bird', 'end', 'ready', 'pause', 'restart', 'play'):
            p = Path(root) / 'templates' / f'{name}.png'
            if p.exists():
                template=cv2.imread(str(p))
                if template is None:continue
                if name in ('end','ready','pause') and np.mean(cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)>150)<.1:
                    self.invalid_templates.append(name)
                    continue
                self.templates[name] = template
        self.last_bird = None
        self.modal_scales={}
        for name in ('ready','end','pause'):
            if name in self.templates:
                gray=cv2.cvtColor(self.templates[name],cv2.COLOR_BGR2GRAY)
                self.modal_scales[name]=[cv2.resize(gray,None,fx=s,fy=s) for s in (.99,.98,1.01,1.02,.97,1.03,.96,1.04,.94,1.06,.90,1.10)]
        self.previous_gates=[]
        self.previous_time=None
        self.gate_speed=cfg['pipe_speed']
        self.gate_seen={}
        self.speed_samples=deque(maxlen=15)
        self.digits = []
        for i in range(10):
            p = Path(root) / 'templates' / f'digit-{i}.png'
            if p.exists():
                self.digits.append((i, cv2.imread(str(p), 0)))

    def match(self, frame, name):
        t = self.templates.get(name)
        if t is None or t.shape[0] > frame.shape[0] or t.shape[1] > frame.shape[1]:
            return 0., (0, 0), (0, 0)
        result = cv2.matchTemplate(frame, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, pos = cv2.minMaxLoc(result)
        if score<.82 and name in self.modal_scales:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            best_size=(t.shape[1],t.shape[0])
            for candidate in self.modal_scales[name]:
                if candidate.shape[0]>gray.shape[0] or candidate.shape[1]>gray.shape[1]:continue
                _,quality,_,location=cv2.minMaxLoc(cv2.matchTemplate(gray,candidate,cv2.TM_CCOEFF_NORMED))
                if quality>score:
                    score,pos,best_size=quality,location,(candidate.shape[1],candidate.shape[0])
                if score>.9:break
            return score,pos,best_size
        return score, pos, (t.shape[1], t.shape[0])

    def detect(self, raw, timestamp=None):
        if self.cfg.get('planner')=='source_euler':
            self.gate_speed=self.cfg['pipe_speed']
        now=time.perf_counter() if timestamp is None else timestamp
        width = self.cfg['width']
        frame = cv2.resize(raw, (width, round(raw.shape[0] * width / raw.shape[1])))
        h, w = frame.shape[:2]
        center=frame[round(h*.12):round(h*.85),round(w*.12):round(w*.88)]
        pixels=center[::4,::4].astype(np.int16)
        paper=(pixels[:,:,2]>210)&(pixels[:,:,1]>200)&(pixels[:,:,1]-pixels[:,:,0]>8)&(pixels[:,:,2]-pixels[:,:,0]>12)
        modal=bool(paper.mean()>.20)
        scores={name:self.match(center,name)[0] for name in ('end','ready','pause')} if modal else {}
        winner=max(scores,key=scores.get) if scores and max(scores.values())>.82 else None
        ended=winner=='end'
        ready=winner=='ready'
        paused=winner=='pause'
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bird = None
        if not ended and not ready and not paused and not modal:
            if 'bird' in self.templates and self.cfg['bird_mode'] == 'template':
                quality, (x, y), (bw, bh) = self.match(frame, 'bird')
                if quality >= self.cfg['template_threshold'] and x < w * .45:
                    bird = Bird(x+bw/2, y+bh/2, bw, bh, quality)
            else:
                mask = cv2.inRange(hsv, np.array(self.cfg['bird_hsv_low']), np.array(self.cfg['bird_hsv_high']))
                mask[:, round(w*.43):] = 0
                mask[:round(h*.09)] = 0  # score cards
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                for c in contours:
                    x, y, bw, bh = cv2.boundingRect(c)
                    if 12 < bw < 75 and 10 < bh < 75 and cv2.contourArea(c) > 90 and .08*w < x < .4*w:
                        cx,cy=x+bw/2,y+bh/2
                        if self.cfg.get('body_centroid',False):
                            moments=cv2.moments(c)
                            if moments['m00']:
                                cx=moments['m10']/moments['m00'];cy=moments['m01']/moments['m00']
                        candidates.append(Bird(cx,cy,bw+6,bh+6))
                if candidates:
                    bird = min(candidates, key=lambda b: abs(b.x-w*.24) + (abs(b.y-self.last_bird.y)*.2 if self.last_bird else 0))
                self.last_bird = bird
        # Long vertical shafts distinguish pipes from trees/ground.
        pm = cv2.inRange(hsv, np.array(self.cfg['pipe_hsv_low']), np.array(self.cfg['pipe_hsv_high']))
        pm[round(h*.97):] = 0
        long = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((25, 3), np.uint8))
        contours, _ = cv2.findContours(long, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shafts = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if 18 <= bw <= 105 and bh >= 25 and (y < h*.20 or y+bh > h*.87):
                shafts.append((x, y, bw, bh))
        gates = []
        for x, y, bw, bh in shafts:
            if y > h*.20:
                continue
            # Include lip at both ends; columns from observed artwork.
            center = round(x+bw/2)
            occupied = pm[:, max(0, center-3):center+4].mean(axis=1) > 100
            top = y+bh
            while top < h and occupied[top]:
                top += 1
            # Trees share the pipe hue. A pipe lip spans almost the complete
            # gate width; a center-column green pixel is insufficient.
            lip_left=max(0,x-10);lip_right=min(w,x+bw+10)
            lip=pm[:,lip_left:lip_right].mean(axis=1)>255*.85
            rows=np.flatnonzero(lip[min(h,top+35):])
            bottom=top+35+int(rows[0]) if len(rows) else h
            if bottom>h*.86:
                lower=[sy for sx,sy,sw,sh in shafts if sy>top+35 and abs(sx+sw/2-center)<10]
                if lower:bottom=min(lower)
            if bottom-top<40 or bottom>h*.86:
                continue
            lip=self.cfg.get('pipe_lip_side_px',12)
            left=x-lip if self.cfg.get('planner')=='source_euler' else max(0,x-lip)
            right=left+self.cfg['gate_width_px'] if self.cfg.get('planner')=='source_euler' else min(w,x+bw+lip)
            if self.cfg.get('planner')=='source_euler' and x==0:
                right=min(w,x+bw+lip)
                left=right-self.cfg['gate_width_px']
            gates.append(Gate(left,right,top,bottom))
        gates.sort(key=lambda g:g.x)
        if self.cfg.get('reconstruct_entering_gate',False):
            expected_width=self.cfg.get('gate_width_px',86)
            gates=[Gate(g.x,g.x+expected_width,g.top,g.bottom,.6,'reconstructed_entering')
                   if g.right>=w-1 and g.right-g.x<expected_width*.95 else g for g in gates]
        # HUD occludes a top pipe when it is beside the bird. Preserve its
        # complete collision box briefly using image-derived motion.
        if not modal and self.previous_time is not None and 0<now-self.previous_time<.2:
            dt=now-self.previous_time
            completed=[];used=set();seen={}
            for i,prev in enumerate(self.previous_gates):
                predicted=Gate(prev.x-self.gate_speed*dt,prev.right-self.gate_speed*dt,prev.top,prev.bottom)
                matches=[(abs(g.right-predicted.right)+abs(g.top-prev.top),j,g) for j,g in enumerate(gates) if j not in used and abs(g.right-predicted.right)<25 and abs(g.top-prev.top)<15]
                if matches:
                    _,j,g=min(matches,key=lambda v:v[0]);used.add(j)
                    if g.right-g.x<(prev.right-prev.x)*.8:
                        g=predicted
                        if self.cfg.get('track_gates',False):g=Gate(g.x,g.right,g.top,g.bottom,.65,'tracked_partial')
                    else:
                        measured=(prev.right-g.right)/dt
                        if 40<measured<350:
                            if self.cfg.get('planner')=='source_euler':
                                self.gate_speed=self.cfg['pipe_speed']
                            elif self.cfg.get('track_gates',False):
                                self.speed_samples.append(measured)
                                self.gate_speed=float(np.median(self.speed_samples))
                            else:self.gate_speed=.9*self.gate_speed+.1*measured
                        if g.right-g.x<prev.right-prev.x:
                            g=Gate(g.right-(prev.right-prev.x),g.right,g.top,g.bottom)
                    completed.append(g);seen[len(completed)-1]=now
                elif now-self.gate_seen.get(i,now)<.3 and predicted.right>(bird.x-bird.w/2-10 if bird else 0):
                    if self.cfg.get('track_gates',False):predicted=Gate(predicted.x,predicted.right,predicted.top,predicted.bottom,.5,'tracked_occlusion')
                    completed.append(predicted);seen[len(completed)-1]=self.gate_seen.get(i,now)
            for j,g in enumerate(gates):
                if j not in used:completed.append(g);seen[len(completed)-1]=now
            order=sorted(range(len(completed)),key=lambda i:completed[i].x)
            gates=[completed[i] for i in order];self.gate_seen={j:seen[i] for j,i in enumerate(order)}
        else:
            self.gate_seen={i:now for i in range(len(gates))}
        self.previous_gates=gates if not modal else []
        if self.cfg.get('track_gates',False) and not modal:
            expected_width=self.cfg.get('gate_width_px',86)
            reliable=[];reliable_seen={}
            for index,gate in enumerate(gates):
                width=gate.right-gate.x
                if gate.source=='image' and width<expected_width*.7:
                    # Không nhận một phần cột bị HUD che là toàn bộ cột.
                    if gate.x<30:gate=Gate(gate.right-expected_width,gate.right,gate.top,gate.bottom,.6,'reconstructed_edge')
                    else:continue
                reliable_seen[len(reliable)]=self.gate_seen.get(index,now)
                reliable.append(gate)
            gates=reliable
            self.previous_gates=gates
            self.gate_seen=reliable_seen
        self.previous_time=now
        point=self.modal_action(frame) if ready or ended else None
        return Observation(bird, gates, frame, ended or ready or paused or modal, self.read_score(frame),ready,ended,point)

    def modal_action(self,frame):
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        yellow=cv2.inRange(hsv,np.array([18,110,170]),np.array([38,255,255]))
        contours,_=cv2.findContours(yellow,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        h,w=frame.shape[:2];buttons=[]
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            if .20*w<bw<.65*w and 18<bh<75 and x+bw/2>.52*w and y+bh/2>.25*h and cv2.contourArea(contour)>.65*bw*bh:
                buttons.append((bw*bh,(x+bw/2,y+bh/2)))
        return max(buttons,key=lambda item:item[0])[1] if buttons else None

    def read_score(self, frame):
        if len(self.digits) != 10:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # UI score: small card at top left, do not read the best-score card.
        roi = gray[20:55, 12:120]
        hits = []
        for digit, t in self.digits:
            if t.shape[0] > roi.shape[0] or t.shape[1] > roi.shape[1]:
                continue
            r = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
            yy, xx = np.where(r>.85)
            for y, x in zip(yy, xx):
                hits.append((float(r[y,x]), int(x), digit, t.shape[1]))
        selected=[]
        for hit in sorted(hits, reverse=True):
            if all(abs(hit[1]-s[1]) > max(hit[3],s[3])*.6 for s in selected):
                selected.append(hit)
        if not selected:
            return None
        return int(''.join(str(s[2]) for s in sorted(selected,key=lambda s:s[1])))
