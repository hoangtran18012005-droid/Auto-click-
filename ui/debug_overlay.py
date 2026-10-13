import cv2

def draw(obs, decision, physics, fps):
    f=obs.frame.copy()
    b=obs.bird
    if b:
        cv2.rectangle(f,(int(b.x-b.w/2),int(b.y-b.h/2)),(int(b.x+b.w/2),int(b.y+b.h/2)),(0,200,255),2)
    for g in obs.gates:
        cv2.rectangle(f,(int(g.x),int(g.top)),(int(g.right),int(g.bottom)),(255,180,0),2)
        cv2.line(f,(int(g.x),int((g.top+g.bottom)/2)),(int(g.right),int((g.top+g.bottom)/2)),(255,255,255),1)
    for a,bp in zip(decision.path,decision.path[1:]):
        cv2.line(f,tuple(map(int,a)),tuple(map(int,bp)),(0,0,255),2)
    cv2.putText(f,f'{fps:.1f} FPS | {decision.reason}',(10,75),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),2)
    cv2.putText(f,f'v={physics.v:.0f} g={physics.g:.0f} speed={physics.speed:.0f}',(10,95),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
    return f
