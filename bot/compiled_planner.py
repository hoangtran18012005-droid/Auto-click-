"""Kernel tùy chọn Numba: cùng max-min clearance và beam, không đọc game state."""
import numpy as np
from numba import njit

def warmup():
    kernel(200.,0.,687.4,-264.2,141.,np.empty((0,4)),145.,26.,24.5,0.,356.,100.,.045,1/30,1,.18,1.,200,2.,8.)
    for action in (0,1):
        kernel(200.,0.,687.4,-264.2,141.,np.empty((0,4)),145.,26.,24.5,0.,356.,100.,.045,1/30,1,.18,1.,200,2.,8.,action)
        kernel(200.,0.,687.4,-264.2,141.,np.empty((0,4)),145.,26.,24.5,0.,356.,100.,.045,1/30,1,.18,1.,200,2.,8.,action,True)
    # Khớp đầy đủ kiểu đối số của đường chạy live trước khi click VÀO ĐUA.
    for action in (-1,0,1):
        kernel(200.,0.,687.4,-264.2,141.,np.empty((0,4)),145.,26.,24.5,0.,356.,100.,.045,1/30,1,.18,1.,200,2.,8.,action,False,1/30)

@njit(cache=True)
def bounds(y,v,g,start,end):
    a=y+v*start+.5*g*start*start;b=y+v*end+.5*g*end*end
    apex=min(end,max(start,-v/g));c=y+v*apex+.5*g*apex*apex
    return min(a,b,c),max(a,b,c)

@njit(cache=True)
def kernel(y,v,g,jump,speed,gates,x,hw,hh,ceiling,floor,prefix,latency,dt,steps,cooldown,since,beam,bin_y,bin_v,first_action=-1,rank_suffix=False,first_dt=0.,rect_mode=False):
    # Prefix giống nhau ở mọi nhánh: tránh để nó làm bão hòa xếp hạng beam.
    ys=np.array([y]);vs=np.array([v]);scores=np.array([np.inf if rank_suffix else prefix]);elapsed=np.array([since]);first=np.array([False])
    for step in range(steps):
        duration=max(dt,first_dt) if step==0 else dt
        offset=latency if step==0 else latency+max(dt,first_dt)+(step-1)*dt
        n=len(ys);ny=np.empty(2*n);nv=np.empty(2*n);ns=np.empty(2*n);ne=np.empty(2*n);nf=np.empty(2*n,np.bool_);keys=np.empty(2*n,np.int64);count=0
        for parent in range(n):
            for action in range(2):
                if step==0 and first_action>=0 and action!=first_action:continue
                if action and elapsed[parent]<cooldown-1e-9:continue
                velocity=jump if action else vs[parent]
                low,high=bounds(ys[parent],velocity,g,0,duration)
                margin=min(low-hh-ceiling,floor-high-hh)
                for gate in gates:
                    entry=max(0.,(gate[0]-x-hw)/speed-offset)
                    exit=min(duration,(gate[1]-x+hw)/speed-offset)
                    if entry<=exit:
                        low,high=bounds(ys[parent],velocity,g,entry,exit)
                        if rect_mode:
                            margin=min(margin,max(gate[2]-high-hh,low-hh-gate[3]))
                        else:margin=min(margin,low-hh-gate[2],gate[3]-high-hh)
                ny[count]=ys[parent]+velocity*duration+.5*g*duration*duration;nv[count]=velocity+g*duration
                ns[count]=min(scores[parent],margin);ne[count]=min(cooldown,duration if action else elapsed[parent]+duration)
                nf[count]=bool(action) if step==0 else first[parent]
                ky=int(np.rint(ny[count]/bin_y))+100000;kv=int(np.rint(nv[count]/bin_v))+100000
                ke=int(np.rint(ne[count]/dt))
                keys[count]=((ky*300000+kv)*1000+ke)*2+int(nf[count]);count+=1
        if count==0:return False,-1e6
        order=np.argsort(keys[:count]);chosen=np.empty(count,np.int64);size=0
        for index in order:
            if size==0 or keys[index]!=keys[chosen[size-1]]:
                chosen[size]=index;size+=1
            elif ns[index]>ns[chosen[size-1]]:chosen[size-1]=index
        ranking=np.argsort(-ns[chosen[:size]])[:beam];selected=chosen[ranking]
        ys=ny[selected];vs=nv[selected];scores=ns[selected];elapsed=ne[selected];first=nf[selected]
    index=np.argmax(scores)
    return first[index],min(prefix,scores[index])
