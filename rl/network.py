"""MLP, Huber và Adam bằng NumPy: không thêm thư viện vào bot Windows."""
import json
from pathlib import Path
import numpy as np
from .env import STATE_SIZE,SCHEMA

class Network:
    def __init__(self,seed=0,hidden=64):
        self.hidden=hidden;rng=np.random.default_rng(seed)
        self.params={}
        for i,(a,b) in enumerate(((STATE_SIZE,hidden),(hidden,hidden),(hidden,2))):
            self.params[f'w{i}']=(rng.normal(size=(a,b))*np.sqrt(2/a)).astype(np.float32)
            self.params[f'b{i}']=np.zeros(b,np.float32)
        self.m={k:np.zeros_like(v) for k,v in self.params.items()};self.v={k:np.zeros_like(v) for k,v in self.params.items()};self.updates=0
    def copy_from(self,other):
        for k in self.params:self.params[k][...]=other.params[k]
    def expanded(self,hidden,seed=0):
        """Tăng số neuron mà vẫn giữ chính xác hàm Q của checkpoint cũ."""
        if hidden<=self.hidden:raise ValueError('Hidden mới phải lớn hơn hidden cũ.')
        out=Network(seed,hidden);old=self.hidden
        out.params['w0'][:,:old]=self.params['w0'];out.params['b0'][:old]=self.params['b0']
        out.params['w1'][:old,:old]=self.params['w1'];out.params['w1'][old:,:old]=0
        out.params['b1'][:old]=self.params['b1']
        out.params['w2'][:old]=self.params['w2'];out.params['w2'][old:]=0
        out.params['b2'][...]=self.params['b2']
        return out
    def forward(self,x,cache=False):
        x=np.asarray(x,np.float32);z1=x@self.params['w0']+self.params['b0'];a1=np.maximum(z1,0)
        z2=a1@self.params['w1']+self.params['b1'];a2=np.maximum(z2,0);q=a2@self.params['w2']+self.params['b2']
        return (q,(x,z1,a1,z2,a2)) if cache else q
    def fit(self,x,actions,targets,lr=3e-4):
        q,(x,z1,a1,z2,a2)=self.forward(x,True);n=len(x);error=q[np.arange(n),actions]-targets
        loss=np.mean(np.where(abs(error)<=1,.5*error**2,abs(error)-.5))
        dq=np.zeros_like(q);dq[np.arange(n),actions]=np.clip(error,-1,1)/n
        dz2=(dq@self.params['w2'].T)*(z2>0);dz1=(dz2@self.params['w1'].T)*(z1>0)
        grads=dict(w2=a2.T@dq,b2=dq.sum(0),w1=a1.T@dz2,b1=dz2.sum(0),w0=x.T@dz1,b0=dz1.sum(0))
        self._apply_gradients(grads,lr)
        return float(loss)
    def _apply_gradients(self,grads,lr):
        norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads.values()));scale=min(1.,10/max(norm,1e-8))
        self.updates+=1;t=self.updates
        for k,g in grads.items():
            g=g*scale;self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            self.params[k]-=lr*(self.m[k]/(1-.9**t))/(np.sqrt(self.v[k]/(1-.999**t))+1e-8)
    def _backward(self,dq,cache):
        x,z1,a1,z2,a2=cache
        dz2=(dq@self.params['w2'].T)*(z2>0);dz1=(dz2@self.params['w1'].T)*(z1>0)
        return dict(w2=a2.T@dq,b2=dq.sum(0),w1=a1.T@dz2,b1=dz2.sum(0),w0=x.T@dz1,b0=dz1.sum(0))
    def fit_actions(self,x,actions,weights,lr=1e-5,margin=.05,anchors=None,anchor_values=None,anchor_weight=1.):
        """Học thứ tự Q theo nhãn planner; neo Q cũ ở trạng thái thông thường."""
        q,cache=self.forward(x,True);n=len(q);rows=np.arange(n);other=1-actions
        weights=np.asarray(weights,np.float32)
        if margin<=0 or np.any(weights<0) or weights.sum()<=0:raise ValueError('Margin/trọng số không hợp lệ.')
        error=np.maximum(0,margin+q[rows,other]-q[rows,actions]);weight=weights/weights.sum()
        loss=float(np.sum(.5*weight*error**2));dq=np.zeros_like(q)
        dq[rows,other]=weight*error;dq[rows,actions]=-weight*error
        grads=self._backward(dq,cache)
        if anchors is not None and len(anchors):
            aq,acache=self.forward(anchors,True);residual=aq-anchor_values
            loss+=float(anchor_weight*.5*np.mean(residual**2))
            ag=self._backward(anchor_weight*residual/residual.size,acache)
            grads={k:g+ag[k] for k,g in grads.items()}
        self._apply_gradients(grads,lr)
        return loss
    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        arrays=dict(self.params)
        arrays.update({f'm_{k}':v for k,v in self.m.items()});arrays.update({f'v_{k}':v for k,v in self.v.items()})
        arrays['metadata']=np.array(json.dumps(dict(metadata or {},schema=SCHEMA,hidden=self.hidden,updates=self.updates)))
        temp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(temp,**arrays);temp.replace(path)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta.get('schema')!=SCHEMA:raise ValueError('Checkpoint không cùng schema state.')
            net=cls(hidden=meta['hidden']);net.updates=meta['updates']
            for k in net.params:
                net.params[k][...]=data[k];net.m[k][...]=data[f'm_{k}'];net.v[k][...]=data[f'v_{k}']
        return net,meta

def double_targets(online,target,next_states,rewards,terminated,discounts,next_masks):
    q=online.forward(next_states);chosen=np.argmax(np.where(next_masks,q,-1e9),axis=1)
    values=target.forward(next_states)[np.arange(len(chosen)),chosen]
    return rewards+discounts*(~terminated)*values

class Replay:
    def __init__(self,capacity=100000):
        self.capacity=capacity;self.size=0;self.pos=0
        self.states=np.empty((capacity,STATE_SIZE),np.float32);self.next_states=np.empty_like(self.states)
        self.actions=np.empty(capacity,np.int64);self.rewards=np.empty(capacity,np.float32)
        self.terminated=np.empty(capacity,bool);self.discounts=np.empty(capacity,np.float32);self.masks=np.empty((capacity,2),bool)
    def add(self,state,action,reward,next_state,terminated,discount,mask):
        i=self.pos;self.states[i]=state;self.actions[i]=action;self.rewards[i]=reward;self.next_states[i]=next_state
        self.terminated[i]=terminated;self.discounts[i]=discount;self.masks[i]=mask
        self.pos=(i+1)%self.capacity;self.size=min(self.size+1,self.capacity)
    def sample(self,n,rng):
        indices=rng.integers(0,self.size,n)
        return tuple(a[indices] for a in (self.states,self.actions,self.rewards,self.next_states,self.terminated,self.discounts,self.masks))

def mixed_batch(regular,hard,count,fraction,rng):
    """Trộn replay thông thường và tình huống trước va chạm; giữ nguyên transition."""
    if count<1 or not 0<=fraction<=1:raise ValueError('Cỡ batch/tỷ lệ không hợp lệ.')
    difficult_count=int(count*fraction) if hard.size else 0
    regular_count=count-difficult_count
    if difficult_count and regular_count:
        a=regular.sample(regular_count,rng);b=hard.sample(difficult_count,rng)
        return tuple(np.concatenate((x,y)) for x,y in zip(a,b))
    return (hard if difficult_count else regular).sample(count,rng)
