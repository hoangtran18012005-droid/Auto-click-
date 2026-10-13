import numpy as np
import pytest
from rl.env import FlappyEnv,EnvConfig,STATE_SIZE
from rl.network import Network,Replay,double_targets

def test_state_seed_and_no_hidden_future_leak():
    a=FlappyEnv();b=FlappyEnv();s,_=a.reset(13);t,_=b.reset(13)
    assert s.shape==(STATE_SIZE,) and np.array_equal(s,t)
    assert np.all(s[8:20]==0)
    a.plan[-1]['top']=200
    assert np.array_equal(a.state(),b.state())

def test_single_flap_queue_and_cooldown_mask():
    env=FlappyEnv();env.reset();_,_,_,_,info=env.step(1)
    assert info['executed_action']==1 and not info['action_mask'][1]
    _,_,_,_,info=env.step(1)
    assert info['executed_action']==0
    assert env.sim.v<0

def test_truncation_is_not_death():
    env=FlappyEnv(EnvConfig(max_steps=1));env.reset();_,_,done,truncated,info=env.step(0)
    assert not done and truncated
    with pytest.raises(RuntimeError):env.step(0)

def test_double_target_uses_online_choice_and_target_value_with_mask():
    class Fake:
        def __init__(self,values):self.values=np.asarray(values,np.float32)
        def forward(self,x):return self.values
    online=Fake([[2,1],[2,3],[8,7]]);target=Fake([[10,100],[50,20],[30,40]])
    result=double_targets(online,target,np.zeros((3,STATE_SIZE)),np.array([1,1,1]),np.array([False,False,True]),np.array([.5,.5,.5]),np.array([[True,True],[True,False],[True,True]]))
    assert result==pytest.approx([6,26,1])

def test_network_learns_fixed_targets_and_checkpoint_roundtrips(tmp_path):
    net=Network(3,32);x=np.random.default_rng(4).normal(size=(32,STATE_SIZE)).astype(np.float32)
    a=np.arange(32)%2;y=np.full(32,.5,np.float32)
    before=np.mean((net.forward(x)[np.arange(32),a]-y)**2)
    for _ in range(150):net.fit(x,a,y,.001)
    after=np.mean((net.forward(x)[np.arange(32),a]-y)**2)
    assert after<before*.1
    path=tmp_path/'model.npz';net.save(path,dict(steps=150));restored,meta=Network.load(path)
    assert np.array_equal(restored.forward(x),net.forward(x)) and meta['steps']==150

def test_replay_ring_overwrites_oldest_and_keeps_terminal_flag():
    replay=Replay(2)
    for i in range(3):replay.add(np.full(STATE_SIZE,i),0,i,np.zeros(STATE_SIZE),i==2,.995,[True,False])
    assert replay.size==2 and sorted(replay.rewards.tolist())==[1,2]
    assert replay.terminated[0]

def test_rl_oracle_does_not_drift_from_source_simulator_before_pipe_entry():
    from sim.source_engine import SourceSimulator,sequence
    env=FlappyEnv();env.reset(1);oracle=SourceSimulator(sequence(1,104),535)
    for i in range(10):
        action=int(i==0)
        if action:oracle.flap()
        oracle.elapsed(1/30);env.step(action)
        assert env.sim.y==pytest.approx(oracle.y) and env.sim.v==pytest.approx(oracle.v)

@pytest.mark.parametrize('y,kind',[(464,'ground'),(-11,'ceiling')])
def test_terminal_log_identifies_sprite_boundary(y,kind):
    env=FlappyEnv();env.reset();env.sim.y=y;env.sim.dead=True
    assert env.info()['failure']['kind']==kind

def test_terminal_log_identifies_lip_collision_outside_body():
    env=FlappyEnv();env.reset();env.sim.y=130;env.sim.tick=0;env.sim.dead=True
    env.sim.pipes=[dict(x=250,top=170,bottom=350)]
    assert env.info()['failure']['kind']=='top_lip'

@pytest.mark.parametrize('config',[EnvConfig(latency=-1),EnvConfig(noise_y=float('nan')),EnvConfig(frame_drop_probability=1.1),EnvConfig(gamma=0)])
def test_invalid_environment_is_rejected(config):
    with pytest.raises(ValueError):FlappyEnv(config)

def test_mixed_replay_keeps_fraction_rewards_terminal_and_action_masks():
    from rl.network import mixed_batch
    regular=Replay(4);hard=Replay(4)
    regular.add(np.zeros(STATE_SIZE),0,1,np.zeros(STATE_SIZE),False,.995,[True,True])
    hard.add(np.ones(STATE_SIZE),1,-2,np.ones(STATE_SIZE),True,.99,[True,False])
    s,a,r,ns,done,d,mask=mixed_batch(regular,hard,16,.25,np.random.default_rng(1))
    assert np.sum(s[:,0]==1)==4 and np.sum(done)==4
    assert np.all(a[done]==1) and np.all(r[done]==-2)
    assert np.all(d[done]==np.float32(.99)) and np.all(mask[done,1]==False)

def test_mixed_replay_falls_back_before_first_pipe_failure():
    from rl.network import mixed_batch
    regular=Replay(2);hard=Replay(2)
    regular.add(np.zeros(STATE_SIZE),0,1,np.zeros(STATE_SIZE),False,.995,[True,True])
    batch=mixed_batch(regular,hard,8,1,np.random.default_rng(1))
    assert len(batch[0])==8 and not np.any(batch[4])

def test_planner_imitation_margin_updates_selected_action_in_right_direction():
    net=Network(3,16);x=np.zeros((8,STATE_SIZE),np.float32)
    loss=net.fit_actions(x,np.ones(8,np.int64),np.ones(8,np.float32),lr=.001)
    assert loss==pytest.approx(.5*.05**2)
    assert np.all(net.forward(x)[:,1]>net.forward(x)[:,0])

def test_planner_imitation_zero_weight_cannot_change_excluded_example():
    a=Network(4,16);b=Network(4,16);x=np.zeros((2,STATE_SIZE),np.float32)
    a.fit_actions(x,np.array([0,1]),np.array([1,0]),lr=.001)
    b.fit_actions(x[:1],np.array([0]),np.array([1]),lr=.001)
    assert all(np.array_equal(a.params[k],b.params[k]) for k in a.params)

def test_planner_imitation_anchor_gradient_reduces_previous_q_error():
    net=Network(4,16);x=np.zeros((8,STATE_SIZE),np.float32)
    targets=np.full((8,2),.3,np.float32)
    before=np.mean((net.forward(x)-targets)**2)
    for _ in range(80):
        net.fit_actions(x,np.ones(8,np.int64),np.ones(8,np.float32),lr=.005,
                        anchors=x,anchor_values=targets,anchor_weight=10)
    assert np.mean((net.forward(x)-targets)**2)<before*.1

def test_difficult_state_uses_only_visible_pipe_and_entry_time():
    from rl.imitation import difficult
    env=FlappyEnv();s,_=env.reset();assert not difficult(s)
    s[8:12]=[100/900,100/535,280/535,1]
    assert difficult(s)
    s[8]=500/900;assert not difficult(s)
    s[12:16]=[600/900,250/535,430/535,1]
    assert not difficult(s)
    s[12]=400/900;assert difficult(s)

def test_planner_teacher_reads_simulator_state_without_capture(monkeypatch):
    from rl.imitation import PlannerTeacher
    from sim.source_engine import SourceSimulator
    env=FlappyEnv();state,info=env.reset(4)
    for i in range(6):state,_,_,_,info=env.step(int(i==0))
    assert np.any(state[11:20:4]>.5)
    monkeypatch.setattr(SourceSimulator,'observe',lambda self: (_ for _ in ()).throw(AssertionError('Không dùng ảnh.')))
    teacher=PlannerTeacher()
    try:
        action,accepted,decision=teacher.label(env,state)
        assert action in (0,1) and isinstance(accepted,bool)
        assert decision.coast_margin is not None and decision.flap_margin is not None
    finally:teacher.close()

def test_expert_rollout_uses_teacher_action_and_one_query_per_available_frame():
    from rl.imitation import collect
    from types import SimpleNamespace
    class Net:
        def forward(self,x):return np.tile(np.array([[0.,1.]],np.float32),(len(x),1))
    class Teacher:
        def __init__(self):self.calls=0;self.closed=False
        def label(self,env,state):
            self.calls+=1
            return 1,True,SimpleNamespace(coast_margin=1.,flap_margin=2.)
        def close(self):self.closed=True
    teacher=Teacher()
    states,labels,anchors,meta=collect(Net(),[12345],EnvConfig(max_steps=15),query_every=1,teacher=teacher,expert_rollout=True)
    assert teacher.closed and meta['queries']==teacher.calls
    assert np.all(labels==1) and len(anchors)==len(states)
    assert meta['expert_rollout'] and meta['action_counts']==[0,len(labels)]

def test_expanding_network_preserves_previous_q_and_checkpoint(tmp_path):
    old=Network(4,16);wide=old.expanded(32,seed=8)
    x=np.random.default_rng(3).normal(size=(20,STATE_SIZE)).astype(np.float32)
    assert np.allclose(wide.forward(x),old.forward(x),atol=1e-6)
    path=tmp_path/'wide.npz';wide.save(path)
    restored,meta=Network.load(path)
    assert meta['hidden']==32 and np.array_equal(restored.forward(x),wide.forward(x))
    with pytest.raises(ValueError):old.expanded(16)
