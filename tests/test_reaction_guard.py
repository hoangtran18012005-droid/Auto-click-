import numpy as np
from bot.compiled_planner import kernel


def test_guard_checks_collision_until_delayed_next_frame():
    gates=np.array([[78.,164.,62.,181.]])
    # Trạng thái gần frame cuối của lượt 512, đã dự đoán hết input latency.
    args=(145.568,278.415,710.38,-248.58494,139.6,gates,145.,26.,23.,0.,356.,100.,.04,1/30,1,.18,.25,200,2.,8.,0,False)
    _,ordinary=kernel(*args,1/30)
    _,guarded=kernel(*args,.05)
    assert ordinary>0
    assert guarded<0
