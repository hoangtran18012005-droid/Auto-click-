"""Khôi phục thứ tự khe từ ảnh/log; không dùng state của game."""
import json
from pathlib import Path
import numpy as np
from bot.vision import Gate


def rebuild_sequence(path, speed=141., spacing=195.):
    frames=json.loads(Path(path).read_text(encoding='utf-8'))['frames']
    tracks=[];active={};last_time=None
    for frame in frames:
        now=frame['t'];dt=0 if last_time is None else now-last_time
        used=set();next_active={}
        for gate in sorted(frame.get('gates',[]),key=lambda g:g['x']):
            candidates=[(abs(previous['right']-speed*dt-gate['right']),index)
                        for index,previous in active.items() if index not in used
                        and abs(previous['top']-gate['top'])<12]
            candidates=[pair for pair in candidates if pair[0]<35]
            if candidates:index=min(candidates)[1]
            else:index=len(tracks);tracks.append([])
            used.add(index);next_active[index]=gate
            if gate['x']>0 and 75<gate['right']-gate['x']<100:
                tracks[index].append((gate['x']+speed*now,gate['top'],gate['bottom']))
        active=next_active;last_time=now
    good=[np.median(samples,axis=0) for samples in tracks if len(samples)>=3]
    good.sort(key=lambda row:row[0])
    # Cùng một cột có thể tách track sau che khuất; gộp theo offset thời gian.
    merged=[]
    for row in good:
        if merged and abs(row[0]-merged[-1][0])<spacing*.4:
            continue
        merged.append(row)
    if not merged:raise ValueError('Log không có đủ quan sát cột để khôi phục.')
    first_x=float(merged[0][0])
    gates=[Gate(first_x+i*spacing,first_x+i*spacing+86,float(row[1]),float(row[2])) for i,row in enumerate(merged)]
    residuals=[float(row[0]-(first_x+i*spacing)) for i,row in enumerate(merged)]
    return gates,{'source':str(path),'gates':len(gates),'offset_residual_median_px':float(np.median(np.abs(residuals))),
                  'offset_residual_max_px':max(map(abs,residuals)),
                  'note':'Giữ thứ tự/độ cao khe; chuẩn hóa speed=141 và spacing=195, không tái hiện đúng thời gian renderer.'}
