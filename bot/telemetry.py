"""Tổng hợp JSONL sau khi dừng; không đọc màn hình hoặc gửi input."""
import json,statistics
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np

def summarize_session(path,metadata):
    values=defaultdict(list);count=0;first=last=None;actions=0;decisions=Counter();diagnostics=Counter()
    fields=('capture_ms','handoff_ms','detection_ms','physics_ms','controller_ms','input_ms','capture_to_input_ms','total_ms','fps_instant','previous_wait_requested_ms','previous_wait_actual_ms')
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            row=json.loads(line)
            if row.get('kind')=='diagnostic':
                if row.get('event'):
                    diagnostics[row['event']]+=1
                    continue
                diagnostics['captured_frames']+=1
                for key in ('ready','game_over','ended','bird_detected','focused'):
                    diagnostics[key]+=bool(row.get(key))
                continue
            count+=1;actions+=bool(row.get('flap'))
            if first is None:first=row['t']
            last=row['t'];decisions[row.get('decision','unknown')]+=1
            for key in fields:
                if row.get(key) is not None:values[key].append(row[key])
    result=dict(metadata,frames=count,flaps=actions,observed_span_s=(last-first if count else 0),decisions=dict(decisions),diagnostics=dict(diagnostics),metrics={})
    for key,items in values.items():result['metrics'][key]=dict(median=statistics.median(items),p95=float(np.percentile(items,95)))
    return result
