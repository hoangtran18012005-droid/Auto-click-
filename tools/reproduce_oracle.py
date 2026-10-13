"""Tái hiện số clearance lượt 17 trước khi chẩn đoán/sửa planner."""
import argparse
import json
from pathlib import Path
import numpy as np
from exact_clearance import maxmin_clearance,BX

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,default=ROOT/'logs/local-native-17-final-confirm.json')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/oracle-run17.json')
    args=parser.parse_args();frames=json.loads(args.log.read_text(encoding='utf-8'))['frames']
    rows=[]
    for frame in frames:
        if not 38<=frame['t']<=42.2:continue
        gates=[g for g in frame['gates'] if g['right']>BX-26]
        margin=maxmin_clearance(frame['y'],frame['v'],gates,every=2)
        rows.append({'t':frame['t'],'y':frame['y'],'v':frame['v'],'margin_px':margin,'flap':frame['flap']})
    closest=min(rows,key=lambda row:abs(row['t']-42.19))
    source=min(frames,key=lambda row:abs(row['t']-42.19));gates=[g for g in source['gates'] if g['right']>BX-26]
    actions={str(action):maxmin_clearance(source['y'],source['v'],gates,every=2,first_action=action) for action in (False,True)}
    summary={'frames':len(rows),'safe':sum(r['margin_px']>=0 for r in rows),
             'min_margin_px':min(r['margin_px'] for r in rows),'max_margin_px':max(r['margin_px'] for r in rows),
             'closest_42_19':closest,'first_action_values_px':actions,
             'velocity_note':'Tái hiện nguyên bản dùng v ghi trong log, kể cả frame đã ghi đè sau flap; regret xử lý riêng.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'summary':summary,'frames':rows},indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
