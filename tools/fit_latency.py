"""Đo phản ứng từ ảnh/log sau click; không suy rằng mọi lệnh đều được áp dụng."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.regret import reconstruct_velocity
from bot.physics import project
from tools.time_alignment import input_time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern',default='local-step6-candidate-30-*.json')
    parser.add_argument('--series',type=Path,help='Chỉ các lượt kết thúc có điểm đã xác nhận trong artifact chuỗi.')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/latency-fit.json')
    args=parser.parse_args();rows=[]
    selected=None
    if args.series:
        series=json.loads(args.series.read_text(encoding='utf-8'))
        selected={f"local-{e['name']}" for e in series['episodes'] if e.get('score') is not None}
    for path in sorted((ROOT/'logs').glob(args.pattern)):
        if selected is not None and path.stem not in selected:continue
        frames=json.loads(path.read_text(encoding='utf-8'))['frames']
        for index,f in enumerate(frames):
            if not f['flap']:continue
            velocity=reconstruct_velocity(f,frames[index-1] if index else None)
            if velocity is None:continue
            start=input_time(f,'input_start')
            response=None
            for following in frames[index+1:]:
                if following['t']>start+.16 or following['flap']:break
                expected=project(f['y'],velocity,f['g'],following['t']-f['t'])[0]
                if following['y']<expected-4:
                    response=max(0,following['t']-start);break
            rows.append({'run':path.stem,'t':f['t'],'latency_ms':1000*response if response is not None else None,'pre_flap_v':velocity})
    values=[r['latency_ms'] for r in rows if r['latency_ms'] is not None]
    result={'method':'Frame đầu lệch thấp hơn coast prediction 4 px, cửa sổ 160 ms; có bias bbox và thời gian renderer. Không phải timestamp input nội bộ.',
            'commands':len(rows),'unconfirmed':sum(r['latency_ms'] is None for r in rows),
            'p10_ms':float(np.percentile(values,10)) if values else None,'median_ms':float(np.median(values)) if values else None,'p90_ms':float(np.percentile(values,90)) if values else None,'rows':rows}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
