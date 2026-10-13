"""Tổng hợp lượt live đã kết thúc có ảnh điểm; không suy điểm từ thời gian sống."""
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',type=int,default=1000)
    parser.add_argument('--pattern',default='goal1000-*-pilot.json')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/goal1000-status.json')
    args=parser.parse_args();groups=[];completed=[]
    for path in sorted((ROOT/'evaluation').glob(args.pattern)):
        data=json.loads(path.read_text(encoding='utf-8'))
        valid=[e for e in data.get('episodes',[]) if e.get('score') is not None
               and e.get('runner',{}).get('reason')=='modal/end' and e.get('image') and Path(e['image']).is_file()]
        scores=[e['score'] for e in valid]
        groups.append(dict(series=path.name,config=valid[0]['runner'].get('config_path') if valid else None,
                           completed=len(valid),scores=scores,median=float(np.median(scores)) if scores else None,
                           best=max(scores) if scores else None,interruptions=len(data.get('interruptions',[]))))
        completed.extend(valid)
    best=max(completed,key=lambda e:e['score']) if completed else None
    artifact=dict(target=args.target,status='target_score_recorded' if best and best['score']>=args.target else 'in_progress',
                  note='Điểm OCR chỉ được nhận khi có modal/end và tệp ảnh. Cần xem ảnh kỷ lục để xác nhận trước khi báo đạt mục tiêu.',
                  completed=len(completed),best={k:best[k] for k in ('name','score','image')} if best else None,groups=groups)
    args.output.write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(artifact,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
