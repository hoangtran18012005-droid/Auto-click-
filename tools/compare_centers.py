"""So bbox/centroid trên ảnh cuối lượt đã lưu; dữ liệu ngắn, không tự bật cờ."""
import argparse,json,re
from pathlib import Path
import cv2,numpy as np,yaml
from bot.vision import Vision
from tools.fit_physics import fit_segment

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,help='JSON danh sách tên thư mục ảnh để đóng băng dataset.')
    parser.add_argument('--output',type=Path,default=ROOT/'evaluation/center-comparison.json')
    args=parser.parse_args()
    folders=[ROOT/'screenshots'/name for name in json.loads(args.manifest.read_text(encoding='utf-8'))] if args.manifest else sorted((ROOT/'screenshots').glob('local-*'))
    cfg=yaml.safe_load((ROOT/'evaluation/live-calibration.yaml').read_text(encoding='utf-8'));results={}
    for mode in ('bbox','centroid'):
        values=[];offsets=[]
        for folder in folders:
            log=ROOT/'logs'/f'{folder.name}.json'
            if not log.exists():continue
            frames=json.loads(log.read_text(encoding='utf-8'))['frames'];flaps=[f['t']+.045 for f in frames if f['flap']]
            vision=Vision(dict(cfg,body_centroid=mode=='centroid'),ROOT);points=[];previous=None;last_flap=None
            for file in sorted(folder.glob('*.jpg')):
                match=re.search(r'-(\d+\.\d+)s',file.name)
                if not match:continue
                timestamp=float(match[1]);image=cv2.imread(str(file));obs=vision.detect(image,timestamp)
                intervening=[t for t in flaps if previous is not None and previous<t<=timestamp+.06]
                if intervening:
                    fitted=fit_segment(points,last_flap,.0)
                    if fitted:values.append(dict(fitted,run=folder.name))
                    points=[];last_flap=max(intervening)
                if obs.bird and not obs.ended:
                    points.append({'t':timestamp,'y':obs.bird.y});offsets.append(obs.bird.y)
                previous=timestamp
            fitted=fit_segment(points,last_flap,.0)
            if fitted:values.append(dict(fitted,run=folder.name))
        result={'segments':len(values),'fits':values}
        for key in ('g','jump','rmse_px'):
            data=[v[key] for v in values if v[key] is not None]
            result[key]={'n':len(data),'median':float(np.median(data)),'std':float(np.std(data))} if data else None
        results[mode]=result
    results['source_folders']=[folder.name for folder in folders]
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({mode:{k:v for k,v in results[mode].items() if k!='fits'} for mode in ('bbox','centroid')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
