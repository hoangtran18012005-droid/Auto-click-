"""So dự đoán coast một frame kế tiếp trên cùng log; không gọi đây là điểm live."""
import argparse,json
from pathlib import Path
import numpy as np,yaml
from bot.physics import Physics,project
from bot.vision import Bird,Gate

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--pattern',default='local-canvas-click-pilot-*.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'));errors=[];events=[];runs=[]
    for path in sorted((ROOT/'logs').glob(args.pattern)):
        data=json.loads(path.read_text(encoding='utf-8'));frames=data['frames'];model=Physics(cfg);last_input=-1e9;previous=None
        for f in frames:
            if 'y' not in f:continue
            model.update(Bird(f['x'],f['y'],f['w'],f['h']),[Gate(**g) for g in f['gates']],f['t'])
            if previous is not None and f['t']-last_input>.18 and not f['flap'] and not previous['flap']:
                dt=f['t']-previous['t']
                if 0<dt<.12:
                    old=project(previous['y'],previous.get('pre_flap_v',previous['v']),previous['g'],dt)[0]-f['y']
                    new=project(previous['y'],previous['estimated_v'],model.g,dt)[0]-f['y']
                    errors.append({'run':path.stem,'t':f['t'],'old_error_px':old,'new_error_px':new})
            previous=dict(f,estimated_v=model.v)
            if f['flap']:
                sent=f.get('input_start_t') if cfg.get('input_timestamp_origin')=='start' else f.get('input_sent_t')
                if sent is None:
                    completion=(f.get('capture_to_input_ms') or 0)-f.get('capture_ms',0)
                    if cfg.get('input_timestamp_origin')=='start':completion-=f.get('input_ms') or 0
                    sent=f['t']+completion/1000
                last_input=sent;model.flap(sent)
        events.extend([dict(event,run=path.stem) for event in model.input_events]);runs.append(path.name)
    summary={'runs':runs,'coast_samples':len(errors),'confirmed':sum(e['status']=='confirmed' for e in events),'unconfirmed':sum(e['status']=='unconfirmed' for e in events)}
    for key in ('old','new'):
        values=np.array([r[key+'_error_px'] for r in errors])
        summary[key]={'rmse_px':float(np.sqrt(np.mean(values**2))),'median_abs_px':float(np.median(abs(values))),'p95_abs_px':float(np.percentile(abs(values),95))}
    args.output.write_text(json.dumps({'summary':summary,'events':events,'errors':errors,'config':cfg},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
