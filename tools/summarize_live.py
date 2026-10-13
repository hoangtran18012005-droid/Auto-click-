"""Tổng hợp chuỗi live và trạng thái cuối; phân loại hình học là giả thuyết."""
import argparse,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--series',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    series=json.loads(args.series.read_text(encoding='utf-8'))
    rows=[];frames=[]
    for episode in series['episodes']:
        if episode.get('score') is None:continue
        path=ROOT/'logs'/f"local-{episode['name']}.json"
        data=json.loads(path.read_text(encoding='utf-8'))
        valid=[f for f in data['frames'] if 'y' in f]
        frames.extend(valid)
        if not valid:continue
        last=valid[-1];bird_left=last['x']-last['w']/2;bird_right=last['x']+last['w']/2
        distances={'ceiling':last['y']-last['h']/2-series['config'].get('world_ceiling_px',0),
                   'floor':series['config'].get('world_floor_px',356)-last['y']-last['h']/2}
        overlaps=[]
        for gate in last['gates']:
            if gate['x']<=bird_right and gate['right']>=bird_left:
                distances['pipe_top']=last['y']-last['h']/2-gate['top']
                distances['pipe_bottom']=gate['bottom']-last['y']-last['h']/2
                overlaps.append(gate)
        edge=min(distances,key=distances.get)
        action=next((f for f in reversed(valid) if f['flap']),None)
        rows.append({'name':episode['name'],'score':episode['score'],'span_s':data['summary']['observed_span_s'],
            'last_frame':last,'overlapping_gates':overlaps,'edge_distances_px':distances,
            'nearest_edge':edge,'last_action':action,
            'interpretation':'Khoảng cách bbox ở frame cuối trước modal; không chứng minh hitbox hoặc thời điểm mất đường an toàn.'})
    metrics={}
    for key in ('capture_ms','handoff_ms','transfer_ms','detection_ms','physics_ms','controller_ms','input_ms','capture_to_input_ms','total_ms','fps'):
        values=[f[key] for f in frames if f.get(key) is not None]
        if values:metrics[key]={'samples':len(values),'median':float(np.median(values)),'p05':float(np.percentile(values,5)),'p95':float(np.percentile(values,95))}
    scores=[r['score'] for r in rows];spans=[r['span_s'] for r in rows]
    result={'series':str(args.series),'completed':len(rows),'scores':scores,
        'score_median':float(np.median(scores)),'score_q25':float(np.percentile(scores,25)),
        'score_q75':float(np.percentile(scores,75)),'score_min':min(scores),'score_max':max(scores),
        'reaching_1':sum(s>=1 for s in scores),'reaching_10':sum(s>=10 for s in scores),
        'reaching_50':sum(s>=50 for s in scores),'reaching_100':sum(s>=100 for s in scores),
        'span_median_s':float(np.median(spans)),'span_max_s':max(spans),'frames':len(frames),
        'metrics_pooled_frames':metrics,'interruptions':series.get('interruptions',[]),'episodes':rows}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('episodes','interruptions')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
