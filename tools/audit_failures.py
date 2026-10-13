"""Đối chiếu các frame trước va chạm với tìm kiếm rộng, cùng cooldown/pending/biên."""
import argparse,json,time
from pathlib import Path
import numpy as np
from bot.physics import Physics
from bot.controller import Controller
from bot.vision import Bird,Gate,Observation
from bot.compiled_planner import warmup
from tools.time_alignment import input_time,cooldown_time

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs',nargs='+',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--stride',type=int,default=3)
    parser.add_argument('--beam',type=int,default=4000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();warmup();rows=[];summaries=[]
    for path in args.logs:
        data=json.loads(path.read_text(encoding='utf-8'));frames=data['frames'];cfg=data['summary']['config'];last_command=-1e9;events=[];selected=[]
        for index,f in enumerate(frames):
            if 'y' not in f:continue
            if f['t']>=frames[-1]['t']-args.seconds and index%args.stride==0:
                model=Physics(dict(cfg,freeze_physics=True));model.v=f.get('pre_flap_v',f['v']);model.g=f['g'];model.jump=f['jump'];model.speed=f['speed']
                model.pending=[event for event in events if event>f['t']]
                obs=Observation(Bird(f['x'],f['y'],f['w'],f['h']),[Gate(**g) for g in f['gates']],np.zeros((495,600,3),np.uint8))
                age=(f.get('handoff_ms',0)+f.get('detection_ms',0)+f.get('physics_ms',0))/1000
                for label,width,separate in [('current',cfg.get('beam_width',200),cfg.get('separate_actions',False)),('wide_reference',args.beam,True)]:
                    settings=dict(cfg,use_numba=True,planner='maxmin',beam_width=width,separate_actions=separate,adaptive_beam=False)
                    controller=Controller(settings);controller.last_flap=last_command
                    start=time.perf_counter();decision=controller.decide(obs,model,f['t'],age)
                    record={'run':path.stem,'t':f['t'],'label':label,'beam':width,'margin':decision.plan_margin,'coast_margin':decision.coast_margin,'flap_margin':decision.flap_margin,'action':decision.flap,'logged_action':f['flap'],'y':f['y'],'v':model.v,'controller_ms':1000*(time.perf_counter()-start)}
                    rows.append(record);selected.append(record)
            if f['flap']:
                last_command=cooldown_time(f,cfg)
                sent=input_time(f,'start' if cfg.get('input_timestamp_origin')=='start' else 'end')
                events.append(sent+cfg.get('input_effect_latency',.045))
        wide=[r for r in selected if r['label']=='wide_reference']
        lost=next((r for r in wide if r['margin']<0),None)
        summaries.append({'run':path.stem,'sampled_states':len(wide),'first_negative_in_window':lost,'limitations':'Tham chiếu beam rộng, không chứng minh tối ưu; dùng v từ log, input latency dự kiến và cùng cooldown. Frame âm đầu tiên trong cửa sổ không nhất thiết là âm đầu tiên toàn lượt.'})
        print(path.name,'frame:',len(wide),'âm đầu:',lost['t'] if lost else None,flush=True)
    args.output.write_text(json.dumps({'summary':summaries,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
