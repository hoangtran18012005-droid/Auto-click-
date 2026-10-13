"""Dùng chung mốc input để replay không nhầm API start, end và frame capture."""

def input_time(frame,origin='input_start'):
    if origin=='frame':return frame['t']
    sent=frame.get('input_sent_t')
    if sent is None:
        sent=frame['t']+max(0,((frame.get('capture_to_input_ms') or 0)-frame.get('capture_ms',0))/1000)
    if origin in ('start','input_start'):
        start=frame.get('input_start_t')
        return start if start is not None else sent-(frame.get('input_ms') or 0)/1000
    return sent

def cooldown_time(frame,cfg):
    return input_time(frame,'start') if cfg.get('align_input_cooldown',False) else frame['t']
