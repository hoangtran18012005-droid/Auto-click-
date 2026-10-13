import json, numpy as np
from pathlib import Path
G, J, SPD, BX = 687.4, -264.2, 141.0, 145.0

def maxmin_clearance(y0, v0, gates, g=G, jump=J, spd=SPD, hw=26., hh=24.5, dt=1/60, every=1, qy=1., qv=4., ymin=0., ymax=400., first_action=None, horizon=None):
    """Tìm CHÍNH XÁC (lưới mịn) clearance lớn nhất có thể đạt: max over kế hoạch của min khoảng cách tới mép cột (px).
    >=0: tồn tại quỹ đạo an toàn với hitbox=bbox. every=2 → chỉ quyết định mỗi 2 bước (30 Hz)."""
    end = max((gt['right'] for gt in gates),default=BX-hw) + hw - BX
    K=max(1,int(max(0,end) / spd / dt) + 1) if horizon is None else max(1,round(horizon/dt))
    Y, V, C = np.array([y0]), np.array([v0]), np.array([1e3])
    for k in range(1, K + 1):
        tau = k * dt; can = (k - 1) % every == 0
        if k==1 and first_action is not None:
            can=False
            if first_action:V=np.full_like(V,jump)
        Y2 = np.r_[Y, Y] if can else Y; V2 = np.r_[V, np.full_like(V, jump)] if can else V
        C2 = np.r_[C, C] if can else C
        V2 = V2 + g * dt; Y2 = Y2 + V2 * dt
        c = np.minimum(Y2 - hh - ymin, ymax - (Y2 + hh))
        for gt in gates:
            gx, gr = gt['x'] - spd * tau, gt['right'] - spd * tau
            if gx < BX + hw and gr > BX - hw: c = np.minimum(c, np.minimum(Y2 - hh - gt['top'], gt['bottom'] - (Y2 + hh)))
        C2 = np.minimum(C2, c); ok = C2 > -40
        if not ok.any(): return -40.
        Y, V, C = Y2[ok], V2[ok], C2[ok]
        key = np.round(Y / qy).astype(np.int64) * 100000 + np.round(V / qv).astype(np.int64)
        o = np.lexsort((-C, key)); ks = key[o]; o = o[np.r_[True, ks[1:] != ks[:-1]]]; Y, V, C = Y[o], V[o], C[o]
    return float(C.max())

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Chạy oracle tham chiếu trên log đã lưu.')
    parser.add_argument('--log',type=Path,default=Path(__file__).parent/'logs/local-native-17-final-confirm.json')
    args=parser.parse_args()
    D=json.loads(args.log.read_text(encoding='utf-8'));F=D['frames'];T=np.array([f['t'] for f in F])
    import time; t0 = time.time()
    print(' t      y      v    gates(top-bot)                      m*(60Hz) m*(30Hz) m*(30Hz,g+5%) m*(hh=20)')
    for i in range(int(np.argmin(abs(T - 38.0))), len(F), 4):
        f = F[i]; gs = [gt for gt in f['gates'] if gt['right'] > BX - 26]
        if f['t'] > 43.7: break
        a = maxmin_clearance(f['y'], f['v'], gs); b = maxmin_clearance(f['y'], f['v'], gs, every=2)
        c = maxmin_clearance(f['y'], f['v'], gs, g=G * 1.05, every=2); d = maxmin_clearance(f['y'], f['v'], gs, hh=20., every=2)
        print(f"{f['t']:6.2f} {f['y']:6.1f} {f['v']:6.0f}  " + ' '.join(f"{gt['top']}-{gt['bottom']}" for gt in gs[:3]).ljust(28) + f"{a:8.1f} {b:8.1f} {c:10.1f} {d:10.1f}", flush=True)
    print('elapsed %.0fs' % (time.time() - t0))
