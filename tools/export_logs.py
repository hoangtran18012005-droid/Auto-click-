"""Gom log ứng dụng và ảnh kết thúc đã có; không chụp ảnh mới."""
import argparse,time,zipfile
from pathlib import Path

def export_recent(root,count=3):
    if not 1<=count<=20:raise ValueError('Số phiên xuất phải từ 1 đến 20.')
    root=Path(root);sessions=sorted((root/'logs').glob('run-*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)[:count]
    if not sessions:raise ValueError('Chưa có log ứng dụng. Chạy một phiên rồi nhấn F8.')
    output=root/'exports'/f'logs-{time.time_ns()}.zip';output.parent.mkdir(exist_ok=True)
    files=set()
    for session in sessions:
        files.add(session);files.update((root/'logs').glob(session.stem+'*.json'))
        files.update((root/'screenshots').glob(session.stem+'*.png'))
        files.update((root/'screenshots').glob(session.stem+'*.jpg'))
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):archive.write(path,path.relative_to(root))
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--last',type=int,default=3)
    args=parser.parse_args();print(export_recent(Path(__file__).resolve().parents[1],args.last))
