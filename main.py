from pathlib import Path
from bot.screen_capture import dpi_aware

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Auto Player dùng ảnh màn hình, chạy trên Windows.')
    parser.add_argument('--config',type=Path,help='Hồ sơ YAML riêng; mặc định profiles/live-20261013-latency.yaml.')
    args=parser.parse_args()
    dpi_aware()
    from ui.app import App
    root=Path(__file__).resolve().parent
    for folder in ('logs','screenshots','templates'):
        (root/folder).mkdir(exist_ok=True)
    App(root,args.config or root/'profiles/live-20261013-latency.yaml').run()
