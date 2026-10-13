"""Đóng gói source chạy độc lập; loại dữ liệu gameplay và môi trường phát triển."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def build_release():
    files=[]
    for name in ('main.py','local_runner.py','exact_clearance.py','config.yaml','requirements.txt','requirements-tested.txt','requirements-dev.txt','install.ps1','run.bat','README.md','REPORT.md','REPORT-history.md','RL_README.md'):
        files.append(ROOT/name)
    for folder in ('bot','ui','sim','tests','rl'):
        files.extend((ROOT/folder).rglob('*.py'))
    files.extend((ROOT/'profiles').glob('*.yaml'))
    files.extend((ROOT/'templates').glob('*.png'))
    for name in ('read_score.ps1','export_logs.py','benchmark_standalone.py','benchmark_source.py','time_alignment.py','regret.py','run_series.py','build_release.py'):
        files.append(ROOT/'tools'/name)
    files.append(ROOT/'screenshots/local-native-6-controller/13-4.488s.jpg')
    for name in ('standalone-simulation.json','standalone-observed-simulation.json'):
        files.append(ROOT/'evaluation'/name)
    for name in ('live-grid-timer.yaml','goal1000-timer-summary.json'):
        files.append(ROOT/'evaluation'/name)
    files.extend((ROOT/'evaluation').glob('source-*.json'))
    files.extend((ROOT/'evaluation').glob('dqn-*.json'))
    for folder in ('dqn-offline','dqn-continued','dqn-curriculum','dqn-finetune'):
        files.append(ROOT/'models'/folder/'best.npz')
        if folder in ('dqn-curriculum','dqn-finetune'):
            files.append(ROOT/'models'/folder/'last.npz')
        files.extend((ROOT/'models'/folder).glob('train-*.jsonl'))
    for folder in ('dqn-imitation','dqn-imitation-critical','dqn-imitation-risk',
                   'dqn-imitation-expanded','dqn-imitation-one-epoch',
                   'dqn-imitation-expert','dqn-imitation-wide'):
        for name in ('best.npz','last.npz','dataset.npz','dataset-summary.json','imitation.jsonl'):
            path=ROOT/'models'/folder/name
            if path.exists():
                files.append(path)
    files.extend((ROOT/'backups').glob('before-source-physics-*.zip'))
    files.append(ROOT/'backups/before-offline-dqn.zip')
    files.append(ROOT/'backups/before-dqn-curriculum.zip')
    files.append(ROOT/'backups/before-planner-imitation.zip')
    files.append(ROOT/'backups/windows-reachability-before-short-veto.yaml')
    files.append(ROOT/'backups/windows-reachability-user-before-veto12.yaml')
    output=ROOT.parent/'FlappyBirdAutoPlayer.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path,Path('flappy-bird-bot')/path.relative_to(ROOT))
    return output

if __name__=='__main__':print(build_release())
