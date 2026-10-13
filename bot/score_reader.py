"""OCR sau khi kết thúc lượt; không đọc DOM và không chặn vòng điều khiển."""
import re,subprocess,unicodedata
from pathlib import Path

def parse_result(text):
    text=''.join(c for c in unicodedata.normalize('NFD',text.lower().replace('đ','d')) if not unicodedata.combining(c))
    text=' '.join(text.split())
    anchor=re.search(r'va\s*cham',text)
    if not anchor:return None
    section=re.split(r'ky\s*luc|top\s*5',text[anchor.end():],maxsplit=1)[0]
    match=re.search(r'\b([0-9]+|o)\s*diem\b',section)
    return (0 if match[1]=='o' else int(match[1])) if match else None

def read_result(image_path,root):
    result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(Path(root)/'tools/read_score.ps1'),'-ImagePath',str(image_path)],capture_output=True,timeout=15,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    text=result.stdout.decode('utf-8',errors='replace')
    return parse_result(text) if result.returncode==0 else None
