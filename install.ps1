$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cần Python 3.12 64-bit và Python Launcher (py).' }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Cài thư viện thất bại; kiểm tra thông báo pip ở trên.' }
Write-Host 'Đã cài. Chạy run.bat hoặc .\.venv\Scripts\python.exe main.py'
