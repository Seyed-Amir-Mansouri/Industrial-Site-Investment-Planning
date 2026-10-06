@echo off
setlocal
cd /d "%~dp0.."
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" python -m venv "%CD%\.venv"
if errorlevel 1 goto :fail
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
cd /d "%~dp0"
"%VENV_PY%" manage.py migrate --noinput
if errorlevel 1 goto :fail
start "Capacity Planner" "%VENV_PY%" manage.py runserver 0.0.0.0:9000 --noreload
powershell -NoProfile -Command "do { Start-Sleep -Seconds 1; try { $r = Invoke-WebRequest -Uri 'http://localhost:9000/' -UseBasicParsing -TimeoutSec 2 } catch { $r = $null } } until ($r); Start-Process 'http://localhost:9000/'"
exit /b 0
:fail
pause
exit /b 1
