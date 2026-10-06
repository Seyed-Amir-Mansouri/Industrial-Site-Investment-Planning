@echo off
setlocal
cd /d "%~dp0.."
set "PY_CMD="
py -3 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3"
if not defined PY_CMD (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)
if not defined PY_CMD goto :fail
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
if not exist "%VENV_PY%" %PY_CMD% -m venv "%CD%\.venv"
if errorlevel 1 goto :fail
"%VENV_PY%" -c "import django, pandas, numpy, scipy, sklearn, joblib, pyarrow, linopy, highspy, polars, xarray" >nul 2>&1
if errorlevel 1 "%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
cd /d "%~dp0"
"%VENV_PY%" manage.py migrate --check >nul 2>&1
if errorlevel 1 "%VENV_PY%" manage.py migrate --noinput
if errorlevel 1 goto :fail
curl -s -o nul http://localhost:9000/
if errorlevel 1 start "Capacity Planner" "%VENV_PY%" manage.py runserver 0.0.0.0:9000 --noreload
set "TRIES=0"
:wait
curl -s -o nul http://localhost:9000/
if not errorlevel 1 goto :open
set /a TRIES+=1
if %TRIES% GEQ 60 goto :fail
ping -n 2 127.0.0.1 >nul
goto :wait
:open
start "" http://localhost:9000/
exit /b 0
:fail
pause
exit /b 1
