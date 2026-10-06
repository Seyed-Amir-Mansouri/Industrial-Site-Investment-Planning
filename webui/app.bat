@echo off
setlocal
cd /d "%~dp0.."
set "DATA_DIR=%CD%\data_exchange\01_dispatch_output__train_input"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
call :fetch "%DATA_DIR%\elec_samples.parquet" 1EK45o9fBQUdrSi0rlWZI1HkrAKzZic_s
if errorlevel 1 goto :fail
call :fetch "%DATA_DIR%\h2_samples.parquet" 1IsUIwpnYwm10-bS7tCVMrOxcyDaet_uP
if errorlevel 1 goto :fail
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
:fetch
if exist "%~1" exit /b 0
curl -L --fail -o "%~1" "https://drive.usercontent.google.com/download?id=%~2&export=download&confirm=t"
if errorlevel 1 exit /b 1
for %%F in ("%~1") do if %%~zF LSS 1000000 (
    del "%%~F"
    exit /b 1
)
exit /b 0
