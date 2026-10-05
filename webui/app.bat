@echo off
setlocal

cd /d "%~dp0"

set "PY="
set "PY_NAME="

if defined PLANNER_PYTHON if exist "%PLANNER_PYTHON%" (
    set "PY=%PLANNER_PYTHON%"
    set "PY_NAME=PLANNER_PYTHON"
)

if not defined PY if exist "%LOCALAPPDATA%\Python\handy\Scripts\python.exe" (
    set "PY=%LOCALAPPDATA%\Python\handy\Scripts\python.exe"
    set "PY_NAME=handy venv"
)
if not defined PY if exist "%USERPROFILE%\handy\Scripts\python.exe" (
    set "PY=%USERPROFILE%\handy\Scripts\python.exe"
    set "PY_NAME=handy venv"
)

if not defined PY if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
    set "PY_NAME=.venv"
)
if not defined PY if exist "%~dp0venv\Scripts\python.exe" (
    set "PY=%~dp0venv\Scripts\python.exe"
    set "PY_NAME=venv"
)
if not defined PY if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PY=%~dp0..\.venv\Scripts\python.exe"
    set "PY_NAME=.venv (project root)"
)
if not defined PY if exist "%~dp0..\venv\Scripts\python.exe" (
    set "PY=%~dp0..\venv\Scripts\python.exe"
    set "PY_NAME=venv (project root)"
)

if not defined PY (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PY=python"
        set "PY_NAME=main Python (PATH)"
    )
)

if not defined PY (
    echo No usable Python found.
    echo Install Python, or set PLANNER_PYTHON to a python.exe path.
    pause
    exit /b 1
)

if /i "%PY%"=="python" (
    for /f "delims=" %%P in ('where python') do if not defined PY_FULL set "PY_FULL=%%P"
) else (
    for %%I in ("%PY%") do set "PY_FULL=%%~fI"
)

echo ============================================================
echo Python environment : %PY_NAME%
echo Interpreter path   : %PY_FULL%
echo ============================================================
"%PY%" -c "import sys; print(sys.version)"
echo.

"%PY%" manage.py migrate --noinput
if errorlevel 1 (
    echo Database migration failed.
    echo If the error is "No module named django", install the requirements:
    echo   "%PY_FULL%" -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo Capacity Planner running at:
echo   http://localhost:9000
echo Press Ctrl+C to stop.
echo.

start "" cmd /c "ping -n 4 127.0.0.1 >nul & start http://localhost:9000"

"%PY%" manage.py runserver 0.0.0.0:9000 --noreload
pause
