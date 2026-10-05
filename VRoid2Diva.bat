@echo off
rem VRoid -> Project DIVA Mega Mix+ converter launcher.
rem Double-click to open the converter, or drag a .vrm file onto this file.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Python 3.9 or newer is required. Get it from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" during installation^)
  pause
  exit /b 1
)
%PY% -c "import numpy, PIL, cryptography" >nul 2>nul
if errorlevel 1 (
  echo Installing required packages ^(first run only^)...
  %PY% -m pip install --user -r requirements.txt
)
%PY% -m vroid2diva %*
if errorlevel 1 pause
