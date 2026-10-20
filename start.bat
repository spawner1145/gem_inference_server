@echo off
setlocal EnableExtensions DisableDelayedExpansion
pushd "%~dp0"
set "_GEM_PYTHON=%GEM_PYTHON%"
if defined _GEM_PYTHON goto ready
if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" set "_GEM_PYTHON=%VIRTUAL_ENV%\Scripts\python.exe"
if defined _GEM_PYTHON goto ready
if exist "%~dp0.venv\Scripts\python.exe" set "_GEM_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined _GEM_PYTHON set "_GEM_PYTHON=python"
:ready
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
echo Python: "%_GEM_PYTHON%"
echo GEM WebUI default: http://127.0.0.1:6700/
"%_GEM_PYTHON%" -m gem_server %*
set "_GEM_EXIT=%ERRORLEVEL%"
if not "%_GEM_EXIT%"=="0" if not "%GEM_NO_PAUSE%"=="1" pause
popd
exit /b %_GEM_EXIT%
