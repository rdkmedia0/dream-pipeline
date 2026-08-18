@echo off
REM Launches the Dream Pipeline web GUI on Windows.
REM
REM Dynamic on purpose: this project folder is meant to be shareable
REM (e.g. over SMB) and run from multiple machines against the same
REM data, so nothing here is a fixed absolute path. Everything is
REM resolved relative to THIS script's own location (%~dp0), and the
REM Python virtual environment lives in this machine's own per-user
REM %APPDATA% (never inside the shared folder itself -- a venv bakes in
REM absolute paths and OS-specific binaries, so one made on machine A
REM is useless on machine B; see setup_installer.py's
REM _default_venv_dir() for the full reasoning, this script just
REM mirrors its Windows branch).
REM
REM First run on a new machine: no venv exists yet at that per-user
REM path, so this runs setup_installer.py first (installs Python
REM packages into a fresh venv there) before launching. Every run after
REM that just launches directly -- no reinstall, no prompts.

setlocal
set "SCRIPT_DIR=%~dp0"
set "PIPELINE_DIR=%SCRIPT_DIR%_pipeline"

if not exist "%PIPELINE_DIR%\dream_step.py" (
    echo ERROR: %PIPELINE_DIR%\dream_step.py not found -- this script must stay next to the _pipeline\ folder.
    exit /b 1
)

set "VENV_PY=%APPDATA%\Dream Pipeline\venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: no python found on PATH -- install Python first ^(python.org^), then re-run this script.
        exit /b 1
    )
    echo No pipeline environment found for this machine yet -- installing Python packages...
    echo ^(this only happens once per machine; it installs into %APPDATA%\Dream Pipeline\venv, never into the shared folder^)
    REM Only the venv/pip step -- NOT setup_installer.py's full
    REM interactive main() (which also asks about Ollama/ComfyUI/model
    REM downloads via input() prompts). Everything after Python packages
    REM is handled by the GUI's own dependency checks/install actions
    REM once it's actually running -- this wrapper's only job is to get
    REM to a launchable GUI, fast and non-interactively.
    python -c "import sys; sys.path.insert(0, r'%PIPELINE_DIR%'); import setup_installer; setup_installer.install_pip_requirements()"
)

if not exist "%VENV_PY%" (
    echo ERROR: setup did not produce a usable environment ^(see messages above^) -- fix the issue and re-run this script.
    exit /b 1
)

"%VENV_PY%" "%PIPELINE_DIR%\dream_step.py" --web %*
