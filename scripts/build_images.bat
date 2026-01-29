@echo off
setlocal EnableDelayedExpansion

REM ====== User-configurable parameters ======
rem set "BLENDER_EXE=D:\blender-3.6.23-windows-x64\blender.exe"
set "BLENDER_EXE=D:\blender-5.0.1-windows-x64\blender.exe"
REM ==========================================

set "PROJECT_ROOT=%~dp0.."
set "DATA_DIR=e:/data/raw"
set "IMAGES_DIR=e:/data/images"
set "RESOLUTION=1024"

if not exist "%IMAGES_DIR%" mkdir "%IMAGES_DIR%"

REM Run Python as module to support relative imports
REM Redirect stderr (2>nul) to suppress Blender internal messages
echo "%BLENDER_EXE%" -b --python-expr "import sys; sys.path.insert(0, r'%PROJECT_ROOT%'); from scripts.build_images import main; main()" -- --data "%DATA_DIR%" --images "%IMAGES_DIR%" --resolution %RESOLUTION%
"%BLENDER_EXE%" -b --python-expr "import sys; sys.path.insert(0, r'%PROJECT_ROOT%'); from scripts.build_images import main; main()" -- --data "%DATA_DIR%" --images "%IMAGES_DIR%" --resolution %RESOLUTION% 2>nul

endlocal

pause