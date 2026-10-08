@echo off
rem EdgeLens launcher (Windows): double-click this file.
rem Finds Python 3.10-3.13 and hands over to bootstrap.py.
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "NONINT=0"
for %%A in (%*) do if /I "%%~A"=="--noninteractive" set "NONINT=1"

set "PYEXE="
for %%V in (3.13 3.12 3.11 3.10) do call :trypy "py -%%V"
if defined PYEXE goto :found
call :trypy "python"
if defined PYEXE goto :found
goto :nopython

:trypy
if defined PYEXE exit /b 0
%~1 -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
if not errorlevel 1 set "PYEXE=%~1"
exit /b 0

:found
%PYEXE% bootstrap.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo EdgeLens exited with code %RC%.
    if "%NONINT%"=="0" pause
)
exit /b %RC%

:nopython
echo Could not find Python 3.10-3.13.
echo 找不到可用的 Python（需要 3.10 到 3.13）。
echo.
echo Please install Python 3.12 from https://www.python.org/downloads/windows/
echo 請到上面的網址下載並安裝 Python 3.12。
echo IMPORTANT: tick "Add python.exe to PATH" on the first installer page.
echo 重點：安裝畫面第一頁要勾選 "Add python.exe to PATH"。
echo Then double-click this file again. 裝好後再雙擊一次這個檔案。
if "%NONINT%"=="0" pause
exit /b 1
