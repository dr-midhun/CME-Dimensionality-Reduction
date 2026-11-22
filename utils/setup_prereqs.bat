@echo off
rem Setup prerequisites for Windows
setlocal enabledelayedexpansion

echo =======================================================
echo Setting up prerequisites for CME-fast on Windows...
echo =======================================================

rem Check for winget
where winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Detected winget package manager.
    echo Checking / installing CMake and Visual Studio Build Tools...
    winget install --id Kitware.CMake --accept-package-agreements --accept-source-agreements
    winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--passive --add Microsoft.VisualStudio.Workload.VCTools" --accept-package-agreements --accept-source-agreements
    echo.
    echo Installation initiated/completed.
    echo Please make sure to open the "x64 Native Tools Command Prompt for VS 2022"
    echo to build the project.
    exit /b 0
)

rem Check for choco (Chocolatey)
where choco >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Detected Chocolatey package manager.
    echo Installing cmake and visualstudio2022buildtools...
    choco install -y cmake visualstudio2022buildtools visualstudio2022-workload-vctools
    echo.
    echo Please open "x64 Native Tools Command Prompt for VS 2022" to build the project.
    exit /b 0
)

echo Warning: Neither winget nor chocolatey was found on your system.
echo Please manually install:
echo   1. Visual Studio 2022 (with "Desktop development with C++" workload)
echo      https://visualstudio.microsoft.com/visual-cpp-build-tools/
echo   2. CMake (optional, for cmake-based builds)
echo      https://cmake.org/download/
echo.
echo Once installed, launch the "x64 Native Tools Command Prompt for VS"
echo and run: nmake -f Makefile.win
exit /b 1
