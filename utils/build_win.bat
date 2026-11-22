@echo off
rem Build script for Windows using MSVC (x64 Native Tools Command Prompt)
setlocal

set CXX=cl
set CXXFLAGS=/O2 /std:c++17 /openmp /W3 /EHsc /MD /nologo
set INC=/Iinclude
set SRC=src\network.cpp src\state_space.cpp src\transition.cpp src\eigenbasis.cpp src\eigenbasis_cpu.cpp
set BUILD=build

if not exist %BUILD% mkdir %BUILD%

if "%1"=="clean" (
    if exist %BUILD% rmdir /s /q %BUILD%
    if exist *.obj del /q *.obj
    echo Cleaned.
    exit /b 0
)

echo Building cme_cli.exe and test_small.exe...
%CXX% %CXXFLAGS% %INC% apps\cme_cli.cpp %SRC% /Fe:%BUILD%\cme_cli.exe
if errorlevel 1 exit /b 1

%CXX% %CXXFLAGS% %INC% tests\test_small.cpp %SRC% /Fe:%BUILD%\test_small.exe
if errorlevel 1 exit /b 1

rem Clean intermediate object files created in root directory
if exist *.obj del /q *.obj

echo Build successful.

if "%1"=="test" (
    echo Running tests...
    %BUILD%\test_small.exe
)
