@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.29
if %errorlevel% neq 0 exit /b %errorlevel%
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 exit /b %errorlevel%
ninja -C build
if %errorlevel% neq 0 exit /b %errorlevel%
echo Build succeeded: build\Morphon.exe
