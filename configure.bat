@echo off
setlocal EnableExtensions

REM Configure the Visual Studio build tree for this checkout.
REM Relocatable: every project path is derived from %~dp0, so moving the
REM checkout does not leave stale absolute paths in the build tree.
REM
REM External ONNX Runtime SDK (required by the 5D adapter):
REM   set SUZERO_ONNXRUNTIME_ROOT=<sdk>
REM or unpack it into extern\onnxruntime-win-x64-1.24.3\ (git-ignored).
REM
REM Environment overrides:
REM   SUZERO_BUILD_DIR          build directory (default: .\build)
REM   SUZERO_ONNXRUNTIME_ROOT   prebuilt ONNX Runtime SDK
REM Extra arguments are forwarded to cmake.

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not defined SUZERO_BUILD_DIR set "SUZERO_BUILD_DIR=%ROOT%build"

if not defined SUZERO_ONNXRUNTIME_ROOT (
  if exist "%ROOT%extern\onnxruntime-win-x64-1.24.3\include\onnxruntime_cxx_api.h" (
    set "SUZERO_ONNXRUNTIME_ROOT=%ROOT%extern\onnxruntime-win-x64-1.24.3"
  )
)

if not defined SUZERO_ONNXRUNTIME_ROOT (
  echo [ERROR] SUZERO_ONNXRUNTIME_ROOT is not set.
  echo [INFO]  Set it to a prebuilt ONNX Runtime SDK ^(include\, lib\^), or
  echo [INFO]  unpack one into "%ROOT%extern\onnxruntime-win-x64-1.24.3".
  exit /b 1
)

REM A CMake cache stores the absolute source directory. After moving the
REM checkout, reuse fails with "different than the directory where
REM CMakeCache.txt was created"; --fresh regenerates it in place.
set "SUZERO_FRESH="
if not exist "%SUZERO_BUILD_DIR%\CMakeCache.txt" goto :configure
set "ROOT_FWD=%ROOT:\=/%"
set "ROOT_FWD=%ROOT_FWD:~0,-1%"
findstr /I /C:"CMAKE_HOME_DIRECTORY:INTERNAL=%ROOT_FWD%" "%SUZERO_BUILD_DIR%\CMakeCache.txt" >nul 2>&1
if errorlevel 1 (
  echo [INFO] Build cache belongs to a different source path; using --fresh.
  set "SUZERO_FRESH=--fresh"
)

:configure
cmake -S "%ROOT%." -B "%SUZERO_BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 ^
  -DSUZERO_BUILD_5DCHESS=ON ^
  -DSUZERO_BUILD_PYTHON_BINDINGS=ON ^
  -DSUZERO_ONNXRUNTIME_ROOT="%SUZERO_ONNXRUNTIME_ROOT%" ^
  %SUZERO_FRESH% %*
exit /b %errorlevel%
