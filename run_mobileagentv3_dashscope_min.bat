@echo off
setlocal

:: Minimal, ASCII-only, CMD-safe script.
:: Args (optional): %1=adb.exe path, %2=DashScope API Key, %3=instruction
if not "%~1"=="" set "ADB_PATH=%~1"
if not "%~2"=="" set "INSTRUCTION=%~2"

:: Defaults
if "%MODEL%"=="" set "MODEL=gui-owl-7b"
if "%BASE_URL%"=="" set "BASE_URL=http://192.168.3.144:8000/v1"
if "%COOR_TYPE%"=="" set "COOR_TYPE=abs"
if "%API_KEY%"=="" set "API_KEY="

:: Interactive fill (minimal)
if "%ADB_PATH%"=="" set /p ADB_PATH=ADB path (adb.exe): 
if "%INSTRUCTION%"=="" set /p INSTRUCTION=Instruction: 

:: Config summary (as requested)
echo === Config Summary ===
echo ADB_PATH   = "%ADB_PATH%"
echo BASE_URL   = "%BASE_URL%"
echo MODEL      = "%MODEL%"
echo COOR_TYPE  = "%COOR_TYPE%"
echo API_KEY    = "%API_KEY%"
echo INSTRUCTION= "%INSTRUCTION%"

python ".\Mobile-Agent-v3\mobile_v3\run_mobileagentv3.py" --adb_path "%ADB_PATH%" --api_key "%API_KEY%" --base_url "%BASE_URL%" --model "%MODEL%" --instruction "%INSTRUCTION%" --coor_type "%COOR_TYPE%"

endlocal
pause
