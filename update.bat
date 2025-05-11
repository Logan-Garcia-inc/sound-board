
@echo off
setlocal

:: Set the URL and destination path
set "https://github.com/Logan-Garcia-inc/sound-board/raw/refs/heads/main/soundboard/main.py"
set "DEST=%~dp0main.py"

echo Downloading file from %URL%...
curl -L -o "%DEST%" "%URL%"

if %ERRORLEVEL% equ 0 (
    echo File successfully downloaded to %DEST%.
) else (
    echo Failed to download file.
)

endlocal
pause
