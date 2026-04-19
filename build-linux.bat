@echo off
setlocal

set GOOS=linux
set GOARCH=amd64
set CGO_ENABLED=0
set OUTPUT=app

echo Building %OUTPUT% for %GOOS%/%GOARCH% ...
go build -trimpath -ldflags="-s -w" -o %OUTPUT% .
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo Build succeeded: %OUTPUT%
exit /b 0
