param(
    [int]$Port = 5000,
    [switch]$NoBuild,
    [switch]$Foreground
)

$ErrorActionPreference = 'Stop'

$portalDir = Split-Path -Parent $PSScriptRoot
$logsDir = Join-Path $portalDir 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdoutLog = Join-Path $logsDir ("portal-{0}-stdout.log" -f $timestamp)
$stderrLog = Join-Path $logsDir ("portal-{0}-stderr.log" -f $timestamp)
$pidFile = Join-Path $portalDir '.portal.pid'

$dotnetArgs = @('run', '--urls', "http://localhost:$Port")
if ($NoBuild) {
    $dotnetArgs += '--no-build'
}

Set-Location -LiteralPath $portalDir

if ($Foreground) {
    Write-Host "Starting portal in foreground on http://localhost:$Port"
    Write-Host "Logs: $stdoutLog"
    & dotnet @dotnetArgs 2>&1 | Tee-Object -FilePath $stdoutLog
    exit $LASTEXITCODE
}

$proc = Start-Process `
    -FilePath 'dotnet' `
    -ArgumentList $dotnetArgs `
    -WorkingDirectory $portalDir `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii

Write-Host "Portal started in background"
Write-Host "URL: http://localhost:$Port"
Write-Host "PID: $($proc.Id)"
Write-Host "PID file: $pidFile"
Write-Host "STDOUT: $stdoutLog"
Write-Host "STDERR: $stderrLog"
Write-Host "Stop command: .\\scripts\\stop-portal.ps1 -Port $Port"
