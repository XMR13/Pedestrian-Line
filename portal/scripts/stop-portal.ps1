param(
    [int]$Port = 5000,
    [int]$ProcessId,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$portalDir = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $portalDir '.portal.pid'

$targets = @()

if ($ProcessId -gt 0) {
    $targets += $ProcessId
}
else {
    try {
        $byPort = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop |
            Select-Object -ExpandProperty OwningProcess -Unique
        if ($byPort) {
            $targets += $byPort
        }
    }
    catch {
        # No listener on this port.
    }

    if ($targets.Count -eq 0 -and (Test-Path -LiteralPath $pidFile)) {
        $rawPid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
        $parsedPid = 0
        if ([int]::TryParse(($rawPid | Out-String).Trim(), [ref]$parsedPid) -and $parsedPid -gt 0) {
            $targets += $parsedPid
        }
    }
}

$targets = $targets | Select-Object -Unique

if ($targets.Count -eq 0) {
    Write-Host "No portal process found (port=$Port)."
    exit 0
}

foreach ($target in $targets) {
    try {
        $proc = Get-Process -Id $target -ErrorAction Stop
        Stop-Process -Id $target -Force:$Force.IsPresent
        Write-Host "Stopped PID $target ($($proc.ProcessName))"
    }
    catch {
        Write-Warning "Failed to stop PID ${target}: $($_.Exception.Message)"
    }
}

if (Test-Path -LiteralPath $pidFile) {
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
}
