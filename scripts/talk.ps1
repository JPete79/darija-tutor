<#
One command for a voice conversation: start the HADRA server if needed, wait until it is ready, then chat.

    .\talk.cmd                                  # Noual, atlas-chat-9b, voice on
    .\talk.cmd --persona abdul-ali --speed 0.85 # any cli.py flags are passed through
    .\talk.cmd -Restart                         # stop the old HADRA server and start a fresh one

The server opens in its own minimized window and keeps running after the chat ends, so the next
run starts instantly. Close that window (or use -Restart / -StopServer) to stop it.
#>
param(
    [switch]$Restart,
    [switch]$StopServer,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ChatArgs
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$port = 8021
$health = "http://127.0.0.1:$port/health"

function Test-Ready {
    try { return (Invoke-RestMethod $health -TimeoutSec 2).ready } catch { return $false }
}

function Stop-Hadra {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

if ($StopServer) { Stop-Hadra; Write-Host "HADRA server stopped."; return }
if ($Restart) { Write-Host "Stopping old HADRA server ..."; Stop-Hadra; Start-Sleep -Seconds 1 }

if (Test-Ready) {
    Write-Host "HADRA server already running."
} else {
    Write-Host "Starting HADRA server (about a minute) ..."
    $server = Start-Process -FilePath "$root\.venv-hadra\python.exe" -ArgumentList "scripts\hadra_server.py" `
        -WorkingDirectory $root -WindowStyle Minimized -PassThru
    $deadline = (Get-Date).AddMinutes(4)
    while (-not (Test-Ready)) {
        if ($server.HasExited) { throw "HADRA server exited (code $($server.ExitCode)); run it by hand to see the error." }
        if ((Get-Date) -gt $deadline) { throw "HADRA server not ready after 4 minutes." }
        Start-Sleep -Seconds 2
    }
    Write-Host "HADRA server ready."
}

# Defaults first so any flag given on the command line overrides them.
& "$root\.venv\Scripts\python.exe" -m darija_tutor.cli discussion --model atlas-chat-9b --voice @ChatArgs
