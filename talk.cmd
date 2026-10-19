@echo off
rem Voice chat in one command; see scripts\talk.ps1. Bypass avoids PowerShell's script-blocking policy.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\talk.ps1" %*
