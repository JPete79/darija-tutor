# Run commands

Quick reference for starting, stopping and restarting the tutor. All commands are PowerShell, run from the repo root (`C:\Users\jimpe\dev\darija-tutor`).

## One command

```powershell
.\talk.cmd
```

This starts the HADRA server in a minimized window, or reuses it if it's already running. It waits until the server is ready, then starts a voice chat (Noual, atlas-chat-9b). You don't need to activate any environment. You can add any chat flags:

```powershell
.\talk.cmd --persona abdul-ali --topic "food in Fes" --speed 0.85
.\talk.cmd -Restart        # stop the old HADRA server first, then start fresh
.\talk.cmd -StopServer     # just stop the HADRA server
```

The server keeps running after you quit the chat, so the next `.\talk.cmd` starts right away.

## Step by step

To do it by hand instead, you need **two terminals**: one for the HADRA voice server, one for the chat.

## 1. Stop any old HADRA server

If the server is still running in a terminal, press `Ctrl+C` there.
If you closed that terminal or don't know where it is, stop whatever is listening on port 8021:

```powershell
Get-NetTCPConnection -LocalPort 8021 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

Check it is gone (no output means it stopped):

```powershell
Get-NetTCPConnection -LocalPort 8021 -State Listen -ErrorAction SilentlyContinue
```

Optional: free GPU memory by unloading the LLM from Ollama:

```powershell
ollama ps
ollama stop atlas-chat-9b
```

## 2. Start the HADRA voice server (terminal 1)

HADRA uses its own environment (`.venv-hadra`), so call its Python directly; no need to activate anything:

```powershell
.venv-hadra\python.exe scripts\hadra_server.py
```

Wait for `ready: speakers [...] on http://127.0.0.1:8021` (takes about a minute). Leave this terminal open.

Check it from another terminal (optional):

```powershell
Invoke-RestMethod http://127.0.0.1:8021/health
```

## 3. Enter the main virtual environment (terminal 2)

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt now starts with `(.venv)`. If PowerShell blocks the script, run this once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Leave the environment with `deactivate`.

## 4. Start talking (terminal 2)

Voice conversation with Noual (female voice, F1):

```powershell
python -m darija_tutor.cli discussion --model atlas-chat-9b --voice
```

With Abdul-Ali (male voice, M1):

```powershell
python -m darija_tutor.cli discussion --model atlas-chat-9b --voice --persona abdul-ali
```

On a chosen topic, slower speech:

```powershell
python -m darija_tutor.cli discussion --model atlas-chat-9b --voice --topic "food in Fes" --speed 0.85
```

Text-only (no HADRA server needed):

```powershell
python -m darija_tutor.cli discussion --model atlas-chat-9b
```

Lessons (text only):

```powershell
python -m darija_tutor.cli lesson greetings --model atlas-chat-9b
python -m darija_tutor.cli lesson verbs_past_kteb --model atlas-chat-9b
```

Without activating `.venv`, put `.venv\Scripts\python.exe` in place of `python` in any command above.

Useful flags: `--learner male|female`, `--show arabic arabizi english`, `--width 60` (if Arabic rows wrap), `--speaker F1..F4|M1..M3`.

## While chatting

| Type | Does |
|---|---|
| `/slow`, `/fast` | speed down / up by 0.1 |
| `/speed 0.8` | set speed (0.5 to 1.1) |
| `/sentences` | speak one sentence at a time |
| `/words` | speak one word at a time |
| `/normal` | speak the whole reply at once |
| `/repeat [mode]` | say the last reply again |
| `/clear` | forget the conversation, start fresh |
| `/help` | list these controls |
| `Ctrl+C` | quit the chat |

## Full restart, in short

```powershell
# terminal 1
Get-NetTCPConnection -LocalPort 8021 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
.venv-hadra\python.exe scripts\hadra_server.py

# terminal 2 (after "ready")
.venv\Scripts\Activate.ps1
python -m darija_tutor.cli discussion --model atlas-chat-9b --voice
```
