"""Terminal chat, optionally spoken aloud.

    python -m darija_tutor.cli lesson [lesson_name] [--persona noual|abdul-ali]
    python -m darija_tutor.cli discussion [--topic "..."] [--persona noual|abdul-ali] [--voice]
"""
import argparse
import sys

from . import config, tts
from .translit import for_terminal
from .tutor import SCRIPTS, Tutor, format_reply, spoken_text


def main() -> None:
    # Windows consoles default to cp1252, which can't print or read Arabic.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="discussion", choices=["lesson", "discussion"])
    ap.add_argument("lesson", nargs="?", help="lesson file name in lessons/ (lesson mode)")
    ap.add_argument("--persona", default=config.DEFAULT_PERSONA, choices=list(config.PERSONAS))
    ap.add_argument("--topic", help="conversation topic (discussion mode)")
    ap.add_argument("--show", nargs="+", default=list(SCRIPTS), choices=SCRIPTS,
                    help="views to print: arabic arabizi english (default: all)")
    ap.add_argument("--logical", action="store_true",
                    help="print Arabic in raw logical order (copy-friendly; shows reversed/unjoined "
                         "in terminals without right-to-left support)")
    ap.add_argument("--width", type=int,
                    help="max characters per Arabic row (default: terminal width minus 10); "
                         "lower it, e.g. --width 60, if Arabic rows still wrap")
    ap.add_argument("--learner", choices=["male", "female"], help="your gender, for correct 'you' forms")
    ap.add_argument("--model", help=f"Ollama model name (default {config.LLM_MODEL})")
    ap.add_argument("--voice", action="store_true", help="speak replies (needs scripts/hadra_server.py running)")
    ap.add_argument("--speed", type=float,
                    help="starting voice speed, below 1.0 is slower (default 0.95); change it "
                         "during the chat with /slow, /fast or /speed 0.8")
    ap.add_argument("--speaker", choices=["F1", "F2", "F3", "F4", "M1", "M2", "M3"],
                    help="HADRA voice (default: the persona's)")
    args = ap.parse_args()

    tutor = Tutor(args.mode, args.lesson, model=args.model, persona=args.persona, topic=args.topic,
                  learner_gender=args.learner)
    name = tutor.persona["name"]
    # Lesson replies are free-form teaching text; only discussion uses the Arabic + EN: format.
    base = (lambda r: format_reply(r, tuple(args.show))) if args.mode == "discussion" else str
    render = base if args.logical else (lambda r: for_terminal(base(r), args.width))
    speaker = args.speaker or tutor.persona["speaker"]
    voice = args.voice
    if voice and args.mode != "discussion":
        print("(--voice only applies to discussion mode for now; lesson replies mix English and Darija)")
        voice = False
    if voice and not tts.available():
        print(f"(voice server not ready at {config.TTS_URL}; start it with: "
              r".venv-hadra\python.exe scripts\hadra_server.py -- continuing text-only)")
        voice = False

    speaker_voice = tts.Voice(speaker, args.speed if args.speed is not None else 0.95)
    last_spoken = ""

    def show(reply: str) -> None:
        nonlocal last_spoken
        # Visual (right-to-left) mode puts the name on its own line so wrapped Arabic rows align cleanly.
        print(f"\n{name}:{' ' if args.logical else chr(10)}{render(reply)}", flush=True)
        last_spoken = spoken_text(reply)
        if voice:
            speaker_voice.say(last_spoken)

    def command(line: str) -> None:
        """Voice controls typed during the chat; they are never sent to the model."""
        parts = line[1:].lower().split()
        cmd, arg = (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else "")
        v = speaker_voice
        if cmd in ("slow", "slower"):
            print(f"(speed {v.set_speed(v.speed - 0.1)})")
        elif cmd in ("fast", "faster"):
            print(f"(speed {v.set_speed(v.speed + 0.1)})")
        elif cmd == "speed":
            try:
                print(f"(speed {v.set_speed(float(arg))})")
            except ValueError:
                print("(usage: /speed 0.8   -- between 0.5 and 1.1, lower is slower)")
        elif cmd in v.MODES:
            v.mode = cmd
            print(f"(speaking mode: {cmd})")
        elif cmd == "clear":
            nonlocal last_spoken
            tutor.history = tutor.history[:1]      # keep only the system prompt
            last_spoken = ""
            print("(conversation cleared; the tutor has forgotten everything said so far)")
        elif cmd == "repeat":
            if arg and arg not in v.MODES:
                print("(usage: /repeat [normal|sentences|words])")
            elif not last_spoken:
                print("(nothing to repeat yet)")
            elif voice:
                v.say(last_spoken, arg or None)
            else:
                print("(voice is off; start with --voice)")
        else:
            print("Voice controls (type during the chat):\n"
                  "  /slow  /fast         speed down/up by 0.1\n"
                  "  /speed 0.8           set speed (0.5-1.1; lower is slower)\n"
                  "  /sentences           speak one sentence at a time, with pauses\n"
                  "  /words               speak one word at a time\n"
                  "  /normal              speak the whole reply at once\n"
                  "  /repeat [mode]       say the last reply again (optionally in sentences/words mode)\n"
                  "  /clear               forget the conversation so far (a fresh start on a new topic)\n"
                  f"Now: speed {v.speed}, mode {v.mode}")

    show(tutor.say())
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.startswith("/"):
            command(text)
            continue
        show(tutor.say(text))


if __name__ == "__main__":
    main()
