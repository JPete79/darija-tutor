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
    ap.add_argument("--learner", choices=["male", "female"], help="your gender, for correct 'you' forms")
    ap.add_argument("--model", help=f"Ollama model name (default {config.LLM_MODEL})")
    ap.add_argument("--voice", action="store_true", help="speak replies (needs scripts/hadra_server.py running)")
    ap.add_argument("--speaker", choices=["F1", "F2", "F3", "F4", "M1", "M2", "M3"],
                    help="HADRA voice (default: the persona's)")
    args = ap.parse_args()

    tutor = Tutor(args.mode, args.lesson, model=args.model, persona=args.persona, topic=args.topic,
                  learner_gender=args.learner)
    name = tutor.persona["name"]
    # Lesson replies are free-form teaching text; only discussion uses the Arabic + EN: format.
    base = (lambda r: format_reply(r, tuple(args.show))) if args.mode == "discussion" else str
    render = base if args.logical else (lambda r: for_terminal(base(r)))
    speaker = args.speaker or tutor.persona["speaker"]
    voice = args.voice
    if voice and args.mode != "discussion":
        print("(--voice only applies to discussion mode for now; lesson replies mix English and Darija)")
        voice = False
    if voice and not tts.available():
        print(f"(voice server not ready at {config.TTS_URL}; start it with: "
              r".venv-hadra\python.exe scripts\hadra_server.py -- continuing text-only)")
        voice = False

    def show(reply: str) -> None:
        # Visual (right-to-left) mode puts the name on its own line so wrapped Arabic rows align cleanly.
        print(f"\n{name}:{' ' if args.logical else chr(10)}{render(reply)}", flush=True)
        if voice:
            tts.play(tts.synthesize(spoken_text(reply), speaker))

    show(tutor.say())
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        show(tutor.say(text))


if __name__ == "__main__":
    main()
