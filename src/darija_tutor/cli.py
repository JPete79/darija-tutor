"""Text-only terminal chat.

    python -m darija_tutor.cli lesson [lesson_name] [--persona noual|abdul-ali]
    python -m darija_tutor.cli discussion [--topic "..."] [--persona noual|abdul-ali]
"""
import argparse
import sys

from . import config
from .translit import for_terminal
from .tutor import SCRIPTS, Tutor, format_reply


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
    args = ap.parse_args()

    tutor = Tutor(args.mode, args.lesson, model=args.model, persona=args.persona, topic=args.topic,
                  learner_gender=args.learner)
    name = tutor.persona["name"]
    # Lesson replies are free-form teaching text; only discussion uses the Arabic + EN: format.
    base = (lambda r: format_reply(r, tuple(args.show))) if args.mode == "discussion" else str
    render = base if args.logical else (lambda r: for_terminal(base(r)))
    print(f"{name}: {render(tutor.say())}")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        print(f"\n{name}: {render(tutor.say(text))}")


if __name__ == "__main__":
    main()
