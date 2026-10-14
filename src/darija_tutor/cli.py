"""Text-only terminal chat.

    python -m darija_tutor.cli lesson [lesson_name] [--persona noual|abdul-ali]
    python -m darija_tutor.cli discussion [--topic "..."] [--persona noual|abdul-ali]
"""
import argparse

from . import config
from .tutor import Tutor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="discussion", choices=["lesson", "discussion"])
    ap.add_argument("lesson", nargs="?", help="lesson file name in lessons/ (lesson mode)")
    ap.add_argument("--persona", default=config.DEFAULT_PERSONA, choices=list(config.PERSONAS))
    ap.add_argument("--topic", help="conversation topic (discussion mode)")
    args = ap.parse_args()

    tutor = Tutor(args.mode, args.lesson, persona=args.persona, topic=args.topic)
    name = tutor.persona["name"]
    print(f"{name}: {tutor.say()}")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        print(f"\n{name}: {tutor.say(text)}")


if __name__ == "__main__":
    main()
