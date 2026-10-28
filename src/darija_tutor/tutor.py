import re

import yaml

from . import config, llm
from .translit import to_arabizi

SCRIPTS = ("arabic", "arabizi", "english")


def spoken_text(reply: str) -> str:
    """The part of a discussion reply to read aloud: the Arabic lines, not the `EN:` gloss."""
    lines = [l.strip() for l in reply.strip().splitlines()]
    return " ".join(l for l in lines if l and not l.upper().startswith("EN:"))


def pair_lines(reply: str) -> list[tuple[str, str]]:
    """Split a reply into (arabic, english) pairs. The model writes each Darija line followed by
    its own `EN:` line; an `EN:` line with no Arabic before it (or extra ones) is kept as its own pair."""
    pairs: list[list[str]] = []
    for line in (l.strip() for l in reply.strip().splitlines()):
        if not line:
            continue
        if line.upper().startswith("EN:"):
            text = line[3:].strip()
            if pairs and not pairs[-1][1]:
                pairs[-1][1] = text
            elif pairs:
                pairs[-1][1] += " " + text      # a second EN line for the same Arabic line
            else:
                pairs.append(["", text])
        else:
            pairs.append([line, ""])
    return [(a, e) for a, e in pairs]


def format_reply(reply: str, show: tuple[str, ...] = SCRIPTS) -> str:
    """Render a reply in the chosen views, one block per sentence: Arabic, Arabizi, then English."""
    blocks = []
    for arabic, english in pair_lines(reply):
        lines = []
        if "arabic" in show and arabic:
            lines.append(arabic)
        if "arabizi" in show and arabic:
            lines.append(to_arabizi(arabic))
        if "english" in show and english:
            lines.append("EN: " + english)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if len(blocks) > 1 else "".join(blocks)


def load_prompt(mode: str, version: str = config.PROMPT_VERSION) -> str:
    """The mode's system prompt; versioned files (`<mode>_system_<version>.md`) win over unversioned."""
    path = config.PROMPTS_DIR / f"{mode}_system_{version}.md"
    if not path.exists():
        path = config.PROMPTS_DIR / f"{mode}_system.md"
    return path.read_text(encoding="utf-8")


def load_lesson(name: str) -> dict:
    path = config.LESSONS_DIR / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def lesson_context(lesson: dict) -> str:
    lines = [f"Lesson: {lesson['title']}"]
    if lesson.get("instructions"):
        lines.append(lesson["instructions"].strip())
    if lesson.get("phrases"):
        lines.append("Target phrases:")
        for p in lesson["phrases"]:
            lines.append(f"- {p['darija']} ({p['translit']}) = {p['english']}")
    for v in lesson.get("verbs", []):
        lines.append(f"Verb: {v['darija']} ({v['translit']}) = {v['english']}")
        for tense, forms in v["conjugations"].items():
            lines.append(f"  {tense}:")
            for person, f in forms.items():
                lines.append(f"    {person}: {f['darija']} ({f['translit']})")
    return "\n".join(lines)


def gender_rules(tutor_gender: str, learner_gender: str | None) -> str:
    """Darija marks gender on adjectives, participles and 'you' forms, so tell the model both."""
    rules = (
        f"You are a {tutor_gender} speaker. When you refer to yourself, use {tutor_gender} "
        f"forms of adjectives and participles (e.g. {'farhana' if tutor_gender == 'female' else 'farhan'} "
        f"for 'happy')."
    )
    if learner_gender:
        rules += (
            f" The learner is {learner_gender}: address them with {learner_gender} forms "
            "wherever Darija distinguishes gender."
        )
    return rules


class Tutor:
    """Holds the conversation history for one session in `lesson` or `discussion` mode."""

    def __init__(
        self,
        mode: str,
        lesson: str | None = None,
        model: str | None = None,
        persona: str = config.DEFAULT_PERSONA,
        topic: str | None = None,
        learner_gender: str | None = None,
        prompt: str = config.PROMPT_VERSION,
    ):
        if mode not in ("lesson", "discussion"):
            raise ValueError(f"unknown mode: {mode}")
        self.persona = config.PERSONAS[persona]
        intro = f"Your name is {self.persona['name']}. " + gender_rules(
            self.persona["gender"], learner_gender
        )
        if config.EXPERTISE_PROMPT and self.persona.get("expertise"):
            intro += (
                f" Your special interests and strongest knowledge are: {self.persona['expertise']}. "
                "When the conversation touches them, share concrete details with enthusiasm; for other "
                "topics you are still helpful. Being passionate about a subject does not mean you are "
                "never wrong, so still follow the rules about flagging uncertainty."
            )
        system = intro + "\n\n" + load_prompt(mode, prompt)
        if mode == "lesson":
            system += "\n\n" + lesson_context(load_lesson(lesson or "greetings"))
        elif topic:
            system += f"\n\nThe learner has chosen this topic; open with it: {topic}"
        self.mode = mode
        self.model = model
        self.history = [{"role": "system", "content": system}]

    def say(self, user_text: str | None = None) -> str:
        """Send the user's turn (or None to let the tutor open) and return the reply."""
        if user_text is None:
            self.history.append({"role": "user", "content": "Begin."})
        else:
            self.history.append({"role": "user", "content": user_text})
        reply = llm.chat(self.history, self.model)
        self.last_usage = dict(llm.last_usage)  # before any translation request overwrites it
        if self.mode == "discussion":
            reply = self.complete_translations(reply)
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def complete_translations(self, reply: str) -> str:
        """Make sure every Arabic line has its own `EN:` line, rebuilding the reply as clean pairs.

        Small models drop the pairing on long answers, so any Arabic line left without English is
        sent back in one short, focused translation request.
        """
        pairs = pair_lines(reply)
        missing = [i for i, (ar, en) in enumerate(pairs) if ar and not en]
        if missing:
            numbered = "\n".join(f"{n}. {pairs[i][0]}" for n, i in enumerate(missing, 1))
            answer = llm.chat(
                [{"role": "user", "content": (
                    "Translate each numbered Moroccan Darija sentence into English, literally and "
                    "closely, without explanations or extra text. Reply with the same numbering, one "
                    "line per sentence, like '1. translation'.\n\n" + numbered)}],
                self.model,
            )
            found = {int(n): text.strip() for n, text in re.findall(r"^\s*(\d+)[.)]\s*(.+)$", answer, re.M)}
            for n, i in enumerate(missing, 1):
                if n in found:
                    pairs[i] = (pairs[i][0], found[n])
        lines = []
        for ar, en in pairs:
            lines += [l for l in (ar, f"EN: {en}" if en else "") if l]
        return "\n".join(lines)
