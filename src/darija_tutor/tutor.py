import yaml

from . import config, llm


def load_prompt(mode: str) -> str:
    return (config.PROMPTS_DIR / f"{mode}_system.md").read_text(encoding="utf-8")


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


class Tutor:
    """Holds the conversation history for one session in `lesson` or `discussion` mode."""

    def __init__(
        self,
        mode: str,
        lesson: str | None = None,
        model: str | None = None,
        persona: str = config.DEFAULT_PERSONA,
        topic: str | None = None,
    ):
        if mode not in ("lesson", "discussion"):
            raise ValueError(f"unknown mode: {mode}")
        self.persona = config.PERSONAS[persona]
        system = f"Your name is {self.persona['name']}.\n\n" + load_prompt(mode)
        if mode == "lesson":
            system += "\n\n" + lesson_context(load_lesson(lesson or "greetings"))
        elif topic:
            system += f"\n\nThe learner has chosen this topic; open with it: {topic}"
        self.model = model
        self.history = [{"role": "system", "content": system}]

    def say(self, user_text: str | None = None) -> str:
        """Send the user's turn (or None to let the tutor open) and return the reply."""
        if user_text is None:
            self.history.append({"role": "user", "content": "Begin."})
        else:
            self.history.append({"role": "user", "content": user_text})
        reply = llm.chat(self.history, self.model)
        self.history.append({"role": "assistant", "content": reply})
        return reply
