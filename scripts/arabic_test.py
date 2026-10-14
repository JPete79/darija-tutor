"""Print one Darija sentence several ways to see which one your terminal shows correctly.

    .venv\\Scripts\\python.exe scripts\\arabic_test.py

The sentence is: "What do you want to talk about today?"  Read each line and note which looks right
(connected letters, first word at the right edge).
"""
import sys

import arabic_reshaper
from bidi import get_display

import darija_tutor.translit as translit

sys.stdout.reconfigure(encoding="utf-8")
text = "شنو بغيتي تهضر عليه ليوم؟"
shaped = arabic_reshaper.reshape(text)

print("code file :", translit.__file__)
print("arabizi   :", translit.to_arabizi(text), "   (expect: ... lyum?)")
print()
print("A raw logical order        :", text)
print("B shaped only              :", shaped)
print("C shaped + bidi (CLI mode) :", get_display(shaped))
print("D bidi only (no shaping)   :", get_display(text))
print()
print("Correct looks like: connected letters, first word (shnu) at the RIGHT edge, ? at the left.")

# Long text: the CLI wraps to the terminal width BEFORE reordering, so rows read top to bottom.
long_text = ("سمح ليا على الغلطة. الجامعة الكراويين هي جامعة قديمة ففاس، تأسست فعام 859. "
             "هادي وحدة من أقدم الجامعات فالعالم، وكتعرف حتى بسميتها الحقيقية.")
print()
print("E long text as the CLI prints it (rows must read TOP to BOTTOM, each right-to-left):")
print(translit.for_terminal(long_text))
print("The first sentence (sorry about the mistake) must be on the TOP row, '859' near the first rows.")
