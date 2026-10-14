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
