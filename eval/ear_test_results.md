# Ear test results (2026-10-24)

Scripts: `eval/ear_test.md`. Persona Noual (F1 voice), HADRA on, num_ctx 2048, temperature 0.6. Not blind.
Same setup for all three: same CLI code (no model-specific branches), same system prompt (`prompts/discussion_system.md`), persona, temperature, num_ctx, Q4_K_M, and post-processing. Differences: each model's own chat template (Atlas/Gemma 2 has no system role, so its Modelfile folds the prompt into the first user turn; Gemma 4 uses Ollama's renderer with a system role; `think: False` only affects Gemma 4). Caveat: the prompt was refined on Atlas replies in earlier sessions, so it may favour Atlas. The Arabizi lines are written by the app's converter, not the models.
The user's ratings are the verdict; "Observed" notes are a first read of the transcripts, for the user to confirm or overrule.

## GPU fit next to HADRA (8 GB RTX 4070 Laptop)

| | e4b-tulu-1000 | gemma-4-e4b-ours | atlas-chat-9b |
|---|---|---|---|
| GPU total incl. HADRA (~1.0 GB) | 4.2 GB | 4.2 GB | 7.4 of 8.2 GB |
| Model on GPU | all | all | all (43/43 layers) |
| Reply time (LLM only) | ~1.3 s | ~1.3 s | 1.2-2 s (first 5 s: load) |
| Context at end of cafe chat | - | ~1,835 / 2,048 | 1,567 / 2,048 |

E4B also keeps ~2.9 GB in shared system memory: its per-layer embedding tables, by design, not a spill.
One 8-turn chat nearly fills num_ctx 2048 (system prompt ~1,000 tokens + three lines per sentence). A repeat run of gemma-4-e4b-ours by mistake (up-arrow) hit the limit (`truncated = 1` in the Ollama log): replies cut off mid-word. Atlas has no room for a larger context next to HADRA; E4B does.

## Conversation 1: cafe

### e4b-tulu-1000
- **Arabizi input: failed.** Ignored the order and looped (same 3-line block repeated; same reply to the same line). A direct probe: Arabizi line ignored, the same line in Arabic script understood. TULU data is Arabic script only.
- Rerun with Arabic-script lines, completed. Observed: opener narrates the role ("ana knqul lik: '...'"); "7lua wla balzbda" (coffee "with butter") twice; English question not answered (no "how to say"); "mskr balzrba" (misread "b chwiya d sukkar"); planted mistake not corrected; did not simplify (repeated its previous line); bill 40 dh. MSA-looking: akhra, asf, lmjmu3, mmt3, "kif ymkn li nkhdm m3ak".
- User rating: _pending_

### gemma-4-e4b-ours
- **Arabizi input: works.**
- Observed: opener ignores the cafe ("what do you want to talk about?"); then good role-play ("ha hi qhutk, tfdl"; "3ndna lkrusan"). English question answered but with MSA ("balrja2, skr qlil", "nsf skr"). Planted mistake corrected by recast ("lqhua barda bzaf?"). Simplify: talked about simplifying instead of doing it. Bill: placeholder "(ndkr ltmn)". Goodbye natural ("bs7tk akhti, thlay frask ... nharik zin"). 3-4 lines per turn (longest). Addressed the learner as female. MSA-looking: akhra, tfdl (= prefer), n3tdr, ntklm.
- User rating: _pending_

### atlas-chat-9b
- **Arabizi input: works.**
- Observed: in role from the first line ("mr7ba bikm f qhwa fas"). Misread "nss nss" as "lqhua mzduja" (double espresso). English question answered in Darija ("bla shkr" / "shuia shkr", but sugar spelled شكر, not سكر) and reused it in turn 4. Planted mistake not corrected (in-role apology). Simplify: promised ("ghadi nhdr bshuia"), replies already short. Bill: placeholder "(hna s3r lqhua dyalk)". Goodbye natural ("llh yshl 3lik"). Shortest replies, one sentence pair per turn. Addressed the learner as male. MSA-looking: mzduja, ql, akhra.
- User rating: _pending_

## Conversation 2: medina small talk

### e4b-tulu-1000
- **Arabizi input: partly understood** (greetings, studies, the English question; not *skhun*, not the holiday question). No Arabic-script rerun.
- Observed: opener out of role ("thank you for this opportunity ... as Noual, who was learning this years ago"); "do you remember anything in the medina?" asked 3 times (opener, turns 2 and 7) and turn 3 repeated verbatim in turn 4 (looping). Answered "labas?" with "mtbdlatsh bzaf"; "kifash nti?"; plural "kt3lmwa" for one person. Planted mistake ignored. "lju mzyan lyum" correct. Misread *skhun* as rain. Holiday question not answered. "m3 lslama" fine.
- User rating: _pending_

### gemma-4-e4b-ours
- **Arabizi input: works** for all 8 lines. Context at the end 1,839 / 2,048, no truncation.
- Observed: in role from the opener ("labas 3lik? sh7al hadi ma shftksh"), but 3-4 lines and up to 3 questions per turn. Natural: "labas 3lya l7md llh", "tbark llh 3lik! shnu ktqra baldbt?", "lju zuin lyum", "ghadi nmshi nzur shi khutati falbadia" (answered the holiday question with *ghadi*), "thlay frask ... ntlaqaw in sha2 llh". Understood *skhun*. Planted mistake ignored; one nonsense line ("are you still thinking about something empty?"). Gender flip: *nta* earlier, *akhti* in the goodbye. MSA-looking: fi lmdina lqdima, ant, aw, ltqs, frayy (فرأيي).
- User rating: _pending_

### atlas-chat-9b
- **Arabizi input: works** for all 8 lines. Context at the end 1,419 / 2,048; GPU peak again 7.4 GB, all on GPU.
- Observed: in role throughout, one short line per turn, answers every question and follows up on content. Natural: "slam! fin knti had layam?", "l7md llh, kulshi mzyan. nta w l3ailtk?", "lyum lju zuin", "m3ak l7q, lju zuin bzaf, wlkn shuia skhun", "m3rftsh mazal, wlkn ymkn nmshi lshi blasa f lb7r wla ljbl". Asked "fin ktqra?" right after the learner said where. Planted mistake ignored. EN line wrong once: "nhar s3id" translated as "see you later, goodbye". Consistent masculine address (*nta*). MSA-looking: ltkhss (mild).
- User rating: _pending_

## First read, both conversations (before the user's ratings)

| | e4b-tulu-1000 | gemma-4-e4b-ours | atlas-chat-9b |
|---|---|---|---|
| Reads Arabizi | cafe no, medina partly | yes | yes |
| Stays in role | weak (meta openers, loops) | yes, after a slow cafe opener | yes |
| English question | cafe no, medina yes | yes, but cafe answer in MSA | yes, in Darija |
| Corrects the planted mistake | no / no | cafe yes (recast) / no | no / no |
| Reply length | medium | long (3-4 lines, several questions) | short (1 line) |
| Template leaks | - | price placeholder | price placeholder |
| GPU with HADRA | 4.2 GB | 4.2 GB | 7.4 GB |

## User's verdict

Atlas-Chat-9B performed best overall and looks like the better starting point for training (2026-10-24). Detailed ratings to follow.

## Free conversation about Fes (no topic, user's own lines; kept in git-ignored `data/fes_questions.md`)

One chat each, lines not identical (user adapted lines 3-8 for Atlas), so an impression, not a measurement. **User: Atlas felt worse than gemma-4-e4b-ours in informal discussion.**
- gemma-4-e4b-ours: engaged with the learner's places (after misreading them first and being corrected), added no facts, the "not sure" remark fired on the learner's own statements, 3-4 lines + a question per turn.
- atlas-chat-9b: ignored the specifics, the same praise + generic question every turn ("hadi 7aja zuina bzaf!"), read the last line backwards (offered to teach English), gender mix, MSA calques (mtir llahtmam, rai3a).
- Reading: the scripted role-plays reward script-following (Atlas strongest); free chat rewards engaging with what the learner said (E4B better). A tutor needs both.

## Fes corrections

_none yet_
