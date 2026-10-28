# Ear test: scripted conversations

The user's Fes ear is the quality judge. Each model gets exactly the same situation and the same learner lines, so differences come from the model alone. Run from the project folder in PowerShell, one model at a time, with nothing else using the GPU (no scoring runs).

Models (2026-10-24): `e4b-tulu-1000` (run 2, best trained checkpoint), `gemma-4-e4b-ours` (released E4B, our pipeline), `atlas-chat-9b` (current tutor). Always name `--model`: without it the CLI falls back to `atlas-chat-2b`.

## Starting a conversation

With voice (starts the HADRA server if needed; the first start takes about a minute, then it keeps running in its own minimized window):

```
.\talk.cmd --model e4b-tulu-1000 --topic "<topic below>"
```

Text only: `.venv\Scripts\python -m darija_tutor.cli discussion --model e4b-tulu-1000 --topic "<topic below>"`

The tutor (Noual, female) speaks first. Paste each line at the `>` prompt (not at `PS C:\...>`, which is PowerShell). Ctrl+C ends the conversation. `.\talk.cmd -StopServer` stops HADRA afterwards.

## Conversation 1: cafe

Topic: `Role-play: you are a waiter in a cafe in Fes, and the learner is a customer who wants to order. Stay in the role.`

| # | Learner line | Tests |
|---|---|---|
| 1 | `salam, bghit wa7d l9hwa nss nss 3afak.` | ordering in Arabizi |
| 2 | `wach 3ndkum chi croissant?` | French loanword, staying in role |
| 3 | `How do I ask for less sugar?` | English question mid role-play |
| 4 | `bghit atay b chwiya d sukkar.` | reusing what it taught |
| 5 | `l9hwa bard bzaf.` | planted mistake: l9hwa is feminine, so *barda* |
| 6 | `afak hder b chwiya, ma fhemtch.` | simplifies and still answers |
| 7 | `ch7al l7ssab?` | the bill, numbers, prices |
| 8 | `chokran, bslama!` | natural goodbye |

## Conversation 2: medina small talk

Topic: `Role-play: you know the learner, a student, and you meet by chance in the medina of Fes. Have a short, friendly chat about everyday things. Stay in the role.`

| # | Learner line | Tests |
|---|---|---|
| 1 | `salam! labas 3lik? kif dayra?` | greeting chain, asks back |
| 2 | `ana mzyan, l7amdulillah. w l3a2ila? kolchi bikhir?` | family, politeness formulas |
| 3 | `ana kanqra f jami3a hna f Fes.` | small talk about studies |
| 4 | `lbare7 ana mchat l jami3a m3a sa7bi.` | planted mistake: with *ana* it is *mchit* |
| 5 | `How do I say "the weather is nice today"?` | English question mid chat |
| 6 | `ljaw zwin lyoum, walakin skhun chwiya.` | follows up on its own answer |
| 7 | `chno ghadi tdiri f l3otla?` | plans, future with *ghadi* |
| 8 | `wakha, nchofek men b3d, bslama!` | goodbye formulas |

## Conversation 3: Fes chat (no topic)

Added 2026-10-28 for the prompt v1 vs v2 comparison: free talk, where the first ear test found generic replies and the "not sure" remark firing on the learner's own statements. No `--topic`.

| # | Learner line | Tests |
|---|---|---|
| 1 | `salam! ana 3icht f Fes chi 3am, f lmdina l9dima.` | engages with a statement; no "not sure" remark |
| 2 | `kont kanqra darija f wa7d lmadrasa 9rib l Bab Boujloud.` | picks up a detail (place name), no remark |
| 3 | `imta tbnat jami3at l9arawiyin?` | factual question: the remark should appear |
| 4 | `dar dyali kanet kbir bzaf.` | planted mistake: *dar* is feminine, so *kbira* |
| 5 | `chno kat3jbek nti f Fes?` | opinion flagged as only an opinion |
| 6 | `afak 3awdi b klam sahl, ma fhemtch mzyan.` | simplifies and repeats, does not just promise |
| 7 | `How do I say "I miss Fes"?` | English question mid chat |
| 8 | `chokran bzaf, nchofek mn b3d!` | natural goodbye |

(Lines use feminine *dayra* / *tdiri* / *nti* / *3awdi* for Noual; with `--persona abdul-ali` use *dayr* / *tdir* / *nta* / *3awd*.)

## Automatic replay (prompt and model comparisons)

`eval/ear_replay.py` sends the same lines without typing, text only, and writes one transcript per model and prompt version to `data/ear_runs/` (Arabic, Arabizi, EN, plus simple counts: sentences and questions per reply, "not sure" remarks, bracket placeholders, context used). It does not replace the ear: the counts flag things to look at, and the transcripts are for reading and rating.

```
.venv\Scripts\python eval\ear_replay.py --model atlas-chat-9b --prompt v1 v2
```

Stop HADRA first (`.\talk.cmd -StopServer`); one model at a time. Temperature stays 0.6 as in the app, with a fixed seed so a rerun gives the same text; `--seeds 0 1 2` shows how much a reply varies by chance.

## Rating (1-5 per model and conversation)

1. Sounds like real Darija, as a waiter / acquaintance in Fes would talk
2. MSA words or phrasing (write them down)
3. Stays in the role for all 8 turns
4. Corrects the planted mistake naturally and correctly
5. `EN:` lines are accurate translations
6. Simplifies on request without just promising to
7. With voice: HADRA pronunciation (separate from the model's Darija)

Write down concrete examples, good and bad: they feed the Fes corrections list. Ideally rate before checking which model it was.
