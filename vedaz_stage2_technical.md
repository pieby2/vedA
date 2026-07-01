# Stage 2 — Hands-On Technical Task (about 4–6 hours)

**Role:** AI Engineer — AI Astrologers, Vedaz
**Time:** ~4–6 hours. Quality matters more than finishing everything. A smaller, working submission beats a big broken one.
**Submit:** A GitHub repo link or a ZIP, containing your code, your generated files, and a short `README.md`.

---

## What this task is really about

In Stage 1 you showed you understand our voice. Now we want to see you **build the tools** that keep that voice consistent at scale.

When you have thousands of example chats instead of 15, you can't check them all by hand. You need code that does three jobs:
1. **Checks** that the chats are well-formed and follow our safety rules.
2. **Creates** more good chats automatically.
3. **Tests** how well an AI assistant is doing.

That's exactly what the three tasks below ask for. Each one should actually run. You'll also write a few sentences explaining your choices — but the working code is the main thing.

You can use any AI model/API you like (we use Together AI with DeepSeek, but anything is fine for this task). Don't hard-code any secret keys into the repo — read them from an environment variable.

---

## Task 1 — A "chat checker" script (required)

Write a script that reads a `.jsonl` file of chats and prints a report. It should:

- Confirm each chat has the right shape (a `system` message, then alternating `user` and `assistant` turns).
- Count how long each chat is (roughly, in words or tokens).
- Find duplicate or near-duplicate chats.
- Split the chats into a "training" set and a smaller "test" set.
- **The important part:** automatically flag any chat that breaks our safety rules — for example, text that predicts death or illness, promises a medical/money result, or pressures someone to pay for a remedy.

You decide how to detect rule-breaking — simple keyword rules, an AI model that reads each chat, or a mix. In your README, explain why you chose your method and where it might miss things.

*Why we ask this: turning a fuzzy rule like "don't scare people" into something a computer can catch is a core part of the job.*

---

## Task 2 — A "chat generator" script (required)

Write a script that uses an AI model to **create new example chats** in our format and voice. It should:

- Take a topic or situation as input (for example: "career delay, Hindi" or "marriage compatibility, skeptical user").
- Ask the AI to produce a full chat in the correct JSON format.
- Read the AI's reply safely and save valid chats to a `.jsonl` file.
- **Run each new chat through your checker from Task 1** and throw away any that break the rules, keeping only the good ones.

Generate at least 10 good chats this way and include them in your submission.

*Why we ask this: this is close to the real day-to-day work — using AI to build training data, then automatically filtering out the bad output.*

---

## Task 3 — A "quality tester" script (required)

Write a script that measures how good and how safe an AI assistant's answers are. It should:

- Take a small set of test questions (10–15 — you can write your own, or reuse the test split from Task 1).
- Send each question to an AI model and collect the answer.
- **Grade each answer** — for example, by asking a second AI model to score it on: Did it follow our safety rules? Was it warm and helpful? Did it stay honest about astrology's limits?
- Print a simple results table (question, answer, scores).

*Why we ask this: before we trust any assistant with real users, we need a repeatable way to measure if it's working and safe — not just a gut feeling.*

---

## Optional — Train a small model (only if you have time and tools)

If you want to go further, fine-tune a small open AI model on the example chats (a lightweight method like LoRA in a free Colab notebook is totally fine) and show a before/after example. **This is a bonus, not required** — we know it needs GPU access and time, and we won't mark you down for skipping it.

---

## What to include in your submission

- Your three scripts (Tasks 1–3), with clear instructions in `README.md` on how to run them.
- The `.jsonl` file of chats you generated in Task 2.
- The results table from Task 3.
- A short note (a few sentences per task) explaining the main choices you made and what you'd improve with more time.
- Anything from the optional task, if you did it.

---

## How we'll judge this

| What we look at | Roughly how much it counts | What "good" looks like |
|---|---|---|
| Working code | 35% | The scripts run, do what's asked, and are clean enough to read. |
| Safety rule detection | 25% | Your checker actually catches scary, dishonest, or fear-selling chats — and you're honest about its blind spots. |
| Quality of generated chats | 20% | The chats from Task 2 sound like Vedaz and pass your own checks. |
| Testing/measuring | 15% | Your quality tester gives a sensible, repeatable score for safety and helpfulness. |
| Clear explanation | 5% | Your README and notes are easy to follow. |

**Notes:**
- You may use AI tools to help you write code, but you own every line and we will ask you about it in the interview.
- There's no single right answer — we care about your reasoning and your care for the user.
- If something is unclear, make a sensible assumption, write it down, and keep going.
