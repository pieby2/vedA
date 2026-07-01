# Vedaz AI Astrologer - Stage 2 Submission

This repository contains the required scripts for the Vedaz AI Astrologers Stage 2 hands-on technical task. 

## Setup Instructions
1. Install the required Python packages:
   ```bash
   pip install google-genai
   ```
2. Set your Google Gemini API key as an environment variable:
   - On Windows (PowerShell): `$env:GEMINI_API_KEY="your_api_key_here"`
   - On Windows (CMD): `set GEMINI_API_KEY="your_api_key_here"`
   - On Mac/Linux: `export GEMINI_API_KEY="your_api_key_here"`

## Task 1: Chat Checker (`task1_checker.py`)
**Usage**: `python task1_checker.py <input.jsonl>`
*(Defaults to `vedaz_new_chats.jsonl` if no file is provided)*

**What it does:**
- Validates the JSON format (alternating `system`, `user`, `assistant` messages).
- Calculates the approximate word count.
- Checks for duplicate chats using MD5 hashing of the content.
- Uses the LLM via `google-genai` to evaluate the chats against safety rules (avoiding fear-mongering, fatalistic predictions, or guarantees). Falls back to regex heuristics if the API key is not present.
- Splits valid chats into `train_chats.jsonl` (80%) and `test_chats.jsonl` (20%).

**Design Choices & Blind Spots:**
I chose a hybrid approach for safety detection. Simple regex rules (heuristics) are extremely fast but brittle—they might miss veiled threats or overly confident predictions that don't use specific trigger words. 
Therefore, I implemented an **LLM-as-a-judge** layer as the primary safety checker. The LLM can understand the *nuance* of a response (e.g., distinguishing between "you have a short life" vs "I cannot predict your lifespan").
*Blind spot:* The LLM checker adds latency and cost. For a massive dataset (millions of chats), running an LLM on every chat might be too expensive, in which case a specialized, smaller fine-tuned classifier (like a BERT safety classifier) would be more efficient.

## Task 2: Chat Generator (`task2_generator.py`)
**Usage**: `python task2_generator.py`

**What it does:**
- Takes a predefined list of 12 tricky scenarios (e.g., domestic abuse, crypto predictions).
- Calls the Gemini model to synthesize a new chat following the system persona.
- Parses the output into JSON.
- **Passes the generated chat through the checker from Task 1**. If it fails safety or formatting, it discards the chat and tries again.
- Saves the valid chats to `generated_chats.jsonl` incrementally until it collects 10 valid chats.

**Design Choices & Blind Spots:**
I wrapped the generation in a retry loop and directly imported the validation pipeline from Task 1. This ensures that the generated dataset is always clean.
*Blind spot:* If the system prompt for generation is weak, the model might repeatedly generate unsafe outputs, causing an infinite loop. I added a maximum attempt limit (25) to prevent this.

## Task 3: Quality Tester (`task3_tester.py`)
**Usage**: `python task3_tester.py`

**What it does:**
- Iterates over 10 difficult test prompts designed to trigger unsafe or fatalistic responses.
- Generates a response from the baseline AI.
- Uses a separate prompt to evaluate the AI's response on three axes (1 to 5):
  1. **Safety**: Did it avoid predicting death/finance or violating privacy?
  2. **Honesty**: Did it acknowledge the limits of astrology?
  3. **Helpfulness**: Was it empathetic?
- Outputs the results to the terminal and saves them to `task3_results.md`.

**Design Choices & Blind Spots:**
Using an LLM to evaluate another LLM (LLM-as-a-judge) is standard practice for subjective metrics like "helpfulness" and "honesty". It provides a quantifiable score that can be tracked across model iterations.
*Blind spot:* LLM judges can suffer from biases (e.g., preferring longer answers or answers that match its own training data). It's crucial to occasionally audit the judge's scores manually.
